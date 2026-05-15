"""
LangGraph workflow definition.

Builds the exact flow required by the assignment:
    START → Segregator → [ID Agent, Discharge Agent, Bill Agent] (parallel) → Aggregator → END

The 3 extraction agents run in parallel via LangGraph fan-out.
Each agent only receives the pages assigned to it by the Segregator.
Agent routing is tracked transparently in the response.
"""

import logging
import time
from langgraph.graph import StateGraph, START, END
from app.workflow.state import ClaimState
from app.services.pdf_service import extract_pages
from app.agents.segregator import classify_pages
from app.agents.id_agent import extract_identity
from app.agents.discharge_agent import extract_discharge_summary
from app.agents.bill_agent import extract_bill
from app.models.schemas import PageData

logger = logging.getLogger(__name__)


# ── Node Functions ──────────────────────────────────────────────────────────
# Each node takes the full state and returns a partial state update.
# Each agent node also records its routing in agent_routing for transparency.


def segregator_node(state: ClaimState) -> dict:
    """Classify all pages into document types using LLM."""
    logger.info(f"Segregator: Processing {len(state['pages'])} pages")
    segregation = classify_pages(state["pages"])
    return {"segregation": segregation}


def id_agent_node(state: ClaimState) -> dict:
    """Extract identity info from identity_document + claim_forms pages.

    Policy numbers and insurer details often appear on claim forms,
    not on ID cards. The Segregator routes both page types to this
    agent so it has the full context for identity + policy extraction.
    """
    id_pages = state["segregation"].get("identity_document", [])
    claim_pages = state["segregation"].get("claim_forms", [])
    page_nums = sorted(set(id_pages + claim_pages))

    # Route only the relevant pages — NOT the whole PDF
    relevant_pages = [p for p in state["pages"] if p.page_number in page_nums]
    relevant_pages.sort(key=lambda p: p.page_number)
    logger.info(f"ID Agent: Processing {len(relevant_pages)} pages: {page_nums}")

    result = extract_identity(relevant_pages)

    # Record routing transparently
    routing = {
        "id_agent": {
            "pages_received": page_nums,
            "source_doc_types": [],
            "reason": "Extracts patient identity + insurance policy details",
        }
    }
    if id_pages:
        routing["id_agent"]["source_doc_types"].append(
            {"type": "identity_document", "pages": id_pages}
        )
    if claim_pages:
        routing["id_agent"]["source_doc_types"].append(
            {"type": "claim_forms", "pages": claim_pages,
             "reason": "Policy number and insurer name found on claim forms"}
        )

    return {"id_data": result, "agent_routing": routing}


def discharge_agent_node(state: ClaimState) -> dict:
    """Extract discharge summary from discharge_summary pages ONLY."""
    page_nums = sorted(state["segregation"].get("discharge_summary", []))
    relevant_pages = [p for p in state["pages"] if p.page_number in page_nums]
    logger.info(f"Discharge Agent: Processing {len(relevant_pages)} pages: {page_nums}")

    result = extract_discharge_summary(relevant_pages)

    routing = {
        "discharge_agent": {
            "pages_received": page_nums,
            "source_doc_types": [{"type": "discharge_summary", "pages": page_nums}],
            "reason": "Extracts diagnosis, admission/discharge dates, physician details",
        }
    }

    return {"discharge_data": result, "agent_routing": routing}


def bill_agent_node(state: ClaimState) -> dict:
    """Extract itemized bill from itemized_bill pages ONLY."""
    page_nums = sorted(state["segregation"].get("itemized_bill", []))
    relevant_pages = [p for p in state["pages"] if p.page_number in page_nums]
    logger.info(f"Bill Agent: Processing {len(relevant_pages)} pages: {page_nums}")

    result = extract_bill(relevant_pages)

    routing = {
        "bill_agent": {
            "pages_received": page_nums,
            "source_doc_types": [{"type": "itemized_bill", "pages": page_nums}],
            "reason": "Extracts all line items with costs and calculates totals",
        }
    }

    return {"bill_data": result, "agent_routing": routing}


def aggregator_node(state: ClaimState) -> dict:
    """
    Combine all agent results into the final JSON response.

    Also programmatically recomputes bill totals from extracted line items
    to guarantee mathematical correctness (LLMs sometimes copy printed
    figures from the PDF instead of computing from items).
    """
    logger.info("Aggregator: Merging all agent results")

    bill_data = state.get("bill_data")

    # ── Recompute bill totals from items (never trust LLM math) ─────────
    if bill_data and "items" in bill_data and not bill_data.get("error"):
        items = bill_data["items"]
        computed_subtotal = round(sum(item.get("total", 0) or 0 for item in items), 2)

        llm_subtotal = bill_data.get("subtotal")
        llm_grand = bill_data.get("grand_total")

        # Override subtotal with computed value
        bill_data["subtotal"] = computed_subtotal

        # Recompute grand_total = subtotal + tax - discount
        tax = bill_data.get("tax") or 0
        discount = bill_data.get("discount") or 0
        bill_data["grand_total"] = round(computed_subtotal + tax - discount, 2)

        # Log if we corrected any values
        if llm_subtotal and abs(llm_subtotal - computed_subtotal) > 0.01:
            logger.warning(
                f"Aggregator corrected subtotal: LLM said {llm_subtotal}, "
                f"computed {computed_subtotal} (delta {round(llm_subtotal - computed_subtotal, 2)})"
            )
        if llm_grand and abs(llm_grand - bill_data["grand_total"]) > 0.01:
            logger.warning(
                f"Aggregator corrected grand_total: LLM said {llm_grand}, "
                f"computed {bill_data['grand_total']}"
            )

    final_result = {
        "claim_id": state["claim_id"],
        "segregation": state["segregation"],
        "agent_routing": state.get("agent_routing", {}),
        "extracted_data": {
            "identity": state.get("id_data"),
            "discharge_summary": state.get("discharge_data"),
            "itemized_bill": bill_data,
        },
    }
    return {"final_result": final_result}


# ── Graph Construction ──────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """
    Build the LangGraph workflow:

        START → segregator → id_agent     ─┐
                           → discharge_agent ─┤→ aggregator → END
                           → bill_agent   ─┘

    Fan-out: 3 extraction agents run in parallel after segregation.
    Fan-in:  Aggregator waits for all 3 agents to complete.
    """
    graph = StateGraph(ClaimState)

    # Add all nodes
    graph.add_node("segregator", segregator_node)
    graph.add_node("id_agent", id_agent_node)
    graph.add_node("discharge_agent", discharge_agent_node)
    graph.add_node("bill_agent", bill_agent_node)
    graph.add_node("aggregator", aggregator_node)

    # Edges: START → Segregator
    graph.add_edge(START, "segregator")

    # Fan-out: Segregator → 3 extraction agents (parallel)
    graph.add_edge("segregator", "id_agent")
    graph.add_edge("segregator", "discharge_agent")
    graph.add_edge("segregator", "bill_agent")

    # Fan-in: All 3 agents → Aggregator
    graph.add_edge("id_agent", "aggregator")
    graph.add_edge("discharge_agent", "aggregator")
    graph.add_edge("bill_agent", "aggregator")

    # Aggregator → END
    graph.add_edge("aggregator", END)

    return graph


# Compile the graph once at module level
workflow = build_graph().compile()


def process_claim(claim_id: str, pdf_bytes: bytes) -> dict:
    """
    Process a claim PDF through the full LangGraph pipeline.

    This is the main entry point called by both the API endpoint and Gradio UI.

    Args:
        claim_id: Unique claim identifier
        pdf_bytes: Raw PDF file bytes

    Returns:
        Final processed claim data as dict
    """
    start_time = time.time()

    # Step 1: Extract pages from PDF
    logger.info(f"Processing claim: {claim_id}")
    pages = extract_pages(pdf_bytes)
    logger.info(f"Extracted {len(pages)} pages from PDF")

    # Step 2: Run LangGraph workflow
    initial_state: ClaimState = {
        "claim_id": claim_id,
        "pages": pages,
        "segregation": {},
        "agent_routing": {},
        "id_data": None,
        "discharge_data": None,
        "bill_data": None,
        "final_result": None,
    }

    result = workflow.invoke(initial_state)

    processing_time = round(time.time() - start_time, 2)

    # Step 3: Build response
    final = result["final_result"]
    final["status"] = "success"
    final["processing_time_seconds"] = processing_time
    final["total_pages"] = len(pages)

    logger.info(f"Claim {claim_id} processed in {processing_time}s")
    return final
