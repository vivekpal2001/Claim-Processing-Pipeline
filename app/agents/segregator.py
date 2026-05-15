"""
Segregator Agent — Classifies PDF pages into 9 document types using LLM.

This is the brain of the pipeline. It analyzes every page and routes
them to the appropriate extraction agents.
"""

import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import get_llm
from app.models.schemas import SegregationResult, PageData, DOCUMENT_TYPES

logger = logging.getLogger(__name__)

SEGREGATOR_SYSTEM_PROMPT = """You are an expert document classifier for insurance claim processing.

Analyze each page and classify it into exactly ONE of these 9 document types:
1. claim_forms — Insurance claim application forms
2. cheque_or_bank_details — Cheque images or bank account details
3. identity_document — ID cards (Aadhaar, PAN, Passport), identity proofs
4. itemized_bill — Hospital bills with line items and costs
5. discharge_summary — Medical discharge summaries from hospitals
6. prescription — Doctor prescriptions for medication
7. investigation_report — Lab reports, diagnostic test results
8. cash_receipt — Payment receipts
9. other — Anything that doesn't fit above categories

Rules:
- Each page gets EXACTLY one classification
- Use the exact type names listed above
- Provide a confidence score (0.0 to 1.0) for each classification
- If a page has no text, classify as "other" with low confidence

Return a JSON object with this exact structure:
{
    "classifications": [
        {"page_number": 1, "document_type": "identity_document", "confidence": 0.95},
        {"page_number": 2, "document_type": "discharge_summary", "confidence": 0.88}
    ]
}"""


def classify_pages(pages: list[PageData]) -> dict[str, list[int]]:
    """
    Classify all pages into document types using LLM.

    Args:
        pages: List of PageData extracted from PDF

    Returns:
        Dict mapping document_type → list of page numbers
        e.g. {"identity_document": [1, 2], "discharge_summary": [3, 4]}
    """
    llm = get_llm(temperature=0.0)

    # Build page content for the prompt
    pages_text = "\n\n".join(
        f"--- PAGE {p.page_number} ---\n{p.text}" for p in pages
    )

    human_message = (
        f"Classify each of the following {len(pages)} pages into document types.\n\n"
        f"{pages_text}"
    )

    # Use structured output for reliable JSON parsing
    structured_llm = llm.with_structured_output(SegregationResult)

    try:
        result: SegregationResult = structured_llm.invoke(
            [
                SystemMessage(content=SEGREGATOR_SYSTEM_PROMPT),
                HumanMessage(content=human_message),
            ]
        )
    except Exception as e:
        logger.error(f"Segregator LLM call failed: {e}")
        # Fallback: classify all pages as "other"
        return {"other": [p.page_number for p in pages]}

    # Build the routing map: doc_type → [page_numbers]
    routing: dict[str, list[int]] = {dt: [] for dt in DOCUMENT_TYPES}

    for classification in result.classifications:
        doc_type = classification.document_type
        if doc_type in routing:
            routing[doc_type].append(classification.page_number)
        else:
            logger.warning(f"Unknown doc type '{doc_type}' for page {classification.page_number}")
            routing["other"].append(classification.page_number)

    # Remove empty entries for cleaner output
    routing = {k: v for k, v in routing.items() if v}

    logger.info(f"Segregation result: {routing}")
    return routing
