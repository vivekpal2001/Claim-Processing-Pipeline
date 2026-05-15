"""
LangGraph state schema for the claim processing workflow.

Defines the TypedDict that flows through all graph nodes.
Each agent writes to its own field to avoid conflicts during parallel execution.
The agent_routing field uses a custom reducer to merge dicts from parallel agents.
"""

from typing import TypedDict, Annotated
from app.models.schemas import PageData


def _merge_dicts(existing: dict, new: dict) -> dict:
    """Reducer: merge agent_routing dicts from parallel fan-out nodes."""
    merged = existing.copy() if existing else {}
    merged.update(new or {})
    return merged


class ClaimState(TypedDict):
    """
    Shared state that flows through the LangGraph workflow.

    Fields:
        claim_id:        Unique claim identifier from the API request
        pages:           All pages extracted from the PDF (text content)
        segregation:     Segregator output: doc_type → [page_numbers]
        agent_routing:   Transparent record of which pages each agent received and why
                         Uses a merge reducer so parallel agents don't overwrite each other
        id_data:         ID Agent extraction result
        discharge_data:  Discharge Summary Agent extraction result
        bill_data:       Itemized Bill Agent extraction result
        final_result:    Aggregated output from all agents
    """
    claim_id: str
    pages: list[PageData]
    segregation: dict[str, list[int]]
    agent_routing: Annotated[dict[str, dict], _merge_dicts]
    id_data: dict | None
    discharge_data: dict | None
    bill_data: dict | None
    final_result: dict | None

