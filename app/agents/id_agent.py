"""
ID Agent — Extracts identity information from identity document pages.

Processes ONLY pages classified as 'identity_document' by the Segregator.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import get_llm
from app.models.schemas import IdentityData, PageData

logger = logging.getLogger(__name__)

ID_AGENT_PROMPT = """You are an expert at extracting identity information from documents.

Extract the following fields from the provided document pages:
- patient_name: Full name of the patient/insured person
- date_of_birth: Date of birth (any format found)
- id_type: Type of ID (e.g., Aadhaar, PAN, Passport, Driving License, Voter ID)
- id_number: The ID number/document number
- policy_number: Insurance policy number if present
- insurer_name: Name of the insurance company if present
- additional_details: Any other relevant identity details as key-value pairs

If a field is not found in the documents, set it to null.

Return a JSON object matching this exact structure:
{
    "patient_name": "John Doe",
    "date_of_birth": "1990-01-15",
    "id_type": "Aadhaar",
    "id_number": "1234-5678-9012",
    "policy_number": "POL-2024-001",
    "insurer_name": "Star Health Insurance",
    "additional_details": {"address": "123 Main St"}
}"""


def extract_identity(pages: list[PageData]) -> dict | None:
    """
    Extract identity information from the given pages.

    Args:
        pages: Only the pages classified as 'identity_document'

    Returns:
        Extracted identity data as dict, or None if no pages provided
    """
    if not pages:
        logger.info("ID Agent: No identity document pages to process")
        return None

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(IdentityData)

    pages_text = "\n\n".join(
        f"--- PAGE {p.page_number} ---\n{p.text}" for p in pages
    )

    try:
        result: IdentityData = structured_llm.invoke(
            [
                SystemMessage(content=ID_AGENT_PROMPT),
                HumanMessage(content=f"Extract identity information from these pages:\n\n{pages_text}"),
            ]
        )
        logger.info(f"ID Agent extracted: {result.patient_name}")
        return result.model_dump()
    except Exception as e:
        logger.error(f"ID Agent extraction failed: {e}")
        return {"error": str(e)}
