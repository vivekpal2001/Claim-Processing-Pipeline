"""
ID Agent — Extracts identity information from identity document pages.

Processes ONLY pages classified as 'identity_document' by the Segregator.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import get_llm
from app.models.schemas import IdentityData, PageData

logger = logging.getLogger(__name__)

ID_AGENT_PROMPT = """You are an expert at extracting identity and insurance information from documents.

You will receive identity documents (ID cards, Aadhaar, PAN, etc.) AND claim forms.
Extract the following fields by combining information from ALL provided pages:

- patient_name: Full name of the patient/insured person (from ID card or claim form)
- date_of_birth: Date of birth (any format found)
- id_type: Type of government ID (e.g., Aadhaar, PAN, Passport, Driving License, State ID)
- id_number: The government ID number/document number
- policy_number: Insurance policy number (usually found on the CLAIM FORM, not the ID card)
- insurer_name: Name of the insurance company (usually found on the CLAIM FORM header)
- additional_details: Any other relevant details as key-value pairs (address, blood group, gender, member_id, etc.)

IMPORTANT:
- Policy number and insurer name are almost NEVER on the ID card — look for them on claim forms.
- If the same field appears on multiple pages, prefer the most complete/detailed version.
- If a field is truly not found on any page, set it to null.

Return a JSON object matching this exact structure:
{
    "patient_name": "John Doe",
    "date_of_birth": "1990-01-15",
    "id_type": "Aadhaar",
    "id_number": "1234-5678-9012",
    "policy_number": "POL-2024-001",
    "insurer_name": "Star Health Insurance",
    "additional_details": {"address": "123 Main St", "member_id": "MEM-123"}
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
