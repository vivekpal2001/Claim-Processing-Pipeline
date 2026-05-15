"""
Discharge Summary Agent — Extracts medical discharge information.

Processes ONLY pages classified as 'discharge_summary' by the Segregator.
"""

import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import get_llm
from app.models.schemas import DischargeSummaryData, PageData

logger = logging.getLogger(__name__)

DISCHARGE_AGENT_PROMPT = """You are an expert at extracting medical discharge summary information.

Extract the following fields from the provided discharge summary pages:
- patient_name: Full name of the patient
- admission_date: Date of hospital admission (any format found)
- discharge_date: Date of hospital discharge (any format found)
- diagnosis: List of all diagnoses mentioned
- procedures: List of all procedures/treatments performed
- treating_physician: Name of the treating doctor
- hospital_name: Name of the hospital
- clinical_notes: Any additional clinical notes or observations

If a field is not found, set it to null (or empty list for list fields).

Return a JSON object matching this exact structure:
{
    "patient_name": "John Doe",
    "admission_date": "2024-01-10",
    "discharge_date": "2024-01-15",
    "diagnosis": ["Acute Appendicitis", "Peritonitis"],
    "procedures": ["Laparoscopic Appendectomy"],
    "treating_physician": "Dr. Smith",
    "hospital_name": "City General Hospital",
    "clinical_notes": "Patient recovered well post-surgery"
}"""


def extract_discharge_summary(pages: list[PageData]) -> dict | None:
    """
    Extract discharge summary information from the given pages.

    Args:
        pages: Only the pages classified as 'discharge_summary'

    Returns:
        Extracted discharge data as dict, or None if no pages provided
    """
    if not pages:
        logger.info("Discharge Agent: No discharge summary pages to process")
        return None

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(DischargeSummaryData)

    pages_text = "\n\n".join(
        f"--- PAGE {p.page_number} ---\n{p.text}" for p in pages
    )

    try:
        result: DischargeSummaryData = structured_llm.invoke(
            [
                SystemMessage(content=DISCHARGE_AGENT_PROMPT),
                HumanMessage(content=f"Extract discharge summary from these pages:\n\n{pages_text}"),
            ]
        )
        logger.info(f"Discharge Agent extracted for patient: {result.patient_name}")
        return result.model_dump()
    except Exception as e:
        logger.error(f"Discharge Agent extraction failed: {e}")
        return {"error": str(e)}
