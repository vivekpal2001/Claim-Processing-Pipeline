"""
Pydantic schemas for request/response models and LLM structured output.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Document Types ──────────────────────────────────────────────────────────

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]


# ── PDF Page Data ───────────────────────────────────────────────────────────

class PageData(BaseModel):
    """Represents a single page extracted from the PDF."""
    page_number: int
    text: str


# ── Segregator Output ──────────────────────────────────────────────────────

class PageClassification(BaseModel):
    """Classification result for a single page."""
    page_number: int = Field(description="1-indexed page number")
    document_type: str = Field(description="One of the 9 document types")
    confidence: float = Field(description="Confidence score between 0 and 1", ge=0, le=1)


class SegregationResult(BaseModel):
    """Complete segregation output from the Segregator Agent."""
    classifications: list[PageClassification]


# ── ID Agent Output ─────────────────────────────────────────────────────────

class IdentityData(BaseModel):
    """Extracted identity document information."""
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_type: Optional[str] = Field(None, description="e.g. Aadhaar, PAN, Passport")
    id_number: Optional[str] = None
    policy_number: Optional[str] = None
    insurer_name: Optional[str] = None
    additional_details: Optional[dict] = None


# ── Discharge Summary Agent Output ─────────────────────────────────────────

class DischargeSummaryData(BaseModel):
    """Extracted discharge summary information."""
    patient_name: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    diagnosis: list[str] = Field(default_factory=list)
    procedures: list[str] = Field(default_factory=list)
    treating_physician: Optional[str] = None
    hospital_name: Optional[str] = None
    clinical_notes: Optional[str] = None


# ── Itemized Bill Agent Output ──────────────────────────────────────────────

class BillItem(BaseModel):
    """A single line item in the itemized bill."""
    description: str = ""
    quantity: Optional[int] = None
    unit_price: Optional[float] = None
    total: float = 0.0


class ItemizedBillData(BaseModel):
    """Extracted itemized bill information."""
    items: list[BillItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    grand_total: Optional[float] = None


# ── API Response ────────────────────────────────────────────────────────────

class ClaimResponse(BaseModel):
    """Final API response returned to the client."""
    claim_id: str
    status: str
    processing_time_seconds: float
    segregation: dict[str, list[int]]
    extracted_data: dict
