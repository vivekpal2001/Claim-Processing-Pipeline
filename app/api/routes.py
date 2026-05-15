"""
FastAPI API routes.

POST /api/process — Process a PDF claim through the LangGraph pipeline.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.workflow.graph import process_claim

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Claims"])


@router.post("/process")
async def process_claim_endpoint(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF claim document"),
):
    """
    Process a PDF claim document through the AI pipeline.

    - Segregates pages into 9 document types using LLM
    - Extracts identity, discharge summary, and billing data
    - Returns aggregated JSON with all extracted information
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    # Read file bytes
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Process through LangGraph pipeline
    try:
        logger.info(f"API: Processing claim {claim_id}, file: {file.filename}")
        result = process_claim(claim_id, pdf_bytes)
        return result
    except Exception as e:
        logger.error(f"API: Processing failed for claim {claim_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Claim processing failed: {str(e)}",
        )
