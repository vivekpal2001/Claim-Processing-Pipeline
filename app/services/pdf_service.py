"""
PDF parsing service using PyMuPDF (fitz).

Extracts text content from each page of a PDF document.
"""

import fitz  # PyMuPDF
from app.models.schemas import PageData


def extract_pages(pdf_bytes: bytes) -> list[PageData]:
    """
    Extract text content from each page of a PDF.

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        List of PageData with 1-indexed page numbers and text content
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        # Fallback text for scanned/image-only pages
        pages.append(
            PageData(
                page_number=page_num + 1,
                text=text if text else "[No text content - possibly scanned image]",
            )
        )

    doc.close()
    return pages
