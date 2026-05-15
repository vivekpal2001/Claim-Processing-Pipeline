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

        # TODO: For scanned PDFs with little/no text, add vision support:
        # if len(text) < 50:
        #     pix = page.get_pixmap(dpi=200)
        #     image_bytes = pix.tobytes("png")
        #     image_base64 = base64.b64encode(image_bytes).decode()
        #     → Send image to LLM via multimodal message

        pages.append(
            PageData(
                page_number=page_num + 1,  # 1-indexed
                text=text if text else "[No text content - possibly scanned image]",
            )
        )

    doc.close()
    return pages
