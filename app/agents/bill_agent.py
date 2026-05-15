"""
Itemized Bill Agent — Extracts billing line items and totals.

Processes ONLY pages classified as 'itemized_bill' by the Segregator.
Uses JSON mode as primary method with robust fallback parsing.
"""

import json
import re
import logging
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm_service import get_llm
from app.models.schemas import ItemizedBillData, PageData

logger = logging.getLogger(__name__)

BILL_AGENT_PROMPT = """You are an expert at extracting itemized billing information from hospital bills.

Extract ALL line items from the provided billing pages.

CRITICAL RULES:
1. Extract EVERY line item you can find across all bills/invoices on the pages.
2. If quantity or unit_price aren't clear, set them to null but still include the item.
3. ALL numeric fields MUST be pre-computed final numbers (e.g. 6329.30).
4. NEVER use arithmetic expressions like "6113.0 + 216.3" — always compute the result.
5. If there are multiple bills/invoices, combine all items into ONE unified list.
6. For each item, compute: total = quantity * unit_price.
7. For negative amounts like discounts, use negative numbers: -11.47

TOTALS — VERY IMPORTANT:
- "subtotal": COMPUTE this yourself by adding up ALL the individual item "total" values you extracted. Do NOT copy any printed subtotal from the document.
- "tax": Sum of all tax amounts across all bills. If multiple tax lines exist, add them together into one number.
- "discount": Sum of all discount amounts across all bills as a positive number. If no discount, use null.
- "grand_total": COMPUTE this yourself as: subtotal + tax - discount. Do NOT copy any printed total from the document.

Return ONLY valid JSON. No comments, no trailing commas, no arithmetic operators.

Return a JSON object matching this exact structure:
{
    "items": [
        {"description": "Room Charges (5 days)", "quantity": 5, "unit_price": 2000.0, "total": 10000.0},
        {"description": "Surgery Fee", "quantity": 1, "unit_price": 50000.0, "total": 50000.0},
        {"description": "Medicines", "quantity": null, "unit_price": null, "total": 8500.0}
    ],
    "subtotal": 68500.0,
    "tax": 1233.0,
    "discount": 5000.0,
    "grand_total": 64733.0
}"""


def _clean_llm_json(raw: str) -> str:
    """
    Clean common LLM JSON mistakes before parsing.

    Handles:
    - Markdown code fences (```json ... ```)
    - Arithmetic expressions (6113.0 + 216.3 → 6329.3)
    - Em-dashes used as minus signs (–11.47 → -11.47)
    - Trailing commas before ] or }
    - Comments (// ... or /* ... */)
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Replace em-dashes and en-dashes with minus signs
    text = text.replace("–", "-").replace("—", "-")

    # Remove single-line comments (// ...)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)

    # Remove multi-line comments (/* ... */)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Evaluate arithmetic expressions in numeric values
    def _eval_match(match):
        expr = match.group(1)
        try:
            result = eval(expr)  # Safe: only numbers and operators
            return str(round(result, 2))
        except Exception:
            return match.group(0)

    text = re.sub(
        r'(?<=:)\s*([\d.]+(?:\s*[+\-*/]\s*[\d.]+)+)',
        lambda m: " " + _eval_match(m),
        text,
    )

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()


def _extract_json_from_text(text: str) -> dict:
    """Extract and parse JSON from potentially messy LLM output."""
    cleaned = _clean_llm_json(text)

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with regex
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: extract between first { and last }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1:
        subset = cleaned[first_brace : last_brace + 1]
        return json.loads(subset)

    raise ValueError("Could not extract valid JSON from LLM response")


def extract_bill(pages: list[PageData]) -> dict | None:
    """
    Extract itemized bill information from the given pages.

    Uses JSON mode as the primary method (faster than tool calling).
    Falls back to raw text parsing if JSON mode fails.

    Args:
        pages: Only the pages classified as 'itemized_bill'

    Returns:
        Extracted bill data as dict, or None if no pages provided
    """
    if not pages:
        logger.info("Bill Agent: No itemized bill pages to process")
        return None

    llm = get_llm(temperature=0.0)

    pages_text = "\n\n".join(
        f"--- PAGE {p.page_number} ---\n{p.text}" for p in pages
    )

    messages = [
        SystemMessage(content=BILL_AGENT_PROMPT),
        HumanMessage(content=f"Extract all billing items and totals from these pages:\n\n{pages_text}"),
    ]

    # Attempt 1: JSON mode (faster than tool calling)
    try:
        response = llm.invoke(
            messages,
            response_format={"type": "json_object"},
        )
        parsed = _extract_json_from_text(response.content)
        result = ItemizedBillData(**parsed)
        logger.info(f"Bill Agent extracted {len(result.items)} items, total: {result.grand_total}")
        return result.model_dump()
    except Exception as e:
        logger.warning(f"Bill Agent JSON mode failed, trying raw fallback: {e}")

    # Attempt 2: Fallback — raw text with robust cleaning
    try:
        response = llm.invoke(messages)
        parsed = _extract_json_from_text(response.content)
        result = ItemizedBillData(**parsed)
        logger.info(f"Bill Agent (fallback) extracted {len(result.items)} items, total: {result.grand_total}")
        return result.model_dump()
    except Exception as e2:
        logger.error(f"Bill Agent fallback also failed: {e2}")
        return {"error": f"Extraction failed: {str(e2)}"}
