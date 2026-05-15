"""
Claim Processing Pipeline — FastAPI Application Entry Point.

Serves both:
    - REST API at /api/process (for programmatic access)
    - Gradio UI at / (for browser-based interaction)

Run with: uvicorn app.main:app --reload
"""

import logging
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import gradio as gr

from app.api.routes import router as api_router
from app.frontend.gradio_ui import create_gradio_app

# ── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Claim Processing Pipeline",
    description="AI-powered PDF claim processing with LangGraph multi-agent orchestration",
    version="1.0.0",
)

# Mount API routes
app.include_router(api_router)


# Root redirect to Gradio UI
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the Gradio UI."""
    return RedirectResponse(url="/ui")


# Health check
@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "claim-processing-pipeline"}


# ── Mount Gradio ────────────────────────────────────────────────────────────

gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")
