"""
Gradio frontend UI — embedded in the FastAPI app.

Provides a user-friendly interface for uploading PDFs and viewing
extracted claim data, all served from the same port as the API.
"""

import json
import gradio as gr
from app.workflow.graph import process_claim


def handle_process(claim_id: str, pdf_file) -> tuple[str, str]:
    """
    Gradio handler: process uploaded PDF through the pipeline.

    Args:
        claim_id: User-provided claim ID
        pdf_file: Uploaded PDF file path (from Gradio)

    Returns:
        Tuple of (status_message, json_result)
    """
    if not claim_id or not claim_id.strip():
        return "❌ Please enter a Claim ID", ""

    if pdf_file is None:
        return "❌ Please upload a PDF file", ""

    try:
        # Read the PDF bytes from the uploaded file
        with open(pdf_file, "rb") as f:
            pdf_bytes = f.read()

        result = process_claim(claim_id.strip(), pdf_bytes)
        return "✅ Processing Complete!", json.dumps(result, indent=2, default=str)

    except Exception as e:
        return f"❌ Processing Failed: {str(e)}", ""


def create_gradio_app() -> gr.Blocks:
    """Create the Gradio UI for claim processing."""

    with gr.Blocks(
        title="Claim Processing Pipeline",
    ) as app:
        # Header
        gr.Markdown(
            """
            <div class="main-header">
                <h1>🏥 Claim Processing Pipeline</h1>
                <p>Upload a PDF claim document to extract structured data using AI agents</p>
            </div>
            """,
        )

        with gr.Row():
            # ── Left Column: Inputs ─────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")

                claim_id_input = gr.Textbox(
                    label="Claim ID",
                    placeholder="e.g. CLM-2026-001",
                    info="Unique identifier for this claim",
                )

                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )

                submit_btn = gr.Button(
                    "🚀 Process Claim",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown(
                    """
                    ---
                    ### 🔄 Pipeline Flow
                    ```
                    PDF Upload
                      ↓
                    Segregator (AI classifies pages)
                      ↓
                    ┌─────────┬──────────┬──────────┐
                    ID Agent  Discharge  Bill Agent
                    └─────────┴──────────┴──────────┘
                      ↓
                    Aggregator → JSON Result
                    ```
                    """,
                )

            # ── Right Column: Output ────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-box"],
                )

                result_output = gr.Code(
                    label="Extracted Data (JSON)",
                    language="json",
                    lines=30,
                )

        # Wire up the submit button
        submit_btn.click(
            fn=handle_process,
            inputs=[claim_id_input, pdf_input],
            outputs=[status_output, result_output],
        )

    return app
