# 🏥 Claim Processing Pipeline

AI-powered PDF claim processing using **FastAPI + LangGraph + Gradio**.

## Architecture

```
POST /api/process (PDF + claim_id)
        ↓
┌─────────────────────────────────┐
│      LangGraph Workflow         │
│                                 │
│  Segregator Agent (LLM)        │
│  Classifies pages → 9 types    │
│        ↓  ↓  ↓                 │
│  ┌─────┬─────┬──────┐         │
│  │ ID  │Disch│ Bill │  ← parallel│
│  │Agent│Agent│Agent │         │
│  └──┬──┴──┬──┴──┬───┘         │
│     └─────┼─────┘              │
│        Aggregator              │
└─────────────────────────────────┘
        ↓
   JSON Response

UI: Gradio at /ui  |  API: FastAPI at /api/process
```

## Quick Start

```bash
# 1. Clone & setup
git clone <repo-url>
cd claim-processing-pipeline

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (get from https://aistudio.google.com)

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

Then open:
- **Gradio UI**: http://localhost:8000/ui
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

## API Usage

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-2026-001" \
  -F "file=@./sample_claim.pdf"
```

## Switching LLM Provider

Change **one line** in `.env`:

```bash
# Use Gemini (default)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key

# OR use OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
```

Zero code changes required — both providers use the same LangChain interface.

## Tech Stack

- **FastAPI** — REST API framework
- **LangGraph** — Multi-agent orchestration with parallel fan-out
- **Gradio** — Embedded web UI (same port as API)
- **Gemini Flash / OpenAI** — LLM for page classification & data extraction
- **PyMuPDF** — PDF text extraction
- **Pydantic** — Structured output schemas

## Project Structure

```
app/
├── main.py              # FastAPI + Gradio entry point
├── config.py            # Settings (LLM provider toggle)
├── api/routes.py        # POST /api/process endpoint
├── agents/
│   ├── segregator.py    # Page classifier (9 document types)
│   ├── id_agent.py      # Identity document extractor
│   ├── discharge_agent.py # Discharge summary extractor
│   └── bill_agent.py    # Itemized bill extractor
├── workflow/
│   ├── state.py         # LangGraph state schema
│   └── graph.py         # LangGraph graph definition
├── services/
│   ├── llm_service.py   # LLM abstraction (Gemini ↔ OpenAI)
│   └── pdf_service.py   # PDF parsing
├── models/schemas.py    # Pydantic models
└── frontend/gradio_ui.py # Gradio UI
```