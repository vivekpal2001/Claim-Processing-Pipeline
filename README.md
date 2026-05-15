# Claim Processing Pipeline

So this is basically an AI system that takes messy multi-page insurance claim PDFs and pulls out all the important stuff — patient info, hospital details, billing items — and spits it out as clean structured JSON.

I built it using **LangGraph** for orchestrating multiple LLM agents that each specialize in one thing. The whole thing runs as a FastAPI server with a Gradio UI for easy testing.

## The Problem

Insurance companies get these thick PDF claim packets. Each packet has like 10-18 pages — some are ID cards, some are hospital bills, some are lab reports, discharge summaries, etc. Manually going through all that is painful and error-prone. So the idea is: let AI agents handle it.

Live : https://web-production-7b006.up.railway.app/ui/

## How the Pipeline Works

Pretty straightforward flow:

```
Upload PDF
    ↓
Page Segregator (figures out what each page is)
    ↓
┌──────────────┬─────────────────┬──────────────┐
│  ID Agent    │ Discharge Agent │  Bill Agent  │  ← these 3 run in parallel
└──────────────┴─────────────────┴──────────────┘
    ↓
Aggregator (merges everything + verifies bill math)
    ↓
Final JSON output
```

**Step by step:**

1. PDF gets uploaded through the API or the web UI
2. **Segregator** reads every page and classifies it (claim form? ID card? hospital bill? etc.)
3. Based on that classification, pages get routed to 3 specialist agents:
   - **ID Agent** gets identity docs + claim forms (because policy numbers are usually on claim forms, not ID cards)
   - **Discharge Agent** gets discharge summary pages
   - **Bill Agent** gets the itemized bill pages
4. All 3 agents run at the same time (parallel fan-out in LangGraph)
5. **Aggregator** collects all results and does a sanity check on the bill math — it recalculates subtotals from the line items instead of blindly trusting whatever the LLM said

## Project Structure

```
app/
├── main.py                  # entry point — FastAPI + Gradio
├── config.py                # loads .env settings
├── api/routes.py            # POST /api/process endpoint
├── agents/
│   ├── segregator.py        # classifies pages into 9 types
│   ├── id_agent.py          # extracts patient + policy info
│   ├── discharge_agent.py   # extracts medical discharge data
│   └── bill_agent.py        # extracts line items + totals
├── workflow/
│   ├── state.py             # shared state that flows between nodes
│   └── graph.py             # the actual LangGraph DAG
├── services/
│   ├── llm_service.py       # swap between Groq/Gemini/OpenAI here
│   └── pdf_service.py       # text extraction from PDFs
├── models/schemas.py        # pydantic models for validation
└── frontend/gradio_ui.py    # web UI
```

## Getting Started

```bash
# clone it
git clone <repo-url>
cd claim-processing-pipeline

# set up virtual env
python -m venv .venv
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# set up your env variables
cp .env.example .env
# open .env and paste your API key
```

## Running It

```bash
uvicorn app.main:app --reload
```

Then open:
- `http://localhost:8000/ui` — Gradio web interface (upload PDFs here)
- `http://localhost:8000/docs` — Swagger API docs

Or hit the API directly:

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-001" \
  -F "file=@your-claim.pdf"
```

## Switching LLM Providers

The cool part — you can swap the LLM backend by changing literally one line in `.env`:

```env
# pick one
LLM_PROVIDER=groq     # Llama 3.3 70B — fast and free (100k tokens/day limit)
LLM_PROVIDER=gemini   # Gemini 2.0 Flash — generous free tier
LLM_PROVIDER=openai   # GPT-4o-mini — paid but reliable
```

I went with **Groq** as the default because it's free and insanely fast. But heads up — the free tier has a daily token limit, so if you're running lots of tests, you might hit 429 rate limits. Just wait an hour or switch to Gemini.

## Some Tricky Things I Dealt With

**LLM math is unreliable.** The bill agent would sometimes just copy the printed subtotal from the PDF instead of actually adding up the items. So the aggregator now ignores whatever the LLM says and recalculates everything from the extracted line items. Problem solved.

**LLM JSON output is messy.** Especially with Llama models on Groq — they'd sometimes output stuff like `"total": 6113.0 + 216.3` (an arithmetic expression instead of a number), or use em-dashes (–) instead of minus signs, or leave trailing commas. The bill agent has a whole JSON cleaning pipeline that fixes all of this before parsing.

**Policy numbers aren't on ID cards.** This one took a while to figure out. The ID agent was returning `policy_number: null` because it was only looking at the identity document page. But policy numbers are on the claim form. So the router now sends both `identity_document` AND `claim_forms` pages to the ID agent, and the response transparently shows this in the `agent_routing` field.

**JSON mode > tool calling.** I switched from LangChain's `with_structured_output` (which uses tool/function calling under the hood) to plain `response_format: json_object`. It's faster because the model doesn't have to reason about tool schemas — it just writes JSON directly.

## What I'd Add Next

- **Vision support** — right now it only works with text-based PDFs. Scanned documents (images) would need a multimodal LLM
- **Caching** — no point re-classifying the same PDF if it's uploaded twice
- **Combo page handling** — some pages have two document types on them (like a prescription on top and a receipt on the bottom). Right now each page gets exactly one classification
- **Better rate limiting** — auto-retry with backoff instead of just failing
- **Docker + deployment** — containerize it and throw it on Render or Railway

## Tech Stack

- FastAPI + Gradio (API + UI)
- LangGraph (multi-agent orchestration)
- LangChain (LLM abstraction)
- Groq / Gemini / OpenAI (LLM providers)
- PyMuPDF (PDF parsing)
- Pydantic (data validation)
