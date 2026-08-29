# Supplier Risk Prediction Agent

An agentic AI tool that predicts supplier risk in a supply-chain environment.
Built as a hybrid **ML + LLM agent** system: a classical ML model scores every
supplier instantly, and an LLM-powered agent is invoked **on demand** to
investigate specific flagged suppliers and produce a narrative report with
recommendations.

## Why hybrid (ML + Agent), not "just ask an LLM"?

Sending 250 suppliers' raw data to an LLM to "guess" a risk number is slow,
non-deterministic, expensive, and hard to audit. Risk scoring from structured
numeric features is a supervised ML problem, so it's solved with a
`RandomForest` model trained on the data — millisecond inference, deterministic,
explainable via feature importance.

The LLM/agent is reserved for what LLMs are actually good at: reading
unstructured signals (news, financial trend, logistics) and **reasoning**
about a *specific* supplier to produce a human-readable investigation report.
This is the core architecture decision this project is built around.

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────────────────────┐
│   Streamlit Frontend │  REST  │              FastAPI Backend              │
│  (dashboard, agent   │───────▶│                                            │
│   investigate, chat) │        │  ┌──────────────┐   ┌───────────────────┐ │
└─────────────────────┘        │  │  ML Risk      │   │  Agentic Layer     │ │
                                │  │  Model        │   │  (agent.py)        │ │
                                │  │ (RandomForest)│   │  tool-calling loop │ │
                                │  │ scores ALL    │   │  invoked ONLY on   │ │
                                │  │ suppliers,    │   │  "Investigate"     │ │
                                │  │ instantly     │   │  click, per        │ │
                                │  └──────┬───────┘   │  supplier           │ │
                                │         │            └────────┬──────────┘ │
                                │         │                     │            │
                                │         ▼                     ▼            │
                                │   data/suppliers.csv    backend/tools.py   │
                                │   (synthetic data)      (news, financial,  │
                                │                          logistics tools)  │
                                └──────────────────────────────────────────┘
                                              │
                                              ▼
                                   Anthropic Claude API
                              (LIVE mode; falls back to a
                               deterministic MOCK agent if
                               no ANTHROPIC_API_KEY is set)
```

**Request flow for "Investigate" (the agentic part):**
1. User picks a flagged supplier in the UI → `POST /suppliers/{id}/investigate`
2. Agent sends the supplier's profile + ML risk score to Claude with 3 tools
   available: `get_news_sentiment`, `get_financial_trend`, `get_logistics_status`
3. Claude decides which tools it needs, we execute them (max 4 iterations),
   feed results back
4. Claude returns a compact JSON report: summary, risk drivers, recommendation,
   next actions
5. Report is cached in-memory by `supplier_id` — re-opening the same supplier
   costs zero extra tokens

## Project structure

```
supplier-risk-agent/
├── data/
│   ├── generate_synthetic_data.py   # synthetic supplier dataset generator
│   └── suppliers.csv                # generated data (250 suppliers)
├── backend/
│   ├── main.py                      # FastAPI app (endpoints)
│   ├── ml_model.py                  # RandomForest training + inference
│   ├── agent.py                     # agentic tool-calling loop + mock fallback
│   ├── tools.py                     # simulated news/financial/logistics tools
│   └── schemas.py                   # pydantic request/response models
├── frontend/
│   └── app.py                       # Streamlit UI (3 tabs)
├── models/                          # trained model artifacts (.joblib)
├── requirements.txt
└── README.md
```

## Setup & run

```bash
git clone <this-repo>
cd supplier-risk-agent
python -m venv venv && source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. generate synthetic data
python data/generate_synthetic_data.py

# 2. train the ML model (also auto-trains on first API call if skipped)
python backend/ml_model.py

# 3. (optional) enable LIVE agent mode with a real LLM
export ANTHROPIC_API_KEY=sk-ant-...      # skip this to run in MOCK mode

# 4. start backend
uvicorn backend.main:app --reload --port 8000

# 5. start frontend (new terminal)
streamlit run frontend/app.py
```

Open the Streamlit URL (usually `http://localhost:8501`). No API key is
required to fully demo the app — it runs in **MOCK mode** with the same tool
calls and a rule-based synthesis, so the demo is never blocked by network/API
availability.

## Key design decisions

| Decision | Why |
|---|---|
| RandomForest for bulk scoring, LLM only for deep-dive | Cost/latency: LLM calls are O(1) per user action instead of O(n) suppliers |
| Structured JSON tool outputs, not raw text | Fewer input tokens per agent turn |
| `claude-haiku` by default, not a large model | Tool orchestration + summarization doesn't need a frontier model |
| Hard cap on tool-call iterations (4) | Bounds worst-case latency/cost of a single investigation |
| In-memory report cache by supplier_id | Re-viewing a report costs zero extra LLM calls |
| MOCK mode fallback with identical tool calls | App is demoable with zero API key / offline |
| ML model persisted to `models/*.joblib` | No retraining cost on every API restart |
| Synthetic data with a weighted-formula label + noise | Learnable ground truth without using confidential real data |

