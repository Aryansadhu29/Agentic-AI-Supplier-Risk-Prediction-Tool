"""
FastAPI Backend
----------------
Endpoints:
  GET  /suppliers                -> bulk ML-scored supplier list (no LLM, fast)
  GET  /suppliers/{id}           -> single supplier detail
  POST /suppliers/{id}/investigate -> agentic deep-dive report (LLM, on demand)
  POST /chat                     -> ask natural-language questions over the portfolio

Run:
    uvicorn backend.main:app --reload --port 8000
"""

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.ml_model import RiskModel
from backend import agent
from backend.schemas import InvestigationReport, ChatRequest, ChatResponse

DATA_PATH = "data/suppliers.csv"

app = FastAPI(title="Supplier Risk Agent API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = RiskModel()
_scored_df_cache: pd.DataFrame | None = None


def get_scored_df() -> pd.DataFrame:
    """Bulk-score all suppliers once and cache in memory. Cheap (ms), so this
    is recomputed on process start and whenever /refresh is called -- never
    needs an LLM call."""
    global _scored_df_cache
    if _scored_df_cache is None:
        df = pd.read_csv(DATA_PATH)
        _scored_df_cache = _model.score_dataframe(df)
    return _scored_df_cache


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": "live" if os.environ.get("ANTHROPIC_API_KEY") else "mock"}


@app.get("/suppliers")
def list_suppliers(risk_category: str | None = None, country: str | None = None):
    df = get_scored_df()
    if risk_category:
        df = df[df["predicted_risk_category"] == risk_category]
    if country:
        df = df[df["country"] == country]
    return df.to_dict(orient="records")


@app.get("/suppliers/{supplier_id}")
def get_supplier(supplier_id: str):
    df = get_scored_df()
    row = df[df["supplier_id"] == supplier_id]
    if row.empty:
        raise HTTPException(404, f"Supplier {supplier_id} not found")
    return row.iloc[0].to_dict()


@app.post("/suppliers/{supplier_id}/investigate", response_model=InvestigationReport)
def investigate(supplier_id: str):
    df = get_scored_df()
    row = df[df["supplier_id"] == supplier_id]
    if row.empty:
        raise HTTPException(404, f"Supplier {supplier_id} not found")
    ml_prediction = {
        "predicted_risk_score": float(row.iloc[0]["predicted_risk_score"]),
        "predicted_risk_category": row.iloc[0]["predicted_risk_category"],
    }
    report = agent.investigate_supplier(supplier_id, ml_prediction)
    return report


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Lightweight portfolio Q&A. Deliberately NOT a full RAG-over-LLM pipeline
    for every question -- most questions ('which suppliers in China are
    high risk') are just dataframe filters. We answer those directly with
    pandas (zero tokens, instant). Only if the question can't be resolved by
    a filter do we fall back to the LLM with a SMALL summarized context
    (aggregate stats, not all 250 rows) to keep tokens low.
    """
    df = get_scored_df()
    q = req.question.lower()

    matched = df
    filtered = False
    for country in df["country"].unique():
        if country.lower() in q:
            matched = matched[matched["country"] == country]
            filtered = True
    for cat in ["low", "medium", "high", "critical"]:
        if cat in q:
            matched = matched[matched["predicted_risk_category"].str.lower() == cat]
            filtered = True
    if "single source" in q or "single-source" in q:
        matched = matched[matched["single_source_flag"] == 1]
        filtered = True

    if filtered:
        names = matched["supplier_name"].tolist()[:15]
        answer = (
            f"Found {len(matched)} matching supplier(s). "
            f"Examples: {', '.join(names[:8])}"
            + (" ..." if len(matched) > 8 else "")
        )
        return ChatResponse(answer=answer, matched_supplier_ids=matched["supplier_id"].tolist())

    # Fallback: small aggregate summary, no LLM needed for it
    summary = df["predicted_risk_category"].value_counts().to_dict()
    answer = (
        "I can filter suppliers by country, risk category, or single-source "
        f"dependency. Current portfolio breakdown: {summary}"
    )
    return ChatResponse(answer=answer, matched_supplier_ids=None)
