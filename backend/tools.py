"""
Agent Tools
-----------
These are the "actions" the agent can take when investigating a supplier.
In a real PwC/enterprise deployment these would call real systems:
    - get_news_sentiment      -> a news/NLP API (e.g. GDELT, NewsAPI + sentiment model)
    - get_financial_trend     -> D&B / credit bureau / Moody's API
    - get_logistics_status    -> a freight/ports API or internal ERP
    - get_ml_risk_assessment  -> our own trained model (ml_model.py)

For this assignment we SIMULATE the first three with seeded, deterministic
"synthetic" responses (seeded by supplier_id, so a given supplier always
returns the same simulated news/financials -- useful for a stable demo).

Design note: each tool returns a small, structured JSON object -- not a wall
of prose. This keeps the number of tokens the agent has to read down to the
essentials, which directly reduces LLM cost and latency (see report.md,
"Token & Latency Optimization").
"""

import hashlib
import random
import pandas as pd

DATA_PATH = "data/suppliers.csv"


def _seeded_rng(supplier_id: str) -> random.Random:
    seed = int(hashlib.sha256(supplier_id.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def get_supplier_record(supplier_id: str) -> dict:
    """Tool: fetch the supplier's structured profile from the master data set."""
    df = pd.read_csv(DATA_PATH)
    row = df[df["supplier_id"] == supplier_id]
    if row.empty:
        return {"error": f"Supplier {supplier_id} not found"}
    return row.iloc[0].to_dict()


NEWS_TEMPLATES_NEGATIVE = [
    "Regional labor strikes reported near supplier's main facility",
    "Local currency devaluation raising input cost pressure",
    "Regulatory investigation opened into industry compliance practices",
    "Port congestion delaying outbound shipments in the region",
    "Raw material shortage reported across the supplier's category",
]
NEWS_TEMPLATES_NEUTRAL = [
    "Supplier announced routine facility maintenance schedule",
    "No significant news detected in the last 90 days",
    "Industry trade publication mentioned supplier in a market overview",
]
NEWS_TEMPLATES_POSITIVE = [
    "Supplier announced new ISO certification",
    "Supplier expanded production capacity with new facility",
    "Positive quarterly earnings reported by supplier's parent company",
]


def get_news_sentiment(supplier_id: str, supplier_name: str, country: str) -> dict:
    """Tool: simulated news/sentiment scan for the supplier and its region."""
    rng = _seeded_rng(supplier_id + "news")
    sentiment_bucket = rng.choices(
        ["negative", "neutral", "positive"], weights=[0.35, 0.4, 0.25]
    )[0]
    templates = {
        "negative": NEWS_TEMPLATES_NEGATIVE,
        "neutral": NEWS_TEMPLATES_NEUTRAL,
        "positive": NEWS_TEMPLATES_POSITIVE,
    }[sentiment_bucket]
    headlines = rng.sample(templates, k=min(2, len(templates)))
    sentiment_score = {
        "negative": round(rng.uniform(-1.0, -0.3), 2),
        "neutral": round(rng.uniform(-0.2, 0.2), 2),
        "positive": round(rng.uniform(0.3, 1.0), 2),
    }[sentiment_bucket]
    return {
        "supplier_id": supplier_id,
        "sentiment_label": sentiment_bucket,
        "sentiment_score": sentiment_score,
        "headlines": headlines,
        "region_context": f"{country} regional stability monitor: "
        + ("elevated concern" if sentiment_bucket == "negative" else "normal"),
    }


def get_financial_trend(supplier_id: str, financial_health_score: float) -> dict:
    """Tool: simulated quarter-over-quarter financial trend."""
    rng = _seeded_rng(supplier_id + "fin")
    trend_direction = rng.choices(["declining", "stable", "improving"], weights=[0.3, 0.45, 0.25])[0]
    delta = {
        "declining": -round(rng.uniform(3, 12), 1),
        "stable": round(rng.uniform(-1.5, 1.5), 1),
        "improving": round(rng.uniform(3, 10), 1),
    }[trend_direction]
    return {
        "supplier_id": supplier_id,
        "current_financial_health_score": financial_health_score,
        "trend_direction": trend_direction,
        "qoq_change_pts": delta,
        "credit_watchlist": trend_direction == "declining" and financial_health_score < 50,
    }


def get_logistics_status(supplier_id: str, country: str, single_source_flag: int) -> dict:
    """Tool: simulated logistics / concentration-risk snapshot."""
    rng = _seeded_rng(supplier_id + "log")
    disruption_probability = round(rng.uniform(0.05, 0.6), 2)
    return {
        "supplier_id": supplier_id,
        "region": country,
        "estimated_disruption_probability_90d": disruption_probability,
        "single_source_dependency": bool(single_source_flag),
        "alternate_suppliers_identified": 0 if single_source_flag else rng.randint(1, 4),
    }


TOOL_REGISTRY = {
    "get_supplier_record": get_supplier_record,
    "get_news_sentiment": get_news_sentiment,
    "get_financial_trend": get_financial_trend,
    "get_logistics_status": get_logistics_status,
}
