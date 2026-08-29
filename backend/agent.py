"""
Agentic Investigation Layer
----------------------------
This is the "agentic AI" part of the assignment: given ONE flagged supplier,
the agent autonomously decides which tools to call (news, financials,
logistics, ML score), reads the results, and synthesizes a structured risk
report with a recommendation -- the way a human risk analyst would.

Two modes, selected automatically:
  1. LIVE mode  -- uses the Anthropic Claude API with native tool-use
     (function calling). Real agentic loop: the model chooses tools,
     we execute them, feed results back, repeat until it answers.
  2. MOCK mode  -- if no ANTHROPIC_API_KEY is set, we run the same tool
     calls deterministically and generate the report with a rule-based
     template instead of an LLM. This means the whole app is demoable
     end-to-end with zero API key / zero internet, which matters a lot
     for a live interview demo.

TOKEN & LATENCY OPTIMIZATION (see report.md for full write-up):
  - Model used: claude-haiku (fast + cheap) by default, not a large model --
    this task is tool-orchestration + summarization, not deep reasoning.
  - Tool outputs are pre-summarized, structured JSON (tools.py), never raw
    dumps -- smaller input tokens per turn.
  - Hard cap of MAX_TOOL_ITERATIONS turns to bound worst-case latency/cost.
  - System prompt enforces a compact JSON output schema (max_tokens capped)
    instead of free-form prose -- smaller, predictable output tokens.
  - Results are cached in-memory per supplier_id so re-opening the same
    supplier in the UI costs zero extra LLM calls.
"""

import os
import json
from typing import Optional

from backend import tools

MODEL_NAME = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")
MAX_TOOL_ITERATIONS = 4
MAX_OUTPUT_TOKENS = 700

_REPORT_CACHE: dict[str, dict] = {}

TOOL_SPECS = [
    {
        "name": "get_news_sentiment",
        "description": "Scan recent news/sentiment for this supplier and its region.",
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "supplier_name": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["supplier_id", "supplier_name", "country"],
        },
    },
    {
        "name": "get_financial_trend",
        "description": "Get the quarter-over-quarter financial health trend for this supplier.",
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "financial_health_score": {"type": "number"},
            },
            "required": ["supplier_id", "financial_health_score"],
        },
    },
    {
        "name": "get_logistics_status",
        "description": "Get logistics disruption probability and single-source concentration risk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "supplier_id": {"type": "string"},
                "country": {"type": "string"},
                "single_source_flag": {"type": "integer"},
            },
            "required": ["supplier_id", "country", "single_source_flag"],
        },
    },
]

SYSTEM_PROMPT = """You are a supply-chain risk analyst agent for a procurement team.
You are given one supplier's structured profile and its ML-predicted risk score.
Use the available tools (news sentiment, financial trend, logistics status) to
gather extra context, then respond with ONLY a compact JSON object, no prose
outside the JSON, with exactly these keys:
{
  "risk_summary": "2-3 sentence plain-English summary of why this supplier is risky or not",
  "key_risk_drivers": ["driver 1", "driver 2", "driver 3"],
  "recommendation": "one of: Monitor | Diversify Sourcing | Renegotiate Terms | Immediate Escalation",
  "recommended_actions": ["action 1", "action 2"]
}
Be concise. Do not call more than 3 tools total."""


def _get_llm_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic  # lazy import: keeps mock mode dependency-free
    return anthropic.Anthropic(api_key=api_key)


def investigate_supplier(supplier_id: str, ml_prediction: dict, use_cache: bool = True) -> dict:
    if use_cache and supplier_id in _REPORT_CACHE:
        cached = dict(_REPORT_CACHE[supplier_id])
        cached["from_cache"] = True
        return cached

    record = tools.get_supplier_record(supplier_id)
    client = _get_llm_client()

    if client is None:
        report = _mock_investigate(record, ml_prediction)
    else:
        report = _live_investigate(client, record, ml_prediction)

    report["from_cache"] = False
    _REPORT_CACHE[supplier_id] = report
    return report


# ---------------------------------------------------------------------------
# LIVE mode: real agentic tool-use loop against the Anthropic API
# ---------------------------------------------------------------------------
def _live_investigate(client, record: dict, ml_prediction: dict) -> dict:
    user_context = {
        "supplier_profile": record,
        "ml_risk_assessment": ml_prediction,
    }
    messages = [
        {
            "role": "user",
            "content": (
                "Investigate this supplier and produce the JSON report.\n\n"
                f"{json.dumps(user_context)}"
            ),
        }
    ]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SPECS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return _parse_json_report(final_text)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = tools.TOOL_REGISTRY.get(block.name)
            result = fn(**block.input) if fn else {"error": "unknown tool"}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return {
        "risk_summary": "Agent exceeded max tool iterations; returning ML-only assessment.",
        "key_risk_drivers": [],
        "recommendation": "Monitor",
        "recommended_actions": [],
    }


def _parse_json_report(text: str) -> dict:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "risk_summary": text.strip()[:400],
            "key_risk_drivers": [],
            "recommendation": "Monitor",
            "recommended_actions": [],
        }


# ---------------------------------------------------------------------------
# MOCK mode: same tool calls, deterministic rule-based synthesis.
# Lets the whole product be demoed with zero API key / zero internet.
# ---------------------------------------------------------------------------
def _mock_investigate(record: dict, ml_prediction: dict) -> dict:
    news = tools.get_news_sentiment(record["supplier_id"], record["supplier_name"], record["country"])
    fin = tools.get_financial_trend(record["supplier_id"], record["financial_health_score"])
    log = tools.get_logistics_status(record["supplier_id"], record["country"], record["single_source_flag"])

    drivers = []
    if fin["trend_direction"] == "declining":
        drivers.append(f"Financial health declining ({fin['qoq_change_pts']} pts QoQ)")
    if news["sentiment_label"] == "negative":
        drivers.append(f"Negative regional news sentiment ({news['headlines'][0]})")
    if log["single_source_dependency"]:
        drivers.append("Single-source dependency with no qualified backup supplier")
    if record["quality_defect_ppm"] > 400:
        drivers.append(f"Elevated defect rate ({record['quality_defect_ppm']:.0f} ppm)")
    if record["on_time_delivery_rate"] < 85:
        drivers.append(f"On-time delivery below target ({record['on_time_delivery_rate']:.1f}%)")
    if not drivers:
        drivers.append("No major red flags detected in current data")

    category = ml_prediction["predicted_risk_category"]
    if category == "Critical":
        recommendation = "Immediate Escalation"
        actions = ["Escalate to procurement risk committee this week",
                   "Begin qualifying an alternate supplier in parallel"]
    elif category == "High":
        recommendation = "Diversify Sourcing" if log["single_source_dependency"] else "Renegotiate Terms"
        actions = ["Request updated financial statements",
                   "Evaluate 1-2 alternate suppliers for this category"]
    elif category == "Medium":
        recommendation = "Monitor"
        actions = ["Add to quarterly risk review", "Track financial trend next quarter"]
    else:
        recommendation = "Monitor"
        actions = ["No immediate action required", "Continue standard quarterly monitoring"]

    summary = (
        f"{record['supplier_name']} is assessed as {category} risk "
        f"(score {ml_prediction['predicted_risk_score']}/100). "
        f"Primary concerns: {', '.join(drivers[:2]).lower()}."
    )

    return {
        "risk_summary": summary,
        "key_risk_drivers": drivers[:4],
        "recommendation": recommendation,
        "recommended_actions": actions,
        "_tool_calls_made": ["get_news_sentiment", "get_financial_trend", "get_logistics_status"],
        "_mode": "mock (no ANTHROPIC_API_KEY set)",
    }
