"""
Streamlit Frontend
--------------------
A deliberately simple, single-file UI. Three tabs:
  1. Dashboard   - all suppliers, ML-scored, instantly (no LLM calls)
  2. Investigate - pick one flagged supplier, run the AGENT, see its report
  3. Ask         - natural language questions over the portfolio

Run (after starting the backend):
    streamlit run frontend/app.py
"""

import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Supplier Risk Agent", layout="wide")

RISK_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#ef6c00", "Critical": "#c62828"}


@st.cache_data(ttl=30)
def fetch_suppliers():
    r = requests.get(f"{API_URL}/suppliers", timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def investigate_supplier(supplier_id: str):
    r = requests.post(f"{API_URL}/suppliers/{supplier_id}/investigate", timeout=60)
    r.raise_for_status()
    return r.json()


def ask_question(question: str):
    r = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=30)
    r.raise_for_status()
    return r.json()


st.title("🔗 Supplier Risk Prediction Agent")
st.caption(
    "ML model scores every supplier instantly. The AI agent is only invoked "
    "on demand to investigate a specific flagged supplier -- keeping cost and "
    "latency low. See README for architecture."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    mode_badge = "🟢 LIVE (Claude API)" if health["llm_mode"] == "live" else "🟡 MOCK (no API key set — rule-based agent)"
    st.info(f"Agent mode: {mode_badge}")
except Exception:
    st.error(f"Cannot reach backend at {API_URL}. Start it with: `uvicorn backend.main:app --reload`")
    st.stop()

tab_dashboard, tab_investigate, tab_ask = st.tabs(["📊 Dashboard", "🕵️ Investigate", "💬 Ask"])

with tab_dashboard:
    df = fetch_suppliers()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Suppliers", len(df))
    col2.metric("Critical Risk", int((df["predicted_risk_category"] == "Critical").sum()))
    col3.metric("High Risk", int((df["predicted_risk_category"] == "High").sum()))
    col4.metric(
        "Total Spend at High+ Risk",
        f"${df[df['predicted_risk_category'].isin(['High','Critical'])]['annual_spend_usd'].sum():,.0f}",
    )

    st.subheader("Risk Distribution")
    dist = df["predicted_risk_category"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
    st.bar_chart(dist)

    st.subheader("Supplier Portfolio")
    colf1, colf2 = st.columns(2)
    with colf1:
        risk_filter = st.multiselect(
            "Filter by risk category", ["Low", "Medium", "High", "Critical"],
            default=["High", "Critical"],
        )
    with colf2:
        country_filter = st.multiselect("Filter by country", sorted(df["country"].unique()))

    view = df.copy()
    if risk_filter:
        view = view[view["predicted_risk_category"].isin(risk_filter)]
    if country_filter:
        view = view[view["country"].isin(country_filter)]

    view = view.sort_values("predicted_risk_score", ascending=False)
    st.dataframe(
        view[[
            "supplier_id", "supplier_name", "country", "category",
            "predicted_risk_category", "predicted_risk_score",
            "prediction_confidence", "annual_spend_usd",
        ]],
        use_container_width=True,
        hide_index=True,
    )

with tab_investigate:
    st.write("Pick a supplier (flagged suppliers are pre-sorted to the top) and let the agent investigate.")
    df = fetch_suppliers().sort_values("predicted_risk_score", ascending=False)
    options = [
        f"{row.supplier_id} — {row.supplier_name} ({row.predicted_risk_category}, score {row.predicted_risk_score})"
        for row in df.itertuples()
    ]
    choice = st.selectbox("Supplier", options)
    supplier_id = choice.split(" — ")[0]

    row = df[df["supplier_id"] == supplier_id].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", row["predicted_risk_score"])
    c2.metric("Risk Category", row["predicted_risk_category"])
    c3.metric("On-Time Delivery", f"{row['on_time_delivery_rate']}%")
    c4.metric("Financial Health", row["financial_health_score"])

    if st.button("🔍 Run Agent Investigation", type="primary"):
        with st.spinner("Agent is calling tools (news, financials, logistics) and reasoning..."):
            report = investigate_supplier(supplier_id)

        badge = "📦 cached result" if report.get("from_cache") else "🆕 freshly generated"
        st.caption(badge)

        st.markdown(f"### Recommendation: **{report['recommendation']}**")
        st.write(report["risk_summary"])

        st.markdown("**Key risk drivers:**")
        for d in report["key_risk_drivers"]:
            st.markdown(f"- {d}")

        st.markdown("**Recommended actions:**")
        for a in report["recommended_actions"]:
            st.markdown(f"- {a}")

with tab_ask:
    st.write("Ask a question about the supplier portfolio, e.g. *\"Which suppliers in China are high risk?\"*")
    q = st.text_input("Your question")
    if st.button("Ask") and q:
        with st.spinner("Thinking..."):
            resp = ask_question(q)
        st.write(resp["answer"])
