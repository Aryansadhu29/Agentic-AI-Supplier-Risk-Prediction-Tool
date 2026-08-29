"""
Synthetic Supplier Data Generator
----------------------------------
Why synthetic data?
- No real supplier/vendor data is available (and using real data would raise
  confidentiality issues).
- We need LABELED data to train a supervised risk model, so we generate
  features AND derive a realistic risk label from a weighted formula + noise.
  This keeps the ML model genuinely learnable (not random) while being 100%
  fabricated.

Run:
    python data/generate_synthetic_data.py
Produces:
    data/suppliers.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SUPPLIERS = 250

COUNTRIES = {
    # country: baseline geopolitical/logistics risk (0-100)
    "USA": 15, "Germany": 12, "India": 30, "China": 45, "Vietnam": 38,
    "Mexico": 28, "Brazil": 35, "Taiwan": 40, "Poland": 20, "Turkey": 42,
    "South Africa": 44, "Bangladesh": 48, "Philippines": 33,
}

CATEGORIES = [
    "Electronics", "Raw Metals", "Plastics", "Semiconductors",
    "Packaging", "Textiles", "Chemicals", "Logistics Services",
]

SUPPLIER_NAME_PREFIXES = [
    "Global", "Apex", "Prime", "Summit", "Nova", "Vertex", "Atlas",
    "Pioneer", "Orion", "Meridian", "Ironclad", "Stellar", "Union",
    "Fortress", "Horizon", "Zenith", "Continental", "Precision",
]
SUPPLIER_NAME_SUFFIXES = [
    "Industries", "Manufacturing", "Components", "Supply Co.",
    "Materials", "Technologies", "Logistics", "Group", "Traders",
]


def generate_suppliers(n=N_SUPPLIERS) -> pd.DataFrame:
    rows = []
    countries = list(COUNTRIES.keys())

    for i in range(1, n + 1):
        country = np.random.choice(countries)
        geo_risk_base = COUNTRIES[country]

        on_time_delivery_rate = np.clip(np.random.normal(90, 10), 40, 100)
        quality_defect_ppm = max(0, np.random.exponential(300))
        financial_health_score = np.clip(np.random.normal(65, 20), 5, 100)
        days_payable_outstanding = np.clip(np.random.normal(45, 20), 10, 120)
        single_source_flag = np.random.choice([0, 1], p=[0.7, 0.3])
        geopolitical_risk_index = np.clip(
            geo_risk_base + np.random.normal(0, 8), 0, 100
        )
        past_disruptions_count = np.random.poisson(1.2)
        avg_lead_time_days = np.clip(np.random.normal(30, 15), 3, 120)
        price_volatility_pct = np.clip(np.random.exponential(8), 0, 60)
        esg_score = np.clip(np.random.normal(60, 18), 0, 100)
        years_relationship = np.clip(np.random.exponential(6), 0.2, 30)
        annual_spend_usd = int(np.clip(np.random.lognormal(11, 1.2), 5_000, 20_000_000))

        name = f"{np.random.choice(SUPPLIER_NAME_PREFIXES)} {np.random.choice(SUPPLIER_NAME_SUFFIXES)}"

        rows.append({
            "supplier_id": f"SUP-{i:04d}",
            "supplier_name": name,
            "country": country,
            "category": np.random.choice(CATEGORIES),
            "annual_spend_usd": annual_spend_usd,
            "on_time_delivery_rate": round(on_time_delivery_rate, 1),
            "quality_defect_ppm": round(quality_defect_ppm, 1),
            "financial_health_score": round(financial_health_score, 1),
            "days_payable_outstanding": round(days_payable_outstanding, 1),
            "single_source_flag": single_source_flag,
            "geopolitical_risk_index": round(geopolitical_risk_index, 1),
            "past_disruptions_count": past_disruptions_count,
            "avg_lead_time_days": round(avg_lead_time_days, 1),
            "price_volatility_pct": round(price_volatility_pct, 1),
            "esg_score": round(esg_score, 1),
            "years_relationship": round(years_relationship, 1),
        })

    df = pd.DataFrame(rows)
    df = _compute_risk_label(df)
    return df


def _compute_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive a ground-truth risk score (0-100) from a domain-informed weighted
    formula + random noise. This is what a supply-chain risk SME would
    roughly weight -- it gives the ML model a real signal to learn instead of
    fitting noise, while staying fully synthetic.
    """
    risk = (
        (100 - df["on_time_delivery_rate"]) * 0.45
        + (df["quality_defect_ppm"] / 40)
        + (100 - df["financial_health_score"]) * 0.30
        + df["single_source_flag"] * 9
        + df["geopolitical_risk_index"] * 0.22
        + df["past_disruptions_count"] * 4
        + df["price_volatility_pct"] * 0.35
        + (100 - df["esg_score"]) * 0.08
        - df["years_relationship"] * 0.4  # longer relationship -> slightly lower risk
    )
    risk = risk + np.random.normal(0, 3.5, size=len(df))  # noise
    risk = np.clip(risk, 0, 100)
    df["risk_score"] = risk.round(1)

    def bucket(s):
        if s < 25:
            return "Low"
        elif s < 42:
            return "Medium"
        elif s < 60:
            return "High"
        else:
            return "Critical"

    df["risk_category"] = df["risk_score"].apply(bucket)
    return df


if __name__ == "__main__":
    df = generate_suppliers()
    out_path = "data/suppliers.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic suppliers -> {out_path}")
    print(df["risk_category"].value_counts())
