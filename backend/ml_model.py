"""
ML Risk Scoring Layer
----------------------
WHY THIS EXISTS (design decision, ask about this in the interview):
Sending every supplier's raw data to an LLM to "guess" a risk score would be
slow, expensive, non-deterministic, and hard to audit. Risk scoring from
structured numeric features is a textbook supervised ML problem, so we solve
it with a classical model (RandomForest). This gives us:
  - Millisecond inference for all suppliers at once (bulk dashboard load)
  - Deterministic, explainable, versionable scores
  - Zero LLM token cost for the 90% of the workflow that is just "show me
    the numbers"

The LLM/agent layer (agent.py) is reserved for the 10% of the workflow that
actually needs reasoning, synthesis, and natural language: investigating a
SPECIFIC flagged supplier and producing a narrative report + recommendation.
This is the "tiered architecture" pattern -- cheap deterministic model first,
expensive generative model only on demand.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

FEATURE_COLUMNS = [
    "on_time_delivery_rate",
    "quality_defect_ppm",
    "financial_health_score",
    "days_payable_outstanding",
    "single_source_flag",
    "geopolitical_risk_index",
    "past_disruptions_count",
    "avg_lead_time_days",
    "price_volatility_pct",
    "esg_score",
    "years_relationship",
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLF_PATH = os.path.join(MODEL_DIR, "risk_classifier.joblib")
REG_PATH = os.path.join(MODEL_DIR, "risk_regressor.joblib")


def train(csv_path="data/suppliers.csv"):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS]
    y_cat = df["risk_category"]
    y_score = df["risk_score"]

    X_train, X_test, ycat_train, ycat_test, yscore_train, yscore_test = train_test_split(
        X, y_cat, y_score, test_size=0.2, random_state=42, stratify=y_cat
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, ycat_train)
    acc = accuracy_score(ycat_test, clf.predict(X_test))

    reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    reg.fit(X_train, yscore_train)
    mae = mean_absolute_error(yscore_test, reg.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, CLF_PATH)
    joblib.dump(reg, REG_PATH)

    print(f"Classifier hold-out accuracy: {acc:.3f}")
    print(f"Regressor hold-out MAE: {mae:.2f} risk points")
    print(f"Feature importances (regressor):")
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, reg.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {feat:28s} {imp:.3f}")

    return clf, reg


class RiskModel:
    """Thin wrapper used by the API layer. Loads once, scores in bulk."""

    def __init__(self):
        if not (os.path.exists(CLF_PATH) and os.path.exists(REG_PATH)):
            train()
        self.clf = joblib.load(CLF_PATH)
        self.reg = joblib.load(REG_PATH)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS]
        df = df.copy()
        df["predicted_risk_score"] = self.reg.predict(X).round(1)
        df["predicted_risk_category"] = self.clf.predict(X)
        # class probabilities -> useful for "confidence" in the UI
        proba = self.clf.predict_proba(X)
        df["prediction_confidence"] = proba.max(axis=1).round(2)
        return df

    def score_one(self, row: dict) -> dict:
        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        score = float(self.reg.predict(X)[0])
        category = self.clf.predict(X)[0]
        confidence = float(self.clf.predict_proba(X).max())
        return {
            "predicted_risk_score": round(score, 1),
            "predicted_risk_category": category,
            "prediction_confidence": round(confidence, 2),
        }


if __name__ == "__main__":
    train()
