"""Baseline churn prediction model, added on top of the descriptive analytics
in insights.py to answer a forward-looking question: which currently active
customers are most likely to churn next?

This is a plain logistic regression, no hyperparameter search and no
cross-validation beyond one train/test split, trained on the full historical
snapshot so it won't catch drift over time. Good enough to rank relative
churn risk for an MVP, not meant as a calibrated production score.

Scores go into a separate `churn_risk_scores` table rather than into
`customers`, so the curated table stays purely ETL output.

The `predicted_churn` flag does not use the usual 0.5 cutoff. Instead the
threshold is picked by maximizing expected net retention value on the test
set, using the RETENTION_* cost assumptions below.

Run as: python -m src.analytics.churn_model
Produces:
  - data/processed/model_metrics.json (test metrics, top coefficients,
    threshold curve and recommendation)
  - `churn_risk_scores` table in telco_churn.db
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DB_PATH = PROCESSED_DIR / "telco_churn.db"
METRICS_PATH = PROCESSED_DIR / "model_metrics.json"

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "num_services"]
CATEGORICAL_FEATURES = [
    "gender", "senior_citizen", "partner", "dependents", "internet_service",
    "contract", "payment_method", "paperless_billing", "multiple_lines",
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]
TARGET = "churn"

# total_charges was dropped: it's highly correlated with tenure_months (r=0.83)
# since total_charges roughly equals tenure_months times monthly_charges. Keeping
# it made its own coefficient unreliable (it came out positive, which reads as
# "customers who paid more churn more," an artifact of the correlation rather
# than a real effect) without adding meaningful predictive power.

# Cost assumptions for the threshold search below. No real campaign-cost data
# exists for this dataset, so these are guesses, adjust them to match a real
# retention program.
RETENTION_OFFER_COST = 15.0      # USD per outreach (call/email/discount admin)
RETENTION_SUCCESS_RATE = 0.30    # odds a contacted at-risk customer actually stays
RETAINED_VALUE_MONTHS = 12       # months of billing a retained customer preserves
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def _evaluate_thresholds(y_test: pd.Series, y_proba: np.ndarray, monthly_charges_test: pd.Series) -> list[dict]:
    """Score every candidate threshold by precision, recall, and net dollar
    value: revenue saved from true positives minus outreach cost for
    everyone flagged, false positives included."""
    y_test_arr = y_test.to_numpy()
    charges_arr = monthly_charges_test.to_numpy()
    n_actual_churn = y_test_arr.sum()

    curve = []
    for threshold in THRESHOLD_GRID:
        pred = (y_proba >= threshold).astype(int)
        tp_mask = (pred == 1) & (y_test_arr == 1)
        fp_mask = (pred == 1) & (y_test_arr == 0)
        n_flagged = int(pred.sum())
        tp = int(tp_mask.sum())

        precision = tp / n_flagged if n_flagged > 0 else 0.0
        recall = tp / n_actual_churn if n_actual_churn > 0 else 0.0

        benefit = RETENTION_SUCCESS_RATE * RETAINED_VALUE_MONTHS * charges_arr[tp_mask].sum()
        cost = RETENTION_OFFER_COST * n_flagged
        net_value = benefit - cost

        curve.append({
            "threshold": float(threshold),
            "customers_flagged": n_flagged,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "expected_net_value": round(float(net_value), 2),
        })
    return curve


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_score() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    eval_pipeline = _build_pipeline()
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)
    y_proba = eval_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        "test_set_size": len(y_test),
        "train_set_size": len(y_train),
        "model": "LogisticRegression(class_weight=balanced)",
        "note": (
            "Baseline model on a single train/test split, no hyperparameter tuning. "
            "Intended to rank relative churn risk, not as a calibrated production score."
        ),
    }

    # Top coefficients for explainability (which features push risk up/down most).
    feature_names = eval_pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = eval_pipeline.named_steps["model"].coef_[0]
    top_idx = coefs.argsort()
    top_risk_increasing = [
        {"feature": feature_names[i], "coef": round(float(coefs[i]), 3)}
        for i in top_idx[-5:][::-1]
    ]
    top_risk_decreasing = [
        {"feature": feature_names[i], "coef": round(float(coefs[i]), 3)}
        for i in top_idx[:5]
    ]
    metrics["top_risk_increasing_features"] = top_risk_increasing
    metrics["top_risk_decreasing_features"] = top_risk_decreasing

    # pick the threshold with the highest expected net value instead of
    # defaulting to 0.5
    threshold_curve = _evaluate_thresholds(y_test, y_proba, X_test["monthly_charges"])
    best = max(threshold_curve, key=lambda r: r["expected_net_value"])
    metrics["business_value_analysis"] = {
        "assumptions": {
            "retention_offer_cost_usd": RETENTION_OFFER_COST,
            "retention_success_rate": RETENTION_SUCCESS_RATE,
            "retained_value_months": RETAINED_VALUE_MONTHS,
            "note": "Guessed values, not from real campaign data. Adjust the constants in churn_model.py to match actual retention economics.",
        },
        "threshold_curve": threshold_curve,
        "recommended_threshold": best["threshold"],
        "recommended_threshold_rationale": (
            f"Maximizes expected net retention value (${best['expected_net_value']:,.0f} on the "
            f"{len(y_test)}-customer test set) rather than defaulting to 0.5; flags "
            f"{best['customers_flagged']} test customers at precision={best['precision']}, recall={best['recall']}."
        ),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Model metrics written to {METRICS_PATH}")
    print(f"  accuracy={metrics['accuracy']} precision={metrics['precision']} "
          f"recall={metrics['recall']} roc_auc={metrics['roc_auc']}")
    print(f"  business-optimal threshold={best['threshold']} "
          f"(expected_net_value=${best['expected_net_value']:,.0f} on test set)")

    # retrain on everything for the actual risk-scoring table, now that the
    # held-out metrics above are recorded
    final_pipeline = _build_pipeline()
    final_pipeline.fit(X, y)
    df["churn_probability"] = final_pipeline.predict_proba(X)[:, 1]
    df["predicted_churn"] = (df["churn_probability"] >= best["threshold"]).astype(int)

    scores = df[["customer_id", "churn", "churn_probability", "predicted_churn"]].copy()
    scores["churn_probability"] = scores["churn_probability"].round(4)
    scores.to_sql("churn_risk_scores", conn, index=False, if_exists="replace")
    conn.commit()
    conn.close()
    print(f"Wrote churn_probability for {len(scores)} customers to `churn_risk_scores` table "
          f"(predicted_churn flag uses threshold={best['threshold']}).")


if __name__ == "__main__":
    train_and_score()
