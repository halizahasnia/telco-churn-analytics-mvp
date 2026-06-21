"""Cleaning and transformation: staging (raw strings) to analytics-ready dataframe.

TotalCharges has 11 blank-string rows, all with tenure=0 (brand-new customers
who haven't been billed yet). They're set to 0.0 instead of dropped, since
dropping real customers would skew churn stats toward longer-tenure ones.

tenure_bucket and num_services are derived columns for segmentation, not
present in the raw dataset.
"""
from __future__ import annotations

import pandas as pd

YES_NO_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]

ADDON_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, 60, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-60", "61-72"]


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    for col in YES_NO_COLUMNS:
        df[col] = (df[col] == "Yes").astype(int)

    df["churn"] = (df["Churn"] == "Yes").astype(int)

    df["tenure_bucket"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)

    df["num_services"] = df[ADDON_SERVICE_COLUMNS].eq("Yes").sum(axis=1) + df["PhoneService"]

    df = df.rename(columns={
        "customerID": "customer_id",
        "gender": "gender",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure_months",
        "PhoneService": "phone_service",
        "MultipleLines": "multiple_lines",
        "InternetService": "internet_service",
        "OnlineSecurity": "online_security",
        "OnlineBackup": "online_backup",
        "DeviceProtection": "device_protection",
        "TechSupport": "tech_support",
        "StreamingTV": "streaming_tv",
        "StreamingMovies": "streaming_movies",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
    })

    return df.drop(columns=["Churn"])
