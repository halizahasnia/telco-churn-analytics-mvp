"""Data quality checks run on the raw (staging) dataframe before transformation.

Each check returns a (passed, detail) tuple. Kept as plain functions instead of
a rules engine since the dataset is small and the rule set is fixed.
"""
from __future__ import annotations

import pandas as pd

EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
}


def check_schema(df: pd.DataFrame) -> tuple[bool, str]:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing expected columns: {sorted(missing)}"
    return True, "All expected columns present."


def check_primary_key_unique(df: pd.DataFrame) -> tuple[bool, str]:
    dupes = df["customerID"].duplicated().sum()
    if dupes:
        return False, f"{dupes} duplicate customerID rows found."
    return True, "customerID is unique across all rows."


def check_churn_domain(df: pd.DataFrame) -> tuple[bool, str]:
    bad = set(df["Churn"].unique()) - {"Yes", "No"}
    if bad:
        return False, f"Unexpected Churn values: {bad}"
    return True, "Churn column only contains Yes/No."


def check_total_charges_numeric(df: pd.DataFrame) -> tuple[bool, str]:
    blank = df["TotalCharges"].str.strip().eq("").sum()
    non_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce").isna().sum()
    if non_numeric:
        return False, (
            f"{non_numeric} rows have a non-numeric TotalCharges "
            f"({blank} of which are blank strings, typically tenure=0 new customers)."
        )
    return True, "TotalCharges parses cleanly to numeric."


def check_monthly_charges_range(df: pd.DataFrame) -> tuple[bool, str]:
    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    if (charges <= 0).any() or charges.isna().any():
        return False, "MonthlyCharges has non-positive or non-numeric values."
    return True, "MonthlyCharges values are all positive numbers."


CHECKS = [
    check_schema,
    check_primary_key_unique,
    check_churn_domain,
    check_total_charges_numeric,
    check_monthly_charges_range,
]


def run_quality_checks(df: pd.DataFrame) -> list[dict]:
    """Run all checks and return a report. Does not raise; failures like the
    TotalCharges issue are expected and handled in the cleaning step."""
    report = []
    for check in CHECKS:
        passed, detail = check(df)
        report.append({"check": check.__name__, "passed": passed, "detail": detail})
    return report
