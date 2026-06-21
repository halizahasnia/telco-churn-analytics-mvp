"""Raw data ingestion: read the source CSV as-is, no transformation."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it from Kaggle "
            "(blastchar/telco-customer-churn) and place it there."
        )
    return pd.read_csv(path, dtype=str)
