"""Pipeline orchestrator: Raw CSV -> Staging (raw strings) -> Cleaned -> Curated SQLite.

Run as: python -m src.pipeline.build_db
Produces:
  - data/processed/quality_report.json  (data quality check results)
  - data/processed/telco_churn.db       (analytics-ready SQLite database, table `customers`)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.pipeline.clean import clean
from src.pipeline.ingest import load_raw
from src.pipeline.quality_checks import run_quality_checks

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DB_PATH = PROCESSED_DIR / "telco_churn.db"
QUALITY_REPORT_PATH = PROCESSED_DIR / "quality_report.json"


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    staging_df = load_raw()
    report = run_quality_checks(staging_df)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2))
    failed = [r for r in report if not r["passed"]]
    print(f"Quality checks: {len(report) - len(failed)}/{len(report)} passed.")
    for r in failed:
        print(f"  [HANDLED IN CLEANING] {r['check']}: {r['detail']}")

    curated_df = clean(staging_df)

    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    curated_df.to_sql("customers", conn, index=False, if_exists="replace")
    conn.execute("CREATE INDEX idx_contract ON customers(contract)")
    conn.execute("CREATE INDEX idx_churn ON customers(churn)")
    conn.commit()
    conn.close()

    print(f"Curated dataset written to {DB_PATH} ({len(curated_df)} rows, table `customers`).")


if __name__ == "__main__":
    main()
