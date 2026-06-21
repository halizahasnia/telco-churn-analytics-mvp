"""Guardrailed SQL execution against the curated analytics database.

This is the only way the LLM ever touches the data, so it's locked down on
purpose instead of acting like a general SQL executor: single SELECT
statements only, table references parsed and checked against an allowlist
(not just a substring match), DDL/DML/PRAGMA keywords blocked, the connection
opened read-only at the SQLite level as a second line of defense, and a row
cap so a runaway query can't flood the LLM context.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "telco_churn.db"

ALLOWED_TABLES = {"customers", "churn_risk_scores"}
ROW_LIMIT = 200

BLOCKED_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|VACUUM|REPLACE|TRIGGER)\b",
    re.IGNORECASE,
)

TABLE_REF_PATTERN = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)


class UnsafeQueryError(Exception):
    pass


def validate_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise UnsafeQueryError("Multiple statements are not allowed.")
    if not sql.upper().startswith("SELECT"):
        raise UnsafeQueryError("Only SELECT statements are allowed.")
    if BLOCKED_KEYWORDS.search(sql):
        raise UnsafeQueryError("Query contains a disallowed keyword.")
    referenced_tables = {t.lower() for t in TABLE_REF_PATTERN.findall(sql)}
    if not referenced_tables:
        raise UnsafeQueryError("Query must reference at least one known table.")
    disallowed = referenced_tables - ALLOWED_TABLES
    if disallowed:
        raise UnsafeQueryError(f"Query references disallowed table(s): {disallowed}")
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT {ROW_LIMIT}"
    return sql


def run_query(sql: str) -> list[dict]:
    safe_sql = validate_sql(sql)
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(safe_sql)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchmany(ROW_LIMIT)
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


SCHEMA_DESCRIPTION = """\
Table `customers` (one row per customer, 7043 rows):
  customer_id TEXT, gender TEXT, senior_citizen INT(0/1), partner INT(0/1),
  dependents INT(0/1), tenure_months INT, phone_service INT(0/1),
  multiple_lines TEXT, internet_service TEXT('DSL'/'Fiber optic'/'No'),
  online_security TEXT, online_backup TEXT, device_protection TEXT,
  tech_support TEXT, streaming_tv TEXT, streaming_movies TEXT,
  contract TEXT('Month-to-month'/'One year'/'Two year'),
  paperless_billing INT(0/1), payment_method TEXT, monthly_charges REAL,
  total_charges REAL, churn INT(0/1), tenure_bucket TEXT, num_services INT

Table `churn_risk_scores` (one row per customer, from a baseline ML model, join on customer_id):
  customer_id TEXT, churn INT(0/1) [actual, historical], churn_probability REAL(0-1)
  [model-predicted likelihood of churn; use this for "who is at risk" questions,
  filtering churn_probability descending and customers.churn = 0 for currently active customers]
"""
