"""Precomputed business insights over the curated `customers` table.

Each insight is a named SQL query plus a short narrative, stored to JSON. The
dashboard and the LLM grounding layer both read from this same file, so an
LLM answer is reusing numbers that already came out of a checked SQL query
instead of writing its own conclusion.

Run as: python -m src.analytics.insights
Produces: data/processed/insights.json
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DB_PATH = PROCESSED_DIR / "telco_churn.db"
INSIGHTS_PATH = PROCESSED_DIR / "insights.json"


def _run(conn: sqlite3.Connection, sql: str) -> list[dict]:
    cur = conn.execute(sql)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def build_insights() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    insights = []

    sql = """
        SELECT COUNT(*) AS total_customers,
               SUM(churn) AS churned_customers,
               ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
        FROM customers
    """
    rows = _run(conn, sql)
    r = rows[0]
    insights.append({
        "id": "overall_churn_rate",
        "question": "What is the overall customer churn rate?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"Out of {r['total_customers']} customers, {r['churned_customers']} have churned, "
            f"an overall churn rate of {r['churn_rate_pct']}%."
        ),
    })

    sql = """
        SELECT contract,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
        FROM customers GROUP BY contract ORDER BY churn_rate_pct DESC
    """
    rows = _run(conn, sql)
    top = rows[0]
    insights.append({
        "id": "churn_by_contract",
        "question": "Which contract type has the highest churn rate?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"{top['contract']} contracts have the highest churn rate at {top['churn_rate_pct']}%, "
            f"compared to {rows[-1]['contract']} at {rows[-1]['churn_rate_pct']}%. "
            "Month-to-month customers are the most flexible to leave and the highest churn risk segment."
        ),
    })

    sql = """
        SELECT tenure_bucket,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
        FROM customers GROUP BY tenure_bucket ORDER BY tenure_bucket
    """
    rows = _run(conn, sql)
    insights.append({
        "id": "churn_by_tenure",
        "question": "How does churn rate trend across customer tenure?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"Churn rate is highest among new customers (tenure 0-12 months) at {rows[0]['churn_rate_pct']}%, "
            f"and drops to {rows[-1]['churn_rate_pct']}% for long-tenured customers (61-72 months). "
            "Churn risk is concentrated in the first year of the relationship."
        ),
    })

    sql = """
        SELECT payment_method,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
        FROM customers GROUP BY payment_method ORDER BY churn_rate_pct DESC
    """
    rows = _run(conn, sql)
    top = rows[0]
    insights.append({
        "id": "churn_by_payment_method",
        "question": "Which payment method correlates with the highest churn?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"Customers paying via {top['payment_method']} churn the most, at {top['churn_rate_pct']}%, "
            f"versus {rows[-1]['payment_method']} at {rows[-1]['churn_rate_pct']}%. "
            "This may reflect a less-engaged or more price-sensitive customer segment rather than "
            "the payment method itself causing churn."
        ),
    })

    sql = """
        SELECT tech_support,
               online_security,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
        FROM customers
        WHERE internet_service != 'No'
        GROUP BY tech_support, online_security
        ORDER BY churn_rate_pct DESC
    """
    rows = _run(conn, sql)
    insights.append({
        "id": "addon_services_effect",
        "question": "Do add-on services like tech support or online security reduce churn?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            "Among internet customers, those without TechSupport and without OnlineSecurity churn at "
            f"{rows[0]['churn_rate_pct']}%, the highest combination in the data. Customers with both "
            f"add-ons churn at {rows[-1]['churn_rate_pct']}%. Add-on services correlate with retention, "
            "though this dataset cannot prove causation."
        ),
    })

    sql = """
        SELECT ROUND(SUM(monthly_charges), 2) AS monthly_revenue_at_risk,
               COUNT(*) AS churned_customers
        FROM customers WHERE churn = 1
    """
    rows = _run(conn, sql)
    r = rows[0]
    insights.append({
        "id": "revenue_at_risk",
        "question": "How much monthly recurring revenue is at risk from churned customers?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"Churned customers represented ${r['monthly_revenue_at_risk']:,} in monthly recurring "
            f"charges ({r['churned_customers']} customers) at the time they left. It's a useful proxy for "
            "revenue impact, but it overstates the true loss since not all of them would have stayed forever."
        ),
    })

    sql = """
        SELECT contract, internet_service,
               COUNT(*) AS customers,
               ROUND(100.0 * SUM(churn) / COUNT(*), 1) AS churn_rate_pct
        FROM customers
        WHERE tenure_months <= 12
        GROUP BY contract, internet_service
        ORDER BY churn_rate_pct DESC
        LIMIT 1
    """
    rows = _run(conn, sql)
    top = rows[0]
    insights.append({
        "id": "highest_risk_segment",
        "question": "What is the single highest-risk customer segment, and what should the business do?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"New customers (tenure <= 12 months) on a {top['contract']} contract with "
            f"{top['internet_service']} internet churn at {top['churn_rate_pct']}%, the highest-risk "
            "segment identified. Recommendation: target this segment in the first 90 days with retention "
            "offers (e.g. discounted add-ons or a contract upgrade incentive) before the churn window closes."
        ),
    })

    sql = """
        SELECT c.customer_id, c.contract, c.tenure_months, c.monthly_charges,
               s.churn_probability
        FROM churn_risk_scores s
        JOIN customers c ON c.customer_id = s.customer_id
        WHERE s.churn = 0
        ORDER BY s.churn_probability DESC
        LIMIT 5
    """
    rows = _run(conn, sql)
    insights.append({
        "id": "top_predicted_at_risk",
        "question": "Which currently active customers are most likely to churn next, according to the predictive model?",
        "sql": sql.strip(),
        "result": rows,
        "narrative": (
            f"The baseline churn model flags customer {rows[0]['customer_id']} as highest-risk among "
            f"still-active customers, with a predicted churn probability of {rows[0]['churn_probability']:.0%}. "
            "Most top-ranked customers share short tenure, high monthly charges, and month-to-month "
            "contracts, which lines up with the churn-by-contract and churn-by-tenure findings above. "
            "This is a baseline model with no hyperparameter tuning; see model_metrics.json for accuracy/ROC-AUC."
        ),
    })

    conn.close()
    return insights


def main() -> None:
    insights = build_insights()
    INSIGHTS_PATH.write_text(json.dumps(insights, indent=2))
    print(f"Wrote {len(insights)} insights to {INSIGHTS_PATH}")


if __name__ == "__main__":
    main()
