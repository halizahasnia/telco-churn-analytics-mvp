"""MVP interface: one dashboard page + one natural-language prompt box.

Run with: streamlit run app/streamlit_app.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm.grounding import answer_question  # noqa: E402

DB_PATH = ROOT / "data" / "processed" / "telco_churn.db"
INSIGHTS_PATH = ROOT / "data" / "processed" / "insights.json"
MODEL_METRICS_PATH = ROOT / "data" / "processed" / "model_metrics.json"

st.set_page_config(page_title="Telco Churn Analytics MVP", layout="wide")


@st.cache_data
def load_insights() -> list[dict]:
    return json.loads(INSIGHTS_PATH.read_text())


@st.cache_data
def load_customers() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df


st.title("Telco Customer Churn Analytics MVP")
st.caption(
    "Dataset: IBM/Kaggle Telco Customer Churn (7,043 customers). "
    "Analytics-ready data is precomputed; the prompt box below is grounded in that same data."
)

tab_dashboard, tab_ask = st.tabs(["📊 Dashboard", "💬 Ask a Question"])

with tab_dashboard:
    insights = load_insights()
    df = load_customers()

    col1, col2, col3 = st.columns(3)
    overall = next(i for i in insights if i["id"] == "overall_churn_rate")["result"][0]
    col1.metric("Total customers", overall["total_customers"])
    col2.metric("Churned customers", overall["churned_customers"])
    col3.metric("Churn rate", f"{overall['churn_rate_pct']}%")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Churn rate by contract type")
        contract = pd.DataFrame(next(i for i in insights if i["id"] == "churn_by_contract")["result"])
        st.bar_chart(contract.set_index("contract")["churn_rate_pct"])
        st.caption(next(i for i in insights if i["id"] == "churn_by_contract")["narrative"])

    with right:
        st.subheader("Churn rate by tenure bucket")
        tenure = pd.DataFrame(next(i for i in insights if i["id"] == "churn_by_tenure")["result"])
        st.bar_chart(tenure.set_index("tenure_bucket")["churn_rate_pct"])
        st.caption(next(i for i in insights if i["id"] == "churn_by_tenure")["narrative"])

    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Churn rate by payment method")
        pay = pd.DataFrame(next(i for i in insights if i["id"] == "churn_by_payment_method")["result"])
        st.bar_chart(pay.set_index("payment_method")["churn_rate_pct"])
        st.caption(next(i for i in insights if i["id"] == "churn_by_payment_method")["narrative"])

    with right2:
        st.subheader("Revenue at risk & highest-risk segment")
        rev = next(i for i in insights if i["id"] == "revenue_at_risk")["result"][0]
        st.metric("Monthly revenue at risk", f"${rev['monthly_revenue_at_risk']:,}")
        st.caption(next(i for i in insights if i["id"] == "highest_risk_segment")["narrative"])

    st.divider()
    st.subheader("Predicted churn risk (baseline ML model)")
    if MODEL_METRICS_PATH.exists():
        metrics = json.loads(MODEL_METRICS_PATH.read_text())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", metrics["accuracy"])
        m2.metric("Precision", metrics["precision"])
        m3.metric("Recall", metrics["recall"])
        m4.metric("ROC-AUC", metrics["roc_auc"])
        st.caption(
            f"{metrics['model']}. {metrics['note']} "
            f"Trained on {metrics['train_set_size']} rows, evaluated on {metrics['test_set_size']} held-out rows."
        )

        at_risk_insight = next(i for i in insights if i["id"] == "top_predicted_at_risk")
        st.markdown("**Top currently-active customers by predicted churn risk:**")
        st.dataframe(pd.DataFrame(at_risk_insight["result"]))
        st.caption(at_risk_insight["narrative"])

        st.markdown("**Cost-sensitive decision threshold (business value, not just accuracy):**")
        bva = metrics["business_value_analysis"]
        a = bva["assumptions"]
        st.caption(
            f"Assumptions (illustrative, adjust in `churn_model.py`): retention outreach costs "
            f"${a['retention_offer_cost_usd']:.0f}/customer, {a['retention_success_rate']:.0%} of contacted "
            f"at-risk customers are actually retained, and a retained customer preserves "
            f"{a['retained_value_months']} months of billing value."
        )
        curve_df = pd.DataFrame(bva["threshold_curve"]).set_index("threshold")
        st.line_chart(curve_df["expected_net_value"])
        st.write(
            f"Recommended threshold: **{bva['recommended_threshold']}** "
            f"(default would be 0.5). {bva['recommended_threshold_rationale']}"
        )
    else:
        st.info("Model metrics not found. Run `python -m src.analytics.churn_model` to generate them.")

    with st.expander("View underlying curated data (first 100 rows)"):
        st.dataframe(df.head(100))

with tab_ask:
    st.subheader("Ask a question about this dataset")
    st.caption(
        "Answers are grounded in the curated `customers` table or in the precomputed insights above. "
        "The LLM is not allowed to answer from general knowledge."
    )

    suggested = [i["question"] for i in load_insights()]
    suggested.append("What is the average monthly charge for customers on a two-year contract?")
    cols = st.columns(2)
    for idx, q in enumerate(suggested[:6]):
        if cols[idx % 2].button(q, key=f"sugg_{idx}"):
            st.session_state["question_input"] = q

    question = st.text_input(
        "Your question",
        key="question_input",
        placeholder="e.g. Which customer segment has the highest churn risk?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving data and composing answer..."):
            try:
                result = answer_question(question)
            except Exception as exc:
                st.error(
                    "The LLM call failed. Check that an API key is configured in `.env` "
                    f"for the provider set in LLM_PROVIDER. Details: {exc}"
                )
            else:
                if result["grounded"]:
                    st.success(result["answer"])
                else:
                    st.warning(result["answer"])

                rows = result.get("rows") or []
                if len(rows) > 1:
                    chart_df = pd.DataFrame(rows)
                    numeric_cols = chart_df.select_dtypes(include="number").columns.tolist()
                    label_cols = [c for c in chart_df.columns if c not in numeric_cols]
                    if numeric_cols and label_cols:
                        st.bar_chart(chart_df.set_index(label_cols[0])[numeric_cols[0]])

                with st.expander("How was this answer generated?"):
                    st.write(f"**Source:** {result['source']}")
                    st.code(result["detail"], language="sql" if result["source"] == "generated_sql" else None)

                follow_ups = result.get("follow_ups") or []
                if follow_ups:
                    st.caption("Follow-up questions:")
                    fcols = st.columns(len(follow_ups))
                    for idx, fq in enumerate(follow_ups):
                        if fcols[idx].button(fq, key=f"followup_{idx}"):
                            st.session_state["question_input"] = fq
                            st.rerun()
