"""Turns a natural-language question into a grounded answer, in two LLM calls.

First call routes the question: reuse a precomputed insight, write a
read-only SQL SELECT, or admit the question can't be answered from this
table. The model never sees raw rows at this stage, just the schema and the
insight catalog. Second call writes the final answer from whatever got
retrieved in step one, told to say "can't answer" rather than guess if the
data doesn't actually cover the question.

So every answer traces back to either a checked SQL aggregate in
insights.json or a fresh query that passed the allowlist in sql_tool.py.
Nothing gets answered off the top of the model's head.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.llm.client import chat
from src.llm.sql_tool import SCHEMA_DESCRIPTION, run_query

INSIGHTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "insights.json"


def _load_insights() -> list[dict]:
    return json.loads(INSIGHTS_PATH.read_text())


def _route(question: str, insights: list[dict]) -> dict:
    catalog = "\n".join(f"- {i['id']}: {i['question']}" for i in insights)
    system_prompt = f"""You are a routing engine for an analytics assistant over a single SQLite table.

{SCHEMA_DESCRIPTION}

Precomputed insights already available (id: question):
{catalog}

Decide how to answer the user's question. Respond with ONLY a JSON object, no prose, in one of these three shapes:
1. {{"action": "use_insight", "insight_id": "<id from the list above>"}}
2. {{"action": "sql", "sql": "<single read-only SELECT statement against the customers table>"}}
3. {{"action": "unanswerable", "reason": "<why this cannot be answered from the customers table>"}}

Rules:
- Prefer "use_insight" if an existing insight already answers the question.
- Use "sql" only for SELECT/aggregate queries against the `customers` table. Never write INSERT/UPDATE/DELETE/DDL.
- Use "unanswerable" if the question is about data not present in this table (e.g. dates, regions, support tickets, names) or is not a data question at all.
"""
    raw = chat(system_prompt, question)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"action": "unanswerable", "reason": "Could not interpret the question into a valid query."}


def _compose(question: str, grounding_text: str) -> str:
    system_prompt = (
        "You are an analytics assistant. Answer the user's question using ONLY the data below. "
        "Do not use any outside knowledge or assumptions. If the data below does not actually answer "
        "the question, say so explicitly instead of guessing. Keep the answer to 2-4 sentences, "
        "include the relevant numbers, and avoid restating the raw data verbatim.\n\n"
        f"DATA:\n{grounding_text}"
    )
    return chat(system_prompt, question)


def _suggest_follow_ups(question: str, answer: str, insights: list[dict]) -> list[str]:
    """Asks for 2-3 natural follow-up questions answerable from this same
    table, so the suggestion is grounded in what's actually queryable rather
    than the model inventing something off-topic."""
    catalog = "\n".join(f"- {i['question']}" for i in insights)
    system_prompt = f"""A user just asked an analytics question and got an answer. Suggest 2-3 short, natural follow-up questions they might ask next about the same `customers` dataset (a telco churn table).

Questions already available as a reference for what's queryable:
{catalog}

Respond with ONLY a JSON array of strings, no prose. Example: ["question one?", "question two?"]"""
    user_prompt = f"Original question: {question}\nAnswer given: {answer}"
    raw = chat(system_prompt, user_prompt)
    try:
        suggestions = json.loads(raw)
        if isinstance(suggestions, list):
            return [str(s) for s in suggestions][:3]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def answer_question(question: str) -> dict:
    """Returns {answer, grounded, source, detail, rows, follow_ups}.
    `detail` holds the SQL or insight id used. `rows` is the underlying data
    (for optional charting), empty for unanswerable questions."""
    insights = _load_insights()
    route = _route(question, insights)
    action = route.get("action")

    if action == "use_insight":
        insight = next((i for i in insights if i["id"] == route.get("insight_id")), None)
        if insight is None:
            return _unanswerable("The routed insight id was not found in the catalog.")
        grounding_text = f"{insight['narrative']}\nRaw result: {insight['result']}"
        answer = _compose(question, grounding_text)
        follow_ups = _suggest_follow_ups(question, answer, insights)
        return {
            "answer": answer,
            "grounded": True,
            "source": "precomputed_insight",
            "detail": insight["id"],
            "rows": insight["result"],
            "follow_ups": follow_ups,
        }

    if action == "sql":
        sql = route.get("sql", "")
        try:
            rows = run_query(sql)
        except Exception as exc:  # covers UnsafeQueryError and sqlite3 errors
            return _unanswerable(f"Generated query could not be executed safely: {exc}")
        if not rows:
            return _unanswerable("The query ran but returned no matching data.")
        answer = _compose(question, f"SQL: {sql}\nResult rows: {rows}")
        follow_ups = _suggest_follow_ups(question, answer, insights)
        return {
            "answer": answer,
            "grounded": True,
            "source": "generated_sql",
            "detail": sql,
            "rows": rows,
            "follow_ups": follow_ups,
        }

    return _unanswerable(route.get("reason", "This question cannot be answered from the available dataset."))


def _unanswerable(reason: str) -> dict:
    return {
        "answer": f"I can't answer that from this dataset. {reason}",
        "grounded": False,
        "source": "none",
        "detail": reason,
        "rows": [],
        "follow_ups": [],
    }
