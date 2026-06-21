# Telco Customer Churn: LLM-Powered Analytics MVP

This is a small MVP built around the IBM/Kaggle Telco Customer Churn dataset.
It cleans the raw data into an analytics-ready table, computes a set of
business insights plus a baseline churn prediction model, and exposes both
through a Streamlit dashboard with a chat box on top. The chat box only
answers from data that's actually in the database; it doesn't get to make
things up.

Full reasoning behind the design choices is in `docs/architecture.md`.

## Dataset

Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), the IBM sample dataset mirrored on Kaggle. 7,043 customers, 21 raw columns covering demographics, subscribed services, contract/billing info, and a Churn (Yes/No) label.

I picked this one because it fits the "Customer Churn / Subscription" category well: a telco running month-to-month vs. term contracts looks a lot like most subscription businesses. It also has a small, realistic data quality wrinkle (blank `TotalCharges` strings) without being messy enough to eat the whole 5 days.

Business problem: figure out which customers are likely to leave and what's driving it, so retention spend goes to the right people.

A few things worth flagging up front: this is a single snapshot in time, there's no historical trend across periods, `customerID` is already an anonymized key rather than real PII, and the churn label reflects whoever left in the month the data was pulled, not necessarily who's at risk today.

## Project layout

```
data/raw/              raw CSV as downloaded from Kaggle
data/processed/        analytics-ready SQLite DB + precomputed insights (generated, not committed)
src/pipeline/          ingestion, cleaning, data quality checks, DB build
src/analytics/         precomputed business-question insights (SQL + narrative) and the churn risk model
src/llm/               LLM client, guardrailed SQL tool, grounding orchestration
app/streamlit_app.py   dashboard tab + ask-a-question tab
docs/architecture.md   architecture document
Dockerfile, docker-compose.yml, docker-entrypoint.sh   container setup
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, or pick a different provider below
```

### Getting the dataset

The raw CSV isn't committed to this repo (Kaggle's redistribution terms are unclear, and it's easy enough to grab yourself):

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Place it at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` in this repo.

If you skip this, `python -m src.pipeline.build_db` will fail with a clear error pointing back to these two steps.

### Picking an LLM provider

`src/llm/client.py` doesn't hardcode a vendor. Set `LLM_PROVIDER` in `.env` to whichever of these fits:

| Provider | Cost | Setup |
|---|---|---|
| `openai` (default) | pay per token | set `OPENAI_API_KEY` |
| `anthropic` | pay per token | set `ANTHROPIC_API_KEY` |
| `gemini` | free tier available | set `GOOGLE_API_KEY` |
| `ollama` | free, fully local, no key | install [Ollama](https://ollama.com), run `ollama serve`, `ollama pull llama3.2`, set `LLM_PROVIDER=ollama` |

The Ollama path exists so the whole thing can be built and demoed without spending anything or signing up for an API. It just points at Ollama's local OpenAI-compatible endpoint (`http://localhost:11434/v1`). A small local model is noticeably weaker at returning clean JSON for the routing step than a hosted model, and the grounding layer already treats a bad parse as "unanswerable" rather than crashing. Switching to a paid provider later only means editing `.env`.

If you want to go this route:

1. Install Ollama from [ollama.com](https://ollama.com).
2. Start the server: `ollama serve` (leave this running in its own terminal, or as a background service, while you use the app).
3. Pull a model once: `ollama pull llama3.2`.
4. In `.env`, set `LLM_PROVIDER=ollama`. The default base URL and model already match what was just pulled, so `OLLAMA_BASE_URL`/`OLLAMA_MODEL` don't need to be set unless you want a different model.

## Running it

1. Build the analytics-ready dataset (raw CSV to cleaned to curated SQLite):
   ```bash
   python -m src.pipeline.build_db
   ```
2. Train the baseline churn risk model. It writes a `churn_risk_scores` table that both the dashboard and the LLM layer read from:
   ```bash
   python -m src.analytics.churn_model
   ```
3. Precompute the business-question insights (run this after the model step, since one insight joins against `churn_risk_scores`):
   ```bash
   python -m src.analytics.insights
   ```
4. Launch the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open the URL it prints. The dashboard tab works with no API key at all. The "Ask a Question" tab needs either a real LLM key in `.env`, or `LLM_PROVIDER=ollama` with Ollama running locally.

## The churn risk model

The mandatory part of this project is descriptive: SQL aggregates answering things like "which segment churns most." On top of that, `src/analytics/churn_model.py` adds a forward-looking piece, a baseline logistic regression that scores every customer's probability of churning, so you can also ask "who's likely to leave next" instead of only "who already left and why."

It's deliberately plain: one-hot encoded categoricals, scaled numeric features, no hyperparameter search. See `docs/architecture.md` for why that's the right call here, and for a note on a multicollinearity issue I found and fixed (`total_charges` was dropped from the features; it tracked `tenure_months` too closely and was producing a misleading coefficient). On a held-out 20% test split it gets accuracy 0.74, precision 0.51, recall 0.78, ROC-AUC 0.84 (full numbers in `data/processed/model_metrics.json`, along with the top coefficients pushing risk up or down).

Predictions land in their own `churn_risk_scores` table joined to `customers`, rather than being folded into the curated table, so that table's lineage stays purely ETL.

One thing worth calling out: the `predicted_churn` flag doesn't use the standard 0.5 cutoff. Instead the threshold is picked by maximizing expected net retention value on the test set, using a few disclosed assumptions about outreach cost, retention success rate, and how much billing value a retained customer preserves (the `RETENTION_*` constants in `churn_model.py`). That points to threshold 0.1 (precision ~0.34, recall ~0.99) rather than 0.5 (precision 0.51, recall 0.78), because outreach is cheap relative to what's lost when a real churner slips through. The full curve and reasoning is in `docs/architecture.md`.

Both the dashboard and the LLM grounding layer can use this model's output: there's a dedicated insight (`top_predicted_at_risk`), and `churn_risk_scores` is a valid table for generated SQL too.

## How the chatbot stays grounded

The prompt interface is never allowed to just answer from general knowledge:

1. A routing call decides between three options: reuse an existing precomputed insight, write a single read-only `SELECT` against `customers` or `churn_risk_scores`, or admit the question can't be answered from this data.
2. Any generated SQL has to pass an allowlist check first (SELECT only, single statement, known tables only, no DDL/DML) before it runs against a read-only SQLite connection.
3. A second call writes the actual answer, using only what got retrieved in step one. It's told to say it can't answer rather than guess if that data doesn't cover the question.

On top of that, a grounded answer also gets a quick bar chart when the underlying rows have more than one row and a numeric column, and a third call suggests 2-3 follow-up questions to ask next. The follow-up suggestions are best-effort, the model occasionally proposes something this dataset can't actually answer (no date or region columns, for instance), but clicking one still goes through the same routing/validation path above, so a bad suggestion gets a "can't answer" response rather than a made-up one.

More detail, including the security/privacy side of this, is in `docs/architecture.md`.

## Running it in Docker

```bash
docker compose up --build
```

This builds the image, runs the pipeline (build, model, insights) on container start, and serves the app at `http://localhost:8501`. The compose file mounts `data/raw` and `data/processed` as volumes, so you still need the CSV in `data/raw/` first (see above), and `.env` for whichever LLM provider you're using.

If you're using `LLM_PROVIDER=ollama`, point `OLLAMA_BASE_URL` at `http://host.docker.internal:11434/v1` in `.env` instead of `localhost`, since the container needs the host's hostname to reach Ollama running outside it. The compose file already adds the `host.docker.internal` mapping needed for this on Linux too (it works out of the box on Mac and Windows).

## What's not finished

- The churn model is a single-split baseline, no tuning or cross-validation. Treat `churn_probability` as a relative ranking, not a calibrated number you'd hang a budget decision on without more work.
- One static CSV snapshot, no streaming or incremental ingestion.
- The SQL routing call doesn't retry on a bad query; a malformed one is just reported as unanswerable.
- No auth on the Streamlit app, so it's not meant to run anywhere multi-tenant.
- No automated evaluation of LLM answer quality, just manual spot-checking.
- `data/processed/` is generated locally and isn't committed; run the three pipeline commands above before launching the app.
