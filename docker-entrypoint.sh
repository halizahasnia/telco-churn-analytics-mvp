#!/bin/sh
set -e

echo "Building analytics-ready dataset..."
python -m src.pipeline.build_db
python -m src.analytics.churn_model
python -m src.analytics.insights

echo "Starting Streamlit..."
exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
