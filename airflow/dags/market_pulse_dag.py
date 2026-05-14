"""Airflow DAG: market_pulse_daily_pipeline.

Runs the full Market Pulse ingestion → training → evaluation pipeline
daily at 06:00 UTC (after US market close).  Even when Airflow is not
deployed this file demonstrates the intended orchestration design.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Default task arguments ─────────────────────────────────────────────────── #

_DEFAULT_ARGS: dict = {
    "owner": "market-pulse-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ── Helper ─────────────────────────────────────────────────────────────────── #


def _run(module: str, *extra_args: str) -> None:
    """Run a Python module as a subprocess so it inherits the container env."""
    cmd = [sys.executable, "-m", module, *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Module {module!r} exited with code {result.returncode}"
        )


# ── Task callables ─────────────────────────────────────────────────────────── #


def ingest_prices() -> None:
    """Download OHLCV price data from Yahoo Finance."""
    from src.data_ingestion.yahoo_finance import run as _run_prices

    _run_prices()


def ingest_news() -> None:
    """Fetch headlines from RSS feeds and NewsData.io."""
    from src.data_ingestion.news_rss import fetch_all_feeds
    from src.data_ingestion.newsdata_api import fetch_latest_newsdata
    from src.utils.helpers import append_csv
    import pandas as pd
    from src.config import RAW_DIR

    articles = fetch_all_feeds()
    if articles:
        df = pd.DataFrame(articles)
        append_csv(df, RAW_DIR / "news_articles.csv")

    newsdata = fetch_latest_newsdata()
    if newsdata:
        nd_df = pd.DataFrame(newsdata)
        append_csv(nd_df, RAW_DIR / "newsdata_articles.csv")


def ingest_reddit() -> None:
    """Scrape Reddit posts from finance subreddits."""
    from src.data_ingestion.reddit_scraper import fetch_latest_reddit
    from src.utils.helpers import append_csv
    import pandas as pd
    from src.config import RAW_DIR

    posts = fetch_latest_reddit()
    if posts:
        df = pd.DataFrame(posts)
        append_csv(df, RAW_DIR / "reddit_posts.csv")


def run_sentiment() -> None:
    """Label all text data with ensemble sentiment scores."""
    from src.config import RAW_DIR, PROCESSED_DIR
    from src.sentiment.ensemble import run_sentiment_pipeline

    run_sentiment_pipeline(
        input_path=RAW_DIR / "all_text_data.csv",
        output_path=PROCESSED_DIR / "sentiment_labeled.csv",
    )


def build_features() -> None:
    """Compute technical indicators, aggregate sentiment, and build sequences."""
    _run("src.feature_engineering.technical_indicators")
    _run("src.feature_engineering.sentiment_aggregator")
    _run("src.feature_engineering.dataset_builder")


def evaluate_models() -> None:
    """Retrain all sequence models and log results to MLflow."""
    _run("scripts.run_training")


def notify(**context: object) -> None:  # noqa: ANN003
    """Log a pipeline completion summary to stdout (extend with email/Slack)."""
    execution_date = context.get("execution_date", "unknown")
    dag_run = context.get("dag_run")
    state = getattr(dag_run, "state", "unknown") if dag_run else "unknown"
    print(
        f"[market_pulse_dag] Pipeline finished | "
        f"execution_date={execution_date} | state={state}"
    )


# ── DAG definition ─────────────────────────────────────────────────────────── #

with DAG(
    dag_id="market_pulse_daily_pipeline",
    description="Daily Market Pulse ingestion → sentiment → features → training pipeline",
    schedule_interval="0 6 * * *",  # 06:00 UTC daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args=_DEFAULT_ARGS,
    tags=["market-pulse", "mlops", "finance"],
) as dag:

    t_ingest_prices = PythonOperator(
        task_id="ingest_prices",
        python_callable=ingest_prices,
    )

    t_ingest_news = PythonOperator(
        task_id="ingest_news",
        python_callable=ingest_news,
    )

    t_ingest_reddit = PythonOperator(
        task_id="ingest_reddit",
        python_callable=ingest_reddit,
    )

    t_run_sentiment = PythonOperator(
        task_id="run_sentiment",
        python_callable=run_sentiment,
    )

    t_build_features = PythonOperator(
        task_id="build_features",
        python_callable=build_features,
    )

    t_evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
    )

    t_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
        provide_context=True,
    )

    # ── Task dependencies ──────────────────────────────────────────────────── #
    (
        t_ingest_prices
        >> t_ingest_news
        >> t_ingest_reddit
        >> t_run_sentiment
        >> t_build_features
        >> t_evaluate_models
        >> t_notify
    )
