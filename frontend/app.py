"""Streamlit dashboard for the Market Pulse Predictor.

Calls the FastAPI backend (default: http://localhost:8000) via HTTP.

Start with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────────────────── #
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"]
MODELS = ["rnn", "lstm", "gru", "bilstm_attention"]
TIMEOUT = 10  # seconds

st.set_page_config(
    page_title="Market Pulse Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── API helpers ───────────────────────────────────────────────────────────── #

def _get(path: str, **kwargs) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}. Is the backend running?")
    except requests.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    return None


def _post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE}.")
    except requests.HTTPError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.title("Market Pulse Predictor")
    st.markdown("*Real-time sentiment-driven market forecasting*")
    st.divider()

    selected_ticker = st.selectbox("Ticker", TICKERS, index=0)
    selected_model = st.selectbox("Model", MODELS, index=3)
    date_end = st.date_input("Date range — end", value=datetime.today())
    date_start = st.date_input("Date range — start", value=datetime.today() - timedelta(days=90))

    st.divider()
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()

    # API health indicator
    health = _get("/health")
    if health:
        st.success(f"API Online  |  {len(health.get('models_loaded', []))} checkpoints loaded")
    else:
        st.error("API Offline")


# ── Tabs ──────────────────────────────────────────────────────────────────── #

tab_pred, tab_hist, tab_compare, tab_status = st.tabs(
    ["Live Prediction", "Historical Analysis", "Model Comparison", "Pipeline Status"]
)


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Live Prediction
# ════════════════════════════════════════════════════════════════════════════ #

with tab_pred:
    st.subheader(f"Live Prediction — {selected_ticker}")

    col_pred, col_sent = st.columns([1, 1])

    with col_pred:
        if st.button("Run Prediction", type="primary", use_container_width=True):
            with st.spinner("Running inference …"):
                result = _post("/predict", {"ticker": selected_ticker, "model": selected_model})

            if result:
                pred = result["prediction"]
                direction = pred["direction"].upper()
                confidence = pred["confidence"]

                colour = {"UP": "green", "DOWN": "red", "NEUTRAL": "gray"}.get(direction, "gray")
                arrow = {"UP": "▲", "DOWN": "▼", "NEUTRAL": "■"}.get(direction, "")

                st.markdown(
                    f"""
                    <div style='text-align:center; padding:24px; border-radius:12px;
                                background:#1e1e2e; border: 2px solid {colour};'>
                        <h1 style='color:{colour}; font-size:3rem; margin:0'>
                            {arrow} {direction}
                        </h1>
                        <p style='color:#aaa; font-size:1.2rem; margin:4px 0 0 0'>
                            Confidence: <b>{confidence:.1%}</b>
                        </p>
                        <p style='color:#666; font-size:0.85rem'>
                            Model: {result['model_used'].upper()} &nbsp;|&nbsp;
                            {result['timestamp'][:19].replace('T', ' ')} UTC
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if pred.get("predicted_return") is not None:
                    st.metric("Predicted return", f"{pred['predicted_return']:.2%}")
                if pred.get("volatility_spike_risk") is not None:
                    st.metric("Volatility spike risk", f"{pred['volatility_spike_risk']:.1%}")
        else:
            st.info("Press **Run Prediction** to fetch a live forecast.")

    with col_sent:
        st.markdown("**Latest Sentiment**")
        sent_data = _get(f"/sentiment/{selected_ticker}")
        if sent_data:
            dominant = sent_data["dominant_sentiment"]
            mean_score = sent_data["mean_compound"]
            n_articles = sent_data["article_count"]

            sent_colour = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
            bar_colour = sent_colour.get(dominant, "#95a5a6")

            st.markdown(
                f"""
                <div style='padding:16px; border-radius:10px; background:#1e1e2e;
                            border-left:4px solid {bar_colour}'>
                    <b style='color:{bar_colour}'>{dominant.capitalize()}</b>
                    &nbsp; compound score: <b>{mean_score:+.3f}</b><br>
                    <small style='color:#888'>{n_articles} articles &nbsp;|&nbsp;
                    last updated: {sent_data.get('last_updated', 'N/A')}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider()

            # Ratio bar chart
            ratios = {
                "Positive": sent_data["positive_ratio"],
                "Negative": sent_data["negative_ratio"],
                "Neutral": sent_data["neutral_ratio"],
            }
            ratio_df = pd.DataFrame.from_dict(ratios, orient="index", columns=["Ratio"])
            st.bar_chart(ratio_df, color=["#2ecc71"])
        else:
            st.info("No sentiment data available. Run the sentiment pipeline first.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Historical Analysis
# ════════════════════════════════════════════════════════════════════════════ #

with tab_hist:
    st.subheader(f"Historical Analysis — {selected_ticker}")

    try:
        import yfinance as yf

        raw = yf.download(
            selected_ticker,
            start=date_start,
            end=date_end,
            auto_adjust=True,
            progress=False,
        )

        if not raw.empty:
            # Flatten MultiIndex if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]

            col_price, col_vol = st.columns([3, 1])
            with col_price:
                st.markdown("**Close Price**")
                st.line_chart(raw["close"])
            with col_vol:
                st.markdown("**Volume**")
                st.bar_chart(raw["volume"])

            # Sentiment trend
            sent_csv = pd.read_csv(
                "data/processed/daily_sentiment_features.csv",
                parse_dates=["date"],
            ) if (pd.io.common.file_exists("data/processed/daily_sentiment_features.csv")) else None  # type: ignore[attr-defined]

            if sent_csv is not None:
                sub = sent_csv[
                    (sent_csv["ticker"] == selected_ticker) &
                    (sent_csv["date"] >= pd.Timestamp(date_start)) &
                    (sent_csv["date"] <= pd.Timestamp(date_end))
                ].set_index("date")
                if not sub.empty and "daily_sentiment_mean" in sub.columns:
                    st.markdown("**Daily Sentiment (ensemble compound)**")
                    st.area_chart(sub["daily_sentiment_mean"])
        else:
            st.warning(f"No price data returned for {selected_ticker} in the selected range.")

    except ImportError:
        st.error("yfinance not installed. Run: pip install yfinance")
    except Exception as exc:
        st.warning(f"Could not load historical data: {exc}")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Model Comparison
# ════════════════════════════════════════════════════════════════════════════ #

with tab_compare:
    st.subheader("Model Comparison")

    models_data = _get("/models")
    if models_data and models_data.get("models"):
        df = pd.DataFrame(models_data["models"])

        # Filter to ticker or show all
        show_all = st.checkbox("Show all tickers", value=False)
        if not show_all:
            df = df[df["ticker"] == selected_ticker]

        # Metrics table
        display_cols = ["model_name", "ticker", "checkpoint_exists",
                        "test_accuracy", "test_f1_macro", "test_rmse"]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(
            df[display_cols].style.background_gradient(
                subset=[c for c in ["test_accuracy", "test_f1_macro"] if c in display_cols],
                cmap="Greens",
            ),
            use_container_width=True,
        )

        # Bar chart comparison
        metric_opts = [c for c in ["test_accuracy", "test_f1_macro", "test_rmse"] if c in df.columns]
        if metric_opts:
            selected_metric = st.selectbox("Plot metric", metric_opts)
            pivot = df.pivot_table(
                index="ticker", columns="model_name", values=selected_metric, aggfunc="first"
            )
            if not pivot.empty:
                st.bar_chart(pivot)

        # Confusion matrix images
        st.markdown("**Confusion Matrices**")
        import pathlib
        cm_files = sorted(pathlib.Path("results").glob(f"cm_*_{selected_ticker}.png"))
        if cm_files:
            cols_cm = st.columns(min(len(cm_files), 4))
            for i, f in enumerate(cm_files[:4]):
                cols_cm[i % 4].image(str(f), caption=f.stem, use_container_width=True)
        else:
            st.info("No confusion matrix images found in results/. Train models first.")
    else:
        st.info("No model data available. Train models first with scripts/run_training.py.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 4 — Pipeline Status
# ════════════════════════════════════════════════════════════════════════════ #

with tab_status:
    st.subheader("Data Pipeline Status")

    import pathlib

    def _file_info(path: str) -> dict:
        p = pathlib.Path(path)
        if not p.exists():
            return {"exists": False, "rows": "—", "last_modified": "—", "size_kb": "—"}
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = round(p.stat().st_size / 1024, 1)
            try:
                rows = len(pd.read_csv(p))
            except Exception:
                rows = "?"
            return {"exists": True, "rows": rows, "last_modified": mtime, "size_kb": size_kb}
        except Exception:
            return {"exists": True, "rows": "?", "last_modified": "?", "size_kb": "?"}

    files = {
        "Price data": "data/raw/price_data.csv",
        "News articles": "data/raw/news_articles.csv",
        "Reddit posts": "data/raw/reddit_posts.csv",
        "NewsData.io": "data/raw/newsdata_articles.csv",
        "All text (unified)": "data/raw/all_text_data.csv",
        "Sentiment labeled": "data/processed/sentiment_labeled.csv",
        "Technical features": "data/processed/technical_features.csv",
        "Daily sentiment": "data/processed/daily_sentiment_features.csv",
        "X_train": "data/features/X_train.npy",
        "X_test": "data/features/X_test.npy",
    }

    status_rows = []
    for label, path in files.items():
        info = _file_info(path)
        status_rows.append({
            "File": label,
            "Status": "✅" if info["exists"] else "❌",
            "Rows / Shape": info["rows"],
            "Last modified": info["last_modified"],
            "Size (KB)": info["size_kb"],
        })

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Source breakdown from all_text_data.csv
    all_text_path = pathlib.Path("data/raw/all_text_data.csv")
    if all_text_path.exists():
        try:
            text_df = pd.read_csv(all_text_path)
            source_counts = text_df["source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Article Count"]
            st.markdown("**Articles by source**")
            st.bar_chart(source_counts.set_index("Source"))
        except Exception:
            pass

    st.divider()

    col_retrain, _ = st.columns([1, 2])
    with col_retrain:
        st.markdown("**Trigger Retraining**")
        if st.button("Retrain all models", type="secondary", use_container_width=True):
            resp = _post("/retrain", {"tickers": None, "models": None, "target": "direction_3class"})
            if resp:
                st.success(f"Retrain job queued — ID: {resp.get('job_id', 'N/A')}")
