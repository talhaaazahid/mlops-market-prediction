---
title: Market Pulse Predictor
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Equity direction forecasting with sentiment + TA
---

# Market Pulse Predictor

A real-time equity-direction forecasting system that fuses financial news sentiment with technical indicators to predict 5-day forward direction (Up / Down) for six large-cap US equities.

## Architecture

- **FastAPI** backend (internal port 8000) — `/predict`, `/sentiment`, `/models`, `/health`, `/retrain`
- **Streamlit** frontend (port 7860, public) — 4-tab interactive dashboard
- **PyTorch** sequence models — RNN, LSTM, GRU, BiLSTM-Attention
- **Sentiment ensemble** — FinBERT (0.7 weight) + VADER (0.3 weight)

## Models loaded from HuggingFace Hub

Model checkpoints are downloaded lazily from
[`Maarij-Aqeel/market-pulse-models`](https://huggingface.co/Maarij-Aqeel/market-pulse-models)
on first prediction request.

## Source

Full source code, training scripts, MLflow tracking, DVC pipeline, and CI/CD configuration:
**https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction**

## Tickers

AAPL · MSFT · GOOGL · AMZN · TSLA · META

## Quick test

Click the **"Live Prediction"** tab, pick a ticker, choose the BiLSTM-Attention model, and click *Predict*.
