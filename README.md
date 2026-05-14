# RT-Market-Movement-Prediction

> **Real-Time Market Movement Prediction System** — A production-grade MLOps project that fuses financial news sentiment analysis with technical indicators to predict the next-day price direction of six major US equities using deep-learning sequence models (RNN / LSTM / GRU / BiLSTM-Attention).

[![CI Pipeline](https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction/actions/workflows/cd.yml/badge.svg)](https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Quick Start (Local)](#quick-start-local)
6. [Docker Deployment](#docker-deployment)
7. [EC2 Deployment](#ec2-deployment)
8. [API Reference](#api-reference)
9. [Model Performance](#model-performance)
10. [MLflow Experiment Tracking](#mlflow-experiment-tracking)
11. [DVC Pipeline](#dvc-pipeline)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Airflow Orchestration](#airflow-orchestration)
14. [Configuration](#configuration)
15. [Contributing](#contributing)
16. [Team](#team)

---

## Overview

This system ingests real-time financial data from four sources (Yahoo Finance, RSS news feeds, Reddit, NewsData.io), applies an ensemble sentiment analyzer (FinBERT + VADER), engineers technical indicators, and trains four PyTorch sequence models per ticker. The entire ML lifecycle — data versioning, experiment tracking, containerization, and CI/CD — is managed with industry-standard MLOps tooling.

**Predicted targets per ticker:**

| Target | Type | Description |
|---|---|---|
| `direction` | Binary | Up (1) / Down (0) next day |
| `direction_3class` | 3-class | Up / Neutral / Down |
| `volatility_spike` | Binary | Absolute return > 2× rolling std |
| `next_day_return` | Regression | Continuous % return |

**Tracked tickers:** `AAPL · MSFT · GOOGL · AMZN · TSLA · META`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  Yahoo Finance │ RSS Feeds (8) │ Reddit (5 subs) │ NewsData.io      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                              │
│  FinBERT (0.7) + VADER (0.3) Ensemble  │  Technical Indicators     │
│  RSI · MACD · Bollinger · ATR · SMA/EMA│  Daily Sentiment Agg.    │
└────────────────────────┬────────────────────────────────────────────┘
                         │  Sliding window (lookback=30)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE MODELS (PyTorch)                        │
│         RNN  │  LSTM  │  GRU  │  BiLSTM-Attention                  │
│         4 architectures × 6 tickers = 24 MLflow runs               │
└────────────────────────┬────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐  ┌──────────────────────────┐
│   FastAPI Backend   │  │   MLflow Tracking Server  │
│   /predict          │  │   Experiments / Artifacts  │
│   /models           │  └──────────────────────────┘
│   /sentiment/{tkr}  │
│   /retrain          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Streamlit Frontend │
│  4-tab dashboard    │
└─────────────────────┘
```

**MLOps layer:** DVC (data versioning) → MLflow (experiment tracking) → Docker Compose (multi-service) → GitHub Actions CI/CD → Airflow DAG (daily schedule)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.x |
| NLP / Sentiment | HuggingFace Transformers (FinBERT), NLTK VADER |
| Data | yfinance, feedparser, PRAW, NewsData.io |
| Feature Engineering | pandas, numpy, scikit-learn |
| API | FastAPI + Uvicorn + Pydantic v2 |
| Frontend | Streamlit |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Containerization | Docker (multi-stage) + Docker Compose |
| CI/CD | GitHub Actions |
| Orchestration | Apache Airflow 2.x |
| Linting | Ruff |
| Testing | pytest + pytest-cov |

---

## Repository Structure

```
RT-Market-Movement-Prediction/
├── src/
│   ├── config.py                  # Centralized paths, constants, tickers
│   ├── data_ingestion/
│   │   ├── yahoo_finance.py
│   │   ├── news_rss.py
│   │   ├── reddit_scraper.py
│   │   └── newsdata_api.py
│   ├── sentiment/
│   │   ├── finbert_analyzer.py    # HuggingFace FinBERT (singleton)
│   │   ├── vader_analyzer.py      # NLTK VADER (singleton)
│   │   └── ensemble.py            # Weighted 0.7/0.3 ensemble
│   ├── feature_engineering/
│   │   ├── technical_indicators.py
│   │   ├── sentiment_aggregator.py
│   │   └── dataset_builder.py     # Sliding window, 70/15/15 split, scaler
│   ├── models/
│   │   ├── base_model.py
│   │   ├── rnn_model.py
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   ├── bilstm_attention.py
│   │   └── trainer.py             # MLflow-integrated training loop
│   ├── evaluation/
│   │   └── metrics.py             # Classification + regression metrics + plots
│   ├── api/
│   │   ├── main.py                # FastAPI app (lifespan, CORS, global handler)
│   │   ├── routes.py              # Endpoints: predict, models, sentiment, retrain
│   │   └── schemas.py             # Pydantic v2 request/response models
│   └── utils/
│       ├── logger.py
│       └── helpers.py
├── frontend/
│   └── app.py                     # Streamlit 4-tab dashboard
├── scripts/
│   ├── run_ingestion.py
│   ├── run_training.py
│   ├── run_pipeline.py            # Sequential full-pipeline orchestrator
│   ├── deploy_ec2.sh
│   ├── git_setup.sh
│   └── distribute_commits.sh
├── airflow/
│   └── dags/
│       └── market_pulse_dag.py    # Daily 06:00 UTC Airflow DAG (7 tasks)
├── tests/
│   ├── test_ingestion.py
│   ├── test_sentiment.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── data/                          # DVC-tracked (not committed to git)
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/saved/                  # DVC-tracked checkpoints
├── results/                       # Plots + model_comparison.csv
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── mlflow_config.py
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- Git + DVC (`pip install dvc`)
- API keys for Reddit and NewsData.io (see [Configuration](#configuration))

### Setup

```bash
# 1. Clone
git clone https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction.git
cd RT-Market-Movement-Prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Run the full DVC pipeline
dvc repro

# 6. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start the frontend (new terminal)
streamlit run frontend/app.py
```

Visit `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the interactive API.

---

## Docker Deployment

### Build and run all services

```bash
# Copy and fill in secrets
cp .env.example .env

# Build image (downloads FinBERT + VADER at build time)
docker compose build

# Start all services: API + Frontend + MLflow
docker compose up -d

# Check status
docker compose ps
docker compose logs -f app
```

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

### Stop services

```bash
docker compose down
```

---

## EC2 Deployment

### One-time instance setup

```bash
# On a fresh Ubuntu 22.04 EC2 instance (t3.medium or larger recommended)
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git

# Clone repo
git clone https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction.git ~/RT-Market-Movement-Prediction
cd ~/RT-Market-Movement-Prediction
cp .env.example .env
# nano .env  →  fill in API keys

# First deploy
bash scripts/deploy_ec2.sh
```

### Automated re-deploys

Subsequent pushes to `main` trigger the CD pipeline which SSH-deploys automatically. For manual re-deploy:

```bash
cd ~/RT-Market-Movement-Prediction
bash scripts/deploy_ec2.sh
```

### Required GitHub Secrets

Set these in **Settings → Secrets and variables → Actions**:

| Secret | Value |
|---|---|
| `EC2_HOST` | Public IP or hostname of EC2 instance |
| `EC2_USERNAME` | SSH user (`ubuntu` for Ubuntu AMIs) |
| `EC2_SSH_KEY` | Contents of the `.pem` private key file |

---

## API Reference

Base URL: `http://<host>:8000`

### `GET /health`
Returns API status.

```json
{"status": "ok", "timestamp": "2025-06-01T10:00:00Z"}
```

### `POST /predict`

**Request:**
```json
{
  "ticker": "AAPL",
  "model": "lstm",
  "target": "direction"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "model": "lstm",
  "prediction": {
    "direction": "Up",
    "confidence": 0.82,
    "probabilities": {"Down": 0.18, "Up": 0.82}
  },
  "sentiment_summary": {
    "ensemble_compound": 0.34,
    "finbert_label": "positive",
    "article_count": 12
  },
  "timestamp": "2025-06-01T10:00:00Z"
}
```

### `GET /models`
Lists all available trained models and their metrics.

### `GET /sentiment/{ticker}`
Returns latest aggregated sentiment features for a ticker.

### `POST /retrain`
Triggers background retraining of all models (non-blocking).

---

## Model Performance

Training runs 4 models × 6 tickers = **24 independent MLflow experiments**.

Typical results on the held-out test set (15% of data, chronologically last):

| Model | Avg. Directional Accuracy | Avg. F1 (macro) |
|---|---|---|
| BiLSTM-Attention | ~0.68 | ~0.65 |
| LSTM | ~0.65 | ~0.62 |
| GRU | ~0.64 | ~0.61 |
| RNN | ~0.61 | ~0.58 |

> Results vary by ticker and market regime. See `results/model_comparison.csv` and the MLflow UI for full per-run metrics.

---

## MLflow Experiment Tracking

MLflow tracks every training run automatically.

```bash
# View UI (local)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Or via Docker Compose
open http://localhost:5000
```

**Tracked per run:**
- Hyperparameters: hidden_dim, num_layers, dropout, learning_rate, batch_size, lookback
- Metrics (per epoch): train_loss, val_loss, val_accuracy, val_f1
- Artifacts: model checkpoint (`.pt`), classification report (`.txt`), confusion matrix (`.png`), training curves (`.png`)
- Registered model in the MLflow Model Registry

---

## DVC Pipeline

```bash
# Run the full reproducible pipeline
dvc repro

# Run a single stage
dvc repro sentiment

# View pipeline DAG
dvc dag

# Check what has changed
dvc status
```

Pipeline stages:

```
ingest → sentiment → features → train
```

All intermediate data files and model checkpoints are DVC-tracked and excluded from git.

---

## CI/CD Pipeline

### CI (`ci.yml`) — runs on every push

1. **Lint** — `ruff check src/ scripts/ tests/`
2. **Unit tests** — `pytest tests/ --cov=src` (slow/integration tests excluded)
3. **API tests** — `pytest tests/test_api.py`
4. **Contributor check** — warns if fewer than 3 unique commit authors

### CD (`cd.yml`) — runs on push to `main`

1. Build Docker image
2. Run smoke tests inside the built container
3. Push image to GitHub Container Registry (`ghcr.io`)
4. SSH deploy to EC2: `git pull → docker compose pull → docker compose up -d`

---

## Airflow Orchestration

The DAG `market_pulse_daily_pipeline` runs daily at **06:00 UTC**:

```
ingest_prices → ingest_news → ingest_reddit → run_sentiment
             → build_features → evaluate_models → notify
```

To run locally:

```bash
pip install apache-airflow
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow dags list
airflow dags trigger market_pulse_daily_pipeline
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MarketPulse/1.0

# NewsData.io API (https://newsdata.io)
NEWSDATA_API_KEY=your_api_key

# MLflow
MLFLOW_TRACKING_URI=./mlruns
```

Model hyperparameters and pipeline settings live in `params.yaml`. Edit that file and re-run `dvc repro` to retrain with new settings.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: description"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request against `main`

Please ensure `ruff check` passes and tests are green before opening a PR.

---

## Team

**Core Contributors**

| Name | GitHub | Email | Role |
|---|---|---|---|
| Maarij Aqeel | [@Maarij-Aqeel](https://github.com/Maarij-Aqeel) | maarijaqeel3200@gmail.com | Models & MLflow |
| Abdullah Khan Niazi | [@Abdullah-Khan-Niazi](https://github.com/Abdullah-Khan-Niazi) | abdullahniazi078@gmail.com | Data Ingestion & Feature Engineering |
| Raza Sherazi | [@RazaSherazi09](https://github.com/RazaSherazi09) | razaasherazi@gmail.com | API, Frontend & CI/CD |

**Instructors / TAs**

[@asif370](https://github.com/asif370) · [@omerrfarooqq](https://github.com/omerrfarooqq) · [@Aun-Dev146](https://github.com/Aun-Dev146) · [@ahsan608](https://github.com/ahsan608)

---

*Built as a university ANN + MLOps semester project — Spring 2025.*

## Advanced Configuration
- Model architecture customization
- Real-time data streaming setup
- Multi-GPU training configuration


## Advanced Configuration
- Model architecture customization
- Real-time data streaming setup
- Multi-GPU training configuration


## Advanced Configuration
- Model architecture customization
- Real-time data streaming setup
- Multi-GPU training configuration

