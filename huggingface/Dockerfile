# HuggingFace Spaces deployment (Docker SDK)
# Single container running FastAPI (internal port 8000) + Streamlit (public port 7860).

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download FinBERT and VADER at build time so cold-start is fast
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('FinBERT downloaded')"
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY params.yaml mlflow_config.py ./
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

# Bundle the processed feature CSVs so the API can build inference windows
# and the dashboard "Data Pipeline Status" panel resolves. ~12 MB total.
COPY data/processed/technical_features.csv      ./data/processed/technical_features.csv
COPY data/processed/daily_sentiment_features.csv ./data/processed/daily_sentiment_features.csv
COPY data/processed/sentiment_labeled.csv        ./data/processed/sentiment_labeled.csv

# Bundle results/ (model_comparison.csv + confusion-matrix PNGs) so the
# "Model Comparison" dashboard tab renders metrics and confusion matrices.
COPY results/ ./results/

# Required by HuggingFace Spaces — they expect the app on port 7860
EXPOSE 7860

# Environment defaults (overridable via Space secrets)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_BASE_URL=http://localhost:8000 \
    MLFLOW_TRACKING_URI=./mlruns \
    HF_HOME=/app/.hfcache \
    TRANSFORMERS_CACHE=/app/.hfcache

# Pre-create writable cache dirs (HF Spaces filesystem is read-only outside /app)
RUN mkdir -p /app/.hfcache /app/data/raw /app/data/processed /app/data/features /app/models /app/results

CMD ["./start.sh"]
