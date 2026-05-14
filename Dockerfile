# ── Stage: builder ───────────────────────────────────────────────────────── #
# Install dependencies in a separate layer so they are cached between builds.

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system packages needed by some Python deps (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download FinBERT model weights at build time so the container starts fast.
# Model files are stored in the HuggingFace cache (~/.cache/huggingface/).
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert'); \
print('FinBERT downloaded successfully')"

# Also download VADER lexicon
RUN python -c "\
import nltk; nltk.download('vader_lexicon', quiet=True); \
print('VADER lexicon downloaded')"


# ── Stage: runtime ────────────────────────────────────────────────────────── #

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /root/.cache

# Copy application source
COPY . .

# Create data and model directories
RUN mkdir -p data/raw data/processed data/features models/saved results mlruns

# Environment defaults (overridden by docker-compose / .env at runtime)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose both the API and Streamlit ports
EXPOSE 8000 8501

# Default: run the FastAPI backend
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
