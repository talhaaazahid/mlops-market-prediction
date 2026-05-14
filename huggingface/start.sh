#!/bin/bash
# Launch script for the HuggingFace Spaces single-container deployment.
# Runs FastAPI on internal port 8000 and Streamlit on public port 7860.

set -e

echo "============================================"
echo "  Market Pulse Predictor — HF Spaces Boot   "
echo "============================================"
echo "  HF_MODEL_REPO  = ${HF_MODEL_REPO:-(unset)}"
echo "  API_BASE_URL   = ${API_BASE_URL:-http://localhost:8000}"
echo "============================================"

# Start FastAPI in the background — it will lazily download model weights
# from HF Hub on first /predict if checkpoints aren't bundled in the image.
uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &
API_PID=$!

# Wait briefly for the API to become reachable so Streamlit doesn't show
# transient connection errors on the first page load.
echo "Waiting for FastAPI to come up …"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "FastAPI ready (took ${i}s)."
        break
    fi
    sleep 1
done

# If FastAPI died during startup, exit so Spaces flags the failure.
if ! kill -0 $API_PID 2>/dev/null; then
    echo "ERROR: FastAPI failed to start — exiting."
    exit 1
fi

# Start Streamlit on the HF-Spaces-required port (7860).
echo "Starting Streamlit on port 7860 …"
exec streamlit run frontend/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableXsrfProtection false \
    --server.enableCORS false \
    --browser.gatherUsageStats false
