# HuggingFace Deployment Guide

Two-step deployment: **(1) upload model checkpoints to HuggingFace Model Hub**, then **(2) deploy the inference stack as a HuggingFace Space**.

---

## Prerequisites

1. A HuggingFace account → https://huggingface.co/join
2. A **write-token** → https://huggingface.co/settings/tokens (click "Create new token", scope = "Write")
3. Save the token in your shell:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
   ```
4. Install the HuggingFace CLI:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login   # paste the token when prompted
   ```

---

## Step 1 — Upload model checkpoints to HF Model Hub

The 24 trained `.pt` files plus the fitted scaler and feature-name list are uploaded to a public model repo so the Space can download them at runtime.

```bash
# From the project root
python scripts/upload_to_hf.py --repo Maarij-Aqeel/market-pulse-models
```

You should see:
```
Creating new repo: Maarij-Aqeel/market-pulse-models
Uploading 27 files to Maarij-Aqeel/market-pulse-models …
  rnn_AAPL.pt → checkpoints/rnn_AAPL.pt
  ...
✅ Upload complete. View at: https://huggingface.co/Maarij-Aqeel/market-pulse-models
```

Total upload: ~50–80 MB. Takes 2–5 minutes on a normal connection.

---

## Step 2 — Create the HuggingFace Space

### 2a. Create the Space on the HuggingFace website

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `market-pulse-predictor`
   - **License:** MIT
   - **SDK:** **Docker** *(important — pick Docker, not Streamlit)*
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public
3. Click **Create Space**

You'll get an empty repo at `https://huggingface.co/spaces/<your-username>/market-pulse-predictor`.

### 2b. Set the Space secrets

In the Space → **Settings → Variables and secrets**, add:

| Name | Value | Type |
|---|---|---|
| `HF_MODEL_REPO` | `Maarij-Aqeel/market-pulse-models` | Variable |
| `NEWSDATA_API_KEY` | `<your key>` | Secret |
| `TWELVE_DATA_API_KEY` | `<your key>` | Secret |

The Reddit credentials are optional — the API will still work without them, just with empty social-media sentiment.

### 2c. Push the deployment files to the Space

Clone the empty Space repo somewhere temporary:

```bash

```

Copy the deployment files from your project. From the project root:

```bash
PROJECT=/Users/macbookair/Projects/Python/Projects/MLOPS/Market-Movement_predictoin
SPACE=/tmp/hf-space

# Required files
cp $PROJECT/huggingface/Dockerfile  $SPACE/Dockerfile
cp $PROJECT/huggingface/start.sh    $SPACE/start.sh
cp $PROJECT/huggingface/README.md   $SPACE/README.md
cp $PROJECT/requirements.txt        $SPACE/requirements.txt
cp $PROJECT/params.yaml             $SPACE/params.yaml
cp $PROJECT/mlflow_config.py        $SPACE/mlflow_config.py

# Source code
mkdir -p $SPACE/src $SPACE/frontend $SPACE/scripts
cp -r $PROJECT/src/        $SPACE/src/
cp -r $PROJECT/frontend/   $SPACE/frontend/
cp -r $PROJECT/scripts/    $SPACE/scripts/

# Push
cd $SPACE
git add .
git commit -m "feat: initial deployment to HuggingFace Spaces"
git push
```

The Space will start building automatically. You can watch the build at:
`https://huggingface.co/spaces/Maarij-Aqeel/market-pulse-predictor?logs=container`

First build takes **10–15 minutes** (downloads FinBERT during the Docker build).

### 2d. Verify

Once the Space shows **"Running"** (green dot), open it in your browser:

`https://huggingface.co/spaces/Maarij-Aqeel/market-pulse-predictor`

You should see the Streamlit dashboard. Click the **Live Prediction** tab → select `META` + `bilstm_attention` → **Predict**. The first request will be slow (~5–10s) because the model is downloaded from HF Hub on demand; subsequent requests are instant.

---

## Updating the Space later

After making changes locally:

```bash
# Re-copy the changed files to /tmp/hf-space, then:
cd /tmp/hf-space
git add .
git commit -m "<change description>"
git push
```

The Space auto-rebuilds and redeploys on every push.

---

## Troubleshooting

**Build fails with "out of memory"** — the FinBERT download in the Dockerfile uses ~2GB of build-time RAM. Free Spaces have 16GB during build, so this should be fine; if it fails, comment out the FinBERT pre-download line and let the model load lazily on first request.

**Streamlit can't reach FastAPI** — check that both processes started in the Space logs. The `start.sh` script waits 30s for FastAPI to be ready before launching Streamlit.

**Model downloads fail at inference time** — verify `HF_MODEL_REPO` is set in Space secrets and that the model repo is public (or the Space has a HF token with read access).

**Cold start takes 30–60 seconds** — normal for free CPU spaces after they go to sleep. The Space wakes up on the first HTTP request and stays alive for ~48 hours of inactivity.

---

## Three-pronged deployment summary

| Surface | Purpose | Audience |
|---|---|---|
| **HF Space (this guide)** | Public live demo | Professor — clickable URL, no setup |
| **GHCR Docker image** (auto-built by CI) | Production-ready container | Reviewers — `docker pull ghcr.io/maarij-aqeel/market-pulse-predictor` |
| **Local `docker compose up`** | Architecture deep-dive | Yourself during the demo — shows MLflow + FastAPI + Streamlit + healthchecks |

## Production Deployment
- Zero-downtime deployment strategy
- Health check endpoints
- Automatic rollback on failure

