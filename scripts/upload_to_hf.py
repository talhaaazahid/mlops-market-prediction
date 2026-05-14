"""Upload trained model checkpoints and feature artefacts to HuggingFace Model Hub.

Creates the repo if it doesn't exist, then uploads:
  - All .pt files in models/saved/
  - data/features/scaler.pkl
  - data/features/feature_names.json
  - data/features/dataset_metadata.json
  - results/model_comparison.csv

Usage:
    # 1. Get a write-token from https://huggingface.co/settings/tokens
    # 2. Export it:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
    # 3. Run:
    python scripts/upload_to_hf.py --repo Maarij-Aqeel/market-pulse-models

Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from src.config import (
    DATASET_METADATA_FILE,
    FEATURE_NAMES_FILE,
    MODEL_COMPARISON_FILE,
    MODELS_DIR,
    SAVED_MODELS_DIR,
    SCALER_FILE,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_REPO = "Maarij-Aqeel/market-pulse-models"


def _ensure_repo(api: HfApi, repo_id: str, token: str) -> None:
    """Create the model repo if it does not already exist."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        log.info("Repo exists: %s", repo_id)
    except RepositoryNotFoundError:
        log.info("Creating new repo: %s", repo_id)
        create_repo(repo_id=repo_id, repo_type="model", token=token, private=False)


def _collect_files() -> list[tuple[Path, str]]:
    """Return (local_path, repo_path) tuples for all artefacts to upload."""
    pairs: list[tuple[Path, str]] = []

    # Model checkpoints — trainer writes directly to MODELS_DIR; older builds
    # may have written to MODELS_DIR/saved, so check both.
    seen: set[str] = set()
    for search_dir in (MODELS_DIR, SAVED_MODELS_DIR):
        if not search_dir.exists():
            continue
        for ckpt in sorted(search_dir.glob("*.pt")):
            if ckpt.name in seen:
                continue
            seen.add(ckpt.name)
            pairs.append((ckpt, f"checkpoints/{ckpt.name}"))

    # Feature engineering artefacts (needed for inference)
    if SCALER_FILE.exists():
        pairs.append((SCALER_FILE, "features/scaler.pkl"))
    if FEATURE_NAMES_FILE.exists():
        pairs.append((FEATURE_NAMES_FILE, "features/feature_names.json"))
    if DATASET_METADATA_FILE.exists():
        pairs.append((DATASET_METADATA_FILE, "features/dataset_metadata.json"))

    # Comparison table for documentation
    if MODEL_COMPARISON_FILE.exists():
        pairs.append((MODEL_COMPARISON_FILE, "results/model_comparison.csv"))

    return pairs


def _build_model_card(repo_id: str, n_checkpoints: int) -> str:
    """Render a Markdown model card for the repo's README."""
    return f"""---
license: mit
tags:
  - finance
  - time-series
  - sentiment-analysis
  - pytorch
  - lstm
  - bilstm-attention
library_name: pytorch
---

# Market Pulse Predictor — Sequence Models

Trained checkpoints for the [RT-Market-Movement-Prediction](https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction) project.

## Contents

- **{n_checkpoints} model checkpoints** (`checkpoints/*.pt`)
  - 4 architectures (RNN, LSTM, GRU, BiLSTM-Attention) × 6 tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, META)
- **Feature artefacts** (`features/`) — fitted StandardScaler, feature-name list, dataset metadata
- **Model comparison** (`results/model_comparison.csv`) — full test metrics

## Inference

```python
from huggingface_hub import hf_hub_download
import torch
import pickle

ckpt = hf_hub_download(repo_id="{repo_id}",
                       filename="checkpoints/bilstm_attention_META.pt")
scaler_path = hf_hub_download(repo_id="{repo_id}",
                              filename="features/scaler.pkl")

state_dict = torch.load(ckpt, map_location="cpu")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
```

See the [GitHub repository](https://github.com/Maarij-Aqeel/RT-Market-Movement-Prediction) for full inference code, training scripts, and the FastAPI/Streamlit serving stack.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload trained models to HF Hub.")
    parser.add_argument("--repo", default=_DEFAULT_REPO,
                        help=f"HuggingFace repo ID (default: {_DEFAULT_REPO})")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"),
                        help="HF write token (or export HF_TOKEN env var)")
    args = parser.parse_args()

    if not args.token:
        log.error("No HF token provided. Pass --token or set HF_TOKEN env var.")
        log.error("Get a token at https://huggingface.co/settings/tokens (write scope).")
        sys.exit(1)

    api = HfApi()
    _ensure_repo(api, args.repo, args.token)

    files = _collect_files()
    if not files:
        log.error("No artefacts found to upload. Run training first.")
        sys.exit(1)

    log.info("Uploading %d files to %s …", len(files), args.repo)
    for local_path, repo_path in files:
        log.info("  %s → %s", local_path.name, repo_path)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.repo,
            token=args.token,
            commit_message=f"upload {repo_path}",
        )

    # Upload the model card last
    n_ckpts = sum(1 for p, _ in files if p.suffix == ".pt")
    card = _build_model_card(args.repo, n_ckpts)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        token=args.token,
        commit_message="docs: model card",
    )

    log.info("✅ Upload complete. View at: https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    main()
