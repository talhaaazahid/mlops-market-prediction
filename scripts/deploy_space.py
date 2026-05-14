"""Create the HuggingFace Space and upload all deployment files in one step.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
    python scripts/deploy_space.py

The script:
  1. Creates (or reuses) the Space `<USERNAME>/<SPACE_NAME>` with Docker SDK.
  2. Stages all deployment files in a flattened layout in a temp directory.
  3. Uploads the staged folder to the Space in a single commit.

After it finishes, you only need to set the Space secrets in the browser UI
(HF_MODEL_REPO, NEWSDATA_API_KEY, TWELVE_DATA_API_KEY).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login

USERNAME = "Maarij-Aqeel"
SPACE_NAME = "market-pulse-predictor"
PROJECT = Path(__file__).resolve().parent.parent


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: Set HF_TOKEN env var first (export HF_TOKEN=hf_...)")

    login(token=token)
    api = HfApi()
    repo_id = f"{USERNAME}/{SPACE_NAME}"

    print(f"[1/3] Creating Space {repo_id} (Docker SDK) ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        print(f"[2/3] Staging deployment files in {tmp} ...")

        # Root-level files (flattened — Space expects Dockerfile + start.sh at root)
        shutil.copy(PROJECT / "huggingface" / "Dockerfile", tmp / "Dockerfile")
        shutil.copy(PROJECT / "huggingface" / "start.sh",   tmp / "start.sh")
        shutil.copy(PROJECT / "huggingface" / "README.md",  tmp / "README.md")
        shutil.copy(PROJECT / "requirements.txt",           tmp / "requirements.txt")
        shutil.copy(PROJECT / "params.yaml",                tmp / "params.yaml")
        shutil.copy(PROJECT / "mlflow_config.py",           tmp / "mlflow_config.py")

        ignore = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".DS_Store", ".pytest_cache",
        )
        shutil.copytree(PROJECT / "src",      tmp / "src",      ignore=ignore)
        shutil.copytree(PROJECT / "frontend", tmp / "frontend", ignore=ignore)
        shutil.copytree(PROJECT / "scripts",  tmp / "scripts",  ignore=ignore)

        # Bundle processed feature CSVs (the Dockerfile COPYs these in)
        processed_dst = tmp / "data" / "processed"
        processed_dst.mkdir(parents=True, exist_ok=True)
        for csv_name in (
            "technical_features.csv",
            "daily_sentiment_features.csv",
            "sentiment_labeled.csv",
        ):
            shutil.copy(PROJECT / "data" / "processed" / csv_name, processed_dst / csv_name)

        # Bundle results/ (model_comparison.csv + confusion matrix PNGs)
        shutil.copytree(PROJECT / "results", tmp / "results", ignore=ignore)

        (tmp / "start.sh").chmod(0o755)

        print(f"[3/3] Uploading folder to {repo_id} ...")
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="space",
            commit_message="feat: initial deployment to HuggingFace Spaces",
        )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    print()
    print("=" * 60)
    print(" Space deployed!")
    print("=" * 60)
    print(f"  URL:        {space_url}")
    print(f"  Build logs: {space_url}?logs=container")
    print()
    print("Next: set these in Settings -> Variables and secrets:")
    print(f"  HF_MODEL_REPO       = {USERNAME}/market-pulse-models   (variable)")
    print( "  NEWSDATA_API_KEY    = <your key>                       (secret)")
    print( "  TWELVE_DATA_API_KEY = <your key>                       (secret)")
    print()
    print("First build takes ~10-15 min (FinBERT downloads during Docker build).")


if __name__ == "__main__":
    main()
