"""MLflow tracking configuration.

Import this module before any MLflow calls to ensure the tracking URI and
experiment are set up correctly for both local and Dockerised environments.

Usage:
    import mlflow_config  # noqa: F401  — side-effect import
    import mlflow
    with mlflow.start_run(): ...
"""

from __future__ import annotations

import os

import mlflow

# ── Tracking URI ──────────────────────────────────────────────────────────── #
# Local default: ./mlruns  |  Docker: http://mlflow:5000
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Experiment ────────────────────────────────────────────────────────────── #
EXPERIMENT_NAME = "market-pulse-predictor"

try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as exc:
    # Non-fatal: tracking server may not be available yet at import time
    import warnings
    warnings.warn(f"MLflow setup warning: {exc}", stacklevel=2)

# ── Auto-logging (optional — disable if manual logging is preferred) ──────── #
# mlflow.pytorch.autolog(log_every_n_epoch=5)
