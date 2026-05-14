"""Unified training loop with MLflow integration for all sequence models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.config import MODELS_DIR, get_params
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)
from src.models.base_model import BaseSequenceModel
from src.utils.logger import get_logger

log = get_logger(__name__)

_REGRESSION_TARGET = "next_day_return"


def _remap_labels(y: np.ndarray) -> np.ndarray:
    """Remap direction_3class labels {-1, 0, 1} → {0, 1, 2} for CrossEntropyLoss.

    Args:
        y: Integer label array possibly containing -1.

    Returns:
        Remapped array with non-negative integers.
    """
    unique = np.unique(y)
    if -1 in unique:
        return y + 1  # {-1, 0, 1} → {0, 1, 2}
    return y


class Trainer:
    """Trains a BaseSequenceModel with early stopping, LR scheduling, and MLflow.

    Args:
        model: Any model inheriting from BaseSequenceModel.
        target: Name of the target variable being predicted.
        ticker: Ticker symbol (used for MLflow tagging and checkpoint names).
        config: Optional hyperparameter overrides. Falls back to params.yaml.
        results_dir: Where to save plots and artefacts.
    """

    def __init__(
        self,
        model: BaseSequenceModel,
        target: str = "direction_3class",
        ticker: str = "ALL",
        config: dict[str, Any] | None = None,
        results_dir: Path | str = Path("results"),
    ) -> None:
        self.model = model
        self.target = target
        self.ticker = ticker
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        params = get_params()
        cfg = params["training"]
        if config:
            cfg = {**cfg, **config}

        self.lr: float = float(cfg.get("learning_rate", 0.001))
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.max_epochs: int = int(cfg.get("max_epochs", 100))
        self.patience: int = int(cfg.get("early_stopping_patience", 10))
        self.max_grad_norm: float = float(cfg.get("max_grad_norm", 1.0))
        self.weight_decay: float = float(cfg.get("weight_decay", 0.0))

        self.is_regression = (target == _REGRESSION_TARGET)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.MSELoss() if self.is_regression else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,   # L2 regularisation
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

    # ── Data helpers ──────────────────────────────────────────────────────── #

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        if self.is_regression:
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        else:
            y_t = torch.tensor(_remap_labels(y.astype(int)), dtype=torch.long)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    # ── Single epoch ─────────────────────────────────────────────────────── #

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> tuple[float, list[int], list[int]]:
        """Run one epoch of training or evaluation.

        Args:
            loader: DataLoader for the split.
            train: If True, computes gradients and updates weights.

        Returns:
            Tuple of (mean_loss, all_true_labels, all_pred_labels).
        """
        self.model.train(train)
        total_loss = 0.0
        all_true: list[int] = []
        all_pred: list[int] = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                logits = self.model(X_batch)

                if self.is_regression:
                    loss = self.criterion(logits, y_batch)
                    preds = logits.detach().cpu().numpy().flatten()
                    all_pred.extend(preds.tolist())
                    all_true.extend(y_batch.cpu().numpy().flatten().tolist())
                else:
                    loss = self.criterion(logits, y_batch.squeeze(-1))
                    preds = logits.argmax(dim=-1).detach().cpu().numpy()
                    all_pred.extend(preds.tolist())
                    all_true.extend(y_batch.squeeze(-1).cpu().numpy().tolist())

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_loss += loss.item() * len(X_batch)

        return total_loss / max(len(loader.dataset), 1), all_true, all_pred

    # ── Main training entry-point ─────────────────────────────────────────── #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        experiment_name: str = "market-pulse-predictor",
    ) -> dict[str, Any]:
        """Train the model and log everything to MLflow.

        Args:
            X_train: Training sequences (N, lookback, features).
            y_train: Training labels (N,).
            X_val: Validation sequences.
            y_val: Validation labels.
            X_test: Test sequences.
            y_test: Test labels.
            experiment_name: MLflow experiment name.

        Returns:
            Results dict with metrics, paths, and best epoch info.
        """
        model_name = self.model.get_model_name()
        run_name = f"{model_name}_{self.ticker}_{int(time.time())}"

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)
        test_loader = self._make_loader(X_test, y_test, shuffle=False)

        # ── Class weighting (classification only) ──────────────────────────── #
        # Inverse-frequency weights so under-represented classes (e.g. Neutral)
        # contribute proportionally to the loss. Without this, the model
        # collapses to predicting only the majority class.
        if not self.is_regression:
            y_train_remapped = _remap_labels(y_train.astype(int))
            n_classes = int(self.model.fc.out_features)
            counts = np.bincount(y_train_remapped, minlength=n_classes).astype(float)
            counts[counts == 0] = 1.0  # avoid div-by-zero
            weights = counts.sum() / (n_classes * counts)
            class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            log.info("Class weights for %s: %s", self.ticker,
                     {i: round(float(w), 3) for i, w in enumerate(weights)})

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        checkpoint_path = MODELS_DIR / f"{model_name}_{self.ticker}.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            # ── Log parameters ─────────────────────────────────────────── #
            params = get_params()
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "ticker": self.ticker,
                    "target": self.target,
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "num_layers": self.model.num_layers,
                    "output_dim": self.model.output_dim,
                    "dropout": self.model.dropout,
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    "max_epochs": self.max_epochs,
                    "early_stopping_patience": self.patience,
                    "lookback_days": params["features"]["lookback_days"],
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "total_parameters": self.model.count_parameters(),
                }
            )
            mlflow.set_tags(
                {
                    "ticker": self.ticker,
                    "model_type": model_name,
                    "experiment_name": experiment_name,
                    "device": str(self.device),
                }
            )

            # ── Training loop ──────────────────────────────────────────── #
            log.info(
                "Training %s on %s | device=%s epochs=%d",
                model_name.upper(), self.ticker, self.device, self.max_epochs,
            )

            for epoch in range(1, self.max_epochs + 1):
                train_loss, tr_true, tr_pred = self._run_epoch(train_loader, train=True)
                val_loss, val_true, val_pred = self._run_epoch(val_loader, train=False)

                self.scheduler.step(val_loss)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if not self.is_regression:
                    tr_acc = np.mean(np.array(tr_true) == np.array(tr_pred))
                    val_acc = np.mean(np.array(val_true) == np.array(val_pred))
                    train_accs.append(tr_acc)
                    val_accs.append(val_acc)
                    from sklearn.metrics import f1_score
                    val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
                else:
                    tr_acc = val_acc = val_f1 = 0.0
                    train_accs.append(0.0)
                    val_accs.append(0.0)

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1": val_f1,
                    },
                    step=epoch,
                )

                if epoch % 10 == 0 or epoch == 1:
                    log.info(
                        "Epoch %3d/%d | train_loss=%.4f val_loss=%.4f "
                        "val_acc=%.3f val_f1=%.3f",
                        epoch, self.max_epochs,
                        train_loss, val_loss, val_acc, val_f1,
                    )

                # Early stopping + checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        log.info("Early stopping at epoch %d", epoch)
                        break

            # ── Load best weights and evaluate on test ─────────────────── #
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            _, test_true, test_pred = self._run_epoch(test_loader, train=False)

            # ── Final metrics ──────────────────────────────────────────── #
            if self.is_regression:
                final_metrics = compute_regression_metrics(
                    np.array(test_true), np.array(test_pred)
                )
            else:
                final_metrics = compute_classification_metrics(
                    np.array(test_true), np.array(test_pred)
                )

            mlflow.log_metrics(
                {f"test_{k}": v for k, v in final_metrics.items() if isinstance(v, (int, float))}
            )
            log.info("Test metrics for %s on %s: %s", model_name, self.ticker, final_metrics)

            # ── Artifacts ─────────────────────────────────────────────── #
            curves_path = self.results_dir / f"curves_{model_name}_{self.ticker}.png"
            plot_training_curves(
                train_losses, val_losses,
                train_accs if not self.is_regression else None,
                val_accs if not self.is_regression else None,
                save_path=curves_path,
                title=f"{model_name.upper()} — {self.ticker}",
            )
            mlflow.log_artifact(str(curves_path))

            if not self.is_regression:
                class_names = [str(c) for c in sorted(set(test_true))]
                cm_path = self.results_dir / f"cm_{model_name}_{self.ticker}.png"
                plot_confusion_matrix(
                    np.array(test_true),
                    np.array(test_pred),
                    class_names=class_names,
                    save_path=cm_path,
                    title=f"Confusion Matrix — {model_name.upper()} {self.ticker}",
                )
                mlflow.log_artifact(str(cm_path))

            mlflow.log_artifact(str(checkpoint_path))
            mlflow.pytorch.log_model(self.model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id

        result: dict[str, Any] = {
            "model_name": model_name,
            "ticker": self.ticker,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "checkpoint_path": str(checkpoint_path),
            "mlflow_run_id": run_id,
            **{f"test_{k}": v for k, v in final_metrics.items()},
        }
        return result
