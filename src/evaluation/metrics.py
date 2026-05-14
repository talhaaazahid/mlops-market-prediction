"""Comprehensive evaluation metrics and visualisation for all model types.

Provides:
  - compute_classification_metrics   — accuracy, F1, precision, recall
  - compute_regression_metrics       — RMSE, MAE, R², directional accuracy
  - plot_training_curves             — loss + accuracy curves
  - plot_confusion_matrix            — seaborn heatmap
  - plot_model_comparison_bar        — grouped bar chart across models/tickers
  - plot_predictions_vs_actual       — time-series overlay
  - generate_evaluation_report       — full dict ready for MLflow
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from src.utils.logger import get_logger

matplotlib.use("Agg")  # headless rendering — no display required

log = get_logger(__name__)

_RESULTS_DIR = Path("results")


# ── Classification ────────────────────────────────────────────────────────── #

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute full classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        save_report_path: If provided, writes the sklearn classification report
            as a .txt file to this path.

    Returns:
        Dict with keys:
        accuracy, f1_macro, f1_weighted, precision_macro, recall_macro,
        per_class_f1, per_class_precision, per_class_recall.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision, recall, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class_f1 = {str(c): round(float(f), 4) for c, f in zip(classes, f1_per)}
    per_class_precision = {str(c): round(float(p), 4) for c, p in zip(classes, precision)}
    per_class_recall = {str(c): round(float(r), 4) for c, r in zip(classes, recall)}

    report_str = classification_report(y_true, y_pred, zero_division=0)

    if save_report_path is not None:
        save_report_path = Path(save_report_path)
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        save_report_path.write_text(report_str)
        log.info("Classification report saved to %s", save_report_path)

    return {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(float(precision.mean()), 4),
        "recall_macro": round(float(recall.mean()), 4),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "classification_report": report_str,
    }


# ── Regression ────────────────────────────────────────────────────────────── #

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground-truth continuous values.
        y_pred: Model predictions.

    Returns:
        Dict with keys: rmse, mae, r2, directional_accuracy.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Directional accuracy: did we predict the correct sign?
    correct_direction = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = float(correct_direction.mean())

    return {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "directional_accuracy": round(directional_accuracy, 4),
    }


# ── Visualisation helpers ─────────────────────────────────────────────────── #

def _savefig(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    """Save a matplotlib figure and close it.

    Args:
        fig: Figure to save.
        path: Destination file path (.png).
        dpi: Output resolution.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to %s", path)


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    save_path: Path | str = _RESULTS_DIR / "training_curves.png",
    title: str = "Training Curves",
) -> None:
    """Plot loss (and optionally accuracy) curves across epochs.

    Args:
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
        train_accs: Per-epoch training accuracy (omit for regression).
        val_accs: Per-epoch validation accuracy (omit for regression).
        save_path: Output PNG path.
        title: Figure title.
    """
    save_path = Path(save_path)
    has_acc = train_accs is not None and val_accs is not None

    n_plots = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation", linewidth=1.5, linestyle="--")
    best_epoch = int(np.argmin(val_losses)) + 1
    ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.6, label=f"Best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy subplot
    if has_acc:
        ax2 = axes[1]
        ax2.plot(epochs, train_accs, label="Train", linewidth=1.5)
        ax2.plot(epochs, val_accs, label="Validation", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    save_path: Path | str = _RESULTS_DIR / "confusion_matrix.png",
    title: str = "Confusion Matrix",
    normalise: bool = True,
) -> None:
    """Plot a seaborn confusion-matrix heatmap.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        class_names: Display labels for each class.
        save_path: Output PNG path.
        title: Figure title.
        normalise: Show row-normalised percentages when True.
    """
    save_path = Path(save_path)
    cm = confusion_matrix(y_true, y_pred)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm.astype(float) / np.where(row_sums == 0, 1, row_sums)
        fmt = ".2f"
        cbar_label = "Proportion"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"

    labels = class_names or [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), max(4, len(labels) * 1.2)))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)


def plot_model_comparison_bar(
    results_df: pd.DataFrame,
    metric: str = "test_accuracy",
    save_path: Path | str = _RESULTS_DIR / "model_comparison.png",
    title: str = "Model Comparison",
) -> None:
    """Grouped bar chart comparing all models across tickers.

    Args:
        results_df: DataFrame with columns [model_name, ticker, <metric>].
        metric: Column name to plot (e.g. "test_accuracy", "test_f1_macro").
        save_path: Output PNG path.
        title: Figure title.
    """
    save_path = Path(save_path)

    if results_df.empty or metric not in results_df.columns:
        log.warning("plot_model_comparison_bar: results empty or metric '%s' missing", metric)
        return

    pivot = results_df.pivot_table(index="ticker", columns="model_name", values=metric)
    n_tickers = len(pivot)
    n_models = len(pivot.columns)

    fig, ax = plt.subplots(figsize=(max(8, n_tickers * 1.5), 5))
    x = np.arange(n_tickers)
    width = 0.8 / n_models

    colors = plt.cm.tab10(np.linspace(0, 0.8, n_models))  # type: ignore[attr-defined]
    for i, model in enumerate(pivot.columns):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, pivot[model], width, label=model, color=colors[i], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.005,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.05, pivot.values[~np.isnan(pivot.values)].max() * 1.15)
                if not pivot.empty else 1.0)
    fig.tight_layout()
    _savefig(fig, save_path)


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: list | pd.DatetimeIndex | None = None,
    save_path: Path | str = _RESULTS_DIR / "predictions_vs_actual.png",
    title: str = "Predictions vs Actual",
    max_points: int = 200,
) -> None:
    """Time-series overlay of predicted vs actual values (regression).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        dates: Optional date axis. Falls back to integer indices.
        save_path: Output PNG path.
        title: Figure title.
        max_points: Truncate to this many points to keep the chart readable.
    """
    save_path = Path(save_path)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n = min(len(y_true), max_points)
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    x_axis = list(dates)[:n] if dates is not None else list(range(n))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_axis, y_true, label="Actual", linewidth=1.2, alpha=0.8)
    ax.plot(x_axis, y_pred, label="Predicted", linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel("Date" if dates is not None else "Index")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    if dates is not None:
        fig.autofmt_xdate()

    fig.tight_layout()
    _savefig(fig, save_path)


# ── Full report ───────────────────────────────────────────────────────────── #

def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    ticker: str,
    is_regression: bool = False,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
    dates: list | pd.DatetimeIndex | None = None,
    output_dir: Path | str = _RESULTS_DIR,
) -> dict[str, Any]:
    """Generate a full evaluation report with metrics and saved plots.

    Args:
        y_true: Ground-truth labels or values.
        y_pred: Predicted labels or values.
        model_name: Short model identifier (e.g. "lstm").
        ticker: Ticker symbol (e.g. "AAPL").
        is_regression: True for regression targets.
        train_losses: Training loss history (used for curve plot).
        val_losses: Validation loss history.
        train_accs: Training accuracy history (classification only).
        val_accs: Validation accuracy history.
        dates: Date axis for the predictions-vs-actual plot.
        output_dir: Root directory for saving plots and reports.

    Returns:
        Dict containing all scalar metrics plus plot file paths.
        Suitable for logging to MLflow as params/metrics/artifacts.
    """
    output_dir = Path(output_dir)
    tag = f"{model_name}_{ticker}"

    report: dict[str, Any] = {
        "model_name": model_name,
        "ticker": ticker,
    }

    # ── Metrics ───────────────────────────────────────────────────────────── #
    if is_regression:
        metrics = compute_regression_metrics(y_true, y_pred)
        report.update(metrics)
    else:
        report_path = output_dir / f"cls_report_{tag}.txt"
        metrics = compute_classification_metrics(y_true, y_pred, save_report_path=report_path)
        report.update({k: v for k, v in metrics.items() if k != "classification_report"})
        report["classification_report_path"] = str(report_path)

    # ── Plots ─────────────────────────────────────────────────────────────── #
    plot_paths: dict[str, str] = {}

    if train_losses and val_losses:
        curves_path = output_dir / f"curves_{tag}.png"
        plot_training_curves(
            train_losses, val_losses, train_accs, val_accs,
            save_path=curves_path,
            title=f"{model_name.upper()} — {ticker} Training Curves",
        )
        plot_paths["training_curves"] = str(curves_path)

    if not is_regression:
        class_names = [str(c) for c in sorted(set(np.concatenate([y_true, y_pred]).astype(int)))]
        cm_path = output_dir / f"cm_{tag}.png"
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=class_names,
            save_path=cm_path,
            title=f"Confusion Matrix — {model_name.upper()} {ticker}",
        )
        plot_paths["confusion_matrix"] = str(cm_path)
    else:
        pva_path = output_dir / f"pva_{tag}.png"
        plot_predictions_vs_actual(
            y_true, y_pred, dates=dates,
            save_path=pva_path,
            title=f"Predictions vs Actual — {model_name.upper()} {ticker}",
        )
        plot_paths["predictions_vs_actual"] = str(pva_path)

    report["plot_paths"] = plot_paths
    return report


if __name__ == "__main__":
    # Smoke test with synthetic data
    rng = np.random.default_rng(42)
    n = 200

    print("── Classification metrics ──")
    y_true_cls = rng.integers(0, 3, size=n)
    y_pred_cls = rng.integers(0, 3, size=n)
    cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
    print(f"  Accuracy : {cls_metrics['accuracy']}")
    print(f"  F1 macro : {cls_metrics['f1_macro']}")
    print(f"  Per-class F1: {cls_metrics['per_class_f1']}")

    print("\n── Regression metrics ──")
    y_true_reg = rng.normal(0, 0.02, size=n)
    y_pred_reg = y_true_reg + rng.normal(0, 0.005, size=n)
    reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)
    print(f"  RMSE                : {reg_metrics['rmse']}")
    print(f"  MAE                 : {reg_metrics['mae']}")
    print(f"  R²                  : {reg_metrics['r2']}")
    print(f"  Directional accuracy: {reg_metrics['directional_accuracy']}")

    print("\n── Generating sample plots → results/ ──")
    _RESULTS_DIR.mkdir(exist_ok=True)
    fake_losses = (np.exp(-np.linspace(0, 3, 50)) + rng.normal(0, 0.01, 50)).tolist()
    fake_accs = (1 - np.exp(-np.linspace(0, 3, 50)) * 0.5 + rng.normal(0, 0.01, 50)).tolist()
    plot_training_curves(fake_losses, [l * 1.1 for l in fake_losses], fake_accs,
                         [a * 0.95 for a in fake_accs])
    plot_confusion_matrix(y_true_cls, y_pred_cls, class_names=["Down", "Neutral", "Up"])
    print("  Done — check results/ folder")

def calculate_sharpe_ratio(returns):
    Calculate Sharpe ratio.
    return returns.mean() / returns.std() * np.sqrt(252)


def calculate_sharpe_ratio(returns):
    Calculate Sharpe ratio.
    return returns.mean() / returns.std() * np.sqrt(252)

