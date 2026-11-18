"""Evaluation metrics, confusion matrix, ROC-AUC curve, and figure generation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI and headless runs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from phishing_xai.utils import ensure_dir, get_logger

logger = get_logger("phishing_xai.evaluate")
DEFAULT_CLASSES = ["Safe Email", "Phishing Email"]


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics from ground truth labels and positive-class probabilities."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc = 0.5  # Fallback if only 1 class is present in slice

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }


def get_classification_report_str(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    target_names: Optional[List[str]] = None,
) -> str:
    """Return formatted classification report."""
    if target_names is None:
        target_names = DEFAULT_CLASSES
    return classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)


def plot_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    save_path: Union[Path, str],
    class_names: Optional[List[str]] = None,
) -> Path:
    """Generate and save confusion matrix heatmap.
    
    Always saves before display and closes the figure cleanly.
    """
    if class_names is None:
        class_names = DEFAULT_CLASSES

    p = Path(save_path)
    ensure_dir(p.parent)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()

    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to: {p}")
    return p


def plot_roc_curve(
    y_true: Union[List[int], np.ndarray],
    y_prob: Union[List[float], np.ndarray],
    save_path: Union[Path, str],
) -> Path:
    """Generate and save ROC-AUC curve figure."""
    p = Path(save_path)
    ensure_dir(p.parent)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"RoBERTa (AUC = {roc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curve figure to: {p}")
    return p


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Union[Path, str],
) -> Path:
    """Plot training and validation loss and F1 curves over epochs."""
    p = Path(save_path)
    ensure_dir(p.parent)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # F1 plot
    ax2.plot(epochs, history["train_f1"], "o-", label="Train F1", color="#2ca02c")
    ax2.plot(epochs, history["val_f1"], "s--", label="Val F1", color="#d62728")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("F1 Score", fontweight="bold")
    ax2.set_title("Training & Validation F1", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves figure to: {p}")
    return p
