"""Model training loop with validation-based checkpointing, AMP, early stopping, and final test evaluation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from phishing_xai.evaluate import (
    compute_metrics,
    get_classification_report_str,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from phishing_xai.utils import (
    ensure_dir,
    get_device,
    get_environment_metadata,
    get_logger,
    save_json,
)

logger = get_logger("phishing_xai.train")


def evaluate_dataloader(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader; returns average loss, true labels, predicted labels, and probabilities."""
    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(dataloader))
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    return avg_loss, y_true, y_pred, y_prob


def train_model(
    model: PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    early_stopping_patience: int = 3,
    device: Optional[torch.device] = None,
    output_dir: Path | str = Path("results"),
    ckpt_dir: Optional[Path | str] = None,
    checkpoint_name: str = "best_model.pt",
    seed: int = 42,
    is_debug: bool = False,
    resume: bool = False,
    keep_latest: bool = False,
) -> Dict[str, Any]:
    """Train the model with validation-F1 checkpointing, full resumability, and single-pass final test evaluation.
    
    IMPORTANT: Test set is NEVER touched inside the epoch training loop.
    """
    if device is None:
        device = get_device()

    out_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(ckpt_dir) if ckpt_dir is not None else out_dir
    checkpoint_path = checkpoints_dir / checkpoint_name
    latest_checkpoint_path = checkpoints_dir / "latest_checkpoint.pt"


    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # AMP setup: only active when CUDA is available
    amp_enabled = use_amp and (device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
    }

    start_epoch = 1
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    if resume and latest_checkpoint_path.exists():
        logger.info(f"Resuming training state from: {latest_checkpoint_path}")
        chk = torch.load(latest_checkpoint_path, map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        if amp_enabled and chk.get("scaler_state_dict") is not None:
            scaler.load_state_dict(chk["scaler_state_dict"])
        start_epoch = chk["epoch"] + 1
        best_val_f1 = chk.get("best_val_f1", -1.0)
        best_epoch = chk.get("best_epoch", 0)
        history = chk.get("history", history)
        logger.info(f"Successfully resumed at epoch {start_epoch} (previous best val F1: {best_val_f1:.4f})")

    logger.info(
        f"Starting training: Device={device}, Epochs={start_epoch}..{epochs}, LR={learning_rate}, "
        f"GradAccum={gradient_accumulation_steps}, AMP={amp_enabled}, Patience={early_stopping_patience}"
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_labels_epoch: List[int] = []
        train_probs_epoch: List[float] = []

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            train_loss_total += loss.item() * (gradient_accumulation_steps if gradient_accumulation_steps > 1 else 1)

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                train_probs_epoch.extend(probs.cpu().numpy().tolist())
                train_labels_epoch.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # Epoch metrics
        train_epoch_loss = train_loss_total / len(train_loader)
        train_epoch_metrics = compute_metrics(train_labels_epoch, train_probs_epoch)

        # Validation phase (Test set is NOT touched!)
        val_loss, val_true, val_pred, val_prob = evaluate_dataloader(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_prob)

        history["train_loss"].append(train_epoch_loss)
        history["train_f1"].append(train_epoch_metrics["f1"])
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_epoch_loss:.4f}, Train F1: {train_epoch_metrics['f1']:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # 1. Always save latest checkpoint (model, optimizer, scaler, epoch, best_f1) for full resumability
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if amp_enabled else None,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "history": history,
                "val_metrics": val_metrics,
            },
            latest_checkpoint_path,
        )

        # 2. Checkpoint selection based on validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0

            logger.info(f"--> Validation F1 improved ({best_val_f1:.4f}). Saving best checkpoint: {checkpoint_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_val_f1,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            logger.info(f"Validation F1 did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # Save training curves figure (train vs val only)
    curves_path = out_dir / "loss_and_f1_curves.png"
    plot_training_curves(history, curves_path)

    # -------------------------------------------------------------
    # FINAL EVALUATION: Load best checkpoint and evaluate on test set ONCE
    # -------------------------------------------------------------
    logger.info("=" * 65)
    logger.info(f"Loading best checkpoint from epoch {best_epoch} for final test evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss, test_true, test_pred, test_prob = evaluate_dataloader(model, test_loader, device)
    test_metrics = compute_metrics(test_true, test_prob)
    class_report_str = get_classification_report_str(test_true, test_pred)

    logger.info("FINAL TEST RESULTS (Evaluated exactly once on best checkpoint):")
    logger.info(f"Test Loss:     {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Test F1:       {test_metrics['f1']:.4f}")
    logger.info(f"Test Recall:   {test_metrics['recall']:.4f}")
    logger.info(f"Test Precision:{test_metrics['precision']:.4f}")
    logger.info(f"Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
    logger.info(f"\nClassification Report:\n{class_report_str}")

    # Generate and save figures
    cm_path = out_dir / "confusion_matrix.png"
    roc_path = out_dir / "roc_auc_curve.png"
    plot_confusion_matrix(test_true, test_pred, cm_path)
    plot_roc_curve(test_true, test_prob, roc_path)

    # Compile run summary
    metrics_summary = {
        "debug": is_debug,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_metrics": checkpoint["val_metrics"],
        "final_test_loss": round(test_loss, 4),
        "final_test_metrics": test_metrics,
        "classification_report": class_report_str,
        "history": history,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "confusion_matrix": str(cm_path),
            "roc_curve": str(roc_path),
            "training_curves": str(curves_path),
        },
        "environment": get_environment_metadata(seed=seed, extra_config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "amp_enabled": amp_enabled,
        }),
    }

    metrics_json_path = out_dir / "metrics.json"
    save_json(metrics_summary, metrics_json_path)
    logger.info(f"Saved complete metrics summary to: {metrics_json_path}")

    # Auto-cleanup intermediate resumability checkpoint unless requested otherwise
    if not keep_latest and latest_checkpoint_path.exists():
        try:
            latest_checkpoint_path.unlink()
            logger.info(f"Cleaned up intermediate checkpoint: {latest_checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not remove latest_checkpoint.pt ({e})")

    logger.info("=" * 65)

    return metrics_summary

