"""Random search hyperparameter tuning with incremental saving, trial resumability, and budget controls."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

from phishing_xai.evaluate import compute_metrics
from phishing_xai.model import create_dataloader
from phishing_xai.utils import (
    ensure_dir,
    get_device,
    get_environment_metadata,
    get_logger,
    load_json,
    save_json,
)

logger = get_logger("phishing_xai.tune")

DEFAULT_PARAM_DISTRIBUTIONS: Dict[str, List[Any]] = {
    "train_batch_size": [8, 16, 32],
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "epochs": [3, 4, 5],
}


def sample_parameters(
    param_distributions: Dict[str, List[Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample one parameter combination randomly."""
    return {
        param: rng.choice(values).item() if hasattr(rng.choice(values), "item") else rng.choice(values)
        for param, values in param_distributions.items()
    }


def train_and_eval_trial(
    model_name: str,
    params: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float, Dict[str, float]]:
    """Train temporary model on sampled parameters and return best validation accuracy and F1."""
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=float(params["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()

    amp_enabled = use_amp and (device.type == "cuda")
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_metrics: Dict[str, float] = {}

    epochs = int(params["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation evaluation
        model.eval()
        val_labels: List[int] = []
        val_probs: List[float] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]

                val_probs.extend(probs.cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

        val_metrics = compute_metrics(val_labels, val_probs)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_acc, best_val_f1, best_metrics


def run_random_search(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    tokenizer: PreTrainedTokenizerFast,
    model_name: str = "roberta-base",
    n_iterations: int = 10,
    max_length: int = 256,
    subsample_train: Optional[int] = 4000,
    param_distributions: Optional[Dict[str, List[Any]]] = None,
    eval_batch_size: int = 64,
    device: Optional[torch.device] = None,
    output_dir: Path | str = Path("results"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute random search hyperparameter tuning with incremental saving and trial resumability."""
    if device is None:
        device = get_device()

    out_dir = Path(output_dir)
    # Put tuning artifacts in out_dir/tuning (unless out_dir already has tuning in name)
    tuning_dir = ensure_dir(out_dir if "tuning" in out_dir.name else out_dir / "tuning")

    results_file = tuning_dir / "results.json"
    best_params_file = tuning_dir / "best_params.json"
    csv_file = tuning_dir / "random_search_results.csv"

    params_space = param_distributions or DEFAULT_PARAM_DISTRIBUTIONS
    rng = np.random.default_rng(seed)

    # Subsample training data if budget control specified
    if subsample_train is not None and len(train_texts) > subsample_train:
        logger.info(f"Subsampling training data to {subsample_train} emails for rapid hyperparameter tuning.")
        indices = rng.choice(len(train_texts), size=subsample_train, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]

    # Resumability check: load existing completed iterations
    all_results: List[Dict[str, Any]] = []
    completed_iterations: Dict[int, Dict[str, Any]] = {}
    best_f1 = -1.0
    best_params: Optional[Dict[str, Any]] = None

    if results_file.exists() and results_file.stat().st_size > 0:
        try:
            existing_data = load_json(results_file)
            if isinstance(existing_data, dict) and "all_results" in existing_data:
                all_results = existing_data["all_results"]
            elif isinstance(existing_data, list):
                all_results = existing_data
            for r in all_results:
                completed_iterations[r["iteration"]] = r
                if r.get("val_f1", -1.0) > best_f1:
                    best_f1 = r["val_f1"]
                    best_params = r["params"]
            logger.info(
                f"Resuming tuning session: found {len(completed_iterations)} already-completed iterations. "
                f"Current best F1: {best_f1:.4f}"
            )
        except Exception as e:
            logger.warning(f"Could not parse existing results.json ({e}); starting fresh.")
            all_results = []
            completed_iterations = {}

    logger.info("=" * 65)
    logger.info(f"STARTING RANDOM SEARCH ({n_iterations} iterations, max_len={max_length})")
    logger.info(f"Parameter Space: {params_space}")
    logger.info("=" * 65)

    for iteration in range(1, n_iterations + 1):
        sampled_params = sample_parameters(params_space, rng)

        if iteration in completed_iterations:
            logger.info(f"[Iteration {iteration}/{n_iterations}] Skipping already completed iteration.")
            continue

        logger.info(f"\n[Iteration {iteration}/{n_iterations}] Sampled: {sampled_params}")

        # Create dataloaders with sampled train_batch_size
        train_loader = create_dataloader(
            texts=train_texts,
            labels=train_labels,
            tokenizer=tokenizer,
            batch_size=int(sampled_params["train_batch_size"]),
            shuffle=True,
            max_length=max_length,
        )
        val_loader = create_dataloader(
            texts=val_texts,
            labels=val_labels,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            shuffle=False,
            max_length=max_length,
        )

        try:
            val_acc, val_f1, val_metrics = train_and_eval_trial(
                model_name=model_name,
                params=sampled_params,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )

            result_entry = {
                "iteration": iteration,
                "params": sampled_params,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "val_metrics": val_metrics,
            }
            all_results.append(result_entry)
            completed_iterations[iteration] = result_entry

            logger.info(f"Iteration {iteration} -> Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = sampled_params
                logger.info(f"*** NEW BEST VALIDATION F1: {best_f1:.4f} ***")

            # Incrementally write results.json after EACH completed iteration
            partial_dict = {
                "optimal_parameters": best_params,
                "best_f1_score": float(best_f1),
                "max_length": max_length,
                "subsample_train": subsample_train,
                "all_results": all_results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "search_space": params_space,
                "environment": get_environment_metadata(seed=seed),
            }
            save_json(partial_dict, results_file)
            save_json(
                {
                    "optimal_parameters": best_params,
                    "best_f1_score": float(best_f1),
                    "max_length": max_length,
                    "subsample_train": subsample_train,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                best_params_file,
            )

            # Update CSV
            rows = []
            for r in all_results:
                row = r["params"].copy()
                row["iteration"] = r["iteration"]
                row["val_accuracy"] = r["val_accuracy"]
                row["val_f1"] = r["val_f1"]
                rows.append(row)
            pd.DataFrame(rows).sort_values("val_f1", ascending=False).to_csv(csv_file, index=False)

        except Exception as e:
            logger.error(f"Error in trial {iteration}: {e}")
            continue

    final_results = {
        "optimal_parameters": best_params,
        "best_f1_score": float(best_f1),
        "max_length": max_length,
        "subsample_train": subsample_train,
        "all_results": all_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "search_space": params_space,
        "environment": get_environment_metadata(seed=seed),
    }

    logger.info("=" * 65)
    logger.info(f"OPTIMAL PARAMETERS: {best_params} (Best Val F1: {best_f1:.4f})")
    logger.info(f"Saved artifacts to {tuning_dir}")
    logger.info("=" * 65)

    return final_results
