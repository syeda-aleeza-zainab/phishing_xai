"""BiLSTM random-search tuning driver (Step 6B.2/6B.4 prep).

Samples N configs over emb_dim / hidden / layers / dropout / lr / max_len /
batch_size, trains each for a small epoch budget (default 3 epochs, seed 42)
with the EXISTING :func:`phishing_xai.baselines.build_torch_baseline`, and
selects on VALIDATION precision subject to recall >= 0.85 — the same rule as
the RoBERTa tuning. If no config meets the recall floor, the highest-recall
trial wins (then precision) and the output SAYS SO (``floor_met: false``).

TUNING ONLY: no retraining on the full data, no test-set use anywhere.
Fitting uses the TRAIN split; selection uses VALIDATION only. Writes
``results/bilstm/tuning.json`` (all trials + best config) and prints the
best config. On Kaggle the default output is redirected under
``/kaggle/working/results/bilstm/tuning.json``.

Usage (Kaggle, after the repo + data are attached)::

    python scripts/tune_bilstm.py --n_trials 10 --epochs 3
    python scripts/tune_bilstm.py --n_trials 10 --epochs 3 --output results/bilstm/tuning.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.baselines import (
    TorchTrainConfig,
    build_torch_baseline,
    compute_metrics_full,
    load_splits,
    resolve_output_dir,
    select_best_bilstm_trial,
)
from phishing_xai.utils import get_logger, set_seed

logger = get_logger("phishing_xai.tune_bilstm")

# Random-search grid (Step 6B.2/6B.4 spec). Sampled uniformly by trial index.
SEARCH_SPACE: Dict[str, List[Any]] = {
    "emb_dim": [100, 200],
    "hidden": [64, 128, 256],
    "layers": [1, 2],
    "dropout": [0.2, 0.3, 0.5],
    "lr": [5e-4, 1e-3, 2e-3],
    "max_len": [128, 256],
    "batch_size": [16, 32],
}
DEFAULT_RECALL_FLOOR: float = 0.85


def sample_config(rng: random.Random) -> Dict[str, Any]:
    """Draw one config uniformly from :data:`SEARCH_SPACE`."""
    return {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}


def tune_bilstm(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    *,
    n_trials: int = 10,
    epochs: int = 3,
    seed: int = 42,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> Dict[str, Any]:
    """Run the random search and return ``{"trials", "selection", ...}``.

    Each trial trains ONE seed (``seed``) via :func:`build_torch_baseline`
    and is scored on VALIDATION at threshold 0.5 (``compute_metrics_full``).
    The test set is never touched here.
    """
    rng = random.Random(seed)
    trials: List[Dict[str, Any]] = []
    for i in range(int(n_trials)):
        params = sample_config(rng)
        config = TorchTrainConfig(
            emb_dim=params["emb_dim"], hidden=params["hidden"],
            layers=params["layers"], dropout=params["dropout"],
            lr=params["lr"], max_len=params["max_len"],
            batch_size=params["batch_size"], epochs=int(epochs),
        )
        start = time.perf_counter()
        model, train_info = build_torch_baseline(
            "bilstm", train_texts, train_labels, val_texts, val_labels,
            config=config, seed=seed,
        )
        val_prob = model.predict_proba(val_texts)[:, 1]
        val_metrics = compute_metrics_full(val_labels, val_prob)
        trial = {
            "trial": i,
            "params": params,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "train_info": {
                "best_val_precision": train_info["best_val_precision"],
                "best_val_recall": train_info["best_val_recall"],
                "best_val_f1": train_info["best_val_f1"],
                "epochs_run": train_info["epochs_run"],
                "floor_met": train_info["floor_met"],
            },
            "train_seconds": round(time.perf_counter() - start, 1),
            "n_params": model.n_params,
        }
        trials.append(trial)
        logger.info(
            f"Trial {i}/{int(n_trials) - 1}: {params} -> "
            f"val precision={trial['val_precision']:.4f} "
            f"recall={trial['val_recall']:.4f} f1={trial['val_f1']:.4f}"
        )
    selection = select_best_bilstm_trial(trials, recall_floor=recall_floor)
    return {
        "seed": seed,
        "n_trials": len(trials),
        "epochs": int(epochs),
        "recall_floor": recall_floor,
        "floor_met": selection["floor_met"],
        "best_params": selection["best_params"],
        "best_val_precision": selection["best_val_precision"],
        "best_val_recall": selection["best_val_recall"],
        "best_val_f1": selection["best_val_f1"],
        "selection": selection,
        "trials": trials,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Random-search tune the BiLSTM on VALIDATION only (tuning only; no test use, no full-data retrain)."
    )
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of sampled configs (default: 10)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epoch budget per trial (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for config sampling and for each trial's training (default: 42)")
    parser.add_argument("--recall_floor", type=float, default=DEFAULT_RECALL_FLOOR,
                        help="Recall floor for precision-first selection on validation (default: 0.85)")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/bilstm/tuning.json, Kaggle-redirected under /kaggle/working)")
    parser.add_argument("--debug", action="store_true",
                        help="Small subset (200 train / 100 val), 1 epoch, 2 trials — smoke test only")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    if args.debug:
        args.n_trials = min(args.n_trials, 2)
        args.epochs = 1

    splits = load_splits(args.data_path, args.splits_path, seed=args.seed)
    train = splits["train"]
    val = splits["val"]
    if args.debug:
        from phishing_xai.baselines import subsample_splits

        splits = subsample_splits(splits, n_train=200, n_val=100, n_test=100, seed=args.seed)
        train, val = splits["train"], splits["val"]
    result = tune_bilstm(
        train["clean_text"].astype(str).tolist(),
        train["label"].astype(int).tolist(),
        val["clean_text"].astype(str).tolist(),
        val["label"].astype(int).tolist(),
        n_trials=args.n_trials, epochs=args.epochs,
        seed=args.seed, recall_floor=args.recall_floor,
    )
    result["splits"] = {"train": len(train), "val": len(val)}
    result["selection_rule"] = (
        "Highest validation precision among trials with validation recall >= "
        f"{args.recall_floor}; if none meets the floor, highest recall then "
        "precision (floor_met=false)."
    )

    out_path = Path(args.output) if args.output else resolve_output_dir("results/bilstm") / "tuning.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"BiLSTM tuning: {result['n_trials']} trials, {result['epochs']} epochs each, seed {args.seed}")
    print(f"Best config: {json.dumps(result['best_params'])}")
    print(f"Best val: precision={result['best_val_precision']:.4f} "
          f"recall={result['best_val_recall']:.4f} f1={result['best_val_f1']:.4f}")
    if result["floor_met"]:
        print(f"Recall floor {args.recall_floor} MET on validation.")
    else:
        print(f"WARNING: NO trial met recall >= {args.recall_floor} on validation; "
              f"selected highest recall then precision (floor_met=false).")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
