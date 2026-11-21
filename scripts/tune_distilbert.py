"""DistilBERT random-search tuning driver (validation only, never test).

Samples N configs over batch_size / lr / epochs (search space below),
trains each for its epoch budget (seed 42) with the EXISTING
:func:`phishing_xai.baselines.train_distilbert` (test set never touched:
``test_texts=None``), and selects on VALIDATION precision subject to
recall >= 0.85 — the same precision-first rule as the RoBERTa tuning
(Stage B) and the BiLSTM tuning driver. If no trial meets the recall
floor, the highest-recall trial wins (then precision) and the output
SAYS SO (``floor_met: false``).

TUNING ONLY: no full-data retraining, no test-set use anywhere. Fitting
uses the TRAIN split (balanced with the train-only synonym augmentation,
then subsampled to ``--subsample_train`` like the RoBERTa tuning);
selection uses VALIDATION only. Trial checkpoints are NOT saved
(``save_best=False``; the final training re-trains with the best config).
Writes ``results/distilbert/tuning/results.json`` (all trials + best
config, incrementally after each trial) and ``best_params.json``.
On Kaggle the default output is redirected under
``/kaggle/working/results/distilbert/tuning``.

Usage (Kaggle, after the repo + data are attached)::

    python scripts/tune_distilbert.py --n_trials 6
    python scripts/tune_distilbert.py --n_trials 6 --output results/distilbert/tuning/results.json
    python scripts/tune_distilbert.py --debug   # 2 trials, 1 epoch, tiny subset
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.augment import balance_training_data  # noqa: E402
from phishing_xai.baselines import (  # noqa: E402
    compute_metrics_full,
    load_splits,
    resolve_output_dir,
    select_best_bilstm_trial,
    train_distilbert,
)
from phishing_xai.utils import ensure_dir, get_logger, set_seed  # noqa: E402

logger = get_logger("phishing_xai.tune_distilbert")

# Random-search grid (task spec): batch [8, 16, 32] x lr [1e-5, 2e-5, 3e-5,
# 5e-5] x epochs [3, 4] = 24 combos; 6 trials sampled without replacement.
SEARCH_SPACE: Dict[str, List[Any]] = {
    "batch_size": [8, 16, 32],
    "lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "epochs": [3, 4],
}
DEFAULT_RECALL_FLOOR: float = 0.85
DEFAULT_MAX_LENGTH: int = 256
DEFAULT_SUBSAMPLE_TRAIN: int = 4000


def sample_configs(rng: random.Random, n_trials: int) -> List[Dict[str, Any]]:
    """Draw ``n_trials`` UNIQUE configs from :data:`SEARCH_SPACE`."""
    combos = [
        {"batch_size": b, "lr": lr, "epochs": e}
        for b in SEARCH_SPACE["batch_size"]
        for lr in SEARCH_SPACE["lr"]
        for e in SEARCH_SPACE["epochs"]
    ]
    rng.shuffle(combos)
    return combos[: min(int(n_trials), len(combos))]


def tune_distilbert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    *,
    n_trials: int = 6,
    seed: int = 42,
    max_length: int = DEFAULT_MAX_LENGTH,
    subsample_train: int = DEFAULT_SUBSAMPLE_TRAIN,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    output_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    on_trial: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """Run the random search and return ``{"trials", "selection", ...}``.

    Each trial trains ONE seed (``seed``) via :func:`train_distilbert`
    with ``test_texts=None`` and is scored on VALIDATION at threshold 0.5
    (``compute_metrics_full`` on the best-epoch validation probabilities).
    The test set is never touched here. Trial checkpoints are not saved.
    ``on_trial`` (when given) is called with the trials so far after each
    completed trial, so the caller can persist incrementally.
    """
    import numpy as np

    if subsample_train and len(train_texts) > subsample_train:
        # Same subsampling as the RoBERTa tuning (tune.py): uniform random
        # draw from the BALANCED train split (already ~50/50 by class).
        rng_np = np.random.default_rng(seed)
        indices = rng_np.choice(len(train_texts), size=subsample_train, replace=False)
        train_texts = [train_texts[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        logger.info(f"Subsampled balanced train to {subsample_train} for tuning.")
    rng = random.Random(seed)
    configs = sample_configs(rng, n_trials)
    trials: List[Dict[str, Any]] = []
    for i, params in enumerate(configs):
        trial_dir = (Path(output_dir) / f"trial_{i}") if output_dir is not None else None
        start = time.perf_counter()
        trained = train_distilbert(
            train_texts, train_labels, val_texts, val_labels,
            test_texts=None, test_labels=None,
            batch_size=int(params["batch_size"]), lr=float(params["lr"]),
            epochs=int(epochs) if epochs is not None else int(params["epochs"]),
            max_length=int(max_length),
            seed=seed, output_dir=trial_dir or f"results/distilbert/tuning/trial_{i}",
            save_best=False, keep_latest=False,
        )
        trained.pop("_model", None)
        trained.pop("_tokenizer", None)
        val_metrics = compute_metrics_full(val_labels, trained["best_val_y_prob"])
        trial = {
            "trial": i,
            "params": params,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "train_info": {
                "best_val_precision": trained["best_val_metrics"].get("precision"),
                "best_val_recall": trained["best_val_metrics"].get("recall"),
                "best_val_f1": trained["best_val_f1"],
                "best_epoch": trained["best_epoch"],
                "epochs_run": trained["epochs_run"],
                "floor_met": trained["floor_met"],
            },
            "train_seconds": round(time.perf_counter() - start, 1),
            "n_params": trained["n_params"],
        }
        trials.append(trial)
        logger.info(
            f"Trial {i}/{len(configs) - 1}: {params} -> "
            f"val precision={trial['val_precision']:.4f} "
            f"recall={trial['val_recall']:.4f} f1={trial['val_f1']:.4f}"
        )
        if on_trial is not None:
            on_trial(list(trials))
    selection = select_best_bilstm_trial(trials, recall_floor=recall_floor)
    return {
        "model": "distilbert",
        "model_name": "distilbert-base-uncased",
        "seed": seed,
        "n_trials": len(trials),
        "max_length": int(max_length),
        "subsample_train": int(subsample_train) if subsample_train else None,
        "recall_floor": recall_floor,
        "floor_met": selection["floor_met"],
        "best_params": selection["best_params"],
        "best_val_precision": selection["best_val_precision"],
        "best_val_recall": selection["best_val_recall"],
        "best_val_f1": selection["best_val_f1"],
        "selection": selection,
        "selection_rule": (
            "Highest validation precision among trials with validation recall >= "
            f"{recall_floor}; if none meets the floor, highest recall then "
            "precision (floor_met=false)."
        ),
        "search_space": SEARCH_SPACE,
        "splits": {"train": len(train_texts), "val": len(val_texts)},
        "trials": trials,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Random-search tune DistilBERT on VALIDATION only (tuning only; no test use, no full-data retrain)."
    )
    parser.add_argument("--n_trials", type=int, default=6,
                        help="Number of sampled configs (default: 6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for config sampling and for each trial's training (default: 42)")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH,
                        help="Token window (default: 256, RoBERTa Stage B precedent)")
    parser.add_argument("--subsample_train", type=int, default=DEFAULT_SUBSAMPLE_TRAIN,
                        help="Balanced-train subsample per trial (default: 4000, RoBERTa Stage B precedent)")
    parser.add_argument("--recall_floor", type=float, default=DEFAULT_RECALL_FLOOR,
                        help="Recall floor for precision-first selection on validation (default: 0.85)")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/distilbert/tuning/results.json, Kaggle-redirected under /kaggle/working)")
    parser.add_argument("--debug", action="store_true",
                        help="Small subset (200 train / 100 val), 1 epoch, 2 trials — smoke test only")
    args = parser.parse_args(argv)

    set_seed(args.seed)
    if args.debug:
        args.n_trials = min(args.n_trials, 2)
        args.subsample_train = 200

    splits = load_splits(args.data_path, args.splits_path, seed=args.seed)
    train_df = splits["train"]
    val_df = splits["val"]
    if args.debug:
        from phishing_xai.baselines import subsample_splits

        splits = subsample_splits(splits, n_train=200, n_val=100, n_test=100, seed=args.seed)
        train_df, val_df = splits["train"], splits["val"]
    try:
        balanced = balance_training_data(
            train_df, text_col="clean_text", label_col="label", seed=args.seed
        )
    except Exception as exc:  # noqa: BLE001 - same fallback as run_baselines.py
        logger.warning(f"Train augmentation failed ({exc}); using unbalanced train split.")
        balanced = train_df.copy().reset_index(drop=True)

    out_path = (Path(args.output) if args.output
                else resolve_output_dir("results/distilbert/tuning") / "results.json")
    out_dir = ensure_dir(out_path.parent)

    def _write_partial(trials_so_far: List[Dict[str, Any]]) -> None:
        # Incremental write after EACH trial so a preempted Kaggle kernel
        # keeps the completed trials (same pattern as tune.py).
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "distilbert", "model_name": "distilbert-base-uncased",
                "seed": args.seed, "status": "running",
                "n_trials": len(trials_so_far), "trials": trials_so_far,
                "search_space": SEARCH_SPACE,
                "splits": {"train": len(balanced), "val": len(val_df)},
            }, f, indent=2, default=str)

    result = tune_distilbert(
        balanced["clean_text"].astype(str).tolist(),
        balanced["label"].astype(int).tolist(),
        val_df["clean_text"].astype(str).tolist(),
        val_df["label"].astype(int).tolist(),
        n_trials=args.n_trials, seed=args.seed,
        max_length=int(args.max_length),
        subsample_train=int(args.subsample_train),
        recall_floor=float(args.recall_floor),
        output_dir=out_dir,
        epochs=1 if args.debug else None,
        on_trial=_write_partial,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {"optimal_parameters": result["best_params"],
             "best_val_precision": result["best_val_precision"],
             "best_val_recall": result["best_val_recall"],
             "best_val_f1": result["best_val_f1"],
             "floor_met": result["floor_met"]},
            f, indent=2, default=str,
        )

    print(f"DistilBERT tuning: {result['n_trials']} trials, seed {args.seed}")
    print(f"Best config: {json.dumps(result['best_params'])}")
    print(f"Best val: precision={result['best_val_precision']:.4f} "
          f"recall={result['best_val_recall']:.4f} f1={result['best_val_f1']:.4f}")
    if result["floor_met"]:
        print(f"Recall floor {args.recall_floor} MET on validation.")
    else:
        print(f"WARNING: NO trial met recall >= {args.recall_floor} on validation; "
              f"selected highest recall then precision (floor_met=false).")
    print(f"Saved {out_path} and {out_dir / 'best_params.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
