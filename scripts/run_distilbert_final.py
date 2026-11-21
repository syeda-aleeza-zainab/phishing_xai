"""DistilBERT final training (3 seeds, best tuning config, test once per seed).

Trains DistilBERT on the FULL balanced train split for seeds 42, 1, 7
with the best config from ``scripts/tune_distilbert.py``
(``results/distilbert/tuning/results.json``; explicit ``--batch_size`` /
``--lr`` / ``--epochs`` override it). Mixed precision is on when CUDA is
available (AMP inside :func:`phishing_xai.baselines.train_distilbert`).
Each seed is resumable from its per-epoch ``latest_checkpoint.pt`` and
writes its outputs IMMEDIATELY after finishing — before the next seed
starts — so a preempted Kaggle kernel keeps every completed seed:

    results/distilbert/seed_<n>/
        best_distilbert.pt      standalone checkpoint (state dict + config)
        tokenizer/              saved tokenizer files
        config.json             saved model config
        latest_checkpoint.pt    per-epoch resumable state (kept)
        metrics.json            full per-seed metrics (see below)
        test_pred.npy / test_prob.npy   per-email test predictions (McNemar)

``metrics.json`` per seed holds: accuracy, precision, recall, F1,
ROC-AUC, PR-AUC, FPR, confusion matrix, n_params, size MB, train time
(seconds), best epoch, validation metrics (full set at the best epoch),
the validation-fitted threshold view (target recall 0.90, fit on
VALIDATION only), latency (median of 200, batch 1, CPU and GPU when
available) and peak memory. The test set is evaluated EXACTLY ONCE per
seed, inside ``train_distilbert`` on the frozen best state; the latency
timing afterwards is inference-only on that frozen model, not a second
scored evaluation. ``summary.json`` in ``results/distilbert/`` is updated
after each seed and flags the best-validation seed at the end.

Usage (Kaggle, after the repo + data + tuning results are attached)::

    python scripts/run_distilbert_final.py --seeds 42 1 7
    python scripts/run_distilbert_final.py --seeds 42 --resume
    python scripts/run_distilbert_final.py --debug   # 1 seed, tiny subset, 1 epoch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from phishing_xai.augment import balance_training_data  # noqa: E402
from phishing_xai.baselines import (  # noqa: E402
    DEFAULT_SEEDS,
    aggregate_seed_results,
    build_distilbert_wrapper,
    compute_metrics_full,
    fit_threshold_for_recall,
    load_splits,
    measure_latency,
    resolve_output_dir,
    train_distilbert,
)
from phishing_xai.utils import ensure_dir, get_device, get_logger, load_json, save_json, set_seed  # noqa: E402

logger = get_logger("phishing_xai.run_distilbert_final")

DEFAULT_MAX_LENGTH: int = 256


def load_best_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Best tuning config from a tuning ``results.json`` / ``best_params.json``.

    Accepts the ``best_params`` key of ``tune_distilbert.py``'s
    ``results.json`` (``{"batch_size", "lr", "epochs"}``) or its
    ``best_params.json`` (``{"optimal_parameters": {...}}``). Raises
    ``FileNotFoundError`` / ``ValueError`` when the file or the config is
    missing — the final run never guesses a config silently.
    """
    if config_path is None:
        raise ValueError(
            "No tuning config given: pass --config results/distilbert/tuning/results.json "
            "or explicit --batch_size/--lr/--epochs."
        )
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Tuning config not found: {config_path}")
    data = load_json(config_path)
    params = data.get("best_params") or (data.get("optimal_parameters") or {})
    if not params or not all(k in params for k in ("batch_size", "lr", "epochs")):
        raise ValueError(
            f"Tuning config {config_path} has no usable best params "
            f"(looked for best_params / optimal_parameters with batch_size, lr, epochs)."
        )
    return {"batch_size": int(params["batch_size"]), "lr": float(params["lr"]),
            "epochs": int(params["epochs"])}


def run_seed(
    seed: int,
    train_df,
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    config: Dict[str, Any],
    out_root: Path,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    weight_decay: float = 0.01,
    patience: int = 3,
    resume: bool = False,
) -> Dict[str, Any]:
    """Train + evaluate ONE seed; write its metrics.json immediately."""
    set_seed(seed)
    device = get_device()
    seed_dir = ensure_dir(out_root / f"seed_{seed}")
    try:
        balanced = balance_training_data(
            train_df, text_col="clean_text", label_col="label", seed=seed
        )
    except Exception as exc:  # noqa: BLE001 - same fallback as run_baselines.py
        logger.warning(f"Seed {seed}: train augmentation failed ({exc}); using unbalanced train split.")
        balanced = train_df.copy().reset_index(drop=True)
    trained = train_distilbert(
        balanced["clean_text"].astype(str).tolist(),
        balanced["label"].astype(int).tolist(),
        val_texts, val_labels,
        test_texts=test_texts, test_labels=test_labels,
        batch_size=int(config["batch_size"]), lr=float(config["lr"]),
        epochs=int(config["epochs"]), max_length=int(max_length),
        weight_decay=float(weight_decay), patience=int(patience),
        seed=seed, device=device, output_dir=seed_dir,
        checkpoint_name="best_distilbert.pt", save_best=True,
        resume=resume, use_amp=True, keep_latest=True,
    )
    model = trained.pop("_model")
    tokenizer = trained.pop("_tokenizer")
    wrapper = build_distilbert_wrapper(
        model, tokenizer, device=device, max_length=int(max_length),
        extra={"model_name": "distilbert-base-uncased",
               "best_epoch": trained.get("best_epoch")},
    )
    test_prob = np.asarray(trained["y_prob"], dtype=float)
    test_labels_arr = np.asarray(test_labels, dtype=int)

    # Validation-fitted threshold view (target recall 0.90, fit on
    # VALIDATION probabilities of the best epoch only, applied unchanged).
    threshold_view: Dict[str, Any] = {}
    try:
        thr = fit_threshold_for_recall(val_labels, trained["best_val_y_prob"], target_recall=0.90)
        threshold_view = {
            **thr,
            "metrics_at_val_threshold": compute_metrics_full(
                test_labels_arr, test_prob, threshold=thr["threshold"]
            ),
        }
    except Exception as exc:  # noqa: BLE001 - record, do not fail the seed
        logger.warning(f"Seed {seed}: validation-threshold view failed ({exc}).")
        threshold_view = {"status": "NOT RUN YET", "reason": str(exc)}

    # Latency (median of 200, batch 1) on the frozen best model: CPU and,
    # when CUDA is available, GPU via set_device — inference-only timing.
    latency: Dict[str, Any] = {}
    try:
        latency["cpu"] = measure_latency(wrapper.predict_proba, test_texts, n=200)
        latency["cpu"]["device"] = "cpu"
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Seed {seed}: CPU latency failed ({exc}).")
        latency["cpu"] = {"status": "NOT RUN YET", "reason": str(exc)}
    if torch.cuda.is_available():
        try:
            wrapper.set_device(torch.device("cuda"))
            latency["gpu"] = measure_latency(wrapper.predict_proba, test_texts, n=200)
            latency["gpu"]["device"] = "cuda"
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Seed {seed}: GPU latency failed ({exc}).")
            latency["gpu"] = {"status": "NOT RUN YET", "reason": str(exc)}
        finally:
            wrapper.set_device(torch.device("cpu"))
    else:
        latency["gpu"] = {"status": "NOT RUN YET", "reason": "No CUDA device in this environment."}

    metrics = {
        "model": "distilbert",
        "model_name": "distilbert-base-uncased",
        "seed": int(seed),
        "status": "ok",
        "config": {**config, "max_length": int(max_length),
                   "weight_decay": float(weight_decay), "patience": int(patience)},
        "test_metrics": trained["test_metrics"],
        "metrics_at_val_threshold": threshold_view.get("metrics_at_val_threshold"),
        "threshold_val": {k: v for k, v in threshold_view.items()
                          if k != "metrics_at_val_threshold"},
        "best_epoch": trained["best_epoch"],
        "epochs_run": trained["epochs_run"],
        "best_val_f1": trained["best_val_f1"],
        "best_val_metrics": trained["best_val_metrics"],
        "floor_met": trained["floor_met"],
        "train_seconds": trained["train_seconds"],
        "n_params": trained["n_params"],
        "size_mb": trained["size_mb"],
        "checkpoint": trained["checkpoint"],
        "checkpoint_format": trained["checkpoint_format"],
        "peak_memory_mb": trained["peak_memory_mb"],
        "latency_ms_per_email": latency,
        "splits": {"train": len(balanced), "val": len(val_texts), "test": len(test_texts)},
        "protocol": (
            "Test evaluated exactly once per seed, inside train_distilbert on the "
            "frozen best-validation state (precision-first best epoch: validation "
            "precision subject to validation recall >= 0.85). Threshold 0.5 for the "
            "headline metrics; the validation-fitted threshold (target recall 0.90) "
            "is an additional view, fit on VALIDATION only."
        ),
    }
    save_json(metrics, seed_dir / "metrics.json")
    np.save(seed_dir / "test_pred.npy", np.asarray(trained["y_pred"]))
    np.save(seed_dir / "test_prob.npy", test_prob)
    logger.info(f"Seed {seed} done: test P={metrics['test_metrics']['precision']:.4f} "
                f"R={metrics['test_metrics']['recall']:.4f} "
                f"F1={metrics['test_metrics']['f1']:.4f} "
                f"(best epoch {metrics['best_epoch']}) -> {seed_dir / 'metrics.json'}")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DistilBERT final training: seeds 42/1/7 with the best tuning config; test evaluated once per seed."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                        help="Seeds (default 42 1 7)")
    parser.add_argument("--config", type=str, default="results/distilbert/tuning/results.json",
                        help="Tuning results.json (best_params) or best_params.json; explicit --batch_size/--lr/--epochs override it")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="lr")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--output_dir", type=str, default="results/distilbert",
                        help="Output root (auto-redirected under /kaggle/working on Kaggle)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each seed from its latest_checkpoint.pt if present")
    parser.add_argument("--debug", action="store_true",
                        help="Small subset (200 train / 100 val / 100 test), 1 epoch, 1 seed — smoke test only")
    args = parser.parse_args(argv)

    set_seed(args.seeds[0])
    if args.debug:
        args.seeds = args.seeds[:1]
        args.epochs = 1

    explicit = {k: v for k, v in
                (("batch_size", args.batch_size), ("lr", args.lr), ("epochs", args.epochs))
                if v is not None}
    if explicit:
        config = load_best_config(Path(args.config)) if Path(args.config).exists() else {}
        config.update(explicit)
        missing = [k for k in ("batch_size", "lr", "epochs") if k not in config]
        if missing:
            print(f"ERROR: config incomplete, missing {missing}: pass --config or all of "
                  f"--batch_size/--lr/--epochs.", file=sys.stderr)
            return 2
    else:
        config = load_best_config(Path(args.config))
    print(f"Final DistilBERT config: {config} (seeds {args.seeds})")

    splits = load_splits(args.data_path, args.splits_path, seed=args.seeds[0])
    if args.debug:
        from phishing_xai.baselines import subsample_splits

        splits = subsample_splits(splits, n_train=200, n_val=100, n_test=100, seed=args.seeds[0])
    train_df = splits["train"]
    val_texts = splits["val"]["clean_text"].astype(str).tolist()
    val_labels = splits["val"]["label"].astype(int).tolist()
    test_texts = splits["test"]["clean_text"].astype(str).tolist()
    test_labels = splits["test"]["label"].astype(int).tolist()

    out_root = resolve_output_dir(args.output_dir)
    ensure_dir(out_root)

    per_seed: List[Dict[str, Any]] = []
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        try:
            prev = load_json(summary_path)
            per_seed = [m for m in prev.get("per_seed", []) if m.get("seed") not in args.seeds]
        except Exception:  # noqa: BLE001 - corrupt previous summary: start fresh
            per_seed = []
    for seed in args.seeds:
        metrics = run_seed(
            seed, train_df, val_texts, val_labels, test_texts, test_labels,
            config, out_root,
            max_length=int(args.max_length), weight_decay=float(args.weight_decay),
            patience=int(args.patience), resume=args.resume,
        )
        per_seed.append(metrics)
        # Summary updated after EACH seed (a preempted kernel keeps it).
        ok = [m for m in per_seed if m.get("status") == "ok" and m.get("test_metrics")]
        agg = aggregate_seed_results([
            {"seed": m["seed"], "metrics": m["test_metrics"],
             "train_seconds": m["train_seconds"], "n_params": m["n_params"],
             "size_mb": m["size_mb"],
             "latency_cpu_ms": (m.get("latency_ms_per_email") or {}).get("cpu", {}),
             "latency_gpu_ms": (m.get("latency_ms_per_email") or {}).get("gpu", {})}
            for m in ok
        ]) if ok else {"status": "NOT RUN YET"}
        best_val_seed = None
        if ok:
            best_val_seed = max(ok, key=lambda m: (m.get("best_val_f1", -1.0), -m["seed"]))["seed"]
        save_json({
            "model": "distilbert",
            "model_name": "distilbert-base-uncased",
            "config": config,
            "seeds": [m["seed"] for m in per_seed],
            "best_validation_seed": best_val_seed,
            "per_seed": per_seed,
            "aggregate": agg,
        }, summary_path)
        print(f"Seed {seed} saved; summary updated ({summary_path}).")

    ok = [m for m in per_seed if m.get("status") == "ok"]
    print(f"DistilBERT final training done: {len(ok)}/{len(args.seeds)} seeds ok. "
          f"Best-validation seed: "
          f"{max(ok, key=lambda m: (m.get('best_val_f1', -1.0), -m['seed']))['seed'] if ok else None}.")
    return 0 if len(ok) == len(args.seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
