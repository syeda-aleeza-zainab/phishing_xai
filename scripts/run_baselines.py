"""CLI: train and evaluate the Section H baseline ladder.

Usage examples (from the repo root)::

    python scripts/run_baselines.py --models dummy tfidf_lr tfidf_svm --seeds 42 1 7
    python scripts/run_baselines.py --models bilstm --seeds 42 1 7 --epochs 8 --emb_dim 200 --hidden 128 --layers 2 --max_len 256
    python scripts/run_baselines.py --models cnn distilbert --seeds 42
    python scripts/run_baselines.py --models roberta_eval --checkpoint results/best_model.pt
    python scripts/run_baselines.py --models dummy tfidf_lr --analysis --robustness
    python scripts/run_baselines.py --models tfidf_lr bilstm --faithfulness --sample_size 100
    python scripts/run_baselines.py --models tfidf_lr bilstm --multimodal_eval_dir results/multimodal_eval
    python scripts/run_baselines.py --debug --models dummy tfidf_lr

Fairness contract: every model uses ``results/splits.json`` (same splits and
same ``clean_text`` cleaning as RoBERTa), vocab/TF-IDF/tokenizer fit on TRAIN
only, train-only synonym augmentation via ``augment.py``, hyperparameter
search (~8 trials) scored on VALIDATION only, and a single final test
evaluation per model. RoBERTa is eval-only here and is never retrained.
Anything not executed is recorded as ``"NOT RUN YET"``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Add project root / src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.augment import balance_training_data
from phishing_xai.baselines import (
    BASELINE_MODELS,
    DEFAULT_SEEDS,
    RANDOM_SEARCH_TRIALS,
    aggregate_seed_results,
    bootstrap_f1_ci,
    bootstrap_precision_ci,
    build_dummy,
    build_tfidf_lr,
    build_tfidf_svm,
    build_torch_baseline,
    compute_metrics_full,
    error_analysis,
    evaluate_baseline,
    evaluate_explainability,
    evaluate_multimodal,
    evaluate_on_extractor_output,
    evaluate_roberta_checkpoint,
    evaluate_robustness,
    fit_threshold_for_recall,
    load_multimodal_eval_csv,
    load_roberta_metrics,
    load_splits,
    measure_latency,
    mcnemar_test,
    random_search_tfidf,
    resolve_output_dir,
    save_baseline_model,
    select_deployment_model,
    subsample_splits,
    TorchTrainConfig,
)
from phishing_xai.utils import ensure_dir, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("run_baselines")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the baseline runner."""
    parser = argparse.ArgumentParser(description="Train/evaluate Section H baseline models")
    parser.add_argument("--models", nargs="+", default=["dummy", "tfidf_lr"],
                        choices=BASELINE_MODELS,
                        help="Baseline models to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                        help="Seeds (default 42 1 7)")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory holding splits.json / metrics.json / best_model.pt")
    parser.add_argument("--output_dir", type=str, default="results/baselines",
                        help="Output directory (auto-redirected under /kaggle/working on Kaggle)")
    parser.add_argument("--checkpoint", type=str, default="results/best_model.pt",
                        help="RoBERTa checkpoint for --models roberta_eval")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--max_vocab", type=int, default=30000)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3, help="BiLSTM/CNN learning rate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience (val F1)")
    parser.add_argument("--with_cnn", action="store_true", help="Also run the 1D-CNN baseline")
    parser.add_argument("--with_distilbert", action="store_true",
                        help="Also run the DistilBERT baseline (downloads weights if not cached)")
    parser.add_argument("--distilbert_lr", type=float, default=2e-5,
                        help="DistilBERT learning rate (default 2e-5; was hardcoded before 2026-11-20)")
    parser.add_argument("--random_search_trials", type=int, default=RANDOM_SEARCH_TRIALS,
                        help="Random-search trials on validation for TF-IDF models (0 to disable)")
    parser.add_argument("--analysis", action="store_true", help="Write error analysis CSV")
    parser.add_argument("--robustness", action="store_true", help="Evaluate 3 perturbations")
    parser.add_argument("--faithfulness", action="store_true",
                        help="Evaluate explainability faithfulness (comp/suff @ k in {1,3,5})")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Faithfulness sample size")
    parser.add_argument("--multimodal_eval_dir", type=str, default=None,
                        help="Directory holding a multimodal eval CSV (see --eval_csv)")
    parser.add_argument("--eval_csv", type=str, default=None,
                        help="Multimodal eval CSV with text/label (+ optional body_text/ocr_text/pdf_text/qr_text)")
    parser.add_argument("--ood_csv", type=str, default=None,
                        help="Optional second-dataset CSV (columns: Email Text, Email Type) for OOD eval; never auto-downloaded")
    parser.add_argument("--debug", action="store_true",
                        help="Small subset, CPU, 1 epoch (smoke test)")
    parser.add_argument("--no_merge", action="store_true",
                        help="Do NOT merge with an existing baseline_results.json; "
                             "overwrite it. Default merges per-model results keyed "
                             "by model+seed, so separate invocations accumulate the "
                             "ladder instead of only the most recent run surviving.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for subsampling")
    return parser.parse_args(argv)


def merge_baseline_results(existing: Any, new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a fresh baseline run into a previous ``baseline_results.json``.

    Per-model results are keyed by model+seed: a new per-seed entry replaces
    the old entry for the same model and seed, entries for other seeds (and
    models the new run did not touch) are kept, and the aggregate is
    recomputed from the merged per-seed entries with
    :func:`aggregate_seed_results`. Optional blocks (error analysis,
    faithfulness, multimodal) are taken from the new run when it produced
    them and otherwise kept from the previous run; a real
    ``comparisons_vs_roberta`` block is likewise kept when the new run has
    only the NOT RUN YET placeholder. A missing, unreadable or malformed
    previous file returns ``new`` unchanged. Added in Section G (2026-11-19):
    before this fix every invocation rewrote ``baseline_results.json`` from
    scratch, so ``run_complexity_report.py`` only ever saw the most recent
    run instead of the accumulated ladder.
    """
    import copy

    if not isinstance(existing, dict) or not isinstance(existing.get("models"), dict):
        return new
    merged = copy.deepcopy(new)
    merged_models = merged.setdefault("models", {})
    for name, old_entry in existing["models"].items():
        if name not in merged_models:
            merged_models[name] = copy.deepcopy(old_entry)
            continue
        old_seeds = old_entry.get("per_seed") or []
        new_seeds = merged_models[name].get("per_seed") or []
        by_seed = {r.get("seed"): r for r in old_seeds if isinstance(r, dict)}
        for r in new_seeds:
            if isinstance(r, dict):
                by_seed[r.get("seed")] = r
        merged_seeds = [by_seed[s] for s in sorted(by_seed, key=lambda v: (v is None, v))]
        merged_models[name]["per_seed"] = merged_seeds
        ok = [r for r in merged_seeds if r.get("status") != "NOT RUN YET" and "metrics" in r]
        merged_models[name]["aggregate"] = aggregate_seed_results(ok)
    for block in ("error_analysis", "faithfulness", "multimodal"):
        if block not in merged and block in existing:
            merged[block] = copy.deepcopy(existing[block])
    old_comp = existing.get("comparisons_vs_roberta")
    new_comp = merged.get("comparisons_vs_roberta")
    if (isinstance(old_comp, dict) and old_comp.get("status") != "NOT RUN YET"
            and (not isinstance(new_comp, dict) or new_comp.get("status") == "NOT RUN YET")):
        merged["comparisons_vs_roberta"] = copy.deepcopy(old_comp)
    old_seeds = (existing.get("fairness") or {}).get("seeds") or []
    new_seeds = (merged.get("fairness") or {}).get("seeds") or []
    if old_seeds or new_seeds:
        merged.setdefault("fairness", {})["seeds"] = sorted(set(old_seeds) | set(new_seeds))
    merged["merged_with_previous_run"] = True
    return merged


def _balanced_train(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Apply the train-only synonym-augmentation balancing policy.

    Falls back to the unbalanced train split (with a warning) when WordNet is
    unavailable, e.g. offline; the fallback is recorded in the run output.
    """
    try:
        return balance_training_data(train_df, text_col="clean_text", label_col="label", seed=seed)
    except Exception as exc:
        logger.warning(f"Train augmentation failed ({exc}); using unbalanced train split.")
        return train_df.copy().reset_index(drop=True)


def _clean_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON-serialisable fitted artefacts (keys starting with '_')."""
    return {k: v for k, v in extra.items() if not k.startswith("_")}


def _add_val_threshold(
    result: Dict[str, Any],
    model: Any,
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
) -> None:
    """Add the precision-at-recall-0.90 view, threshold fit on VALIDATION only.

    The threshold is chosen on validation predictions (target recall 0.90),
    documented with its validation precision/recall, then applied unchanged to
    the test set. Test predictions at threshold 0.5 (``result["metrics"]``)
    are untouched; this is an additional view, not a replacement.
    """
    try:
        val_prob = model.predict_proba(val_texts)[:, 1]
        thr = fit_threshold_for_recall(val_labels, val_prob, target_recall=0.90)
        test_prob = np.asarray(result["y_prob"], dtype=float)
        result["threshold_val"] = thr
        result["metrics_at_val_threshold"] = compute_metrics_full(
            test_labels, test_prob, threshold=thr["threshold"]
        )
    except Exception as exc:
        logger.warning(f"Validation-threshold view failed: {exc}")
        result["threshold_val"] = {"status": "NOT RUN YET", "reason": str(exc)}


def _fit_sklearn(
    name: str,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    args: argparse.Namespace,
    seed: int,
):
    """Fit a TF-IDF baseline, optionally after a validation-only random search."""
    params: Dict[str, Any] = {}
    if args.random_search_trials and args.random_search_trials > 0 and name in ("tfidf_lr", "tfidf_svm"):
        search = random_search_tfidf(
            train_texts, train_labels, val_texts, val_labels,
            kind=name, n_trials=args.random_search_trials, seed=seed,
        )
        params = search["best_params"]
        logger.info(f"{name} seed {seed}: random search chose {params} "
                    f"(val precision {search['best_val_precision']:.4f}, val recall "
                    f"{search['best_val_recall']:.4f}, floor_met={search['floor_met']})")
    start = time.perf_counter()
    if name == "dummy":
        model = build_dummy(train_labels)
    elif name == "tfidf_lr":
        model = build_tfidf_lr(train_texts, train_labels, **params)
    else:
        model = build_tfidf_svm(train_texts, train_labels, **params)
    train_seconds = time.perf_counter() - start
    pipeline = model.extra.get("_pipeline")
    return model, pipeline, train_seconds, params


def _fit_distilbert(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    args: argparse.Namespace,
    seed: int,
    out_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Fine-tune DistilBERT via :func:`phishing_xai.baselines.train_distilbert`.

    Downloads ``distilbert-base-uncased`` from the HF hub only when it is not
    already cached; any download failure is recorded as NOT RUN YET by the
    caller. The test set is evaluated EXACTLY ONCE, inside
    ``train_distilbert`` on the frozen best state (precision-first best
    epoch: validation precision subject to validation recall >= 0.85).
    The best state is saved as a STANDALONE artefact —
    ``<out_dir>/distilbert/seed_<seed>/best_distilbert.pt`` plus
    ``tokenizer/`` and ``config.json`` — so the checkpoint loads without
    the Hub (see :func:`phishing_xai.baselines.load_distilbert_model`).
    The returned dict carries the fitted pair as ``"_model_wrapper"``
    (a :class:`BaselineModel`); the main loop pops it before the results
    are serialised, and uses it for the per-seed robustness / OOD /
    validation-threshold / latency views that the other baselines get.
    """
    from phishing_xai.baselines import (
        build_distilbert_wrapper,
        measure_latency,
        train_distilbert,
    )

    model_name = "distilbert-base-uncased"
    run_dir = out_dir / "distilbert" / f"seed_{seed}"
    ensure_dir(run_dir)
    trained = train_distilbert(
        train_texts, train_labels, val_texts, val_labels,
        test_texts=test_texts, test_labels=test_labels,
        model_name=model_name,
        batch_size=args.batch_size, lr=args.distilbert_lr,
        epochs=args.epochs, max_length=args.max_len,
        patience=args.patience, seed=seed, device=device,
        output_dir=run_dir,
    )
    train_seconds = float(trained["train_seconds"])
    model = trained.pop("_model")
    tokenizer = trained.pop("_tokenizer")
    wrapper = build_distilbert_wrapper(
        model, tokenizer, device=device, max_length=args.max_len,
        extra={"model_name": model_name, "best_epoch": trained.get("best_epoch"),
               "best_val_f1": trained.get("best_val_f1")},
    )
    result: Dict[str, Any] = {
        "model": "distilbert", "status": "ok",
        "metrics": trained["test_metrics"],
        "n_params": trained["n_params"],
        "size_mb": trained["size_mb"],
        "train_seconds": train_seconds,
        "y_pred": trained["y_pred"], "y_prob": trained["y_prob"],
        "extra": {"model_name": model_name, "best_epoch": trained.get("best_epoch"),
                  "best_val_f1": trained.get("best_val_f1"),
                  "best_val_metrics": trained.get("best_val_metrics"),
                  "floor_met": trained.get("floor_met"),
                  "checkpoint": trained.get("checkpoint"),
                  "checkpoint_format": trained.get("checkpoint_format")},
        "peak_memory_mb": trained.get("peak_memory_mb"),
        "_model_wrapper": wrapper,
    }
    # CPU latency on the fitted wrapper (median of 200, batch 1), plus GPU
    # latency when CUDA is available — the same views evaluate_baseline
    # records for the other baselines.
    try:
        result["latency_cpu_ms"] = measure_latency(wrapper.predict_proba, test_texts, n=200)
    except Exception as exc:
        logger.warning(f"DistilBERT CPU latency measurement failed: {exc}")
        result["latency_cpu_ms"] = {"latency_ms_per_email_median": None, "n_timed": 0}
    if torch.cuda.is_available():
        try:
            wrapper.set_device(torch.device("cuda"))
            result["latency_gpu_ms"] = measure_latency(wrapper.predict_proba, test_texts, n=200)
        except Exception as exc:
            logger.warning(f"DistilBERT GPU latency measurement failed: {exc}")
        finally:
            wrapper.set_device(torch.device("cpu"))
    return result


def _load_ood(path: str) -> pd.DataFrame:
    """Load an optional second-dataset CSV for OOD evaluation (never downloaded)."""
    from phishing_xai.data import load_data

    df = load_data(path)
    return df[["clean_text", "label"]].reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the requested baselines and write all artefacts."""
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    out_dir = resolve_output_dir(args.output_dir)
    ensure_dir(out_dir)
    logger.info(f"Baseline output directory: {out_dir} (device={device})")

    models = list(args.models)
    if args.with_cnn and "cnn" not in models:
        models.append("cnn")
    if args.with_distilbert and "distilbert" not in models:
        models.append("distilbert")

    if args.debug:
        args.epochs = 1
        args.max_len = 64
        logger.info("Debug mode: 1 epoch, max_len=64, small data subset, CPU.")

    # --- Load identical splits -------------------------------------------------
    splits = load_splits(args.data_path, args.splits_path, seed=args.seed)
    if args.debug:
        splits = subsample_splits(splits, n_train=200, n_val=100, n_test=100, seed=args.seed)

    train_texts = splits["train"]["clean_text"].tolist()
    train_labels = splits["train"]["label"].astype(int).tolist()
    val_texts = splits["val"]["clean_text"].tolist()
    val_labels = splits["val"]["label"].astype(int).tolist()
    test_texts = splits["test"]["clean_text"].tolist()
    test_labels = splits["test"]["label"].astype(int).tolist()
    test_ids = splits["test"]["email_id"].tolist() if "email_id" in splits["test"].columns else list(range(len(test_texts)))

    ood_df: Optional[pd.DataFrame] = None
    if args.ood_csv:
        ood_df = _load_ood(args.ood_csv)
        logger.info(f"OOD dataset loaded: {len(ood_df)} rows from {args.ood_csv}")

    roberta_info = load_roberta_metrics(args.results_dir)
    untrained_fallback = bool(roberta_info.get("untrained_fallback", True))
    logger.info(f"RoBERTa metrics status: {roberta_info.get('status')} (untrained_fallback={untrained_fallback})")

    all_results: Dict[str, Any] = {
        "status": "ok",
        "untrained_fallback": untrained_fallback,
        "roberta": roberta_info,
        "splits": {"train": len(train_texts), "val": len(val_texts), "test": len(test_texts)},
        "fairness": {
            "splits": "results/splits.json (identical to RoBERTa)",
            "cleaning": "phishing_xai.data.clean_text",
            "augmentation": "train-only synonym balancing via phishing_xai.augment.balance_training_data",
            "hyperparameter_search": f"~{args.random_search_trials} random trials scored on VALIDATION only; never on test",
            "seeds": list(args.seeds),
            "primary_metric": "phishing-class PRECISION (user decision 2026-11-19); F1/accuracy/recall reported but not used alone for selection",
            "recall_floor": 0.85,
            "thresholds": "fit on VALIDATION only (target recall 0.90); documented with validation precision/recall in threshold_val",
        },
        "models": {},
    }
    # Keep fitted first-seed models for analysis / faithfulness / multimodal.
    fitted: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Any] = {}
    # Best fitted model per savable kind, by VALIDATION F1 (ties: lowest
    # seed). Saved as best_<model>.pt after the loop (6B.2).
    best_models: Dict[str, Dict[str, Any]] = {}

    for name in models:
        per_seed: List[Dict[str, Any]] = []
        for seed in args.seeds:
            set_seed(seed)
            logger.info(f"=== {name} seed {seed} ===")
            balanced_train = _balanced_train(splits["train"], seed=seed)
            b_train_texts = balanced_train["clean_text"].tolist()
            b_train_labels = balanced_train["label"].astype(int).tolist()
            try:
                current_model = None
                if name == "roberta_eval":
                    result = evaluate_roberta_checkpoint(
                        args.checkpoint, test_texts, test_labels, device=device,
                    )
                    result["seed"] = seed
                    result["train_seconds"] = None
                    per_seed.append(result)
                    continue
                elif name == "distilbert":
                    result = _fit_distilbert(
                        b_train_texts, b_train_labels, val_texts, val_labels,
                        test_texts, test_labels, args, seed, out_dir, device,
                    )
                    result["seed"] = seed
                    # Pop the fitted wrapper before serialisation; it drives
                    # the same per-seed robustness / OOD / val-threshold
                    # views the other baselines get (fall through to the
                    # shared block below). The standalone checkpoint was
                    # already saved per seed inside _fit_distilbert, so
                    # distilbert is deliberately NOT in the best_models /
                    # save_baseline_model tracking below.
                    wrapper = result.pop("_model_wrapper", None)
                    if wrapper is not None:
                        current_model = wrapper
                        _add_val_threshold(result, wrapper, val_texts, val_labels,
                                           test_texts, test_labels)
                        if seed == args.seeds[0]:
                            fitted[name] = wrapper
                elif name in ("dummy", "tfidf_lr", "tfidf_svm"):
                    model, pipeline, train_seconds, params = _fit_sklearn(
                        name, b_train_texts, b_train_labels, val_texts, val_labels, args, seed
                    )
                    result = evaluate_baseline(model, test_texts, test_labels)
                    result["seed"] = seed
                    result["train_seconds"] = train_seconds
                    result["extra"] = {"params": params}
                    current_model = model
                    _add_val_threshold(result, model, val_texts, val_labels, test_texts, test_labels)
                    if seed == args.seeds[0]:
                        fitted[name] = model
                        fitted_pipelines[name] = pipeline
                elif name in ("bilstm", "cnn"):
                    config = TorchTrainConfig(
                        emb_dim=args.emb_dim, hidden=args.hidden, layers=args.layers,
                        dropout=args.dropout, max_len=args.max_len, max_vocab=args.max_vocab,
                        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                        patience=args.patience,
                    )
                    start = time.perf_counter()
                    model, train_info = build_torch_baseline(
                        name, b_train_texts, b_train_labels, val_texts, val_labels,
                        config=config, seed=seed, device=device,
                    )
                    train_seconds = time.perf_counter() - start
                    result = evaluate_baseline(model, test_texts, test_labels)
                    result["seed"] = seed
                    result["train_seconds"] = train_seconds
                    result["extra"] = _clean_extra(model.extra)
                    result["extra"]["train_info"] = train_info
                    current_model = model
                    _add_val_threshold(result, model, val_texts, val_labels, test_texts, test_labels)
                    if torch.cuda.is_available():
                        try:
                            model.set_device(torch.device("cuda"))
                            result["latency_gpu_ms"] = measure_latency(
                                model.predict_proba, test_texts, n=200
                            )
                        except Exception as exc:
                            logger.warning(f"GPU latency measurement failed: {exc}")
                        finally:
                            model.set_device(torch.device("cpu"))
                    if seed == args.seeds[0]:
                        fitted[name] = model
                else:  # pragma: no cover - argparse choices guard this
                    raise ValueError(f"Unknown model: {name}")
            except Exception as exc:
                logger.warning(f"{name} seed {seed} failed: {exc}")
                per_seed.append({
                    "model": name, "seed": seed, "status": "NOT RUN YET",
                    "reason": str(exc), "train_seconds": None,
                })
                continue
            # Robustness + OOD per seed (test split only, fitted model of this seed).
            if args.robustness and result.get("status") != "NOT RUN YET" and current_model is not None:
                result["robustness"] = evaluate_robustness(current_model, test_texts, test_labels, seed=seed)
            if ood_df is not None and current_model is not None and result.get("y_pred") is not None:
                ood_prob = current_model.predict_proba(ood_df["clean_text"].tolist())[:, 1]
                result["ood_metrics"] = compute_metrics_full(ood_df["label"].tolist(), ood_prob)
            # Best-seed tracking for the fitted-artefact save (6B.2): score
            # the FITTED model of this seed on VALIDATION (never test) and
            # keep the best; ties go to the lowest seed. Dummy has no
            # savable artefact and is excluded.
            if current_model is not None and name in ("tfidf_lr", "tfidf_svm", "bilstm", "cnn"):
                val_prob = current_model.predict_proba(val_texts)[:, 1]
                val_f1 = compute_metrics_full(val_labels, val_prob)["f1"]
                prev = best_models.get(name)
                if (prev is None or val_f1 > prev["val_f1"] + 1e-12
                        or (abs(val_f1 - prev["val_f1"]) <= 1e-12 and seed < prev["seed"])):
                    best_models[name] = {"val_f1": float(val_f1), "seed": seed, "model": current_model}
            per_seed.append(result)
            # Save per-seed test predictions for McNemar / error analysis.
            if result.get("y_pred") is not None:
                np.save(out_dir / f"{name}_seed{seed}_test_pred.npy", np.asarray(result["y_pred"]))
                np.save(out_dir / f"{name}_seed{seed}_test_prob.npy", np.asarray(result["y_prob"]))
        agg = aggregate_seed_results([r for r in per_seed if r.get("status") != "NOT RUN YET" and "metrics" in r])
        all_results["models"][name] = {"per_seed": per_seed, "aggregate": agg}

    # --- Save best-seed fitted baselines (6B.2) --------------------------------
    # One file per model in the output dir (e.g. --output_dir results/bilstm
    # gives results/bilstm/best_bilstm.pt), loadable later with
    # load_scorer(..., checkpoint=<path>) in the multimodal scripts. The
    # metrics / baseline_results.json schemas are unchanged; the saved path
    # is only logged. A save failure warns instead of failing the run —
    # the metrics already computed stay the primary output.
    for name, best in best_models.items():
        dest = out_dir / f"best_{name}.pt"
        try:
            save_baseline_model(best["model"], dest)
            logger.info(f"Best {name} model (seed {best['seed']}, val F1 "
                        f"{best['val_f1']:.4f}) saved -> {dest}")
        except Exception as exc:
            logger.warning(f"Could not save best {name} model to {dest}: {exc}")

    # --- Statistical comparison vs RoBERTa (needs RoBERTa test predictions) ---
    if "roberta_eval" in all_results["models"]:
        roberta_seed = all_results["models"]["roberta_eval"]["per_seed"]
        roberta_ok = [r for r in roberta_seed if r.get("status") == "ok" and r.get("y_pred") is not None]
        if roberta_ok:
            roberta_pred = np.asarray(roberta_ok[0]["y_pred"])
            comparisons: Dict[str, Any] = {}
            for name, entry in all_results["models"].items():
                if name == "roberta_eval":
                    continue
                ok = [r for r in entry["per_seed"] if r.get("y_pred") is not None]
                if not ok:
                    continue
                pred = np.asarray(ok[0]["y_pred"])
                mc = mcnemar_test(test_labels, roberta_pred, pred)
                ci = bootstrap_f1_ci(test_labels, pred)
                ci_prec = bootstrap_precision_ci(test_labels, pred)
                comparisons[name] = {"mcnemar_vs_roberta": mc, "bootstrap_f1_ci": ci,
                                     "bootstrap_precision_ci": ci_prec}
            all_results["comparisons_vs_roberta"] = comparisons
    else:
        all_results["comparisons_vs_roberta"] = {
            "status": "NOT RUN YET",
            "reason": "RoBERTa test predictions unavailable (no checkpoint/metrics). Run --models roberta_eval with --checkpoint results/best_model.pt.",
        }

    # --- Error analysis (best baseline vs RoBERTa) ---
    if args.analysis:
        roberta_files = sorted(out_dir.glob("roberta_eval_seed*_test_pred.npy"))
        if roberta_files:
            roberta_pred = np.load(roberta_files[0])
            # Best baseline, precision-first: highest precision subject to the
            # recall floor (0.85); see select_deployment_model.
            agg_map = {
                name: entry.get("aggregate", {})
                for name, entry in all_results["models"].items() if name != "roberta_eval"
            }
            selection = select_deployment_model(agg_map, recall_floor=0.85)
            best_name = selection.get("model")
            if best_name is not None:
                best_pred_file = out_dir / f"{best_name}_seed{args.seeds[0]}_test_pred.npy"
                if best_pred_file.exists():
                    best_pred = np.load(best_pred_file)
                    ea = error_analysis(test_labels, roberta_pred, best_pred, test_texts,
                                        email_ids=test_ids, top_n=20)
                    ea_path = out_dir / "error_analysis.csv"
                    ea.to_csv(ea_path, index=False)
                    logger.info(f"Error analysis ({best_name} vs RoBERTa): {len(ea)} rows -> {ea_path}")
                    all_results["error_analysis"] = {"path": str(ea_path), "best_baseline": best_name,
                                                     "selection": selection, "n_rows": len(ea)}
        if "error_analysis" not in all_results:
            logger.warning("Error analysis skipped: RoBERTa test predictions unavailable (NOT RUN YET).")
            all_results["error_analysis"] = {"status": "NOT RUN YET",
                                             "reason": "RoBERTa test predictions unavailable."}

    # --- Faithfulness ---
    if args.faithfulness:
        faith: Dict[str, Any] = {}
        for name, model in fitted.items():
            if name == "dummy":
                continue
            method = "coefficients" if name == "tfidf_lr" else ("integrated_gradients" if name in ("bilstm", "cnn") else "lime")
            try:
                res = evaluate_explainability(
                    model, test_texts, test_labels, method=method,
                    sample_size=args.sample_size, seed=args.seed,
                    pipeline=fitted_pipelines.get(name),
                )
                faith[name] = res
                logger.info(f"Faithfulness {name} ({method}): {res['comprehensiveness']}")
            except Exception as exc:
                logger.warning(f"Faithfulness failed for {name}: {exc}")
                faith[name] = {"status": "NOT RUN YET", "reason": str(exc)}
        all_results["faithfulness"] = faith

    # --- Multimodal hook (H6) ---
    if args.eval_csv or args.multimodal_eval_dir:
        eval_path = Path(args.eval_csv) if args.eval_csv else Path(args.multimodal_eval_dir) / "multimodal_eval.csv"
        try:
            eval_df = load_multimodal_eval_csv(eval_path)
            mm: Dict[str, Any] = {}
            for name, model in fitted.items():
                rows = eval_df.to_dict("records")
                mm[name] = evaluate_on_extractor_output(model, rows)
            ablation = evaluate_multimodal(fitted, eval_df)
            ablation_path = out_dir / "multimodal_ablation.csv"
            ablation.to_csv(ablation_path, index=False)
            all_results["multimodal"] = {"per_model": mm, "ablation_path": str(ablation_path)}
            logger.info(f"Multimodal ablation -> {ablation_path}")
        except FileNotFoundError as exc:
            logger.warning(f"Multimodal eval set not found ({exc}); Section E set not ready yet. "
                           "Re-run with --eval_csv <path> once run_multimodal_eval.py produces it.")
            all_results["multimodal"] = {
                "status": "NOT RUN YET",
                "reason": str(exc),
                "interface": "evaluate_on_extractor_output(model, rows) in phishing_xai.baselines; "
                             "run: python scripts/run_baselines.py --models tfidf_lr bilstm --eval_csv <path>",
            }

    if not args.no_merge:
        prev_path = out_dir / "baseline_results.json"
        if prev_path.exists():
            try:
                all_results = merge_baseline_results(load_json(prev_path), all_results)
                logger.info("Merged with previous baseline_results.json "
                            "(per-model results keyed by model+seed).")
            except Exception as exc:  # corrupt previous file: keep the new run
                logger.warning(f"Could not merge previous baseline_results.json "
                               f"({exc}); writing the new run only.")
    save_json(all_results, out_dir / "baseline_results.json")
    logger.info(f"Baseline results -> {out_dir / 'baseline_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
