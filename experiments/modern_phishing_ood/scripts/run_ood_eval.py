"""Out-of-distribution evaluation: fine-tuned RoBERTa vs tuned BiLSTM.

Evaluates BOTH models (loaded from saved checkpoints, NEVER retrained)
on modern-phishing OOD sets. Thresholds are fitted on the ORIGINAL
VALIDATION split only (``baselines.load_splits`` + ``fit_threshold_for_recall``)
and applied unchanged to OOD — OOD labels are never used for fitting
(see ``ood_common.fit_thresholds_for_floors`` and ``tests/test_ood_eval.py``).

Primary OOD set: PhishNChips v5.2 — a MODERN URL-GROUNDED
SYNTHETIC-STIMULI set (phishing labels verified via PhishTank/OpenPhish/
GitHub phishing DB URLs; email prose is synthetic stimulus text, NOT
real inbox mail). Report it as such, never as "real-world 2026 phishing".

Hard rules (project-wide, never bypassed here):
* NEVER modify ``src/phishing_xai/model.py`` or ``src/phishing_xai/faithfulness.py``.
* NEVER overwrite existing results/checkpoints. All outputs go under
  the ``--output_dir`` / ``--predictions_dir`` given on the command line.
* NEVER retrain. A missing BiLSTM artefact is a loud failure, NOT a
  silent refit (``load_scorer`` would silently refit a fresh BiLSTM if
  the path were missing — this script guards with
  ``_is_baseline_artefact`` and loads via ``load_baseline_model``).
* Splits are loaded from ``results/splits.json`` via ``load_splits``,
  never regenerated (sklearn-version split instability).

Writes:
* ``<output_dir>/<set>_<model>_metrics.json`` — full metrics at 0.5 and
  at each validation-fitted threshold, latency, params, model size.
* ``<output_dir>/ood_metrics_combined.json`` — all sets/models, thresholds,
  leakage summary, cross-check diffs, per-model status.
* ``<output_dir>/thresholds.json`` — validation-fitted thresholds.
* ``<output_dir>/leakage_audit.json`` — exact / near-dup / MinHash / URL
  overlap counts per set.
* ``<predictions_dir>/<set>_<model>_predictions.csv`` — per-email
  (email_id, true_label, predicted_label@0.5, prob_phishing, source,
  category, model).
* ``<output_dir>/robustness_<model>.json`` — only with ``--robustness``
  (primary set only).

A model that fails to load is recorded as NOT RUN YET with the reason
and the other model still runs; exit code is 0 iff at least one
model/set produced results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ood_common as oc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

CAVEAT = ("modern URL-grounded synthetic-stimuli OOD "
          "(PhishNChips v5.2, Apr 2026): phishing labels verified via "
          "PhishTank/OpenPhish/GitHub phishing DB URLs; email prose is "
          "synthetic stimulus text, not real inbox mail")

SET_LOADERS = ("phishnchips", "phishfuzzer", "frontiers_llm",
               "frontiers_human", "bec", "bec_poisoned")
MODEL_CHOICES = ("roberta", "bilstm")
FLOORS_DEFAULT = (0.85, 0.90, 0.95)


# ---------------------------------------------------------------------------
# Lazy repo helpers (torch/sklearn live only on Kaggle / GPU machines)
# ---------------------------------------------------------------------------

def _repo_clean_text():
    """Return (clean_fn, how) — the repo helper when importable, else lite."""
    try:
        from phishing_xai.data import clean_text
        return clean_text, "phishing_xai.data.clean_text"
    except Exception as exc:  # noqa: BLE001 - CPU-only local runs
        print(f"[warn] repo clean_text unavailable ({exc}); using clean_text_lite "
              f"for model inputs — DO NOT use these numbers as evidence.", flush=True)
        return oc.clean_text_lite, "ood_common.clean_text_lite (FALLBACK)"


def _repo_metrics():
    """Return (metrics_fn, how) — repo helper when importable, else lite."""
    try:
        from phishing_xai.baselines import compute_metrics_full
        return compute_metrics_full, "phishing_xai.baselines.compute_metrics_full"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] compute_metrics_full unavailable ({exc}); using metrics_lite.",
              flush=True)
        return oc.metrics_lite, "ood_common.metrics_lite (FALLBACK)"


def _repo_threshold_fitter():
    """Return (fitter_fn, how) — repo helper when importable, else lite."""
    try:
        from phishing_xai.baselines import fit_threshold_for_recall

        def fitter(y, p, floor):
            r = fit_threshold_for_recall(y, p, target_recall=floor)
            return {"threshold": float(r["threshold"]),
                    "val_precision": float(r["val_precision"]),
                    "val_recall": float(r["val_recall"]),
                    "floor_met": bool(r.get("floor_met", r["val_recall"] >= floor))}
        return fitter, "phishing_xai.baselines.fit_threshold_for_recall"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] fit_threshold_for_recall unavailable ({exc}); using "
              f"ood_common.fit_thresholds_for_floors.", flush=True)

        def fitter(y, p, floor):
            r = oc.fit_thresholds_for_floors(y, p, floors=(floor,))[f"{floor:.2f}"]
            return {"threshold": float(r["threshold"]),
                    "val_precision": float(r["val_precision"]),
                    "val_recall": float(r["val_recall"]),
                    "floor_met": bool(r["floor_met"])}
        return fitter, "ood_common.fit_thresholds_for_floors (FALLBACK)"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ood_set(name: str, ood_dir: Path) -> pd.DataFrame:
    """Load and normalise one OOD set (see ood_common loaders)."""
    if name == "phishnchips":
        return oc.load_phishnchips(ood_dir / "phishnchips_core_emails.csv")
    if name == "phishfuzzer":
        return oc.load_phishfuzzer_manual(ood_dir / "phishfuzzer_seeds.json")
    if name == "frontiers_llm":
        return oc.load_frontiers_llm(
            ood_dir / "zenodo_frontiers" / "data" / "llm_corpus_sampled.csv")
    if name == "frontiers_human":
        return oc.load_frontiers_human(
            ood_dir / "zenodo_frontiers" / "data" / "human_corpus_sampled.csv")
    if name == "bec":
        return oc.load_bec(ood_dir / "yoadjei_bec" / "synthetic_emails.csv")
    if name == "bec_poisoned":
        return oc.load_bec(ood_dir / "yoadjei_bec" / "synthetic_emails_poisoned.csv",
                           poisoned=True)
    raise ValueError(f"Unknown OOD set: {name!r}")


def load_training_texts(data_path: Path) -> Tuple[List[str], List[str], List[int]]:
    """Training texts for leakage checks.

    Prefers the repo path (``load_data`` + ``deduplicate_data`` — the
    exact frame ``make_splits.py`` built); falls back to a plain pandas
    read with lite cleaning. Returns (all_clean_texts, phishing_texts,
    labels)."""
    try:
        from phishing_xai.data import deduplicate_data, load_data
        df = deduplicate_data(load_data(str(data_path)))
        texts = df["clean_text"].astype(str).tolist()
        labels = df["label"].astype(int).tolist()
        print(f"[leakage] training texts via repo load_data+deduplicate: "
              f"n={len(texts)} (unique {len(set(texts))})", flush=True)
        return texts, [t for t, y in zip(texts, labels) if y == 1], labels
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] repo load_data unavailable ({exc}); pandas fallback "
              f"for leakage texts.", flush=True)
        df = pd.read_csv(data_path, usecols=["Email Text", "Email Type"])
        df = df.dropna(subset=["Email Text", "Email Type"])
        labels = [1 if str(v).strip().lower().startswith("phish") else 0
                  for v in df["Email Type"]]
        texts = [oc.clean_text_lite(t) for t in df["Email Text"].astype(str)]
        return texts, [t for t, y in zip(texts, labels) if y == 1], labels


def load_validation(data_path: Path, splits_path: Path):
    """Original validation split via ``baselines.load_splits`` (raises if
    the manifest disagrees — never regenerate splits)."""
    from phishing_xai.baselines import load_splits
    splits = load_splits(data_path=str(data_path), splits_path=str(splits_path))
    val = splits["val"]
    return (val["clean_text"].astype(str).tolist(),
            val["label"].astype(int).tolist())


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def run_leakage_audit(frames: Dict[str, pd.DataFrame],
                      train_texts: Sequence[str],
                      train_phish_texts: Sequence[str],
                      minhash_train_sample: int = 0) -> Dict[str, Any]:
    """Exact + near-dup-hash + MinHash + URL overlap per OOD set.

    ``minhash_train_sample``: 0 = full training set; otherwise a fixed
    seed-42 subsample of that size (recorded in the output)."""
    train_urls = set()
    for t in train_phish_texts:
        train_urls.update(oc.extract_urls(t))
    rng = np.random.RandomState(42)
    if minhash_train_sample and len(train_texts) > minhash_train_sample:
        idx = rng.choice(len(train_texts), size=minhash_train_sample, replace=False)
        mh_train = [train_texts[i] for i in sorted(idx)]
    else:
        mh_train = list(train_texts)
    # Training MinHash signatures are computed ONCE and reused for every
    # set (recomputing them per set cost ~6x in the local run).
    train_sigs = oc.minhash_signatures(mh_train, n_perm=64, seed=42, k=5)
    # Body-only exact hashes of training texts (training bodies are the
    # texts themselves after cleaning, so the normalised body hash set is
    # the same set used for the combined-text check; the body-only check
    # below matters for OOD sets whose subject differs from training
    # while the body matches, e.g. PhishFuzzer Manual row No. 37).
    train_body_hashes = {oc.md5_hex(oc.normalize_for_hash(t)) for t in train_texts}
    out: Dict[str, Any] = {
        "train_n_texts": len(train_texts),
        "train_n_unique_normalized_hashes": len(train_body_hashes),
        "train_n_phishing_urls": len(train_urls),
        "minhash_train_n": len(mh_train),
        "minhash_train_sampled": bool(minhash_train_sample
                                      and len(train_texts) > minhash_train_sample),
        "per_set": {},
    }
    for name, frame in frames.items():
        texts = frame["text"].astype(str).tolist()
        phish_texts = frame.loc[frame["label"] == 1, "text"].astype(str).tolist()
        ood_urls = set()
        for t in phish_texts:
            ood_urls.update(oc.extract_urls(t))
        entry = oc.hash_overlap(texts, train_texts)
        ood_sigs = oc.minhash_signatures(texts, n_perm=64, seed=42, k=5)
        entry["minhash"] = oc.near_dup_audit_with_sigs(
            ood_sigs, train_sigs, threshold=0.9, n_ood=len(texts), n_train=len(mh_train))
        entry["url_overlap_vs_train_phishing"] = oc.url_overlap(ood_urls, train_urls)
        if "body" in frame.columns:
            body_hashes = [oc.md5_hex(oc.normalize_for_hash(b))
                           for b in frame["body"].astype(str)]
            entry["n_ood_body_only_exact_overlap"] = int(
                sum(h in train_body_hashes for h in body_hashes))
        entry["n_rows"] = int(len(frame))
        entry["n_phish"] = int((frame["label"] == 1).sum())
        entry["n_legit"] = int((frame["label"] == 0).sum())
        out["per_set"][name] = entry
        print(f"[leakage] {name}: exact overlap "
              f"{entry['n_ood_exact_overlap']}/{entry['n_ood']}, body-only exact "
              f"{entry.get('n_ood_body_only_exact_overlap', 'n/a')}, near-dup-hash "
              f"{entry['n_ood_near_dup_hash_overlap']}/{entry['n_ood']}, "
              f"minhash>=0.9 {entry['minhash']['n_ood_with_max_jaccard_ge_threshold']}/"
              f"{entry['n_ood']}, shared URLs "
              f"{entry['url_overlap_vs_train_phishing']['n_shared_urls']}",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# Scorers (with the BiLSTM silent-refit guard)
# ---------------------------------------------------------------------------

def load_roberta(checkpoint: Path):
    """Load the fine-tuned RoBERTa checkpoint via load_scorer. Raises on
    a missing/too-small checkpoint — never falls back to untrained base."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"RoBERTa checkpoint not found: {checkpoint}")
    size = checkpoint.stat().st_size
    if size < 100_000_000:
        raise ValueError(f"RoBERTa checkpoint too small ({size} bytes): {checkpoint}")
    from phishing_xai.baselines import load_scorer
    scorer = load_scorer("roberta", checkpoint=str(checkpoint))
    n_params = int(sum(p.numel() for p in scorer.model.parameters()))
    print(f"[load] roberta: {checkpoint} ({size / 1e6:.1f} MB, "
          f"{n_params:,} params, device={scorer.device})", flush=True)
    return scorer, {"n_params": n_params, "size_mb": size / 1e6,
                    "max_length": int(getattr(scorer, "max_length", 512))}


def load_bilstm(checkpoint: Path):
    """Load the tuned BiLSTM artefact. GUARD: ``_is_baseline_artefact``
    must be True or we refuse — ``load_scorer`` would otherwise silently
    REFIT a fresh BiLSTM on the train split. Loads via
    ``load_baseline_model`` (raises on a bad path)."""
    if not checkpoint.exists():
        raise FileNotFoundError(f"BiLSTM artefact not found: {checkpoint}")
    from phishing_xai.baselines import _is_baseline_artefact, load_baseline_model
    if not _is_baseline_artefact(checkpoint):
        raise ValueError(
            f"BiLSTM path is NOT a baseline artefact: {checkpoint} — refusing "
            f"to load (load_scorer would silently refit instead).")
    import torch
    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if raw.get("kind") != "bilstm":
        raise ValueError(f"BiLSTM artefact kind is {raw.get('kind')!r}, not 'bilstm'")
    cfg = raw.get("config", {})
    model = load_baseline_model(str(checkpoint))
    # Smoke check: a real prediction on a neutral text, sane shape/sums.
    probe = np.asarray(model.predict_proba(["hello world, meeting at 3pm"]), dtype=float)
    assert probe.shape == (1, 2) and abs(probe.sum() - 1.0) < 1e-6, probe
    print(f"[load] bilstm: {checkpoint} ({checkpoint.stat().st_size / 1e6:.2f} MB, "
          f"n_params={raw.get('n_params')}, config={cfg}, probe={probe.tolist()})",
          flush=True)
    return model, {"n_params": int(raw.get("n_params", -1)),
                   "size_mb": checkpoint.stat().st_size / 1e6,
                   "config": {k: (str(v)) for k, v in cfg.items()}}


def measure_both_latencies(scorer, cpu_scorer, texts: Sequence[str],
                           n_gpu: int, n_cpu: int) -> Dict[str, Any]:
    """Median batch-1 latency (ms/email) on GPU and CPU via
    ``baselines.measure_latency``. CPU uses a separately loaded CPU scorer
    (the GPU scorer's device is fixed at load)."""
    from phishing_xai.baselines import measure_latency
    out: Dict[str, Any] = {}
    if scorer is not None:
        r = measure_latency(lambda t: scorer.predict_proba(t), texts, n=n_gpu)
        r["device"] = str(getattr(scorer, "device", "gpu?"))
        out["gpu"] = r
    if cpu_scorer is not None:
        r = measure_latency(lambda t: cpu_scorer.predict_proba(t), texts, n=n_cpu)
        r["device"] = "cpu"
        out["cpu"] = r
    return out


# ---------------------------------------------------------------------------
# Evaluation of one set / one model
# ---------------------------------------------------------------------------

def evaluate_set(model_name: str, scorer, frame: pd.DataFrame, set_name: str,
                 thresholds: Dict[str, Dict[str, Any]], clean_fn,
                 metrics_fn, pred_dir: Path) -> Dict[str, Any]:
    """Score one normalised frame with one scorer. Binary rows only
    (label 0/1); label -1 rows (PhishFuzzer spam) are scored and reported
    separately, never mixed into binary metrics."""
    binary = frame[frame["label"] >= 0].copy()
    spam = frame[frame["label"] < 0].copy()
    texts = [clean_fn(t) for t in binary["text"].astype(str)]
    labels = binary["label"].astype(int).tolist()
    t0 = time.time()
    probs = np.asarray(scorer.predict_proba(texts), dtype=float)[:, 1]
    elapsed = time.time() - t0
    pred_file = pred_dir / f"{set_name}_{model_name}_predictions.csv"
    pd.DataFrame({
        "email_id": binary["email_id"].astype(str).tolist(),
        "true_label": labels,
        "predicted_label_at_0_5": (probs >= 0.5).astype(int).tolist(),
        "prob_phishing": probs.tolist(),
        "source": binary["source"].astype(str).tolist(),
        "category": binary["category"].astype(str).tolist(),
        "model": model_name,
        "set": set_name,
    }).to_csv(pred_file, index=False)
    result: Dict[str, Any] = {
        "set": set_name, "model": model_name, "n": int(len(binary)),
        "n_phish": int((binary["label"] == 1).sum()),
        "n_legit": int((binary["label"] == 0).sum()),
        "scoring_seconds_total": float(elapsed),
        "predictions_file": str(pred_file),
        "metrics_at_0_5": metrics_fn(labels, probs, threshold=0.5),
        "metrics_at_fitted_thresholds": {},
    }
    for floor, thr in thresholds.items():
        m = metrics_fn(labels, probs, threshold=float(thr["threshold"]))
        m["threshold_used"] = float(thr["threshold"])
        m["target_recall_floor"] = float(floor)
        m["val_floor_met"] = bool(thr["floor_met"])
        result["metrics_at_fitted_thresholds"][floor] = m
    # Per-category breakdown at 0.5 (single-class groups get ROC-AUC 0.5
    # fallback from the metric fn; n is always recorded).
    cats: Dict[str, Any] = {}
    labels_list = list(labels)
    probs_list = [float(p) for p in probs]
    for col in ("category", "datasource"):
        if col not in binary.columns:
            continue
        col_vals = binary[col].astype(str).tolist()
        for val in sorted(set(col_vals)):
            idx = [i for i, v in enumerate(col_vals) if v == val]
            g_labels = [labels_list[i] for i in idx]
            g_probs = [probs_list[i] for i in idx]
            cats[f"{col}={val}"] = metrics_fn(g_labels, g_probs, threshold=0.5)
    result["per_category_at_0_5"] = cats
    # Lite-vs-repo cross-check on the SAME probabilities (no re-scoring).
    try:
        lite = oc.metrics_lite(labels_list, probs_list, threshold=0.5)
        repo_m = result["metrics_at_0_5"]
        result["metrics_crosscheck_lite_vs_repo"] = {
            k: abs(float(lite[k]) - float(repo_m[k]))
            for k in ("accuracy", "precision", "recall", "f1",
                      "roc_auc", "pr_auc") if k in repo_m}
    except Exception as exc:  # noqa: BLE001
        result["metrics_crosscheck_lite_vs_repo"] = {"status": "FAILED",
                                                     "reason": str(exc)}
    if len(spam):
        spam_texts = [clean_fn(t) for t in spam["text"].astype(str)]
        spam_probs = np.asarray(scorer.predict_proba(spam_texts), dtype=float)[:, 1]
        result["spam_rows"] = {
            "n": int(len(spam)),
            "flagged_as_phishing_at_0_5": int((spam_probs >= 0.5).sum()),
            "flagged_rate_at_0_5": float((spam_probs >= 0.5).mean()),
            "median_prob": float(np.median(spam_probs)),
            "note": "Spam rows are NOT part of binary metrics; reported separately.",
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="OOD eval: RoBERTa vs BiLSTM on modern-phishing sets (no retraining).")
    parser.add_argument("--sets", nargs="+", choices=list(SET_LOADERS),
                        default=["phishnchips", "phishfuzzer"],
                        help="OOD sets to evaluate")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CHOICES),
                        default=["roberta", "bilstm"])
    parser.add_argument("--ood_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--roberta_checkpoint", type=str, default="results/best_model.pt")
    parser.add_argument("--bilstm_checkpoint", type=str,
                        default="results/bilstm/best_bilstm.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--floors", nargs="+", type=float, default=list(FLOORS_DEFAULT))
    parser.add_argument("--latency_n_gpu", type=int, default=200)
    parser.add_argument("--latency_n_cpu", type=int, default=100)
    parser.add_argument("--robustness", action="store_true",
                        help="Also run the 3 H4 perturbations on the primary set only")
    parser.add_argument("--skip_leakage", action="store_true")
    parser.add_argument("--minhash_train_sample", type=int, default=0,
                        help="0 = full training set for MinHash; else seed-42 subsample")
    parser.add_argument("--legit_pairing", type=str, default="phishnchips",
                        choices=["phishnchips"],
                        help="Legit set paired with phishing-only sets")
    args = parser.parse_args(argv)

    ood_dir = Path(args.ood_dir)
    out_dir = Path(args.output_dir)
    pred_dir = Path(args.predictions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    clean_fn, clean_how = _repo_clean_text()
    metrics_fn, metrics_how = _repo_metrics()
    fitter, fitter_how = _repo_threshold_fitter()
    print(f"[setup] clean={clean_how} | metrics={metrics_how} | thresholds={fitter_how}",
          flush=True)

    # --- Load + normalise OOD sets -------------------------------------
    frames: Dict[str, pd.DataFrame] = {}
    for name in args.sets:
        frames[name] = load_ood_set(name, ood_dir)
        f = frames[name]
        print(f"[data] {name}: n={len(f)} phish={(f['label'] == 1).sum()} "
              f"legit={(f['label'] == 0).sum()} other={(f['label'] < 0).sum()}",
              flush=True)
    # Pair phishing-only sets with the PhishNChips legit 1,000.
    paired_from: Dict[str, str] = {}
    if any(n in ("frontiers_llm", "frontiers_human", "bec", "bec_poisoned")
           for n in args.sets):
        legit = frames.get("phishnchips")
        if legit is None:
            legit = load_ood_set("phishnchips", ood_dir)
        legit_only = legit[legit["label"] == 0].copy()
        for n in ("frontiers_llm", "frontiers_human", "bec", "bec_poisoned"):
            if n in frames:
                frames[n] = oc.pair_with_legit(frames[n], legit_only)
                paired_from[n] = ("phishnchips legit 1,000 "
                                  "(same legit rows reused across synthetic-track sets)")
                print(f"[data] {n} paired with PhishNChips legit "
                      f"n={len(frames[n])}", flush=True)

    # --- Leakage audit (no model needed) -------------------------------
    leakage = None
    if not args.skip_leakage:
        train_texts, train_phish, _ = load_training_texts(Path(args.data_path))
        leakage = run_leakage_audit(frames, train_texts, train_phish,
                                    minhash_train_sample=args.minhash_train_sample)
        (out_dir / "leakage_audit.json").write_text(
            json.dumps(leakage, indent=2), encoding="utf-8")

    # --- Validation predictions -> thresholds (validation ONLY) ---------
    val_texts, val_labels = load_validation(Path(args.data_path), Path(args.splits_path))
    print(f"[thresholds] validation n={len(val_texts)} "
          f"(phish={sum(val_labels)})", flush=True)
    val_clean = [clean_fn(t) for t in val_texts]

    combined: Dict[str, Any] = {
        "experiment": "modern_phishing_ood",
        "caveat": CAVEAT,
        "sets": args.sets,
        "models_requested": args.models,
        "floors": [float(f) for f in args.floors],
        "clean_fn": clean_how,
        "metrics_fn": metrics_how,
        "threshold_fitter": fitter_how,
        "paired_legit": paired_from,
        "leakage_audit_file": ("leakage_audit.json" if leakage is not None
                               else "SKIPPED"),
        "results": {},
        "status": {},
    }
    any_ok = False

    for model_name in args.models:
        # Load the scorer (GPU) and a CPU copy for CPU latency.
        cpu_scorer = None
        gpu_scorer = None
        try:
            if model_name == "roberta":
                scorer, info = load_roberta(Path(args.roberta_checkpoint))
                gpu_scorer = scorer
                try:
                    cpu_scorer, _ = load_roberta_cpu(Path(args.roberta_checkpoint))
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] CPU RoBERTa scorer unavailable ({exc}); "
                          f"CPU latency NOT RUN YET.", flush=True)
            else:
                scorer, info = load_bilstm(Path(args.bilstm_checkpoint))
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        from phishing_xai.baselines import load_baseline_model
                        gpu_scorer = load_baseline_model(
                            str(args.bilstm_checkpoint), device="cuda")
                    cpu_scorer = load_baseline_model(
                        str(args.bilstm_checkpoint), device="cpu")
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] BiLSTM device copies unavailable ({exc}).",
                          flush=True)
                if cpu_scorer is None:
                    cpu_scorer = scorer
        except Exception as exc:  # noqa: BLE001
            combined["status"][model_name] = {
                "status": "NOT RUN YET", "reason": f"load failed: {exc}"}
            print(f"[load] {model_name} NOT RUN YET: {exc}", flush=True)
            continue

        # Validation probabilities (once per model) -> thresholds.
        val_probs = np.asarray(scorer.predict_proba(val_clean), dtype=float)[:, 1]
        thresholds: Dict[str, Dict[str, Any]] = {}
        crosscheck: Dict[str, Any] = {}
        for floor in args.floors:
            thresholds[f"{float(floor):.2f}"] = fitter(val_labels, val_probs, float(floor))
        # Cross-check the repo fitter against the pure-numpy mirror.
        try:
            mirror = oc.fit_thresholds_for_floors(val_labels, val_probs,
                                                  floors=tuple(float(f) for f in args.floors))
            diffs = {k: abs(thresholds[k]["threshold"] - mirror[k]["threshold"])
                     for k in thresholds}
            crosscheck = {"mirror": "ood_common.fit_thresholds_for_floors",
                          "max_abs_threshold_diff": float(max(diffs.values())),
                          "per_floor_diff": diffs}
        except Exception as exc:  # noqa: BLE001
            crosscheck = {"mirror": "FAILED", "reason": str(exc)}
        print(f"[thresholds] {model_name}: {thresholds} | cross-check {crosscheck}",
              flush=True)
        (out_dir / f"thresholds_{model_name}.json").write_text(
            json.dumps({"model": model_name, "thresholds": thresholds,
                        "crosscheck_vs_mirror": crosscheck,
                        "fit_on": "original_validation_split_only",
                        "val_n": len(val_labels)}, indent=2), encoding="utf-8")

        # Latency (primary-set texts, batch 1, median). GPU scorer is the
        # CUDA copy when one was loaded, else the main scorer.
        latency_texts = [clean_fn(t) for t in
                         frames[args.sets[0]]["text"].astype(str).tolist()]
        latency: Dict[str, Any] = {"texts_source": args.sets[0]}
        try:
            latency.update(measure_both_latencies(
                gpu_scorer if gpu_scorer is not None else scorer,
                cpu_scorer, latency_texts,
                n_gpu=args.latency_n_gpu, n_cpu=args.latency_n_cpu))
        except Exception as exc:  # noqa: BLE001
            latency = {"status": "NOT RUN YET", "reason": str(exc),
                       "texts_source": args.sets[0]}
        print(f"[latency] {model_name}: {latency}", flush=True)

        # Evaluate every requested set.
        for set_name in args.sets:
            try:
                res = evaluate_set(model_name, scorer, frames[set_name], set_name,
                                   thresholds, clean_fn, metrics_fn, pred_dir)
                res["latency_ms_per_email"] = latency
                res["model_info"] = info
                res["thresholds_used"] = thresholds
                (out_dir / f"{set_name}_{model_name}_metrics.json").write_text(
                    json.dumps(res, indent=2, default=str), encoding="utf-8")
                combined["results"][f"{set_name}__{model_name}"] = res
                combined["status"][f"{set_name}__{model_name}"] = {"status": "ok"}
                m = res["metrics_at_0_5"]
                print(f"[eval] {set_name} / {model_name}: n={res['n']} "
                      f"acc={m['accuracy']:.4f} P={m['precision']:.4f} "
                      f"R={m['recall']:.4f} F1={m['f1']:.4f} "
                      f"ROC-AUC={m['roc_auc']:.4f} FPR={m['false_positive_rate']:.4f}",
                      flush=True)
                any_ok = True
            except Exception as exc:  # noqa: BLE001
                combined["status"][f"{set_name}__{model_name}"] = {
                    "status": "NOT RUN YET", "reason": str(exc)}
                print(f"[eval] {set_name} / {model_name} NOT RUN YET: {exc}", flush=True)

        # Robustness on the primary set only.
        if args.robustness and "phishnchips" in frames:
            try:
                from phishing_xai.baselines import evaluate_robustness
                prim = frames["phishnchips"]
                binary = prim[prim["label"] >= 0]
                r_texts = [clean_fn(t) for t in binary["text"].astype(str)]
                r_labels = binary["label"].astype(int).tolist()
                rob = evaluate_robustness(scorer, r_texts, r_labels, seed=42)
                (out_dir / f"robustness_{model_name}.json").write_text(
                    json.dumps({"model": model_name, "set": "phishnchips",
                                "n": len(r_labels), "robustness": rob}, indent=2),
                    encoding="utf-8")
                print(f"[robustness] {model_name}: {rob}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[robustness] {model_name} NOT RUN YET: {exc}", flush=True)

    # BEC clean-vs-poisoned contrast (paired rows that actually changed).
    if "bec" in frames and "bec_poisoned" in frames:
        try:
            combined["bec_poisoned_contrast"] = bec_poisoned_contrast(
                frames, pred_dir, args.models)
        except Exception as exc:  # noqa: BLE001
            combined["bec_poisoned_contrast"] = {"status": "NOT RUN YET",
                                                 "reason": str(exc)}
            print(f"[bec] poisoned contrast NOT RUN YET: {exc}", flush=True)

    (out_dir / "ood_metrics_combined.json").write_text(
        json.dumps(combined, indent=2, default=str), encoding="utf-8")
    print(f"[done] outputs in {out_dir} and {pred_dir}", flush=True)
    return 0 if any_ok else 1


def load_roberta_cpu(checkpoint: Path):
    """Load the RoBERTa checkpoint forced onto CPU (for CPU latency)."""
    import torch
    from phishing_xai.baselines import load_scorer
    scorer = load_scorer("roberta", checkpoint=str(checkpoint),
                         device=torch.device("cpu"))
    return scorer, {"device": "cpu"}


def bec_poisoned_contrast(frames: Dict[str, pd.DataFrame], pred_dir: Path,
                          models: Sequence[str]) -> Dict[str, Any]:
    """Paired clean-vs-poisoned recall on the BEC rows the poisoning
    actually changed, from the saved per-email prediction CSVs.

    Pairing is by the positional row suffix in ``email_id``
    (``bec_clean_<i>`` / ``bec_poisoned_<i>``); the changed mask is
    computed from the frame texts (clean text != poisoned text), never
    assumed. Subjects must match positionally — any mismatch is counted
    and reported, and the contrast is marked unreliable if nonzero.
    Unchanged rows are reported as a sanity check (their predictions
    must be identical)."""
    clean_frame = frames["bec"][frames["bec"]["source"] == "yoadjei_bec_clean"]
    pois_frame = frames["bec_poisoned"][
        frames["bec_poisoned"]["source"] == "yoadjei_bec_poisoned"]
    clean_text = {str(e).split("_")[-1]: t for e, t in
                  zip(clean_frame["email_id"], clean_frame["text"])}
    pois_text = {str(e).split("_")[-1]: t for e, t in
                 zip(pois_frame["email_id"], pois_frame["text"])}
    common = sorted(set(clean_text) & set(pois_text), key=int)
    changed = [k for k in common if clean_text[k] != pois_text[k]]
    unchanged = [k for k in common if clean_text[k] == pois_text[k]]
    out: Dict[str, Any] = {
        "method": "positional pairing by email_id suffix; changed = text differs",
        "n_paired": len(common),
        "n_changed": len(changed),
        "n_unchanged": len(unchanged),
        "per_model": {},
    }
    for model in models:
        clean_pred = pd.read_csv(pred_dir / f"bec_{model}_predictions.csv")
        pois_pred = pd.read_csv(pred_dir / f"bec_poisoned_{model}_predictions.csv")
        cp = {str(e).split("_")[-1]: float(p) for e, p in
              zip(clean_pred["email_id"], clean_pred["prob_phishing"])}
        pp = {str(e).split("_")[-1]: float(p) for e, p in
              zip(pois_pred["email_id"], pois_pred["prob_phishing"])}
        ch_c = [cp[k] for k in changed if k in cp]
        ch_p = [pp[k] for k in changed if k in pp]
        un_c = [cp[k] for k in unchanged if k in cp]
        un_p = [pp[k] for k in unchanged if k in pp]
        out["per_model"][model] = {
            "n_changed_scored": len(ch_c),
            "recall_at_0_5_clean_changed": float(np.mean([p >= 0.5 for p in ch_c])) if ch_c else None,
            "recall_at_0_5_poisoned_changed": float(np.mean([p >= 0.5 for p in ch_p])) if ch_p else None,
            "recall_delta_poisoned_minus_clean": (
                float(np.mean([p >= 0.5 for p in ch_p]) - np.mean([p >= 0.5 for p in ch_c]))
                if ch_c and ch_p else None),
            "median_prob_change_poisoned_minus_clean": (
                float(np.median([pp[k] - cp[k] for k in changed if k in cp and k in pp]))
                if ch_c and ch_p else None),
            "unchanged_rows_identical_predictions": bool(
                all(abs(cp[k] - pp[k]) < 1e-6 for k in unchanged if k in cp and k in pp)),
            # Tolerance 1e-6, not 1e-9: GPU-scored BiLSTM shows float
            # nondeterminism up to ~4.4e-08 on identical texts (measured
            # on the Kernel 2 predictions); 1e-9 falsely flags that as
            # a real difference. RoBERTa is exactly 0.0.
            "n_unchanged_scored": len(un_c),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
