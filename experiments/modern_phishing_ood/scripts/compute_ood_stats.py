"""Statistics + Tables A–F for the modern-phishing OOD experiment.

Reads the SAVED per-email predictions (never re-scores) and the metrics
JSONs written by ``run_ood_eval.py``, and computes:

* paired McNemar (RoBERTa vs BiLSTM) on identical OOD examples —
  ERRORS ONLY, explicitly NOT a precision test;
* paired bootstrap 95% CIs for precision / recall / F1 (1,000 resamples,
  seed 42) and for ROC-AUC / PR-AUC, per model per set;
* Tables A–F (see below), every cell traced to a metrics JSON path or
  marked NOT RUN YET with the reason. No invented cells.

Repo helpers are used when importable (``baselines.mcnemar_test``,
``bootstrap_precision_ci``, ``bootstrap_f1_ci``). On CPU-only machines
those imports fail (``baselines`` imports torch), so this module ships
fallback implementations that mirror the repo formulas exactly
(McNemar chi2 with continuity correction, exact binomial p when b+c<25;
percentile bootstrap, resamples with a single class skipped) and says so
wherever the fallback is used. Bootstrap CIs for recall and for AUCs do
not exist in the repo; they follow the same percentile scheme and are
labelled as extensions.

Tables:
* A — original in-distribution test (``results/RESULTS.md`` values) vs
  modern OOD (PhishNChips), per model.
* B — human-vs-AI (synthetic track only): Frontiers human phishing
  recall vs LLM phishing recall, per model, with FPR on the SHARED
  PhishNChips legit 1,000.
* C — temporal — expected NOT RUN YET (no per-email timestamps exist).
* D — robustness (from ``robustness_<model>.json``).
* E — statistical comparison (McNemar + bootstrap CIs).
* F — efficiency (latency CPU/GPU + params + model size).

Checkpoints are single (one RoBERTa checkpoint, one BiLSTM artefact):
multi-seed variability is NOT claimed; ``n_seeds`` is recorded as 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# In-distribution reference (Kaggle Tesla T4, 2026-11-19, seed 42,
# test n=3,495, threshold 0.5) — from results/RESULTS.md. Quoted, not
# recomputed; the source file is recorded in every Table A cell.
IN_DIST = {
    "roberta": {"precision": 0.9690, "recall": 0.9932, "f1": 0.9809,
                "accuracy": 0.9854, "roc_auc": 0.9986, "n": 3495,
                "source": "results/RESULTS.md (Stage B)"},
    "bilstm": {"precision": 0.9719, "recall": 0.9428, "f1": 0.9571,
               "accuracy": None, "roc_auc": None, "n": 3495,
               "source": "results/RESULTS.md (tuned BiLSTM, 3 seeds: "
                         "precision 0.9719 +/- 0.0015)"},
}

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Repo helpers (preferred) / mirrors (fallback, documented)
# ---------------------------------------------------------------------------

def _repo_stats():
    """Return (mcnemar, boot_prec, boot_f1, how) from the repo if possible."""
    try:
        from phishing_xai.baselines import (  # noqa: E402
            bootstrap_f1_ci,
            bootstrap_precision_ci,
            mcnemar_test,
        )
        return mcnemar_test, bootstrap_precision_ci, bootstrap_f1_ci, "repo"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] repo stats helpers unavailable ({exc}); using mirrors "
              f"of baselines.py:356-460.", flush=True)
        return (mcnemar_mirror, bootstrap_precision_ci_mirror,
                bootstrap_f1_ci_mirror, "mirror")


def mcnemar_mirror(y_true, pred_a, pred_b):
    """Mirror of ``baselines.mcnemar_test`` (:356): b = A wrong/B right,
    c = A right/B wrong; chi2 with continuity correction; exact binomial
    p when b+c < 25; {b:0, c:0} -> p=1.0."""
    from scipy.stats import binomtest, chi2 as chi2_dist
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    a_wrong = pred_a != y_true
    b_wrong = pred_b != y_true
    b = int((a_wrong & ~b_wrong).sum())
    c = int((~a_wrong & b_wrong).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    if n < 25:
        p_value = float(binomtest(min(b, c), n, 0.5,
                                  alternative="two-sided").pvalue)
        return {"b": b, "c": c, "statistic": None, "p_value": p_value}
    stat = float((abs(b - c) - 1) ** 2 / n)
    return {"b": b, "c": c, "statistic": stat,
            "p_value": float(chi2_dist.sf(stat, 1))}


def _bootstrap_ci(values_fn, y_true, y_pred_or_prob, n_resamples=1000,
                  seed=42, ci=0.95):
    """Percentile bootstrap CI, repo scheme (baselines.py:389-460):
    resample (y_true, predictions) pairs with replacement; skip
    resamples with a single class in y_true."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_other = np.asarray(y_pred_or_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(values_fn(y_true[idx], y_other[idx]))
    if not stats:
        return {"value": float(values_fn(y_true, y_other)),
                "ci_low": None, "ci_high": None, "n_resamples": 0}
    stats = np.asarray(stats, dtype=float)
    lo = (1 - ci) / 2
    return {"value": float(values_fn(y_true, y_other)),
            "ci_low": float(np.percentile(stats, 100 * lo)),
            "ci_high": float(np.percentile(stats, 100 * (1 - lo))),
            "n_resamples": int(len(stats))}


def _precision(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


def _recall(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return tp / (tp + fn) if (tp + fn) else 0.0


def _f1(y, p):
    pr, rc = _precision(y, p), _recall(y, p)
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


def bootstrap_precision_ci_mirror(y_true, y_pred, n_resamples=1000, seed=42, ci=0.95):
    r = _bootstrap_ci(_precision, y_true, y_pred, n_resamples, seed, ci)
    return {"precision": r["value"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "n_resamples": r["n_resamples"]}


def bootstrap_f1_ci_mirror(y_true, y_pred, n_resamples=1000, seed=42, ci=0.95):
    r = _bootstrap_ci(_f1, y_true, y_pred, n_resamples, seed, ci)
    return {"f1": r["value"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "n_resamples": r["n_resamples"]}


def bootstrap_recall_ci(y_true, y_pred, n_resamples=1000, seed=42, ci=0.95):
    """Recall CI — same percentile scheme as the repo's precision/F1 CIs
    (extension; no recall CI exists in the repo)."""
    r = _bootstrap_ci(_recall, y_true, y_pred, n_resamples, seed, ci)
    return {"recall": r["value"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "n_resamples": r["n_resamples"]}


def _roc_auc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return 0.5


def _pr_auc(y, p):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return 0.0


def bootstrap_auc_ci(y_true, y_prob, n_resamples=1000, seed=42, ci=0.95):
    """Bootstrap 95% CIs for ROC-AUC and PR-AUC (extension; the repo has
    no AUC bootstrap). Same resampling scheme as the repo CIs."""
    r_roc = _bootstrap_ci(_roc_auc, y_true, y_prob, n_resamples, seed, ci)
    r_pr = _bootstrap_ci(_pr_auc, y_true, y_prob, n_resamples, seed, ci)
    return {"roc_auc": r_roc["value"], "roc_auc_ci_low": r_roc["ci_low"],
            "roc_auc_ci_high": r_roc["ci_high"],
            "pr_auc": r_pr["value"], "pr_auc_ci_low": r_pr["ci_low"],
            "pr_auc_ci_high": r_pr["ci_high"],
            "n_resamples": r_roc["n_resamples"]}


def paired_diff_ci(y_true, prob_a, prob_b, threshold=0.5, n_resamples=1000,
                   seed=42, ci=0.95):
    """Bootstrap CI of the PAIRED difference (A − B) in precision/recall/F1
    on identical examples. A CI excluding 0 is the paired analogue of a
    significance claim for that metric (McNemar itself is errors-only)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    pa = (np.asarray(prob_a, dtype=float) >= threshold).astype(int)
    pb = (np.asarray(prob_b, dtype=float) >= threshold).astype(int)
    n = len(y)
    out = {}
    for name, fn in (("precision", _precision), ("recall", _recall), ("f1", _f1)):
        diffs = []
        for _ in range(n_resamples):
            idx = rng.randint(0, n, size=n)
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
        diffs = np.asarray(diffs, dtype=float)
        lo = (1 - ci) / 2
        out[name] = {
            "diff_A_minus_B": float(fn(y, pa) - fn(y, pb)),
            "ci_low": float(np.percentile(diffs, 100 * lo)) if len(diffs) else None,
            "ci_high": float(np.percentile(diffs, 100 * (1 - lo))) if len(diffs) else None,
            "n_resamples": int(len(diffs)),
        }
    return out


# ---------------------------------------------------------------------------
# Loading predictions
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: Path) -> Dict[str, pd.DataFrame]:
    """All ``<set>_<model>_predictions.csv`` files, keyed ``set__model``."""
    out = {}
    for path in sorted(pred_dir.glob("*_predictions.csv")):
        stem = path.name[: -len("_predictions.csv")]
        model = stem.rsplit("_", 1)[-1]
        set_name = stem[: -len(model) - 1]
        out[f"{set_name}__{model}"] = pd.read_csv(path)
    return out


def align_pair(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two prediction frames on email_id (identical examples
    only). Raises if the join is empty."""
    merged = pred_a.merge(pred_b, on="email_id", suffixes=("_a", "_b"))
    if len(merged) == 0:
        raise ValueError("align_pair: no shared email_ids")
    return merged


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _cell(value, source):
    return {"value": value, "source": source}


def _not_run(reason):
    return {"value": "NOT RUN YET", "source": None, "reason": reason}


def build_tables(results_dir: Path, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Tables A–F from metrics JSONs + stats. Every cell carries its
    source path; missing inputs become NOT RUN YET cells, never guesses."""
    def load_metrics(set_name, model):
        p = results_dir / f"{set_name}_{model}_metrics.json"
        if not p.exists():
            return None, str(p)
        return json.loads(p.read_text()), str(p)

    tables: Dict[str, Any] = {}

    # Table A: original vs modern (PhishNChips), threshold 0.5.
    table_a = {"title": "A: in-distribution test vs modern OOD (PhishNChips v5.2)",
               "note": "PhishNChips is modern URL-grounded synthetic-stimuli OOD, "
                       "not real-world phishing.",
               "rows": {}}
    for model in ("roberta", "bilstm"):
        m, src = load_metrics("phishnchips", model)
        row = {"in_distribution_test": {
            k: _cell(IN_DIST[model][k], IN_DIST[model]["source"])
            for k in ("precision", "recall", "f1", "accuracy", "roc_auc", "n")},
            "modern_ood_phishnchips": {}}
        if m is None:
            for k in ("precision", "recall", "f1", "accuracy", "roc_auc", "n"):
                row["modern_ood_phishnchips"][k] = _not_run(
                    f"phishnchips/{model} not evaluated yet (no {src})")
        else:
            mm = m["metrics_at_0_5"]
            for k in ("precision", "recall", "f1", "accuracy", "roc_auc", "n"):
                row["modern_ood_phishnchips"][k] = _cell(mm.get(k), src)
        table_a["rows"][model] = row
    tables["A_original_vs_modern"] = table_a

    # Table B: human-vs-AI (synthetic track only).
    table_b = {"title": "B: human-authored vs LLM-generated phishing "
                        "(synthetic track only)",
               "note": "Human side is 2007-2008 corpora (TREC-07/CEAS-08/Nazario/"
                       "Nigerian/Enron), NOT modern. Both phishing sets share "
                       "the same PhishNChips legit 1,000 as the paired legit set.",
               "rows": {}}
    for model in ("roberta", "bilstm"):
        row = {}
        for set_name, label in (("frontiers_human", "human_old_corpora"),
                                ("frontiers_llm", "llm_generated")):
            m, src = load_metrics(set_name, model)
            if m is None:
                row[label] = {"recall_at_0_5": _not_run(
                    f"{set_name}/{model} not evaluated yet (no {src})")}
            else:
                mm = m["metrics_at_0_5"]
                row[label] = {
                    "recall_at_0_5": _cell(mm["recall"], src),
                    "precision_at_0_5": _cell(mm["precision"], src),
                    "fpr_on_shared_legit": _cell(mm["false_positive_rate"], src),
                    "n": _cell(m["n"], src),
                }
        table_b["rows"][model] = row
    tables["B_human_vs_ai"] = table_b

    # Table C: temporal — no per-email timestamps exist anywhere.
    tables["C_temporal"] = {
        "title": "C: temporal degradation",
        "rows": {"status": _not_run(
            "No per-email timestamps exist in any OOD set (PhishNChips has "
            "only the April 2026 release date; PhishFuzzer has Year only). "
            "Temporal analysis is not possible without dated emails.")},
    }

    # Table D: robustness (primary set only).
    table_d = {"title": "D: robustness on PhishNChips (F1 at 0.5)",
               "rows": {}}
    for model in ("roberta", "bilstm"):
        p = results_dir / f"robustness_{model}.json"
        if not p.exists():
            table_d["rows"][model] = {"status": _not_run(
                f"robustness kernel not run yet (no {p})")}
            continue
        d = json.loads(p.read_text())
        row = {"n": _cell(d["n"], str(p))}
        for kind in ("truncated", "urls_removed", "typo_synonym"):
            row[kind] = _cell(d["robustness"][kind]["f1"], str(p))
        # Clean F1 comes from the primary metrics file.
        m, src = load_metrics("phishnchips", model)
        row["clean"] = (_cell(m["metrics_at_0_5"]["f1"], src) if m is not None
                        else _not_run("phishnchips metrics missing"))
        table_d["rows"][model] = row
    tables["D_robustness"] = table_d

    # Table E: statistical comparison (from stats).
    tables["E_statistical_comparison"] = {
        "title": "E: RoBERTa vs BiLSTM — paired statistics (McNemar is "
                 "errors-only, NOT a precision test)",
        "rows": stats.get("per_set", {}),
    }

    # Table F: efficiency.
    table_f = {"title": "F: efficiency (latency median, batch 1; params; size)",
               "rows": {}}
    for model in ("roberta", "bilstm"):
        m, src = load_metrics("phishnchips", model)
        if m is None:
            m, src = load_metrics("frontiers_llm", model)
        if m is None:
            table_f["rows"][model] = {"status": _not_run(
                "no evaluated set with latency yet")}
            continue
        lat = m.get("latency_ms_per_email", {})
        info = m.get("model_info", {})
        table_f["rows"][model] = {
            "latency_gpu_ms": _cell(lat.get("gpu", {}).get("latency_ms_per_email_median")
                                    if isinstance(lat.get("gpu"), dict) else None, src),
            "latency_cpu_ms": _cell(lat.get("cpu", {}).get("latency_ms_per_email_median")
                                    if isinstance(lat.get("cpu"), dict) else None, src),
            "latency_n_gpu": _cell(lat.get("gpu", {}).get("n_timed")
                                   if isinstance(lat.get("gpu"), dict) else None, src),
            "latency_n_cpu": _cell(lat.get("cpu", {}).get("n_timed")
                                   if isinstance(lat.get("cpu"), dict) else None, src),
            "n_params": _cell(info.get("n_params"), src),
            "model_size_mb": _cell(info.get("size_mb"), src),
            "latency_texts_source": _cell(lat.get("texts_source"), src),
        }
    tables["F_efficiency"] = table_f
    return tables


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OOD stats + Tables A-F.")
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_resamples", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args(argv)

    pred_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcnemar_fn, boot_prec_fn, boot_f1_fn, how = _repo_stats()
    print(f"[stats] helpers: {how} (mcnemar/bootstrap from "
          f"{'phishing_xai.baselines' if how == 'repo' else 'local mirrors'})",
          flush=True)

    preds = load_predictions(pred_dir)
    print(f"[stats] prediction files: {sorted(preds)}", flush=True)
    sets = sorted({k.rsplit("__", 1)[0] for k in preds})
    stats: Dict[str, Any] = {
        "method": "paired McNemar (errors only) + paired bootstrap 95% CIs "
                  f"(n_resamples={args.n_resamples}, seed={args.seed})",
        "helpers": how,
        "n_seeds": 1,
        "multi_seed_note": "Checkpoints are single (one RoBERTa checkpoint, "
                          "one BiLSTM artefact); no 3-seed variability is claimed.",
        "per_set": {},
    }
    for set_name in sets:
        key_r = f"{set_name}__roberta"
        key_b = f"{set_name}__bilstm"
        entry: Dict[str, Any] = {"set": set_name}
        for key, model in ((key_r, "roberta"), (key_b, "bilstm")):
            if key not in preds:
                entry[model] = {"status": "NOT RUN YET",
                                "reason": f"no predictions file for {key}"}
                continue
            df = preds[key]
            binary = df[df["true_label"] >= 0]
            y = binary["true_label"].astype(int).to_numpy()
            p = binary["prob_phishing"].astype(float).to_numpy()
            pred = (p >= 0.5).astype(int)
            entry[model] = {
                "n": int(len(binary)),
                "bootstrap_precision_ci": boot_prec_fn(
                    y, pred, n_resamples=args.n_resamples, seed=args.seed),
                "bootstrap_recall_ci": bootstrap_recall_ci(
                    y, pred, n_resamples=args.n_resamples, seed=args.seed),
                "bootstrap_f1_ci": boot_f1_fn(
                    y, pred, n_resamples=args.n_resamples, seed=args.seed),
                "bootstrap_auc_ci": bootstrap_auc_ci(
                    y, p, n_resamples=args.n_resamples, seed=args.seed),
            }
        if key_r in preds and key_b in preds:
            merged = align_pair(preds[key_r], preds[key_b])
            binary = merged[merged["true_label_a"] >= 0]
            y = binary["true_label_a"].astype(int).to_numpy()
            pa = (binary["prob_phishing_a"].astype(float).to_numpy() >= 0.5).astype(int)
            pb = (binary["prob_phishing_b"].astype(float).to_numpy() >= 0.5).astype(int)
            entry["paired_n"] = int(len(binary))
            entry["mcnemar_roberta_vs_bilstm"] = mcnemar_fn(y, pa, pb)
            entry["mcnemar_note"] = ("McNemar compares overall errors only; "
                                     "it is NOT a precision test.")
            entry["paired_diff_ci_roberta_minus_bilstm"] = paired_diff_ci(
                y, binary["prob_phishing_a"].astype(float).to_numpy(),
                binary["prob_phishing_b"].astype(float).to_numpy(),
                threshold=0.5, n_resamples=args.n_resamples, seed=args.seed)
        else:
            entry["mcnemar_roberta_vs_bilstm"] = {
                "status": "NOT RUN YET",
                "reason": "both models' predictions are needed for a paired test"}
        stats["per_set"][set_name] = entry
        print(f"[stats] {set_name}: "
              f"{entry.get('mcnemar_roberta_vs_bilstm', {})}", flush=True)

    tables = build_tables(results_dir, stats)
    (out_dir / "ood_stats.json").write_text(json.dumps(stats, indent=2, default=str),
                                            encoding="utf-8")
    (out_dir / "ood_tables.json").write_text(json.dumps(tables, indent=2, default=str),
                                             encoding="utf-8")
    print(f"[done] wrote {out_dir / 'ood_stats.json'} and "
          f"{out_dir / 'ood_tables.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
