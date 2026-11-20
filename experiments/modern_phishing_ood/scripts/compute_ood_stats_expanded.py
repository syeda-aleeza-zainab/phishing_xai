"""Expanded OOD statistical comparison (Task 7 + Task 8).

Extends ``compute_ood_stats.py`` (RoBERTa vs BiLSTM only) to EVERY model
pair present in the saved per-email predictions — TF-IDF+SVC and
TF-IDF+LR included once their Kernel 4 files exist — and adds:

* paired confusion-matrix comparisons (per-model confusion matrices,
  prediction-agreement table, error table) per set per pair;
* McNemar on paired errors ONLY (explicitly NOT a precision test);
* paired bootstrap 95% CIs for precision / recall / F1 differences and
  for ROC-AUC / PR-AUC differences (1,000 resamples, seed 42);
* multiple-comparison correction: Holm-Bonferroni across the McNemar
  family WITHIN each set (primary), plus a global Holm across all
  pairs x sets as a sensitivity check; Bonferroni-adjusted p reported
  for reference. "Significant" means Holm-adjusted p < 0.05 (McNemar)
  or a paired-difference CI excluding 0 (metric differences) — a bare
  numerical gap is never called significant;
* H1a / H1b verdicts (Task 8) on the primary set: H1a = RoBERTa better
  RANKING than BiLSTM (ROC-AUC / PR-AUC paired differences); H1b =
  RoBERTa better OPERATIONAL detection under the predefined deployment
  criteria (recall floor 0.85 at threshold 0.5 and at the
  validation-fitted thresholds). Ranking and deployment readiness are
  judged independently — a ranking win is not a deployment win.

Reads ONLY saved prediction CSVs and metrics JSONs; never re-scores.
Missing models are reported as NOT RUN YET with the reason — the
four-model tables are incomplete until the TF-IDF OOD run lands.

Outputs (new files; ``statistical_report.md`` / ``ood_stats.json`` are
never overwritten):
* ``<out_dir>/ood_stats_expanded.json`` — full per-set per-pair numbers;
* ``<out_dir>/statistical_comparison_expanded.md`` — the readable report.

Usage:
    python experiments/modern_phishing_ood/scripts/compute_ood_stats_expanded.py \
        --predictions_dir experiments/modern_phishing_ood/predictions \
        --results_dir experiments/modern_phishing_ood/results \
        --out_dir experiments/modern_phishing_ood/reports
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compute_ood_stats as base  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

MODELS = ("roberta", "bilstm", "tfidf_svm", "tfidf_lr")
SETS = ("phishnchips", "phishfuzzer", "frontiers_llm", "frontiers_human",
        "bec", "bec_poisoned")
RECALL_FLOOR = 0.85
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def holm_adjust(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Sort ascending; adjusted p_(i) = max over j<=i of (m-j+1) * p_(j),
    capped at 1.0. Returns [] for an empty input.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def bonferroni_adjust(p_values: List[float]) -> List[float]:
    """Bonferroni-adjusted p-values (reference only; Holm is primary)."""
    m = len(p_values)
    return [min(1.0, m * p) for p in p_values]


# ---------------------------------------------------------------------------
# Paired AUC-difference CIs
# ---------------------------------------------------------------------------

def paired_auc_diff_ci(y_true, prob_a, prob_b, n_resamples=1000,
                       seed=42, ci=0.95) -> Dict[str, Any]:
    """Bootstrap 95% CI of the PAIRED difference (A - B) in ROC-AUC and
    PR-AUC on identical examples. Same percentile scheme as
    ``compute_ood_stats._bootstrap_ci`` (resample pairs, skip single-class
    resamples). A CI excluding 0 is the paired analogue of a significance
    claim for the ranking difference."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    n = len(y)
    out: Dict[str, Any] = {}
    for name, fn in (("roc_auc", base._roc_auc), ("pr_auc", base._pr_auc)):
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
            "significant": bool(len(diffs) and
                                (float(np.percentile(diffs, 100 * lo)) > 0
                                 or float(np.percentile(diffs, 100 * (1 - lo))) < 0)),
        }
    return out


# ---------------------------------------------------------------------------
# Per-pair comparison
# ---------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred) -> List[List[int]]:
    """[[TN, FP], [FN, TP]] at threshold 0.5."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    return [[int(((p == 0) & (y == 0)).sum()), int(((p == 1) & (y == 0)).sum())],
            [int(((p == 0) & (y == 1)).sum()), int(((p == 1) & (y == 1)).sum())]]


def compare_pair(merged: pd.DataFrame, model_a: str, model_b: str,
                 n_resamples: int, seed: int) -> Dict[str, Any]:
    """Full paired comparison of two models on identical (merged) rows."""
    binary = merged[merged["true_label_a"] >= 0].copy()
    y = binary["true_label_a"].astype(int).to_numpy()
    pa_prob = binary["prob_phishing_a"].astype(float).to_numpy()
    pb_prob = binary["prob_phishing_b"].astype(float).to_numpy()
    pa = (pa_prob >= 0.5).astype(int)
    pb = (pb_prob >= 0.5).astype(int)
    out: Dict[str, Any] = {
        "models": [model_a, model_b],
        "n_paired": int(len(binary)),
        "confusion_matrix_a": confusion_matrix(y, pa),
        "confusion_matrix_b": confusion_matrix(y, pb),
        # Prediction-agreement table [[A=1,B=1],[A=1,B=0],[A=0,B=1],[A=0,B=0]].
        "prediction_agreement": [
            [int(((pa == 1) & (pb == 1)).sum()), int(((pa == 1) & (pb == 0)).sum())],
            [int(((pa == 0) & (pb == 1)).sum()), int(((pa == 0) & (pb == 0)).sum())],
        ],
        # Error table [[both correct],[A wrong B right],[A right B wrong],[both wrong]].
        "error_table": {
            "both_correct": int(((pa == y) & (pb == y)).sum()),
            "a_wrong_b_right": int(((pa != y) & (pb == y)).sum()),
            "a_right_b_wrong": int(((pa == y) & (pb != y)).sum()),
            "both_wrong": int(((pa != y) & (pb != y)).sum()),
        },
    }
    mc = base.mcnemar_mirror(y, pa, pb)
    mc["note"] = ("McNemar compares overall errors only; it is NOT a "
                  "precision test and says nothing about which model has "
                  "higher precision or recall.")
    out["mcnemar"] = mc
    diffs = base.paired_diff_ci(y, pa_prob, pb_prob, threshold=0.5,
                                n_resamples=n_resamples, seed=seed)
    for metric, d in diffs.items():
        d["significant"] = bool(
            d["ci_low"] is not None and d["ci_high"] is not None
            and (d["ci_low"] > 0 or d["ci_high"] < 0))
    out["paired_metric_diff_ci"] = diffs
    out["paired_auc_diff_ci"] = paired_auc_diff_ci(
        y, pa_prob, pb_prob, n_resamples=n_resamples, seed=seed)
    return out


# ---------------------------------------------------------------------------
# H1a / H1b (primary set, RoBERTa vs BiLSTM)
# ---------------------------------------------------------------------------

def evaluate_h1(primary_pair: Optional[Dict[str, Any]],
                results_dir: Path) -> Dict[str, Any]:
    """H1a (ranking) and H1b (operational detection), judged independently.

    H1a: RoBERTa ranks phishing above legit better than BiLSTM on the
    primary OOD set — supported iff the paired ROC-AUC difference CI
    AND the paired PR-AUC difference CI both exclude 0 in RoBERTa's
    favour (each reported separately too).
    H1b: RoBERTa is the better OPERATIONAL detector under the predefined
    deployment criteria — recall floor 0.85 at threshold 0.5 and at the
    validation-fitted thresholds. Supported iff RoBERTa meets the floor
    where BiLSTM does not (or dominates it operationally); rejected if
    neither meets the floor. A ranking win (H1a) is NOT a deployment win.
    """
    out: Dict[str, Any] = {"recall_floor": RECALL_FLOOR,
                           "primary_set": "phishnchips"}
    if primary_pair is None:
        out["status"] = "NOT RUN YET"
        out["reason"] = "no paired RoBERTa/BiLSTM primary predictions"
        return out
    auc = primary_pair["paired_auc_diff_ci"]
    roc_sig = bool(auc["roc_auc"]["significant"]
                   and auc["roc_auc"]["diff_A_minus_B"] > 0)
    pr_sig = bool(auc["pr_auc"]["significant"]
                  and auc["pr_auc"]["diff_A_minus_B"] > 0)
    out["H1a_ranking"] = {
        "roc_auc_diff_roberta_minus_bilstm": auc["roc_auc"],
        "pr_auc_diff_roberta_minus_bilstm": auc["pr_auc"],
        "verdict": ("SUPPORTED" if (roc_sig and pr_sig) else
                    "PARTIALLY SUPPORTED (ROC-AUC only)" if roc_sig else
                    "PARTIALLY SUPPORTED (PR-AUC only)" if pr_sig else
                    "REJECTED"),
        "note": "Ranking only. A ranking win does not imply deployment "
                "readiness (see H1b).",
    }
    # H1b: operational criteria from the metrics JSONs.
    def _recall_at(model: str, where: str) -> Optional[float]:
        p = results_dir / f"phishnchips_{model}_metrics.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        if where == "0.5":
            return (d.get("metrics_at_0_5") or {}).get("recall")
        return ((d.get("metrics_at_fitted_thresholds") or {}).get(where) or {}).get("recall")
    op: Dict[str, Any] = {}
    for model in ("roberta", "bilstm"):
        op[model] = {
            "recall_at_0_5": _recall_at(model, "0.5"),
            "recall_at_fitted_0.85": _recall_at(model, "0.85"),
            "recall_at_fitted_0.90": _recall_at(model, "0.90"),
            "recall_at_fitted_0.95": _recall_at(model, "0.95"),
        }
        op[model]["floor_met_at_0_5"] = (
            op[model]["recall_at_0_5"] is not None
            and op[model]["recall_at_0_5"] >= RECALL_FLOOR)
        op[model]["floor_met_at_any_fitted"] = any(
            (op[model][k] is not None and op[model][k] >= RECALL_FLOOR)
            for k in ("recall_at_fitted_0.85", "recall_at_fitted_0.90",
                      "recall_at_fitted_0.95"))
    out["H1b_operational"] = op
    r_floor = op["roberta"]["floor_met_at_0_5"] or op["roberta"]["floor_met_at_any_fitted"]
    b_floor = op["bilstm"]["floor_met_at_0_5"] or op["bilstm"]["floor_met_at_any_fitted"]
    if r_floor and not b_floor:
        verdict = "SUPPORTED"
    elif r_floor and b_floor:
        verdict = ("NOT SUPPORTED as a RoBERTa advantage — BOTH models meet "
                   "the recall floor; H1b asked whether RoBERTa is better, "
                   "and a shared pass is not a RoBERTa win")
    else:
        verdict = ("REJECTED — NEITHER model meets the predefined recall "
                   "floor 0.85 on the primary OOD set, at threshold 0.5 or "
                   "at any validation-fitted threshold")
    out["H1b_operational"]["verdict"] = verdict
    out["H1b_operational"]["note"] = (
        "Deployment criteria are recall floor 0.85 (predefined) plus the "
        "operational metrics in the Final Decision (precision, FPR, "
        "latency). H1b is judged on those, never on ROC-AUC/PR-AUC.")
    return out


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "NOT RUN YET"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def render_markdown(stats: Dict[str, Any]) -> str:
    """Readable report from the expanded stats JSON."""
    lines: List[str] = []
    lines.append("# OOD Statistical Comparison — Expanded (RoBERTa vs BiLSTM vs TF-IDF+SVC vs TF-IDF+LR)")
    lines.append("")
    lines.append("Paired comparisons on IDENTICAL OOD examples (per-email predictions "
                 "aligned on `email_id`). McNemar compares overall errors ONLY — it is "
                 "explicitly NOT a precision test. A difference is called significant "
                 "only when its paired bootstrap 95% CI excludes 0 (metric differences) "
                 "or its Holm-adjusted McNemar p < 0.05; a bare numerical gap is "
                 "reported as a numerical difference only, never as significant.")
    lines.append("")
    lines.append(f"Method: {stats['method']}")
    lines.append("")
    lines.append(f"Multiple-comparison correction: {stats['correction']}")
    lines.append("")
    lines.append("Original RoBERTa-vs-BiLSTM findings are RETAINED (see "
                 "`statistical_report.md` / `ood_stats.json`); this report ADDS the "
                 "TF-IDF pairs and the correction. Nothing in the original evaluation "
                 "was changed.")
    lines.append("")
    tfidf = stats["models_present"]
    if "tfidf_svm" in tfidf and "tfidf_lr" in tfidf:
        tfidf_note = "TF-IDF OOD: COMPLETED."
    else:
        tfidf_note = ("TF-IDF OOD: NOT RUN YET — no *_tfidf_*_predictions.csv "
                      "files found; TF-IDF pairs below are NOT RUN YET, not zero.")
    lines.append(f"Models present in the predictions dir: {', '.join(tfidf)}. {tfidf_note}")
    lines.append("")
    for set_name in SETS:
        entry = stats["per_set"].get(set_name)
        if entry is None:
            continue
        lines.append(f"## {set_name} (n={entry['n']})")
        lines.append("")
        for pair_key, comp in entry["pairs"].items():
            if comp.get("status") == "NOT RUN YET":
                lines.append(f"### {pair_key}\n\nNOT RUN YET — {comp.get('reason', '')}\n")
                continue
            a, b = comp["models"]
            mc = comp["mcnemar"]
            lines.append(f"### {a} vs {b} (n paired = {comp['n_paired']})")
            lines.append("")
            lines.append("| confusion (rows = truth, cols = predicted) | "
                         f"{a} [[TN, FP], [FN, TP]] | {b} [[TN, FP], [FN, TP]] |")
            lines.append("|---|---|---|")
            lines.append(f"| confusion matrix | {comp['confusion_matrix_a']} | {comp['confusion_matrix_b']} |")
            lines.append(f"| prediction agreement [[both phish, {a} only], [{b} only, both safe]] | "
                         f"{comp['prediction_agreement']} | |")
            et = comp["error_table"]
            lines.append(f"| errors: both correct / {a} wrong only / {b} wrong only / both wrong | "
                         f"{et['both_correct']} / {et['a_wrong_b_right']} / {et['a_right_b_wrong']} / {et['both_wrong']} | |")
            lines.append("")
            lines.append(f"McNemar (errors only): b={mc['b']} (A wrong, B right), c={mc['c']} "
                         f"(A right, B wrong), raw p={_fmt(mc['p_value'])}, "
                         f"Holm-adjusted p={_fmt(comp.get('mcnemar_p_holm'))}, "
                         f"Bonferroni-adjusted p={_fmt(comp.get('mcnemar_p_bonferroni'))} — "
                         f"{'SIGNIFICANT (Holm p < 0.05)' if comp.get('mcnemar_significant_holm') else 'not significant after Holm correction'}. "
                         f"McNemar is NOT a precision test.")
            lines.append("")
            lines.append("| paired difference (A − B) | value | 95% CI | significant? |")
            lines.append("|---|---|---|---|")
            for metric, d in comp["paired_metric_diff_ci"].items():
                lines.append(f"| {metric} | {_fmt(d['diff_A_minus_B'])} | "
                             f"[{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}] | "
                             f"{'yes (CI excludes 0)' if d['significant'] else 'no — numerical difference only'} |")
            for metric, d in comp["paired_auc_diff_ci"].items():
                lines.append(f"| {metric} | {_fmt(d['diff_A_minus_B'])} | "
                             f"[{_fmt(d['ci_low'])}, {_fmt(d['ci_high'])}] | "
                             f"{'yes (CI excludes 0)' if d['significant'] else 'no — numerical difference only'} |")
            lines.append("")
    h = stats.get("hypotheses", {})
    if h:
        lines.append("## Hypotheses (primary OOD, PhishNChips v5.2)")
        lines.append("")
        h1a = h.get("H1a_ranking", {})
        lines.append(f"### H1a — RoBERTa better RANKING than BiLSTM: **{h1a.get('verdict', 'NOT RUN YET')}**")
        lines.append("")
        for k in ("roc_auc_diff_roberta_minus_bilstm", "pr_auc_diff_roberta_minus_bilstm"):
            d = h1a.get(k, {})
            lines.append(f"- {k}: diff {_fmt(d.get('diff_A_minus_B'))}, 95% CI "
                         f"[{_fmt(d.get('ci_low'))}, {_fmt(d.get('ci_high'))}], "
                         f"significant: {d.get('significant')}")
        lines.append("")
        h1b = h.get("H1b_operational", {})
        lines.append(f"### H1b — RoBERTa better OPERATIONAL detection (recall floor 0.85): **{h1b.get('verdict', 'NOT RUN YET')}**")
        lines.append("")
        for model in ("roberta", "bilstm"):
            m = h1b.get(model, {})
            lines.append(f"- {model}: recall at 0.5 = {_fmt(m.get('recall_at_0_5'))}, "
                         f"at fitted 0.85 = {_fmt(m.get('recall_at_fitted_0.85'))}, "
                         f"at fitted 0.90 = {_fmt(m.get('recall_at_fitted_0.90'))}, "
                         f"at fitted 0.95 = {_fmt(m.get('recall_at_fitted_0.95'))}; "
                         f"floor met at 0.5: {m.get('floor_met_at_0_5')}, "
                         f"at any fitted threshold: {m.get('floor_met_at_any_fitted')}")
        lines.append("")
        lines.append("H1a and H1b are independent: a ranking win is not a deployment win, "
                     "and a deployment failure is not a ranking failure.")
        lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(f"TF-IDF pairs: {stats['tfidf_status']}")
    lines.append("")
    lines.append(f"Generated by `compute_ood_stats_expanded.py` from saved prediction "
                 f"CSVs only (never re-scored). Bootstrap: {N_BOOTSTRAP} resamples, "
                 f"seed {BOOTSTRAP_SEED}.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Expanded OOD statistical comparison.")
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

    preds = base.load_predictions(pred_dir)
    print(f"[stats-expanded] prediction files: {sorted(preds)}", flush=True)
    models_present = sorted({k.rsplit("__", 1)[1] for k in preds})
    sets_present = sorted({k.rsplit("__", 1)[0] for k in preds})

    stats: Dict[str, Any] = {
        "method": ("paired McNemar (errors only) + paired bootstrap 95% CIs "
                   f"(n_resamples={args.n_resamples}, seed={args.seed}) for "
                   "precision/recall/F1 and ROC-AUC/PR-AUC differences, on "
                   "identical examples aligned on email_id"),
        "correction": ("Holm-Bonferroni across the McNemar family WITHIN each "
                       "set (primary); global Holm across all pairs x sets "
                       "reported as a sensitivity check; Bonferroni-adjusted p "
                       "for reference. Significant = Holm-adjusted p < 0.05."),
        "models_present": models_present,
        "tfidf_status": ("COMPLETED" if "tfidf_svm" in models_present
                         and "tfidf_lr" in models_present else
                         "NOT RUN YET — no TF-IDF OOD prediction files found"),
        "n_seeds": 1,
        "multi_seed_note": ("Checkpoints are single (one RoBERTa checkpoint, "
                            "one BiLSTM artefact, one TF-IDF refit per model); "
                            "no multi-seed variability is claimed."),
        "per_set": {},
    }

    # Collect raw McNemar p-values per set for Holm (family = pairs in set),
    # and globally for the sensitivity check.
    all_raw: List[Tuple[str, str, float]] = []
    for set_name in SETS:
        if set_name not in sets_present:
            continue
        entry: Dict[str, Any] = {"set": set_name}
        # n = binary rows of any present model file for this set.
        n_rows = None
        for k, df in preds.items():
            if k.startswith(set_name + "__"):
                n_rows = int((df["true_label"] >= 0).sum())
                break
        entry["n"] = n_rows
        entry["pairs"] = {}
        set_raw: List[Tuple[str, float]] = []
        for model_a, model_b in itertools.combinations(MODELS, 2):
            key_a, key_b = f"{set_name}__{model_a}", f"{set_name}__{model_b}"
            pair_key = f"{model_a}_vs_{model_b}"
            if key_a not in preds or key_b not in preds:
                missing = [m for m, k in ((model_a, key_a), (model_b, key_b))
                           if k not in preds]
                entry["pairs"][pair_key] = {
                    "status": "NOT RUN YET",
                    "reason": f"no predictions file for {', '.join(missing)}",
                }
                continue
            merged = base.align_pair(preds[key_a], preds[key_b])
            comp = compare_pair(merged, model_a, model_b,
                                args.n_resamples, args.seed)
            entry["pairs"][pair_key] = comp
            set_raw.append((pair_key, float(comp["mcnemar"]["p_value"])))
            all_raw.append((set_name, pair_key,
                            float(comp["mcnemar"]["p_value"])))
        # Holm within this set.
        if set_raw:
            keys = [k for k, _ in set_raw]
            raw = [p for _, p in set_raw]
            holm = holm_adjust(raw)
            bonf = bonferroni_adjust(raw)
            for k, h, bf in zip(keys, holm, bonf):
                c = entry["pairs"][k]
                c["mcnemar_p_holm"] = h
                c["mcnemar_p_bonferroni"] = bf
                c["mcnemar_significant_holm"] = bool(h < 0.05)
        stats["per_set"][set_name] = entry
        print(f"[stats-expanded] {set_name}: pairs done", flush=True)

    # Global Holm across all pairs x sets (sensitivity check).
    if all_raw:
        raw = [p for _, _, p in all_raw]
        holm = holm_adjust(raw)
        stats["global_holm_sensitivity"] = [
            {"set": s, "pair": k, "raw_p": p, "holm_adjusted_p": h,
             "significant_holm_global": bool(h < 0.05)}
            for (s, k, p), h in zip(all_raw, holm)
        ]

    primary = stats["per_set"].get("phishnchips", {}).get("pairs", {}).get(
        "roberta_vs_bilstm")
    stats["hypotheses"] = evaluate_h1(primary, results_dir)

    (out_dir / "ood_stats_expanded.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8")
    (out_dir / "statistical_comparison_expanded.md").write_text(
        render_markdown(stats), encoding="utf-8")
    print(f"[done] wrote {out_dir / 'ood_stats_expanded.json'} and "
          f"{out_dir / 'statistical_comparison_expanded.md'}", flush=True)
    print(f"[done] H1a: {stats['hypotheses'].get('H1a_ranking', {}).get('verdict')} | "
          f"H1b: {stats['hypotheses'].get('H1b_operational', {}).get('verdict')}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
