"""CLI: generate the Step 6B.5 system-complexity report (S1-S4).

Reads ONLY already-produced JSON files (never trains, never loads a model)
and produces::

    <results_dir>/system_complexity_report.md
    <results_dir>/system_complexity_report.json
    <results_dir>/system_complexity_plots/*.png   (4 plots, when data exists)

Systems: S1 TF-IDF+LR, S2 BiLSTM, S3 DistilBERT (skipped unless run),
S4 RoBERTa. Any missing input is recorded as ``NOT RUN YET`` with the
reason; nothing is fabricated. Exit 0 even when inputs are missing;
exit non-zero only on an unreadable/corrupt JSON input file.

The verdict rule mirrors ``scripts/run_complexity_report.py::build_verdict``
(same 1.0-precision-point / overlapping-bootstrap-CI / McNemar p > 0.05
conditions, same precision-first selection) and extends it to BOTH the
text test set and the multimodal held-out set. The text-only verdict from
``build_verdict`` is also reproduced in the JSON for reference.

Usage::

    python scripts/run_system_complexity_report.py --results_dir results
    python scripts/run_system_complexity_report.py --results_dir results --ood_csv data/ood.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root / src to path (same pattern as run_complexity_report.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.baselines import DATASET_CAVEAT
from phishing_xai.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("run_system_complexity_report")

NOT_RUN = "NOT RUN YET"

SYSTEMS: Dict[str, str] = {
    "S1": "tfidf_lr",
    "S2": "bilstm",
    "S3": "distilbert",
    "S4": "roberta_eval",
}
SYSTEM_LABELS: Dict[str, str] = {
    "S1": "S1 TF-IDF+LR",
    "S2": "S2 BiLSTM",
    "S3": "S3 DistilBERT",
    "S4": "S4 RoBERTa",
}
RECALL_FLOOR = 0.85
PRECISION_TOL_POINTS = 1.0

# Candidate input files (all optional). ``kind`` is used by the reader.
INPUT_SPECS: List[Tuple[str, str]] = [
    ("baselines/baseline_results.json", "text test-set ladder (S1-S4 aggregates, McNemar, bootstrap CIs, robustness, faithfulness)"),
    ("metrics.json", "RoBERTa test metrics (fallback for S4 if baseline_results.json has no roberta_eval)"),
    ("bilstm/tuning.json", "BiLSTM tuning (validation precision/recall, best config)"),
    ("bilstm/faithfulness.json", "BiLSTM (6B.3) faithfulness summary"),
    ("bilstm/alpha_ablation.json", "BiLSTM (6B.3) alpha-ablation summary"),
    ("bilstm/explanations/explanations.json", "BiLSTM per-email top words (run_explain.py --model bilstm)"),
    ("bilstm/explanations.json", "BiLSTM per-email top words (alternate location)"),
    ("explanations/explanations.json", "RoBERTa per-email top words (run_explain.py --model roberta)"),
    ("faithfulness.json", "Stage C RoBERTa faithfulness summary"),
    ("alpha_ablation.json", "Stage C RoBERTa alpha-ablation summary"),
    ("multimodal_eval/multimodal_eval.json", "RoBERTa (S4) multimodal 4-ablation table + held-out metrics"),
    ("bilstm/multimodal_eval.json", "BiLSTM (S2) multimodal 4-ablation table + held-out metrics"),
    ("bilstm/multimodal_eval/multimodal_eval.json", "BiLSTM (S2) multimodal 4-ablation table (alternate location)"),
    ("multimodal_eval_rule_based/multimodal_eval.json", "Rule-based multimodal run (reference only, not a system)"),
]

MM_VARIANTS = ("a", "b", "c", "d_weighted", "d_max")
MM_VARIANT_LABELS = {
    "a": "(a) best part",
    "b": "(b) concatenated context",
    "c": "(c) context + URL risk",
    "d_weighted": "(d) full fusion, weighted",
    "d_max": "(d) full fusion, max",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the system-complexity report."""
    parser = argparse.ArgumentParser(description="Generate Step 6B.5 system-complexity report")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory holding the already-produced JSON inputs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <results_dir>)")
    parser.add_argument("--ood_csv", type=str, default=None,
                        help="Optional second-dataset CSV; noted in the report, never evaluated here")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _mean(entry: Any) -> Optional[float]:
    """Mean of a ``{"mean": ..}`` aggregate, a bare number, or None."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("mean")
        return float(val) if isinstance(val, (int, float)) and not math.isnan(val) else None
    if isinstance(entry, (int, float)) and not math.isnan(entry):
        return float(entry)
    return None


def _std(entry: Any) -> Optional[float]:
    """Std of a ``{"mean", "std"}`` aggregate, else None."""
    if isinstance(entry, dict):
        val = entry.get("std")
        return float(val) if isinstance(val, (int, float)) and not math.isnan(val) else None
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a number for the table, or NOT RUN YET for missing values."""
    if value is None:
        return NOT_RUN
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return NOT_RUN
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _mean_std_str(entry: Any) -> str:
    """Render a ``{"mean": .., "std": ..}`` aggregate as ``mean +/- std``."""
    mean = _mean(entry)
    if mean is None:
        return NOT_RUN
    std = _std(entry)
    return f"{mean:.4f} +/- {std:.4f}" if std is not None else f"{mean:.4f}"


def _load_optional(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Load an optional JSON input.

    Returns ``(data, status)`` where status is ``"ok"``, ``"missing: ..."``
    or raises ``ValueError`` for a present-but-corrupt/unreadable file (the
    caller turns that into a non-zero exit).
    """
    if not path.exists():
        return None, f"missing: {path} not found"
    try:
        data = load_json(path)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Unreadable/corrupt JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Unreadable/corrupt JSON at {path}: top level is not an object")
    return data, "ok"


def _first_present(paths: List[Path]) -> Optional[Path]:
    """First path in the list that exists, else None."""
    for p in paths:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------

def load_inputs(results_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    """Load every optional input; returns ``(inputs, corrupt_errors)``.

    ``inputs`` maps a short key to the parsed JSON (or None when missing).
    A present-but-corrupt file is recorded in ``corrupt_errors`` and the
    entry is left as None; ``main`` exits non-zero in that case.
    """
    inputs: Dict[str, Any] = {}
    errors: List[str] = []
    status: Dict[str, str] = {}
    for rel, _desc in INPUT_SPECS:
        try:
            data, st = _load_optional(results_dir / rel)
        except ValueError as exc:
            errors.append(str(exc))
            data, st = None, "corrupt"
        inputs[rel] = data
        status[rel] = st
    inputs["_status"] = status
    return inputs, errors


def _faithfulness_at_5_from_summary(summary: Any) -> Optional[Dict[str, Any]]:
    """Comprehensiveness/sufficiency @5 from a faithfulness summary JSON.

    Accepts both the ``run_faithfulness.py`` shape
    (``summary[method].comprehensiveness_mean["5"]``) and the
    ``baseline_results.json`` faithfulness shape
    (``faithfulness[model].comprehensiveness["5"]``).
    Prefers the LITA magnitude-preserving method; falls back to any method
    whose name starts with ``lita``.
    """
    if not isinstance(summary, dict):
        return None
    # baseline_results.json shape: {"comprehensiveness": {"5": ..}, "sufficiency": {"5": ..}}
    if isinstance(summary.get("comprehensiveness"), dict):
        comp = summary["comprehensiveness"].get("5", summary["comprehensiveness"].get(5))
        suff = summary.get("sufficiency", {}).get("5", summary.get("sufficiency", {}).get(5))
        if comp is None and suff is None:
            return None
        return {"comprehensiveness": comp, "sufficiency": suff,
                "method": summary.get("method") or summary.get("explanation_method")}
    # run_faithfulness.py shape: {"summary": {method: {comprehensiveness_mean: {"5": ..}}}}
    table = summary.get("summary") if isinstance(summary.get("summary"), dict) else summary
    if not isinstance(table, dict):
        return None
    for method in ("lita_magnitude", "lita", "lita_minmax"):
        entry = table.get(method)
        if isinstance(entry, dict):
            comp = (entry.get("comprehensiveness_mean") or {}).get("5")
            suff = (entry.get("sufficiency_mean") or {}).get("5")
            if comp is not None or suff is not None:
                return {"comprehensiveness": comp, "sufficiency": suff, "method": method}
    for method, entry in table.items():
        if isinstance(entry, dict) and str(method).startswith("lita"):
            comp = (entry.get("comprehensiveness_mean") or {}).get("5")
            suff = (entry.get("sufficiency_mean") or {}).get("5")
            if comp is not None or suff is not None:
                return {"comprehensiveness": comp, "sufficiency": suff, "method": method}
    return None


def _mm_from_eval_file(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Held-out metrics + ablation table from one multimodal_eval.json."""
    if not isinstance(data, dict):
        return {"status": NOT_RUN, "reason": "multimodal_eval.json not found"}
    test_metrics = data.get("test_metrics") or {}
    full = test_metrics.get("d_weighted") or {}
    out: Dict[str, Any] = {
        "status": "ok" if full else NOT_RUN,
        "precision": full.get("precision"),
        "recall": full.get("recall"),
        "f1": full.get("f1"),
        "false_positive_rate": full.get("false_positive_rate"),
        "n": full.get("n"),
        "untrained_fallback": bool(data.get("untrained_fallback", False)),
        "ablation": {
            v: {
                "precision": (test_metrics.get(v) or {}).get("precision"),
                "recall": (test_metrics.get(v) or {}).get("recall"),
                "f1": (test_metrics.get(v) or {}).get("f1"),
            }
            for v in MM_VARIANTS if isinstance(test_metrics.get(v), dict)
        },
        # Optional per-backbone-vs-RoBERTa statistics (written by a future
        # run_baselines.py --multimodal_eval_dir comparison, or by hand from
        # the per-row predictions). Absent in current outputs.
        "mcnemar_p_vs_roberta": (data.get("mcnemar_vs_roberta") or {}).get("p_value"),
        "bootstrap_ci_low": (data.get("bootstrap_precision_ci") or {}).get("ci_low"),
        "bootstrap_ci_high": (data.get("bootstrap_precision_ci") or {}).get("ci_high"),
    }
    # A backbone file may also carry a "baselines" block (tfidf_lr on the
    # same synthetic set) — handled by the caller for S1.
    return out


def build_multimodal_by_system(inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Held-out multimodal metrics per system (S1-S4)."""
    mm: Dict[str, Dict[str, Any]] = {}
    roberta_mm = _mm_from_eval_file(inputs.get("multimodal_eval/multimodal_eval.json"))
    mm["S4"] = roberta_mm
    # S1: TF-IDF+LR evaluated on the same synthetic set (baselines block of
    # the RoBERTa multimodal file) or the baseline_results.json multimodal block.
    s1: Dict[str, Any] = {"status": NOT_RUN, "reason": "no TF-IDF+LR multimodal run found"}
    roberta_file = inputs.get("multimodal_eval/multimodal_eval.json") or {}
    tfidf_block = (roberta_file.get("baselines") or {}).get("tfidf_lr") or {}
    if isinstance(tfidf_block, dict) and tfidf_block.get("status") == "ok":
        s1 = {"status": "ok", "precision": tfidf_block.get("precision"),
              "recall": tfidf_block.get("recall"), "f1": tfidf_block.get("f1"),
              "false_positive_rate": tfidf_block.get("false_positive_rate"),
              "n": tfidf_block.get("n"), "ablation": None,
              "mcnemar_p_vs_roberta": None, "bootstrap_ci_low": None,
              "bootstrap_ci_high": None,
              "reason": "from baselines.tfidf_lr in multimodal_eval.json (no McNemar/CI stored)"}
    mm["S1"] = s1
    bilstm_file = (
        inputs.get("bilstm/multimodal_eval.json")
        or inputs.get("bilstm/multimodal_eval/multimodal_eval.json")
    )
    mm["S2"] = _mm_from_eval_file(bilstm_file)
    mm["S3"] = {"status": NOT_RUN,
                "reason": "DistilBERT multimodal run not present (skipped, like S3 on the text set)"}
    # baseline_results.json "multimodal" block (run_baselines.py
    # --multimodal_eval_dir) overrides / fills per-model stats where present.
    baseline = inputs.get("baselines/baseline_results.json") or {}
    mm_block = baseline.get("multimodal") if isinstance(baseline.get("multimodal"), dict) else {}
    for sys_id, model in SYSTEMS.items():
        entry = mm_block.get(model)
        if isinstance(entry, dict) and entry.get("precision") is not None:
            cur = mm.get(sys_id, {})
            cur.update({
                "status": "ok",
                "precision": entry.get("precision", cur.get("precision")),
                "recall": entry.get("recall", cur.get("recall")),
                "f1": entry.get("f1", cur.get("f1")),
                "mcnemar_p_vs_roberta": (entry.get("mcnemar_vs_roberta") or {}).get("p_value"),
                "bootstrap_ci_low": (entry.get("bootstrap_precision_ci") or {}).get("ci_low"),
                "bootstrap_ci_high": (entry.get("bootstrap_precision_ci") or {}).get("ci_high"),
            })
            mm[sys_id] = cur
    return mm


def build_system_rows(inputs: Dict[str, Any], mm: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one row per system (S1-S4) from the loaded inputs."""
    baseline = inputs.get("baselines/baseline_results.json") or {}
    models = baseline.get("models", {}) if isinstance(baseline.get("models"), dict) else {}
    comparisons = baseline.get("comparisons_vs_roberta", {}) if isinstance(baseline.get("comparisons_vs_roberta"), dict) else {}
    faith_block = baseline.get("faithfulness", {}) if isinstance(baseline.get("faithfulness"), dict) else {}
    metrics_json = inputs.get("metrics.json") or {}
    rows: List[Dict[str, Any]] = []
    for sys_id, model in SYSTEMS.items():
        entry = models.get(model) if isinstance(models.get(model), dict) else {}
        agg = entry.get("aggregate", {}) if isinstance(entry.get("aggregate"), dict) else {}
        per_seed = entry.get("per_seed", []) if isinstance(entry.get("per_seed"), list) else []
        row: Dict[str, Any] = {"system": sys_id, "model": model, "label": SYSTEM_LABELS[sys_id],
                               "status": "ok" if agg else NOT_RUN, "reasons": {}}
        if not agg:
            row["reasons"]["text"] = (
                f"{model} not in baseline_results.json"
                + (" (DistilBERT skipped: run_baselines.py --with_distilbert / --models distilbert not executed)"
                   if model == "distilbert" else " (run_baselines.py not executed for this model yet)")
            )
        # S4 fallback: results/metrics.json (single-run values, no std).
        if not agg and model == "roberta_eval" and metrics_json:
            flat = metrics_json.get("test_metrics") or metrics_json.get("metrics") or metrics_json
            if isinstance(flat, dict) and flat.get("precision") is not None:
                agg = {k: flat.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")}
                row["status"] = "ok (metrics.json, single run, no seed std)"
                row["reasons"]["text"] = "from results/metrics.json (single run); baseline_results.json has no roberta_eval"
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"):
            row[key] = agg.get(key)
        row["precision_ci"] = None
        comp = comparisons.get(model) if isinstance(comparisons.get(model), dict) else {}
        mcnemar = comp.get("mcnemar_vs_roberta", {}) if isinstance(comp.get("mcnemar_vs_roberta"), dict) else {}
        row["mcnemar_p_vs_roberta"] = mcnemar.get("p_value")
        ci = comp.get("bootstrap_precision_ci", {}) if isinstance(comp.get("bootstrap_precision_ci"), dict) else {}
        if ci.get("ci_low") is not None and ci.get("ci_high") is not None:
            row["precision_ci"] = {"ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
        row["params"] = agg.get("n_params")
        row["size_mb"] = agg.get("size_mb")
        row["train_seconds"] = agg.get("train_seconds")
        row["latency_cpu_ms"] = agg.get("latency_cpu_ms_median")
        row["latency_gpu_ms"] = agg.get("latency_gpu_ms_median")
        # Peak memory: recorded per seed by evaluate_baseline; aggregate does
        # not carry it, so take the max over seeds.
        peak_cpu = [r.get("peak_memory_mb", {}).get("peak_cpu_rss_mb")
                    for r in per_seed if isinstance(r.get("peak_memory_mb"), dict)]
        peak_cpu = [v for v in peak_cpu if isinstance(v, (int, float))]
        peak_cuda = [r.get("peak_memory_mb", {}).get("peak_cuda_mb")
                     for r in per_seed if isinstance(r.get("peak_memory_mb"), dict)]
        peak_cuda = [v for v in peak_cuda if isinstance(v, (int, float))]
        row["peak_memory_cpu_mb"] = max(peak_cpu) if peak_cpu else None
        row["peak_memory_cuda_mb"] = max(peak_cuda) if peak_cuda else None
        if row["peak_memory_cpu_mb"] is None:
            row["reasons"]["peak_memory"] = "no per-seed peak_memory_mb in baseline_results.json"
        # Robustness F1: first per-seed entry that carries it.
        robust = None
        for r in per_seed:
            if isinstance(r.get("robustness"), dict):
                robust = r["robustness"]
                break
        row["robustness_f1"] = (
            {k: (v or {}).get("f1") for k, v in robust.items()} if isinstance(robust, dict) else None
        )
        if row["robustness_f1"] is None:
            row["reasons"]["robustness"] = "no per-seed robustness block in baseline_results.json"
        # Explanation time per email: no script records it.
        row["explanation_time_s_per_email"] = None
        row["reasons"]["explanation_time"] = (
            "no script records explanation seconds/email (run_explain.py / run_faithfulness.py "
            "record no timing); NOT RUN YET"
        )
        # Faithfulness @5: baseline block first, then the per-model summary file.
        faith = None
        if isinstance(faith_block.get(model), dict):
            faith = _faithfulness_at_5_from_summary(faith_block[model])
        if faith is None and model == "bilstm":
            faith = _faithfulness_at_5_from_summary(inputs.get("bilstm/faithfulness.json"))
        if faith is None and model == "roberta_eval":
            faith = _faithfulness_at_5_from_summary(inputs.get("faithfulness.json"))
        row["faithfulness_at_5"] = faith
        if faith is None:
            row["reasons"]["faithfulness"] = "no faithfulness @5 found (baseline_results.json faithfulness block or faithfulness.json)"
        # Multimodal held-out (full-fusion d_weighted) precision/recall.
        mm_entry = mm.get(sys_id, {})
        row["mm_precision"] = mm_entry.get("precision")
        row["mm_recall"] = mm_entry.get("recall")
        row["mm_f1"] = mm_entry.get("f1")
        row["mm_n"] = mm_entry.get("n")
        row["mm_status"] = mm_entry.get("status", NOT_RUN)
        row["mm_mcnemar_p_vs_roberta"] = mm_entry.get("mcnemar_p_vs_roberta")
        row["mm_precision_ci"] = (
            {"ci_low": mm_entry["bootstrap_ci_low"], "ci_high": mm_entry["bootstrap_ci_high"]}
            if mm_entry.get("bootstrap_ci_low") is not None and mm_entry.get("bootstrap_ci_high") is not None
            else None
        )
        if row["mm_precision"] is None:
            row["reasons"]["multimodal"] = mm_entry.get("reason", "no multimodal_eval.json for this backbone")
        row["mm_untrained_fallback"] = bool(mm_entry.get("untrained_fallback", False))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Top-word agreement (LSTM vs RoBERTa)
# ---------------------------------------------------------------------------

def _top5_from_explanation(rec: Dict[str, Any]) -> Optional[Tuple[List[str], List[float]]]:
    """Top-5 words by |LITA score| for one explanation record.

    Prefers ``word_level_attributions`` (``lita_score`` per word); falls
    back to ``lime_top_words`` (top 15, already ranked). Words are
    lower-cased so the two models' tokenisations compare on the same terms.
    Returns ``(words, scores)`` in rank order, or None when neither field
    is present.
    """
    attrs = rec.get("word_level_attributions")
    if isinstance(attrs, list) and attrs:
        ranked = sorted(
            (a for a in attrs if isinstance(a, dict) and a.get("word")),
            key=lambda a: -abs(float(a.get("lita_score", a.get("lime_score", 0.0)))),
        )[:5]
        if ranked:
            return ([str(a["word"]).lower() for a in ranked],
                    [float(a.get("lita_score", a.get("lime_score", 0.0))) for a in ranked])
    lime_top = rec.get("lime_top_words")
    if isinstance(lime_top, list) and lime_top:
        words, scores = [], []
        for item in lime_top[:5]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                words.append(str(item[0]).lower())
                scores.append(float(item[1]))
            elif isinstance(item, dict) and item.get("word"):
                words.append(str(item["word"]).lower())
                scores.append(float(item.get("score", item.get("lime_score", 0.0))))
        if words:
            return words, scores
    return None


def _spearman(rank_a: List[float], rank_b: List[float]) -> Optional[float]:
    """Spearman rho as Pearson on ranks; None when either side is constant."""
    n = len(rank_a)
    if n < 2:
        return None
    ma, mb = sum(rank_a) / n, sum(rank_b) / n
    num = sum((a - ma) * (b - mb) for a, b in zip(rank_a, rank_b))
    den_a = math.sqrt(sum((a - ma) ** 2 for a in rank_a))
    den_b = math.sqrt(sum((b - mb) ** 2 for b in rank_b))
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def compute_agreement(
    roberta_expl: Optional[Dict[str, Any]],
    bilstm_expl: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Mean Jaccard + Spearman agreement of top-5 words, per matched email.

    Matches explanation records by ``id``. For Spearman, the union of the
    two top-5 sets is ranked in each model (a word absent from a model's
    top 5 gets rank 6); rho is the Pearson correlation of those ranks.
    Returns NOT RUN YET (with the reason) when either explanations file is
    missing, has no per-email top words, or shares no email ids.
    """
    missing = []
    if not isinstance(roberta_expl, dict):
        missing.append("RoBERTa explanations.json not found")
    if not isinstance(bilstm_expl, dict):
        missing.append("BiLSTM explanations.json not found")
    if missing:
        return {"status": NOT_RUN, "reason": "; ".join(missing), "n_emails": 0,
                "mean_jaccard": None, "mean_spearman": None}
    roberta_recs = roberta_expl.get("explanations") or []
    bilstm_recs = bilstm_expl.get("explanations") or []
    r_map = {str(r.get("id")): r for r in roberta_recs if isinstance(r, dict) and r.get("id") is not None}
    b_map = {str(r.get("id")): r for r in bilstm_recs if isinstance(r, dict) and r.get("id") is not None}
    common = sorted(set(r_map) & set(b_map))
    if not common:
        return {"status": NOT_RUN,
                "reason": "no shared email ids between the RoBERTa and BiLSTM explanations files "
                          "(run_explain.py must be run for both models with the same --num_samples)",
                "n_emails": 0, "mean_jaccard": None, "mean_spearman": None}
    jaccards, spearmans, per_email = [], [], []
    for email_id in common:
        r_top = _top5_from_explanation(r_map[email_id])
        b_top = _top5_from_explanation(b_map[email_id])
        if r_top is None or b_top is None:
            continue
        r_words, b_words = r_top[0], b_top[0]
        r_set, b_set = set(r_words), set(b_words)
        union = r_set | b_set
        jaccard = len(r_set & b_set) / len(union) if union else 0.0
        r_rank = {w: (r_words.index(w) + 1 if w in r_set else 6) for w in union}
        b_rank = {w: (b_words.index(w) + 1 if w in b_set else 6) for w in union}
        rho = _spearman([r_rank[w] for w in union], [b_rank[w] for w in union])
        jaccards.append(jaccard)
        if rho is not None:
            spearmans.append(rho)
        per_email.append({"email_id": email_id, "jaccard": jaccard, "spearman": rho,
                          "roberta_top5": r_words, "bilstm_top5": b_words})
    if not jaccards:
        return {"status": NOT_RUN,
                "reason": "explanation records carry no per-email top words "
                          "(neither word_level_attributions nor lime_top_words found)",
                "n_emails": 0, "mean_jaccard": None, "mean_spearman": None}
    return {"status": "ok", "n_emails": len(jaccards),
            "mean_jaccard": float(sum(jaccards) / len(jaccards)),
            "mean_spearman": (float(sum(spearmans) / len(spearmans)) if spearmans else None),
            "n_spearman": len(spearmans),
            "roberta_method": roberta_expl.get("model"),
            "bilstm_method": bilstm_expl.get("model"),
            "per_email": per_email}


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _check_set(simple_prec: Optional[float], s4_prec: Optional[float],
               ci: Optional[Dict[str, Any]], p_value: Optional[float]) -> Dict[str, Any]:
    """One set's three conditions: gap <= 1.0pt, CI overlap, McNemar p > 0.05.

    CI overlap uses the same definition as run_complexity_report.build_verdict:
    the simple model's bootstrap precision CI contains the RoBERTa precision.
    """
    missing: List[str] = []
    if simple_prec is None:
        missing.append("simple-model precision missing")
    if s4_prec is None:
        missing.append("RoBERTa precision missing")
    gap = None if (simple_prec is None or s4_prec is None) else (s4_prec - simple_prec) * 100.0
    ci_overlaps = None
    if ci is None or ci.get("ci_low") is None or ci.get("ci_high") is None or s4_prec is None:
        missing.append("bootstrap precision CI missing")
    else:
        ci_overlaps = bool(ci["ci_low"] <= s4_prec <= ci["ci_high"])
    if p_value is None:
        missing.append("McNemar p missing")
    passes = (gap is not None and gap <= PRECISION_TOL_POINTS
              and ci_overlaps is True and p_value is not None and p_value > 0.05)
    # "Significant" means a POSITIVE precision gap AND McNemar p < 0.05.
    # A negative gap is never a gain, and a positive gap with p >= 0.05 (or a
    # missing p) is never called significant (fixed 2026-11-19: the old rule
    # flagged gap > 1.0pt alone, p <= 0.05 alone, or a non-overlapping CI
    # alone, so a -1.30-point gap and a +5.88-point gap with p = 0.625 were
    # both labelled "significant precision gain").
    significant = (gap is not None and gap > 0
                   and p_value is not None and p_value < 0.05)
    return {"gap_points": gap, "ci_overlaps": ci_overlaps, "mcnemar_p": p_value,
            "missing": missing, "passes": bool(passes), "significant": bool(significant)}


def _cost_multiples(rows: List[Dict[str, Any]], simple_model: str) -> Tuple[str, str]:
    """Latency and parameter multiples of S4 over a simple model."""
    by_model = {r["model"]: r for r in rows}
    s4, simple = by_model.get("roberta_eval", {}), by_model.get(simple_model, {})
    lat_s4, lat_simple = _mean(s4.get("latency_cpu_ms")), _mean(simple.get("latency_cpu_ms"))
    lat = f"{lat_s4 / lat_simple:.1f}x" if lat_s4 and lat_simple else "unknown"
    p_s4, p_simple = s4.get("params"), simple.get("params")
    par = f"{p_s4 / p_simple:.1f}x" if p_s4 and p_simple else "unknown"
    return lat, par


def build_system_verdict(rows: List[Dict[str, Any]], mm: Dict[str, Dict[str, Any]]) -> str:
    """Code-generated verdict over BOTH the text test set and the multimodal held-out set.

    * No-significant-gain branch: some simple system (S1 or S2) passes all
      three conditions (gap <= 1.0 precision point, overlapping bootstrap CI,
      McNemar p > 0.05) on BOTH sets -> the Transformer is not statistically
      justified by accuracy; value is the XAI + multimodal layer; the
      passing simple system is the recommended lightweight deployment.
    * Significant-gain branch: some check shows a POSITIVE precision gap
      with McNemar p < 0.05 -> the gain size and the cost multiples
      (latency x, params x). A negative gap is reported as the Transformer
      being below the baseline (never a "gain"), and a positive but
      non-significant gap is reported neutrally as no statistically
      significant difference.
    * Otherwise (a needed statistic is missing on either set) -> the verdict
      is explicitly undecided and lists what is missing, instead of claiming
      a gain that was never measured.
    """
    by_model = {r["model"]: r for r in rows}
    s4 = by_model.get("roberta_eval", {})
    s4_prec = _mean(s4.get("precision"))
    candidates = [("tfidf_lr", "S1"), ("bilstm", "S2")]
    results: Dict[str, Dict[str, Any]] = {}
    for model, sys_id in candidates:
        row = by_model.get(model, {})
        text = _check_set(_mean(row.get("precision")), s4_prec,
                          row.get("precision_ci"), row.get("mcnemar_p_vs_roberta"))
        mm_entry = mm.get(sys_id, {})
        mm_check = _check_set(mm_entry.get("precision"), mm.get("S4", {}).get("precision"),
                              mm_entry.get("mm_precision_ci") if "mm_precision_ci" in mm_entry else
                              ({"ci_low": mm_entry.get("bootstrap_ci_low"),
                                "ci_high": mm_entry.get("bootstrap_ci_high")}
                               if mm_entry.get("bootstrap_ci_low") is not None else None),
                              mm_entry.get("mcnemar_p_vs_roberta"))
        # Untrained-fallback multimodal numbers are pipeline-demonstration
        # output, not detection evidence: they must never drive a verdict.
        if mm_entry.get("untrained_fallback") or mm.get("S4", {}).get("untrained_fallback"):
            mm_check["missing"].append(
                "multimodal numbers are UNTRAINED FALLBACK (fine-tuned checkpoint missing) — not detection evidence")
            mm_check["passes"] = False
            mm_check["significant"] = False
        results[model] = {"sys_id": sys_id, "text": text, "mm": mm_check}
    # Branch 1: a simple system passes on BOTH sets.
    passing = [m for m, r in results.items() if r["text"]["passes"] and r["mm"]["passes"]]
    if passing:
        # Highest-precision passing candidate wins.
        passing.sort(key=lambda m: -(_mean(by_model[m].get("precision")) or 0.0))
        best = passing[0]
        t, m = results[best]["text"], results[best]["mm"]
        return (
            f"The Transformer is not statistically justified by accuracy on this data; "
            f"the project's value is the XAI + multimodal layer, and {best} "
            f"({results[best]['sys_id']}) is the recommended lightweight deployment. "
            f"Evidence: on the text test set {best} is within {t['gap_points']:.2f} precision "
            f"points of RoBERTa (RoBERTa {_mean(s4.get('precision')):.4f} vs {best} "
            f"{_mean(by_model[best].get('precision')):.4f}), its 95% bootstrap precision CI "
            f"overlaps RoBERTa, and McNemar's test gives p = {t['mcnemar_p']:.4f} (> 0.05); "
            f"on the multimodal held-out set it is within {m['gap_points']:.2f} precision points "
            f"(RoBERTa {mm.get('S4', {}).get('precision')} vs {best} "
            f"{mm.get(results[best]['sys_id'], {}).get('precision')}), CI overlaps, "
            f"McNemar p = {m['mcnemar_p']:.4f} (> 0.05)."
        )
    # Branch 2: a real gain is measured somewhere (positive gap, p < 0.05);
    # otherwise a negative gap, or a non-significant difference, stated
    # plainly and never as a "significant gain".
    checks = []
    for model, r in results.items():
        for set_name, check, s4_prec_here, simple_prec_here in (
                ("text test set", r["text"], _mean(s4.get("precision")),
                 _mean(by_model[model].get("precision"))),
                ("multimodal held-out set", r["mm"], mm.get("S4", {}).get("precision"),
                 mm.get(r["sys_id"], {}).get("precision"))):
            if check["gap_points"] is None:
                continue
            # Untrained-fallback multimodal numbers are pipeline-demonstration
            # output, not detection evidence: they must never drive any verdict
            # branch (significant, negative, or neutral).
            if any("UNTRAINED FALLBACK" in m for m in check["missing"]):
                continue
            checks.append({"model": model, "sys_id": r["sys_id"], "set_name": set_name,
                           "check": check, "s4_prec": s4_prec_here,
                           "simple_prec": simple_prec_here})
    significant = [c for c in checks if c["check"]["significant"]]
    if significant:
        # First in traversal order (candidate order, text set before multimodal),
        # matching the previous reporting order.
        best = significant[0]
        lat, par = _cost_multiples(rows, best["model"])
        check = best["check"]
        p_text = f"McNemar p = {check['mcnemar_p']:.4f}" if check["mcnemar_p"] is not None else "McNemar p unavailable"
        return (
            f"Verdict (precision-first): Transformer gives a significant precision gain of "
            f"{check['gap_points']:.2f} points on the {best['set_name']} "
            f"(RoBERTa {best['s4_prec']:.4f} vs best simple baseline {best['model']} "
            f"{best['simple_prec']:.4f}; {p_text}) at {lat} latency and "
            f"{par} parameters. Whether that gain justifies the complexity depends on the "
            "deployment budget: if latency and model size allow it, RoBERTa is the precision "
            "choice; otherwise the simple baseline is the pragmatic deployment model."
        )
    negative = [c for c in checks if c["check"]["gap_points"] < 0]
    if negative:
        # First in traversal order (candidate order, text set before multimodal).
        best = negative[0]
        lat, par = _cost_multiples(rows, best["model"])
        check = best["check"]
        p_text = f"McNemar p = {check['mcnemar_p']:.4f}" if check["mcnemar_p"] is not None else "McNemar p unavailable"
        return (
            f"Verdict (precision-first): Transformer is below the best simple baseline by "
            f"{abs(check['gap_points']):.2f} precision points on the {best['set_name']} "
            f"(RoBERTa {best['s4_prec']:.4f} vs {best['model']} "
            f"{best['simple_prec']:.4f}; {p_text}) at {lat} latency and "
            f"{par} parameters. Its complexity is NOT justified by a precision gain on this "
            "dataset; value lies in the XAI/multimodal layer."
        )
    if checks:
        # Positive (or zero) gaps, none significant: neutral wording.
        best = checks[0]
        check = best["check"]
        lat, par = _cost_multiples(rows, best["model"])
        p_text = f"McNemar p = {check['mcnemar_p']:.4f}" if check["mcnemar_p"] is not None else "McNemar p unavailable"
        return (
            f"Verdict (precision-first): no statistically significant difference in precision "
            f"on the {best['set_name']} (gap {check['gap_points']:.2f} points, RoBERTa "
            f"{best['s4_prec']:.4f} vs {best['model']} {best['simple_prec']:.4f}; {p_text}) "
            f"at {lat} latency and {par} parameters. The measured difference does not support "
            "a significance claim, so the Transformer's complexity is not justified by accuracy "
            "on this set; value lies in the XAI/multimodal layer."
        )
    # Branch 3: undecided — a needed statistic is missing.
    missing_bits = []
    for model, r in results.items():
        for set_name, check in (("text test set", r["text"]), ("multimodal held-out set", r["mm"])):
            if check["missing"]:
                missing_bits.append(f"{model} on the {set_name}: {'; '.join(check['missing'])}")
    if s4_prec is None:
        extra = (" — " + "; ".join(missing_bits)) if missing_bits else ""
        return (
            f"{NOT_RUN}: no executed RoBERTa (S4) precision is available yet, so no verdict "
            "can be drawn. Run the baselines on the full splits (Kaggle GPU for RoBERTa/BiLSTM) "
            f"and re-run this report.{extra}"
        )
    return (
        f"{NOT_RUN}: the verdict cannot be decided yet — a statistic the rule needs is missing: "
        + "; ".join(missing_bits)
        + ". The multimodal held-out comparison in particular needs per-backbone McNemar p and "
          "bootstrap precision CIs (not stored by run_multimodal_eval.py today); until those are "
          "recorded, no 'significant gain' claim is made either."
    )


def recommended_deployment(rows: List[Dict[str, Any]]) -> str:
    """Highest precision subject to recall >= 0.85; tie-break lowest CPU latency.

    Mirrors ``baselines.select_deployment_model`` (same precision-first rule
    and floor) except the tie-break is lowest inference latency, as specified
    for this report, instead of fewest parameters. If no system meets the
    floor, the highest-precision system overall is named with that stated.
    """
    entries = []
    for r in rows:
        prec, rec = _mean(r.get("precision")), _mean(r.get("recall"))
        if prec is None or rec is None:
            continue
        entries.append({"model": r["model"], "system": r["system"], "precision": prec,
                        "recall": rec, "latency": _mean(r.get("latency_cpu_ms")) or float("inf")})
    if not entries:
        return NOT_RUN
    eligible = [e for e in entries if e["recall"] >= RECALL_FLOOR]
    if eligible:
        pool, floor_note = eligible, f"recall floor {RECALL_FLOOR} met"
    else:
        pool = entries
        best_rec = max(e["recall"] for e in entries)
        floor_note = (f"no system met recall floor {RECALL_FLOOR} "
                      f"(best recall among systems: {best_rec:.4f}); floor actually used: none, "
                      "highest-precision system overall")
    pool.sort(key=lambda e: (-e["precision"], e["latency"]))
    best = pool[0]
    return (f"{best['model']} ({best['system']}, precision {best['precision']:.4f}, "
            f"recall {best['recall']:.4f}; {floor_note})")


def text_only_verdict(baseline: Dict[str, Any]) -> Optional[str]:
    """The Step-H text-only verdict, reused verbatim from run_complexity_report.

    Imported lazily so this module still loads if that script is absent.
    Returns None when the import fails.
    """
    try:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from run_complexity_report import build_verdict  # noqa: E402
    except Exception as exc:  # pragma: no cover - import fallback
        logger.warning(f"Could not import run_complexity_report.build_verdict ({exc}); skipping text-only verdict.")
        return None
    summary = {
        "models": {name: (e.get("aggregate", {}) if isinstance(e, dict) else {})
                   for name, e in baseline.get("models", {}).items()},
        "comparisons_vs_roberta": baseline.get("comparisons_vs_roberta", {}),
    }
    return build_verdict(summary)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_precision_vs_latency(rows: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    """Plot 1: precision vs inference latency (CPU, median of 200) per system."""
    data = [r for r in rows if r.get("status") != NOT_RUN
            and _mean(r.get("precision")) is not None and _mean(r.get("latency_cpu_ms")) is not None]
    if not data:
        return None
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in data:
        ax.scatter([_mean(r["latency_cpu_ms"])], [_mean(r["precision"])], s=110, alpha=0.75)
        ax.annotate(r["system"], (_mean(r["latency_cpu_ms"]), _mean(r["precision"])),
                    fontsize=10, xytext=(5, 5), textcoords="offset points")
    lats = [_mean(r["latency_cpu_ms"]) for r in data]
    if max(lats) / max(min(lats), 1e-9) > 10:
        ax.set_xscale("log")
    ax.set_xlabel("Inference latency, CPU (ms/email, median of 200, batch size 1)")
    ax.set_ylabel("Test precision (phishing class)")
    ax.set_title("Precision vs inference latency per system")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_faithfulness(rows: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    """Plot 2: faithfulness @5 (comprehensiveness / sufficiency) per model."""
    data = [(r, r["faithfulness_at_5"]) for r in rows
            if isinstance(r.get("faithfulness_at_5"), dict)
            and r["faithfulness_at_5"].get("comprehensiveness") is not None]
    if not data:
        return None
    plt = _plot_setup()
    import numpy as np
    labels = [r["system"] for r, _ in data]
    comp = [f["comprehensiveness"] for _, f in data]
    suff = [f["sufficiency"] if f.get("sufficiency") is not None else 0.0 for _, f in data]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, comp, 0.4, label="Comprehensiveness @5")
    ax.bar(x + 0.2, suff, 0.4, label="Sufficiency @5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Faithfulness @5")
    ax.set_title("Faithfulness @5 per model (higher comprehensiveness, lower sufficiency = better)")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multimodal_ablation(mm: Dict[str, Dict[str, Any]], out_path: Path) -> Optional[Path]:
    """Plot 3: multimodal ablation bars (precision per variant) per backbone."""
    plt = _plot_setup()
    import numpy as np
    series = {}
    for sys_id in ("S1", "S2", "S4"):
        entry = mm.get(sys_id, {})
        ablation = entry.get("ablation") or {}
        if ablation:
            series[sys_id] = ablation
        elif entry.get("precision") is not None:
            # Only the full-fusion point is known (e.g. S1 from the baselines block).
            series[sys_id] = {"d_weighted": {"precision": entry.get("precision")}}
    if not series:
        return None
    variants = [v for v in MM_VARIANTS if any(v in s for s in series.values())]
    if not variants:
        return None
    x = np.arange(len(variants))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (sys_id, ablation) in enumerate(sorted(series.items())):
        vals = [(ablation.get(v) or {}).get("precision") for v in variants]
        vals = [v if v is not None else 0.0 for v in vals]
        ax.bar(x + (i - len(series) / 2 + 0.5) * width, vals, width, label=sys_id)
    ax.set_xticks(x)
    ax.set_xticklabels([MM_VARIANT_LABELS.get(v, v) for v in variants], rotation=15, ha="right")
    ax.set_ylabel("Held-out precision (multimodal eval set)")
    ax.set_title("Multimodal ablation: precision per fusion variant and backbone")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_agreement(agreement: Dict[str, Any], out_path: Path) -> Optional[Path]:
    """Plot 4: LSTM vs RoBERTa mean top-5 word agreement (Jaccard, Spearman)."""
    if agreement.get("status") != "ok":
        return None
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [agreement["mean_jaccard"],
            agreement["mean_spearman"] if agreement.get("mean_spearman") is not None else 0.0]
    bars = ax.bar(["Mean Jaccard\n(top-5 words)", "Mean Spearman rho\n(top-5 ranks)"], vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Agreement")
    ax.set_title(f"LSTM vs RoBERTa top-5 word agreement (n={agreement['n_emails']} emails)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

SAMPLING_NOTE = (
    "Sampling comparability (checked against the scripts, 2026-11-19). "
    "run_faithfulness.py and run_ablation.py draw their emails with "
    "test_df.sample(n=sample_size, random_state=seed) from the same test split, "
    "so the SAME --seed and SAME --sample_size give the SAME emails for RoBERTa and "
    "for BiLSTM (--model bilstm) — provided the splits are identical. The sampled "
    "email ids are verifiable post-hoc in faithfulness_records.jsonl (one record per "
    "email, with email_id); faithfulness.json / alpha_ablation.json themselves store "
    "only the summary plus metadata seed/sample_count, no per-email words. "
    "run_explain.py is DIFFERENT: it takes test_df.head(num_samples) — the first N "
    "test rows, seed-independent — so its emails are NOT the faithfulness sample. "
    "Agreement between models is still valid from explanations.json because both "
    "models explain the same head(N) emails, but only when both runs use the same "
    "--num_samples AND different --output_dir values (the default "
    "results/explanations/explanations.json would be overwritten by the second run). "
    "Per-email top words ARE stored by run_explain.py (word_level_attributions with "
    "lita_score per word, plus lime_top_words top 15, keyed by id) — so no code change "
    "is needed for agreement; the discipline is CLI-level (same --num_samples, "
    "separate --output_dir). If a future explain run stops storing per-email words, "
    "agreement falls back to NOT RUN YET."
)

def render_markdown(rows: List[Dict[str, Any]], verdict: str, recommended: str,
                    agreement: Dict[str, Any], mm: Dict[str, Dict[str, Any]],
                    inputs: Dict[str, Any], ood_csv: Optional[str],
                    plots: Dict[str, Optional[Path]], plot_dir_name: str) -> str:
    """Render ``system_complexity_report.md``."""
    lines: List[str] = []
    lines.append("# Step 6B.5: System-Complexity Report (S1-S4)")
    lines.append("")
    lines.append("Pure JSON analysis: this report reads only already-produced JSON files. "
                 "No model was trained or loaded, no GPU was used. Any cell reading "
                 f"{NOT_RUN} means its input file was absent when this report ran; "
                 "nothing was fabricated.")
    lines.append("")
    lines.append(f"**Dataset caveat:** {DATASET_CAVEAT}")
    lines.append("")
    lines.append("## Inputs read (all optional)")
    lines.append("")
    lines.append("| input file | status |")
    lines.append("|---|---|")
    status = inputs.get("_status", {})
    for rel, desc in INPUT_SPECS:
        st = status.get(rel, "missing")
        lines.append(f"| `{rel}` ({desc}) | {st} |")
    lines.append("")
    lines.append("## System table (S1-S4)")
    lines.append("")
    lines.append("Precision is the primary metric (user decision 2026-11-19); F1, recall, "
                 "ROC-AUC and FPR are reported alongside. Mean +/- std is across seeds where "
                 "the source has per-seed values; a bare number is a single run.")
    lines.append("")
    lines.append("| system | precision mean +/- std | recall mean +/- std | FPR | F1 mean +/- std | ROC-AUC | McNemar p vs S4 | bootstrap 95% CI (precision) | params | size MB | train time (s) | latency CPU (ms) | latency GPU (ms) | peak memory CPU/CUDA (MB) | explanation time/email (s) | faithfulness @5 comp/suff | robustness F1 (trunc / urls / typo) | multimodal held-out precision / recall |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") == NOT_RUN:
            reason = r["reasons"].get("text", "")
            lines.append(f"| {r['label']} | {NOT_RUN} ({reason}) | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} |")
            continue
        robust = r.get("robustness_f1")
        robust_str = (
            f"{_fmt(robust.get('truncated'))} / {_fmt(robust.get('urls_removed'))} / {_fmt(robust.get('typo_synonym'))}"
            if isinstance(robust, dict) else NOT_RUN
        )
        faith = r.get("faithfulness_at_5")
        faith_str = (
            f"{_fmt(faith.get('comprehensiveness'))} / {_fmt(faith.get('sufficiency'))}"
            f" ({faith.get('method')})"
            if isinstance(faith, dict) and faith.get("comprehensiveness") is not None else NOT_RUN
        )
        ci = r.get("precision_ci")
        ci_str = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]" if ci else NOT_RUN
        peak = (
            f"{_fmt(r.get('peak_memory_cpu_mb'), 1)} / {_fmt(r.get('peak_memory_cuda_mb'), 1)}"
            if r.get("peak_memory_cpu_mb") is not None else NOT_RUN
        )
        mm_str = (
            f"{_fmt(r.get('mm_precision'))} / {_fmt(r.get('mm_recall'))}"
            f"{' (UNTRAINED FALLBACK)' if r.get('mm_untrained_fallback') else ''}"
            if r.get("mm_precision") is not None else f"{NOT_RUN} ({r['reasons'].get('multimodal', '')})"
        )
        lines.append(
            f"| {r['label']} | {_mean_std_str(r.get('precision'))} | {_mean_std_str(r.get('recall'))} | "
            f"{_fmt(_mean(r.get('false_positive_rate')))} | {_mean_std_str(r.get('f1'))} | "
            f"{_fmt(_mean(r.get('roc_auc')))} | {_fmt(r.get('mcnemar_p_vs_roberta'))} | {ci_str} | "
            f"{_fmt(r.get('params'), 0)} | {_fmt(r.get('size_mb'), 2)} | "
            f"{_fmt(_mean(r.get('train_seconds')), 1)} | {_fmt(_mean(r.get('latency_cpu_ms')), 2)} | "
            f"{_fmt(_mean(r.get('latency_gpu_ms')), 2)} | {peak} | "
            f"{NOT_RUN} ({r['reasons'].get('explanation_time', '')}) | {faith_str} | {robust_str} | {mm_str} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(f"**Recommended deployment model:** {recommended}")
    lines.append("")
    lines.append("## Top-word agreement (LSTM vs RoBERTa)")
    lines.append("")
    if agreement.get("status") == "ok":
        lines.append(f"Matched emails: {agreement['n_emails']}. "
                     f"Mean Jaccard (top-5 word sets): {agreement['mean_jaccard']:.4f}. "
                     f"Mean Spearman rho (top-5 ranks, absent word = rank 6): "
                     f"{agreement['mean_spearman']:.4f} (n={agreement['n_spearman']}). "
                     "Top 5 per email is by |LITA score| from word_level_attributions "
                     "(lime_top_words fallback) in the two explanations.json files.")
    else:
        lines.append(f"Agreement: {NOT_RUN} — {agreement.get('reason', '')}")
    lines.append("")
    lines.append("## Sampling note")
    lines.append("")
    lines.append(SAMPLING_NOTE)
    lines.append("")
    lines.append("## Out-of-distribution check")
    lines.append("")
    if ood_csv:
        lines.append(f"OOD CSV supplied: `{ood_csv}`. No evaluated OOD result was found in the "
                     f"inputs, so the OOD test is {NOT_RUN}. (An OOD result would have to be "
                     "produced by run_baselines.py --ood_csv first and recorded in "
                     "baseline_results.json.)")
    else:
        lines.append(f"OOD test: {NOT_RUN} (no --ood_csv supplied and no OOD result in the inputs).")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for key, label in (("precision_vs_latency", "Plot 1: precision vs latency per system"),
                       ("faithfulness", "Plot 2: faithfulness @5 per model"),
                       ("multimodal_ablation", "Plot 3: multimodal ablation bars per backbone"),
                       ("agreement", "Plot 4: LSTM vs RoBERTa top-word agreement")):
        p = plots.get(key)
        if p is not None:
            lines.append(f"### {label}\n\n![{label}]({plot_dir_name}/{p.name})")
        else:
            lines.append(f"### {label}\n\n{NOT_RUN} (no data to plot yet).")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Generate the system-complexity report from already-produced JSONs."""
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir
    plot_dir = out_dir / "system_complexity_plots"
    ensure_dir(out_dir)
    ensure_dir(plot_dir)
    inputs, corrupt_errors = load_inputs(results_dir)
    if corrupt_errors:
        for err in corrupt_errors:
            logger.error(err)
            print(f"ERROR: {err}", file=sys.stderr)
        return 2
    mm = build_multimodal_by_system(inputs)
    rows = build_system_rows(inputs, mm)
    roberta_expl = inputs.get("explanations/explanations.json")
    bilstm_expl = inputs.get("bilstm/explanations/explanations.json") or inputs.get("bilstm/explanations.json")
    agreement = compute_agreement(roberta_expl, bilstm_expl)
    verdict = build_system_verdict(rows, mm)
    recommended = recommended_deployment(rows)
    baseline = inputs.get("baselines/baseline_results.json") or {}
    plots: Dict[str, Optional[Path]] = {
        "precision_vs_latency": plot_precision_vs_latency(rows, plot_dir / "precision_vs_latency.png"),
        "faithfulness": plot_faithfulness(rows, plot_dir / "faithfulness_at_5.png"),
        "multimodal_ablation": plot_multimodal_ablation(mm, plot_dir / "multimodal_ablation.png"),
        "agreement": plot_agreement(agreement, plot_dir / "top_word_agreement.png"),
    }
    md = render_markdown(rows, verdict, recommended, agreement, mm, inputs,
                         args.ood_csv, plots, plot_dir.name)
    md_path = out_dir / "system_complexity_report.md"
    md_path.write_text(md, encoding="utf-8")
    report_json = {
        "systems": rows,
        "multimodal_by_system": mm,
        "verdict": verdict,
        "recommended_deployment_model": recommended,
        "text_only_verdict": text_only_verdict(baseline),
        "agreement": agreement,
        "precision_first": True,
        "primary_metric": "phishing-class precision (user decision 2026-11-19)",
        "recall_floor": RECALL_FLOOR,
        "precision_tolerance_points": PRECISION_TOL_POINTS,
        "dataset_caveat": DATASET_CAVEAT,
        "sampling_note": SAMPLING_NOTE,
        "ood_csv": args.ood_csv,
        "ood_status": NOT_RUN,
        "inputs_status": inputs.get("_status", {}),
        "plots": {k: (str(v) if v else NOT_RUN) for k, v in plots.items()},
        "plot_dir": str(plot_dir),
    }
    save_json(report_json, out_dir / "system_complexity_report.json")
    logger.info(f"System-complexity report -> {md_path} and {out_dir / 'system_complexity_report.json'}")
    logger.info(f"Verdict: {verdict}")
    logger.info(f"Recommended deployment model: {recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
