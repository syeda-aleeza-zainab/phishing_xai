"""CLI: generate the Section H complexity-justification report.

Reads ``<results_dir>/baselines/baseline_results.json`` (written by
``scripts/run_baselines.py``) and produces::

    results/complexity_report.md
    results/complexity_report.json
    results/complexity_f1_vs_latency.png   (Plot 1, if any real numbers exist)
    results/complexity_f1_vs_train_time.png (Plot 2, if any real numbers exist)

The verdict paragraph is written BY CODE from the numbers
(:func:`build_verdict`), with both branches implemented and unit-tested.
When no executed baseline numbers exist, every table cell reads
``NOT RUN YET`` and the verdict says so plainly; nothing is invented.

Usage::

    python scripts/run_complexity_report.py --results_dir results
    python scripts/run_complexity_report.py --results_dir results --ood_csv data/ood.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root / src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.baselines import DATASET_CAVEAT, select_deployment_model
from phishing_xai.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("run_complexity_report")

NOT_RUN = "NOT RUN YET"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the complexity report."""
    parser = argparse.ArgumentParser(description="Generate Section H complexity report")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory holding baselines/baseline_results.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <results_dir>)")
    parser.add_argument("--ood_csv", type=str, default=None,
                        help="Optional second-dataset CSV already evaluated by run_baselines.py; noted in the report, never downloaded")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode (same output, smaller plots)")
    return parser.parse_args(argv)


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
    if not isinstance(entry, dict) or entry.get("mean") is None:
        return NOT_RUN
    mean, std = entry["mean"], entry.get("std")
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return NOT_RUN
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def build_verdict(summary: Dict[str, Any]) -> str:
    """Rule-based verdict paragraph, written by code from executed numbers.

    PRECISION-FIRST (user decision 2026-11-19): the headline judgement is on
    phishing-class PRECISION gain, with the F1-based branch logic kept as a
    secondary statement.

    * If the best simple baseline (highest precision subject to recall >= 0.85)
      is within 1.0 PRECISION point of RoBERTa, its bootstrap precision CI
      overlaps RoBERTa's precision, and McNemar p > 0.05 -> the transformer
      gives no statistically significant precision gain; complexity is NOT
      justified by accuracy alone, and its value lies in the XAI / multimodal
      layer.
    * Otherwise -> the transformer gives a significant precision gain of X
      points at Yx latency and Zx parameters (with the F1 gap as secondary).

    ``summary`` maps model name -> aggregate dict as produced by
    ``aggregate_seed_results`` plus ``comparisons_vs_roberta`` entries.
    With no real numbers, returns the NOT RUN YET verdict.
    """
    models = summary.get("models", {})
    simple_aggs = {n: e for n, e in models.items() if n != "roberta_eval"}
    roberta = models.get("roberta_eval", {}) if isinstance(models, dict) else {}

    def _mean(entry: Any, key: str) -> Optional[float]:
        if not isinstance(entry, dict):
            return None
        val = entry.get(key)
        if isinstance(val, dict):
            return val.get("mean")
        return val if isinstance(val, (int, float)) else None

    roberta_prec = _mean(roberta, "precision")
    roberta_f1 = _mean(roberta, "f1")
    selection = select_deployment_model(simple_aggs, recall_floor=0.85)
    best_simple = selection.get("model")
    best_prec = selection.get("precision")
    best_f1 = _mean(simple_aggs.get(best_simple, {}), "f1") if best_simple else None
    if roberta_prec is None or best_prec is None or best_simple is None:
        return (
            f"{NOT_RUN}: no executed baseline or RoBERTa precision numbers are "
            "available yet, so no verdict can be drawn. Run "
            "scripts/run_baselines.py on the full splits (Kaggle GPU for "
            "RoBERTa/BiLSTM) and re-run this report."
        )
    gap_points = (roberta_prec - best_prec) * 100.0
    comparisons = summary.get("comparisons_vs_roberta", {})
    comp = comparisons.get(best_simple, {}) if isinstance(comparisons, dict) else {}
    mcnemar = comp.get("mcnemar_vs_roberta", {}) if isinstance(comp, dict) else {}
    p_value = mcnemar.get("p_value") if isinstance(mcnemar, dict) else None
    ci_prec = comp.get("bootstrap_precision_ci", {}) if isinstance(comp, dict) else {}
    ci_low, ci_high = ci_prec.get("ci_low"), ci_prec.get("ci_high")
    ci_overlaps = (
        ci_low is not None and ci_high is not None and ci_low <= roberta_prec <= ci_high
    )
    f1_gap = (
        f" (F1 gap: {(roberta_f1 - best_f1) * 100.0:.2f} points, RoBERTa {roberta_f1:.4f} vs {best_simple} {best_f1:.4f})"
        if roberta_f1 is not None and best_f1 is not None else ""
    )
    if gap_points <= 1.0 and ci_overlaps and p_value is not None and p_value > 0.05:
        return (
            f"Verdict (precision-first): the best simple baseline ({best_simple}, "
            f"precision {best_prec:.4f}, recall {selection.get('recall'):.4f}) is within "
            f"{gap_points:.2f} precision points of RoBERTa (precision {roberta_prec:.4f}), "
            f"its 95% bootstrap precision CI [{ci_low:.4f}, {ci_high:.4f}] overlaps RoBERTa, "
            f"and McNemar's test gives p = {p_value:.4f} (> 0.05). Transformer gives no "
            "statistically significant precision gain on this dataset, so its complexity "
            "is NOT justified by accuracy alone. Value lies in the XAI/multimodal layer: "
            "LIME / Integrated Gradients / LITA explanations and image/PDF/QR handling "
            f"that the simple baselines do not provide.{f1_gap}"
        )
    # Significant-gain branch: quantify the cost of the gain.
    roberta_lat = _latency_of(models.get("roberta_eval", {}))
    simple_lat = _latency_of(models.get(best_simple, {}))
    latency_ratio = f"{roberta_lat / simple_lat:.1f}x" if roberta_lat and simple_lat else "unknown"
    roberta_params = models.get("roberta_eval", {}).get("n_params")
    simple_params = models.get(best_simple, {}).get("n_params")
    param_ratio = (
        f"{roberta_params / simple_params:.1f}x"
        if roberta_params and simple_params else "unknown"
    )
    p_text = f"McNemar p = {p_value:.4f}" if p_value is not None else "McNemar p unavailable"
    return (
        f"Verdict (precision-first): Transformer gives a significant precision gain of "
        f"{gap_points:.2f} points (RoBERTa {roberta_prec:.4f} vs best simple baseline "
        f"{best_simple} {best_prec:.4f}; {p_text}) at {latency_ratio} latency and "
        f"{param_ratio} parameters.{f1_gap} Whether that gain justifies the complexity "
        "depends on the deployment budget: if latency and model size allow it, RoBERTa "
        "is the precision choice; otherwise the simple baseline is the pragmatic "
        "deployment model."
    )


def _latency_of(entry: Dict[str, Any], key: str = "latency_cpu_ms_median") -> Optional[float]:
    """Median latency (ms/email) from an aggregate entry, if present."""
    lat = entry.get(key)
    if isinstance(lat, dict):
        return lat.get("mean")
    return None


def recommended_deployment_model(summary: Dict[str, Any]) -> str:
    """Highest-precision model subject to a recall floor of 0.85.

    Precision-first rule (user decision 2026-11-19). If no model meets
    recall >= 0.85, the floor actually used (the chosen model's recall) is
    stated explicitly in the returned string. Returns NOT RUN YET when no
    executed numbers exist.
    """
    selection = select_deployment_model(summary.get("models", {}), recall_floor=0.85)
    if selection.get("model") is None:
        return NOT_RUN
    floor_note = (
        f"recall floor 0.85 met (recall {selection['recall']:.4f})"
        if selection["floor_met"]
        else f"no model met recall floor 0.85 (best recall among models: "
             f"{selection['floor_used']:.4f}); floor actually used: none, highest-precision model overall"
    )
    return (
        f"{selection['model']} (precision {selection['precision']:.4f}, "
        f"recall {selection['recall']:.4f}; {floor_note})"
    )


def build_table_rows(baseline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build one table row per model from ``baseline_results.json`` content.

    Rows are ranked by precision first (mean precision descending); models
    with no executed numbers sort last with status NOT RUN YET.
    """
    models = baseline_results.get("models", {})
    comparisons = baseline_results.get("comparisons_vs_roberta", {})
    faithfulness = baseline_results.get("faithfulness", {})
    rows: List[Dict[str, Any]] = []
    for name, entry in models.items():
        agg = entry.get("aggregate", {}) if isinstance(entry, dict) else {}
        per_seed = entry.get("per_seed", []) if isinstance(entry, dict) else []
        ok = [r for r in per_seed if isinstance(r, dict) and r.get("status") != NOT_RUN and "metrics" in r]
        if not ok and not agg:
            rows.append({"model": name, "status": NOT_RUN})
            continue
        comp = comparisons.get(name, {}) if isinstance(comparisons, dict) else {}
        mcnemar = comp.get("mcnemar_vs_roberta", {}) if isinstance(comp, dict) else {}
        robust: Dict[str, Any] = {}
        threshold_val: Optional[Dict[str, Any]] = None
        for r in ok:
            if isinstance(r.get("robustness"), dict):
                robust = r["robustness"]
            if isinstance(r.get("threshold_val"), dict) and r["threshold_val"].get("threshold") is not None:
                threshold_val = r["threshold_val"]
            if robust and threshold_val:
                break
        faith = faithfulness.get(name, {}) if isinstance(faithfulness, dict) else {}
        faith_5 = None
        if isinstance(faith, dict):
            faith_5 = {
                "comp": faith.get("comprehensiveness", {}).get("5"),
                "suff": faith.get("sufficiency", {}).get("5"),
            }
        rows.append({
            "model": name,
            "status": "ok",
            "params": agg.get("n_params"),
            "size_mb": agg.get("size_mb"),
            "train_seconds": (agg.get("train_seconds") or {}).get("mean") if isinstance(agg.get("train_seconds"), dict) else None,
            "latency_cpu_ms": _latency_of(agg),
            "latency_gpu_ms": _latency_of(agg, "latency_gpu_ms_median"),
            "accuracy": agg.get("accuracy"),
            "precision": agg.get("precision"),
            "recall": agg.get("recall"),
            "f1": agg.get("f1"),
            "roc_auc": agg.get("roc_auc"),
            "precision_at_recall_0.90": agg.get("precision_at_recall_0.90"),
            "threshold_val": threshold_val,
            "mcnemar_p_vs_roberta": mcnemar.get("p_value") if isinstance(mcnemar, dict) else None,
            "robustness_f1": {k: v.get("f1") for k, v in robust.items()} if robust else None,
            "faithfulness_comp_suff_at_5": faith_5,
        })
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    not_run_rows = [r for r in rows if r.get("status") != "ok"]

    def _prec_mean(row: Dict[str, Any]) -> float:
        prec = row.get("precision")
        if isinstance(prec, dict) and prec.get("mean") is not None:
            return float(prec["mean"])
        return -1.0

    ok_rows.sort(key=_prec_mean, reverse=True)
    return ok_rows + not_run_rows


def plot_f1_vs_latency(rows: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    """Plot 1: F1 vs inference latency (log-x), bubble size = parameters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = [r for r in rows if r.get("status") == "ok" and r.get("f1") and r.get("latency_cpu_ms")]
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in data:
        f1 = r["f1"]["mean"] if isinstance(r["f1"], dict) else r["f1"]
        params = r.get("params") or 1
        size = 40 + 300 * (np.log10(max(params, 1)) / np.log10(125_000_000))
        ax.scatter([r["latency_cpu_ms"]], [f1], s=size, alpha=0.7)
        ax.annotate(r["model"], (r["latency_cpu_ms"], f1), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency, CPU (ms/email, median, batch size 1)")
    ax.set_ylabel("Test F1 (phishing class)")
    ax.set_title("F1 vs inference latency (bubble size = parameters)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_f1_vs_train_time(rows: List[Dict[str, Any]], out_path: Path) -> Optional[Path]:
    """Plot 2: F1 vs training time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [r for r in rows if r.get("status") == "ok" and r.get("f1") and r.get("train_seconds")]
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in data:
        f1 = r["f1"]["mean"] if isinstance(r["f1"], dict) else r["f1"]
        ax.scatter([r["train_seconds"]], [f1], s=90, alpha=0.7)
        ax.annotate(r["model"], (r["train_seconds"], f1), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Training wall-clock (seconds)")
    ax.set_ylabel("Test F1 (phishing class)")
    ax.set_title("F1 vs training time")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_markdown(
    rows: List[Dict[str, Any]],
    verdict: str,
    recommended: str,
    baseline_results: Dict[str, Any],
    ood_csv: Optional[str],
    plot1: Optional[Path],
    plot2: Optional[Path],
) -> str:
    """Render ``complexity_report.md`` from the table rows and verdict."""
    lines: List[str] = []
    lines.append("# Section H: Baseline and Complexity-Justification Report")
    lines.append("")
    lines.append("Baselines use the identical splits (``results/splits.json``), the same "
                 "``clean_text`` cleaning, train-only augmentation, and a small random "
                 "search (~8 trials) scored on VALIDATION only. The test set was touched "
                 "once per model for the final evaluation. RoBERTa is eval-only here.")
    lines.append("")
    lines.append(f"**Dataset caveat:** {DATASET_CAVEAT}")
    lines.append("")
    if baseline_results.get("untrained_fallback"):
        lines.append("> **Note:** RoBERTa weights were missing when the baselines ran "
                     "(``untrained_fallback: true``); any RoBERTa cell below marked "
                     "NOT RUN YET was not executed and must be filled on Kaggle with the checkpoint.")
        lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append("Models are ranked by phishing-class PRECISION first (mean precision "
                 "descending); recall, F1 and accuracy are reported alongside and are "
                 "never hidden. Precision @ recall 0.90 is the precision achieved at the "
                 "threshold fit on VALIDATION for target recall 0.90; that threshold and "
                 "its validation precision/recall are recorded per model in "
                 "``baseline_results.json`` (``threshold_val``).")
    lines.append("")
    lines.append("| model | precision mean +/- std | recall mean +/- std | F1 mean +/- std | precision @ recall 0.90 | accuracy | ROC-AUC | params | size MB | train time (s) | latency CPU (ms) | latency GPU (ms) | McNemar p vs RoBERTa | robustness F1 (trunc / urls / typo) | faithfulness comp/suff @5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("status") == NOT_RUN:
            lines.append(f"| {r['model']} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} | {NOT_RUN} |")
            continue
        robust = r.get("robustness_f1")
        robust_str = (
            f"{_fmt(robust.get('truncated'))} / {_fmt(robust.get('urls_removed'))} / {_fmt(robust.get('typo_synonym'))}"
            if isinstance(robust, dict) else NOT_RUN
        )
        faith = r.get("faithfulness_comp_suff_at_5")
        faith_str = (
            f"{_fmt(faith.get('comp'))} / {_fmt(faith.get('suff'))}"
            if isinstance(faith, dict) and faith.get("comp") is not None else NOT_RUN
        )
        lines.append(
            f"| {r['model']} | {_mean_std_str(r.get('precision'))} | {_mean_std_str(r.get('recall'))} | "
            f"{_mean_std_str(r.get('f1'))} | {_mean_std_str(r.get('precision_at_recall_0.90'))} | "
            f"{_fmt((r.get('accuracy') or {}).get('mean') if isinstance(r.get('accuracy'), dict) else r.get('accuracy'))} | "
            f"{_fmt((r.get('roc_auc') or {}).get('mean') if isinstance(r.get('roc_auc'), dict) else r.get('roc_auc'))} | "
            f"{_fmt(r.get('params'), 0)} | {_fmt(r.get('size_mb'), 2)} | "
            f"{_fmt(r.get('train_seconds'), 1)} | {_fmt(r.get('latency_cpu_ms'), 2)} | "
            f"{_fmt(r.get('latency_gpu_ms'), 2)} | "
            f"{_fmt(r.get('mcnemar_p_vs_roberta'))} | {robust_str} | {faith_str} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(f"**Recommended deployment model:** {recommended}")
    lines.append("")
    lines.append("## Explainability")
    lines.append("")
    faith = baseline_results.get("faithfulness", {})
    if isinstance(faith, dict) and faith:
        for name, res in faith.items():
            if isinstance(res, dict) and res.get("status") != NOT_RUN:
                lines.append(
                    f"- {name} ({res.get('method')}): comprehensiveness @5 "
                    f"{_fmt(res.get('comprehensiveness', {}).get('5'))}, sufficiency @5 "
                    f"{_fmt(res.get('sufficiency', {}).get('5'))} (n={res.get('n_evaluated')})."
                )
            else:
                lines.append(f"- {name}: {NOT_RUN} ({res.get('reason', '') if isinstance(res, dict) else ''})")
    else:
        lines.append(f"Explainability comparison: {NOT_RUN} (re-run run_baselines.py with --faithfulness).")
    lines.append("")
    lines.append("Whether RoBERTa explanations are more faithful than the baselines can only "
                 "be judged from the executed numbers above; where a cell reads NOT RUN YET, "
                 "that comparison is NOT RUN YET.")
    lines.append("")
    if ood_csv:
        lines.append(f"## Out-of-distribution check\n\nOOD dataset supplied: ``{ood_csv}`` "
                     "(evaluated by run_baselines.py --ood_csv; see baseline_results.json).")
        lines.append("")
    if plot1 is not None:
        lines.append(f"## Plot 1: F1 vs inference latency\n\n![F1 vs latency]({plot1.name})")
        lines.append("")
    if plot2 is not None:
        lines.append(f"## Plot 2: F1 vs training time\n\n![F1 vs training time]({plot2.name})")
        lines.append("")
    if plot1 is None and plot2 is None:
        lines.append(f"Plots: {NOT_RUN} (no executed numbers to plot yet).")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Generate the complexity report from executed baseline results."""
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir
    ensure_dir(out_dir)
    baselines_path = results_dir / "baselines" / "baseline_results.json"
    if baselines_path.exists():
        baseline_results = load_json(baselines_path)
    else:
        logger.warning(f"No baseline results at {baselines_path}; report will be all NOT RUN YET.")
        baseline_results = {"status": NOT_RUN, "models": {}, "untrained_fallback": True}
    rows = build_table_rows(baseline_results)
    # Verdict input: aggregates + comparisons in the shape build_verdict expects.
    summary: Dict[str, Any] = {
        "models": {
            name: (entry.get("aggregate", {}) if isinstance(entry, dict) else {})
            for name, entry in baseline_results.get("models", {}).items()
        },
        "comparisons_vs_roberta": baseline_results.get("comparisons_vs_roberta", {}),
    }
    verdict = build_verdict(summary)
    recommended = recommended_deployment_model(summary)
    plot1 = plot_f1_vs_latency(rows, out_dir / "complexity_f1_vs_latency.png")
    plot2 = plot_f1_vs_train_time(rows, out_dir / "complexity_f1_vs_train_time.png")
    md = render_markdown(rows, verdict, recommended, baseline_results, args.ood_csv, plot1, plot2)
    md_path = out_dir / "complexity_report.md"
    md_path.write_text(md, encoding="utf-8")
    report_json = {
        "table": rows,
        "verdict": verdict,
        "recommended_deployment_model": recommended,
        "precision_first": True,
        "primary_metric": "phishing-class precision (user decision 2026-11-19)",
        "recall_floor": 0.85,
        "dataset_caveat": DATASET_CAVEAT,
        "untrained_fallback": bool(baseline_results.get("untrained_fallback", True)),
        "plots": {
            "f1_vs_latency": str(plot1) if plot1 else NOT_RUN,
            "f1_vs_train_time": str(plot2) if plot2 else NOT_RUN,
        },
    }
    save_json(report_json, out_dir / "complexity_report.json")
    logger.info(f"Complexity report -> {md_path} and {out_dir / 'complexity_report.json'}")
    logger.info(f"Verdict: {verdict}")
    logger.info(f"Recommended deployment model: {recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
