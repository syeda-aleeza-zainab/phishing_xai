#!/usr/bin/env python3
"""Regenerate the system-complexity verdicts from the MEASURED rerun numbers.

Reads the rerun multimodal eval files (results/rerun_mm/floor_0.80 and
floor_0.85, produced by Kaggle kernel phishing-xai-rerun-multimodal-floors),
recomputes McNemar p and bootstrap precision CIs from the saved per-row
test predictions with the repo's own stats functions (baselines.mcnemar_test,
baselines.bootstrap_precision_ci; 1,000 resamples, seed 42), then calls the
FIXED verdict generator (run_system_complexity_report.build_system_verdict)
with those numbers plus the measured RoBERTa CPU latency from
results/rerun_latency/latency_explain_time.json.

No GPU, no retraining, no threshold fitting: everything here is computed
from already-executed numbers. Writes results/rerun_mm/verdicts.json.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from phishing_xai.baselines import bootstrap_precision_ci, mcnemar_test  # noqa: E402
from run_system_complexity_report import _check_set, build_system_verdict  # noqa: E402

SIMPLE_MODELS = ("tfidf_lr", "bilstm", "tfidf_svm", "rule_based")


def load_test_rows(floor: str, model: str):
    path = REPO / "results" / "rerun_mm" / floor / model / "multimodal_eval.json"
    data = json.loads(path.read_text())
    rows = [r for r in data["rows"] if r["split"] == "test"]
    tm = data["test_metrics"]["d_weighted"]
    return rows, tm


def precision(y_true, y_pred):
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    return tp / (tp + fp) if tp + fp else 0.0


def floor_verdict(floor: str, rows_text, latency_cpu_ms_roberta: float):
    """System verdict (text + multimodal) for one recall floor."""
    per_model = {}
    for model in ("roberta",) + SIMPLE_MODELS:
        rows, tm = load_test_rows(floor, model)
        y_true = [r["label"] for r in rows]
        y_pred = [r["pred_d_weighted"] for r in rows]
        per_model[model] = {
            "precision": tm["precision"],
            "recall": tm["recall"],
            "f1": tm["f1"],
            "n": len(rows),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    roberta = per_model["roberta"]
    stats = {}
    for model in SIMPLE_MODELS:
        m = per_model[model]
        mc = mcnemar_test(roberta["y_true"], roberta["y_pred"], m["y_pred"])
        ci = bootstrap_precision_ci(m["y_true"], m["y_pred"], n_resamples=1000, seed=42)
        stats[model] = {"mcnemar_p_vs_roberta": mc["p_value"], "mcnemar_b": mc["b"],
                        "mcnemar_c": mc["c"], "bootstrap_ci_low": ci["ci_low"],
                        "bootstrap_ci_high": ci["ci_high"]}
    mm = {
        "S4": {"status": "ok", "precision": roberta["precision"], "recall": roberta["recall"],
               "f1": roberta["f1"], "n": roberta["n"], "untrained_fallback": False},
        "S1": {"status": "ok", "precision": per_model["tfidf_lr"]["precision"],
               "recall": per_model["tfidf_lr"]["recall"], "f1": per_model["tfidf_lr"]["f1"],
               "n": per_model["tfidf_lr"]["n"], "untrained_fallback": False,
               "mcnemar_p_vs_roberta": stats["tfidf_lr"]["mcnemar_p_vs_roberta"],
               "bootstrap_ci_low": stats["tfidf_lr"]["bootstrap_ci_low"],
               "bootstrap_ci_high": stats["tfidf_lr"]["bootstrap_ci_high"]},
        "S2": {"status": "ok", "precision": per_model["bilstm"]["precision"],
               "recall": per_model["bilstm"]["recall"], "f1": per_model["bilstm"]["f1"],
               "n": per_model["bilstm"]["n"], "untrained_fallback": False,
               "mcnemar_p_vs_roberta": stats["bilstm"]["mcnemar_p_vs_roberta"],
               "bootstrap_ci_low": stats["bilstm"]["bootstrap_ci_low"],
               "bootstrap_ci_high": stats["bilstm"]["bootstrap_ci_high"]},
    }
    # Text rows: unchanged from the item-1 regeneration, except RoBERTa's CPU
    # latency is now the MEASURED median (item 3) instead of unknown.
    rows = json.loads((REPO / "results" / "system_complexity_report.json").read_text())["systems"]
    for r in rows:
        if r["model"] == "roberta_eval":
            r["latency_cpu_ms"] = {"mean": latency_cpu_ms_roberta, "std": 0.0, "n": 1}
    verdict = build_system_verdict(rows, mm)
    # Standalone multimodal check vs the best simple baseline on this set:
    # highest weighted-fusion precision among the simple backbones.
    best_simple = max(SIMPLE_MODELS, key=lambda m: per_model[m]["precision"])
    mm_check = _check_set(per_model[best_simple]["precision"], roberta["precision"],
                          {"ci_low": stats[best_simple]["bootstrap_ci_low"],
                           "ci_high": stats[best_simple]["bootstrap_ci_high"]},
                          stats[best_simple]["mcnemar_p_vs_roberta"])
    return {"floor": floor, "verdict": verdict, "mm_check": mm_check,
            "best_simple_multimodal": best_simple, "stats": stats,
            "multimodal": {m: {k: per_model[m][k] for k in ("precision", "recall", "f1", "n")}
                           for m in per_model}}


def main() -> None:
    latency = json.loads((REPO / "results" / "rerun_latency" / "latency_explain_time.json").read_text())
    roberta_cpu_ms = latency["latency_ms_per_email"]["roberta_cpu"]["latency_ms_per_email_median"]
    rows_text = None  # text rows come from the saved system report inside floor_verdict
    out = {}
    for floor in ("floor_0.80", "floor_0.85"):
        out[floor] = floor_verdict(floor, rows_text, roberta_cpu_ms)
        print(f"===== {floor} =====")
        print(out[floor]["verdict"])
        print(f"best simple baseline (multimodal, weighted fusion): "
              f"{out[floor]['best_simple_multimodal']}")
        print(f"multimodal check vs best simple: {out[floor]['mm_check']}")
        print()
    (REPO / "results" / "rerun_mm" / "verdicts.json").write_text(json.dumps(out, indent=2))
    print("Wrote results/rerun_mm/verdicts.json")


if __name__ == "__main__":
    main()
