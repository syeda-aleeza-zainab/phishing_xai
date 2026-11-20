"""Tests for scripts/run_system_complexity_report.py (Step 6B.5).

All inputs are SYNTHETIC JSONs written into tmp_path; no real results,
no model loading, no GPU. The script under test reads only JSON files.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_system_complexity_report.py"
_SPEC = importlib.util.spec_from_file_location("run_system_complexity_report", _SCRIPT)
run_system_complexity_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_system_complexity_report)

main = run_system_complexity_report.main
NOT_RUN = run_system_complexity_report.NOT_RUN


def _agg(precision, recall, f1=0.95, roc_auc=0.99, fpr=0.03,
         params=1_000_000, size_mb=4.0, train_s=60.0, lat_cpu=2.0, lat_gpu=None):
    agg = {
        "precision": {"mean": precision, "std": 0.005, "n": 3},
        "recall": {"mean": recall, "std": 0.005, "n": 3},
        "f1": {"mean": f1, "std": 0.005, "n": 3},
        "accuracy": {"mean": 0.97, "std": 0.005, "n": 3},
        "roc_auc": {"mean": roc_auc, "std": 0.005, "n": 3},
        "false_positive_rate": {"mean": fpr, "std": 0.005, "n": 3},
        "n_params": params,
        "size_mb": size_mb,
        "train_seconds": {"mean": train_s, "std": 1.0, "n": 3},
        "latency_cpu_ms_median": {"mean": lat_cpu, "std": 0.1, "n": 3},
    }
    if lat_gpu is not None:
        agg["latency_gpu_ms_median"] = {"mean": lat_gpu, "std": 0.1, "n": 3}
    return agg


def _per_seed(seed=42, f1=0.95):
    return [{
        "seed": seed,
        "metrics": {"precision": 0.97, "recall": 0.93, "f1": f1},
        "latency_cpu_ms": {"latency_ms_per_email_median": 2.0},
        "peak_memory_mb": {"peak_cpu_rss_mb": 512.0, "peak_cuda_mb": None},
        "robustness": {"truncated": {"f1": 0.90}, "urls_removed": {"f1": 0.91},
                      "typo_synonym": {"f1": 0.89}},
    }]


def _baseline_results(s1_prec=0.970, s4_prec=0.975, s1_p=0.40, s1_ci=(0.960, 0.985),
                      s2_prec=0.965, s2_p=0.30, s2_ci=(0.955, 0.980),
                      mm_s1_prec=0.940, mm_s1_p=0.30, mm_s1_ci=(0.930, 0.960),
                      s0_prec=0.982, s0_p=0.253, s0_ci=(0.9746, 0.9889)):
    return {
        "models": {
            "tfidf_lr": {"aggregate": _agg(s1_prec, 0.93, params=500_000, size_mb=8.0,
                                           train_s=30.0, lat_cpu=2.0),
                         "per_seed": _per_seed()},
            "bilstm": {"aggregate": _agg(s2_prec, 0.92, params=5_000_000, size_mb=20.0,
                                         train_s=600.0, lat_cpu=15.0),
                       "per_seed": _per_seed()},
            "distilbert": {"aggregate": _agg(0.972, 0.93, params=66_000_000, size_mb=260.0,
                                             train_s=1800.0, lat_cpu=40.0, lat_gpu=5.0),
                           "per_seed": _per_seed()},
            "roberta_eval": {"aggregate": _agg(s4_prec, 0.94, params=125_000_000,
                                               size_mb=500.0, train_s=3600.0,
                                               lat_cpu=50.0, lat_gpu=8.0),
                             "per_seed": _per_seed()},
            "tfidf_svm": {"aggregate": _agg(s0_prec, 0.99, params=500_000, size_mb=8.0,
                                            train_s=30.0, lat_cpu=2.0),
                          "per_seed": _per_seed()},
        },
        "comparisons_vs_roberta": {
            "tfidf_lr": {"mcnemar_vs_roberta": {"p_value": s1_p},
                         "bootstrap_precision_ci": {"ci_low": s1_ci[0], "ci_high": s1_ci[1]}},
            "bilstm": {"mcnemar_vs_roberta": {"p_value": s2_p},
                       "bootstrap_precision_ci": {"ci_low": s2_ci[0], "ci_high": s2_ci[1]}},
            "distilbert": {"mcnemar_vs_roberta": {"p_value": 0.60},
                           "bootstrap_precision_ci": {"ci_low": 0.960, "ci_high": 0.985}},
            "tfidf_svm": {"mcnemar_vs_roberta": {"p_value": s0_p},
                          "bootstrap_precision_ci": {"ci_low": s0_ci[0], "ci_high": s0_ci[1]}},
        },
        "faithfulness": {
            "tfidf_lr": {"comprehensiveness": {"5": 0.30}, "sufficiency": {"5": 0.10}},
            "bilstm": {"comprehensiveness": {"5": 0.35}, "sufficiency": {"5": 0.08}},
            "roberta_eval": {"comprehensiveness": {"5": 0.45}, "sufficiency": {"5": 0.05}},
        },
        "multimodal": {
            "tfidf_lr": {"precision": mm_s1_prec, "recall": 0.90,
                         "mcnemar_vs_roberta": {"p_value": mm_s1_p},
                         "bootstrap_precision_ci": {"ci_low": mm_s1_ci[0], "ci_high": mm_s1_ci[1]}},
        },
    }


def _mm_eval(backbone_prec=0.945, backbone="roberta", with_stats=False):
    data = {
        "metadata": {"models": [backbone], "seed": 42, "n_test": 96},
        "test_metrics": {
            "a": {"precision": 0.50, "recall": 1.0, "f1": 0.67, "n": 96},
            "b": {"precision": 0.55, "recall": 0.95, "f1": 0.70, "n": 96},
            "c": {"precision": 0.60, "recall": 0.90, "f1": 0.72, "n": 96},
            "d_weighted": {"precision": backbone_prec, "recall": 0.85, "f1": 0.89,
                           "false_positive_rate": 0.02, "n": 96},
            "d_max": {"precision": 0.56, "recall": 0.67, "f1": 0.61, "n": 96},
        },
    }
    if backbone == "roberta":
        data["baselines"] = {"tfidf_lr": {"status": "ok", "n": 120, "precision": 0.94,
                                          "recall": 0.90, "f1": 0.92, "accuracy": 0.92,
                                          "false_positive_rate": 0.05}}
    if with_stats:
        data["mcnemar_vs_roberta"] = {"p_value": 0.35}
        data["bootstrap_precision_ci"] = {"ci_low": 0.930, "ci_high": 0.960}
    return data


def _explanations(model, tops):
    """tops: {email_id: [(word, lita_score), ...]} in rank order."""
    recs = []
    for email_id, words in tops.items():
        recs.append({
            "id": email_id,
            "model": model,
            "explanation_method": "lita" if model == "roberta" else "lime_ig",
            "word_level_attributions": [
                {"word_idx": i, "word": w, "ti_score": 0.1, "lime_score": s, "lita_score": s}
                for i, (w, s) in enumerate(words)
            ],
            "lime_top_words": [[w, s] for w, s in words],
        })
    return {"explanations": recs, "model": model}


_ROBERTA_TOPS = {
    "email_1": [("urgent", 0.9), ("verify", 0.8), ("account", 0.7), ("click", 0.6), ("password", 0.5)],
    "email_2": [("invoice", 0.9), ("payment", 0.8), ("due", 0.7), ("click", 0.6), ("link", 0.5)],
    "email_3": [("security", 0.9), ("alert", 0.8), ("verify", 0.7), ("account", 0.6), ("now", 0.5)],
}
_BILSTM_TOPS = {
    "email_1": [("verify", 0.9), ("account", 0.8), ("urgent", 0.7), ("login", 0.6), ("click", 0.5)],
    "email_2": [("payment", 0.9), ("invoice", 0.8), ("due", 0.7), ("link", 0.6), ("now", 0.5)],
    "email_3": [("alert", 0.9), ("security", 0.8), ("account", 0.7), ("verify", 0.6), ("update", 0.5)],
}


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_all_inputs(results: Path, **overrides) -> None:
    """Write the full synthetic input set; overrides replace single files."""
    _write(results / "baselines" / "baseline_results.json",
           overrides.get("baseline_results", _baseline_results()))
    _write(results / "multimodal_eval" / "multimodal_eval.json",
           overrides.get("mm_roberta", _mm_eval(0.945, "roberta")))
    _write(results / "bilstm" / "multimodal_eval.json",
           overrides.get("mm_bilstm", _mm_eval(0.940, "bilstm", with_stats=True)))
    _write(results / "explanations" / "explanations.json",
           overrides.get("expl_roberta", _explanations("roberta", _ROBERTA_TOPS)))
    _write(results / "bilstm" / "explanations" / "explanations.json",
           overrides.get("expl_bilstm", _explanations("bilstm", _BILSTM_TOPS)))
    _write(results / "faithfulness.json",
           {"summary": {"lita_magnitude": {"comprehensiveness_mean": {"5": 0.45},
                                            "sufficiency_mean": {"5": 0.05}}}})
    _write(results / "bilstm" / "faithfulness.json",
           {"summary": {"lita_magnitude": {"comprehensiveness_mean": {"5": 0.35},
                                            "sufficiency_mean": {"5": 0.08}}}})


def test_all_inputs_present(tmp_path):
    """(a) All inputs present: S1-S4 + S0 table, best-baseline verdict, recommended line, 4 plots."""
    results = tmp_path / "results"
    _write_all_inputs(results)
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    md = (results / "system_complexity_report.md").read_text(encoding="utf-8")
    report = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))
    for label in ("S1 TF-IDF+LR", "S2 BiLSTM", "S3 DistilBERT", "S4 RoBERTa",
                  "S0 TF-IDF+SVC"):
        assert label in md
    # Headline verdict is vs the best simple baseline (SVC 0.982 > RoBERTa 0.975).
    assert "below the best simple baseline" in md
    assert "TF-IDF+SVC" in report["verdict"]
    assert "GPU latency multiple: NOT MEASURED" in report["verdict"]
    # The older LR-based verdict is kept for traceability.
    assert "not statistically justified" in report["verdict_vs_tfidf_lr"]
    assert "tfidf_lr" in report["verdict_vs_tfidf_lr"]
    assert "Recommended deployment model:" in md
    assert "tfidf_svm" in report["recommended_deployment_model"]  # highest precision, recall >= 0.85
    plot_dir = results / "system_complexity_plots"
    for name in ("precision_vs_latency.png", "faithfulness_at_5.png",
                 "multimodal_ablation.png", "top_word_agreement.png"):
        assert (plot_dir / name).exists(), f"missing plot {name}"
    # Agreement: email_1 Jaccard 4/6; mean over 3 emails is (4/6 + 4/6 + 4/6)/3.
    assert report["agreement"]["status"] == "ok"
    assert report["agreement"]["n_emails"] == 3
    assert report["agreement"]["mean_jaccard"] == pytest.approx(4 / 6)
    assert 0.0 < report["agreement"]["mean_spearman"] <= 1.0
    assert report["agreement"]["roberta_method"] == "roberta"
    assert report["agreement"]["bilstm_method"] == "bilstm"


def test_no_significant_gain_names_s1(tmp_path):
    """(b) S1 within 1.0pt + overlapping CI + p>0.05 on both sets -> names S1 (LR verdict)."""
    results = tmp_path / "results"
    _write_all_inputs(results)
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    md = (results / "system_complexity_report.md").read_text(encoding="utf-8")
    report = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))
    verdict = report["verdict_vs_tfidf_lr"]
    assert "The Transformer is not statistically justified by accuracy on this data" in verdict
    assert "tfidf_lr" in verdict
    assert "recommended lightweight deployment" in verdict
    assert "The Transformer is not statistically justified by accuracy on this data" in md


def test_significant_gain_reports_size_and_cost(tmp_path):
    """(c) S4 significantly better -> gain size + latency/params multiples (vs SVC)."""
    results = tmp_path / "results"
    _write_all_inputs(results, baseline_results=_baseline_results(
        s1_prec=0.90, s4_prec=0.99, s1_p=0.001, s1_ci=(0.88, 0.92),
        s2_prec=0.89, s2_p=0.001, s2_ci=(0.87, 0.91),
        mm_s1_prec=0.80, mm_s1_p=0.001, mm_s1_ci=(0.78, 0.82),
        s0_prec=0.90, s0_p=0.001, s0_ci=(0.88, 0.92)))
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    verdict = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))["verdict"]
    assert "significant precision gain" in verdict
    assert "9.00 precision points" in verdict  # (0.99 - 0.90) * 100
    assert "25.0x CPU latency" in verdict  # 50 ms / 2 ms
    assert "250.0x parameters" in verdict  # 125M / 500k
    assert "GPU latency multiple: NOT MEASURED" in verdict


def test_missing_inputs_not_run_yet_exit_0(tmp_path):
    """(d) No inputs at all -> NOT RUN YET markers, exit 0, no crash."""
    results = tmp_path / "results"
    results.mkdir()
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    md = (results / "system_complexity_report.md").read_text(encoding="utf-8")
    report = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))
    assert NOT_RUN in md
    assert NOT_RUN in report["verdict"]
    assert report["recommended_deployment_model"] == NOT_RUN
    assert report["agreement"]["status"] == NOT_RUN
    assert not (results / "system_complexity_plots" / "precision_vs_latency.png").exists()


def test_corrupt_json_exits_nonzero(tmp_path):
    """Corrupt present-but-unreadable JSON -> non-zero exit (not silent NOT RUN YET)."""
    results = tmp_path / "results"
    (results / "baselines").mkdir(parents=True)
    (results / "baselines" / "baseline_results.json").write_text("{not json", encoding="utf-8")
    rc = main(["--results_dir", str(results)])
    assert rc != 0


def test_agreement_missing_explanations(tmp_path):
    """No explanations files -> agreement NOT RUN YET with a reason, exit 0."""
    results = tmp_path / "results"
    _write(results / "baselines" / "baseline_results.json", _baseline_results())
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    report = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))
    assert report["agreement"]["status"] == NOT_RUN
    assert "explanations.json not found" in report["agreement"]["reason"]


def test_mm_only_significant_gain_no_text_s4(tmp_path):
    """Gain measured only on the multimodal set (no text S4 precision) must not crash.

    Regression: the gain branch used to format the text-set RoBERTa precision
    even when the multimodal set triggered it (TypeError on None).
    Rule change 2026-11-19: a positive gap with NO McNemar p stored must NOT
    be called a "significant precision gain" (significance needs gap > 0 AND
    p < 0.05); the verdict is neutral instead.
    """
    results = tmp_path / "results"
    _write(results / "multimodal_eval" / "multimodal_eval.json", _mm_eval(1.0, "roberta"))
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    # Headline verdict needs text precision (absent here); the LR-based
    # verdict carries the multimodal-only logic.
    verdict = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))["verdict_vs_tfidf_lr"]
    assert "significant precision gain" not in verdict
    assert "no statistically significant difference" in verdict
    assert "multimodal held-out set" in verdict
    assert "6.00 points" in verdict  # (1.00 - 0.94) * 100, S1 baselines block vs S4 d_weighted
    assert "McNemar p unavailable" in verdict


def test_untrained_fallback_mm_never_drives_verdict(tmp_path):
    """Multimodal numbers flagged untrained_fallback must not produce a gain verdict."""
    results = tmp_path / "results"
    mm_data = _mm_eval(1.0, "roberta")
    mm_data["untrained_fallback"] = True
    _write(results / "multimodal_eval" / "multimodal_eval.json", mm_data)
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    verdict = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))["verdict_vs_tfidf_lr"]
    assert "significant precision gain" not in verdict
    assert "UNTRAINED FALLBACK" in verdict


# --- Verdict wording rules (fixed 2026-11-19) --------------------------------
# "Significant gain" requires a POSITIVE precision gap AND McNemar p < 0.05.
# A negative gap is "below the baseline"; a positive non-significant gap is
# neutral ("no statistically significant difference"); p exactly 0.05 is not
# significant. Direct unit tests on build_system_verdict / _check_set.

def _verdict_rows(s1_prec, s2_prec, s4_prec, s1_p, s2_p,
                 s1_ci=(0.960, 0.985), s2_ci=(0.955, 0.980)):
    """Minimal rows in the shape build_system_verdict reads."""
    def _row(model, prec, p, ci, lat=2.0, params=500_000):
        return {"model": model, "system": model, "precision": {"mean": prec},
                "recall": {"mean": 0.93}, "latency_cpu_ms": {"mean": lat},
                "params": params, "precision_ci": {"ci_low": ci[0], "ci_high": ci[1]},
                "mcnemar_p_vs_roberta": p}
    return [
        _row("tfidf_lr", s1_prec, s1_p, s1_ci),
        _row("bilstm", s2_prec, s2_p, s2_ci, lat=15.0, params=5_000_000),
        _row("roberta_eval", s4_prec, None, (0.960, 0.985), lat=50.0,
             params=125_000_000),
    ]


def _verdict_mm(s1_prec, s2_prec, s4_prec, s1_p, s2_p,
                s1_ci=(0.930, 0.960), s2_ci=(0.930, 0.960)):
    """Minimal multimodal dict in the shape build_system_verdict reads."""
    def _entry(prec, p, ci):
        return {"precision": prec, "mcnemar_p_vs_roberta": p,
                "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1]}
    return {"S1": _entry(s1_prec, s1_p, s1_ci),
            "S2": _entry(s2_prec, s2_p, s2_ci),
            "S4": {"precision": s4_prec}}


def test_check_set_significant_needs_positive_gap_and_small_p():
    """_check_set: significant only when gap > 0 AND p < 0.05."""
    check_set = run_system_complexity_report._check_set
    ci = {"ci_low": 0.90, "ci_high": 0.99}
    assert check_set(0.90, 0.99, ci, 0.001)["significant"] is True   # +9.0pt, p<0.05
    assert check_set(0.94, 1.00, ci, 0.625)["significant"] is False  # +6.0pt, p>=0.05
    assert check_set(0.982, 0.969, ci, 0.001)["significant"] is False  # -1.3pt, p<0.05
    assert check_set(0.94, 0.99, ci, 0.05)["significant"] is False   # p exactly 0.05
    assert check_set(0.94, 0.99, ci, None)["significant"] is False   # p missing


def test_system_verdict_positive_significant_gain():
    """Positive gap + p < 0.05 -> 'significant precision gain' with the size."""
    rows = _verdict_rows(0.90, 0.92, 0.99, 0.001, 0.001)
    mm = _verdict_mm(0.95, 0.96, 0.99, 0.001, 0.002)
    verdict = run_system_complexity_report.build_system_verdict(rows, mm)
    assert "significant precision gain of 9.00 points" in verdict
    assert "text test set" in verdict


def test_system_verdict_positive_nonsignificant_neutral():
    """Positive gap + p = 0.625 -> neutral wording, never 'significant gain'."""
    rows = _verdict_rows(0.9412, 0.95, 1.00, 0.625, 0.50,
                         s1_ci=(0.930, 0.960), s2_ci=(0.930, 0.960))
    mm = _verdict_mm(0.9412, 0.95, 1.00, 0.625, 0.50)
    verdict = run_system_complexity_report.build_system_verdict(rows, mm)
    assert "significant precision gain" not in verdict
    assert "no statistically significant difference" in verdict
    assert "5.88 points" in verdict


def test_system_verdict_negative_gap_below_baseline():
    """RoBERTa below the simple baselines -> 'below the baseline by X points'."""
    rows = _verdict_rows(0.9820, 0.9719, 0.9690, 0.253, 0.0002,
                         s1_ci=(0.9746, 0.9889), s2_ci=(0.9613, 0.9789))
    mm = _verdict_mm(0.9412, 1.00, 1.00, 0.625, 1.00)
    verdict = run_system_complexity_report.build_system_verdict(rows, mm)
    assert "significant precision gain" not in verdict
    assert "below the best simple baseline by 1.30 precision points" in verdict
    assert "text test set" in verdict


def test_system_verdict_p_exactly_boundary_not_significant():
    """McNemar p exactly 0.05 is NOT < 0.05 -> neutral, never 'significant gain'."""
    rows = _verdict_rows(0.94, 0.95, 0.99, 0.05, 0.10)
    mm = _verdict_mm(0.96, 0.97, 0.99, 0.20, 0.30)
    verdict = run_system_complexity_report.build_system_verdict(rows, mm)
    assert "significant precision gain" not in verdict
    assert "no statistically significant difference" in verdict


# --- Cross-load guard + best-baseline verdict (fixed 2026-11-20) -----------

def test_agreement_refuses_cross_loaded_roberta_file(tmp_path):
    """A RoBERTa explanations file declaring model 'bilstm' is refused,
    never reported as agreement (regression: the 2026-11-19 report showed
    Jaccard/Spearman 1.0000 from exactly this mix-up)."""
    results = tmp_path / "results"
    _write_all_inputs(results, expl_roberta=_explanations("bilstm", _BILSTM_TOPS))
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    report = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))
    assert report["agreement"]["status"] == NOT_RUN
    assert "cross-loaded" in report["agreement"]["reason"]
    assert report["agreement"]["mean_jaccard"] is None


def test_agreement_refuses_identical_files(tmp_path):
    """Byte-identical RoBERTa/BiLSTM explanation records are refused: one
    run overwrote the other's file, so no cross-model comparison exists."""
    results = tmp_path / "results"
    _write_all_inputs(results, expl_roberta=_explanations("roberta", _BILSTM_TOPS))
    rc = main(["--results_dir", str(results)])
    assert rc == 0
    report = json.loads((results / "system_complexity_report.json").read_text(encoding="utf-8"))
    assert report["agreement"]["status"] == NOT_RUN
    assert "identical" in report["agreement"]["reason"]


def _best_verdict_rows():
    """Rows in the shape build_best_baseline_verdict reads (real numbers)."""
    def _row(model, prec, rec, f1, roc, lat_cpu, lat_gpu, params, size,
             p_vs_roberta=None, expl=None, rob=None, mm_p=None, mm_r=None,
             mm_p_value=None):
        return {"model": model, "precision": {"mean": prec},
                "recall": {"mean": rec}, "f1": {"mean": f1},
                "roc_auc": {"mean": roc},
                "latency_cpu_ms": {"mean": lat_cpu} if lat_cpu is not None else None,
                "latency_gpu_ms": {"mean": lat_gpu} if lat_gpu is not None else None,
                "params": params, "size_mb": size,
                "mcnemar_p_vs_roberta": p_vs_roberta,
                "explanation_time_s_per_email": ({"median": expl} if expl is not None else None),
                "robustness_f1": rob,
                "mm_precision": mm_p, "mm_recall": mm_r,
                "mm_mcnemar_p_vs_roberta": mm_p_value}
    return [
        _row("tfidf_svm", 0.9820, 0.9892, 0.9856, 0.9989, 8.47, None, 3_972_261, 54.81,
             p_vs_roberta=0.2530, expl=None,
             rob={"clean": 0.9845, "truncated": 0.9726, "urls_removed": 0.9849,
                  "typo_synonym": 0.9835},
             mm_p=0.9556, mm_r=0.8958, mm_p_value=0.0703),
        _row("bilstm", 0.9719, 0.9428, 0.9571, 0.9942, 19.90, 8.37, 6_236_034, 23.79,
             p_vs_roberta=0.0002, expl=1.87,
             rob={"clean": 0.9658, "truncated": 0.9658, "urls_removed": 0.9654,
                  "typo_synonym": 0.9637},
             mm_p=1.0, mm_r=1.0, mm_p_value=1.0),
        _row("roberta_eval", 0.9690, 0.9932, 0.9809, 0.9986, 242.50, 15.50,
             124_647_170, 475.57, expl=30.87,
             rob={"clean": 0.9809, "truncated": 0.9773, "urls_removed": 0.9755,
                  "typo_synonym": 0.9638},
             mm_p=1.0, mm_r=0.9792),
    ]


def test_best_baseline_verdict_below_svc_with_explicit_multiples():
    """RoBERTa 1.30 points below SVC (p=0.253): CPU multiple explicit,
    GPU multiple NOT MEASURED, never fabricated."""
    rows = _best_verdict_rows()
    inputs = {"paired_mcnemar_seed42.json": {
        "pairs": {"tfidf_svm_vs_bilstm": {"b": 16, "c": 65,
                                          "p_exact_two_sided": 3.64e-08}}}}
    verdict, details = run_system_complexity_report.build_best_baseline_verdict(
        rows, {}, inputs)
    assert "below the best simple baseline" in verdict
    assert "1.30 precision points" in verdict
    assert "McNemar p = 0.2530" in verdict
    assert "28.6x CPU latency" in verdict  # 242.50 / 8.47
    assert "31.4x parameters" in verdict  # 124647170 / 3972261
    assert "GPU latency multiple: NOT MEASURED" in verdict
    assert "significant precision gain" not in verdict
    assert details["roberta_vs_svc"]["gpu_latency_multiple"] is None
    assert details["roberta_vs_svc"]["cpu_latency_multiple"] == pytest.approx(242.50 / 8.47, rel=1e-3)
    assert details["bilstm_vs_svc"]["cpu_latency_multiple"] == pytest.approx(19.90 / 8.47, rel=1e-3)
    assert details["bilstm_vs_svc"]["mcnemar_p_vs_svc"] == pytest.approx(3.64e-08, rel=1e-2)
    assert "SVC is significantly better than BiLSTM" in verdict
