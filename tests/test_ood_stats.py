"""Unit tests for the OOD stats script (tests/test_ood_stats.py).

CPU-only, no GPU, no checkpoints, no real predictions: all fixtures are
tiny and hand-built. The repo stats helpers are NOT imported here (they
need torch); the tests exercise the local mirrors, which copy the repo
formulas, and the table builder's NOT RUN YET behaviour.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "experiments" / "modern_phishing_ood" / "scripts"))

import compute_ood_stats as cs


def test_mcnemar_mirror_counts_and_exact_p():
    # A wrong 8x, B wrong 2x, both wrong/right elsewhere. b+c = 10 < 25
    # -> exact binomial p.
    y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])
    pa = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])  # A wrong on 10
    pb = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])  # B wrong on 2
    r = cs.mcnemar_mirror(y, pa, pb)
    assert r["b"] == 8  # A wrong, B right
    assert r["c"] == 0  # A right, B wrong
    assert r["statistic"] is None  # exact path
    assert 0.0 < r["p_value"] < 0.05


def test_mcnemar_mirror_no_discordant_pairs_is_one():
    y = np.array([1, 0, 1, 0])
    assert cs.mcnemar_mirror(y, y, y)["p_value"] == 1.0


def test_mcnemar_mirror_chi2_path_large_n():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=200)
    pa = y.copy()
    pa[:40] = 1 - pa[:40]  # A wrong on 40
    pb = y.copy()
    pb[40:50] = 1 - pb[40:50]  # B wrong on 10, disjoint from A's errors
    r = cs.mcnemar_mirror(y, pa, pb)
    assert r["b"] == 40 and r["c"] == 10
    assert r["statistic"] is not None
    assert r["statistic"] == pytest.approx((abs(40 - 10) - 1) ** 2 / 50)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, size=200)
    p = (y * 0.7 + rng.rand(200) * 0.3 > 0.5).astype(int)
    r = cs.bootstrap_precision_ci_mirror(y, p, n_resamples=200, seed=42)
    assert r["ci_low"] <= r["precision"] <= r["ci_high"]
    assert r["n_resamples"] == 200
    r2 = cs.bootstrap_recall_ci(y, p, n_resamples=200, seed=42)
    assert r2["ci_low"] <= r2["recall"] <= r2["ci_high"]
    r3 = cs.bootstrap_f1_ci_mirror(y, p, n_resamples=200, seed=42)
    assert r3["ci_low"] <= r3["f1"] <= r3["ci_high"]


def test_bootstrap_auc_ci_reasonable():
    rng = np.random.RandomState(2)
    y = np.array([1] * 50 + [0] * 50)
    p = np.concatenate([rng.uniform(0.6, 1.0, 50), rng.uniform(0.0, 0.4, 50)])
    r = cs.bootstrap_auc_ci(y, p, n_resamples=200, seed=42)
    assert r["roc_auc"] == pytest.approx(1.0)
    assert r["roc_auc_ci_low"] <= 1.0 <= r["roc_auc_ci_high"]
    assert 0.0 < r["pr_auc"] <= 1.0


def test_paired_diff_ci_zero_for_identical_models():
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.4, 0.6, 0.3])
    d = cs.paired_diff_ci(y, p, p, n_resamples=100, seed=42)
    for metric in ("precision", "recall", "f1"):
        assert d[metric]["diff_A_minus_B"] == 0.0
        assert d[metric]["ci_low"] == 0.0 and d[metric]["ci_high"] == 0.0


def test_align_pair_inner_joins_on_email_id():
    a = pd.DataFrame({"email_id": ["e1", "e2", "e3"], "prob_phishing": [0.9, 0.2, 0.8],
                      "true_label": [1, 0, 1]})
    b = pd.DataFrame({"email_id": ["e2", "e3", "e4"], "prob_phishing": [0.1, 0.7, 0.5],
                      "true_label": [0, 1, 0]})
    m = cs.align_pair(a, b)
    assert m["email_id"].tolist() == ["e2", "e3"]
    with pytest.raises(ValueError):
        cs.align_pair(a, pd.DataFrame({"email_id": ["zz"], "prob_phishing": [0.5],
                                       "true_label": [0]}))


def test_build_tables_marks_missing_cells_not_run_yet(tmp_path):
    tables = cs.build_tables(tmp_path, {"per_set": {}})
    assert tables["A_original_vs_modern"]["rows"]["roberta"]["modern_ood_phishnchips"]["precision"]["value"] == "NOT RUN YET"
    assert tables["B_human_vs_ai"]["rows"]["roberta"]["llm_generated"]["recall_at_0_5"]["value"] == "NOT RUN YET"
    assert tables["C_temporal"]["rows"]["status"]["value"] == "NOT RUN YET"
    assert tables["D_robustness"]["rows"]["roberta"]["status"]["value"] == "NOT RUN YET"
    assert tables["F_efficiency"]["rows"]["roberta"]["status"]["value"] == "NOT RUN YET"
    # In-distribution side of Table A is always present (quoted).
    assert tables["A_original_vs_modern"]["rows"]["roberta"]["in_distribution_test"]["precision"]["value"] == 0.9690


def test_build_tables_reads_metrics_json(tmp_path):
    (tmp_path / "phishnchips_roberta_metrics.json").write_text(
        __import__("json").dumps({
            "metrics_at_0_5": {"precision": 0.5, "recall": 0.6, "f1": 0.55,
                               "accuracy": 0.7, "roc_auc": 0.8, "n": 2000},
            "latency_ms_per_email": {"gpu": {"latency_ms_per_email_median": 15.0,
                                             "n_timed": 200},
                                     "cpu": {"latency_ms_per_email_median": 240.0,
                                             "n_timed": 100},
                                     "texts_source": "phishnchips"},
            "model_info": {"n_params": 124647170, "size_mb": 476.0},
        }), encoding="utf-8")
    tables = cs.build_tables(tmp_path, {"per_set": {}})
    cell = tables["A_original_vs_modern"]["rows"]["roberta"]["modern_ood_phishnchips"]["precision"]
    assert cell["value"] == 0.5
    assert "phishnchips_roberta_metrics.json" in cell["source"]
    f = tables["F_efficiency"]["rows"]["roberta"]
    assert f["latency_gpu_ms"]["value"] == 15.0
    assert f["n_params"]["value"] == 124647170


# --- Expanded comparison (Task 7 + Task 8) ---------------------------------

def _load_expanded():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compute_ood_stats_expanded",
        Path(__file__).resolve().parent.parent
        / "experiments" / "modern_phishing_ood" / "scripts"
        / "compute_ood_stats_expanded.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_holm_adjust_known_values():
    """Holm: sorted p (0.01, 0.04, 0.05), m=3 -> (0.03, 0.08, 0.08)."""
    mod = _load_expanded()
    adj = mod.holm_adjust([0.01, 0.04, 0.05])
    assert adj == pytest.approx([0.03, 0.08, 0.08])
    # Monotone non-decreasing in sorted order; capped at 1.0.
    assert mod.holm_adjust([0.6, 0.7]) == pytest.approx([1.0, 1.0])
    assert mod.holm_adjust([]) == []
    assert mod.holm_adjust([0.0359]) == pytest.approx([0.0359])


def test_paired_auc_diff_ci_zero_for_identical_models():
    """Identical probabilities -> diff 0 and a CI containing 0."""
    mod = _load_expanded()
    rng = np.random.RandomState(0)
    y = np.array([1] * 50 + [0] * 50)
    p = rng.rand(100)
    out = mod.paired_auc_diff_ci(y, p, p, n_resamples=200, seed=42)
    assert out["roc_auc"]["diff_A_minus_B"] == pytest.approx(0.0)
    assert out["roc_auc"]["ci_low"] <= 0.0 <= out["roc_auc"]["ci_high"]
    assert out["roc_auc"]["significant"] is False


def test_compare_pair_mcnemar_is_errors_only():
    """Two models with identical precision but different errors: McNemar
    can be significant while the precision difference is 0."""
    mod = _load_expanded()
    df = pd.DataFrame({
        "email_id": [f"e{i}" for i in range(100)],
        "true_label": [1] * 50 + [0] * 50,
        # A: TP=40, FN=10, FP=0 -> precision 1.0, recall 0.80.
        # B: TP=45, FN=5,  FP=0 -> precision 1.0, recall 0.90.
        # Same precision, different errors: A-only-wrong = 5 rows.
        "true_label_a": [1] * 50 + [0] * 50,
        "prob_phishing_a": [0.9] * 40 + [0.1] * 10 + [0.1] * 50,
        "true_label_b": [1] * 50 + [0] * 50,
        "prob_phishing_b": [0.9] * 45 + [0.1] * 5 + [0.1] * 50,
    })
    comp = mod.compare_pair(df, "a", "b", n_resamples=200, seed=42)
    assert comp["n_paired"] == 100
    assert comp["mcnemar"]["b"] + comp["mcnemar"]["c"] > 0
    assert comp["paired_metric_diff_ci"]["precision"]["diff_A_minus_B"] == pytest.approx(0.0)


def test_load_predictions_parses_multitoken_model_names(tmp_path):
    """Regression: phishnchips_tfidf_lr -> set phishnchips, model tfidf_lr
    (the old last-token split read model 'lr', set 'phishnchips_tfidf')."""
    import compute_ood_stats as cs
    for name in ["phishnchips_roberta_predictions.csv",
                 "phishnchips_tfidf_svm_predictions.csv",
                 "frontiers_llm_tfidf_lr_predictions.csv",
                 "bec_poisoned_bilstm_predictions.csv"]:
        pd.DataFrame({"email_id": ["e1"], "true_label": [1],
                      "prob_phishing": [0.9]}).to_csv(tmp_path / name, index=False)
    preds = cs.load_predictions(tmp_path)
    assert sorted(preds) == ["bec_poisoned__bilstm",
                             "frontiers_llm__tfidf_lr",
                             "phishnchips__roberta",
                             "phishnchips__tfidf_svm"]
