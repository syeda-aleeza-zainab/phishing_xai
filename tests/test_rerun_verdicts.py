"""Tests for scripts/regen_rerun_verdicts.py (fix-and-rerun round, 2026-11-19).

The regen script recomputes McNemar p and bootstrap precision CIs from the
saved per-row rerun predictions (results/rerun_mm/floor_0.80 and floor_0.85,
executed on Kaggle) with the repo's own stats functions, then calls the FIXED
verdict generator. These tests assert the recomputed statistics against the
values recorded in results/rerun_mm/verdicts.json and assert the fixed rule
on the rerun numbers: a zero gap with p = 1.0 is never "significant", and the
old bug (a +5.88-point gap with p = 0.625 labelled a significant gain) is
gone. No model loading, no GPU.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_SCRIPT = REPO / "scripts" / "regen_rerun_verdicts.py"
_SPEC = importlib.util.spec_from_file_location("regen_rerun_verdicts", _SCRIPT)
regen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(regen)

from phishing_xai.baselines import bootstrap_precision_ci, mcnemar_test  # noqa: E402
from run_system_complexity_report import _check_set  # noqa: E402

VERDICTS = json.loads((REPO / "results" / "rerun_mm" / "verdicts.json").read_text())


def _test_rows(floor, model):
    data = json.loads((REPO / "results" / "rerun_mm" / floor / model / "multimodal_eval.json").read_text())
    rows = [r for r in data["rows"] if r["split"] == "test"]
    assert len(rows) == 96
    return [r["label"] for r in rows], [r["pred_d_weighted"] for r in rows]


@pytest.mark.parametrize("floor", ["floor_0.80", "floor_0.85"])
def test_rerun_mcnemar_matches_recorded_verdicts(floor):
    """McNemar p recomputed from the saved rows equals verdicts.json."""
    y_true, roberta_pred = _test_rows(floor, "roberta")
    for model in ("tfidf_lr", "bilstm", "tfidf_svm", "rule_based"):
        _, pred = _test_rows(floor, model)
        mc = mcnemar_test(y_true, roberta_pred, pred)
        assert mc["p_value"] == pytest.approx(
            VERDICTS[floor]["stats"][model]["mcnemar_p_vs_roberta"], abs=1e-12
        )


@pytest.mark.parametrize("floor", ["floor_0.80", "floor_0.85"])
def test_rerun_bootstrap_ci_matches_recorded_verdicts(floor):
    """Bootstrap precision CI recomputed from the saved rows equals verdicts.json."""
    for model in ("tfidf_lr", "bilstm", "tfidf_svm", "rule_based"):
        y_true, pred = _test_rows(floor, model)
        ci = bootstrap_precision_ci(y_true, pred, n_resamples=1000, seed=42)
        assert ci["ci_low"] == pytest.approx(
            VERDICTS[floor]["stats"][model]["bootstrap_ci_low"], abs=1e-12
        )
        assert ci["ci_high"] == pytest.approx(
            VERDICTS[floor]["stats"][model]["bootstrap_ci_high"], abs=1e-12
        )


@pytest.mark.parametrize("floor", ["floor_0.80", "floor_0.85"])
def test_multimodal_best_simple_is_bilstm_with_zero_gap(floor):
    """Best simple baseline on the rerun multimodal set is BiLSTM at P 1.00:
    zero gap vs RoBERTa, p = 1.0, CI overlaps, never significant."""
    check = VERDICTS[floor]["mm_check"]
    assert VERDICTS[floor]["best_simple_multimodal"] == "bilstm"
    assert check["gap_points"] == pytest.approx(0.0, abs=1e-9)
    assert check["mcnemar_p"] == pytest.approx(1.0, abs=1e-12)
    assert check["ci_overlaps"] is True
    assert check["significant"] is False
    assert check["passes"] is True


def test_old_buggy_gain_would_not_be_significant():
    """The old bug: a +5.88-point gap with McNemar p = 0.625 was called a
    significant gain. Under the fixed rule it is not significant."""
    check = _check_set(0.9412, 1.0, {"ci_low": 0.8718, "ci_high": 1.0}, 0.625)
    assert check["gap_points"] == pytest.approx(5.88, abs=0.01)
    assert check["significant"] is False


@pytest.mark.parametrize("floor", ["floor_0.80", "floor_0.85"])
def test_system_verdict_is_the_negative_text_verdict(floor):
    """The fixed generator's system verdict names the below-baseline text
    result (0.11 points, p = 0.3491) with the measured 223.7x latency."""
    verdict = VERDICTS[floor]["verdict"]
    assert "below the best simple baseline by 0.11 precision points" in verdict
    assert "McNemar p = 0.3491" in verdict
    assert "223.7x latency" in verdict
    assert "significant precision gain" not in verdict


def test_floor_085_only_rule_based_moved():
    """At floor 0.85 the learned backbones are unchanged; only rule-based
    moved (precision 1.00 -> 0.75)."""
    for model in ("roberta", "bilstm", "tfidf_lr", "tfidf_svm"):
        assert VERDICTS["floor_0.80"]["multimodal"][model] == VERDICTS["floor_0.85"]["multimodal"][model]
    assert VERDICTS["floor_0.80"]["multimodal"]["rule_based"]["precision"] == pytest.approx(1.0)
    assert VERDICTS["floor_0.85"]["multimodal"]["rule_based"]["precision"] == pytest.approx(0.75)
