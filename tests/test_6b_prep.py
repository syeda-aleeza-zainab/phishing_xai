"""Tests for the 6B.2/6B.4 prep (2026-11-19).

(a) ``select_best_bilstm_trial`` picks by the precision / recall-floor rule
    on fabricated trial rows — no real model is trained here.
(b) ``RuleBasedScorer`` returns (n, 2) rows summing to 1 and its scores do
    not depend on body words (only on the URLs in the text); ``load_scorer``
    builds it with no data, no fit and no checkpoint.
(c) The bigger BiLSTM config (emb_dim 200, hidden 256, layers 2) builds and
    predicts in a tiny 1-epoch CPU run.

All tests are fast (no real dataset, no GPU, no downloads).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.baselines import (
    RuleBasedScorer,
    TorchTrainConfig,
    build_torch_baseline,
    load_scorer,
    select_best_bilstm_trial,
)

SUSPICIOUS_URL = "http://123.45.67.89/verify"


def _tiny_frames():
    """Tiny train/val frames (same shape as tests/test_scorer_adapter.py)."""
    train = pd.DataFrame({
        "clean_text": [
            "verify your password now please",
            "verify your password now urgent",
            "verify your password now today",
            "team meeting notes for tomorrow",
            "team meeting notes for today",
            "team meeting notes for review",
        ],
        "label": [1, 1, 1, 0, 0, 0],
    })
    val = pd.DataFrame({
        "clean_text": [
            "verify your password now please",
            "team meeting notes for tomorrow",
        ],
        "label": [1, 0],
    })
    return {"train": train, "val": val}


def _trial(params, precision, recall, f1):
    return {"params": params, "val_precision": precision,
            "val_recall": recall, "val_f1": f1}


# ---------------------------------------------------------------------------
# (a) Tuning selection rule on fabricated trial rows (no training).
# ---------------------------------------------------------------------------
def test_select_best_trial_prefers_precision_among_floor_meeters():
    trials = [
        _trial({"hidden": 64}, 0.90, 0.86, 0.88),   # meets floor, lower precision
        _trial({"hidden": 128}, 0.97, 0.90, 0.93),  # meets floor, best precision
        _trial({"hidden": 256}, 0.99, 0.80, 0.89),  # higher precision but BELOW floor
    ]
    sel = select_best_bilstm_trial(trials, recall_floor=0.85)
    assert sel["floor_met"] is True
    assert sel["best_params"] == {"hidden": 128}
    assert sel["best_val_precision"] == pytest.approx(0.97)
    assert sel["best_val_recall"] == pytest.approx(0.90)
    assert sel["n_eligible"] == 2


def test_select_best_trial_no_floor_met_picks_highest_recall_then_precision():
    trials = [
        _trial({"hidden": 64}, 0.99, 0.50, 0.66),   # best precision, low recall
        _trial({"hidden": 128}, 0.80, 0.70, 0.75),  # best recall
        _trial({"hidden": 256}, 0.85, 0.70, 0.77),  # tied recall, better precision
    ]
    sel = select_best_bilstm_trial(trials, recall_floor=0.85)
    assert sel["floor_met"] is False
    # Highest recall (0.70, tied) then highest precision wins: hidden 256.
    assert sel["best_params"] == {"hidden": 256}
    assert sel["best_val_recall"] == pytest.approx(0.70)
    assert sel["best_val_precision"] == pytest.approx(0.85)
    assert sel["n_eligible"] == 0


def test_select_best_trial_empty_raises():
    with pytest.raises(ValueError):
        select_best_bilstm_trial([], recall_floor=0.85)


# ---------------------------------------------------------------------------
# (b) Rule-based scorer: shape, row sums, body-word independence.
# ---------------------------------------------------------------------------
def test_rule_based_scorer_shape_rows_sum_and_no_url_is_zero():
    scorer = RuleBasedScorer()
    assert scorer.name == "rule_based"
    probs = scorer.predict_proba([
        f"please verify now {SUSPICIOUS_URL}",
        "hello team meeting notes",
        "",
    ])
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3))
    # IP-literal http URL carries static risk (> 0); plain words score 0.
    assert probs[0, 1] > 0.0
    assert probs[1, 1] == pytest.approx(0.0)
    assert probs[2, 1] == pytest.approx(0.0)
    assert scorer.predict_proba([]).shape == (0, 2)


def test_rule_based_scorer_ignores_body_words():
    scorer = RuleBasedScorer()
    # Same URL, completely different body words -> identical scores.
    a = scorer.predict_proba([f"hello team meeting notes {SUSPICIOUS_URL}"])[0]
    b = scorer.predict_proba([f"urgent password reset now {SUSPICIOUS_URL}"])[0]
    np.testing.assert_allclose(a, b)
    # Worst-URL rule: adding a benign URL does not lower the score.
    c = scorer.predict_proba(
        [f"hello team {SUSPICIOUS_URL} see also https://example.com/login"]
    )[0]
    np.testing.assert_allclose(a, c)


def test_load_scorer_rule_based_needs_no_data_no_fit(monkeypatch):
    import phishing_xai.baselines as bl

    def _boom(*args, **kwargs):
        raise AssertionError("rule_based must not touch the dataset/splits")

    monkeypatch.setattr(bl, "load_splits", _boom)
    scorer = load_scorer("rule_based")
    assert isinstance(scorer, RuleBasedScorer)
    assert scorer.name == "rule_based"
    probs = scorer.predict_proba([f"click here {SUSPICIOUS_URL}"])
    assert probs.shape == (1, 2)
    np.testing.assert_allclose(probs.sum(axis=1), [1.0])


def test_score_email_with_rule_based_ignores_body_words():
    from scripts.run_multimodal_eval import score_email

    def _extraction(body_text):
        return {
            "body_text": body_text,
            "parts": [
                {"kind": "subject", "text": "team update"},
                {"kind": "body_text", "text": body_text},
            ],
            "links": [SUSPICIOUS_URL],
            "flags": [],
            "headers": {},
        }

    out_a = score_email(_extraction(f"hello team meeting notes {SUSPICIOUS_URL}"),
                        RuleBasedScorer())
    out_b = score_email(_extraction(f"urgent password reset now {SUSPICIOUS_URL}"),
                        RuleBasedScorer())
    # Same URLs/flags, different body words -> identical ablation scores
    # through the same fuse_email path the other backbones use.
    for key in ("score_a", "score_b", "score_c", "score_d_weighted", "score_d_max"):
        assert out_a[key] == pytest.approx(out_b[key])
        assert 0.0 <= out_a[key] <= 1.0


def test_rule_based_cli_choices_in_eval_and_predict_help():
    for script in ("run_multimodal_eval", "run_multimodal_predict"):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / f"{script}.py"), "--help"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0
        assert "rule_based" in proc.stdout


# ---------------------------------------------------------------------------
# (c) Bigger BiLSTM config builds (tiny 1-epoch CPU run).
# ---------------------------------------------------------------------------
def test_bigger_bilstm_config_builds_and_predicts():
    frames = _tiny_frames()
    model, _info = build_torch_baseline(
        "bilstm",
        frames["train"]["clean_text"].tolist(),
        frames["train"]["label"].tolist(),
        frames["val"]["clean_text"].tolist(),
        frames["val"]["label"].tolist(),
        config=TorchTrainConfig(epochs=1, emb_dim=200, hidden=256, layers=2,
                                dropout=0.3, max_len=32, batch_size=4),
        seed=42,
    )
    assert model.name == "bilstm"
    assert model.extra["emb_dim"] == 200
    assert model.extra["hidden"] == 256
    assert model.extra["layers"] == 2
    probs = model.predict_proba([
        "verify your password now please",
        "team meeting notes for tomorrow",
    ])
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))
