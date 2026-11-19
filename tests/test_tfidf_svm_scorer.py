"""Tests for the tfidf_svm scorer adapter + robustness/latency helpers.

Tiny fixtures only: no GPU, no internet, no real checkpoint, no real
data (except the --help subprocess check, which only parses args).
The full-data fit of tfidf_svm is exercised on Kaggle, not here.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.baselines import (
    BaselineModel,
    BaselineScorer,
    SCORER_MODELS,
    build_tfidf_svm,
    load_baseline_model,
    load_scorer,
    save_baseline_model,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_robustness_loaded = _load_script("run_robustness_loaded")
run_latency_explain_time = _load_script("run_latency_explain_time")

TRAIN_TEXTS = [
    "hello team meeting tomorrow please review the report",
    "quarterly report attached for your review team",
    "lunch meeting scheduled for friday afternoon",
    "please confirm your attendance at the team meeting",
    "project update meeting notes shared with the team",
    "holiday schedule and office meeting calendar",
    "urgent verify your account now click here claim prize",
    "your password expired click link verify account immediately",
    "congratulations you won a prize click to claim money now",
    "verify your bank account urgently or it will be closed",
    "claim your free gift card now limited time offer",
    "suspicious login attempt verify password immediately click",
]
TRAIN_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
TEST_TEXTS = [
    "team meeting notes for tomorrow",
    "verify your account click here now",
]
TEST_LABELS = [0, 1]


def test_scorer_models_includes_tfidf_svm():
    """load_scorer advertises tfidf_svm alongside tfidf_lr/bilstm."""
    assert "tfidf_svm" in SCORER_MODELS
    assert "tfidf_lr" in SCORER_MODELS


def test_tfidf_svm_scorer_proba_sums_to_one():
    """BaselineScorer around a fitted tfidf_svm returns (n, 2) rows summing to 1."""
    model = build_tfidf_svm(TRAIN_TEXTS, TRAIN_LABELS)
    scorer = BaselineScorer(model)
    assert scorer.name == "tfidf_svm"
    probs = scorer.predict_proba(TEST_TEXTS)
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-6)
    assert scorer.tokenize_words("hello world") == ["hello", "world"]
    assert scorer.truncate("one two three four", max_length=2) == "one two"


def test_load_scorer_tfidf_svm_dispatches_to_build(monkeypatch):
    """load_scorer('tfidf_svm') fits via build_tfidf_svm on the train split.

    The real builders / splits are stubbed so this stays a fast unit
    test; the real full-data fit runs on Kaggle.
    """
    calls = {}

    def fake_load_splits(*args, **kwargs):
        return {"train": pd.DataFrame({"clean_text": TRAIN_TEXTS, "label": TRAIN_LABELS})}

    def fake_build(texts, labels):
        calls["texts"] = list(texts)
        calls["labels"] = list(labels)
        return BaselineModel("tfidf_svm", "sklearn",
                             lambda ts: np.tile(np.array([0.4, 0.6]), (len(ts), 1)))

    monkeypatch.setattr("phishing_xai.baselines.load_splits", fake_load_splits)
    monkeypatch.setattr("phishing_xai.baselines.build_tfidf_svm", fake_build)
    scorer = load_scorer("tfidf_svm")
    assert isinstance(scorer, BaselineScorer)
    assert scorer.name == "tfidf_svm"
    assert calls["labels"] == TRAIN_LABELS
    probs = scorer.predict_proba(["some text"])
    assert probs.shape == (1, 2)


def test_load_scorer_tfidf_svm_loads_saved_artefact(tmp_path):
    """A saved tfidf_svm artefact loads through load_scorer (no refit)."""
    model = build_tfidf_svm(TRAIN_TEXTS, TRAIN_LABELS)
    artefact = tmp_path / "best_tfidf_svm.pt"
    save_baseline_model(model, artefact)
    assert artefact.exists()
    scorer = load_scorer("tfidf_svm", checkpoint=artefact)
    assert isinstance(scorer, BaselineScorer)
    assert scorer.name == "tfidf_svm"
    expected = model.predict_proba(TEST_TEXTS)
    np.testing.assert_allclose(scorer.predict_proba(TEST_TEXTS), expected, atol=1e-9)


def test_load_scorer_unknown_model_still_rejected():
    """Unknown model kinds still raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model kind"):
        load_scorer("not_a_model")


def test_run_multimodal_eval_accepts_tfidf_svm_model():
    """The --model choices of run_multimodal_eval.py include tfidf_svm."""
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_multimodal_eval.py"), "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert out.returncode == 0
    assert "tfidf_svm" in out.stdout


def test_run_robustness_for_scorer_with_stub():
    """run_robustness_for_scorer returns clean + 3 perturbations for a stub scorer."""

    class StubScorer:
        name = "stub"

        def predict_proba(self, texts):
            # Perfect ranking: phishing texts contain "phish", safe do not.
            rows = []
            for t in texts:
                p = 0.9 if "phish" in t else 0.1
                rows.append([1.0 - p, p])
            return np.asarray(rows, dtype=float)

    texts = ["safe hello world"] * 4 + ["phish verify account click"] * 4
    labels = [0] * 4 + [1] * 4
    res = run_robustness_loaded.run_robustness_for_scorer(StubScorer(), texts, labels)
    assert res["clean_at_0_5"]["f1"] == pytest.approx(1.0)
    assert set(res["robustness"].keys()) == {"truncated", "urls_removed", "typo_synonym"}
    for kind, vals in res["robustness"].items():
        assert 0.0 <= vals["f1"] <= 1.0


def test_latency_summarise_seconds():
    """Median/mean/std helper for per-email seconds."""
    summary = run_latency_explain_time._summarise_seconds([1.0, 2.0, 3.0, 4.0])
    assert summary["median"] == pytest.approx(2.5)
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["n"] == 4
    assert len(summary["per_email"]) == 4
    empty = run_latency_explain_time._summarise_seconds([])
    assert empty["median"] is None and empty["n"] == 0


def test_time_explanations_uses_stubbed_explain_path(monkeypatch):
    """time_explanations times the explain_word_scores path once per email."""
    calls = []

    def fake_explain(scorer, text, lime_samples=100, seed=42):
        calls.append(text)
        return ["w"], [0.1], [0.2], "lita"

    monkeypatch.setattr(
        run_latency_explain_time, "explain_word_scores", fake_explain,
    )
    items = [{"id": "email_1", "text": "hello world"}, {"id": "email_2", "text": "foo bar"}]
    out = run_latency_explain_time.time_explanations(object(), items, lime_samples=5, seed=42)
    assert out["n"] == 2
    assert out["method"] == "lita"
    assert out["email_ids"] == ["email_1", "email_2"]
    assert len(calls) == 2
