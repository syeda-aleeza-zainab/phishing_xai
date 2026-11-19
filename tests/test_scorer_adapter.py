"""Tests for the 6B.1 scorer adapter (TextScorer).

No GPU, no internet, no real checkpoint needed. RoBERTa paths are exercised
only through monkeypatched ``score_text`` / ``load_trained_roberta``; the
baseline paths use tiny synthetic frames fitted on the spot. The fake
scorer defined here lives in this file only.
"""

import json
import re
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
    BaselineScorer,
    RobertaScorer,
    build_tfidf_lr,
    build_torch_baseline,
    explain_word_scores,
    load_scorer,
    TorchTrainConfig,
)
from phishing_xai.explain import split_words
from phishing_xai.multimodal import build_concatenated_context, score_texts

TRIGGER_WORDS = ("urgent", "verify", "password")


class FakeLstmScorer:
    """Deterministic stand-in scorer (test file only).

    P(phishing) = 0.9 when any trigger word is present, else 0.1; empty /
    whitespace text maps to [1.0, 0.0] (the same empty convention as
    ``score_text``). ``tokenize_words`` is the shared ``split_words`` and
    ``truncate`` keeps the first ``max_length`` words.
    """

    name = "fake_lstm"

    def predict_proba(self, texts):
        rows = []
        for text in texts:
            text = "" if text is None else str(text)
            if not text.strip():
                rows.append([1.0, 0.0])
                continue
            lowered = text.lower()
            p = 0.9 if any(w in lowered for w in TRIGGER_WORDS) else 0.1
            rows.append([1.0 - p, p])
        if not rows:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(rows, dtype=float)

    def tokenize_words(self, text):
        return split_words(text)

    def truncate(self, text, max_length=None):
        words = split_words(text or "")
        ml = max_length if max_length is not None else 512
        return " ".join(words[:ml])


class _FakeHFTokenizer:
    """Minimal HF-style tokenizer: one token per whitespace word."""

    def __call__(self, text, max_length=512, truncation=True, return_offsets_mapping=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        offsets = [(0, 0)] + spans[:max_length] + [(0, 0)]
        return {"offset_mapping": offsets}


def _write_eml(path: Path, subject: str, body: str) -> Path:
    path.write_text(
        f"From: a@example.com\nSubject: {subject}\nContent-Type: text/plain; charset=utf-8\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def _tiny_frames():
    """Tiny train/val frames whose repeated words survive min_df=2."""
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
    return {"train": train, "val": val, "test": val.copy()}


# ---------------------------------------------------------------------------
# (1) Fake scorer contract.
# ---------------------------------------------------------------------------
def test_fake_scorer_shape_rows_sum_and_empty():
    scorer = FakeLstmScorer()
    probs = scorer.predict_proba(["please verify your password", "hello team"])
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))
    np.testing.assert_allclose(probs[0], [0.1, 0.9])
    np.testing.assert_allclose(probs[1], [0.9, 0.1])
    np.testing.assert_allclose(scorer.predict_proba([""]), [[1.0, 0.0]])
    np.testing.assert_allclose(scorer.predict_proba(["   "]), [[1.0, 0.0]])
    assert scorer.predict_proba([]).shape == (0, 2)
    assert scorer.tokenize_words("verify your password") == ["verify", "your", "password"]
    assert scorer.truncate("a b c d e", 3) == "a b c"


# ---------------------------------------------------------------------------
# (2) RobertaScorer parity with score_text (monkeypatched).
# ---------------------------------------------------------------------------
def test_roberta_scorer_parity_with_score_text(monkeypatch):
    import scripts.run_multimodal_predict as predict_mod

    calls = []

    def fake_score_text(model, tokenizer, text, device=None, max_length=512):
        calls.append((text, max_length))
        if not (text or "").strip():
            return (0, 1.0, 0.0)
        if "phish" in text:
            return (1, 0.2, 0.8)
        return (0, 0.7, 0.3)

    monkeypatch.setattr(predict_mod, "score_text", fake_score_text)
    scorer = RobertaScorer(model="m", tokenizer="t", device="cpu", max_length=8)
    assert scorer.name == "roberta"
    probs = scorer.predict_proba(["", "hello phish", "x" * 1000])
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs[0], [1.0, 0.0])
    np.testing.assert_allclose(probs[1], [0.2, 0.8])
    np.testing.assert_allclose(probs[2], [0.7, 0.3])
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(3))
    # Over-long text went through score_text with the scorer's max_length.
    assert calls[2] == ("x" * 1000, 8)
    assert scorer.predict_proba([]).shape == (0, 2)
    assert scorer.tokenize_words("a b") == ["a", "b"]


# ---------------------------------------------------------------------------
# (3) load_scorer("roberta") delegates to the existing loader.
# ---------------------------------------------------------------------------
def test_load_scorer_roberta_delegates_to_existing_loader(monkeypatch, tmp_path):
    import scripts.run_multimodal_predict as predict_mod

    seen = {}

    def fake_loader(ckpt, **kwargs):
        seen["checkpoint"] = Path(ckpt)
        seen["kwargs"] = kwargs
        return ("tok", "mod", "dev")

    monkeypatch.setattr(predict_mod, "load_trained_roberta", fake_loader)
    ckpt = tmp_path / "best_model.pt"
    scorer = load_scorer("roberta", checkpoint=ckpt)
    assert isinstance(scorer, RobertaScorer)
    assert scorer.model == "mod"
    assert scorer.tokenizer == "tok"
    assert scorer.device == "dev"
    assert seen["checkpoint"] == ckpt

    with pytest.raises(ValueError):
        load_scorer("no_such_model")


# ---------------------------------------------------------------------------
# (4) load_scorer("tfidf_lr") fits on the train split, never the eval set.
# ---------------------------------------------------------------------------
def test_load_scorer_tfidf_lr_fits_on_train_split_only(monkeypatch):
    import phishing_xai.baselines as bl

    frames = _tiny_frames()
    calls = []
    opened_eval_paths = []
    real_open = open

    def fake_load_splits(*args, **kwargs):
        calls.append((args, kwargs))
        return frames

    def tracking_open(path, *args, **kwargs):
        if "multimodal_eval" in str(path):
            opened_eval_paths.append(str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(bl, "load_splits", fake_load_splits)
    monkeypatch.setattr("builtins.open", tracking_open)
    scorer = load_scorer("tfidf_lr")
    assert isinstance(scorer, BaselineScorer)
    assert scorer.name == "tfidf_lr"
    assert len(calls) == 1
    # The loader was pointed at the real dataset/splits, not the eval set.
    assert "Phishing_Email.csv" in str(calls[0][1].get("data_path", ""))
    assert "splits.json" in str(calls[0][1].get("splits_path", ""))
    assert opened_eval_paths == []
    probs = scorer.predict_proba(["verify your password now", "team meeting notes"])
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))
    # Fitted on the train frame: the phishing phrasing scores higher.
    assert probs[0, 1] > probs[1, 1]


# ---------------------------------------------------------------------------
# (5) predict_eml with the fake scorer: no checkpoint, no torch model.
# ---------------------------------------------------------------------------
def test_predict_eml_with_fake_scorer_no_checkpoint(tmp_path):
    import scripts.run_multimodal_predict as predict_mod

    eml = _write_eml(
        tmp_path / "sample.eml", "team update", "please verify your password now"
    )
    result = predict_mod.predict_eml(
        eml_path=eml, model=None, tokenizer=None, device=None, scorer=FakeLstmScorer(),
    )
    assert result["model"] == "fake_lstm"
    assert result["final_class"] == 1
    assert result["context_prob_phishing"] == pytest.approx(0.9)
    assert "fusion_weighted" in result and "fusion_max" in result
    # Hand-computed fusion: context 0.9, max part 0.9, no URL/flag evidence.
    # weighted = 0.45*0.9 + 0.20*0.9 = 0.585 (corroborated: 2 active channels).
    assert result["fusion_weighted"]["final_score"] == pytest.approx(0.585)
    assert result["fusion_weighted"]["corroborated"] is True
    assert result["fusion_max"]["final_score"] == pytest.approx(0.9)
    # Explain path with the fake scorer: LIME only, no TI fabricated.
    explained = predict_mod.predict_eml(
        eml_path=eml, model=None, tokenizer=None, device=None,
        explain=True, lime_samples=20, scorer=FakeLstmScorer(),
    )
    assert explained["explanation_method"] == "lime_only"
    assert explained["ti_available"] is False
    assert explained["alpha_effective"] == pytest.approx(1.0)
    assert len(explained["top_5_lita_words"]) > 0


# ---------------------------------------------------------------------------
# (6) score_email with the fake scorer: all ablation scores in [0, 1].
# ---------------------------------------------------------------------------
def test_score_email_with_fake_scorer_scores_in_range():
    from scripts.run_multimodal_eval import score_email

    extraction = {
        "body_text": "please verify your password now",
        "parts": [
            {"kind": "subject", "text": "team update"},
            {"kind": "body_text", "text": "please verify your password now"},
            {"kind": "image", "text": "hello world"},
            {"kind": "pdf_text", "text": "invoice attached"},
        ],
        "links": [],
        "flags": [],
        "headers": {},
    }
    out = score_email(extraction, FakeLstmScorer())
    for key in ("score_a", "score_b", "score_c", "score_d_weighted", "score_d_max"):
        assert key in out
        assert 0.0 <= out[key] <= 1.0
    assert out["score_a"] == pytest.approx(0.9)
    assert out["score_d_max"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# (7) Multimodal faithfulness with the fake scorer: lime_only.
# ---------------------------------------------------------------------------
def test_multimodal_faithfulness_with_fake_scorer_lime_only(tmp_path):
    from scripts.run_multimodal_faithfulness import (
        evaluate_multimodal_samples_faithfulness,
    )

    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    # No Subject header: the body must be the winning (only) scorable part.
    # (The shared winning-part key prefers a non-body part over body_text
    # even at a lower score, so a subject part would win here instead.)
    (samples_dir / "s1.eml").write_text(
        "From: a@example.com\nContent-Type: text/plain; charset=utf-8\n\n"
        "please verify your password now to avoid account closure today\n",
        encoding="utf-8",
    )
    rows = evaluate_multimodal_samples_faithfulness(
        samples_dir=samples_dir,
        checkpoint_path=tmp_path / "missing.pt",
        k=5,
        lime_samples=50,
        scorer=FakeLstmScorer(),
        model="fake_lstm",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "ok"
    assert row["part_kind"] == "body_text"
    assert row["model"] == "fake_lstm"
    assert row["explanation_method"] == "lime_only"
    assert row["ti_available"] is False
    assert row["alpha_effective"] == pytest.approx(1.0)
    for key in ("comprehensiveness_k5", "sufficiency_k5",
                "random_comprehensiveness_k5", "random_sufficiency_k5"):
        assert -1.0 <= row[key] <= 1.0
    # LIME's top words are the triggers; dropping them moves the score,
    # while a random drop usually leaves a trigger behind.
    assert row["random_comprehensiveness_k5"] != row["comprehensiveness_k5"]
    assert len(row["top_5_words"]) == 5


# ---------------------------------------------------------------------------
# (8) explain_word_scores dispatch: tfidf_lr / bilstm / fake.
# ---------------------------------------------------------------------------
def test_explain_word_scores_dispatch_tfidf_lr():
    frames = _tiny_frames()
    model = build_tfidf_lr(
        frames["train"]["clean_text"].tolist(), frames["train"]["label"].tolist()
    )
    scorer = BaselineScorer(model)
    words, lime, second, method = explain_word_scores(
        scorer, "verify your password now please", lime_samples=20, seed=42,
    )
    assert method == "lime_coefficients"
    assert words == ["verify", "your", "password", "now", "please"]
    assert len(lime) == len(second) == len(words)
    assert any(abs(s) > 0 for s in second)


def test_explain_word_scores_dispatch_bilstm():
    frames = _tiny_frames()
    model, _info = build_torch_baseline(
        "bilstm",
        frames["train"]["clean_text"].tolist(),
        frames["train"]["label"].tolist(),
        frames["val"]["clean_text"].tolist(),
        frames["val"]["label"].tolist(),
        config=TorchTrainConfig(epochs=1, emb_dim=16, hidden=16, layers=1,
                                max_len=32, batch_size=4),
        seed=42,
    )
    scorer = BaselineScorer(model)
    words, lime, second, method = explain_word_scores(
        scorer, "verify your password now please", lime_samples=20, seed=42,
    )
    assert method == "lime_ig"
    assert words == ["verify", "your", "password", "now", "please"]
    assert len(lime) == len(second) == len(words)
    assert all(isinstance(s, float) for s in second)


def test_explain_word_scores_dispatch_fake_lime_only():
    words, lime, second, method = explain_word_scores(
        FakeLstmScorer(), "please verify your password now", lime_samples=20, seed=42,
    )
    assert method == "lime_only"
    assert words == ["please", "verify", "your", "password", "now"]
    assert len(lime) == len(second) == len(words)
    assert second == lime
    assert explain_word_scores(FakeLstmScorer(), "", lime_samples=10) == ([], [], [], "lime_only")


# ---------------------------------------------------------------------------
# (9) build_concatenated_context truncation precedence.
# ---------------------------------------------------------------------------
def test_build_concatenated_context_truncation_precedence():
    long_text = " ".join(f"w{i}" for i in range(100))
    extraction = {
        "body_text": long_text,
        "parts": [{"kind": "body_text", "text": long_text}],
        "links": [],
        "flags": [],
        "headers": {},
    }
    # Scorer wins: word-truncated by the scorer.
    by_scorer = build_concatenated_context(extraction, scorer=FakeLstmScorer(), max_length=10)
    assert len(split_words(by_scorer)) == 10
    # Neither scorer nor tokenizer: char cap (max_length * 4).
    by_char = build_concatenated_context(extraction, max_length=10)
    assert len(by_char) == 40
    # Tokenizer: exact token-window truncation via offsets.
    by_tokenizer = build_concatenated_context(
        extraction, tokenizer=_FakeHFTokenizer(), max_length=3
    )
    assert by_tokenizer == "[BODY] w0 w1"


def test_score_texts_empty_and_whitespace():
    assert score_texts(FakeLstmScorer(), ["", "   "]) == [0.0, 0.0]
    assert score_texts(FakeLstmScorer(), ["verify now", "hello"]) == [0.9, 0.1]


# ---------------------------------------------------------------------------
# (10) CLI: --model defaults to roberta in all six scripts; bilstm skips
# the missing-checkpoint error in run_multimodal_predict.
# ---------------------------------------------------------------------------
SCRIPTS = [
    "run_multimodal_predict",
    "run_multimodal_eval",
    "run_multimodal_faithfulness",
    "run_explain",
    "run_faithfulness",
    "run_ablation",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_cli_model_defaults_to_roberta(script):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / f"{script}.py"), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0
    assert "--model" in proc.stdout
    # argparse wraps long help lines, so collapse whitespace first.
    assert "(default: roberta)" in " ".join(proc.stdout.split())


def test_predict_cli_bilstm_skips_missing_checkpoint_error(monkeypatch, tmp_path):
    import scripts.run_multimodal_predict as predict_mod

    eml = _write_eml(
        tmp_path / "sample.eml", "team update", "please verify your password now"
    )
    monkeypatch.setattr(predict_mod, "load_scorer", lambda *a, **k: FakeLstmScorer())
    rc = predict_mod.main([
        str(eml),
        "--model", "bilstm",
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--output_dir", str(tmp_path / "out"),
        "--output_json", str(tmp_path / "out.json"),
    ])
    assert rc == 0
    rows = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["model"] == "fake_lstm"
    assert rows[0]["final_class"] == 1


def test_predict_cli_default_model_is_roberta_missing_checkpoint_errors(tmp_path):
    import scripts.run_multimodal_predict as predict_mod

    eml = _write_eml(tmp_path / "sample.eml", "hi", "hello world")
    # No --model flag: the roberta default still requires the checkpoint.
    rc = predict_mod.main([
        str(eml),
        "--checkpoint", str(tmp_path / "missing.pt"),
    ])
    assert rc == 2
