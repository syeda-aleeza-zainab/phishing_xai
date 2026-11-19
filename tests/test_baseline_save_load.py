"""Tests for fitted-baseline save / load (6B.2).

(a) Tiny BiLSTM (1 epoch) and 1D-CNN save -> load -> ``predict_proba``
    identical (exact equality) on the same texts.
(b) Tiny TF-IDF+LR save -> load -> ``predict_proba`` identical.
(c) ``load_scorer`` with a saved BiLSTM checkpoint returns a scorer whose
    predictions match the pre-save model and does NOT refit (the fit path
    is monkeypatched to raise); a RoBERTa-style ``.pt`` instead falls
    through to fitting; a corrupt (non-zip, marker-less) checkpoint file
    raises instead of silently fitting.
(d) ``load_scorer`` with ``checkpoint=None`` still fits on the train split
    (existing behaviour, tiny synthetic frames via monkeypatched
    ``load_splits`` — no real dataset, no GPU, no downloads).

All tests are CPU-only and tiny; the whole file runs in well under a minute.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.baselines import (
    BaselineScorer,
    TorchTrainConfig,
    build_tfidf_lr,
    build_torch_baseline,
    load_baseline_model,
    load_scorer,
    save_baseline_model,
)

TEXTS = [
    "verify your password now please",
    "team meeting notes for tomorrow",
    "verify your account urgently today",
    "quarterly report attached for review",
]


def _tiny_frames():
    """Tiny train/val frames (same shape as tests/test_6b_prep.py)."""
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


def _tiny_config(**overrides):
    cfg = {"epochs": 1, "emb_dim": 8, "hidden": 8, "layers": 1,
           "dropout": 0.0, "max_len": 16, "max_vocab": 100, "batch_size": 2}
    cfg.update(overrides)
    return TorchTrainConfig(**cfg)


def _tiny_torch(kind, **overrides):
    frames = _tiny_frames()
    model, _info = build_torch_baseline(
        kind,
        frames["train"]["clean_text"].tolist(),
        frames["train"]["label"].tolist(),
        frames["val"]["clean_text"].tolist(),
        frames["val"]["label"].tolist(),
        config=_tiny_config(**overrides),
        seed=42, device=torch.device("cpu"),
    )
    return model


# ---------------------------------------------------------------------------
# (a) Torch round-trips: exact predict_proba equality.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["bilstm", "cnn"])
def test_torch_save_load_predict_proba_identical(tmp_path, kind):
    model = _tiny_torch(kind)
    before = model.predict_proba(TEXTS)
    path = tmp_path / f"best_{kind}.pt"
    saved = save_baseline_model(model, path)
    assert saved == path
    assert path.exists() and path.stat().st_size > 0

    loaded = load_baseline_model(path, device=torch.device("cpu"))
    assert loaded.name == kind
    assert loaded.kind == "torch"
    assert loaded.n_params == model.n_params
    # The fitted artefacts are re-attached, so explainability paths keep working.
    assert "_model" in loaded.extra and "_tokenizer" in loaded.extra
    assert loaded.extra["vocab_size"] == model.extra["vocab_size"]
    after = loaded.predict_proba(TEXTS)
    assert after.shape == before.shape
    np.testing.assert_array_equal(after, before)


def test_torch_save_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_baseline_model(tmp_path / "nope.pt")


def test_load_rejects_roberta_style_checkpoint(tmp_path):
    # A RoBERTa checkpoint (dict with model_state_dict, no baseline marker)
    # is not a baseline artefact: the loader must raise, never misread it.
    path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="not a saved baseline artefact"):
        load_baseline_model(path)


# ---------------------------------------------------------------------------
# (b) TF-IDF+LR round-trip: exact predict_proba equality.
# ---------------------------------------------------------------------------
def test_tfidf_lr_save_load_predict_proba_identical(tmp_path):
    frames = _tiny_frames()
    model = build_tfidf_lr(
        frames["train"]["clean_text"].tolist(),
        frames["train"]["label"].tolist(),
    )
    before = model.predict_proba(TEXTS)
    path = tmp_path / "best_tfidf_lr.pt"
    save_baseline_model(model, path)
    assert path.exists() and path.stat().st_size > 0

    loaded = load_baseline_model(path)
    assert loaded.name == "tfidf_lr"
    assert loaded.kind == "sklearn"
    assert "_pipeline" in loaded.extra
    after = loaded.predict_proba(TEXTS)
    np.testing.assert_array_equal(after, before)


# ---------------------------------------------------------------------------
# (c) load_scorer with a saved checkpoint loads instead of fitting.
# ---------------------------------------------------------------------------
def test_load_scorer_with_saved_bilstm_checkpoint_does_not_refit(monkeypatch, tmp_path):
    model = _tiny_torch("bilstm")
    before = model.predict_proba(TEXTS)
    path = tmp_path / "best_bilstm.pt"
    save_baseline_model(model, path)

    # Prove no refit: the entire fit path must raise if touched.
    def _boom(*args, **kwargs):
        raise AssertionError("load_scorer refit the model instead of loading the checkpoint")

    monkeypatch.setattr("phishing_xai.baselines.load_splits", _boom)
    monkeypatch.setattr("phishing_xai.baselines.build_torch_baseline", _boom)
    monkeypatch.setattr("phishing_xai.baselines.build_tfidf_lr", _boom)

    scorer = load_scorer("bilstm", checkpoint=path)
    assert isinstance(scorer, BaselineScorer)
    assert scorer.name == "bilstm"
    np.testing.assert_array_equal(scorer.predict_proba(TEXTS), before)


def test_load_scorer_with_roberta_checkpoint_fits_instead(monkeypatch, tmp_path):
    # A RoBERTa .pt must NOT be loaded as a baseline: load_scorer falls
    # through to fitting on the (monkeypatched, tiny) train split.
    frames = _tiny_frames()
    monkeypatch.setattr(
        "phishing_xai.baselines.load_splits",
        lambda *args, **kwargs: frames,
    )
    path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}}, path)

    scorer = load_scorer("bilstm", checkpoint=path, bilstm_epochs=1)
    assert isinstance(scorer, BaselineScorer)
    probs = scorer.predict_proba(TEXTS)
    assert probs.shape == (len(TEXTS), 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(TEXTS)))


def test_load_scorer_with_corrupt_checkpoint_raises(tmp_path):
    # A non-zip file with no baseline marker is neither a baseline artefact
    # nor a torch checkpoint: raise, never silently fit.
    path = tmp_path / "best_bilstm.pt"
    path.write_bytes(b"this is not a checkpoint at all" * 10)
    with pytest.raises(RuntimeError):
        load_scorer("bilstm", checkpoint=path)


def test_load_scorer_with_corrupt_baseline_artefact_raises(tmp_path):
    # A file that IS a baseline pickle but truncated must also raise.
    frames = _tiny_frames()
    model = build_tfidf_lr(
        frames["train"]["clean_text"].tolist(),
        frames["train"]["label"].tolist(),
    )
    path = tmp_path / "best_tfidf_lr.pt"
    save_baseline_model(model, path)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])  # truncate: marker gone / pickle broken
    with pytest.raises(RuntimeError):
        load_scorer("tfidf_lr", checkpoint=path)


# ---------------------------------------------------------------------------
# (d) load_scorer with checkpoint=None still fits (existing behaviour).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_kind", ["tfidf_lr", "bilstm", "cnn"])
def test_load_scorer_without_checkpoint_fits(monkeypatch, model_kind):
    frames = _tiny_frames()
    monkeypatch.setattr(
        "phishing_xai.baselines.load_splits",
        lambda *args, **kwargs: frames,
    )
    scorer = load_scorer(model_kind, checkpoint=None, bilstm_epochs=1)
    assert isinstance(scorer, BaselineScorer)
    assert scorer.name == model_kind
    probs = scorer.predict_proba(TEXTS)
    assert probs.shape == (len(TEXTS), 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(TEXTS)))
