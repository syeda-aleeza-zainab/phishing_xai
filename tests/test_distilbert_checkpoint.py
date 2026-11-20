"""Tests for the DistilBERT standalone checkpoint (save / load / train / adapters).

CPU-only, no GPU, no internet, no Hub downloads: the fixture is a tiny
randomly-initialised DistilBERT (``DistilBertConfig`` from scratch, dim 32,
1 layer) plus a tiny WordLevel tokenizer built with the ``tokenizers``
library. Nothing here downloads ``distilbert-base-uncased``.

Covered:
(a) save -> load round-trip: ``predict_proba`` identical (exact equality),
    ``best_distilbert.pt`` + ``tokenizer/`` + ``config.json`` all written,
    payload carries the format marker, ``kind="distilbert"``, model name
    and config.
(b) Load refuses a RoBERTa-style ``.pt`` (state dict only, no marker), a
    baseline artefact, and a missing file — never misreads, never refits.
(c) ``load_scorer("distilbert", checkpoint=...)`` returns a
    ``DistilBertScorer`` with identical predictions and does NOT refit
    (``AutoModelForSequenceClassification.from_pretrained`` is monkeypatched
    to raise: the loader rebuilds from the saved config, never the Hub);
    ``checkpoint=None`` and a missing path raise instead of refitting.
(d) ``train_distilbert`` smoke (1 epoch, 6 texts, CPU): best epoch tracked,
    standalone checkpoint written, test evaluated exactly once
    (``y_prob`` length == n_test), and the ``save_best=False`` variant
    returns test metrics with no checkpoint file.
(e) ``explain_word_scores`` on a ``DistilBertScorer`` takes the LITA branch
    (method ``"lita"``: LIME + TransformerExplainer/Captum), not
    ``"lime_only"`` — the Captum fallback runs locally because
    ``transformers-interpret`` is not installed in this venv.
(f) Adapter smoke: ``run_robustness_loaded.run_robustness_for_scorer``
    accepts the DistilBERT scorer (clean + 3 perturbations), and the OOD
    loader ``run_ood_eval.load_distilbert`` loads the artefact and
    refuses a RoBERTa-style ``.pt``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments" / "modern_phishing_ood" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "modern_phishing_ood" / "scripts"))

from phishing_xai.baselines import (  # noqa: E402
    DISTILBERT_ARTEFACT_FORMAT,
    DistilBertScorer,
    build_distilbert_wrapper,
    explain_word_scores,
    load_distilbert_model,
    load_scorer,
    save_distilbert_model,
    train_distilbert,
)

TEXTS = [
    "verify your password now please",
    "team meeting notes for tomorrow",
    "verify your account urgently today",
    "quarterly report attached for review",
]

VOCAB = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
    "verify": 5, "your": 6, "password": 7, "now": 8, "please": 9,
    "team": 10, "meeting": 11, "notes": 12, "for": 13, "tomorrow": 14,
    "urgent": 15, "today": 16, "account": 17, "quarterly": 18,
    "report": 19, "attached": 20, "review": 21, "urgently": 22,
}


def _tiny_tokenizer():
    """WordLevel tokenizer over the fixture vocab (no download)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(vocab=dict(VOCAB), unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    ft = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    ft.model_max_length = 512
    return ft


def _tiny_model():
    """Randomly-initialised tiny DistilBERT (dim 32, 1 layer, 2 labels)."""
    from transformers import AutoModelForSequenceClassification, DistilBertConfig

    cfg = DistilBertConfig(
        vocab_size=len(VOCAB), dim=32, hidden_dim=64,
        n_layer=1, n_heads=2, num_labels=2,
    )
    cfg.id2label = {0: "Safe Email", 1: "Phishing Email"}
    cfg.label2id = {"Safe Email": 0, "Phishing Email": 1}
    return AutoModelForSequenceClassification.from_config(cfg)


def _tiny_pair():
    return _tiny_model(), _tiny_tokenizer()


def _tiny_frames():
    train_texts = [
        "verify your password now please",
        "verify your password now urgent",
        "verify your password now today",
        "team meeting notes for tomorrow",
        "team meeting notes for today",
        "team meeting notes for review",
    ]
    train_labels = [1, 1, 1, 0, 0, 0]
    val_texts = ["verify your password now please", "team meeting notes for tomorrow"]
    val_labels = [1, 0]
    return train_texts, train_labels, val_texts, val_labels


# ---------------------------------------------------------------------------
# (a) Save / load round-trip: exact predict_proba equality.
# ---------------------------------------------------------------------------
def test_distilbert_save_load_predict_proba_identical(tmp_path):
    model, tokenizer = _tiny_pair()
    wrapper = build_distilbert_wrapper(model, tokenizer, device=torch.device("cpu"))
    before = wrapper.predict_proba(TEXTS)

    path = tmp_path / "best_distilbert.pt"
    saved = save_distilbert_model(model, tokenizer, path, extra={"seed": 42})
    assert saved == path
    assert path.exists() and path.stat().st_size > 0
    assert (tmp_path / "tokenizer").is_dir()
    assert (tmp_path / "tokenizer" / "tokenizer.json").exists()
    assert (tmp_path / "config.json").exists()

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == DISTILBERT_ARTEFACT_FORMAT
    assert payload["kind"] == "distilbert"
    assert payload["model_name"]
    assert payload["config"]["model_type"] == "distilbert"
    assert "state_dict" in payload

    loaded_model, loaded_tokenizer, _payload = load_distilbert_model(
        path, device=torch.device("cpu")
    )
    loaded = build_distilbert_wrapper(
        loaded_model, loaded_tokenizer, device=torch.device("cpu")
    )
    after = loaded.predict_proba(TEXTS)
    assert after.shape == before.shape
    np.testing.assert_array_equal(after, before)
    np.testing.assert_allclose(after.sum(axis=1), np.ones(len(TEXTS)))


def test_distilbert_save_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_distilbert_model(tmp_path / "nope.pt")


# ---------------------------------------------------------------------------
# (b) Load refuses RoBERTa-style .pt and baseline artefacts.
# ---------------------------------------------------------------------------
def test_load_rejects_roberta_style_checkpoint(tmp_path):
    path = tmp_path / "best_model.pt"
    torch.save({"epoch": 1, "model_state_dict": {}, "val_f1": 0.9}, path)
    with pytest.raises(ValueError, match="not a saved DistilBERT artefact"):
        load_distilbert_model(path)


def test_load_rejects_baseline_artefact(tmp_path):
    from phishing_xai.baselines import BASELINE_ARTEFACT_FORMAT, build_tfidf_lr

    frames = _tiny_frames()
    model = build_tfidf_lr(frames[0], frames[1])
    from phishing_xai.baselines import save_baseline_model

    path = tmp_path / "best_tfidf_lr.pt"
    save_baseline_model(model, path)
    assert BASELINE_ARTEFACT_FORMAT  # the file is a baseline artefact, not DistilBERT
    with pytest.raises(ValueError, match="not a saved DistilBERT artefact"):
        load_distilbert_model(path)


# ---------------------------------------------------------------------------
# (c) load_scorer("distilbert") loads, never refits, never falls back.
# ---------------------------------------------------------------------------
def test_load_scorer_distilbert_loads_and_does_not_refit(monkeypatch, tmp_path):
    model, tokenizer = _tiny_pair()
    wrapper = build_distilbert_wrapper(model, tokenizer, device=torch.device("cpu"))
    before = wrapper.predict_proba(TEXTS)
    path = tmp_path / "best_distilbert.pt"
    save_distilbert_model(model, tokenizer, path)

    # Prove no refit and no Hub load: the from_pretrained constructor
    # must raise if touched (the loader rebuilds from the saved config).
    def _boom(*args, **kwargs):
        raise AssertionError("load_scorer refit / re-downloaded instead of loading the checkpoint")

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained", _boom
    )
    scorer = load_scorer("distilbert", checkpoint=path, device=torch.device("cpu"))
    assert isinstance(scorer, DistilBertScorer)
    assert scorer.name == "distilbert"
    # allclose, not exact: the scorer scores one text at a time via
    # score_text while the wrapper scores the batch at once, and float32
    # padding changes the last ~1e-8 (same magnitude the OOD BEC contrast
    # already tolerates for GPU nondeterminism).
    np.testing.assert_allclose(scorer.predict_proba(TEXTS), before, atol=1e-6)


def test_load_scorer_distilbert_without_checkpoint_raises():
    with pytest.raises(ValueError, match="requires a saved DistilBERT artefact"):
        load_scorer("distilbert", checkpoint=None)


def test_load_scorer_distilbert_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scorer("distilbert", checkpoint=tmp_path / "nope.pt")


def test_load_scorer_distilbert_rejects_roberta_checkpoint(tmp_path):
    path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="not a saved DistilBERT artefact"):
        load_scorer("distilbert", checkpoint=path)


# ---------------------------------------------------------------------------
# (d) train_distilbert smoke: 1 epoch, tiny data, CPU, test evaluated once.
# ---------------------------------------------------------------------------
def test_train_distilbert_smoke_saves_standalone_checkpoint(tmp_path, monkeypatch):
    train_texts, train_labels, val_texts, val_labels = _tiny_frames()
    # Train on the tiny fixture pair instead of downloading distilbert-base.
    tiny_model, tiny_tokenizer = _tiny_pair()
    monkeypatch.setattr(
        "phishing_xai.model.load_tokenizer_and_model",
        lambda *args, **kwargs: (tiny_tokenizer, tiny_model),
    )
    out_dir = tmp_path / "seed_42"
    result = train_distilbert(
        train_texts, train_labels, val_texts, val_labels,
        test_texts=TEXTS, test_labels=[1, 0, 1, 0],
        epochs=1, batch_size=2, lr=2e-5, max_length=16, patience=3,
        seed=42, device=torch.device("cpu"), output_dir=out_dir,
    )
    assert result["status"] == "ok"
    assert result["best_epoch"] == 1
    assert result["test_metrics"] is not None
    assert len(result["y_prob"]) == len(TEXTS)
    assert len(result["y_pred"]) == len(TEXTS)
    assert (out_dir / "best_distilbert.pt").exists()
    assert (out_dir / "tokenizer").is_dir()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "latest_checkpoint.pt").exists()
    # The saved checkpoint loads standalone and reproduces the test probs.
    loaded_model, loaded_tokenizer, _ = load_distilbert_model(
        out_dir / "best_distilbert.pt", device=torch.device("cpu")
    )
    loaded = build_distilbert_wrapper(
        loaded_model, loaded_tokenizer, device=torch.device("cpu")
    )
    np.testing.assert_allclose(
        loaded.predict_proba(TEXTS)[:, 1], np.asarray(result["y_prob"]), atol=1e-6
    )
    # No _model/_tokenizer leak into the JSON-plain part of the result.
    jsonable = {k: v for k, v in result.items() if not k.startswith("_")}
    import json

    json.dumps(jsonable, default=str)


def test_train_distilbert_tuning_mode_never_touches_test(tmp_path, monkeypatch):
    train_texts, train_labels, val_texts, val_labels = _tiny_frames()
    tiny_model, tiny_tokenizer = _tiny_pair()
    monkeypatch.setattr(
        "phishing_xai.model.load_tokenizer_and_model",
        lambda *args, **kwargs: (tiny_tokenizer, tiny_model),
    )
    out_dir = tmp_path / "trial_0"
    result = train_distilbert(
        train_texts, train_labels, val_texts, val_labels,
        test_texts=None, test_labels=None,
        epochs=1, batch_size=2, lr=2e-5, max_length=16,
        seed=42, device=torch.device("cpu"), output_dir=out_dir,
        save_best=False,
    )
    assert result["test_metrics"] is None
    assert result["y_prob"] is None
    assert result["checkpoint"] is None
    assert not (out_dir / "best_distilbert.pt").exists()
    assert result["best_val_metrics"]["f1"] >= 0.0


# ---------------------------------------------------------------------------
# (e) explain_word_scores on a DistilBertScorer takes the LITA branch.
# ---------------------------------------------------------------------------
def test_explain_word_scores_distilbert_is_lita(tmp_path):
    model, tokenizer = _tiny_pair()
    path = tmp_path / "best_distilbert.pt"
    save_distilbert_model(model, tokenizer, path)
    scorer = load_scorer("distilbert", checkpoint=path, device=torch.device("cpu"))
    text = "verify your password now please team meeting notes"
    words, lime_scores, second_scores, method_name = explain_word_scores(
        scorer, text, lime_samples=5, seed=42,
    )
    assert method_name == "lita"
    assert len(words) == len(lime_scores) == len(second_scores)
    assert all(isinstance(s, float) for s in second_scores)


# ---------------------------------------------------------------------------
# (f) Adapter smoke: robustness + OOD loader accept the DistilBERT scorer.
# ---------------------------------------------------------------------------
def test_robustness_for_distilbert_scorer(tmp_path):
    from scripts.run_robustness_loaded import run_robustness_for_scorer

    model, tokenizer = _tiny_pair()
    scorer = DistilBertScorer(model, tokenizer, torch.device("cpu"), max_length=16)
    res = run_robustness_for_scorer(
        scorer,
        ["verify your password now please", "team meeting notes for tomorrow"],
        [1, 0],
        seed=42,
    )
    assert set(res["robustness"]) == {"truncated", "urls_removed", "typo_synonym"}
    for kind in ("truncated", "urls_removed", "typo_synonym"):
        assert 0.0 <= res["robustness"][kind]["f1"] <= 1.0


def test_ood_load_distilbert_accepts_artefact_and_refuses_roberta(tmp_path):
    import run_ood_eval

    model, tokenizer = _tiny_pair()
    path = tmp_path / "best_distilbert.pt"
    save_distilbert_model(model, tokenizer, path)
    scorer, info = run_ood_eval.load_distilbert(path)
    assert info["n_params"] > 0
    probs = np.asarray(scorer.predict_proba(TEXTS), dtype=float)
    assert probs.shape == (len(TEXTS), 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(TEXTS)))

    roberta_path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": {}}, roberta_path)
    with pytest.raises(ValueError, match="not a saved DistilBERT artefact"):
        run_ood_eval.load_distilbert(roberta_path)
    with pytest.raises(FileNotFoundError):
        run_ood_eval.load_distilbert(tmp_path / "nope.pt")
