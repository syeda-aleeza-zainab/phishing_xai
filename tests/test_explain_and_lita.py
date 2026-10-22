"""Tests for Phase 4: LIME explainer wrapper, probability correctness, batching, and LITA hybrid scoring."""

import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from phishing_xai.explain import LimeExplainer, split_words, WORD_SPLIT_REGEX
from phishing_xai.lita import (
    build_hybrid_word_attributions,
    combine_scores,
    compute_lita_scores,
    normalize_magnitude_preserving,
    normalize_minmax,
)

TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TINY_MODEL)
    return model, tokenizer


def test_lime_predict_proba_returns_valid_probabilities(tiny_model_and_tokenizer):
    """Verify LimeExplainer.predict_proba returns true softmax probabilities summing to 1.0."""
    model, tokenizer = tiny_model_and_tokenizer
    explainer = LimeExplainer(model, tokenizer, device=torch.device("cpu"), batch_size=2)

    texts = [
        "Urgent: your account has been suspended. Please verify your credentials.",
        "Team meeting tomorrow at 10 AM in conference room B.",
        "Click here to claim your cash lottery prize immediately.",
    ]

    probs = explainer.predict_proba(texts)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (3, 2)
    # All probabilities in [0.0, 1.0]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Each row sums to 1.0
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), rtol=1e-5, atol=1e-5)


def test_lime_batching_equivalence(tiny_model_and_tokenizer):
    """Verify predict_proba yields identical probabilities regardless of batch_size."""
    model, tokenizer = tiny_model_and_tokenizer
    texts = [
        "Email text sample one for batch testing.",
        "Email text sample two with different length.",
        "Email text sample three confirming batched prediction equivalence.",
        "Short text four.",
    ]

    explainer_b1 = LimeExplainer(model, tokenizer, device=torch.device("cpu"), batch_size=1)
    explainer_b4 = LimeExplainer(model, tokenizer, device=torch.device("cpu"), batch_size=4)

    probs_b1 = explainer_b1.predict_proba(texts)
    probs_b4 = explainer_b4.predict_proba(texts)

    np.testing.assert_allclose(probs_b1, probs_b4, rtol=1e-5, atol=1e-5)


def test_lime_explain_on_tiny_model(tiny_model_and_tokenizer):
    """Verify LimeExplainer.explain returns properly formatted word attributions and predictions."""
    model, tokenizer = tiny_model_and_tokenizer
    explainer = LimeExplainer(model, tokenizer, device=torch.device("cpu"), batch_size=4, seed=42)

    text = "Please verify your bank login password immediately."
    word_attributions, pred_class, probs = explainer.explain(text, num_features=5, num_samples=30)

    assert pred_class in (0, 1)
    assert len(probs) == 2
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-5)
    assert isinstance(word_attributions, list)
    assert len(word_attributions) <= 5
    for word, score in word_attributions:
        assert isinstance(word, str)
        assert isinstance(score, float)
        assert np.isfinite(score)


def test_lita_normalization_minmax_vs_magnitude_preserving():
    """Verify properties of min-max vs magnitude-preserving normalization.
    
    Demonstrates that min-max distorts positive magnitudes to negative values,
    whereas magnitude-preserving retains positive signs, zero-baselines, and ratios.
    """
    all_positive = np.array([1.0, 2.0, 3.0])

    # Min-max forces min to -1.0, max to +1.0
    norm_minmax = normalize_minmax(all_positive)
    np.testing.assert_allclose(norm_minmax, np.array([-1.0, 0.0, 1.0]))
    # Notice: 1.0 was turned into -1.0 (distortion)!

    # Magnitude-preserving divides by max(|x|): keeps all positive, preserves 1:2:3 ratios
    norm_mag = normalize_magnitude_preserving(all_positive)
    np.testing.assert_allclose(norm_mag, np.array([1.0 / 3.0, 2.0 / 3.0, 1.0]))
    assert np.all(norm_mag > 0.0), "Magnitude-preserving must retain positive polarity"

    # Zero-baseline test: zero importance stays exactly zero
    sparse = np.array([0.0, 4.0, -2.0])
    norm_mag_sparse = normalize_magnitude_preserving(sparse)
    assert norm_mag_sparse[0] == 0.0
    assert norm_mag_sparse[1] == 1.0
    assert norm_mag_sparse[2] == -0.5

    # Constant scores test: no division by zero
    constant = np.array([5.0, 5.0, 5.0])
    np.testing.assert_allclose(normalize_minmax(constant), np.zeros(3))


def test_lita_combine_scores_alpha_weighting():
    """Verify alpha parameter linearly blends normalized LIME and TI scores."""
    lime_norm = np.array([0.2, 0.6, -0.4])
    ti_norm = np.array([0.8, -0.2, 0.4])

    # Alpha = 0.0: 100% LIME
    lita_alpha_0 = combine_scores(lime_norm, ti_norm, alpha=0.0)
    np.testing.assert_allclose(lita_alpha_0, lime_norm)

    # Alpha = 1.0: 100% TI
    lita_alpha_1 = combine_scores(lime_norm, ti_norm, alpha=1.0)
    np.testing.assert_allclose(lita_alpha_1, ti_norm)

    # Alpha = 0.5: equal blend
    lita_alpha_05 = combine_scores(lime_norm, ti_norm, alpha=0.5)
    expected_05 = 0.5 * lime_norm + 0.5 * ti_norm
    np.testing.assert_allclose(lita_alpha_05, expected_05)

    # Invalid alpha raises ValueError
    with pytest.raises(ValueError):
        combine_scores(lime_norm, ti_norm, alpha=1.5)


def test_build_hybrid_word_attributions_preserves_words_and_aligns():
    """Verify build_hybrid_word_attributions aligns LIME with TI word occurrences without merging repeated words."""
    # Sentence has 4 words, with "bank" repeated at index 1 and index 3
    ti_words = [
        (0, "verify", 0.4),
        (1, "bank", 0.8),
        (2, "your", -0.1),
        (3, "bank", 0.6),
    ]

    # LIME gives feature attributions for vocabulary words
    lime_words = [
        ("bank", 0.5),
        ("verify", 0.3),
    ]

    hybrid = build_hybrid_word_attributions(
        word_attributions_ti=ti_words,
        word_attributions_lime=lime_words,
        alpha=0.5,
        variant="magnitude_preserving",
    )

    assert len(hybrid) == 4
    # Check that both occurrences of "bank" are preserved as distinct items
    assert hybrid[1]["word_idx"] == 1
    assert hybrid[1]["word"] == "bank"
    assert hybrid[3]["word_idx"] == 3
    assert hybrid[3]["word"] == "bank"

    # Word 'your' wasn't in LIME top features -> aligned lime_score should be 0.0
    assert hybrid[2]["lime_score"] == 0.0
    assert np.isfinite(hybrid[2]["lita_score"])

    # All lita_scores must be finite floats
    for item in hybrid:
        assert np.isfinite(item["lita_score"])


def test_shared_word_splitter_contractions_urls_and_repeated_words():
    """Verify shared word splitter preserves contractions, URLs, numbers, and repeated words."""
    from lime.lime_text import IndexedString, LimeTextExplainer

    text = "don't click http://a.b/c now, now"
    tokens = split_words(text)

    # Contractions, URLs, and punctuation are kept distinct
    expected = ["don't", "click", "http://a.b/c", "now", ",", "now"]
    assert tokens == expected

    # Test LIME IndexedString with shared splitter
    idx_str = IndexedString(text, split_expression=split_words)
    assert len(idx_str.inverse_vocab) == 5  # ["don't", "click", "http://a.b/c", "now", ","]
    assert "don't" in idx_str.inverse_vocab
    assert "http://a.b/c" in idx_str.inverse_vocab
