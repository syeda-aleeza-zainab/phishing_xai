"""Unit tests for faithfulness metrics (comprehensiveness, sufficiency), tie-breaking, and alpha ablation."""

import numpy as np
import pytest
import torch

from phishing_xai.faithfulness import (
    compute_instance_faithfulness,
    evaluate_faithfulness_single_method,
    evaluate_random_baseline,
    perturb_text_drop_indices,
    perturb_text_keep_indices,
    select_top_k_indices,
)
from phishing_xai.lita import combine_scores, compute_lita_scores


def test_faithfulness_top_k_deterministic_tie_breaking():
    """Verify select_top_k_indices is strictly deterministic and breaks ties cleanly by sequence position."""
    scores = [0.2, 0.8, 0.8, 0.1, 0.8]
    # Indices 1, 2, and 4 all have the maximum score 0.8.
    # For k=2, stable tie-breaking should pick index 1, then index 2.
    top_2 = select_top_k_indices(scores, k=2)
    assert top_2 == [1, 2]

    # For k=3, it should pick [1, 2, 4]
    top_3 = select_top_k_indices(scores, k=3)
    assert top_3 == [1, 2, 4]

    # Determinism: running 100 times gives identical result
    for _ in range(100):
        assert select_top_k_indices(scores, k=3) == [1, 2, 4]

    # Edge cases
    assert select_top_k_indices([], k=3) == []
    assert select_top_k_indices([0.5], k=0) == []
    assert select_top_k_indices([0.5], k=5) == [0]


def test_perturb_text_drop_and_keep():
    """Verify drop and keep text perturbation preserve words and sequence order."""
    words = ["please", "verify", "your", "bank", "password", "now"]
    top_indices = [1, 4]  # "verify", "password"

    dropped_text = perturb_text_drop_indices(words, top_indices)
    expected_dropped = "please your bank now"
    assert dropped_text == expected_dropped

    kept_text = perturb_text_keep_indices(words, top_indices)
    expected_kept = "verify password"
    assert kept_text == expected_kept


def test_faithfulness_comprehensiveness_and_sufficiency_mock_model():
    """Verify mathematical definitions of comprehensiveness and sufficiency with known probabilities."""
    p_orig = 0.95
    p_drop = 0.20  # Removing key words drastically drops confidence
    p_keep = 0.90  # Keeping key words maintains confidence

    comp, suff = compute_instance_faithfulness(p_orig=p_orig, p_drop=p_drop, p_keep=p_keep)

    # Comprehensiveness = P(orig) - P(drop) = 0.95 - 0.20 = 0.75
    assert pytest.approx(comp, abs=1e-6) == 0.75
    # Sufficiency = P(orig) - P(keep) = 0.95 - 0.90 = 0.05
    assert pytest.approx(suff, abs=1e-6) == 0.05


def test_alpha_ablation_endpoints():
    """Verify alpha=0 matches pure LIME and alpha=1 matches pure TI across both normalization variants."""
    lime_scores = [0.1, 0.9, -0.4, 0.6]
    ti_scores = [0.8, -0.2, 0.5, 0.3]

    for variant in ["magnitude_preserving", "minmax"]:
        # Alpha = 0.0 -> 100% LIME
        scores_alpha_0 = compute_lita_scores(lime_scores, ti_scores, alpha=0.0, variant=variant)
        # Compute pure LIME normalized
        from phishing_xai.lita import normalize_magnitude_preserving, normalize_minmax
        norm_fn = normalize_magnitude_preserving if variant == "magnitude_preserving" else normalize_minmax
        expected_lime = norm_fn(np.array(lime_scores))
        np.testing.assert_allclose(scores_alpha_0, expected_lime, atol=1e-6)

        # Alpha = 1.0 -> 100% TI
        scores_alpha_1 = compute_lita_scores(lime_scores, ti_scores, alpha=1.0, variant=variant)
        expected_ti = norm_fn(np.array(ti_scores))
        np.testing.assert_allclose(scores_alpha_1, expected_ti, atol=1e-6)

        # Alpha = 0.5 -> 50% LIME + 50% TI
        scores_alpha_05 = compute_lita_scores(lime_scores, ti_scores, alpha=0.5, variant=variant)
        expected_hybrid = 0.5 * expected_lime + 0.5 * expected_ti
        np.testing.assert_allclose(scores_alpha_05, expected_hybrid, atol=1e-6)


def test_random_baseline_multi_seed():
    """Verify random baseline averages comprehensiveness and sufficiency across 5 distinct random seeds."""
    from unittest.mock import MagicMock

    words = ["word1", "word2", "word3", "word4", "word5", "word6"]

    # Mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # When predicting, return deterministic function of text length
    def mock_pred(model, tokenizer, text, target_class, device=None, max_length=128):
        # Probability scales with number of words
        n = len(text.split())
        return min(1.0, max(0.0, 0.1 * n))

    # Monkeypatch predict_probability in faithfulness module
    import phishing_xai.faithfulness as f_mod
    orig_pred = f_mod.predict_probability
    f_mod.predict_probability = mock_pred

    try:
        res = evaluate_random_baseline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            words=words,
            target_class=1,
            p_orig=0.6,
            ks=[1, 3],
            num_seeds=5,
            base_seed=42,
        )
        assert "comprehensiveness" in res
        assert "sufficiency" in res
        assert 1 in res["comprehensiveness"]
        assert 3 in res["comprehensiveness"]
        assert np.isfinite(res["comprehensiveness"][1])
        assert np.isfinite(res["sufficiency"][1])
    finally:
        f_mod.predict_probability = orig_pred
