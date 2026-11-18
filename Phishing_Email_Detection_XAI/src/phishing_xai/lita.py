"""LITA: Hybrid Explainability Attribution Scoring.

Combines Local Interpretable Model-agnostic Explanations (LIME) and
Transformers-Interpret (TI / Integrated Gradients) attributions.

Provides two variants:
1. `minmax`: The upstream legacy formulation (min-max scaled to [-1, 1]).
   Note: Known to distort magnitudes by forcing min to -1 and max to +1 regardless of raw scale.
2. `magnitude_preserving`: Improved formulation using signed max-absolute scaling (x / max(|x|)),
   preserving sign polarity, zero-baseline, and relative importance ratios without distortion.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def normalize_minmax(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [-1, 1] using min-max scaling (legacy upstream formulation).
    
    If all scores are identical, returns zeros.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    s_min = np.min(scores)
    s_max = np.max(scores)
    if np.isclose(s_max, s_min):
        return np.zeros_like(scores)
    # Scale to [0, 1] then shift to [-1, 1]
    return 2.0 * (scores - s_min) / (s_max - s_min) - 1.0


def normalize_magnitude_preserving(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize scores by max-absolute value to [-1, 1] while preserving zero and signs.
    
    Unlike min-max scaling:
    - Zero attribution strictly remains 0.0.
    - Positive scores remain positive; negative scores remain negative.
    - Relative magnitude ratios between features are strictly preserved.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    max_abs = np.max(np.abs(scores))
    if max_abs < eps:
        return np.zeros_like(scores)
    return scores / max_abs


def combine_scores(
    lime_norm: np.ndarray,
    ti_norm: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Compute convex combination of normalized attribution scores: (1 - alpha)*LIME + alpha*TI.
    
    At alpha=0.0: pure LIME.
    At alpha=1.0: pure TI.
    At alpha=0.5: balanced hybrid.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be within [0.0, 1.0], got {alpha}")
    return (1.0 - alpha) * np.asarray(lime_norm, dtype=float) + alpha * np.asarray(ti_norm, dtype=float)


def compute_lita_scores(
    lime_scores: Union[List[float], np.ndarray],
    ti_scores: Union[List[float], np.ndarray],
    alpha: float = 0.5,
    variant: str = "magnitude_preserving",
) -> np.ndarray:
    """Compute LITA hybrid scores from parallel arrays of LIME and TI attributions.
    
    Args:
        lime_scores: Attribution scores from LIME.
        ti_scores: Attribution scores from TI / Integrated Gradients.
        alpha: Weight parameter for LIME (1 - alpha weight for TI). Default 0.5.
        variant: Either 'magnitude_preserving' (default) or 'minmax' (legacy).
        
    Returns:
        np.ndarray: Combined LITA attribution scores.
    """
    lime_arr = np.asarray(lime_scores, dtype=float)
    ti_arr = np.asarray(ti_scores, dtype=float)

    if lime_arr.shape != ti_arr.shape:
        raise ValueError(
            f"Shape mismatch between LIME ({lime_arr.shape}) and TI ({ti_arr.shape}) scores."
        )

    if variant == "minmax":
        lime_norm = normalize_minmax(lime_arr)
        ti_norm = normalize_minmax(ti_arr)
    elif variant == "magnitude_preserving":
        lime_norm = normalize_magnitude_preserving(lime_arr)
        ti_norm = normalize_magnitude_preserving(ti_arr)
    else:
        raise ValueError(
            f"Unknown LITA variant '{variant}'. Supported variants: 'magnitude_preserving', 'minmax'."
        )

    return combine_scores(lime_norm, ti_norm, alpha=alpha)


def build_hybrid_word_attributions(
    word_attributions_ti: List[Tuple[int, str, float]],
    word_attributions_lime: List[Tuple[str, float]],
    alpha: float = 0.5,
    variant: str = "magnitude_preserving",
) -> List[Dict[str, Any]]:
    """Align word-level attributions from TI and LIME for a document and compute LITA.
    
    Preserves word occurrences and sequential position order from TI word aggregation.
    
    Args:
        word_attributions_ti: List of (word_idx, word_str, ti_score) from aggregate_subword_attributions_to_words.
        word_attributions_lime: List of (word_str, lime_score) from LimeTextExplainer.
        alpha: Weight parameter for LIME (default 0.5).
        variant: 'magnitude_preserving' or 'minmax'.
        
    Returns:
        List[Dict[str, Any]]: List of dicts with keys:
            - word_idx: integer sequence position
            - word: word string
            - ti_score: raw TI score
            - lime_score: aligned LIME score (0.0 if not in LIME explanation)
            - lita_score: computed hybrid attribution
    """
    # Create lookup map for LIME scores by lowercase cleaned word
    lime_map: Dict[str, float] = {}
    for word, score in word_attributions_lime:
        clean = word.strip().lower()
        if clean:
            lime_map[clean] = float(score)

    ti_raw = np.array([item[2] for item in word_attributions_ti], dtype=float)
    lime_aligned = np.array(
        [lime_map.get(item[1].strip().lower(), 0.0) for item in word_attributions_ti],
        dtype=float,
    )

    if len(word_attributions_ti) == 0:
        return []

    lita_scores = compute_lita_scores(
        lime_scores=lime_aligned,
        ti_scores=ti_raw,
        alpha=alpha,
        variant=variant,
    )

    results: List[Dict[str, Any]] = []
    for idx, (w_idx, w_str, ti_score) in enumerate(word_attributions_ti):
        results.append({
            "word_idx": w_idx,
            "word": w_str,
            "ti_score": float(ti_score),
            "lime_score": float(lime_aligned[idx]),
            "lita_score": float(lita_scores[idx]),
        })

    return results
