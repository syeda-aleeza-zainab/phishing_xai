"""Faithfulness evaluation metrics: Comprehensiveness, Sufficiency, and Random baseline.

Comprehensiveness: P(y|x) - P(y|x \\ top-k). Higher is more faithful.
Sufficiency:       P(y|x) - P(y|top-k).      Lower is more faithful.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from phishing_xai.data import truncate_text_to_model_window
from phishing_xai.explain import (
    LimeExplainer,
    TransformerExplainer,
    aggregate_subword_attributions_to_words,
    split_words,
)
from phishing_xai.lita import compute_lita_scores
from phishing_xai.utils import get_device, get_logger

logger = get_logger("phishing_xai.faithfulness")


def predict_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    target_class: int,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> float:
    """Predict softmax probability of target class for a given text."""
    device = device or get_device()
    model.eval()

    # Handle empty string perturbation
    text_to_encode = text if text.strip() else " "

    encoded = tokenizer(
        text_to_encode,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prob = float(probs[0, target_class].item())

    return prob


def select_top_k_indices(scores: Sequence[float], k: int) -> List[int]:
    """Select indices of top-k scores with deterministic tie-breaking.
    
    Ties are broken stably by index order (earlier index first).
    """
    if not scores or k <= 0:
        return []
    n = len(scores)
    actual_k = min(k, n)
    # Sort primarily by score descending, secondarily by index ascending
    ranked_indices = sorted(range(n), key=lambda i: (-scores[i], i))
    return ranked_indices[:actual_k]


def perturb_text_drop_indices(words: List[str], drop_indices: Sequence[int]) -> str:
    """Create perturbed text by removing words at specified indices."""
    drop_set = set(drop_indices)
    kept = [w for i, w in enumerate(words) if i not in drop_set]
    return " ".join(kept)


def perturb_text_keep_indices(words: List[str], keep_indices: Sequence[int]) -> str:
    """Create perturbed text by keeping ONLY words at specified indices (in original order)."""
    keep_set = set(keep_indices)
    kept = [w for i, w in enumerate(words) if i in keep_set]
    return " ".join(kept)


def compute_instance_faithfulness(
    p_orig: float,
    p_drop: float,
    p_keep: float,
) -> Tuple[float, float]:
    """Compute comprehensiveness and sufficiency for an instance.
    
    Comprehensiveness: P(y|x) - P(y|x \\ top-k). Higher is better.
    Sufficiency:       P(y|x) - P(y|top-k).      Lower is better.
    """
    comprehensiveness = float(p_orig - p_drop)
    sufficiency = float(p_orig - p_keep)
    return comprehensiveness, sufficiency


def evaluate_faithfulness_single_method(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    words: List[str],
    scores: Sequence[float],
    target_class: int,
    p_orig: float,
    ks: Sequence[int] = (1, 3, 5),
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> Dict[str, Dict[int, float]]:
    """Evaluate comprehensiveness and sufficiency across values of k for a single attribution method."""
    comp_by_k: Dict[int, float] = {}
    suff_by_k: Dict[int, float] = {}

    for k in ks:
        top_k_indices = select_top_k_indices(scores, k=k)
        text_drop = perturb_text_drop_indices(words, top_k_indices)
        text_keep = perturb_text_keep_indices(words, top_k_indices)

        p_drop = predict_probability(model, tokenizer, text_drop, target_class, device=device, max_length=max_length)
        p_keep = predict_probability(model, tokenizer, text_keep, target_class, device=device, max_length=max_length)

        comp, suff = compute_instance_faithfulness(p_orig, p_drop, p_keep)
        comp_by_k[k] = comp
        suff_by_k[k] = suff

    return {"comprehensiveness": comp_by_k, "sufficiency": suff_by_k}


def evaluate_random_baseline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    words: List[str],
    target_class: int,
    p_orig: float,
    ks: Sequence[int] = (1, 3, 5),
    num_seeds: int = 5,
    base_seed: int = 42,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> Dict[str, Dict[int, float]]:
    """Compute random baseline by averaging perturbations over multiple random permutations."""
    n_words = len(words)
    comp_accum: Dict[int, List[float]] = {k: [] for k in ks}
    suff_accum: Dict[int, List[float]] = {k: [] for k in ks}

    for seed_idx in range(num_seeds):
        rng = np.random.RandomState(base_seed + seed_idx)
        perm = rng.permutation(n_words).tolist() if n_words > 0 else []

        for k in ks:
            actual_k = min(k, n_words)
            chosen_indices = perm[:actual_k]

            text_drop = perturb_text_drop_indices(words, chosen_indices)
            text_keep = perturb_text_keep_indices(words, chosen_indices)

            p_drop = predict_probability(model, tokenizer, text_drop, target_class, device=device, max_length=max_length)
            p_keep = predict_probability(model, tokenizer, text_keep, target_class, device=device, max_length=max_length)

            comp, suff = compute_instance_faithfulness(p_orig, p_drop, p_keep)
            comp_accum[k].append(comp)
            suff_accum[k].append(suff)

    return {
        "comprehensiveness": {k: float(np.mean(comp_accum[k])) for k in ks},
        "sufficiency": {k: float(np.mean(suff_accum[k])) for k in ks},
    }
