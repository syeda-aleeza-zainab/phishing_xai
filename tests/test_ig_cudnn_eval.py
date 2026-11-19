"""Regression guard for the GPU-only cuDNN bug in ``integrated_gradients_scores``.

On GPU (Kaggle Tesla T4, torch 2.11.0+cu130, captum 0.9.0) the BiLSTM
``explain_word_scores`` dispatch failed with exactly::

    RuntimeError: cudnn RNN backward can only be called in training mode

``integrated_gradients_scores`` calls ``model.eval()`` and Captum then needs
gradients through the BiLSTM; cuDNN's RNN backward requires a forward run in
training mode, while the CPU native RNN supports backward in eval mode — so
the failure is GPU-only. The fix (in ``baselines.py`` only) runs the Captum
attribution with cuDNN disabled, so the LSTM uses the native implementation,
while the model stays in eval mode (dropout off, attributions deterministic).

This test cannot prove the GPU fix locally (no GPU here) — it asserts the
CPU-observable half of the contract: with the production dropout (0.3) the
function returns bit-identical scores on two consecutive calls, leaves the
model in eval mode, and restores the global cuDNN flag afterwards. If the
fix ever regressed to running the model in training mode (dropout on), the
two calls would differ.
"""

import torch

from phishing_xai.baselines import (
    TorchTrainConfig,
    build_torch_baseline,
    integrated_gradients_scores,
)

TRAIN_TEXTS = [
    "hello team meeting tomorrow please review the report",
    "quarterly report attached for your review team",
    "lunch meeting scheduled for friday afternoon",
    "please confirm your attendance at the team meeting",
    "project update meeting notes shared with the team",
    "holiday schedule and office meeting calendar",
    "urgent verify your account now click here claim prize",
    "your password expired click link verify account immediately",
    "congratulations you won prize click link claim now",
    "urgent action required verify bank account click here",
    "security alert verify your login click link now",
    "claim your prize money click here verify account",
]
TRAIN_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
VAL_TEXTS = [
    "team meeting notes for the quarterly review",
    "office lunch schedule friday meeting",
    "verify account click link urgent prize",
    "click here claim prize verify password",
]
VAL_LABELS = [0, 0, 1, 1]


def test_integrated_gradients_deterministic_in_eval_mode():
    """Two IG calls on a dropout=0.3 BiLSTM give identical scores in eval mode."""
    config = TorchTrainConfig(
        emb_dim=8, hidden=8, layers=1, dropout=0.3,
        max_len=16, lr=1e-2, epochs=1, batch_size=4, patience=1,
    )
    model, _ = build_torch_baseline(
        "bilstm", TRAIN_TEXTS, TRAIN_LABELS, VAL_TEXTS, VAL_LABELS,
        config=config, seed=42, device=torch.device("cpu"),
    )
    torch_model = model.extra["_model"]
    tokenizer = model.extra["_tokenizer"]
    text = "urgent verify your account click here"

    cudnn_before = torch.backends.cudnn.enabled
    scores_first = integrated_gradients_scores(
        torch_model, tokenizer, text, n_steps=5, device=torch.device("cpu")
    )
    scores_second = integrated_gradients_scores(
        torch_model, tokenizer, text, n_steps=5, device=torch.device("cpu")
    )

    assert scores_first == scores_second
    assert len(scores_first) == len(text.split())
    assert all(isinstance(s, float) for s in scores_first)
    # The fix must not flip the model into training mode (dropout on).
    assert torch_model.training is False
    assert torch_model.dropout.training is False
    # The cudnn-disable context manager must restore the global flag.
    assert torch.backends.cudnn.enabled == cudnn_before
