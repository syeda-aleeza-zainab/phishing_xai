"""Tests for explainability (Captum vs TI, word-level aggregation), test-set isolation, and pipeline resumability."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from phishing_xai.explain import (
    TransformerExplainer,
    aggregate_subword_attributions_to_words,
)
from phishing_xai.train import train_model
from phishing_xai.tune import run_random_search
from phishing_xai.utils import load_json


TINY_MODEL = "hf-internal-testing/tiny-random-RobertaForSequenceClassification"


class TinyDummyDataset(Dataset):
    """Minimal dataset for fast training loop tests."""

    def __init__(self, size: int = 8, seq_len: int = 16):
        self.size = size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(10, 1000, (self.seq_len,), dtype=torch.long),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.tensor(idx % 2, dtype=torch.long),
        }


# =========================================================================
# Item 4(a): Captum LayerIntegratedGradients vs transformers_interpret
# =========================================================================
def test_captum_vs_transformers_interpret_tiny_model():
    """Verify Captum fallback and transformers_interpret produce matching interfaces, token counts, finite scores,
    and Spearman rank correlation > 0.9 with matching top-5 tokens."""
    from scipy.stats import spearmanr

    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_MODEL,
        id2label={0: "Safe Email", 1: "Phishing Email"},
        label2id={"Safe Email": 0, "Phishing Email": 1},
    )

    explainer = TransformerExplainer(model, tokenizer, device=torch.device("cpu"))
    assert explainer.ti_explainer is not None, "Primary transformers-interpret explainer should be initialized."

    sample_text = "Important security update: verify your bank account credentials."

    # 1. Primary explainer (transformers-interpret)
    ti_attributions = explainer.explain_raw(sample_text, target_class=1)
    # 2. Captum LayerIntegratedGradients fallback
    captum_attributions = explainer._explain_with_captum(sample_text, target_class=1)

    # Verify both return lists of (token, score) tuples
    assert isinstance(ti_attributions, list)
    assert isinstance(captum_attributions, list)
    assert len(ti_attributions) == len(captum_attributions)

    # Same tokens
    ti_tokens = [tok for tok, _ in ti_attributions]
    captum_tokens = [tok for tok, _ in captum_attributions]
    assert ti_tokens == captum_tokens

    # Finite attributions
    for _, score in ti_attributions:
        assert np.isfinite(score)
    for _, score in captum_attributions:
        assert np.isfinite(score)

    ti_scores = [s for _, s in ti_attributions]
    captum_scores = [s for _, s in captum_attributions]

    # Spearman rank correlation > 0.9
    rho, _ = spearmanr(ti_scores, captum_scores)
    assert rho > 0.9, f"Spearman correlation between TI and Captum fallback was {rho:.4f}, expected > 0.9"

    # Top-5 tokens match
    top5_ti = [ti_tokens[i] for i in np.argsort(ti_scores)[-5:][::-1]]
    top5_captum = [captum_tokens[i] for i in np.argsort(captum_scores)[-5:][::-1]]
    assert top5_ti == top5_captum, f"Top 5 tokens mismatch: TI={top5_ti} vs Captum={top5_captum}"


def test_end_to_end_explainer_aggregation_alignment():
    """Verify end-to-end pipeline: explainer -> word aggregation -> expected word strings on both TI and Captum paths."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_MODEL,
        id2label={0: "Safe Email", 1: "Phishing Email"},
        label2id={"Safe Email": 0, "Phishing Email": 1},
    )
    explainer = TransformerExplainer(model, tokenizer, device=torch.device("cpu"))

    text = "don't click http://a.b/c now, now"
    expected_words = ["don't", "click", "http://a.b/c", "now", ",", "now"]

    # 1. Path A: via TI
    ti_raw = explainer.explain_raw(text, target_class=1)
    ti_agg = aggregate_subword_attributions_to_words(text, ti_raw, tokenizer, max_length=128)
    ti_words = [item[1] for item in ti_agg]
    assert ti_words == expected_words, f"TI path aggregated words mismatch: {ti_words} vs {expected_words}"

    # 2. Path B: via Captum fallback
    captum_raw = explainer._explain_with_captum(text, target_class=1)
    captum_agg = aggregate_subword_attributions_to_words(text, captum_raw, tokenizer, max_length=128)
    captum_words = [item[1] for item in captum_agg]
    assert captum_words == expected_words, f"Captum path aggregated words mismatch: {captum_words} vs {expected_words}"


# =========================================================================
# Item 4(b): Word-level aggregation via word_ids with repeated words
# =========================================================================
def test_word_level_aggregation_multi_subword_and_repeated_words():
    """Verify word-level aggregation via word_ids handles multi-subword words and DOES NOT merge repeated words."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)

    # Sentence with multi-subword words and the word 'bank' repeated at distinct positions
    text = "Verify your bank account now, your bank will close."

    # Generate token attributions
    encoding = tokenizer(text, return_offsets_mapping=True, max_length=64, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    # Provide distinct synthetic attribution scores per token
    token_attributions = [(tok, float(idx * 0.1)) for idx, tok in enumerate(tokens)]

    aggregated = aggregate_subword_attributions_to_words(
        text=text,
        token_attributions=token_attributions,
        tokenizer=tokenizer,
        max_length=64,
    )

    assert isinstance(aggregated, list)
    assert len(aggregated) > 0

    # Collect all words and their assigned word indices
    extracted_words = [item[1] for item in aggregated]
    word_indices = [item[0] for item in aggregated]

    # Verify indices are monotonically strictly increasing
    assert word_indices == sorted(list(set(word_indices)))

    # Find occurrences of "bank"
    bank_occurrences = [item for item in aggregated if "bank" in item[1].lower()]

    # Repeated occurrences MUST NOT be merged: exactly 2 distinct entries for 'bank'
    assert len(bank_occurrences) == 2, (
        f"Expected 2 distinct occurrences for repeated word 'bank', but found: {bank_occurrences}"
    )

    first_bank, second_bank = bank_occurrences
    assert first_bank[0] != second_bank[0], "Each occurrence of 'bank' must have a distinct word index"
    assert first_bank[0] < second_bank[0]

    # Verify all scores are finite floats
    for w_idx, w_str, score in aggregated:
        assert isinstance(w_idx, int)
        assert isinstance(w_str, str)
        assert isinstance(score, float)
        assert np.isfinite(score)


# =========================================================================
# Item 4(c): Test split is NOT accessed before final evaluation
# =========================================================================
class SpyDataLoader:
    """Wrapper around DataLoader to strictly monitor and assert access times."""

    def __init__(self, dataloader: DataLoader, name: str):
        self.dataloader = dataloader
        self.name = name
        self.iter_count = 0

    def __iter__(self):
        self.iter_count += 1
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def test_test_set_not_accessed_before_final_evaluation(tmp_path: Path):
    """Strictly prove the test DataLoader is never accessed during epoch training/validation loops."""
    raw_train_loader = DataLoader(TinyDummyDataset(size=8), batch_size=4)
    raw_val_loader = DataLoader(TinyDummyDataset(size=4), batch_size=4)
    raw_test_loader = DataLoader(TinyDummyDataset(size=4), batch_size=4)

    test_spy = SpyDataLoader(raw_test_loader, name="test_loader")

    # Hook into val loader to verify test_loader has 0 accesses during training/validation
    class MonitoredValLoader:
        def __init__(self, loader, test_monitor):
            self.loader = loader
            self.test_monitor = test_monitor

        def __iter__(self):
            # When val loader runs, test loader must NOT have been accessed
            assert self.test_monitor.iter_count == 0, (
                f"Leakage detected! test_loader was accessed {self.test_monitor.iter_count} times during training/val loop!"
            )
            return iter(self.loader)

        def __len__(self):
            return len(self.loader)

    val_monitored = MonitoredValLoader(raw_val_loader, test_spy)

    model = AutoModelForSequenceClassification.from_pretrained(TINY_MODEL)

    output_dir = tmp_path / "train_test_isolation"
    results = train_model(
        model=model,
        train_loader=raw_train_loader,
        val_loader=val_monitored,  # type: ignore
        test_loader=test_spy,      # type: ignore
        epochs=2,
        output_dir=output_dir,
        device=torch.device("cpu"),
        use_amp=False,
    )

    # After the entire train_model run finishes:
    # test_loader must have been accessed EXACTLY ONCE for final evaluation
    assert test_spy.iter_count == 1, (
        f"test_loader was expected to be evaluated exactly once, but was accessed {test_spy.iter_count} times."
    )
    assert "final_test_metrics" in results


# =========================================================================
# Item 5: Training and Tuning Resumability Tests
# =========================================================================
def test_train_model_checkpoint_saving_and_resuming(tmp_path: Path):
    """Verify train_model saves latest_checkpoint.pt each epoch and cleanly resumes from it."""
    train_loader = DataLoader(TinyDummyDataset(size=8), batch_size=4)
    val_loader = DataLoader(TinyDummyDataset(size=4), batch_size=4)
    test_loader = DataLoader(TinyDummyDataset(size=4), batch_size=4)

    output_dir = tmp_path / "resume_test"

    # Run for 1 epoch with keep_latest=True to test resumability
    model = AutoModelForSequenceClassification.from_pretrained(TINY_MODEL)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        output_dir=output_dir,
        device=torch.device("cpu"),
        use_amp=False,
        keep_latest=True,
    )

    latest_ckpt_path = output_dir / "latest_checkpoint.pt"
    assert latest_ckpt_path.exists(), "latest_checkpoint.pt must be created"

    checkpoint_data = torch.load(latest_ckpt_path, map_location="cpu")
    assert checkpoint_data["epoch"] == 1
    assert "model_state_dict" in checkpoint_data
    assert "optimizer_state_dict" in checkpoint_data
    assert "best_val_f1" in checkpoint_data
    assert "history" in checkpoint_data

    # Resume for epoch 2 with keep_latest=True
    model_resume = AutoModelForSequenceClassification.from_pretrained(TINY_MODEL)
    resumed_results = train_model(
        model=model_resume,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=2,
        output_dir=output_dir,
        device=torch.device("cpu"),
        use_amp=False,
        resume=True,
        keep_latest=True,
    )

    final_checkpoint = torch.load(latest_ckpt_path, map_location="cpu")
    assert final_checkpoint["epoch"] == 2
    assert len(resumed_results["history"]["train_loss"]) == 2


def test_ckpt_dir_and_latest_checkpoint_auto_cleanup(tmp_path: Path):
    """Verify --ckpt_dir saves checkpoints to custom directory and latest_checkpoint.pt is auto-deleted upon success."""
    train_loader = DataLoader(TinyDummyDataset(size=8), batch_size=4)
    val_loader = DataLoader(TinyDummyDataset(size=4), batch_size=4)
    test_loader = DataLoader(TinyDummyDataset(size=4), batch_size=4)

    output_dir = tmp_path / "metrics_output"
    custom_ckpt_dir = tmp_path / "isolated_checkpoints"

    model = AutoModelForSequenceClassification.from_pretrained(TINY_MODEL)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        output_dir=output_dir,
        ckpt_dir=custom_ckpt_dir,
        device=torch.device("cpu"),
        use_amp=False,
        keep_latest=False,  # auto-cleanup enabled
    )

    # best_model.pt must be saved in custom_ckpt_dir
    assert (custom_ckpt_dir / "best_model.pt").exists(), "best_model.pt must reside in ckpt_dir"
    # metrics and figures must reside in output_dir
    assert (output_dir / "metrics.json").exists(), "metrics.json must reside in output_dir"
    # latest_checkpoint.pt must be auto-cleaned upon successful completion
    assert not (custom_ckpt_dir / "latest_checkpoint.pt").exists(), "latest_checkpoint.pt must be auto-deleted"



def test_tune_incremental_saving_and_skip_completed(tmp_path: Path):
    """Verify run_random_search saves iterations incrementally and skips completed trials on re-run."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    texts = ["Sample email number one.", "Sample email number two.", "Sample email number three.", "Sample email number four."]
    labels = [0, 1, 0, 1]

    tuning_dir = tmp_path / "tune_test"
    param_dists = {"train_batch_size": [2], "learning_rate": [1e-4], "epochs": [1]}

    # Run 1 iteration
    res1 = run_random_search(
        train_texts=texts,
        train_labels=labels,
        val_texts=texts[:2],
        val_labels=labels[:2],
        tokenizer=tokenizer,
        model_name=TINY_MODEL,
        n_iterations=1,
        max_length=32,
        param_distributions=param_dists,
        device=torch.device("cpu"),
        output_dir=tuning_dir,
        seed=42,
    )

    results_file = tuning_dir / "tuning" / "results.json"
    assert results_file.exists()
    data1 = load_json(results_file)
    assert len(data1["all_results"]) == 1
    assert data1["all_results"][0]["iteration"] == 1

    # Re-run with 2 iterations: iteration 1 should be skipped, iteration 2 executed
    res2 = run_random_search(
        train_texts=texts,
        train_labels=labels,
        val_texts=texts[:2],
        val_labels=labels[:2],
        tokenizer=tokenizer,
        model_name=TINY_MODEL,
        n_iterations=2,
        max_length=32,
        param_distributions=param_dists,
        device=torch.device("cpu"),
        output_dir=tuning_dir,
        seed=42,
    )

    data2 = load_json(results_file)
    assert len(data2["all_results"]) == 2
    iterations = [r["iteration"] for r in data2["all_results"]]
    assert iterations == [1, 2]
