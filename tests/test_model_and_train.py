"""Unit tests for model dynamic collation, evaluation metrics, and training loop checkpointing."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast

from phishing_xai.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curve
from phishing_xai.model import DynamicPaddingCollator, EmailDataset, create_dataloader
from phishing_xai.train import evaluate_dataloader, train_model
from phishing_xai.utils import set_seed


@pytest.fixture
def fast_tokenizer():
    return RobertaTokenizerFast.from_pretrained("roberta-base")


@pytest.fixture
def tiny_model():
    config = RobertaConfig(
        vocab_size=50265,
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=2,
        max_position_embeddings=128,
        num_labels=2,
    )
    return RobertaForSequenceClassification(config)


def test_dynamic_padding_collator(fast_tokenizer):
    texts = [
        "short email",
        "a significantly longer email containing multiple words for tokenization testing",
    ]
    labels = [0, 1]
    dataset = EmailDataset(texts=texts, labels=labels)
    collator = DynamicPaddingCollator(tokenizer=fast_tokenizer, max_length=128)

    batch = [dataset[0], dataset[1]]
    batch_tensors = collator(batch)

    assert "input_ids" in batch_tensors
    assert "attention_mask" in batch_tensors
    assert "labels" in batch_tensors

    input_ids = batch_tensors["input_ids"]
    assert input_ids.shape[0] == 2
    # Verify dynamic padding length is smaller than max_length (128)
    assert input_ids.shape[1] < 128
    # Verify padding token (1 for RoBERTa) exists in first (short) item
    assert (batch_tensors["attention_mask"][0] == 0).any()


def test_compute_metrics():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_plot_saving_and_closing(tmp_path: Path):
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.6, 0.9])

    cm_path = tmp_path / "confusion_matrix.png"
    roc_path = tmp_path / "roc_curve.png"

    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    plot_roc_curve(y_true, y_prob, save_path=roc_path)

    assert cm_path.exists()
    assert cm_path.stat().st_size > 0
    assert roc_path.exists()
    assert roc_path.stat().st_size > 0


def test_train_model_checkpointing(tmp_path: Path, fast_tokenizer, tiny_model):
    """Test train_model on tiny model with dummy data: checks validation checkpointing and single test pass."""
    set_seed(42)
    device = torch.device("cpu")

    train_texts = ["safe email message"] * 4 + ["phishing bank alert"] * 4
    train_labels = [0] * 4 + [1] * 4
    val_texts = ["safe email message"] * 2 + ["phishing bank alert"] * 2
    val_labels = [0] * 2 + [1] * 2
    test_texts = ["safe email message"] * 2 + ["phishing bank alert"] * 2
    test_labels = [0] * 2 + [1] * 2

    train_loader = create_dataloader(train_texts, train_labels, fast_tokenizer, batch_size=4, max_length=32)
    val_loader = create_dataloader(val_texts, val_labels, fast_tokenizer, batch_size=4, max_length=32)
    test_loader = create_dataloader(test_texts, test_labels, fast_tokenizer, batch_size=4, max_length=32)

    summary = train_model(
        model=tiny_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        learning_rate=1e-4,
        device=device,
        output_dir=tmp_path,
        seed=42,
    )

    checkpoint_file = tmp_path / "best_model.pt"
    metrics_file = tmp_path / "metrics.json"

    assert checkpoint_file.exists()
    assert metrics_file.exists()
    assert "best_epoch" in summary
    assert "final_test_metrics" in summary
    assert "accuracy" in summary["final_test_metrics"]
