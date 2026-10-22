"""Model and tokenizer loading, custom PyTorch Dataset, and dynamic padding DataLoader."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.model")


class EmailDataset(Dataset):
    """PyTorch Dataset holding raw text and optional integer labels."""

    def __init__(self, texts: List[str], labels: Optional[List[int]] = None) -> None:
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class DynamicPaddingCollator:
    """Collator that dynamically pads tokenized sequences to the batch maximum length."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encoded = self.tokenizer(
            texts,
            padding=True,  # Dynamic padding to longest sequence in batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch_dict: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        if "label" in batch[0]:
            labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
            batch_dict["labels"] = labels

        return batch_dict


def load_tokenizer_and_model(
    model_name: str = "roberta-base",
    num_labels: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedTokenizerFast, PreTrainedModel]:
    """Load pretrained transformer tokenizer and classification model."""
    logger.info(f"Loading tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    logger.info(f"Loading sequence classification model: {model_name} (num_labels={num_labels})")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    # Set label mappings for interpretability
    model.config.id2label = {0: "Safe Email", 1: "Phishing Email"}
    model.config.label2id = {"Safe Email": 0, "Phishing Email": 1}

    if device is not None:
        model.to(device)

    return tokenizer, model


def create_dataloader(
    texts: List[str],
    labels: Optional[List[int]],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 32,
    shuffle: bool = False,
    max_length: int = 512,
) -> DataLoader:
    """Create a DataLoader with dynamic padding."""
    dataset = EmailDataset(texts=texts, labels=labels)
    collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
