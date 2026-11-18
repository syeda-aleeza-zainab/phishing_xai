"""Configuration dataclasses for Phishing Email Detection XAI pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading, preprocessing, and splitting."""
    data_path: Path = Path("data/Phishing_Email.csv")
    test_size: float = 0.2
    val_size: float = 0.2  # 20% of train+val (16% of total)
    seed: int = 42
    replace_ratio: float = 0.2  # WordNet synonym replacement ratio
    debug: bool = False
    debug_samples: int = 500  # Subsample size for fast CPU debug run


@dataclass
class ModelConfig:
    """Configuration for RoBERTa model and tokenizer."""
    model_name: str = "roberta-base"
    num_labels: int = 2
    max_length: int = 512
    debug_max_length: int = 128


@dataclass
class TrainConfig:
    """Configuration for model training."""
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 5
    debug_epochs: int = 1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    seed: int = 42
    output_dir: Path = Path("results")
    checkpoint_name: str = "best_model.pt"


@dataclass
class XAIConfig:
    """Configuration for explainability and faithfulness evaluations."""
    lime_num_features: int = 15
    lime_num_samples: int = 1000
    lime_batch_size: int = 32
    alpha: float = 0.5
    faithfulness_n_samples: int = 200
    faithfulness_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    alpha_ablation_values: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    random_baseline_seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    seed: int = 42
