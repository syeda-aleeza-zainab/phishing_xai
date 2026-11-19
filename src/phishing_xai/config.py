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
class FusionConfig:
    """Configuration for multimodal fusion (Section D, ``multimodal.py``).

    Precision-first defaults (user priority 2026-11-19: "mera priority focus
    precision hai"). The weighted fusion gives the concatenated-context score
    the largest weight and the single best part only a small weight, so one
    weak part alone cannot flip the verdict: corroboration across channels
    is required. ``threshold`` is a PLACEHOLDER (0.5) until it is tuned on
    VALIDATION on Kaggle; never tune it on test, and never claim it was tuned
    until that validation run exists.
    """
    # Weights of the weighted fusion; they sum to 1.0.
    w_context: float = 0.45   # concatenated subject+body+OCR+PDF+QR score
    w_max_part: float = 0.20  # single highest-scoring part (kept small on purpose)
    w_url: float = 0.20       # worst static URL risk score (Section C)
    w_flags: float = 0.15     # header / structural flag score (Section B)
    # Decision threshold on the fused score. PLACEHOLDER 0.5 — tune on
    # validation only (see module docstring above).
    threshold: float = 0.5
    # A fused score at/above this value is phishing even with a single
    # active channel (very strong evidence); below it, corroboration across
    # at least two channels (or one channel + strong flags) is required.
    high_confidence_threshold: float = 0.75
    # Model window used when truncating the concatenated context.
    max_length: int = 512


@dataclass
class XAIConfig:
    """Configuration for explainability and faithfulness evaluations."""
    lime_num_features: int = 15
    lime_num_samples: int = 1000
    lime_batch_size: int = 32
    alpha: float = 0.5  # LITA weight for LIME (paper Eq. 2): 0.0 = pure TI, 1.0 = pure LIME.
    faithfulness_n_samples: int = 200
    faithfulness_k_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    alpha_ablation_values: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    random_baseline_seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    seed: int = 42
