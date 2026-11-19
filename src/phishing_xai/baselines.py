"""Baseline models and complexity-justification study (Section H).

Model ladder, evaluated on the SAME splits (``results/splits.json``), the SAME
cleaning (``phishing_xai.data.clean_text``) and the SAME train-only
augmentation policy as the RoBERTa pipeline:

1. ``dummy``       majority-class dummy classifier.
2. ``tfidf_lr``    TF-IDF word 1-2gram + LogisticRegression.
3. ``tfidf_svm``   TF-IDF char 3-5gram + LinearSVC, calibrated for probabilities.
4. ``bilstm``      BiLSTM with an embedding trained from scratch (no downloads).
5. ``cnn``         1D-CNN, behind ``--with_cnn`` / ``--models cnn``.
6. ``distilbert``  DistilBERT fine-tune, behind ``--with_distilbert`` / ``--models distilbert``.
7. ``roberta_eval`` eval-only: reads ``results/metrics.json`` or evaluates the
   existing checkpoint. Never retrains RoBERTa here.

Leakage rules (see also ``docs/PROJECT_LOG.md`` constraints recorded in the
Section H log): vocabulary / TF-IDF / tokenizer are fit on the TRAIN split
only; hyperparameter search (~8 trials) is scored on VALIDATION only; the test
set is touched exactly once per model, for the final evaluation. Anything not
executed is reported as ``"NOT RUN YET"``; no metric is ever fabricated.

PRECISION-FIRST (user decision, 2026-11-19): the primary metric is
phishing-class PRECISION (minimise false positives / low FPR). F1, accuracy
and recall are reported everywhere but never used alone for selection.
Models are ranked by precision first; the recommended deployment model is the
highest-precision model subject to a recall floor of 0.85 (on validation
aggregates); any threshold is fit on VALIDATION only and documented with its
validation precision/recall. Recall is always reported alongside precision.

Faithfulness note: ``phishing_xai.faithfulness.evaluate_faithfulness_single_method``
takes a ``(PreTrainedModel, PreTrainedTokenizerFast)`` pair and calls its own
``predict_probability`` internally, so it cannot be reused for non-transformer
baselines. :func:`evaluate_faithfulness_with_proba_fn` implements the identical
comprehensiveness / sufficiency maths (``p_orig - p_drop`` / ``p_orig - p_keep``)
over any ``predict_proba(texts)`` callable, which every baseline here exposes.
"""

from __future__ import annotations

import os
import pickle
import random
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import chi2 as chi2_dist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from phishing_xai.data import deduplicate_data, load_data
from phishing_xai.explain import split_words
from phishing_xai.utils import ensure_dir, get_device, get_logger, load_json, set_seed

logger = get_logger("phishing_xai.baselines")

DEFAULT_SEEDS: List[int] = [42, 1, 7]
FAITHFULNESS_KS: Tuple[int, ...] = (1, 3, 5)
BASELINE_MODELS: List[str] = [
    "dummy",
    "tfidf_lr",
    "tfidf_svm",
    "bilstm",
    "cnn",
    "distilbert",
    "roberta_eval",
]
# Hyperparameter-search effort, fixed by design so the comparison is fair:
# a small random search scored on VALIDATION only, never on test.
RANDOM_SEARCH_TRIALS: int = 8
URL_REGEX = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
HTML_TAG_REGEX = re.compile(r"<[^>]+>")
NON_ASCII_REGEX = re.compile(r"[^\x00-\x7F]")

DATASET_CAVEAT = (
    "Dataset caveat: this is an old Enron-style dataset that is highly separable, "
    "so high simple-model scores do not imply the same performance on modern, "
    "AI-written or multimodal phishing. A harder out-of-distribution test is needed "
    "before any deployment claim."
)


# ---------------------------------------------------------------------------
# Paths / Kaggle detection
# ---------------------------------------------------------------------------

def resolve_output_dir(output_dir: str | Path = "results/baselines") -> Path:
    """Resolve the baseline output directory, auto-detecting Kaggle.

    On Kaggle (``/kaggle`` exists) a relative default output is redirected under
    ``/kaggle/working/results/baselines`` so artefacts survive the session.
    An explicitly absolute ``output_dir`` is always honoured as given.
    """
    p = Path(output_dir)
    if os.path.exists("/kaggle") and not p.is_absolute():
        kaggle_dir = Path("/kaggle/working") / p
        logger.info(f"Kaggle detected; redirecting baseline output to {kaggle_dir}")
        return kaggle_dir
    return p


def is_kaggle() -> bool:
    """Return True when running on Kaggle."""
    return os.path.exists("/kaggle")


# ---------------------------------------------------------------------------
# Splits: identical to the RoBERTa pipeline
# ---------------------------------------------------------------------------

def load_splits(
    data_path: str | Path = "data/Phishing_Email.csv",
    splits_path: str | Path = "results/splits.json",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Load the dataset and rebuild the exact splits in ``splits.json``.

    The manifest indices are ``email_id`` values, i.e. positions in the
    deduplicated frame produced by :func:`phishing_xai.data.load_data` followed
    by :func:`phishing_xai.data.deduplicate_data` (the same order
    ``scripts/make_splits.py`` used). Sizes and label distributions are
    validated against the manifest summary; a mismatch raises ``ValueError``
    instead of silently evaluating on different splits.

    Raises:
        FileNotFoundError: if the CSV or the splits manifest is missing.
    """
    data_path = Path(data_path)
    splits_path = Path(splits_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {data_path}. "
            "Place data/Phishing_Email.csv in the repo root (or pass --data_path) "
            "and re-run. No baseline metric can be produced without it."
        )
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Splits manifest not found at: {splits_path}. "
            "Run scripts/make_splits.py first, or pass --splits_path."
        )
    df = load_data(data_path)
    df = deduplicate_data(df, text_col="clean_text", label_col="label")
    df = df.reset_index(drop=True)
    df["email_id"] = df.index
    manifest = load_json(splits_path)
    out: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        ids = set(manifest[f"{name}_indices"])
        part = df[df["email_id"].isin(ids)].reset_index(drop=True)
        out[name] = part
    summary = manifest.get("summary", {})
    for name in ("train", "val", "test"):
        expected = summary.get(f"{name}_size")
        if expected is not None and len(out[name]) != int(expected):
            raise ValueError(
                f"Split mismatch for {name}: manifest expects {expected} rows, "
                f"rebuilt {len(out[name])}. The CSV or the deduplication step "
                "differs from the one that produced splits.json; refusing to proceed."
            )
        dist = summary.get(f"{name}_label_dist", {})
        if dist:
            got = {str(k): int(v) for k, v in out[name]["label"].value_counts().items()}
            if got != {str(k): int(v) for k, v in dist.items()}:
                raise ValueError(
                    f"Label distribution mismatch for {name}: manifest {dist}, rebuilt {got}."
                )
    logger.info(
        "Loaded identical splits: train=%d val=%d test=%d",
        len(out["train"]), len(out["val"]), len(out["test"]),
    )
    return out


def subsample_splits(
    splits: Dict[str, pd.DataFrame], n_train: int = 200, n_val: int = 100, n_test: int = 100, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """Return small stratified subsamples of each split (``--debug`` mode)."""
    out: Dict[str, pd.DataFrame] = {}
    for name, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        df = splits[name]
        if len(df) <= n:
            out[name] = df
            continue
        parts = []
        for _, group in df.groupby("label"):
            k = max(1, int(round(n * len(group) / len(df))))
            parts.append(group.sample(n=min(k, len(group)), random_state=seed))
        out[name] = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Metrics, statistics, timing
# ---------------------------------------------------------------------------

def compute_metrics_full(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute the full H3 metric set for one evaluation.

    Returns accuracy, precision/recall/F1 for the phishing class (label 1),
    ROC-AUC, PR-AUC (average precision), the confusion matrix as
    ``[[tn, fp], [fn, tp]]`` and the false-positive rate.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        pr_auc = 0.0
    p_at_r = precision_at_recall(y_true_arr, y_prob_arr, target_recall=0.90)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_recall_0.90": p_at_r["precision"],
        "precision_at_recall_0.90_threshold": p_at_r["threshold"],
        "confusion_matrix": cm,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "n": int(len(y_true_arr)),
    }


def precision_at_recall(
    y_true: Sequence[int], y_prob: Sequence[float], target_recall: float = 0.90
) -> Dict[str, float]:
    """Precision at a fixed recall: max precision over thresholds with recall >= target.

    Returns ``{"precision", "threshold", "achieved_recall"}``. If no threshold
    reaches the target recall, the threshold with the highest achievable recall
    is returned and ``achieved_recall`` reports it honestly.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1.
    best: Dict[str, float] = {"precision": 0.0, "threshold": 0.5, "achieved_recall": 0.0}
    for i, thr in enumerate(thresholds):
        if recall[i] >= target_recall and precision[i] > best["precision"]:
            best = {"precision": float(precision[i]), "threshold": float(thr),
                    "achieved_recall": float(recall[i])}
    if best["achieved_recall"] == 0.0:
        # Target unreachable: take the threshold with max recall (then max precision).
        i = int(np.argmax(recall[:-1])) if len(recall) > 1 else 0
        best = {"precision": float(precision[i]),
                "threshold": float(thresholds[i]) if len(thresholds) > 0 else 0.5,
                "achieved_recall": float(recall[i])}
    return best


def fit_threshold_for_recall(
    y_true: Sequence[int], y_prob: Sequence[float], target_recall: float = 0.90
) -> Dict[str, float]:
    """Fit a decision threshold for a target recall.

    MUST be called on VALIDATION predictions only; the returned threshold is
    then applied unchanged to the test set. Returns the threshold plus the
    validation precision/recall it achieves, so the choice is documented.
    """
    res = precision_at_recall(y_true, y_prob, target_recall=target_recall)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= res["threshold"]).astype(int)
    return {
        "threshold": float(res["threshold"]),
        "target_recall": float(target_recall),
        "val_precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "val_recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "val_f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }


def select_deployment_model(
    models_agg: Dict[str, Dict[str, Any]], recall_floor: float = 0.85
) -> Dict[str, Any]:
    """Highest-precision model subject to a recall floor (precision-first rule).

    ``models_agg`` maps model name -> aggregate dict with ``precision`` /
    ``recall`` as ``{"mean": ...}`` (as produced by
    :func:`aggregate_seed_results`). Candidates meeting
    ``recall mean >= recall_floor`` are ranked by precision mean (ties: higher
    recall, then fewer parameters). If NO model meets the floor, the
    highest-precision model overall is returned with ``floor_met=False`` and
    ``floor_used`` set to that model's recall, stated explicitly so the report
    never pretends the 0.85 floor held.
    """
    entries: List[Dict[str, Any]] = []
    for name, agg in models_agg.items():
        prec = agg.get("precision", {}) if isinstance(agg, dict) else {}
        rec = agg.get("recall", {}) if isinstance(agg, dict) else {}
        p_mean = prec.get("mean") if isinstance(prec, dict) else None
        r_mean = rec.get("mean") if isinstance(rec, dict) else None
        if p_mean is None or r_mean is None:
            continue
        entries.append({
            "model": name, "precision": float(p_mean), "recall": float(r_mean),
            "n_params": agg.get("n_params") or float("inf"),
        })
    if not entries:
        return {"model": None, "status": "NOT RUN YET", "floor_met": False,
                "floor_used": recall_floor, "reason": "No executed precision/recall numbers."}
    eligible = [e for e in entries if e["recall"] >= recall_floor]
    if eligible:
        pool, floor_met, floor_used = eligible, True, recall_floor
    else:
        pool, floor_met = entries, False
        floor_used = max(e["recall"] for e in entries)
    pool.sort(key=lambda e: (-e["precision"], -e["recall"], e["n_params"]))
    best = pool[0]
    return {
        "model": best["model"], "precision": best["precision"], "recall": best["recall"],
        "floor_met": floor_met, "floor_used": floor_used,
        "floor_requested": recall_floor,
        "reason": (
            f"Highest precision among models with recall >= {recall_floor}"
            if floor_met
            else f"No model met recall >= {recall_floor} (best recall among models: "
                 f"{floor_used:.4f}); recommending highest-precision model overall "
                 f"({best['model']}, recall {best['recall']:.4f})."
        ),
    }


def mcnemar_test(
    y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]
) -> Dict[str, Any]:
    """McNemar's test between two classifiers on the same test set.

    ``b`` = A wrong, B right; ``c`` = A right, B wrong. Uses the chi-square
    statistic with continuity correction; exact binomial p-value when
    ``b + c < 25``. Returns ``{"b", "c", "statistic", "p_value"}``.
    """
    a = np.asarray(pred_a, dtype=int)
    b_pred = np.asarray(pred_b, dtype=int)
    y = np.asarray(y_true, dtype=int)
    if not (len(a) == len(b_pred) == len(y)):
        raise ValueError("mcnemar_test: prediction arrays must have equal length.")
    a_wrong = a != y
    b_wrong = b_pred != y
    b = int(np.sum(a_wrong & ~b_wrong))
    c = int(np.sum(~a_wrong & b_wrong))
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    if n_discordant < 25:
        # Exact two-sided binomial test, p = 0.5.
        from scipy.stats import binomtest

        p_value = float(binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue)
        statistic = float((abs(b - c) - 1.0) ** 2 / n_discordant)
    else:
        statistic = float((abs(b - c) - 1.0) ** 2 / n_discordant)
        p_value = float(chi2_dist.sf(statistic, 1))
    return {"b": b, "c": c, "statistic": statistic, "p_value": p_value}


def bootstrap_f1_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap percentile CI for F1 (phishing class).

    Returns ``{"f1", "ci_low", "ci_high", "n_resamples"}``.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    n = len(y_true_arr)
    if n == 0:
        raise ValueError("bootstrap_f1_ci: empty input.")
    rng = np.random.RandomState(seed)
    point = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    scores: List[float] = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        # Skip resamples with a single class in y_true (F1 undefined there).
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        scores.append(float(f1_score(y_true_arr[idx], y_pred_arr[idx], zero_division=0)))
    if not scores:
        return {"f1": point, "ci_low": point, "ci_high": point, "n_resamples": 0}
    alpha = 1.0 - ci
    return {
        "f1": point,
        "ci_low": float(np.percentile(scores, 100 * alpha / 2)),
        "ci_high": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "n_resamples": len(scores),
    }


def bootstrap_precision_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap percentile CI for phishing-class precision.

    Same resampling scheme as :func:`bootstrap_f1_ci`; returns
    ``{"precision", "ci_low", "ci_high", "n_resamples"}``.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    n = len(y_true_arr)
    if n == 0:
        raise ValueError("bootstrap_precision_ci: empty input.")
    rng = np.random.RandomState(seed)
    point = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    scores: List[float] = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true_arr[idx])) < 2:
            continue
        scores.append(float(precision_score(y_true_arr[idx], y_pred_arr[idx], zero_division=0)))
    if not scores:
        return {"precision": point, "ci_low": point, "ci_high": point, "n_resamples": 0}
    alpha = 1.0 - ci
    return {
        "precision": point,
        "ci_low": float(np.percentile(scores, 100 * alpha / 2)),
        "ci_high": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "n_resamples": len(scores),
    }


def mean_std(values: Sequence[float]) -> Dict[str, float]:
    """Mean and sample standard deviation across seeds."""
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(len(arr))}


def measure_latency(
    predict_proba_fn: Callable[[List[str]], np.ndarray],
    texts: Sequence[str],
    n: int = 200,
    warmup: int = 5,
) -> Dict[str, float]:
    """Median inference latency in ms/email, batch size 1.

    Warms up on ``warmup`` emails first, then times ``min(n, len(texts))``
    single-email calls and returns the median (robust to outliers).
    """
    sample = [str(t) for t in list(texts)[:n]]
    if not sample:
        raise ValueError("measure_latency: no texts supplied.")
    for text in sample[:warmup]:
        predict_proba_fn([text])
    timings: List[float] = []
    for text in sample:
        start = time.perf_counter()
        predict_proba_fn([text])
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_ms_per_email_median": float(np.median(timings)),
        "latency_ms_per_email_mean": float(np.mean(timings)),
        "n_timed": len(timings),
    }


def peak_memory_mb() -> Dict[str, Any]:
    """Peak memory observed so far: CPU RSS (ru_maxrss) and CUDA if present."""
    out: Dict[str, Any] = {"peak_cpu_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0}
    if torch.cuda.is_available():
        out["peak_cuda_mb"] = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    else:
        out["peak_cuda_mb"] = None
    return out


def count_params_torch(model: nn.Module) -> int:
    """Number of trainable parameters of a torch module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def object_size_mb(obj: Any) -> float:
    """Size in MB of a pickled object (in-memory; no file written)."""
    return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Robustness perturbations (reuses augment.py for the synonym case)
# ---------------------------------------------------------------------------

def apply_perturbation(texts: Sequence[str], kind: str, seed: int = 42) -> List[str]:
    """Apply one robustness perturbation to a list of texts.

    Kinds: ``truncated`` (first 128 whitespace tokens), ``urls_removed``,
    ``typo_synonym`` (WordNet synonym replacement via ``augment.py``; falls
    back to a deterministic adjacent-character swap when WordNet is
    unavailable, e.g. offline).
    """
    rng = random.Random(seed)
    out: List[str] = []
    if kind == "truncated":
        for text in texts:
            out.append(" ".join(str(text).split()[:128]))
    elif kind == "urls_removed":
        for text in texts:
            out.append(URL_REGEX.sub(" ", str(text)))
    elif kind == "typo_synonym":
        try:
            from phishing_xai.augment import synonym_replacement

            for text in texts:
                out.append(synonym_replacement(str(text), replace_ratio=0.2, rng=rng))
        except Exception as exc:  # offline / WordNet missing: typo fallback
            logger.warning(f"synonym_replacement unavailable ({exc}); using typo fallback.")
            for text in texts:
                words = str(text).split()
                new_words = []
                for w in words:
                    if len(w) > 3 and rng.random() < 0.2:
                        i = rng.randrange(len(w) - 1)
                        w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
                    new_words.append(w)
                out.append(" ".join(new_words))
    else:
        raise ValueError(f"Unknown perturbation kind: {kind!r}")
    return out


# ---------------------------------------------------------------------------
# Train-only tokenizer (word level, built from the train split only)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Word-level tokenizer fit on TRAIN texts only.

    Any word unseen in training maps to ``<unk>``; this is what makes the
    no-leakage test meaningful: a test-only token must never enter the vocab.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab: int = 30000, max_len: int = 256) -> None:
        """Create an empty tokenizer with the given vocab cap and max length."""
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def fit(self, texts: Sequence[str]) -> "SimpleTokenizer":
        """Build the vocabulary from training texts only, by frequency."""
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(str(text).split())
        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self

    def encode(self, text: str) -> List[int]:
        """Encode one text to token ids, truncated to ``max_len``.

        Empty text maps to a single ``<unk>`` id so a batch never has a
        zero-length sequence (which ``pack_padded_sequence`` rejects).
        """
        unk = self.word2idx[self.UNK_TOKEN]
        ids = [self.word2idx.get(w, unk) for w in str(text).split()[: self.max_len]]
        return ids or [unk]

    def __len__(self) -> int:
        return len(self.word2idx)


# ---------------------------------------------------------------------------
# Torch text classifiers: BiLSTM and 1D-CNN (embeddings trained from scratch)
# ---------------------------------------------------------------------------

class BiLSTMClassifier(nn.Module):
    """1-2 layer BiLSTM over a from-scratch embedding, max+mean pooling, linear head."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 200,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.3,
        num_labels: int = 2,
        pad_idx: int = 0,
    ) -> None:
        """Build the BiLSTM classifier."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2 * 2, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits of shape ``(batch, num_labels)``."""
        embedded = self.embedding(input_ids)
        lengths = None
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(embedded)
        mask = (input_ids != 0).float().unsqueeze(-1)
        masked = out * mask
        max_pooled, _ = masked.max(dim=1)
        mean_pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat([max_pooled, mean_pooled], dim=1)
        return self.classifier(self.dropout(pooled))


class CNNClassifier(nn.Module):
    """1D-CNN (kernel sizes 3/4/5) over a from-scratch embedding, max-pooled."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 200,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        dropout: float = 0.3,
        num_labels: int = 2,
        pad_idx: int = 0,
    ) -> None:
        """Build the 1D-CNN classifier."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits of shape ``(batch, num_labels)``."""
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = [torch.max(torch.relu(conv(embedded)), dim=2).values for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


@dataclass
class TorchTrainConfig:
    """Hyperparameters for a from-scratch torch text classifier."""

    emb_dim: int = 200
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.3
    max_len: int = 256
    max_vocab: int = 30000
    lr: float = 1e-3
    epochs: int = 8
    batch_size: int = 32
    patience: int = 3
    num_filters: int = 128
    weight_decay: float = 0.0


def _collate_torch(
    batch: List[Tuple[List[int], int]], pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of (token ids, label) pairs to the batch max length."""
    max_len = max(len(ids) for ids, _ in batch)
    input_ids = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask, labels


def train_torch_classifier(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    config: TorchTrainConfig,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train a torch text classifier with early stopping on VALIDATION F1.

    The test set is never touched here. Returns the best-state model (in
    place), training wall-clock seconds, epochs run, best val F1 and history.
    Mixed precision is used only when CUDA is available.
    """
    set_seed(seed)
    device = device or get_device()
    model.to(device)
    train_data = [(tokenizer.encode(t), int(y)) for t, y in zip(train_texts, train_labels)]
    val_data = [(tokenizer.encode(t), int(y)) for t, y in zip(val_texts, val_labels)]
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True,
        collate_fn=_collate_torch, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config.batch_size * 2, shuffle=False, collate_fn=_collate_torch
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Precision-first early stopping (user decision 2026-11-19): the primary
    # score is validation PRECISION subject to validation recall >= 0.85.
    # Epochs below the recall floor are penalised below any floor-meeting
    # epoch; if no epoch meets the floor, the best-F1 state is kept instead
    # and ``floor_met`` reports that honestly.
    recall_floor = 0.85
    best_score = -float("inf")
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1_state: Optional[Dict[str, torch.Tensor]] = None
    best_f1_value = -1.0
    floor_met = False
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {"train_loss": [], "val_f1": [], "val_precision": [], "val_recall": []}
    start = time.perf_counter()
    epochs_run = 0
    for epoch in range(1, config.epochs + 1):
        epochs_run = epoch
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        val_metrics = _torch_val_metrics(model, val_loader, device)
        val_f1, val_precision, val_recall = val_metrics["f1"], val_metrics["precision"], val_metrics["recall"]
        history["train_loss"].append(total_loss / max(1, len(train_loader)))
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        logger.info(f"BiLSTM/CNN epoch {epoch}/{config.epochs}: train_loss={history['train_loss'][-1]:.4f} val_precision={val_precision:.4f} val_recall={val_recall:.4f} val_f1={val_f1:.4f}")
        score = val_precision if val_recall >= recall_floor else val_precision - 2.0
        if val_f1 > best_f1_value:
            best_f1_value = val_f1
            best_f1_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if score > best_score:
            best_score = score
            best_val_precision, best_val_recall, best_val_f1 = val_precision, val_recall, val_f1
            floor_met = bool(val_recall >= recall_floor)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                logger.info(f"Early stopping after epoch {epoch} (patience {config.patience}).")
                break
    train_seconds = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_f1_state is not None:
        model.load_state_dict(best_f1_state)
        best_val_f1 = best_f1_value
    model.eval()
    return {
        "train_seconds": float(train_seconds),
        "epochs_run": epochs_run,
        "best_val_f1": float(best_val_f1),
        "best_val_precision": float(best_val_precision),
        "best_val_recall": float(best_val_recall),
        "recall_floor": recall_floor,
        "floor_met": bool(floor_met),
        "history": history,
    }


def _torch_val_metrics(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    """Validation precision/recall/F1 for a torch classifier (no gradients)."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            logits = model(input_ids.to(device), attention_mask.to(device))
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    model.train()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def torch_predict_proba(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    texts: Sequence[str],
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Batched softmax probabilities (rows sum to 1) for a torch classifier."""
    device = device or get_device()
    model.eval()
    model.to(device)
    probs: List[np.ndarray] = []
    data = [(tokenizer.encode(t), 0) for t in texts]
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=_collate_torch
    )
    with torch.no_grad():
        for input_ids, attention_mask, _ in loader:
            logits = model(input_ids.to(device), attention_mask.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    if not probs:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(probs)


# ---------------------------------------------------------------------------
# Unified baseline wrapper: every model exposes predict_proba(texts)
# ---------------------------------------------------------------------------

class BaselineModel:
    """Uniform wrapper around any baseline so callers only need ``predict_proba``.

    Attributes:
        name: model key (``dummy``, ``tfidf_lr``, ...).
        kind: ``"sklearn"`` or ``"torch"``.
        n_params: trainable parameter count (0 for the dummy).
        size_mb: model size on disk / pickled size in MB.
        extra: free-form metadata (hyperparameters, vocab size, ...).
    """

    def __init__(
        self,
        name: str,
        kind: str,
        predict_proba_fn: Callable[[List[str]], np.ndarray],
        n_params: int = 0,
        size_mb: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create the wrapper around a fitted model."""
        self.name = name
        self.kind = kind
        self._predict_proba_fn = predict_proba_fn
        self.n_params = int(n_params)
        self.size_mb = float(size_mb)
        self.extra: Dict[str, Any] = dict(extra or {})

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return ``(n, 2)`` softmax probabilities; rows sum to 1."""
        texts = ["" if t is None else str(t) for t in texts]
        probs = np.asarray(self._predict_proba_fn(texts), dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape(-1, 1)
        if probs.shape[1] == 1:
            probs = np.hstack([1.0 - probs, probs])
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Return predicted labels (0/1) at threshold 0.5."""
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)

    def set_device(self, device: torch.device) -> None:
        """Move a torch baseline to ``device`` (no-op for sklearn baselines).

        The torch ``predict_proba`` closure reads its device from a mutable
        holder, so this also switches the device used for inference. Used to
        time CPU and GPU latency separately on the same fitted model.
        """
        holder = self.extra.get("_device_holder")
        if holder is not None:
            holder["device"] = device
        model = self.extra.get("_model")
        if model is not None:
            model.to(device)


def _sklearn_n_params(pipeline: Pipeline) -> int:
    """Count TF-IDF vocabulary + classifier coefficients as parameters."""
    total = 0
    for _, step in pipeline.steps:
        if hasattr(step, "vocabulary_"):
            total += len(step.vocabulary_)
        if hasattr(step, "coef_"):
            total += int(np.asarray(step.coef_).size)
        if hasattr(step, "intercept_"):
            total += int(np.asarray(step.intercept_).size)
        if hasattr(step, "calibrated_classifiers_"):
            for calib in step.calibrated_classifiers_:
                est = getattr(calib, "estimator", calib)
                if hasattr(est, "coef_"):
                    total += int(np.asarray(est.coef_).size)
    return total


def build_dummy(train_labels: Sequence[int]) -> BaselineModel:
    """Majority-class dummy: always predicts the majority training label."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(train_labels), 1)), np.asarray(train_labels, dtype=int))
    majority = int(np.bincount(np.asarray(train_labels, dtype=int)).argmax())

    def _proba(texts: List[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0]]) if majority == 0 else np.array([[0.0, 1.0]]), (len(texts), 1))

    return BaselineModel("dummy", "sklearn", _proba, n_params=0, size_mb=0.0,
                         extra={"majority_class": majority})


def build_tfidf_lr(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    C: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = None,
) -> BaselineModel:
    """TF-IDF word 1-2gram + LogisticRegression (fit on train only)."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features,
                                  sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=C, max_iter=1000, solver="liblinear", random_state=42)),
    ])
    pipe.fit(list(train_texts), np.asarray(train_labels, dtype=int))
    wrapper = BaselineModel(
        "tfidf_lr", "sklearn", lambda texts: pipe.predict_proba(list(texts)),
        n_params=_sklearn_n_params(pipe), size_mb=object_size_mb(pipe),
        extra={"C": C, "ngram_range": list(ngram_range), "max_features": max_features,
               "vocab_size": len(pipe.named_steps["tfidf"].vocabulary_)},
    )
    # Fitted pipeline kept for native-coefficient explainability (H5); the
    # "_" prefix keeps it out of JSON serialisation (see _clean_extra).
    wrapper.extra["_pipeline"] = pipe
    return wrapper


def build_tfidf_svm(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    C: float = 1.0,
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: Optional[int] = None,
) -> BaselineModel:
    """TF-IDF char 3-5gram + LinearSVC, calibrated for probabilities.

    Calibration (``CalibratedClassifierCV``, 3-fold on the TRAIN split only)
    is what makes ``predict_proba`` available for a linear SVM.
    """
    base = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                                  max_features=max_features, sublinear_tf=True, min_df=2)),
        ("clf", LinearSVC(C=C, random_state=42)),
    ])
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    calibrated.fit(list(train_texts), np.asarray(train_labels, dtype=int))
    n_params = 0
    for calib in calibrated.calibrated_classifiers_:
        est = getattr(calib, "estimator", calib)
        if hasattr(est, "named_steps"):
            n_params += _sklearn_n_params(est)
    wrapper = BaselineModel(
        "tfidf_svm", "sklearn", lambda texts: calibrated.predict_proba(list(texts)),
        n_params=n_params, size_mb=object_size_mb(calibrated),
        extra={"C": C, "ngram_range": list(ngram_range), "max_features": max_features,
               "calibration": "sigmoid, 3-fold on train only"},
    )
    wrapper.extra["_pipeline"] = calibrated
    return wrapper


def build_torch_baseline(
    kind: str,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    config: TorchTrainConfig,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[BaselineModel, Dict[str, Any]]:
    """Build and train a BiLSTM (``kind="bilstm"``) or 1D-CNN (``kind="cnn"``).

    Tokenizer and embedding are built from the TRAIN split only; early
    stopping uses VALIDATION F1; the test set is never touched here.
    """
    if kind not in ("bilstm", "cnn"):
        raise ValueError(f"Unknown torch baseline kind: {kind!r}")
    device = device or get_device()
    tokenizer = SimpleTokenizer(max_vocab=config.max_vocab, max_len=config.max_len)
    tokenizer.fit(train_texts)
    if kind == "bilstm":
        model: nn.Module = BiLSTMClassifier(
            vocab_size=len(tokenizer), emb_dim=config.emb_dim, hidden=config.hidden,
            layers=config.layers, dropout=config.dropout,
        )
    else:
        model = CNNClassifier(
            vocab_size=len(tokenizer), emb_dim=config.emb_dim,
            num_filters=config.num_filters, dropout=config.dropout,
        )
    train_info = train_torch_classifier(
        model, tokenizer, train_texts, train_labels, val_texts, val_labels,
        config=config, seed=seed, device=device,
    )
    size_mb = float(sum(p.numel() * 4 for p in model.parameters()) / (1024.0 * 1024.0))
    device_holder: Dict[str, Any] = {"device": device}
    wrapper = BaselineModel(
        kind, "torch",
        lambda texts: torch_predict_proba(model, tokenizer, texts, device=device_holder["device"]),
        n_params=count_params_torch(model), size_mb=size_mb,
        extra={"emb_dim": config.emb_dim, "hidden": config.hidden, "layers": config.layers,
               "max_len": config.max_len, "vocab_size": len(tokenizer),
               "best_val_f1": train_info["best_val_f1"],
               "best_val_precision": train_info["best_val_precision"],
               "best_val_recall": train_info["best_val_recall"],
               "recall_floor_met": train_info["floor_met"],
               "epochs_run": train_info["epochs_run"]},
    )
    wrapper.extra["_device_holder"] = device_holder
    # Keep fitted artefacts reachable for explainability (IG on the embedding).
    wrapper.extra["_model"] = model  # type: ignore[assignment]
    wrapper.extra["_tokenizer"] = tokenizer  # type: ignore[assignment]
    wrapper.extra["_device"] = device  # type: ignore[assignment]
    return wrapper, train_info


# ---------------------------------------------------------------------------
# Small random search on VALIDATION only (fairness: ~8 trials, never test)
# ---------------------------------------------------------------------------

def random_search_tfidf(
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    kind: str = "tfidf_lr",
    n_trials: int = RANDOM_SEARCH_TRIALS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Random search over TF-IDF hyperparameters, scored on VALIDATION.

    Precision-first selection (user decision 2026-11-19): the winner is the
    highest validation PRECISION among trials with validation recall >= 0.85.
    If no trial meets the recall floor, the highest-precision trial wins and
    ``floor_met`` is False with the floor actually used reported. The test set
    is not used anywhere in this function.
    """
    rng = random.Random(seed)
    val_labels_arr = np.asarray(val_labels, dtype=int)
    recall_floor = 0.85
    trials: List[Dict[str, Any]] = []
    for _ in range(n_trials):
        if kind == "tfidf_lr":
            params: Dict[str, Any] = {
                "C": float(rng.choice([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])),
                "ngram_range": rng.choice([(1, 1), (1, 2), (1, 3)]),
                "max_features": rng.choice([None, 20000, 50000]),
            }
            model = build_tfidf_lr(train_texts, train_labels, **params)
        elif kind == "tfidf_svm":
            params = {
                "C": float(rng.choice([0.1, 0.5, 1.0, 2.0, 5.0])),
                "ngram_range": rng.choice([(3, 5), (2, 5), (3, 6)]),
                "max_features": rng.choice([None, 50000, 100000]),
            }
            model = build_tfidf_svm(train_texts, train_labels, **params)
        else:
            raise ValueError(f"random_search_tfidf: unsupported kind {kind!r}")
        val_pred = model.predict(val_texts)
        val_precision = float(precision_score(val_labels_arr, val_pred, zero_division=0))
        val_recall = float(recall_score(val_labels_arr, val_pred, zero_division=0))
        val_f1 = float(f1_score(val_labels_arr, val_pred, zero_division=0))
        trials.append({"params": params, "val_precision": val_precision,
                       "val_recall": val_recall, "val_f1": val_f1})
    eligible = [t for t in trials if t["val_recall"] >= recall_floor]
    pool = eligible if eligible else trials
    best = max(pool, key=lambda t: (t["val_precision"], t["val_recall"], t["val_f1"]))
    logger.info(f"Random search ({kind}, {n_trials} trials on val, precision-first): best={best} floor_met={bool(eligible)}")
    return {"best_params": best["params"], "best_val_precision": best["val_precision"],
            "best_val_recall": best["val_recall"], "best_val_f1": best["val_f1"],
            "recall_floor": recall_floor, "floor_met": bool(eligible), "trials": trials}


# ---------------------------------------------------------------------------
# One-model, one-seed evaluation (H3)
# ---------------------------------------------------------------------------

def evaluate_baseline(
    model: BaselineModel,
    texts: Sequence[str],
    labels: Sequence[int],
    measure_latency_flag: bool = True,
    latency_n: int = 200,
) -> Dict[str, Any]:
    """Evaluate a fitted baseline once: metrics, latency, peak memory."""
    texts = [str(t) for t in texts]
    labels_arr = np.asarray(labels, dtype=int)
    y_prob = model.predict_proba(texts)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics_full(labels_arr, y_prob)
    result: Dict[str, Any] = {
        "model": model.name,
        "metrics": metrics,
        "n_params": model.n_params,
        "size_mb": model.size_mb,
        "peak_memory_mb": peak_memory_mb(),
        "y_pred": y_pred.tolist(),
        "y_prob": [float(v) for v in y_prob],
    }
    if measure_latency_flag:
        try:
            result["latency_cpu_ms"] = measure_latency(model.predict_proba, texts, n=latency_n)
        except Exception as exc:
            logger.warning(f"Latency measurement failed for {model.name}: {exc}")
            result["latency_cpu_ms"] = {"latency_ms_per_email_median": None, "n_timed": 0}
    return result


def evaluate_robustness(
    model: BaselineModel,
    texts: Sequence[str],
    labels: Sequence[int],
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a fitted model under the three H4 perturbations (test split)."""
    out: Dict[str, Dict[str, float]] = {}
    for kind in ("truncated", "urls_removed", "typo_synonym"):
        perturbed = apply_perturbation(texts, kind, seed=seed)
        y_prob = model.predict_proba(perturbed)[:, 1]
        metrics = compute_metrics_full(np.asarray(labels, dtype=int), y_prob)
        out[kind] = {"f1": metrics["f1"], "accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]}
    return out


def aggregate_seed_results(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed metric dicts into mean +/- std (H3, 3 seeds)."""
    if not per_seed:
        return {"status": "NOT RUN YET"}
    keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "precision_at_recall_0.90", "false_positive_rate")
    agg: Dict[str, Any] = {"n_seeds": len(per_seed), "seeds": [r.get("seed") for r in per_seed]}
    for key in keys:
        vals = [r["metrics"][key] for r in per_seed if key in r.get("metrics", {})]
        if vals:
            agg[key] = mean_std(vals)
    lat = [r.get("latency_cpu_ms", {}).get("latency_ms_per_email_median") for r in per_seed]
    lat = [v for v in lat if v is not None]
    if lat:
        agg["latency_cpu_ms_median"] = mean_std(lat)
    lat_gpu = [r.get("latency_gpu_ms", {}).get("latency_ms_per_email_median") for r in per_seed]
    lat_gpu = [v for v in lat_gpu if v is not None]
    if lat_gpu:
        agg["latency_gpu_ms_median"] = mean_std(lat_gpu)
    train_times = [r.get("train_seconds") for r in per_seed if r.get("train_seconds") is not None]
    if train_times:
        agg["train_seconds"] = mean_std(train_times)
    agg["n_params"] = per_seed[0].get("n_params")
    agg["size_mb"] = per_seed[0].get("size_mb")
    return agg


# ---------------------------------------------------------------------------
# Error analysis (H4)
# ---------------------------------------------------------------------------

def _categorise_email(text: str) -> Dict[str, Any]:
    """Categorise one email by length, URL / HTML presence and non-English content."""
    words = str(text).split()
    n_words = len(words)
    length_bucket = "short (<50 words)" if n_words < 50 else ("medium (50-200)" if n_words <= 200 else "long (>200)")
    return {
        "n_words": n_words,
        "length_bucket": length_bucket,
        "has_url": bool(URL_REGEX.search(str(text))),
        "has_html": bool(HTML_TAG_REGEX.search(str(text))),
        "non_english": bool(NON_ASCII_REGEX.search(str(text))),
    }


def error_analysis(
    y_true: Sequence[int],
    roberta_pred: Sequence[int],
    baseline_pred: Sequence[int],
    texts: Sequence[str],
    email_ids: Optional[Sequence[Any]] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Emails RoBERTa gets right but the best baseline gets wrong, and vice versa.

    Returns a DataFrame with at most ``top_n`` rows per direction, each row
    categorised by length, URL presence, HTML presence and non-English content.
    """
    y = np.asarray(y_true, dtype=int)
    r = np.asarray(roberta_pred, dtype=int)
    b = np.asarray(baseline_pred, dtype=int)
    rows: List[Dict[str, Any]] = []
    directions = [
        ("roberta_right_baseline_wrong", (r == y) & (b != y)),
        ("baseline_right_roberta_wrong", (b == y) & (r != y)),
    ]
    for direction, mask in directions:
        idxs = np.where(mask)[0]
        # Deterministic order: longest emails first (most informative errors).
        idxs = sorted(idxs.tolist(), key=lambda i: -len(str(texts[i]).split()))
        for i in idxs[:top_n]:
            cat = _categorise_email(str(texts[i]))
            rows.append({
                "direction": direction,
                "email_id": email_ids[i] if email_ids is not None else int(i),
                "true_label": int(y[i]),
                "roberta_pred": int(r[i]),
                "baseline_pred": int(b[i]),
                "text_snippet": str(texts[i])[:200],
                **cat,
            })
    return pd.DataFrame(rows, columns=[
        "direction", "email_id", "true_label", "roberta_pred", "baseline_pred",
        "text_snippet", "n_words", "length_bucket", "has_url", "has_html", "non_english",
    ])


# ---------------------------------------------------------------------------
# Explainability comparison (H5)
# ---------------------------------------------------------------------------

def evaluate_faithfulness_with_proba_fn(
    predict_proba: Callable[[List[str]], np.ndarray],
    words: List[str],
    scores: Sequence[float],
    target_class: int,
    p_orig: float,
    ks: Sequence[int] = FAITHFULNESS_KS,
) -> Dict[str, Dict[int, float]]:
    """Comprehensiveness / sufficiency at each k, for ANY predict_proba callable.

    Same maths as ``phishing_xai.faithfulness.evaluate_faithfulness_single_method``
    (``comp = p_orig - p_drop``, ``suff = p_orig - p_keep``) but driven by a
    model-agnostic ``predict_proba(texts)`` instead of a transformer pair.
    """
    from phishing_xai.faithfulness import (
        compute_instance_faithfulness,
        perturb_text_drop_indices,
        perturb_text_keep_indices,
        select_top_k_indices,
    )

    comp_by_k: Dict[int, float] = {}
    suff_by_k: Dict[int, float] = {}
    for k in ks:
        top_k = select_top_k_indices(scores, k=k)
        text_drop = perturb_text_drop_indices(words, top_k)
        text_keep = perturb_text_keep_indices(words, top_k)
        p_drop = float(predict_proba([text_drop])[0, target_class])
        p_keep = float(predict_proba([text_keep])[0, target_class])
        comp, suff = compute_instance_faithfulness(p_orig, p_drop, p_keep)
        comp_by_k[int(k)] = comp
        suff_by_k[int(k)] = suff
    return {"comprehensiveness": comp_by_k, "sufficiency": suff_by_k}


def tfidf_lr_coefficient_scores(pipeline: Pipeline, text: str) -> List[float]:
    """Word-level scores from TF-IDF+LR coefficients, aligned to ``split_words``.

    Each word's score is its TF-IDF value in this document times the
    logistic-regression coefficient for the phishing class.
    """
    words = split_words(text)
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    coef = np.asarray(clf.coef_[0], dtype=float)
    vocab = vectorizer.vocabulary_
    row = vectorizer.transform([text])
    tfidf = row.toarray()[0]
    scores: List[float] = []
    for word in words:
        key = word.lower()
        idx = vocab.get(key)
        scores.append(float(tfidf[idx] * coef[idx]) if idx is not None else 0.0)
    return scores


def lime_scores_for_text(
    predict_proba: Callable[[List[str]], np.ndarray],
    text: str,
    num_features: Optional[int] = None,
    num_samples: int = 300,
    seed: int = 42,
) -> List[float]:
    """LIME word scores for any ``predict_proba`` callable (model-agnostic).

    Uses the shared ``split_words`` splitter, exactly as the RoBERTa LIME
    explainer does, so scores line up word-for-word with faithfulness.
    """
    from lime.lime_text import LimeTextExplainer

    words = split_words(text)
    if not words:
        return []
    explainer = LimeTextExplainer(
        class_names=["Safe Email", "Phishing Email"], random_state=seed,
        split_expression=split_words,
    )
    probs = predict_proba([text])[0]
    pred_class = int(np.argmax(probs))
    n_features = num_features or min(len(words), 500)
    exp = explainer.explain_instance(
        text_instance=text, classifier_fn=predict_proba,
        num_features=n_features, num_samples=num_samples, labels=[pred_class],
    )
    score_map = {w.strip().lower(): s for w, s in exp.as_list(label=pred_class) if w.strip()}
    return [float(score_map.get(w.lower(), 0.0)) for w in words]


def integrated_gradients_scores(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    text: str,
    target_class: Optional[int] = None,
    device: Optional[torch.device] = None,
    n_steps: int = 50,
) -> List[float]:
    """Captum IntegratedGradients on the embedding layer, word-level.

    The BiLSTM/CNN tokenizer is word-level, so one token is one word and the
    shared ``split_words`` list aligns directly with the token sequence
    (truncated to ``tokenizer.max_len``).
    """
    from captum.attr import LayerIntegratedGradients

    device = device or get_device()
    model.eval()
    model.to(device)
    words = split_words(text)[: tokenizer.max_len]
    if not words:
        return []
    ids = tokenizer.encode(" ".join(words))
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    if target_class is None:
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            target_class = int(torch.argmax(logits, dim=1).item())

    def _forward(inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = model(inputs, mask)
        return torch.softmax(logits, dim=1)[:, target_class]

    lig = LayerIntegratedGradients(_forward, model.embedding)
    attributions, _ = lig.attribute(
        inputs=input_ids, baselines=torch.zeros_like(input_ids),
        additional_forward_args=(attention_mask,), n_steps=n_steps,
        return_convergence_delta=True,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy().tolist()
    scores = [float(s) for s in token_scores[: len(words)]]
    scores += [0.0] * (len(words) - len(scores))
    return scores


def evaluate_explainability(
    model: BaselineModel,
    texts: Sequence[str],
    labels: Sequence[int],
    method: str = "lime",
    sample_size: int = 100,
    seed: int = 42,
    ks: Sequence[int] = FAITHFULNESS_KS,
    pipeline: Optional[Pipeline] = None,
) -> Dict[str, Any]:
    """Mean comprehensiveness / sufficiency at each k for one baseline.

    ``method`` is ``"lime"``, ``"coefficients"`` (TF-IDF+LR only, needs the
    fitted ``pipeline``) or ``"integrated_gradients"`` (torch baselines only,
    needs the fitted model/tokenizer in ``model.extra``).
    """
    rng = random.Random(seed)
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    indices = indices[: min(sample_size, len(indices))]
    comp_accum: Dict[int, List[float]] = {int(k): [] for k in ks}
    suff_accum: Dict[int, List[float]] = {int(k): [] for k in ks}
    evaluated = 0
    for i in indices:
        text = str(texts[i])
        words = split_words(text)
        if len(words) < 2:
            continue
        probs = model.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))
        p_orig = float(probs[pred_class])
        if method == "lime":
            scores = lime_scores_for_text(model.predict_proba, text, seed=seed)
        elif method == "coefficients":
            if pipeline is None:
                raise ValueError("method='coefficients' requires the fitted TF-IDF+LR Pipeline.")
            scores = tfidf_lr_coefficient_scores(pipeline, text)
        elif method == "integrated_gradients":
            torch_model = model.extra.get("_model")
            torch_tokenizer = model.extra.get("_tokenizer")
            if torch_model is None or torch_tokenizer is None:
                raise ValueError("method='integrated_gradients' requires a fitted torch baseline.")
            scores = integrated_gradients_scores(
                torch_model, torch_tokenizer, text, target_class=pred_class,
                device=model.extra.get("_device"),
            )
        else:
            raise ValueError(f"Unknown explainability method: {method!r}")
        res = evaluate_faithfulness_with_proba_fn(
            model.predict_proba, words, scores, pred_class, p_orig, ks=ks
        )
        for k in ks:
            comp_accum[int(k)].append(res["comprehensiveness"][int(k)])
            suff_accum[int(k)].append(res["sufficiency"][int(k)])
        evaluated += 1
    return {
        "model": model.name,
        "method": method,
        "n_evaluated": evaluated,
        "comprehensiveness": {str(k): float(np.mean(v)) if v else None for k, v in comp_accum.items()},
        "sufficiency": {str(k): float(np.mean(v)) if v else None for k, v in suff_accum.items()},
    }


# ---------------------------------------------------------------------------
# RoBERTa eval-only (H2 item 7) and multimodal hook (H6)
# ---------------------------------------------------------------------------

def load_roberta_metrics(results_dir: str | Path = "results") -> Dict[str, Any]:
    """Load RoBERTa metrics from ``results/metrics.json`` if present.

    Returns ``{"status": "NOT RUN YET", "untrained_fallback": True}`` when the
    file (or the checkpoint) is missing; never retrains anything.
    """
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    checkpoint = results_dir / "best_model.pt"
    if not metrics_path.exists():
        return {
            "status": "NOT RUN YET",
            "untrained_fallback": True,
            "reason": f"RoBERTa metrics file not found: {metrics_path}",
            "checkpoint_exists": checkpoint.exists(),
        }
    data = load_json(metrics_path)
    final = data.get("final_test_metrics", data) if isinstance(data, dict) else {}
    return {
        "status": "ok" if final else "NOT RUN YET",
        "untrained_fallback": not checkpoint.exists(),
        "checkpoint_exists": checkpoint.exists(),
        "metrics": final,
        "source": str(metrics_path),
    }


def evaluate_roberta_checkpoint(
    checkpoint: str | Path,
    test_texts: Sequence[str],
    test_labels: Sequence[int],
    model_name: str = "roberta-base",
    max_length: int = 512,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate an EXISTING RoBERTa checkpoint once on the test split.

    Eval-only: no training, no gradient updates. If the checkpoint file does
    not exist, returns ``{"status": "NOT RUN YET", "untrained_fallback": True}``.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return {
            "status": "NOT RUN YET",
            "untrained_fallback": True,
            "reason": f"Checkpoint not found: {checkpoint}",
        }
    from phishing_xai.model import create_dataloader, load_tokenizer_and_model
    from phishing_xai.train import evaluate_dataloader

    device = device or get_device()
    tokenizer, model = load_tokenizer_and_model(model_name=model_name, num_labels=2, device=device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    loader = create_dataloader(
        texts=[str(t) for t in test_texts], labels=[int(y) for y in test_labels],
        tokenizer=tokenizer, batch_size=batch_size, shuffle=False, max_length=max_length,
    )
    _, y_true, y_pred, y_prob = evaluate_dataloader(model, loader, device)
    metrics = compute_metrics_full(y_true, y_prob)
    return {
        "status": "ok",
        "untrained_fallback": False,
        "model": "roberta_eval",
        "metrics": metrics,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "size_mb": float(checkpoint.stat().st_size / (1024.0 * 1024.0)),
        "y_pred": y_pred.tolist(),
        "y_prob": [float(v) for v in y_prob],
    }


# --- Multimodal hook (H6) -------------------------------------------------
# Interface for Section E (run_multimodal_eval.py --models ...):
#   evaluate_on_extractor_output(model, rows) where each row is a dict with
#   "text" (combined extractor output: subject+body+OCR+PDF+QR) and "label".
#   evaluate_multimodal(...) additionally takes per-modality text columns
#   ("body_text", "ocr_text", "pdf_text", "qr_text") and returns a per-model
#   modality-ablation table. Section E wires this by passing its extractor
#   output rows; this module never imports Section E code.

MULTIMODAL_TEXT_COLUMNS = ("body_text", "ocr_text", "pdf_text", "qr_text")


def load_multimodal_eval_csv(path: str | Path) -> pd.DataFrame:
    """Load a multimodal eval CSV.

    Required columns: ``text`` (combined extractor output) and ``label`` (0/1).
    Optional per-modality columns for ablation: ``body_text``, ``ocr_text``,
    ``pdf_text``, ``qr_text`` (missing ones are treated as empty text).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Multimodal eval CSV not found: {path}")
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Multimodal eval CSV missing required columns: {sorted(missing)}")
    return df.reset_index(drop=True)


def evaluate_on_extractor_output(
    model: BaselineModel,
    rows: Sequence[Dict[str, Any]],
    text_key: str = "text",
    label_key: str = "label",
) -> Dict[str, Any]:
    """Evaluate a FITTED baseline on extractor-output text (H6 hook).

    ``rows`` are dicts as produced by Section E's extractor (subject + body +
    OCR + PDF + QR text already combined into ``text``). Returns the standard
    H3 metric dict plus ``n``. This is the function
    ``run_multimodal_eval.py --models tfidf_lr bilstm roberta`` should call
    for the cheap baselines.
    """
    texts = [str(r.get(text_key, "")) for r in rows]
    labels = [int(r[label_key]) for r in rows]
    y_prob = model.predict_proba(texts)[:, 1]
    metrics = compute_metrics_full(np.asarray(labels, dtype=int), y_prob)
    return {"model": model.name, "n": len(texts), "metrics": metrics}


def evaluate_multimodal(
    models: Dict[str, BaselineModel],
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """Modality-ablation table: one row per (model, modality) combination.

    Modalities: ``all`` (the combined ``text`` column), ``body_text``,
    ``ocr_text``, ``pdf_text``, ``qr_text``. A modality whose column is absent
    from ``eval_df`` is reported with F1 ``None`` and status
    ``"column missing"`` rather than invented.
    """
    rows: List[Dict[str, Any]] = []
    modalities: List[Tuple[str, str]] = [("all", "text")] + [(c, c) for c in MULTIMODAL_TEXT_COLUMNS]
    for model_name, model in models.items():
        for modality, col in modalities:
            if col not in eval_df.columns:
                rows.append({"model": model_name, "modality": modality, "n": len(eval_df),
                             "f1": None, "accuracy": None, "status": "column missing"})
                continue
            texts = eval_df[col].fillna("").astype(str).tolist()
            labels = eval_df["label"].astype(int).tolist()
            y_prob = model.predict_proba(texts)[:, 1]
            metrics = compute_metrics_full(np.asarray(labels, dtype=int), y_prob)
            rows.append({"model": model_name, "modality": modality, "n": len(eval_df),
                         "f1": metrics["f1"], "accuracy": metrics["accuracy"], "status": "ok"})
    return pd.DataFrame(rows, columns=["model", "modality", "n", "f1", "accuracy", "status"])
