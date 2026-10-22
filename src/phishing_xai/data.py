"""Data loading, cleaning, deduplication, and stratified dataset splitting."""

import hashlib
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.data")

# Label mappings
LABEL_MAP = {
    "Safe Email": 0,
    "Phishing Email": 1,
}
ID2LABEL = {0: "Safe Email", 1: "Phishing Email"}


@dataclass
class DatasetSplits:
    """Container for train, validation, and test splits."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    def summary(self) -> dict:
        """Return dataset sizes and class balances."""
        return {
            "train_size": len(self.train_df),
            "train_label_dist": self.train_df["label"].value_counts().to_dict(),
            "val_size": len(self.val_df),
            "val_label_dist": self.val_df["label"].value_counts().to_dict(),
            "test_size": len(self.test_df),
            "test_label_dist": self.test_df["label"].value_counts().to_dict(),
        }


def clean_text(text: str) -> str:
    """Clean email text by lowercasing, HTML stripping, and collapsing whitespace.
    
    Filters MarkupResemblesLocatorWarning to prevent spurious warnings on URLs.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
        text = BeautifulSoup(text, "html.parser").get_text()
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_near_dup_hash(text: str, n_chars: int = 200) -> str:
    """Compute MD5 hash of first n_chars alphanumeric characters to detect template-level near-duplicates."""
    if not isinstance(text, str):
        return ""
    norm = re.sub(r"\W+", "", text[:n_chars].lower())
    if not norm:
        norm = re.sub(r"\W+", "", text.lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def truncate_text_to_model_window(
    text: str,
    tokenizer: Any,
    max_length: int = 128,
    pre_tokenization_char_cap: int = 50_000,
) -> str:
    """Truncate text to the exact character boundary that fits within the model's token window.

    1. Caps pre-tokenization character length to prevent catastrophic overhead on adversarial inputs.
    2. Encodes with return_offsets_mapping=True up to max_length tokens.
    3. Finds the end character offset of the last content token.
    4. Slices and returns text[:end_offset].
    Ensures LIME, TI/Captum, aggregation, and faithfulness see the exact same string the model processes.
    """
    if not isinstance(text, str) or not text:
        return ""
    if len(text) > pre_tokenization_char_cap:
        text = text[:pre_tokenization_char_cap]

    try:
        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = encoding.get("offset_mapping", [])
        content_ends = [end for (start, end) in offsets if start != end]
        if not content_ends:
            return text
        cut_point = max(content_ends)
        return text[:cut_point]
    except Exception as e:
        logger.warning(f"Error in truncate_text_to_model_window ({e}), returning raw text capped.")
        return text


def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load raw dataset CSV, drop nulls, clean text, and encode binary labels."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)

    # Validate required columns
    required_cols = {"Email Text", "Email Type"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns. Found: {list(df.columns)}, Required: {required_cols}")

    initial_count = len(df)
    null_count = df[["Email Text", "Email Type"]].isnull().any(axis=1).sum()
    if null_count > 0:
        logger.info(f"Dropping {null_count} rows with null values (from {initial_count} total).")
        df = df.dropna(subset=["Email Text", "Email Type"]).copy()

    # Drop Unnamed: 0 if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Map labels
    unrecognized = set(df["Email Type"].unique()) - set(LABEL_MAP.keys())
    if unrecognized:
        raise ValueError(f"Unrecognized labels in 'Email Type': {unrecognized}")

    df["label"] = df["Email Type"].map(LABEL_MAP).astype(int)

    # Clean text
    df["clean_text"] = df["Email Text"].apply(clean_text)

    # Drop any entries that became completely empty after cleaning
    empty_after_clean = (df["clean_text"].str.len() == 0).sum()
    if empty_after_clean > 0:
        logger.info(f"Dropping {empty_after_clean} rows that became empty after text cleaning.")
        df = df[df["clean_text"].str.len() > 0].copy()

    logger.info(f"Loaded and cleaned {len(df)} emails (Safe: {(df['label'] == 0).sum()}, Phishing: {(df['label'] == 1).sum()}).")
    return df.reset_index(drop=True)


def deduplicate_data(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Deduplicate emails on cleaned text to prevent train/test leakage.
    
    1. Identifies conflicting duplicates (same clean_text with multiple distinct labels).
       Drops ALL rows with conflicting labels (does NOT keep an arbitrary first row).
    2. Drops remaining exact duplicates on clean_text, keeping the first occurrence.
    Must be executed BEFORE any train/val/test splitting.
    """
    initial_count = len(df)

    # 1. Check for conflicting duplicates across labels
    grouped_labels = df.groupby(text_col)[label_col].nunique()
    conflicting_texts = set(grouped_labels[grouped_labels > 1].index)
    if conflicting_texts:
        conflict_mask = df[text_col].isin(conflicting_texts)
        conflict_rows_count = conflict_mask.sum()
        logger.warning(
            f"Conflicting duplicates: identified {len(conflicting_texts)} unique text(s) "
            f"across {conflict_rows_count} total rows with conflicting labels! "
            f"Dropping ALL copies of these conflicting rows."
        )
        df = df[~conflict_mask].copy().reset_index(drop=True)
    else:
        logger.info("No conflicting duplicate texts found across distinct labels.")

    # 2. Regular deduplication on remaining rows
    dup_mask = df.duplicated(subset=[text_col], keep="first")
    dup_count = dup_mask.sum()
    logger.info(f"Deduplication: removed {dup_count} duplicate emails from remaining pool.")

    deduped_df = df[~dup_mask].copy().reset_index(drop=True)
    total_removed = initial_count - len(deduped_df)
    logger.info(
        f"Deduplication summary: removed {total_removed} rows total ({initial_count} -> {len(deduped_df)}). "
        f"Retained unique emails: Safe={(deduped_df[label_col] == 0).sum()}, Phishing={(deduped_df[label_col] == 1).sum()}."
    )
    return deduped_df


def save_split_manifest(splits: DatasetSplits, manifest_path: Path | str = Path("results/splits.json")) -> Path:
    """Save row indices or unique IDs per split into a manifest file for reproducibility."""
    from phishing_xai.utils import save_json
    p = Path(manifest_path)
    manifest = {
        "train_indices": splits.train_df.index.tolist() if "email_id" not in splits.train_df.columns else splits.train_df["email_id"].tolist(),
        "val_indices": splits.val_df.index.tolist() if "email_id" not in splits.val_df.columns else splits.val_df["email_id"].tolist(),
        "test_indices": splits.test_df.index.tolist() if "email_id" not in splits.test_df.columns else splits.test_df["email_id"].tolist(),
        "summary": splits.summary(),
    }
    save_json(manifest, p)
    logger.info(f"Saved split manifest to: {p}")
    return p


def create_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    debug: bool = False,
    debug_samples: int = 500,
    manifest_path: Path | str | None = None,
    group_by_near_dup: bool = True,
) -> DatasetSplits:
    """Create stratified train, validation, and test splits with zero leakage.
    
    - If group_by_near_dup=True: groups on normalized first-200-char hash using StratifiedGroupKFold,
      ensuring template-level near-duplicates never cross split boundaries (0% leakage).
    - Otherwise uses standard stratified train_test_split.
    - Split 1: 80% train+val, 20% test.
    - Split 2: 80% train (64% overall), 20% val (16% overall).
    Optionally saves the split manifest to results/splits.json.
    """
    df = df.copy()
    if "email_id" not in df.columns:
        df["email_id"] = df.index

    if group_by_near_dup and "near_dup_group" not in df.columns:
        df["near_dup_group"] = df["clean_text"].apply(compute_near_dup_hash)

    if debug and len(df) > debug_samples:
        logger.info(f"Debug mode enabled: subsampling {debug_samples} emails before splitting.")
        df, _ = train_test_split(
            df,
            train_size=debug_samples,
            random_state=seed,
            stratify=df["label"],
        )
        df = df.reset_index(drop=True)

    n_splits_test = max(2, int(round(1.0 / test_size)))
    min_class_count = df["label"].value_counts().min()
    has_near_duplicates = ("near_dup_group" in df.columns) and (df["near_dup_group"].nunique() < len(df))

    if group_by_near_dup and has_near_duplicates and min_class_count >= n_splits_test:
        logger.info(f"Splitting with StratifiedGroupKFold on near-duplicate groups (test_folds={n_splits_test}).")
        sgkf1 = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
        train_val_idx, test_idx = next(sgkf1.split(df, df["label"], df["near_dup_group"]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        n_splits_val = max(2, int(round(1.0 / val_size)))
        sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
        train_idx, val_idx = next(sgkf2.split(train_val_df, train_val_df["label"], train_val_df["near_dup_group"]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
    else:
        # Initial split: train_val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["label"],
        )
        # Secondary split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=seed,
            stratify=train_val_df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    # Verification: assert zero overlap on cleaned text
    train_set = set(train_df["clean_text"])
    val_set = set(val_df["clean_text"])
    test_set = set(test_df["clean_text"])

    leakage_train_test = train_set.intersection(test_set)
    leakage_train_val = train_set.intersection(val_set)
    leakage_val_test = val_set.intersection(test_set)

    if leakage_train_test:
        raise RuntimeError(f"Train/Test leakage detected: {len(leakage_train_test)} overlapping texts!")
    if leakage_train_val:
        raise RuntimeError(f"Train/Val leakage detected: {len(leakage_train_val)} overlapping texts!")
    if leakage_val_test:
        raise RuntimeError(f"Val/Test leakage detected: {len(leakage_val_test)} overlapping texts!")

    # Verification: assert zero overlap on near-duplicate groups if enabled
    if group_by_near_dup and "near_dup_group" in train_df.columns:
        train_nd = set(train_df["near_dup_group"])
        val_nd = set(val_df["near_dup_group"])
        test_nd = set(test_df["near_dup_group"])
        if train_nd.intersection(test_nd):
            raise RuntimeError(f"Near-duplicate leakage detected between train and test: {len(train_nd.intersection(test_nd))} groups!")
        if train_nd.intersection(val_nd):
            raise RuntimeError(f"Near-duplicate leakage detected between train and val: {len(train_nd.intersection(val_nd))} groups!")
        if val_nd.intersection(test_nd):
            raise RuntimeError(f"Near-duplicate leakage detected between val and test: {len(val_nd.intersection(test_nd))} groups!")

    splits = DatasetSplits(train_df=train_df, val_df=val_df, test_df=test_df)
    logger.info(f"Created stratified splits successfully: {splits.summary()}")

    if manifest_path is not None:
        save_split_manifest(splits, manifest_path)

    return splits
