"""Unit tests for data loading, cleaning, deduplication, splitting, and augmentation."""

import random
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from phishing_xai.augment import balance_training_data, synonym_replacement
from phishing_xai.data import (
    clean_text,
    compute_near_dup_hash,
    create_stratified_splits,
    deduplicate_data,
    load_data,
    truncate_text_to_model_window,
)
from phishing_xai.utils import get_device, set_seed


@pytest.fixture
def mock_csv_path(tmp_path: Path) -> Path:
    """Create a temporary CSV file with mock emails including duplicates, nulls, and HTML."""
    records = [
        {"Unnamed: 0": 0, "Email Text": "Hello please update your account password at http://fakebank.com", "Email Type": "Phishing Email"},
        {"Unnamed: 0": 1, "Email Text": "Dear colleague the weekly department meeting is at 10am tomorrow", "Email Type": "Safe Email"},
        {"Unnamed: 0": 2, "Email Text": "Dear colleague the weekly department meeting is at 10am tomorrow", "Email Type": "Safe Email"},  # Duplicate
        {"Unnamed: 0": 3, "Email Text": "<p>Urgent: Your <b>PayPal</b> security code is required!</p>\r\n", "Email Type": "Phishing Email"},
        {"Unnamed: 0": 4, "Email Text": np.nan, "Email Type": "Safe Email"},  # Null text
        {"Unnamed: 0": 5, "Email Text": "Invoice #4928 is attached for your review and approval", "Email Type": "Safe Email"},
        {"Unnamed: 0": 6, "Email Text": "Your wire transfer of $5000 is pending click here to cancel", "Email Type": "Phishing Email"},
        {"Unnamed: 0": 7, "Email Text": "Reminder: Annual benefits enrollment closes this Friday", "Email Type": "Safe Email"},
        {"Unnamed: 0": 8, "Email Text": "Security alert: unauthorized login detected from Russia", "Email Type": "Phishing Email"},
        {"Unnamed: 0": 9, "Email Text": "The team lunch will be hosted at the downtown cafe", "Email Type": "Safe Email"},
    ]
    df = pd.DataFrame(records)
    csv_file = tmp_path / "mock_emails.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_clean_text():
    raw_html = "<p>Urgent: Click <a href='http://evil.com'>here</a> now!   \r\n</p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "</a>" not in cleaned
    assert "\r" not in cleaned
    assert "  " not in cleaned
    assert cleaned == "urgent: click here now!"

    # Non-string input safety
    assert clean_text(None) == ""
    assert clean_text(12345) == ""


def test_load_data(mock_csv_path: Path):
    df = load_data(mock_csv_path)
    # The null row should have been dropped (10 rows - 1 null = 9 rows)
    assert len(df) == 9
    assert "Unnamed: 0" not in df.columns
    assert "clean_text" in df.columns
    assert "label" in df.columns
    # Verify label map: Safe Email -> 0, Phishing Email -> 1
    assert set(df["label"].unique()) == {0, 1}
    assert (df[df["Email Type"] == "Safe Email"]["label"] == 0).all()
    assert (df[df["Email Type"] == "Phishing Email"]["label"] == 1).all()


def test_deduplicate_data(mock_csv_path: Path):
    df = load_data(mock_csv_path)
    initial_len = len(df)
    deduped = deduplicate_data(df, text_col="clean_text")
    # One exact duplicate row should have been removed
    assert len(deduped) == initial_len - 1
    # Check that clean_text has no duplicates
    assert deduped["clean_text"].duplicated().sum() == 0


def test_deduplicate_conflicting_labels():
    """Ensure texts appearing with both Safe and Phishing labels have ALL copies dropped."""
    records = [
        {"clean_text": "conflicting email text", "label": 1, "Email Type": "Phishing Email"},
        {"clean_text": "conflicting email text", "label": 0, "Email Type": "Safe Email"},
        {"clean_text": "conflicting email text", "label": 1, "Email Type": "Phishing Email"},
        {"clean_text": "legitimate safe email", "label": 0, "Email Type": "Safe Email"},
        {"clean_text": "legitimate safe email", "label": 0, "Email Type": "Safe Email"},  # Regular duplicate
        {"clean_text": "malicious phishing email", "label": 1, "Email Type": "Phishing Email"},
    ]
    df = pd.DataFrame(records)
    deduped = deduplicate_data(df, text_col="clean_text", label_col="label")

    # The 3 rows with 'conflicting email text' must all be dropped!
    assert "conflicting email text" not in deduped["clean_text"].values
    # From the remaining 3 rows, 1 duplicate 'legitimate safe email' is dropped -> 2 unique rows remain
    assert len(deduped) == 2
    assert set(deduped["clean_text"].values) == {"legitimate safe email", "malicious phishing email"}


def test_save_split_manifest(tmp_path: Path):
    """Ensure split manifest saves valid indices and summary."""
    records = [{"clean_text": f"Email text {i}", "label": i % 2, "Email Type": "Safe Email" if i % 2 == 0 else "Phishing Email"} for i in range(20)]
    df = pd.DataFrame(records)
    manifest_path = tmp_path / "splits.json"

    splits = create_stratified_splits(df, seed=42, manifest_path=manifest_path)
    assert manifest_path.exists()

    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "train_indices" in data
    assert "val_indices" in data
    assert "test_indices" in data
    assert "summary" in data
    assert len(data["train_indices"]) == len(splits.train_df)
    assert len(data["val_indices"]) == len(splits.val_df)
    assert len(data["test_indices"]) == len(splits.test_df)


def test_no_train_val_test_leakage():
    """Ensure zero overlap between train, val, and test splits."""
    # Create synthetic dataset with 100 samples
    records = []
    for i in range(50):
        records.append({"Email Text": f"Phishing notification {i} click link to verify", "Email Type": "Phishing Email"})
        records.append({"Email Text": f"Safe newsletter {i} regarding company updates", "Email Type": "Safe Email"})
    df = pd.DataFrame(records)
    df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
    df["clean_text"] = df["Email Text"].apply(clean_text)

    splits = create_stratified_splits(df, test_size=0.2, val_size=0.2, seed=42)

    train_texts = set(splits.train_df["clean_text"])
    val_texts = set(splits.val_df["clean_text"])
    test_texts = set(splits.test_df["clean_text"])

    # Strict assertion: no text leakage across any split
    assert len(train_texts.intersection(test_texts)) == 0, "Leakage detected between train and test!"
    assert len(train_texts.intersection(val_texts)) == 0, "Leakage detected between train and val!"
    assert len(val_texts.intersection(test_texts)) == 0, "Leakage detected between val and test!"

    # Stratification check (approximately 50% each class)
    assert splits.train_df["label"].mean() == pytest.approx(0.5, abs=0.05)
    assert splits.val_df["label"].mean() == pytest.approx(0.5, abs=0.05)
    assert splits.test_df["label"].mean() == pytest.approx(0.5, abs=0.05)


def test_deterministic_splits():
    """Ensure identical splits are generated when using the same seed."""
    records = [{"Email Text": f"Sample email {i}", "Email Type": "Safe Email" if i % 2 == 0 else "Phishing Email"} for i in range(40)]
    df = pd.DataFrame(records)
    df["label"] = df["Email Type"].map({"Safe Email": 0, "Phishing Email": 1})
    df["clean_text"] = df["Email Text"].apply(clean_text)

    splits_a = create_stratified_splits(df, seed=123)
    splits_b = create_stratified_splits(df, seed=123)

    pd.testing.assert_frame_equal(splits_a.train_df, splits_b.train_df)
    pd.testing.assert_frame_equal(splits_a.val_df, splits_b.val_df)
    pd.testing.assert_frame_equal(splits_a.test_df, splits_b.test_df)


def test_synonym_replacement_deterministic():
    text = "the quick brown fox jumps over the lazy dog"
    rng1 = random.Random(99)
    res1 = synonym_replacement(text, replace_ratio=0.3, rng=rng1)

    rng2 = random.Random(99)
    res2 = synonym_replacement(text, replace_ratio=0.3, rng=rng2)

    assert res1 == res2, "Synonym replacement must be deterministic given identical RNG state"


def test_balance_training_data_only_touches_train():
    """Ensure data augmentation balances the training set and leaves val/test untouched."""
    # Create unbalanced train set: 30 Safe, 10 Phishing
    train_records = []
    for i in range(30):
        train_records.append({"clean_text": f"Safe discussion message number {i}", "label": 0, "Email Type": "Safe Email"})
    for i in range(10):
        train_records.append({"clean_text": f"Urgent account alert notice number {i}", "label": 1, "Email Type": "Phishing Email"})

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame([{"clean_text": "Val email", "label": 0, "Email Type": "Safe Email"}])
    test_df = pd.DataFrame([{"clean_text": "Test email", "label": 1, "Email Type": "Phishing Email"}])

    val_df_copy = val_df.copy()
    test_df_copy = test_df.copy()

    balanced_train = balance_training_data(train_df, seed=42)

    # 1. Balanced train has equal counts
    assert (balanced_train["label"] == 0).sum() == 30
    assert (balanced_train["label"] == 1).sum() == 30
    assert len(balanced_train) == 60

    # 2. Validation and test sets are completely untouched
    pd.testing.assert_frame_equal(val_df, val_df_copy)
    pd.testing.assert_frame_equal(test_df, test_df_copy)


def test_set_seed_and_device():
    set_seed(42)
    val_rand1 = random.random()
    val_np1 = np.random.rand()
    val_torch1 = torch.rand(1).item()

    set_seed(42)
    val_rand2 = random.random()
    val_np2 = np.random.rand()
    val_torch2 = torch.rand(1).item()

    assert val_rand1 == val_rand2
    assert val_np1 == val_np2
    assert val_torch1 == val_torch2

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_text_truncation_long_text():
    """Verify truncate_text_to_model_window correctly caps pre-tokenization and cuts at token boundary."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-RobertaForSequenceClassification")

    # Pathological long text (e.g. 100,000 characters)
    huge_text = "urgent security notification click here to verify credentials " * 2000
    assert len(huge_text) > 50000

    truncated = truncate_text_to_model_window(
        huge_text,
        tokenizer=tokenizer,
        max_length=32,
        pre_tokenization_char_cap=1000,
    )

    assert len(truncated) <= 1000
    assert len(truncated) > 0
    # The truncated text should encode to <= max_length tokens
    encoded = tokenizer(truncated, max_length=32, truncation=False)
    assert len(encoded["input_ids"]) <= 32


def test_near_duplicates_grouping_no_leakage():
    """Verify near-duplicate templates are grouped into the same split with zero cross-split leakage."""
    records = []
    # Create 10 distinct near-duplicate templates, each with 4 minor variations
    for template_id in range(10):
        for var_id in range(4):
            records.append({
                "clean_text": f"Template email number {template_id:04d} with some standard boiler text and unique variant ending {var_id}",
                "label": template_id % 2,
                "Email Type": "Phishing Email" if template_id % 2 == 1 else "Safe Email",
            })
    df = pd.DataFrame(records)

    splits = create_stratified_splits(df, test_size=0.2, val_size=0.2, seed=42, group_by_near_dup=True)

    # Check zero template leakage
    train_groups = set(splits.train_df["clean_text"].apply(compute_near_dup_hash))
    val_groups = set(splits.val_df["clean_text"].apply(compute_near_dup_hash))
    test_groups = set(splits.test_df["clean_text"].apply(compute_near_dup_hash))

    assert len(train_groups.intersection(test_groups)) == 0, "Near-duplicate leaked between train and test!"
    assert len(train_groups.intersection(val_groups)) == 0, "Near-duplicate leaked between train and val!"
    assert len(val_groups.intersection(test_groups)) == 0, "Near-duplicate leaked between val and test!"
