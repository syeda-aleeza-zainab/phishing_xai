"""Tests for ``scripts/run_train.py::resolve_splits`` (canonical-manifest fix).

``run_train.py`` used to regenerate splits with ``create_stratified_splits``
on every run. ``StratifiedGroupKFold`` output is not stable across
scikit-learn versions, so the Kaggle split (train 11,227 / val 2,786 /
test 3,500, sklearn 1.6.1) differs from the canonical manifest (train
11,215 / val 2,803 / test 3,495, sklearn 1.9.1). Full runs must now reuse
the manifest's exact ``email_id`` lists when ``<output_dir>/splits.json``
exists; only a missing manifest generates (and saves) fresh splits, and
debug mode never reads the manifest.

No GPU, no model, no internet needed. The real-data test uses the repo's
own ``data/Phishing_Email.csv`` + ``results/splits.json``; the synthetic
tests use a small throwaway CSV in ``tmp_path``.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.data import deduplicate_data, load_data
from scripts.run_train import resolve_splits

DATA_PATH = REPO_ROOT / "data" / "Phishing_Email.csv"
MANIFEST_PATH = REPO_ROOT / "results" / "splits.json"
CANONICAL_SIZES = {"train": 11215, "val": 2803, "test": 3495}


def _synthetic_csv(path: Path, n_per_class: int = 300) -> Path:
    """Write a small CSV with distinct texts (no near-duplicate groups)."""
    rows = []
    for i in range(n_per_class):
        rows.append({"Email Text": f"Please verify your account number {i} urgently today", "Email Type": "Phishing Email"})
        rows.append({"Email Text": f"Team lunch on Friday at noon bring your own sandwich {i}", "Email Type": "Safe Email"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _deduped_df(csv_path: Path) -> pd.DataFrame:
    return deduplicate_data(load_data(csv_path), text_col="clean_text", label_col="label")


def test_manifest_present_returns_exact_canonical_ids():
    """(a) With results/splits.json present, the exact manifest splits come back."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    before = MANIFEST_PATH.read_bytes()
    df = _deduped_df(DATA_PATH)
    splits = resolve_splits(df, DATA_PATH, REPO_ROOT / "results", seed=42, debug=False)
    assert len(splits.train_df) == CANONICAL_SIZES["train"]
    assert len(splits.val_df) == CANONICAL_SIZES["val"]
    assert len(splits.test_df) == CANONICAL_SIZES["test"]
    for name, frame in (("train", splits.train_df), ("val", splits.val_df), ("test", splits.test_df)):
        assert sorted(frame["email_id"].tolist()) == sorted(manifest[f"{name}_indices"])
    # The manifest path must not rewrite the manifest it read.
    assert MANIFEST_PATH.read_bytes() == before


def test_no_manifest_generates_and_saves_then_reuses(tmp_path):
    """(b) Without a manifest, splits are generated and saved; a second call reuses them exactly."""
    csv_path = _synthetic_csv(tmp_path / "data.csv")
    df = _deduped_df(csv_path)
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    assert not (out_dir / "splits.json").exists()
    splits = resolve_splits(df, csv_path, out_dir, seed=42, debug=False)
    manifest_path = out_dir / "splits.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert len(splits.train_df) == manifest["summary"]["train_size"]
    assert len(splits.val_df) == manifest["summary"]["val_size"]
    assert len(splits.test_df) == manifest["summary"]["test_size"]
    assert len(splits.train_df) + len(splits.val_df) + len(splits.test_df) == len(df)
    # Second call: the manifest path returns the identical email_id sets.
    again = resolve_splits(df, csv_path, out_dir, seed=42, debug=False)
    for name, frame in (("train", again.train_df), ("val", again.val_df), ("test", again.test_df)):
        assert sorted(frame["email_id"].tolist()) == sorted(manifest[f"{name}_indices"])


def test_debug_never_loads_manifest(tmp_path):
    """(c) Debug mode subsamples ~500 rows and ignores an existing manifest."""
    csv_path = _synthetic_csv(tmp_path / "data.csv", n_per_class=300)
    df = _deduped_df(csv_path)
    assert len(df) == 600
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    # First create a full (non-debug) manifest: 600 rows -> 384/96/120.
    full = resolve_splits(df, csv_path, out_dir, seed=42, debug=False)
    assert (len(full.train_df), len(full.val_df), len(full.test_df)) == (384, 96, 120)
    # Debug must return the 500-row subsample split (320/80/100), not the manifest's.
    debug_splits = resolve_splits(df, csv_path, out_dir, seed=42, debug=True)
    assert (len(debug_splits.train_df), len(debug_splits.val_df), len(debug_splits.test_df)) == (320, 80, 100)


def test_manifest_size_mismatch_raises(tmp_path):
    """A manifest whose summary disagrees with its own id lists fails loudly (ValueError)."""
    csv_path = _synthetic_csv(tmp_path / "data.csv")
    df = _deduped_df(csv_path)
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    resolve_splits(df, csv_path, out_dir, seed=42, debug=False)
    manifest_path = out_dir / "splits.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"]["train_size"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Split mismatch for train"):
        resolve_splits(df, csv_path, out_dir, seed=42, debug=False)
