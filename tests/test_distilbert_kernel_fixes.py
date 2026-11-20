"""Regression tests for the two DistilBERT kernel-script bugs (2026-11-20).

Bug 1 — the final-training kernel's summary print loop read
``m['accuracy']`` etc. on a per-seed metrics dict, but the test metrics
live nested under ``m['test_metrics']`` (top-level keys are best_epoch /
best_val_f1 / train_seconds / size_mb / ...). The kernel errored AFTER
all 3 seeds had finished training. The fixed loop reads
``m['test_metrics'][...]``; this test runs the fixed loop's key accesses
against the REAL downloaded seed-42 metrics.json and asserts the known
values, and asserts the old accesses raise KeyError.

Bug 2 — distilbert_text_stats.py read ``test["text"]``, but
``load_splits`` returns frames from load_data + deduplicate_data with
``text_col="clean_text"`` — there is no ``"text"`` column. The fixed
loader reads ``test["clean_text"]``; this test calls ``load_splits`` on
the real data and asserts the column layout and test size.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent  # ~/workspace — no absolute home paths
SEED42_METRICS = (
    WORKSPACE_ROOT
    / "kaggle_downloads"
    / "distilbert_train"
    / "results"
    / "distilbert"
    / "seed_42"
    / "metrics.json"
)


def _fixed_print_loop(seed, m):
    """The fixed Kernel 2 summary print loop (key accesses only)."""
    t = m["test_metrics"]
    return (
        f"seed {seed}: acc={t['accuracy']:.4f} P={t['precision']:.4f} "
        f"R={t['recall']:.4f} F1={t['f1']:.4f} ROC-AUC={t['roc_auc']:.4f} "
        f"PR-AUC={t['pr_auc']:.4f} FPR={t['false_positive_rate']:.4f} "
        f"best_epoch={m['best_epoch']} train_s={m['train_seconds']:.0f} "
        f"size_MB={m['size_mb']:.1f}"
    )


def test_bug1_fixed_print_loop_keys_resolve_on_real_metrics():
    m = json.loads(SEED42_METRICS.read_text())
    # Old (buggy) accesses raise KeyError on the real dict.
    with pytest.raises(KeyError):
        _ = m["accuracy"]
    with pytest.raises(KeyError):
        _ = m["precision"]
    # Fixed accesses resolve and match the verified seed-42 numbers.
    assert m["test_metrics"]["accuracy"] == pytest.approx(0.9805, abs=1e-4)
    assert m["test_metrics"]["precision"] == pytest.approx(0.9868, abs=1e-4)
    assert m["test_metrics"]["recall"] == pytest.approx(0.9614, abs=1e-4)
    assert m["test_metrics"]["f1"] == pytest.approx(0.9739, abs=1e-4)
    assert m["test_metrics"]["roc_auc"] == pytest.approx(0.9987, abs=1e-4)
    assert m["test_metrics"]["false_positive_rate"] == pytest.approx(0.0078, abs=1e-4)
    assert m["test_metrics"]["n"] == 3495
    assert m["best_epoch"] == 3
    assert m["size_mb"] == pytest.approx(255.45, abs=0.01)
    line = _fixed_print_loop(42, m)
    assert line.startswith("seed 42: acc=0.9805 P=0.9868 R=0.9614 F1=0.9739")


def test_bug1_kernel_script_uses_nested_keys():
    src = (
        WORKSPACE_ROOT / "kaggle_kernels" / "distilbert_final" / "distilbert_final.py"
    ).read_text()
    assert "m[\"test_metrics\"]" in src
    assert "m['accuracy']" not in src
    assert "m['precision']" not in src


def test_bug2_text_stats_loader_reads_clean_text():
    from phishing_xai.baselines import load_splits

    splits = load_splits(
        str(REPO_ROOT / "data/Phishing_Email.csv"),
        str(REPO_ROOT / "results/splits.json"),
    )
    test = splits["test"]
    assert len(test) == 3495
    assert "clean_text" in test.columns
    assert "text" not in test.columns
    assert "label" in test.columns
    texts = test["clean_text"].astype(str).tolist()
    assert len(texts) == 3495
    assert all(isinstance(t, str) and len(t) > 0 for t in texts[:50])
    # The old access must fail on the real frame (documents the bug).
    with pytest.raises(KeyError):
        _ = test["text"]


def test_bug2_stats_script_uses_clean_text():
    for p in (
        WORKSPACE_ROOT
        / "kaggle_kernels"
        / "distilbert_text_evals"
        / "distilbert_text_stats.py",
        WORKSPACE_ROOT
        / "kaggle_kernels"
        / "distilbert_text_evals"
        / "distilbert_text_evals.py",
    ):
        src = p.read_text()
        assert 'test["clean_text"]' in src
        assert 'test["text"]' not in src
