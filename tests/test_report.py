"""Unit tests for consolidated reporting: RESULTS.md and summary.json generation."""

import json
from pathlib import Path
import pytest

from phishing_xai.report import append_to_runs_log, generate_consolidated_report


def test_report_skips_debug_metrics(tmp_path: Path):
    """Ensure metrics files flagged with debug: True are ignored and show NOT RUN YET."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create debug metrics.json
    debug_metrics = {
        "debug": True,
        "best_epoch": 1,
        "final_test_metrics": {
            "accuracy": 0.999,
            "f1": 0.999,
        },
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(debug_metrics, f)

    md_path, json_path = generate_consolidated_report(results_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Must be NOT RUN YET despite debug metrics.json existing
    assert summary["final_training_and_test"] == "NOT RUN YET"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    assert "NOT RUN YET" in md_text
    assert "0.999" not in md_text


def test_report_shows_not_run_yet_for_missing_stages(tmp_path: Path):
    """Ensure missing stages report NOT RUN YET across all sections."""
    results_dir = tmp_path / "empty_results"
    results_dir.mkdir()

    md_path, json_path = generate_consolidated_report(results_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["run_metadata"] == "NOT RUN YET"
    assert summary["dataset"] == "NOT RUN YET"
    assert summary["tuning"] == "NOT RUN YET"
    assert summary["final_training_and_test"] == "NOT RUN YET"
    assert summary["explainability"]["faithfulness"] == "NOT RUN YET"
    assert summary["explainability"]["alpha_ablation"] == "NOT RUN YET"
    assert summary["explainability"]["global_words"] == "NOT RUN YET"

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert md_text.count("NOT RUN YET") >= 5


def test_summary_numbers_match_source_artifacts(tmp_path: Path):
    """Verify that numbers in summary.json strictly equal those in the source files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    real_metrics = {
        "debug": False,
        "best_epoch": 3,
        "best_val_f1": 0.9450,
        "final_test_loss": 0.1823,
        "final_test_metrics": {
            "accuracy": 0.9520,
            "f1": 0.9412,
            "recall": 0.9300,
            "precision": 0.9527,
            "roc_auc": 0.9840,
        },
        "environment": {
            "device": "cuda:0",
            "seed": 42,
            "python_version": "3.10.0",
        },
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(real_metrics, f)

    real_tuning = {
        "optimal_parameters": {"learning_rate": 2e-5, "train_batch_size": 16, "epochs": 4},
        "best_f1_score": 0.9511,
        "all_results": [
            {"iteration": 1, "params": {"learning_rate": 2e-5, "train_batch_size": 16, "epochs": 4}, "val_accuracy": 0.95, "val_f1": 0.9511}
        ],
    }
    with open(results_dir / "random_search_results.json", "w", encoding="utf-8") as f:
        json.dump(real_tuning, f)

    md_path, json_path = generate_consolidated_report(results_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Numerical parity check
    assert summary["final_training_and_test"]["final_test_loss"] == 0.1823
    assert summary["final_training_and_test"]["final_test_metrics"]["accuracy"] == 0.9520
    assert summary["final_training_and_test"]["final_test_metrics"]["f1"] == 0.9412
    assert summary["tuning"]["best_f1_score"] == 0.9511
    assert summary["tuning"]["optimal_parameters"]["train_batch_size"] == 16


def test_runs_log_appends_and_never_overwrites(tmp_path: Path):
    """Ensure runs_log.csv preserves history and appends subsequent runs."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    log_file = append_to_runs_log(
        results_dir=results_dir,
        stage="tune",
        seed=42,
        hardware="cpu",
        key_metric="val_f1=0.9200",
        path_to_artifact="results/random_search_results.json",
    )
    assert log_file.exists()

    append_to_runs_log(
        results_dir=results_dir,
        stage="train",
        seed=42,
        hardware="cpu",
        key_metric="test_f1=0.9400",
        path_to_artifact="results/best_model.pt",
    )

    import pandas as pd
    df = pd.read_csv(log_file)
    assert len(df) == 2
    assert df.iloc[0]["stage"] == "tune"
    assert df.iloc[1]["stage"] == "train"


def test_report_idempotence(tmp_path: Path):
    """Running generate_consolidated_report twice yields identical files."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    md1, json1 = generate_consolidated_report(results_dir)
    with open(md1, "r", encoding="utf-8") as f:
        content_md1 = f.read()

    md2, json2 = generate_consolidated_report(results_dir)
    with open(md2, "r", encoding="utf-8") as f:
        content_md2 = f.read()

    assert content_md1 == content_md2
