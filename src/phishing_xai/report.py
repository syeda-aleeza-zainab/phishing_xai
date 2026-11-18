"""Consolidated reporting: reads artifacts from results/ and generates RESULTS.md and summary.json."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phishing_xai.utils import ensure_dir, get_logger, load_json, save_json

logger = get_logger("phishing_xai.report")

RUNS_LOG_COLUMNS = [
    "timestamp",
    "stage",
    "seed",
    "hardware",
    "key_metric",
    "path_to_artifact",
]


def append_to_runs_log(
    results_dir: Path | str,
    stage: str,
    seed: int,
    hardware: str,
    key_metric: str,
    path_to_artifact: Path | str,
) -> Path:
    """Append a completed production stage entry to results/runs_log.csv without overwriting prior entries.
    
    IMPORTANT: Debug runs must NEVER be logged to runs_log.csv.
    """
    if "debug" in stage.lower() or "debug" in str(results_dir).lower():
        logger.info(f"Skipping runs_log append for debug stage: {stage}")
        return Path(results_dir) / "runs_log.csv"

    r_dir = ensure_dir(results_dir)
    log_file = r_dir / "runs_log.csv"

    file_exists = log_file.exists() and log_file.stat().st_size > 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [
        timestamp,
        stage,
        str(seed),
        str(hardware),
        str(key_metric),
        str(Path(path_to_artifact)),
    ]

    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(RUNS_LOG_COLUMNS)
        writer.writerow(row)

    logger.info(f"Appended entry to runs log: {log_file} -> [{stage}]")
    return log_file


def _safe_load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely load JSON file if it exists and is non-empty; returns None on failure."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return None
    try:
        data = load_json(file_path)
        return data if isinstance(data, dict) else None
    except Exception as e:
        logger.warning(f"Could not parse {file_path}: {e}")
        return {"_corrupted": True, "_error": str(e)}


def generate_consolidated_report(
    results_dir: Path | str = Path("results"),
) -> Tuple[Path, Path]:
    """Generate results/RESULTS.md and results/summary.json from non-debug artifacts.
    
    Refuses to execute inside results/debug/ and skips any metrics file with 'debug': True.
    Displays 'NOT RUN YET' for missing stages.
    """
    res_dir = Path(results_dir)
    if res_dir.name.lower() == "debug" or "debug" in res_dir.parts:
        raise ValueError(f"generate_consolidated_report must not be run on a debug directory: {res_dir}")

    ensure_dir(res_dir)

    # 1. Load source files (never check results/debug/)
    metrics_path = res_dir / "metrics.json"
    splits_path = res_dir / "splits.json"
    tune_json_path = res_dir / "tuning" / "best_params.json"
    if not tune_json_path.exists():
        tune_json_path = res_dir / "random_search_results.json"
    tune_results_path = res_dir / "tuning" / "results.json"

    faithfulness_path = res_dir / "faithfulness.json"
    if not faithfulness_path.exists():
        faithfulness_path = res_dir / "faithfulness_results.json"

    ablation_path = res_dir / "alpha_ablation.json"
    if not ablation_path.exists():
        ablation_path = res_dir / "alpha_ablation_results.json"

    global_words_path = res_dir / "global_explanations.json"
    if not global_words_path.exists():
        global_words_path = res_dir / "global_top_words.json"

    metrics_raw = _safe_load_json(metrics_path)
    splits_data = _safe_load_json(splits_path)
    tune_data = _safe_load_json(tune_json_path)
    tune_history = _safe_load_json(tune_results_path)
    faith_data = _safe_load_json(faithfulness_path)
    ablation_data = _safe_load_json(ablation_path)
    words_data = _safe_load_json(global_words_path)

    # Strictly skip debug metrics
    is_debug_metrics = False
    final_metrics = None
    if metrics_raw is not None:
        if metrics_raw.get("_corrupted"):
            final_metrics = metrics_raw
        elif metrics_raw.get("debug") is True or metrics_raw.get("environment", {}).get("config", {}).get("debug") is True:
            is_debug_metrics = True
            final_metrics = None
        else:
            final_metrics = metrics_raw

    # Build limitations dynamically
    limitations: List[str] = [
        "Evaluations conducted with deterministic seed; variance across random seeds may exist.",
        "Deduplication removes duplicate texts and conflicting labels (e.g. 'empty' placeholders) to ensure clean train/test boundary.",
        "Synonym-replacement augmentation balances class frequencies on the training split only.",
    ]

    # Only print tuning proxy limitation if tuning results exist
    if tune_data and not tune_data.get("_corrupted"):
        p_len = tune_data.get("max_length") or tune_data.get("search_space", {}).get("max_length", "proxy")
        p_sub = tune_data.get("subsample_train") or tune_data.get("search_space", {}).get("subsample_train", "subsampled")
        limitations.append(
            f"Hyperparameter tuning utilized a proxy sequence length ({p_len}) and subsampled training set ({p_sub}) to conserve compute."
        )

    summary: Dict[str, Any] = {
        "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_metadata": None,
        "dataset": None,
        "tuning": None,
        "final_training_and_test": None,
        "explainability": {
            "faithfulness": None,
            "alpha_ablation": None,
            "global_words": None,
        },
        "limitations_and_notes": limitations,
    }

    md_lines: List[str] = [
        "# Consolidated Experimental Results & Evaluation Report",
        "",
        f"*Generated on: {summary['report_generated_at']}*",
        "",
    ]

    # --- Section A: Run Metadata ---
    md_lines.append("## 1. Run Metadata")
    env_source = None
    if final_metrics and not final_metrics.get("_corrupted"):
        env_source = final_metrics.get("environment", {})
    elif tune_data and not tune_data.get("_corrupted"):
        env_source = tune_data.get("environment", {})

    if env_source:
        summary["run_metadata"] = {
            "python_version": env_source.get("python_version", "Unknown"),
            "platform": env_source.get("platform", "Unknown"),
            "hardware": env_source.get("device", "Unknown"),
            "cuda_available": env_source.get("cuda_available", False),
            "seed": env_source.get("seed", 42),
            "library_versions": env_source.get("library_versions", {}),
            "config": env_source.get("config", {}),
        }
        meta = summary["run_metadata"]
        md_lines.extend([
            f"- **Execution Hardware / Device**: `{meta['hardware']}` (CUDA available: `{meta['cuda_available']}`)",
            f"- **Platform**: `{meta['platform']}`",
            f"- **Python Version**: `{meta['python_version']}`",
            f"- **Deterministic Seed**: `{meta['seed']}`",
            "- **Library Versions**:",
        ])
        for lib, ver in meta["library_versions"].items():
            md_lines.append(f"  - `{lib}`: `{ver}`")
        if meta["config"]:
            md_lines.append("- **Training Configuration**:")
            for k, v in meta["config"].items():
                md_lines.append(f"  - `{k}`: `{v}`")
    else:
        summary["run_metadata"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET* (Production training/tuning metadata not yet available).")
    md_lines.append("")

    # --- Section B: Dataset ---
    md_lines.append("## 2. Dataset Partitioning & Hygiene")
    if splits_data and not splits_data.get("_corrupted") and "summary" in splits_data:
        sp_sum = splits_data["summary"]
        summary["dataset"] = {
            "raw_rows": splits_data.get("raw_rows"),
            "nulls_dropped": splits_data.get("nulls_dropped"),
            "conflicting_rows_dropped": splits_data.get("conflicting_rows_dropped"),
            "conflicting_unique_texts": splits_data.get("conflicting_unique_texts", []),
            "exact_duplicates_dropped": splits_data.get("exact_duplicates_dropped"),
            "total_unique_retained": splits_data.get("total_unique_retained"),
            "augmentation_samples_added_to_train": splits_data.get("augmentation_samples_added_to_train"),
            "final_balanced_train_size": splits_data.get("final_balanced_train_size"),
            "splits_summary": sp_sum,
            "manifest_file": str(splits_path),
        }

        # Format dataset section
        md_lines.extend([
            "### Preprocessing & Deduplication Hygiene",
            f"- **Raw Rows Loaded**: {splits_data.get('raw_rows', 'Unknown')}",
            f"- **Null Rows Dropped**: {splits_data.get('nulls_dropped', 'Unknown')}",
            f"- **Conflicting-Label Rows Dropped**: {splits_data.get('conflicting_rows_dropped', 'Unknown')} (all copies removed; texts: `{splits_data.get('conflicting_unique_texts', [])}`)",
            f"- **Exact Duplicate Rows Dropped**: {splits_data.get('exact_duplicates_dropped', 'Unknown')}",
            f"- **Total Unique Cleaned Emails Retained**: {splits_data.get('total_unique_retained', 'Unknown')}",
            f"- **WordNet Synonym Augmentations Added (Train Only)**: {splits_data.get('augmentation_samples_added_to_train', 'Unknown')}",
            f"- **Final Balanced Training Set Size**: {splits_data.get('final_balanced_train_size', 'Unknown')}",
            "",
            "### Split Partitions (Double Stratified, Zero Leakage)",
            "",
            "| Split | Total Samples | Safe Emails (Label 0) | Phishing Emails (Label 1) | Positive Ratio |",
            "|---|---|---|---|---|",
        ])
        for split_name, size_key, dist_key in [
            ("Train (pre-augmentation)", "train_size", "train_label_dist"),
            ("Validation", "val_size", "val_label_dist"),
            ("Test", "test_size", "test_label_dist"),
        ]:
            total = sp_sum.get(size_key, 0)
            dist = sp_sum.get(dist_key, {})
            safe = dist.get(0, dist.get("0", 0))
            phish = dist.get(1, dist.get("1", 0))
            ratio = f"{(phish / total * 100):.2f}%" if total > 0 else "0.0%"
            md_lines.append(f"| **{split_name}** | {total} | {safe} | {phish} | {ratio} |")
        md_lines.append("")
        md_lines.append(f"- **Split Manifest**: Saved row IDs to [`{splits_path}`]({splits_path.name})")
    elif splits_data and splits_data.get("_corrupted"):
        summary["dataset"] = "CORRUPTED"
        md_lines.append(f"*ERROR: Dataset splits manifest is corrupted ({splits_data.get('_error')}).*")
    else:
        summary["dataset"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET*")
    md_lines.append("")

    # --- Section C: Hyperparameter Tuning ---
    md_lines.append("## 3. Hyperparameter Tuning (Random Search)")
    if tune_data and not tune_data.get("_corrupted"):
        summary["tuning"] = {
            "search_space": tune_data.get("search_space", {}),
            "proxy_max_length": tune_data.get("max_length"),
            "proxy_subsample_train": tune_data.get("subsample_train"),
            "optimal_parameters": tune_data.get("optimal_parameters", {}),
            "best_f1_score": tune_data.get("best_f1_score", 0.0),
            "all_trials_count": len(tune_data.get("all_results", [])),
        }
        md_lines.extend([
            f"- **Optimal Parameters Found**: `{tune_data.get('optimal_parameters')}`",
            f"- **Best Validation F1 Score**: `{tune_data.get('best_f1_score'):.4f}`",
            f"- **Search Space**: `{tune_data.get('search_space')}`",
            f"- **Proxy Settings**: max_length={tune_data.get('max_length')}, subsample_train={tune_data.get('subsample_train')}",
            "",
            "### Top Trials Table",
            "",
            "| Trial | Learning Rate | Train Batch Size | Epochs | Val Accuracy | Val F1 |",
            "|---|---|---|---|---|---|",
        ])
        results_list = tune_data.get("all_results", [])
        sorted_trials = sorted(results_list, key=lambda x: x.get("val_f1", 0.0), reverse=True)
        for trial in sorted_trials[:5]:
            p = trial.get("params", {})
            md_lines.append(
                f"| {trial.get('iteration', '-')} "
                f"| {p.get('learning_rate', '-')} "
                f"| {p.get('train_batch_size', '-')} "
                f"| {p.get('epochs', '-')} "
                f"| {trial.get('val_accuracy', 0.0):.4f} "
                f"| {trial.get('val_f1', 0.0):.4f} |"
            )
    elif tune_data and tune_data.get("_corrupted"):
        summary["tuning"] = "CORRUPTED"
        md_lines.append(f"*ERROR: Tuning results file is corrupted ({tune_data.get('_error')}).*")
    else:
        summary["tuning"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET* (Random search tuning has not yet been executed in production mode).")
    md_lines.append("")

    # --- Section D: Final Training & Test Evaluation ---
    md_lines.append("## 4. Final Model Training & Test Evaluation")
    if final_metrics and not final_metrics.get("_corrupted"):
        summary["final_training_and_test"] = {
            "best_epoch": final_metrics.get("best_epoch"),
            "best_val_f1": final_metrics.get("best_val_f1"),
            "final_test_loss": final_metrics.get("final_test_loss"),
            "final_test_metrics": final_metrics.get("final_test_metrics", {}),
            "artifacts": final_metrics.get("artifacts", {}),
        }
        tm = final_metrics.get("final_test_metrics", {})
        md_lines.extend([
            f"- **Best Checkpoint Epoch**: `{final_metrics.get('best_epoch')}` (Validation F1: `{final_metrics.get('best_val_f1'):.4f}`)",
            f"- **Test Set Loss**: `{final_metrics.get('final_test_loss'):.4f}`",
            f"- **Test Accuracy**: `{tm.get('accuracy', 0.0) * 100:.2f}%`",
            f"- **Test F1 Score**: `{tm.get('f1', 0.0):.4f}`",
            f"- **Test Recall**: `{tm.get('recall', 0.0):.4f}`",
            f"- **Test Precision**: `{tm.get('precision', 0.0):.4f}`",
            f"- **Test ROC-AUC**: `{tm.get('roc_auc', 0.0):.4f}`",
            "",
            "### Classification Report",
            "```",
            str(final_metrics.get("classification_report", "")).strip(),
            "```",
            "",
            "### Evaluation Figures",
            "- Confusion Matrix: [`confusion_matrix.png`](confusion_matrix.png)",
            "- ROC-AUC Curve: [`roc_auc_curve.png`](roc_auc_curve.png)",
            "- Training & Validation Curves: [`loss_and_f1_curves.png`](loss_and_f1_curves.png)",
        ])
    elif final_metrics and final_metrics.get("_corrupted"):
        summary["final_training_and_test"] = "CORRUPTED"
        md_lines.append(f"*ERROR: Final metrics file is corrupted ({final_metrics.get('_error')}).*")
    elif is_debug_metrics:
        summary["final_training_and_test"] = "NOT RUN YET"
        md_lines.append(
            "*NOT RUN YET* (Notice: Existing metrics file corresponds to a `--debug` smoke test. "
            "Debug runs are excluded per reporting protocol until a full production run is executed)."
        )
    else:
        summary["final_training_and_test"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET* (Final model training and single-pass test evaluation pending).")
    md_lines.append("")

    # --- Section E: Explainability & Faithfulness ---
    md_lines.append("## 5. Explainability & Faithfulness Evaluation")

    # Faithfulness subsection
    md_lines.append("### 5.1 Faithfulness Benchmarks (Comprehensiveness & Sufficiency)")
    if faith_data and not faith_data.get("_corrupted"):
        summary["explainability"]["faithfulness"] = faith_data
        md_lines.extend([
            "Evaluation across test samples (mean +/- std):",
            "",
            "| Method | Comp@1 | Comp@3 | Comp@5 | Suff@1 | Suff@3 | Suff@5 |",
            "|---|---|---|---|---|---|---|",
        ])
        results_map = faith_data.get("summary", faith_data.get("methods", {}))
        for method_name, vals in results_map.items():
            if "comprehensiveness_mean" in vals:
                cm = vals.get("comprehensiveness_mean", {})
                cs = vals.get("comprehensiveness_std", {})
                sm = vals.get("sufficiency_mean", {})
                ss = vals.get("sufficiency_std", {})
                c1 = f"{cm.get('1', 0.0):.4f} (+/- {cs.get('1', 0.0):.4f})"
                c3 = f"{cm.get('3', 0.0):.4f} (+/- {cs.get('3', 0.0):.4f})"
                c5 = f"{cm.get('5', 0.0):.4f} (+/- {cs.get('5', 0.0):.4f})"
                s1 = f"{sm.get('1', 0.0):.4f} (+/- {ss.get('1', 0.0):.4f})"
                s3 = f"{sm.get('3', 0.0):.4f} (+/- {ss.get('3', 0.0):.4f})"
                s5 = f"{sm.get('5', 0.0):.4f} (+/- {ss.get('5', 0.0):.4f})"
            else:
                c1 = vals.get("comp_1", "-")
                c3 = vals.get("comp_3", "-")
                c5 = vals.get("comp_5", "-")
                s1 = vals.get("suff_1", "-")
                s3 = vals.get("suff_3", "-")
                s5 = vals.get("suff_5", "-")
            md_lines.append(f"| **{method_name}** | {c1} | {c3} | {c5} | {s1} | {s3} | {s5} |")
    else:
        summary["explainability"]["faithfulness"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET*")
    md_lines.append("")

    # Alpha ablation subsection
    md_lines.append("### 5.2 Alpha Parameter Ablation Study (LITA Weighting)")
    if ablation_data and not ablation_data.get("_corrupted"):
        summary["explainability"]["alpha_ablation"] = ablation_data
        md_lines.extend([
            "Faithfulness metric variation across alpha values and normalization variants (k=3):",
            "",
            "| Variant | Alpha (TI weight) | Comprehensiveness@3 | Sufficiency@3 |",
            "|---|---|---|---|",
        ])
        if "summary" in ablation_data:
            for variant, alphas_dict in ablation_data["summary"].items():
                for alpha_str, m in alphas_dict.items():
                    comp = f"{m.get('comprehensiveness_mean', 0.0):.4f} (+/- {m.get('comprehensiveness_std', 0.0):.4f})"
                    suff = f"{m.get('sufficiency_mean', 0.0):.4f} (+/- {m.get('sufficiency_std', 0.0):.4f})"
                    md_lines.append(f"| {variant} | {alpha_str} | {comp} | {suff} |")
        else:
            for row in ablation_data.get("ablation_table", []):
                md_lines.append(f"| - | {row.get('alpha')} | {row.get('comp')} | {row.get('suff')} |")
    else:
        summary["explainability"]["alpha_ablation"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET*")
    md_lines.append("")

    # Global top words subsection
    md_lines.append("### 5.3 Global Explanations (Top Attributed Words)")
    if words_data and not words_data.get("_corrupted"):
        summary["explainability"]["global_words"] = words_data
        if isinstance(words_data, dict) and "Phishing Email" in words_data:
            phish_words = [f"{item['word']} ({item['total_score']})" for item in words_data.get("Phishing Email", [])[:15]]
            safe_words = [f"{item['word']} ({item['total_score']})" for item in words_data.get("Safe Email", [])[:15]]
            md_lines.extend([
                f"- **Top Phishing Words**: `{', '.join(phish_words)}`",
                f"- **Top Safe Words**: `{', '.join(safe_words)}`",
            ])
        else:
            md_lines.extend([
                f"- **Top Phishing Words**: `{', '.join(words_data.get('top_phishing_words', []))}`",
                f"- **Top Safe Words**: `{', '.join(words_data.get('top_safe_words', []))}`",
            ])
    else:
        summary["explainability"]["global_words"] = "NOT RUN YET"
        md_lines.append("*NOT RUN YET*")
    md_lines.append("")

    # --- Section F: Limitations & Notes ---
    md_lines.append("## 6. Limitations & Methodological Caveats")
    for note in summary["limitations_and_notes"]:
        md_lines.append(f"- {note}")
    md_lines.append("")

    # Write files
    results_md_path = res_dir / "RESULTS.md"
    summary_json_path = res_dir / "summary.json"

    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    save_json(summary, summary_json_path)

    logger.info(f"Consolidated reports successfully generated: {results_md_path} and {summary_json_path}")
    return results_md_path, summary_json_path
