"""CLI entry point for model training and single-pass test evaluation."""

import argparse
import sys
from pathlib import Path

# Add project root / src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.augment import balance_training_data
from phishing_xai.config import DataConfig, ModelConfig, TrainConfig
from phishing_xai.data import DatasetSplits, create_stratified_splits, deduplicate_data, load_data
from phishing_xai.model import create_dataloader, load_tokenizer_and_model
from phishing_xai.train import train_model
from phishing_xai.utils import get_device, get_logger, load_json, set_seed

logger = get_logger("run_train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for Phishing Email Detection")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv", help="Path to raw dataset CSV")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save checkpoints and metrics")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model identifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence token length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps for gradient accumulation")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for full reproducibility")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision (AMP)")
    parser.add_argument("--best_params_path", "--params", dest="best_params_path", type=str, default=None, help="Path to optimal parameters JSON (e.g. results/tuning/best_params.json)")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory to save large checkpoint files (can be outside OneDrive)")
    parser.add_argument("--keep_latest", action="store_true", help="Retain intermediate latest_checkpoint.pt after training finishes")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_checkpoint.pt if available")
    parser.add_argument("--debug", action="store_true", help="Debug mode: ~500 emails, max_len=128, 1 epoch")
    return parser.parse_args()



def resolve_splits(df, data_path, output_dir, seed: int = 42, debug: bool = False) -> DatasetSplits:
    """Resolve train/val/test splits, preferring the canonical manifest.

    Full (non-debug) runs reuse the exact splits in
    ``<output_dir>/splits.json`` when it already exists, via
    ``phishing_xai.baselines.load_splits`` (lazy import, so importing this
    module never pulls in the baselines module). That loader rebuilds the
    splits from the manifest's ``email_id`` lists and raises ``ValueError``
    when the rebuilt sizes or label distributions disagree with the
    manifest summary, so a stale or foreign manifest fails loudly instead
    of silently training on different splits. ``df`` is unused on the
    manifest path (the loader rebuilds from ``data_path`` itself, exactly
    as the baselines do).

    Why this exists: ``create_stratified_splits`` regenerates splits with
    ``StratifiedGroupKFold``, whose output is NOT stable across
    scikit-learn versions. The canonical manifest (train 11,215 / val
    2,803 / test 3,495, generated locally with scikit-learn 1.9.1) comes
    back as train 11,227 / val 2,786 / test 3,500 on Kaggle
    (scikit-learn 1.6.1, numpy 2.1.3). Regenerating would put RoBERTa on a
    different test set than the baselines, which already read
    ``splits.json``.

    When no manifest exists, behaviour is unchanged: splits are generated
    with ``create_stratified_splits`` and the manifest is saved. Debug
    mode never reads the manifest (it subsamples ~500 rows first, exactly
    as before) and still writes its debug manifest.
    """
    manifest_path = Path(output_dir) / "splits.json"
    if not debug and manifest_path.exists():
        from phishing_xai.baselines import load_splits

        loaded = load_splits(data_path, manifest_path, seed=seed)
        splits = DatasetSplits(
            train_df=loaded["train"], val_df=loaded["val"], test_df=loaded["test"]
        )
        logger.info(
            "Loaded canonical splits from manifest %s: train=%d val=%d test=%d",
            manifest_path, len(splits.train_df), len(splits.val_df), len(splits.test_df),
        )
        return splits
    splits = create_stratified_splits(
        df,
        test_size=0.2,
        val_size=0.2,
        seed=seed,
        debug=debug,
        debug_samples=500,
        manifest_path=manifest_path,
    )
    logger.info(
        "Generated fresh splits (debug=%s, no manifest at %s): train=%d val=%d test=%d; manifest saved to %s",
        debug, manifest_path, len(splits.train_df), len(splits.val_df), len(splits.test_df),
        manifest_path,
    )
    return splits


def main() -> None:
    args = parse_args()

    # 1. Reproducibility
    set_seed(args.seed)
    device = get_device()
    logger.info(f"Target execution device: {device}")

    # 2. Handle output directory and debug isolation
    if args.debug:
        out_dir = Path(args.output_dir)
        if out_dir.name != "debug":
            out_dir = out_dir / "debug"
        args.output_dir = str(out_dir)
        logger.info(f"Debug mode enabled: Output directed to isolated directory: {args.output_dir}")

    # 3. Handle --params / --best_params_path override if provided
    params_path = Path(args.best_params_path) if args.best_params_path else None
    if not args.debug:
        if not args.best_params_path:
            raise ValueError(
                "Production training requires tuned hyperparameters. Please supply --params "
                "(e.g., --params results/tuning/best_params.json), or pass --debug for smoke testing."
            )
        if params_path and not params_path.exists():
            raise FileNotFoundError(
                f"Specified --params file not found: {params_path}. "
                f"Ensure hyperparameter tuning has finished before starting final training."
            )

    if params_path and not params_path.exists() and args.debug:
        candidate = Path("results/debug/tuning") / params_path.name
        if candidate.exists():
            params_path = candidate
        elif (Path("results/debug") / params_path.name).exists():
            params_path = Path("results/debug") / params_path.name

    if params_path and params_path.exists():
        logger.info(f"Loading optimal parameters from: {params_path}")
        tuning_data = load_json(params_path)
        best_params = tuning_data.get("optimal_parameters", tuning_data)
        if "learning_rate" in best_params:
            args.learning_rate = float(best_params["learning_rate"])
        if "train_batch_size" in best_params:
            args.batch_size = int(best_params["train_batch_size"])
        if "epochs" in best_params:
            args.epochs = int(best_params["epochs"])
        logger.info(f"Using tuned params: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.epochs}")

    # 4. Handle --debug settings
    max_length = 128 if args.debug else args.max_length
    epochs = args.epochs if args.best_params_path else (1 if args.debug else args.epochs)
    batch_size = args.batch_size if args.best_params_path else (16 if args.debug else args.batch_size)
    eval_batch_size = 32 if args.debug else args.eval_batch_size

    if args.debug:
        logger.info("=== RUNNING IN FAST CPU DEBUG MODE (~500 samples, max_len=128, 1 epoch) ===")

    # 4. Data Loading, Deduplication, and Leakage-free Stratified Splitting
    df = load_data(args.data_path)
    df = deduplicate_data(df, text_col="clean_text", label_col="label")

    # Full runs reuse the canonical splits.json when present (see
    # resolve_splits); debug mode always regenerates from a ~500-row sample.
    splits = resolve_splits(
        df,
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed,
        debug=args.debug,
    )

    # 5. Synonym Replacement Augmentation (Balanced on Train Set ONLY!)
    balanced_train_df = balance_training_data(
        splits.train_df,
        text_col="clean_text",
        label_col="label",
        replace_ratio=0.2,
        seed=args.seed,
    )

    # 6. Load Tokenizer & Model
    tokenizer, model = load_tokenizer_and_model(
        model_name=args.model_name,
        num_labels=2,
        device=device,
    )

    # 7. Create DataLoaders with Dynamic Padding
    train_loader = create_dataloader(
        texts=balanced_train_df["clean_text"].tolist(),
        labels=balanced_train_df["label"].tolist(),
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        max_length=max_length,
    )
    val_loader = create_dataloader(
        texts=splits.val_df["clean_text"].tolist(),
        labels=splits.val_df["label"].tolist(),
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
        max_length=max_length,
    )
    test_loader = create_dataloader(
        texts=splits.test_df["clean_text"].tolist(),
        labels=splits.test_df["label"].tolist(),
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
        max_length=max_length,
    )

    # 8. Train Model & Run Single-pass Final Test Evaluation
    metrics_summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=not args.no_amp,
        early_stopping_patience=args.early_stopping_patience,
        device=device,
        output_dir=args.output_dir,
        ckpt_dir=args.ckpt_dir,
        seed=args.seed,
        is_debug=args.debug,
        resume=args.resume,
        keep_latest=args.keep_latest,
    )


    # 9. In production mode only: append to run history and update consolidated report
    if not args.debug:
        from phishing_xai.report import append_to_runs_log, generate_consolidated_report

        key_metric_str = f"test_f1={metrics_summary['final_test_metrics']['f1']:.4f}"
        append_to_runs_log(
            results_dir=args.output_dir,
            stage="train",
            seed=args.seed,
            hardware=str(device),
            key_metric=key_metric_str,
            path_to_artifact=Path(args.output_dir) / "metrics.json",
        )
        generate_consolidated_report(args.output_dir)
    else:
        logger.info("Debug run complete: isolated under results/debug (not logged to production runs_log or RESULTS.md).")


if __name__ == "__main__":
    main()
