"""CLI entry point for random search hyperparameter tuning."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.augment import balance_training_data
from phishing_xai.data import create_stratified_splits, deduplicate_data, load_data
from phishing_xai.model import load_tokenizer_and_model
from phishing_xai.tune import run_random_search
from phishing_xai.utils import get_device, get_logger, set_seed

logger = get_logger("run_tune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random Search Hyperparameter Tuning for RoBERTa")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv", help="Path to raw dataset CSV")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save tuning results")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model backbone")
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of random search trials")
    parser.add_argument("--max_length", type=int, default=256, help="Sequence max length for tuning budget")
    parser.add_argument("--subsample_train", type=int, default=4000, help="Train subsample budget size (or 0 for all)")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for validation evaluations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Quick debug tuning trial")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.debug:
        out_dir = Path(args.output_dir)
        if "debug" not in out_dir.parts:
            out_dir = out_dir / "debug"
        args.output_dir = str(out_dir)
        args.n_iterations = 2
        args.max_length = 64
        args.subsample_train = 100
        logger.info(f"Debug tuning enabled: budget={args.n_iterations} iters, max_len={args.max_length}, samples={args.subsample_train}. Output: {args.output_dir}")

    df = load_data(args.data_path)
    df = deduplicate_data(df, text_col="clean_text", label_col="label")

    splits = create_stratified_splits(
        df,
        test_size=0.2,
        val_size=0.2,
        seed=args.seed,
        debug=args.debug,
        debug_samples=200 if args.debug else 500,
    )

    balanced_train_df = balance_training_data(
        splits.train_df,
        text_col="clean_text",
        label_col="label",
        seed=args.seed,
    )

    tokenizer, _ = load_tokenizer_and_model(args.model_name, num_labels=2, device=device)

    subsample = args.subsample_train if args.subsample_train > 0 else None
    param_dists = {"train_batch_size": [16, 32], "learning_rate": [2e-5, 3e-5], "epochs": [1]} if args.debug else None

    results_dict = run_random_search(
        train_texts=balanced_train_df["clean_text"].tolist(),
        train_labels=balanced_train_df["label"].tolist(),
        val_texts=splits.val_df["clean_text"].tolist(),
        val_labels=splits.val_df["label"].tolist(),
        tokenizer=tokenizer,
        model_name=args.model_name,
        n_iterations=args.n_iterations,
        max_length=args.max_length,
        subsample_train=subsample,
        param_distributions=param_dists,
        eval_batch_size=args.eval_batch_size,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if not args.debug:
        from phishing_xai.report import append_to_runs_log, generate_consolidated_report

        key_metric_str = f"best_val_f1={results_dict.get('best_f1_score', 0.0):.4f}"
        append_to_runs_log(
            results_dir=args.output_dir,
            stage="tune",
            seed=args.seed,
            hardware=str(device),
            key_metric=key_metric_str,
            path_to_artifact=Path(args.output_dir) / "tuning" / "results.json",
        )
        generate_consolidated_report(args.output_dir)
    else:
        logger.info("Debug tuning complete: isolated under results/debug (not logged to production runs_log or RESULTS.md).")


if __name__ == "__main__":
    main()
