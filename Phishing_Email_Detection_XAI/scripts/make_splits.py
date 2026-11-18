"""Script to create reproducible stratified dataset splits on the full dataset without training."""

import argparse
import sys
from pathlib import Path

# Add project root / src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.data import (
    create_stratified_splits,
    deduplicate_data,
    load_data,
    save_split_manifest,
)
from phishing_xai.report import generate_consolidated_report
from phishing_xai.utils import ensure_dir, get_logger, set_seed

logger = get_logger("make_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full dataset stratified splits manifest")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv", help="Path to raw dataset CSV")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save splits.json")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation fraction of train+val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def make_splits(
    data_path: Path | str = "data/Phishing_Email.csv",
    output_dir: Path | str = "results",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Load full dataset, drop nulls and conflicting texts, deduplicate, split, and save manifest."""
    set_seed(seed)
    out_dir = ensure_dir(output_dir)
    manifest_path = out_dir / "splits.json"

    # 1. Load raw data and count nulls
    import pandas as pd
    raw_df = pd.read_csv(data_path)
    raw_rows = len(raw_df)
    null_rows = raw_df[["Email Text", "Email Type"]].isnull().any(axis=1).sum()

    df = load_data(data_path)
    cleaned_rows = len(df)

    # 2. Deduplicate: drop all conflicting labels and exact duplicates
    grouped_labels = df.groupby("clean_text")["label"].nunique()
    conflicting_texts = set(grouped_labels[grouped_labels > 1].index)
    conflicting_rows = df["clean_text"].isin(conflicting_texts).sum()

    df_dedup = deduplicate_data(df, text_col="clean_text", label_col="label")
    final_unique_rows = len(df_dedup)
    exact_duplicates_dropped = (cleaned_rows - conflicting_rows) - final_unique_rows

    # 3. Create full stratified splits (debug=False)
    splits = create_stratified_splits(
        df_dedup,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
        debug=False,
    )

    # 4. Compute train augmentation requirement
    safe_train = (splits.train_df["label"] == 0).sum()
    phish_train = (splits.train_df["label"] == 1).sum()
    samples_to_augment = max(0, safe_train - phish_train)

    # 5. Build rich metadata manifest
    manifest_data = {
        "raw_rows": int(raw_rows),
        "nulls_dropped": int(null_rows),
        "conflicting_rows_dropped": int(conflicting_rows),
        "conflicting_unique_texts": list(conflicting_texts),
        "exact_duplicates_dropped": int(exact_duplicates_dropped),
        "total_unique_retained": int(final_unique_rows),
        "augmentation_samples_added_to_train": int(samples_to_augment),
        "final_balanced_train_size": int(len(splits.train_df) + samples_to_augment),
        "summary": {
            "train_size": int(len(splits.train_df)),
            "train_label_dist": {0: int(safe_train), 1: int(phish_train)},
            "val_size": int(len(splits.val_df)),
            "val_label_dist": {
                0: int((splits.val_df["label"] == 0).sum()),
                1: int((splits.val_df["label"] == 1).sum()),
            },
            "test_size": int(len(splits.test_df)),
            "test_label_dist": {
                0: int((splits.test_df["label"] == 0).sum()),
                1: int((splits.test_df["label"] == 1).sum()),
            },
        },
        "train_indices": splits.train_df["email_id"].tolist(),
        "val_indices": splits.val_df["email_id"].tolist(),
        "test_indices": splits.test_df["email_id"].tolist(),
    }

    from phishing_xai.utils import save_json
    save_json(manifest_data, manifest_path)
    logger.info(f"Full dataset splits manifest successfully saved to: {manifest_path}")

    return manifest_data


def main() -> None:
    args = parse_args()
    manifest_data = make_splits(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    print("\n" + "=" * 65)
    print("FULL DATASET SPLITS REPORT")
    print("=" * 65)
    print(f"Raw rows:                   {manifest_data['raw_rows']}")
    print(f"Null rows dropped:          {manifest_data['nulls_dropped']}")
    print(f"Conflicting rows dropped:   {manifest_data['conflicting_rows_dropped']} (texts: {manifest_data['conflicting_unique_texts']})")
    print(f"Exact duplicates dropped:   {manifest_data['exact_duplicates_dropped']}")
    print(f"Total unique rows retained: {manifest_data['total_unique_retained']}")
    print(f"Augmentation count (train): {manifest_data['augmentation_samples_added_to_train']}")
    print(f"Final balanced train size:  {manifest_data['final_balanced_train_size']}")
    print("\nSplit counts:")
    for split_name in ["train", "val", "test"]:
        size = manifest_data["summary"][f"{split_name}_size"]
        dist = manifest_data["summary"][f"{split_name}_label_dist"]
        print(f"  {split_name.capitalize():<6}: {size} (Safe: {dist[0]}, Phishing: {dist[1]})")
    print("=" * 65)

    # Re-generate consolidated report with full-data numbers
    generate_consolidated_report(args.output_dir)


if __name__ == "__main__":
    main()
