"""Download Kaggle Phishing Emails dataset from Hugging Face mirror and verify schema and row count."""

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HF_DATASET_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv"
)
EXPECTED_COLUMNS = {"Email Text", "Email Type"}
EXPECTED_LABELS = {"Safe Email", "Phishing Email"}
MIN_EXPECTED_ROWS = 17500
MAX_EXPECTED_ROWS = 19500


def download_and_verify_dataset(
    target_path: Path | str = "data/Phishing_Email.csv",
    url: str = HF_DATASET_URL,
) -> Path:
    """Download Phishing_Email.csv and verify schema, labels, and row count strictly."""
    csv_path = Path(target_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Downloading dataset from: {url}")
        print(f"Target destination: {csv_path}")
        urllib.request.urlretrieve(url, csv_path)
        print("Download completed successfully.")
    else:
        print(f"Dataset already exists at: {csv_path} (skipping download).")

    # Strict verification
    print("Verifying downloaded dataset...")
    df = pd.read_csv(csv_path)

    # 1. Column verification
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"FATAL: Downloaded CSV missing required columns: {missing_cols}. Found: {list(df.columns)}")

    # 2. Label verification
    actual_labels = set(df["Email Type"].dropna().unique())
    unexpected_labels = actual_labels - EXPECTED_LABELS
    if unexpected_labels:
        raise ValueError(f"FATAL: Unrecognized labels in 'Email Type': {unexpected_labels}. Expected: {EXPECTED_LABELS}")

    # 3. Row count verification (~18,650 rows expected)
    num_rows = len(df)
    if num_rows < MIN_EXPECTED_ROWS or num_rows > MAX_EXPECTED_ROWS:
        raise ValueError(
            f"FATAL: Dataset row count {num_rows} outside expected bounds "
            f"[{MIN_EXPECTED_ROWS}, {MAX_EXPECTED_ROWS}]."
        )

    print("VERIFICATION PASSED:")
    print(f"  Shape:       {df.shape}")
    print(f"  Columns:     {list(df.columns)}")
    print(f"  Row count:   {num_rows} (matches ~18,650 target)")
    print(f"  Label counts:\n{df['Email Type'].value_counts().to_string()}")

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify Kaggle Phishing Emails dataset")
    parser.add_argument("--output_path", type=str, default="data/Phishing_Email.csv", help="Destination file path")
    args = parser.parse_args()

    try:
        download_and_verify_dataset(target_path=args.output_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
