
import pandas as pd
import numpy as np
from pathlib import Path

CORPUS_FILE = Path(r"C:\Users\RommGT\Desktop\articulo4\datasets\unified\human_corpus_full.csv")
OUTPUT_FILE = Path(r"C:\Users\RommGT\Desktop\articulo4\datasets\unified\human_corpus_sampled.csv")

# Token length filter (applies to text = subject + body)
MIN_TOKENS = 30
MAX_TOKENS = 500

# Stratified sampling quotas
QUOTAS = {
    "CEAS_08":         1500,
    "TREC-07":         1500,
    "Nazario":         1000,
    "Nigerian_Fraud":   750,
    "enron_data_fraud_labeled": 250,
}
EXCLUDE = {"lingspam"}

# Reproducibility
RANDOM_SEED = 42

def main():
    if not CORPUS_FILE.exists():
        print(f"[ERROR] Corpus file not found: {CORPUS_FILE}")
        print("        Run 01_unify_datasets.py first.")
        return

    print(f"[INFO] Loading {CORPUS_FILE} ...")
    df = pd.read_csv(CORPUS_FILE)
    print(f"[INFO] Loaded {len(df):,} rows")

    # Step 1: phishing only
    before = len(df)
    df = df[df["label"] == 1].copy()
    print(f"[STEP 1] Phishing only: {len(df):,} rows (dropped {before - len(df):,} legitimate)")

    # Step 2: exclude listed sources
    before = len(df)
    df = df[~df["source"].isin(EXCLUDE)].copy()
    print(f"[STEP 2] Excluded sources {EXCLUDE}: {len(df):,} rows (dropped {before - len(df):,})")

    # Step 3: length filter
    df["n_tokens"] = df["text"].astype(str).map(lambda t: len(t.split()))
    before = len(df)
    df = df[(df["n_tokens"] >= MIN_TOKENS) & (df["n_tokens"] <= MAX_TOKENS)].copy()
    print(f"[STEP 3] Length filter [{MIN_TOKENS}-{MAX_TOKENS} tokens]: "
          f"{len(df):,} rows (dropped {before - len(df):,})")

    # Step 4: stratified sampling
    print(f"\n[STEP 4] Stratified sampling with seed={RANDOM_SEED}")
    print(f"{'Source':<30} {'Available':>10} {'Quota':>10} {'Kept':>10}")
    print("-" * 62)

    sampled_frames = []
    for source, quota in QUOTAS.items():
        subset = df[df["source"] == source]
        available = len(subset)
        if available == 0:
            print(f"{source:<30} {available:>10} {quota:>10} {0:>10}  [WARN: not in corpus]")
            continue
        if available <= quota:
            kept = subset.copy()
            print(f"{source:<30} {available:>10} {quota:>10} {len(kept):>10}  [keeping all]")
        else:
            kept = subset.sample(n=quota, random_state=RANDOM_SEED)
            print(f"{source:<30} {available:>10} {quota:>10} {len(kept):>10}")
        sampled_frames.append(kept)

    sampled = pd.concat(sampled_frames, ignore_index=True)

    # Step 5: shuffle and clean
    sampled = sampled.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    sampled = sampled.drop(columns=["n_tokens"])

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # Summary
    print(f"\n[OK] Saved -> {OUTPUT_FILE}")
    print(f"[OK] Final corpus: {len(sampled):,} phishing emails")
    print(f"\nSource distribution in final corpus:")
    print(sampled["source"].value_counts())

    print(f"\nToken length stats in final corpus:")
    n = sampled["text"].astype(str).map(lambda t: len(t.split()))
    print(f"  mean   = {n.mean():.1f}")
    print(f"  median = {n.median():.1f}")
    print(f"  std    = {n.std():.1f}")
    print(f"  min    = {n.min()}")
    print(f"  max    = {n.max()}")

if __name__ == "__main__":
    main()