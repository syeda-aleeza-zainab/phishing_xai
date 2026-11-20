
import pandas as pd
from pathlib import Path

CORPUS_FILE = Path(r"C:\Users\RommGT\Desktop\articulo4\datasets\unified\human_corpus_full.csv")

def main():
    if not CORPUS_FILE.exists():
        print(f"[ERROR] Corpus file not found: {CORPUS_FILE}")
        print("        Run 01_unify_datasets.py first.")
        return

    df = pd.read_csv(CORPUS_FILE)
    df["n_tokens"] = df["text"].astype(str).map(lambda t: len(t.split()))
    df["n_chars"]  = df["text"].astype(str).map(len)

    print("=" * 70)
    print("CORPUS OVERVIEW")
    print("=" * 70)
    print(f"Total emails: {len(df):,}")
    print(f"Phishing:     {int((df['label']==1).sum()):,}")
    print(f"Legitimate:   {int((df['label']==0).sum()):,}")

    print("\n" + "=" * 70)
    print("BY SOURCE × LABEL")
    print("=" * 70)
    pivot = df.pivot_table(index="source", columns="label",
                           values="text", aggfunc="count", fill_value=0)
    pivot.columns = ["legitimate" if c == 0 else "phishing" for c in pivot.columns]
    pivot["total"] = pivot.sum(axis=1)
    print(pivot.sort_values("total", ascending=False))

    print("\n" + "=" * 70)
    print("TEXT LENGTH STATISTICS (tokens) — PHISHING ONLY")
    print("=" * 70)
    phish = df[df["label"] == 1]
    print(phish.groupby("source")["n_tokens"].describe()[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(1))

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR SUB-SAMPLING")
    print("=" * 70)
    n_phish = int((df["label"] == 1).sum())
    print(f"You have {n_phish:,} phishing emails total across all sources.")
    if n_phish > 5000:
        print(f"  -> Consider stratified sub-sampling to 3,000-5,000 for balanced training.")
    elif n_phish > 1500:
        print(f"  -> Use all phishing samples; size is already manageable.")
    else:
        print(f"  -> Sample is small; use all phishing emails.")

if __name__ == "__main__":
    main()