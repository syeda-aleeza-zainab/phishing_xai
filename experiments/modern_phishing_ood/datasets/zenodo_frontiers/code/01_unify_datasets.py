
import os
import re
import html
import pandas as pd
from pathlib import Path


DATA_DIR = Path(r"C:\Users\RommGT\Desktop\articulo4\datasets")   # <-- CHANGE THIS
OUTPUT_DIR = Path(r"C:\Users\RommGT\Desktop\articulo4\datasets\unified")   # <-- CHANGE THIS
OUTPUT_FILE = OUTPUT_DIR / "human_corpus_full.csv"

MIN_TOKENS = 10        # discard emails shorter than this
MAX_TOKENS = 5000      # discard outliers longer than this

FILE_SCHEMAS = {
    "CEAS_08.csv":                  ("subject", "body",    "label"),
    "Nazario.csv":                  ("subject", "body",    "label"),
    "Nigerian_Fraud.csv":           ("subject", "body",    "label"),
    "TREC-07.csv":                  ("subject", "body",    "label"),
    "lingspam.csv":                 ("subject", "message", "label"),
    "enron_data_fraud_labeled.csv": ("Subject", "Body",    "Label"),
}


HTML_TAG = re.compile(r"<[^>]+>")
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
WS_RE    = re.compile(r"\s+")

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)              # &amp; -> &
    text = HTML_TAG.sub(" ", text)          # remove HTML tags
    text = WS_RE.sub(" ", text).strip()     # collapse whitespace
    return text

def token_count(text):
    return len(text.split())

def load_dataset(filename, subject_col, body_col, label_col):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None

    print(f"[INFO] Loading {filename} ...")
    # encoding fallback for old corpora with mixed encodings
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    # Verify expected columns exist
    missing = [c for c in [subject_col, body_col, label_col] if c not in df.columns]
    if missing:
        print(f"[ERROR] {filename}: missing columns {missing}. Available: {list(df.columns)}")
        return None

    # Build normalized DataFrame
    out = pd.DataFrame({
        "subject": df[subject_col].fillna("").astype(str).map(clean_text),
        "body":    df[body_col].fillna("").astype(str).map(clean_text),
        "label":   pd.to_numeric(df[label_col], errors="coerce"),
        "source":  filename.replace(".csv", ""),
    })

    # Drop rows with invalid labels
    before = len(out)
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    dropped_label = before - len(out)
    if dropped_label > 0:
        print(f"  [INFO] Dropped {dropped_label} rows with invalid labels")

    # Concatenate subject + body into text field
    out["text"] = (out["subject"] + " " + out["body"]).str.strip()

    # Apply length filter
    before = len(out)
    out["n_tokens"] = out["text"].map(token_count)
    out = out[(out["n_tokens"] >= MIN_TOKENS) & (out["n_tokens"] <= MAX_TOKENS)].copy()
    out = out.drop(columns=["n_tokens"])
    dropped_len = before - len(out)
    if dropped_len > 0:
        print(f"  [INFO] Dropped {dropped_len} rows by length filter "
              f"(MIN={MIN_TOKENS}, MAX={MAX_TOKENS} tokens)")

    print(f"  [OK] {filename}: {len(out)} rows kept "
          f"(phishing={int((out['label']==1).sum())}, "
          f"legit={int((out['label']==0).sum())})")
    return out

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for filename, (sc, bc, lc) in FILE_SCHEMAS.items():
        df = load_dataset(filename, sc, bc, lc)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        print("[ERROR] No data loaded. Check DATA_DIR.")
        return

    corpus = pd.concat(frames, ignore_index=True)

    # Reorder columns
    corpus = corpus[["text", "subject", "body", "label", "source"]]

    # Drop exact duplicates by text
    before = len(corpus)
    corpus = corpus.drop_duplicates(subset=["text"]).reset_index(drop=True)
    dropped_dup = before - len(corpus)
    print(f"\n[INFO] Dropped {dropped_dup} exact duplicates")

    # Save
    corpus.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[OK] Saved unified corpus -> {OUTPUT_FILE}")
    print(f"[OK] Total rows: {len(corpus)}")
    print(f"\nLabel distribution:")
    print(corpus["label"].value_counts().rename({0: "legitimate", 1: "phishing"}))
    print(f"\nSource distribution:")
    print(corpus["source"].value_counts())

if __name__ == "__main__":
    main()