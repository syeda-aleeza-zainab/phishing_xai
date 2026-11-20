
import re
import pandas as pd
from pathlib import Path
import config

OUTPUT_FILE = config.LLM_DIR / "llm_corpus_sampled.csv"

# Soft refusal/disclaimer patterns to drop. Keep this list short and obvious.
REFUSAL_PATTERNS = [
    r"\bI (?:cannot|can't|won't|will not|am unable to)\b",
    r"\bI'?m sorry,? but\b",
    r"\bI must decline\b",
    r"\bAs an AI\b",
    r"\bI cannot help (?:with )?(?:that|generating)\b",
    r"\bI'?m not able to\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

WS_RE = re.compile(r"\s+")

def n_tokens(s: str) -> int:
    return len(str(s).split())

def normalize_for_dup(s: str) -> str:
    return WS_RE.sub(" ", str(s).lower()).strip()[:500]

def load_raw(model_key: str):
    path = config.LLM_DIR / f"{model_key}_raw.csv"
    if not path.exists():
        print(f"[WARN] {path} not found; skipping.")
        return None
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {model_key}: {len(df)} rows")
    return df

def main():
    frames = []
    for model_key in config.MODELS.keys():
        df = load_raw(model_key)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        print("[ERROR] No raw LLM files found. Run the generation scripts first.")
        return

    raw = pd.concat(frames, ignore_index=True)
    print(f"\n[INFO] Combined raw: {len(raw)} rows")

    # Build text field
    raw["subject"] = raw["subject"].fillna("").astype(str).str.strip()
    raw["body"]    = raw["body"].fillna("").astype(str).str.strip()
    raw["text"]    = (raw["subject"] + " " + raw["body"]).str.strip()

    n0 = len(raw)
    # Drop empty body
    raw = raw[raw["body"].str.len() > 0]
    print(f"[FILT] Empty body          : -{n0 - len(raw)}  (kept {len(raw)})")

    n1 = len(raw)
    # Drop empty subject (format failures)
    raw = raw[raw["subject"].str.len() > 0]
    print(f"[FILT] Empty subject       : -{n1 - len(raw)}  (kept {len(raw)})")

    n2 = len(raw)
    # Drop refusals
    raw = raw[~raw["text"].astype(str).str.contains(REFUSAL_RE, na=False)]
    print(f"[FILT] Refusals/disclaimers: -{n2 - len(raw)}  (kept {len(raw)})")

    n3 = len(raw)
    # Length filter
    raw["n_tokens"] = raw["text"].map(n_tokens)
    raw = raw[(raw["n_tokens"] >= config.MIN_TOKENS) &
              (raw["n_tokens"] <= config.MAX_TOKENS)]
    print(f"[FILT] Length [{config.MIN_TOKENS}-{config.MAX_TOKENS}]    : "
          f"-{n3 - len(raw)}  (kept {len(raw)})")

    n4 = len(raw)
    # Drop near-duplicates (per model, category) using normalized prefix
    raw["__dup_key"] = raw["model"] + "|" + raw["category"] + "|" + raw["text"].map(normalize_for_dup)
    raw = raw.drop_duplicates(subset="__dup_key").drop(columns="__dup_key")
    print(f"[FILT] Near-duplicates     : -{n4 - len(raw)}  (kept {len(raw)})")

    # Final assembly
    raw["label"]  = 1                 # phishing positive class
    raw["source"] = raw["model"]      # source name = model name
    out = raw[["text", "subject", "body", "label", "source", "model", "category"]]\
            .reset_index(drop=True)

    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n[OK] Saved -> {OUTPUT_FILE}")
    print(f"[OK] Final LLM corpus: {len(out):,} rows")

    print("\nRows per model:")
    print(out["model"].value_counts())
    print("\nRows per (model, category):")
    print(out.groupby(["model", "category"]).size().unstack(fill_value=0))

    n = out["text"].map(n_tokens)
    print(f"\nToken length stats:")
    print(f"  mean={n.mean():.1f}  median={n.median():.1f}  std={n.std():.1f}  "
          f"min={n.min()}  max={n.max()}")

if __name__ == "__main__":
    main()
