import re
import sys
import pandas as pd
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import config
HUMAN_FILE   = config.HUMAN_DIR / "human_corpus_sampled.csv"
LLM_FILE     = config.LLM_DIR   / "llm_corpus_sampled.csv"
FEATURES_DIR = config.DATA_DIR  / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE     = FEATURES_DIR / "corpus_features.csv"

try:
    import spacy
except ImportError:
    print("[ERROR] spacy not installed. Run: pip install spacy")
    sys.exit(1)

SPACY_MODEL = "en_core_web_sm"
print(f"[INFO] Loading spaCy model: {SPACY_MODEL}")
try:
    nlp = spacy.load(SPACY_MODEL, disable=["ner"])
except OSError:
    print(f"[ERROR] spaCy model '{SPACY_MODEL}' not installed.")
    print(f"        Install with: python -m spacy download {SPACY_MODEL}")
    sys.exit(1)

POLITENESS_TERMS = {
    "please", "kindly", "thank", "thanks", "appreciate", "appreciated",
    "would", "could", "may", "regards", "sincerely", "respectfully",
}
URGENCY_TERMS = {
    "urgent", "urgently", "immediately", "now", "asap", "today",
    "right away", "emergency", "critical", "important", "deadline",
    "expire", "expires", "expiring", "expired", "final", "last", "limited",
    "hurry", "quickly", "soon", "promptly",
}
CTA_TERMS = {
    "click", "tap", "open", "download", "install", "verify", "confirm",
    "update", "sign in", "login", "log in", "log-in", "sign-in", "register",
    "enroll", "submit", "complete", "review", "approve", "respond", "reply",
    "call", "contact", "follow", "visit", "go to", "proceed",
}
AUTHORITY_TERMS = {
    "irs", "police", "fbi", "government", "tax authority", "court", "legal",
    "compliance", "audit", "regulator", "regulatory", "official",
    "administrator", "admin", "manager", "executive", "ceo", "director",
    "headquarters", "corporate", "department", "agency", "bureau",
}
TIME_PRESSURE_TERMS = {
    "24 hours", "48 hours", "72 hours", "today", "tomorrow", "tonight",
    "within", "before", "by end of", "deadline", "expires", "expire",
    "expiring", "expired", "closing", "closes", "soon", "shortly",
    "immediately", "asap",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")

FEATURE_NAMES = [
    "ttr", "mean_word_len", "mean_sentence_len_tokens", "yules_k",
    "clause_density", "noun_ratio", "verb_ratio", "mean_parse_depth",
    "imperative_count", "first_person_ratio", "second_person_ratio",
    "politeness_density", "urgency_density",
    "url_density", "cta_density", "authority_density", "time_pressure_density",
]

OUTPUT_COLS = ["id", "text", "label_origin", "source", "category"] + FEATURE_NAMES

def safe_div(a, b):
    return a / b if b else 0.0

def yules_k(token_freqs):
    N = sum(token_freqs.values())
    if N <= 0:
        return 0.0
    M2 = sum(f * f for f in token_freqs.values())
    return 10000 * (M2 - N) / (N * N) if N > 0 else 0.0

def mean_parse_depth(doc):
    """Mean depth of dependency tree across all tokens.
    The ROOT token has depth 0 (its head index equals its own index).
    We walk up the tree counting steps until we reach the ROOT."""
    depths = []
    HARD_CAP = 200
    for sent in doc.sents:
        for tok in sent:
            d = 0
            cur = tok
            # Compare by token index, not by object identity (spaCy
            # may return new Token wrappers each time .head is accessed)
            while cur.head.i != cur.i and d < HARD_CAP:
                d += 1
                cur = cur.head
            depths.append(d)
    if not depths:
        return 0.0
    return sum(depths) / len(depths)

def count_imperatives(doc):
    n = 0
    for sent in doc.sents:
        first = next((t for t in sent if not t.is_space), None)
        if first is None:
            continue
        if first.tag_ == "VB" and first.dep_ in {"ROOT", "ccomp"}:
            n += 1
    return n

def count_terms_in_text(text_lower, term_set):
    n = 0
    for term in term_set:
        if " " in term:
            n += text_lower.count(term)
        else:
            n += len(re.findall(rf"\b{re.escape(term)}\b", text_lower))
    return n

def extract_features(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {k: 0.0 for k in FEATURE_NAMES}

    raw = text
    text_lower = text.lower()
    doc = nlp(text)
    tokens      = [t for t in doc if not t.is_space]
    word_tokens = [t for t in tokens if t.is_alpha]
    n_words  = len(word_tokens)
    n_tokens = len(tokens)
    sents    = list(doc.sents)
    n_sents  = max(1, len(sents))

    # ---- LEXICAL ----
    lower_words = [t.text.lower() for t in word_tokens]
    word_counts = Counter(lower_words)
    ttr            = safe_div(len(word_counts), n_words)
    mean_word_len  = safe_div(sum(len(w) for w in lower_words), n_words)
    mean_sent_toks = safe_div(n_tokens, n_sents)
    yules_k_val    = yules_k(word_counts)

    # ---- SYNTACTIC ----
    n_clauses = sum(1 for t in doc
                    if t.dep_ in {"ROOT", "ccomp", "advcl", "relcl", "xcomp"})
    clause_density = safe_div(n_clauses, n_sents)
    noun_ratio  = safe_div(sum(1 for t in tokens if t.pos_ == "NOUN"), n_tokens)
    verb_ratio  = safe_div(sum(1 for t in tokens if t.pos_ == "VERB"), n_tokens)
    parse_depth = mean_parse_depth(doc)

    # ---- STYLISTIC ----
    imperative_n = count_imperatives(doc)
    first_person_pronouns  = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    second_person_pronouns = {"you", "your", "yours", "yourself", "yourselves"}
    n_first  = sum(1 for w in lower_words if w in first_person_pronouns)
    n_second = sum(1 for w in lower_words if w in second_person_pronouns)
    first_ratio  = safe_div(n_first,  n_words)
    second_ratio = safe_div(n_second, n_words)
    politeness_n = count_terms_in_text(text_lower, POLITENESS_TERMS)
    urgency_n    = count_terms_in_text(text_lower, URGENCY_TERMS)
    politeness_density = safe_div(politeness_n * 100, n_words)
    urgency_density    = safe_div(urgency_n    * 100, n_words)

    # ---- PHISHING-SPECIFIC ----
    n_urls      = len(URL_RE.findall(raw))
    n_ctas      = count_terms_in_text(text_lower, CTA_TERMS)
    n_authority = count_terms_in_text(text_lower, AUTHORITY_TERMS)
    n_time      = count_terms_in_text(text_lower, TIME_PRESSURE_TERMS)
    url_density       = safe_div(n_urls      * 100, n_words)
    cta_density       = safe_div(n_ctas      * 100, n_words)
    authority_density = safe_div(n_authority * 100, n_words)
    time_density      = safe_div(n_time      * 100, n_words)

    return {
        "ttr":                      round(ttr, 4),
        "mean_word_len":            round(mean_word_len, 3),
        "mean_sentence_len_tokens": round(mean_sent_toks, 3),
        "yules_k":                  round(yules_k_val, 3),
        "clause_density":           round(clause_density, 3),
        "noun_ratio":               round(noun_ratio, 4),
        "verb_ratio":               round(verb_ratio, 4),
        "mean_parse_depth":         round(parse_depth, 3),
        "imperative_count":         imperative_n,
        "first_person_ratio":       round(first_ratio, 4),
        "second_person_ratio":      round(second_ratio, 4),
        "politeness_density":       round(politeness_density, 3),
        "urgency_density":          round(urgency_density, 3),
        "url_density":              round(url_density, 3),
        "cta_density":              round(cta_density, 3),
        "authority_density":        round(authority_density, 3),
        "time_pressure_density":    round(time_density, 3),
    }

def load_corpus_robust(path: Path, default_origin: str, prefix: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[INFO] {path.name}: {len(df):,} rows; columns = {list(df.columns)}")

    for required in ("text", "source"):
        if required not in df.columns:
            print(f"[ERROR] {path.name} is missing required column '{required}'.")
            print(f"        Available columns: {list(df.columns)}")
            sys.exit(1)

    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")].copy()
    df = df.reset_index(drop=True)

    if "id" not in df.columns:
        df["id"] = [f"{prefix}_{i:06d}" for i in range(len(df))]
        print(f"[INFO] {path.name}: auto-generated 'id' column.")
    else:
        df["id"] = df["id"].astype(str)
        if df["id"].duplicated().any():
            n_dup = int(df["id"].duplicated().sum())
            print(f"[WARN] {path.name}: {n_dup} duplicate ids; regenerating.")
            df["id"] = [f"{prefix}_{i:06d}" for i in range(len(df))]

    if "category" not in df.columns:
        df["category"] = ""

    df["label_origin"] = default_origin

    return df[["id", "text", "label_origin", "source", "category"]]

def main():
    print("[INFO] Loading human corpus...")
    h = load_corpus_robust(HUMAN_FILE, default_origin="human", prefix="human")
    print("[INFO] Loading LLM corpus...")
    l = load_corpus_robust(LLM_FILE,   default_origin="llm",   prefix="llm")

    h["id"] = "h_" + h["id"].astype(str)
    l["id"] = "l_" + l["id"].astype(str)

    corpus = pd.concat([h, l], ignore_index=True)
    print(f"\n[INFO] Combined corpus size: {len(corpus):,}")
    print(f"  - human: {(corpus['label_origin']=='human').sum():,}")
    print(f"  - llm  : {(corpus['label_origin']=='llm').sum():,}")
    print(f"\n[INFO] Source distribution:")
    print(corpus["source"].value_counts().to_string())

    # NOTE: because of the parse-depth bug fix, do NOT resume from a previous
    # corpus_features.csv. Force a fresh run by deleting the file first.
    if OUT_FILE.exists():
        print(f"\n[WARN] {OUT_FILE.name} already exists. The parse-depth")
        print( "       computation has been corrected; previous values are")
        print( "       no longer valid. Deleting and recomputing from scratch.")
        OUT_FILE.unlink()

    rows_buffer = []
    flush_every = 200
    pbar = tqdm(corpus.itertuples(index=False), total=len(corpus),
                desc="Extracting features")
    for row in pbar:
        feats = extract_features(row.text)
        rows_buffer.append({
            "id":           row.id,
            "text":         row.text,
            "label_origin": row.label_origin,
            "source":       row.source,
            "category":     row.category,
            **feats,
        })
        if len(rows_buffer) >= flush_every:
            mode   = "a" if OUT_FILE.exists() else "w"
            header = not OUT_FILE.exists()
            pd.DataFrame(rows_buffer)[OUTPUT_COLS].to_csv(
                OUT_FILE, mode=mode, header=header,
                index=False, encoding="utf-8")
            rows_buffer = []

    if rows_buffer:
        mode   = "a" if OUT_FILE.exists() else "w"
        header = not OUT_FILE.exists()
        pd.DataFrame(rows_buffer)[OUTPUT_COLS].to_csv(
            OUT_FILE, mode=mode, header=header,
            index=False, encoding="utf-8")

    final = pd.read_csv(OUT_FILE)
    print(f"\n[OK] Saved -> {OUT_FILE}")
    print(f"[OK] Total rows: {len(final):,}")
    print(f"\nLabel origin distribution:")
    print(final["label_origin"].value_counts().to_string())
    print(f"\nSource distribution:")
    print(final["source"].value_counts().to_string())
    print(f"\nFeature summary (mean by label_origin):")
    print(final.groupby("label_origin")[FEATURE_NAMES].mean().round(3).T)

if __name__ == "__main__":
    main()