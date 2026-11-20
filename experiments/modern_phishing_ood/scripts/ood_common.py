"""Shared helpers for the modern-phishing OOD experiment.

Import-light on purpose: only the standard library + numpy + pandas.
Nothing in this module imports ``torch``, ``sklearn`` or any
``phishing_xai`` module at import time, so the unit tests run on CPU-only
machines without a GPU stack. Where a repo helper exists, this module
either

* mirrors its rule exactly and says so in the docstring (the Kaggle
  kernel cross-checks the mirror against the repo helper at runtime and
  logs any difference), or
* is used by ``run_ood_eval.py`` through a lazy import of the repo
  helper with this module as a documented fallback.

Rules mirrored here (with sources):

* ``near_dup_hash`` mirrors ``phishing_xai.data.compute_near_dup_hash``
  (``src/phishing_xai/data.py:61``): MD5 of the first 200 alphanumeric
  characters, lowercased.
* ``fit_thresholds_for_floors`` mirrors
  ``phishing_xai.baselines.precision_at_recall`` /
  ``fit_threshold_for_recall`` (``src/phishing_xai/baselines.py:256`` /
  ``:283``): highest precision over thresholds whose recall is at least
  the floor; if no threshold reaches the floor, the max-recall
  threshold is returned and ``floor_met`` is False.
* ``normalize_for_hash`` is the exact-hash method used by the dataset
  provenance hunt (``reports/dataset_provenance.md`` section 0):
  lowercase, strip non-alphanumerics, collapse whitespace, MD5.

Caveat that travels with every result of this experiment: the primary
OOD set (PhishNChips v5.2) is a modern URL-grounded synthetic-stimuli
set, NOT real-world inbox mail. See ``reports/dataset_provenance.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Text normalisation / hashing (leakage checks)
# ---------------------------------------------------------------------------

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+|www\.[^\s\"'<>)\]]+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize_for_hash(text: str) -> str:
    """Provenance-method normalisation: lowercase, strip non-alphanumerics,
    collapse whitespace. Used for exact-overlap MD5 checks."""
    text = str(text or "").lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def near_dup_hash(text: str, n_chars: int = 200) -> str:
    """Mirror of ``phishing_xai.data.compute_near_dup_hash`` (data.py:61):
    MD5 of the first ``n_chars`` alphanumeric characters (lowercased).
    The Kaggle kernel cross-checks this against the repo helper on a
    sample and logs any mismatch."""
    alnum = re.sub(r"[^a-z0-9]", "", str(text or "").lower())
    return hashlib.md5(alnum[:n_chars].encode("utf-8")).hexdigest()


def clean_text_lite(text: str) -> str:
    """Lightweight stand-in for ``phishing_xai.data.clean_text`` used ONLY
    for leakage hashing / local inspection. It lowercases, strips HTML
    tags with a regex (not BeautifulSoup) and collapses whitespace.
    Model inputs on Kaggle always go through the real
    ``phishing_xai.data.clean_text`` (lazy import in run_ood_eval.py);
    this fallback is never used for scoring when the repo is importable."""
    text = _HTML_TAG_RE.sub(" ", str(text or ""))
    text = text.lower().replace("\r", " ")
    return _WS_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# URL extraction / overlap
# ---------------------------------------------------------------------------

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text, normalised for comparison: lowercased,
    trailing punctuation stripped. Returns a de-duplicated list."""
    found = []
    for m in _URL_RE.finditer(str(text or "")):
        url = m.group(0).lower().rstrip(".,;:!?\"'").rstrip("/")
        if url and url not in found:
            found.append(url)
    return found


def url_overlap(ood_urls: Iterable[str], train_urls: Iterable[str]) -> Dict[str, object]:
    """Overlap between a set of OOD URLs and a set of training URLs."""
    ood = set(ood_urls)
    train = set(train_urls)
    shared = ood & train
    return {
        "n_ood_urls": len(ood),
        "n_train_urls": len(train),
        "n_shared_urls": len(shared),
        "shared_urls": sorted(shared)[:50],
        "shared_url_fraction_of_ood": (len(shared) / len(ood)) if ood else 0.0,
    }


# ---------------------------------------------------------------------------
# MinHash near-duplicate check
# ---------------------------------------------------------------------------

def word_shingles(text: str, k: int = 5) -> set:
    """Word-level k-shingles of the hash-normalised text."""
    words = normalize_for_hash(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _shingle_hashes(shingles: Iterable[str]) -> np.ndarray:
    """Stable 32-bit integer hash per shingle (first 4 bytes of MD5)."""
    arr = np.empty(len(shingles), dtype=np.uint64)
    for i, s in enumerate(shingles):
        arr[i] = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:4], "little")
    return arr


def minhash_signature(text: str, n_perm: int = 64, seed: int = 42, k: int = 5) -> np.ndarray:
    """MinHash signature of one text over word 5-shingles.

    Permutations are ``(a * x + b) mod 2**32`` with ``a`` (odd) / ``b``
    drawn from a fixed ``RandomState(seed)``, so signatures are
    comparable across runs. Arithmetic is done in uint64 (``a`` and ``x``
    are < 2**32, so the product fits) and reduced mod 2**32.
    Empty texts get an all-``2**32-1`` signature (Jaccard 1.0 only
    against another empty text, by construction of the comparison).
    """
    rng = np.random.RandomState(seed)
    a = rng.randint(1, 2**31, size=n_perm, dtype=np.uint64) * 2 + 1  # odd
    b = rng.randint(0, 2**32, size=n_perm, dtype=np.uint64)
    shingles = word_shingles(text, k=k)
    if not shingles:
        return np.full(n_perm, np.iinfo(np.uint32).max, dtype=np.uint64)
    h = _shingle_hashes(sorted(shingles))  # (n_shingles,) uint64 < 2**32
    vals = (a[:, None] * h[None, :] + b[:, None]) % np.uint64(2**32)
    return vals.min(axis=1)


def minhash_signatures(texts: Sequence[str], n_perm: int = 64, seed: int = 42,
                       k: int = 5) -> np.ndarray:
    """Stacked MinHash signatures, shape ``(n_texts, n_perm)``."""
    return np.vstack([minhash_signature(t, n_perm=n_perm, seed=seed, k=k)
                      for t in texts])


def minhash_max_jaccard(ood_sigs: np.ndarray, train_sigs: np.ndarray,
                        chunk: int = 200) -> np.ndarray:
    """For each OOD signature, the maximum estimated Jaccard similarity
    against any training signature (fraction of equal permutation minima).
    Chunked so memory stays bounded; returns shape ``(n_ood,)``."""
    n_ood = ood_sigs.shape[0]
    best = np.zeros(n_ood, dtype=float)
    for start in range(0, n_ood, chunk):
        block = ood_sigs[start:start + chunk]  # (b, p)
        # (b, n_train, p) is too big; loop over train in blocks instead.
        for t_start in range(0, train_sigs.shape[0], 1000):
            t_block = train_sigs[t_start:t_start + 1000]  # (t, p)
            eq = (block[:, None, :] == t_block[None, :, :]).mean(axis=2)
            best[start:start + chunk] = np.maximum(
                best[start:start + chunk], eq.max(axis=1))
    return best


def near_dup_audit(ood_texts: Sequence[str], train_texts: Sequence[str],
                   threshold: float = 0.9, n_perm: int = 64, seed: int = 42,
                   k: int = 5) -> Dict[str, object]:
    """MinHash near-duplicate audit of OOD texts against training texts.

    Returns the method record plus counts: how many OOD texts have an
    estimated Jaccard >= ``threshold`` against ANY training text, and the
    distribution of the per-OOD maximum Jaccard. Full pairwise comparison
    (no sampling) unless the caller subsamples beforehand; the caller
    records the sample sizes in the returned dict."""
    ood_sigs = minhash_signatures(ood_texts, n_perm=n_perm, seed=seed, k=k)
    train_sigs = minhash_signatures(train_texts, n_perm=n_perm, seed=seed, k=k)
    return near_dup_audit_with_sigs(ood_sigs, train_sigs, threshold=threshold,
                                    n_perm=n_perm, seed=seed, k=k,
                                    n_ood=len(ood_texts), n_train=len(train_texts))


def near_dup_audit_with_sigs(ood_sigs: np.ndarray, train_sigs: np.ndarray,
                             threshold: float = 0.9, n_perm: int = 64,
                             seed: int = 42, k: int = 5,
                             n_ood: int = 0, n_train: int = 0) -> Dict[str, object]:
    """Same as :func:`near_dup_audit` but with precomputed signatures,
    so the training signatures are computed once and reused across sets."""
    best = minhash_max_jaccard(ood_sigs, train_sigs)
    return {
        "method": "minhash",
        "n_perm": n_perm,
        "shingle_k_words": k,
        "seed": seed,
        "threshold": threshold,
        "n_ood": int(n_ood or len(ood_sigs)),
        "n_train": int(n_train or len(train_sigs)),
        "n_ood_with_max_jaccard_ge_threshold": int((best >= threshold).sum()),
        "n_ood_with_max_jaccard_ge_0.8": int((best >= 0.8).sum()),
        "n_ood_with_max_jaccard_ge_0.5": int((best >= 0.5).sum()),
        "max_jaccard_median": float(np.median(best)) if len(best) else 0.0,
        "max_jaccard_mean": float(np.mean(best)) if len(best) else 0.0,
        "max_jaccard_max": float(np.max(best)) if len(best) else 0.0,
    }


# ---------------------------------------------------------------------------
# Exact / near-dup-hash overlap (cheap, exact set intersections)
# ---------------------------------------------------------------------------

def hash_overlap(ood_texts: Sequence[str], train_texts: Sequence[str]) -> Dict[str, object]:
    """Exact normalized-MD5 overlap + first-200-char near-dup-hash overlap
    of OOD texts against training texts. Both are exact set intersections
    (no sampling); counts are exact."""
    train_exact = {md5_hex(normalize_for_hash(t)) for t in train_texts}
    train_near = {near_dup_hash(t) for t in train_texts}
    ood_exact = [md5_hex(normalize_for_hash(t)) for t in ood_texts]
    ood_near = [near_dup_hash(t) for t in ood_texts]
    return {
        "method": "exact_md5_of_normalized_text + near_dup_hash(first_200_alnum)",
        "n_ood": int(len(ood_texts)),
        "n_train": int(len(train_texts)),
        "n_train_unique_exact_hashes": len(train_exact),
        "n_train_unique_near_dup_hashes": len(train_near),
        "n_ood_exact_overlap": int(sum(h in train_exact for h in ood_exact)),
        "n_ood_near_dup_hash_overlap": int(sum(h in train_near for h in ood_near)),
    }


# ---------------------------------------------------------------------------
# Threshold fitting — validation ONLY (pure numpy)
# ---------------------------------------------------------------------------

def fit_thresholds_for_floors(val_labels: Sequence[int], val_probs: Sequence[float],
                              floors: Sequence[float] = (0.85, 0.90, 0.95)
                              ) -> Dict[str, Dict[str, object]]:
    """Fit one decision threshold per recall floor on VALIDATION data only.

    Mirrors ``phishing_xai.baselines.precision_at_recall`` /
    ``fit_threshold_for_recall``: among candidate thresholds, pick the one
    with the highest precision whose recall is at least the floor
    (ties: higher recall, then higher threshold). If no threshold reaches
    the floor, return the max-recall threshold with ``floor_met`` False.

    This function accepts validation arrays ONLY — it has no OOD
    parameter at all, so OOD labels cannot influence the fit. The eval
    script calls it with validation predictions and applies the returned
    thresholds unchanged to every OOD set.
    """
    y = np.asarray(val_labels, dtype=int)
    p = np.asarray(val_probs, dtype=float)
    if len(y) == 0:
        raise ValueError("fit_thresholds_for_floors: empty validation arrays")
    if len(y) != len(p):
        raise ValueError("fit_thresholds_for_floors: label/prob length mismatch")
    n_pos = int((y == 1).sum())
    out: Dict[str, Dict[str, object]] = {}
    # Candidate thresholds: each distinct predicted probability, plus 0.5.
    candidates = sorted(set(float(v) for v in p) | {0.5}, reverse=True)
    for floor in floors:
        floor = float(floor)
        best = None  # (precision, recall, threshold)
        for thr in candidates:
            pred = (p >= thr).astype(int)
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            recall = tp / n_pos if n_pos else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            if recall >= floor:
                key = (precision, recall, thr)
                if best is None or key > best[0]:
                    best = (key, precision, recall, thr)
        if best is not None:
            _, precision, recall, thr = best
            out[f"{floor:.2f}"] = {
                "threshold": float(thr), "target_recall": floor,
                "val_precision": float(precision), "val_recall": float(recall),
                "floor_met": True, "fit_on": "original_validation_split_only",
            }
        else:
            # Floor unreachable: max-recall threshold (highest recall,
            # then highest precision), exactly like the repo fallback.
            best_rec, best_prec, best_thr = -1.0, -1.0, 0.5
            for thr in candidates:
                pred = (p >= thr).astype(int)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                recall = tp / n_pos if n_pos else 0.0
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                if (recall, precision) > (best_rec, best_prec):
                    best_rec, best_prec, best_thr = recall, precision, thr
            out[f"{floor:.2f}"] = {
                "threshold": float(best_thr), "target_recall": floor,
                "val_precision": float(best_prec), "val_recall": float(best_rec),
                "floor_met": False, "fit_on": "original_validation_split_only",
            }
    return out


# ---------------------------------------------------------------------------
# Pure-numpy metrics (fallback / local testing only)
# ---------------------------------------------------------------------------

def metrics_lite(y_true: Sequence[int], y_prob: Sequence[float],
                 threshold: float = 0.5) -> Dict[str, object]:
    """Full metric set in pure numpy (accuracy, precision, recall, F1,
    ROC-AUC via Mann-Whitney, PR-AUC as average precision, confusion
    matrix, FPR, FNR). Used ONLY as a fallback when
    ``phishing_xai.baselines.compute_metrics_full`` is not importable
    (e.g. CPU-only local tests) and as a cross-check target on Kaggle.
    On Kaggle the repo helper is used for the reported numbers."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    n = len(y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # ROC-AUC (Mann-Whitney U, ties 0.5).
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        roc_auc = 0.5
    else:
        order = np.argsort(p)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1)
        # Average ranks for ties.
        _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
        tie_mean = np.zeros(n)
        for v in np.unique(p):
            mask = p == v
            tie_mean[mask] = ranks[mask].mean()
        ranks = tie_mean
        roc_auc = float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                        / (len(pos) * len(neg)))
    # Average precision (step function over recall-sorted thresholds).
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tp_cum = np.cumsum(ys == 1)
    fp_cum = np.cumsum(ys == 0)
    prec_at_k = tp_cum / (tp_cum + fp_cum)
    n_pos = int((y == 1).sum())
    pr_auc = float((prec_at_k[ys == 1].sum() / n_pos)) if n_pos else 0.0
    return {
        "accuracy": float((tp + tn) / n) if n else 0.0,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# OOD set normalisation -> (email_id, text, label, source, category)
# ---------------------------------------------------------------------------

def _combine(subject: object, body: object) -> str:
    s = str(subject or "").strip()
    b = str(body or "").strip()
    if s and b:
        return f"{s}\n{b}"
    return s or b


def load_phishnchips(path: str | Path) -> pd.DataFrame:
    """PhishNChips v5.2 core CSV -> normalised frame.

    ``email_content`` is JSON with sender/from/subject/body; text is
    ``subject + body``; label is ``phish_label`` (1 = phishing).
    Keeps datasource/strategy/url_category for stratified reporting."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        try:
            content = json.loads(r["email_content"])
        except (ValueError, TypeError):
            content = {}
        rows.append({
            "email_id": str(r["id"]),
            "text": _combine(content.get("subject", ""), content.get("body", "")),
            "body": str(content.get("body", "")),
            "label": int(r["phish_label"]),
            "source": "phishnchips_v5.2",
            "category": str(r.get("url_category", "")),
            "datasource": str(r.get("datasource", "")),
            "strategy": str(r.get("strategy", "")),
            "url_raw": str(r.get("url_raw", "")),
        })
    return pd.DataFrame(rows)


def load_phishfuzzer_manual(path: str | Path) -> pd.DataFrame:
    """PhishFuzzer seeds JSON, filtered to ``Source == 'Manual'`` (n=300).

    Binary set: Phishing (1) vs Valid (0), n=205. Spam rows (n=95) are
    kept in the frame with label -1 and category 'spam' so their
    predictions can be reported separately (spam flagged as phishing
    rate) — they are NEVER part of binary metrics."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        if r.get("Source") != "Manual":
            continue
        typ = str(r.get("Type", ""))
        label = {"Phishing": 1, "Valid": 0}.get(typ, -1)
        rows.append({
            "email_id": f"phishfuzzer_manual_{r.get('No.', '')}",
            "text": _combine(r.get("Subject", ""), r.get("Body", "")),
            "body": str(r.get("Body", "")),
            "label": label,
            "source": "phishfuzzer_manual_300",
            "category": typ.lower(),
            "year": str(r.get("Year", "")),
            "language": str(r.get("Language", "")),
        })
    return pd.DataFrame(rows)


def load_frontiers_llm(path: str | Path) -> pd.DataFrame:
    """Frontiers 2026 LLM-generated phishing (4,986; label 1 for all).
    Category is ``<model>/<category>``. Phishing-only: must be paired
    with a legitimate set (PhishNChips legit 1,000) at eval time."""
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        text = str(r["text"]) if "text" in df.columns and pd.notna(r.get("text")) \
            else _combine(r.get("subject", ""), r.get("body", ""))
        rows.append({
            "email_id": f"frontiers_llm_{i}",
            "text": text,
            "body": str(r.get("body", "")),
            "label": 1,
            "source": "frontiers_2026_llm",
            "category": f"{r.get('model', '')}/{r.get('category', '')}",
            "model": str(r.get("model", "")),
        })
    return pd.DataFrame(rows)


def load_frontiers_human(path: str | Path) -> pd.DataFrame:
    """Frontiers 2026 human phishing (5,000; label 1 for all) from OLD
    corpora (TREC-07, CEAS-08, Nazario, Nigerian Fraud, Enron). Used ONLY
    as the human-authored comparison slice for Table B (human-vs-AI);
    it is NOT modern and must never be presented as such."""
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        text = str(r["text"]) if "text" in df.columns and pd.notna(r.get("text")) \
            else _combine(r.get("subject", ""), r.get("body", ""))
        rows.append({
            "email_id": f"frontiers_human_{i}",
            "text": text,
            "body": str(r.get("body", "")),
            "label": 1,
            "source": "frontiers_2026_human_old_corpora",
            "category": str(r.get("source", "")),
        })
    return pd.DataFrame(rows)


def load_bec(path: str | Path, poisoned: bool = False) -> pd.DataFrame:
    """Adversarial BEC synthetic emails (Gemini Flash 2.5). Drops the 30
    rows with NaN label. Phishing-only: must be paired with a legitimate
    set at eval time."""
    df = pd.read_csv(path)
    df = df[df["label"].notna()].copy()
    rows = []
    for i, r in df.iterrows():
        rows.append({
            "email_id": f"bec_{'poisoned' if poisoned else 'clean'}_{i}",
            "text": _combine(r.get("subject", ""), r.get("body", "")),
            "body": str(r.get("body", "")),
            "label": int(r["label"]),
            "source": "yoadjei_bec_poisoned" if poisoned else "yoadjei_bec_clean",
            "category": "bec_gemini",
        })
    return pd.DataFrame(rows)


def pair_with_legit(phish_frame: pd.DataFrame, legit_frame: pd.DataFrame,
                    legit_label: int = 0) -> pd.DataFrame:
    """Pair a phishing-only frame with a legitimate frame (e.g. the
    PhishNChips legit 1,000). Returns the concatenated frame; the
    pairing is recorded in the eval output metadata, never hidden."""
    legit = legit_frame[legit_frame["label"] == legit_label].copy()
    return pd.concat([phish_frame, legit], ignore_index=True)
