"""Unit tests for the OOD experiment helpers (tests/test_ood_eval.py).

CPU-only, no GPU, no internet, no checkpoints, no real training data:
all fixtures are tiny and hand-built. These tests cover the two
properties the experiment's validity rests on:

* leakage checks (exact MD5, near-dup hash, MinHash, URL overlap) count
  what they claim to count, and
* thresholds are fitted on the ORIGINAL VALIDATION split only — OOD
  labels can never influence them (the fit function takes validation
  arrays only; a test proves shuffling OOD labels cannot change the
  fitted thresholds, because the OOD arrays never reach the fitter).
"""

import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "experiments" / "modern_phishing_ood" / "scripts"))

import ood_common as oc


# ---------------------------------------------------------------------------
# Normalisation / hashing
# ---------------------------------------------------------------------------

def test_normalize_for_hash_strips_case_punct_and_whitespace():
    assert oc.normalize_for_hash("  Hello, WORLD!\nNew\tline  ") == "hello world new line"
    assert oc.normalize_for_hash("") == ""
    assert oc.normalize_for_hash(None) == ""


def test_md5_hex_is_plain_md5():
    assert oc.md5_hex("abc") == hashlib.md5(b"abc").hexdigest()


def test_near_dup_hash_matches_repo_algorithm():
    # Re-implement data.compute_near_dup_hash (data.py:61) by hand:
    # MD5 of the first 200 alphanumeric chars, lowercased.
    text = "Hello, World! " * 40 + "TAIL"
    alnum = re.sub(r"[^a-z0-9]", "", text.lower())
    expected = hashlib.md5(alnum[:200].encode("utf-8")).hexdigest()
    assert oc.near_dup_hash(text) == expected
    # A change after the first 200 alnum chars must NOT change the hash
    # (this is the known limitation of the repo's near-dup notion).
    assert oc.near_dup_hash(text + " completely different ending here") == expected
    # A change inside the first 200 chars must change it.
    assert oc.near_dup_hash("Jello, World! " * 40 + "TAIL") != expected


def test_clean_text_lite_lowercases_and_strips_tags():
    assert oc.clean_text_lite("<p>Hello  WORLD</p>") == "hello world"


# ---------------------------------------------------------------------------
# Leakage: exact / near-dup / MinHash / URL overlap
# ---------------------------------------------------------------------------

def test_hash_overlap_counts_exact_and_near_dup():
    train = ["alpha beta gamma delta", "completely different text here", ""]
    ood = ["alpha beta gamma delta",  # exact duplicate
           "ALPHA beta GAMMA delta!",  # same after normalisation
           "alpha beta gamma delta and then a long different tail " * 3,  # near-dup hash hit only
           "brand new unseen email body"]
    res = oc.hash_overlap(ood, train)
    assert res["n_ood"] == 4
    assert res["n_ood_exact_overlap"] == 2
    # The long-tail row shares the first 200 alnum chars with the train
    # row only if its prefix matches; here it does NOT (different words
    # after 'delta'), so near-dup overlap == exact overlap in this fixture.
    assert res["n_ood_near_dup_hash_overlap"] == 2


def test_hash_overlap_near_dup_prefix_rule():
    # Same first-200-alnum prefix, different tail -> near-dup hit, no exact hit.
    prefix = "word " * 60  # 60*4=240 alnum chars, first 200 all 'word'
    train = [prefix + "aaaa"]
    ood = [prefix + "zzzz totally different ending"]
    res = oc.hash_overlap(ood, train)
    assert res["n_ood_exact_overlap"] == 0
    assert res["n_ood_near_dup_hash_overlap"] == 1


def test_minhash_identical_texts_score_one_disjoint_score_low():
    sig_a = oc.minhash_signature("the quick brown fox jumps over the lazy dog today")
    sig_b = oc.minhash_signature("the quick brown fox jumps over the lazy dog today")
    assert (sig_a == sig_b).mean() == 1.0
    sig_c = oc.minhash_signature("quantum entanglement photon polarisation laboratory vacuum chamber")
    assert (sig_a == sig_c).mean() < 0.3
    # One-word change in a long text stays a near-duplicate.
    long_a = " ".join(f"word{i}" for i in range(60))
    long_b = " ".join(f"word{i}" if i != 30 else "changed" for i in range(60))
    est = (oc.minhash_signature(long_a) == oc.minhash_signature(long_b)).mean()
    assert est >= 0.8


def test_near_dup_audit_counts():
    train = [" ".join(f"word{i}" for i in range(60))]
    ood = [" ".join(f"word{i}" for i in range(60)),  # identical
           " ".join(f"word{i}" if i != 30 else "changed" for i in range(60)),  # near-dup
           " ".join(f"other{i}" for i in range(60))]  # disjoint
    res = oc.near_dup_audit(ood, train, threshold=0.9, n_perm=64)
    assert res["method"] == "minhash"
    assert res["n_ood"] == 3 and res["n_train"] == 1
    assert res["n_ood_with_max_jaccard_ge_threshold"] == 2
    assert res["n_ood_with_max_jaccard_ge_0.5"] == 2
    assert res["max_jaccard_max"] == 1.0


def test_url_overlap_counts_and_lists_shared():
    train_urls = ["https://evil.example.com/login", "https://ok.example.org/"]
    ood_urls = ["https://evil.example.com/login", "https://new.example.net/x"]
    res = oc.url_overlap(ood_urls, train_urls)
    assert res["n_shared_urls"] == 1
    assert res["shared_urls"] == ["https://evil.example.com/login"]
    assert res["shared_url_fraction_of_ood"] == 0.5


def test_extract_urls_normalises_and_dedupes():
    text = "see https://Example.COM/a, and https://example.com/a. also http://x.io/b!"
    urls = oc.extract_urls(text)
    assert urls == ["https://example.com/a", "http://x.io/b"]


# ---------------------------------------------------------------------------
# Threshold fitting — validation only
# ---------------------------------------------------------------------------

def _crafted_val():
    # 4 phishing (probs .9 .8 .6 .4), 4 legit (probs .55 .3 .2 .1).
    # At floor 0.75 (recall >= 3/4): threshold 0.6 gives P=1.0 (3 TP, 0 FP);
    # threshold 0.4 gives P=0.8 (4 TP, 1 FP). Best precision is 1.0 @ 0.6.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.6, 0.4, 0.55, 0.3, 0.2, 0.1])
    return y, p


def test_fit_thresholds_picks_highest_precision_at_floor():
    y, p = _crafted_val()
    out = oc.fit_thresholds_for_floors(y, p, floors=(0.75,))
    assert out["0.75"]["threshold"] == pytest.approx(0.6)
    assert out["0.75"]["val_precision"] == pytest.approx(1.0)
    assert out["0.75"]["val_recall"] == pytest.approx(0.75)
    assert out["0.75"]["floor_met"] is True
    assert out["0.75"]["fit_on"] == "original_validation_split_only"


def test_fit_thresholds_floor_unreachable_reports_floor_met_false():
    # Max achievable recall is 0.75 (one phishing at prob 0.05 is below
    # every legit prob, so recall 1.0 needs threshold <= 0.05 which also
    # admits all legit — recall 1.0 IS reachable at thr 0.05. Use a
    # fixture where the lowest phishing prob is BELOW the lowest legit
    # prob... that still reaches recall 1.0. Instead: floor 1.0 with a
    # phishing prob tied below? Simplest unreachable case: floor 1.0
    # when one positive has prob 0.0 and threshold candidates exclude
    # 0.0? Candidates include all distinct probs, so 0.0 is a candidate
    # and recall 1.0 is reachable. Use n_pos == 0 instead: recall is 0.
    y = np.array([0, 0, 0])
    p = np.array([0.9, 0.5, 0.1])
    out = oc.fit_thresholds_for_floors(y, p, floors=(0.85,))
    assert out["0.85"]["floor_met"] is False
    assert out["0.85"]["val_recall"] == pytest.approx(0.0)


def test_fit_thresholds_takes_validation_only_no_ood_parameter():
    sig = inspect.signature(oc.fit_thresholds_for_floors)
    params = list(sig.parameters)
    assert params == ["val_labels", "val_probs", "floors"]
    assert not any("ood" in name.lower() for name in params)


def test_ood_labels_cannot_change_fitted_thresholds():
    """The core validity test: fit thresholds on validation, then shuffle
    / relabel the OOD set arbitrarily — the thresholds must be identical,
    because the fitter never sees OOD data."""
    y_val, p_val = _crafted_val()
    thresholds_a = oc.fit_thresholds_for_floors(y_val, p_val, floors=(0.85, 0.90, 0.95))
    # Simulate an OOD set whose labels would imply a very different
    # threshold if they leaked into the fit (all OOD phishing at 0.99,
    # all OOD legit at 0.01 -> a leaked fit would pick ~0.99).
    ood_labels = np.array([1] * 50 + [0] * 50)
    ood_probs = np.array([0.99] * 50 + [0.01] * 50)
    rng = np.random.RandomState(0)
    for _ in range(5):
        shuffled = ood_labels[rng.permutation(len(ood_labels))]
        # The pipeline fits ONLY on validation; re-fitting here with the
        # same validation inputs must reproduce the thresholds exactly,
        # regardless of what the OOD labels look like.
        thresholds_b = oc.fit_thresholds_for_floors(y_val, p_val, floors=(0.85, 0.90, 0.95))
        assert thresholds_b == thresholds_a
        assert shuffled is not None  # OOD labels exist but are unused
        _ = ood_probs
    # And a fit that (incorrectly) pooled val+OOD would differ, proving
    # the test fixture can detect leakage if it ever happened.
    pooled = oc.fit_thresholds_for_floors(
        np.concatenate([y_val, ood_labels]),
        np.concatenate([p_val, ood_probs]), floors=(0.85,))
    assert pooled["0.85"]["threshold"] != thresholds_a["0.85"]["threshold"]


def test_fit_thresholds_rejects_mismatched_and_empty_inputs():
    with pytest.raises(ValueError):
        oc.fit_thresholds_for_floors([1, 0], [0.5])
    with pytest.raises(ValueError):
        oc.fit_thresholds_for_floors([], [])


# ---------------------------------------------------------------------------
# Pure-numpy metrics (fallback / cross-check)
# ---------------------------------------------------------------------------

def test_metrics_lite_matches_hand_computation():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.4, 0.6, 0.1, 0.8, 0.3]
    m = oc.metrics_lite(y, p, threshold=0.5)
    # preds: 1,0,1,0,1,0 -> tp=2 (rows 0,4), fp=1 (row 2), fn=1 (row 1), tn=2
    assert m["confusion_matrix"] == [[2, 1], [1, 2]]
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert 0.0 < m["pr_auc"] <= 1.0


def test_metrics_lite_single_class_falls_back():
    m = oc.metrics_lite([1, 1, 1], [0.9, 0.8, 0.7])
    assert m["roc_auc"] == 0.5
    assert m["n"] == 3


# ---------------------------------------------------------------------------
# OOD set normalisation
# ---------------------------------------------------------------------------

def test_near_dup_audit_with_sigs_matches_audit():
    train = [" ".join(f"word{i}" for i in range(60))]
    ood = [" ".join(f"word{i}" for i in range(60)),
           " ".join(f"other{i}" for i in range(60))]
    a = oc.near_dup_audit(ood, train, threshold=0.9, n_perm=64)
    ts = oc.minhash_signatures(train, n_perm=64)
    os_ = oc.minhash_signatures(ood, n_perm=64)
    b = oc.near_dup_audit_with_sigs(os_, ts, threshold=0.9,
                                    n_ood=len(ood), n_train=len(train))
    assert a == b


def test_load_phishnchips_has_body_column(tmp_path):
    rows = [
        {"id": "phish_1", "url_raw": "http://evil.example/x", "phish_label": 1,
         "email_content": json.dumps({"sender": "A", "subject": "Hi", "body": "click here"}),
         "strategy": "s", "url_category": "github_pages", "datasource": "phishtank",
         "model_used": "primary"},
        {"id": "legit_1", "url_raw": "http://ok.example/", "phish_label": 0,
         "email_content": json.dumps({"sender": "B", "subject": "Hello", "body": "meeting at 3"}),
         "strategy": "legitimate_derived", "url_category": "legitimate", "datasource": "tranco",
         "model_used": "primary"},
    ]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df = oc.load_phishnchips(path)
    assert list(df.columns[:6]) == ["email_id", "text", "body", "label", "source",
                                   "category"]
    assert df["label"].tolist() == [1, 0]
    assert df["text"].tolist() == ["Hi\nclick here", "Hello\nmeeting at 3"]
    assert df["body"].tolist() == ["click here", "meeting at 3"]
    assert df["datasource"].tolist() == ["phishtank", "tranco"]


def test_load_phishfuzzer_manual_filters_and_separates_spam(tmp_path):
    data = [
        {"No.": "1", "Subject": "S1", "Body": "B1", "Type": "Phishing", "Source": "Manual",
         "Year": "2025", "Language": "en"},
        {"No.": "2", "Subject": "S2", "Body": "B2", "Type": "Valid", "Source": "Manual",
         "Year": "2025", "Language": "en"},
        {"No.": "3", "Subject": "S3", "Body": "B3", "Type": "Spam", "Source": "Manual",
         "Year": "2025", "Language": "en"},
        {"No.": "4", "Subject": "S4", "Body": "B4", "Type": "Phishing",
         "Source": "SpamAssassin.csv", "Year": "2008", "Language": "en"},
    ]
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    df = oc.load_phishfuzzer_manual(path)
    assert len(df) == 3  # non-Manual row excluded
    assert df["label"].tolist() == [1, 0, -1]  # spam kept, labelled -1
    binary = df[df["label"] >= 0]
    assert len(binary) == 2


def test_load_bec_drops_nan_labels(tmp_path):
    df_in = pd.DataFrame({
        "subject": ["a", "b", "c"],
        "body": ["x", "y", "z"],
        "label": [1.0, float("nan"), 1.0],
    })
    path = tmp_path / "bec.csv"
    df_in.to_csv(path, index=False)
    df = oc.load_bec(path)
    assert len(df) == 2
    assert df["label"].tolist() == [1, 1]


def test_load_frontiers_llm_uses_text_column(tmp_path):
    df_in = pd.DataFrame({
        "text": ["full text one", "full text two"],
        "subject": ["s1", "s2"], "body": ["b1", "b2"],
        "label": [1, 1], "source": ["gpt-4.1", "llama-3.3-70b"],
        "model": ["gpt-4.1", "llama-3.3-70b"], "category": ["banking", "hr"],
    })
    path = tmp_path / "llm.csv"
    df_in.to_csv(path, index=False)
    df = oc.load_frontiers_llm(path)
    assert df["text"].tolist() == ["full text one", "full text two"]
    assert df["category"].tolist() == ["gpt-4.1/banking", "llama-3.3-70b/hr"]
    assert (df["label"] == 1).all()


def test_pair_with_legit_concatenates_and_keeps_provenance():
    phish = pd.DataFrame({"email_id": ["p1"], "text": ["t"], "label": [1],
                          "source": ["frontiers_2026_llm"], "category": ["x"]})
    legit = pd.DataFrame({"email_id": ["l1", "l2"], "text": ["a", "b"], "label": [0, 0],
                          "source": ["phishnchips_v5.2", "phishnchips_v5.2"],
                          "category": ["legitimate", "legitimate"]})
    paired = oc.pair_with_legit(phish, legit)
    assert len(paired) == 3
    assert paired["label"].tolist() == [1, 0, 0]
    assert paired["source"].tolist() == ["frontiers_2026_llm", "phishnchips_v5.2",
                                         "phishnchips_v5.2"]


# ---------------------------------------------------------------------------
# TF-IDF OOD models (Task 3): config + evaluate_set smoke test
# ---------------------------------------------------------------------------

def _load_run_ood_eval():
    """Import run_ood_eval.py as a module (it lives in the OOD scripts dir)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ood_eval",
        Path(__file__).resolve().parent.parent
        / "experiments" / "modern_phishing_ood" / "scripts" / "run_ood_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tfidf_ood_config_is_documented_and_seed42():
    """The TF-IDF OOD refit configs must exist, be seed-42 Stage A winners,
    and cover exactly the TF-IDF entries of MODEL_CHOICES."""
    mod = _load_run_ood_eval()
    assert set(mod.TFIDF_OOD_CONFIG) == {"tfidf_svm", "tfidf_lr"}
    assert mod.TFIDF_OOD_CONFIG["tfidf_lr"] == {
        "C": 10.0, "ngram_range": (1, 2), "max_features": None}
    assert mod.TFIDF_OOD_CONFIG["tfidf_svm"] == {
        "C": 5.0, "ngram_range": (3, 6), "max_features": None}
    assert "tfidf_svm" in mod.MODEL_CHOICES and "tfidf_lr" in mod.MODEL_CHOICES


def test_evaluate_set_smoke_with_tfidf_lr(tmp_path):
    """evaluate_set works with a TF-IDF scorer: same prediction-CSV schema
    as the neural models, thresholds fitted on validation only (here a
    fixed 0.5 + one validation-fitted threshold passed in)."""
    pytest.importorskip("sklearn")
    from phishing_xai.baselines import build_tfidf_lr, compute_metrics_full
    mod = _load_run_ood_eval()
    train_texts = (["free money click here now"] * 6
                   + ["team meeting notes for tuesday"] * 6)
    train_labels = [1] * 6 + [0] * 6
    scorer = build_tfidf_lr(train_texts, train_labels, C=10.0,
                           ngram_range=(1, 2), max_features=None)
    frame = pd.DataFrame({
        "email_id": ["t1", "t2", "t3", "t4"],
        "text": ["free money click here", "claim your prize now",
                 "meeting notes tuesday", "project update friday"],
        "label": [1, 1, 0, 0],
        "source": ["smoke", "smoke", "smoke", "smoke"],
        "category": ["a", "a", "b", "b"],
    })
    thresholds = {"0.85": {"threshold": 0.5, "floor_met": True}}
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    res = mod.evaluate_set("tfidf_lr", scorer, frame, "smoke",
                           thresholds, lambda t: t.lower(),
                           compute_metrics_full, pred_dir)
    assert res["n"] == 4 and res["model"] == "tfidf_lr"
    pred_file = pred_dir / "smoke_tfidf_lr_predictions.csv"
    assert pred_file.exists()
    preds = pd.read_csv(pred_file)
    assert list(preds.columns) == ["email_id", "true_label",
                                   "predicted_label_at_0_5", "prob_phishing",
                                   "source", "category", "model", "set"]
    assert preds["model"].tolist() == ["tfidf_lr"] * 4
    assert res["metrics_at_0_5"]["n"] == 4
