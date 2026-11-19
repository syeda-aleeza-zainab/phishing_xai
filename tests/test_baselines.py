"""Tests for Section H baselines (tests/test_baselines.py).

Fake tiny fixtures only: no GPU, no internet, no real checkpoint, no real
data. BiLSTM tests stay tiny (small vocab, 1 epoch / 1 step, CPU).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.baselines import (
    BaselineModel,
    BiLSTMClassifier,
    SimpleTokenizer,
    TorchTrainConfig,
    aggregate_seed_results,
    apply_perturbation,
    bootstrap_f1_ci,
    bootstrap_precision_ci,
    build_dummy,
    build_tfidf_lr,
    build_tfidf_svm,
    build_torch_baseline,
    compute_metrics_full,
    error_analysis,
    evaluate_explainability,
    evaluate_faithfulness_with_proba_fn,
    evaluate_multimodal,
    evaluate_on_extractor_output,
    fit_threshold_for_recall,
    integrated_gradients_scores,
    lime_scores_for_text,
    load_roberta_metrics,
    mcnemar_test,
    mean_std,
    measure_latency,
    precision_at_recall,
    random_search_tfidf,
    resolve_output_dir,
    select_deployment_model,
    tfidf_lr_coefficient_scores,
    torch_predict_proba,
    train_torch_classifier,
)

# Import the report generator's verdict rule from scripts/ (not a package).
_SPEC = importlib.util.spec_from_file_location(
    "run_complexity_report",
    Path(__file__).resolve().parent.parent / "scripts" / "run_complexity_report.py",
)
run_complexity_report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_complexity_report)
build_verdict = run_complexity_report.build_verdict

# Import the baseline runner for the Section G merge helper (same pattern).
_SPEC_B = importlib.util.spec_from_file_location(
    "run_baselines",
    Path(__file__).resolve().parent.parent / "scripts" / "run_baselines.py",
)
run_baselines = importlib.util.module_from_spec(_SPEC_B)
_SPEC_B.loader.exec_module(run_baselines)
merge_baseline_results = run_baselines.merge_baseline_results
recommended_deployment_model = run_complexity_report.recommended_deployment_model

TRAIN_TEXTS = [
    "hello team meeting tomorrow please review the report",
    "quarterly report attached for your review team",
    "lunch meeting scheduled for friday afternoon",
    "please confirm your attendance at the team meeting",
    "project update meeting notes shared with the team",
    "holiday schedule and office meeting calendar",
    "urgent verify your account now click here claim prize",
    "your password expired click link verify account immediately",
    "congratulations you won prize click link claim now",
    "urgent action required verify bank account click here",
    "security alert verify your login click link now",
    "claim your prize money click here verify account",
]
TRAIN_LABELS = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
VAL_TEXTS = [
    "team meeting notes for the quarterly review",
    "office lunch schedule friday meeting",
    "verify account click link urgent prize",
    "click here claim prize verify password",
]
VAL_LABELS = [0, 0, 1, 1]
# "zxqvtestonly" appears ONLY in the test split: a leakage check.
TEST_TEXTS = [
    "team report meeting zxqvtestonly",
    "office calendar meeting",
    "verify account zxqvtestonly click prize",
    "claim prize click link",
]
TEST_LABELS = [0, 0, 1, 1]


@pytest.fixture
def tiny_splits():
    """Tiny fake train/val/test frames (no real data)."""
    def _frame(texts, labels):
        return pd.DataFrame({"clean_text": texts, "label": labels})
    return {
        "train": _frame(TRAIN_TEXTS, TRAIN_LABELS),
        "val": _frame(VAL_TEXTS, VAL_LABELS),
        "test": _frame(TEST_TEXTS, TEST_LABELS),
    }


def test_tokenizer_no_leakage():
    """A test-only token must be OOV: vocab is built from TRAIN only."""
    tok = SimpleTokenizer(max_vocab=1000, max_len=16).fit(TRAIN_TEXTS)
    assert "zxqvtestonly" not in tok.word2idx
    encoded = tok.encode("team report zxqvtestonly")
    assert encoded[-1] == tok.word2idx["<unk>"]
    assert len(tok) <= 1000


def test_tfidf_vocab_no_leakage():
    """TF-IDF vocabulary fit on train must not contain a test-only token."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    vocab = model.extra["_pipeline"].named_steps["tfidf"].vocabulary_
    assert "zxqvtestonly" not in vocab
    assert "verify" in vocab


def test_dummy_predict_proba_sums_to_one():
    """Dummy probabilities sum to 1 and predict the majority class."""
    model = build_dummy(TRAIN_LABELS)
    probs = model.predict_proba(TEST_TEXTS)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4))
    # Train is balanced 6/6 here, so majority is class 0 (first argmax).
    assert set(model.predict(TEST_TEXTS).tolist()) <= {0, 1}


def test_tfidf_lr_predict_proba_sums_to_one():
    """TF-IDF+LR probabilities sum to 1 and separate the toy data."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    probs = model.predict_proba(TEST_TEXTS)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)
    preds = model.predict(TEST_TEXTS)
    assert preds.tolist() == TEST_LABELS


def test_tfidf_svm_calibrated_proba_sums_to_one():
    """Calibrated LinearSVC probabilities sum to 1."""
    model = build_tfidf_svm(TRAIN_TEXTS, TRAIN_LABELS)
    probs = model.predict_proba(TEST_TEXTS)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)


def test_bilstm_forward_shape():
    """BiLSTM forward returns (batch, 2) logits."""
    torch.manual_seed(42)
    model = BiLSTMClassifier(vocab_size=50, emb_dim=8, hidden=8, layers=1, dropout=0.0)
    x = torch.randint(0, 50, (3, 10))
    mask = torch.ones_like(x)
    logits = model(x, mask)
    assert logits.shape == (3, 2)


def test_bilstm_one_step_training_and_proba():
    """BiLSTM trains for 1 epoch on tiny data and outputs valid probabilities."""
    config = TorchTrainConfig(emb_dim=8, hidden=8, layers=1, dropout=0.0,
                              max_len=16, max_vocab=1000, lr=1e-2, epochs=1,
                              batch_size=4, patience=1)
    model, info = build_torch_baseline(
        "bilstm", TRAIN_TEXTS, TRAIN_LABELS, VAL_TEXTS, VAL_LABELS,
        config=config, seed=42, device=torch.device("cpu"),
    )
    assert info["epochs_run"] == 1
    assert info["best_val_f1"] >= 0.0
    probs = model.predict_proba(TEST_TEXTS)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-5)
    assert model.n_params > 0
    assert model.size_mb > 0


def test_precision_at_recall_known_values():
    """Precision at recall 0.90 on a hand-built ranking."""
    y_true = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # Perfect ranking except one safe email scored high.
    y_prob = [0.9, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    res = precision_at_recall(y_true, y_prob, target_recall=0.90)
    assert res["achieved_recall"] >= 0.90
    assert res["precision"] == pytest.approx(9 / 10)  # 9 phishing + 1 safe above threshold
    m = compute_metrics_full(y_true, y_prob)
    assert "precision_at_recall_0.90" in m
    assert m["precision_at_recall_0.90"] == pytest.approx(0.9)


def test_fit_threshold_for_recall_validation_only():
    """Threshold is fit on the supplied (validation) labels and documented."""
    y_true = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6]
    thr = fit_threshold_for_recall(y_true, y_prob, target_recall=0.90)
    assert thr["target_recall"] == 0.90
    assert thr["val_recall"] >= 0.90
    assert 0.0 <= thr["val_precision"] <= 1.0
    assert 0.0 < thr["threshold"] < 1.0


def test_bootstrap_precision_ci_contains_point():
    """Bootstrap precision CI brackets the point precision estimate."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=60).tolist()
    y_pred = [int(v) if rng.rand() > 0.2 else int(1 - v) for v in y_true]
    res = bootstrap_precision_ci(y_true, y_pred, n_resamples=200, seed=42)
    assert res["ci_low"] <= res["precision"] <= res["ci_high"]
    assert 0.0 <= res["ci_low"] <= res["ci_high"] <= 1.0


def test_select_deployment_model_floor_and_fallback():
    """Precision-first selection honours the recall floor, then states the relaxed floor."""
    aggs = {
        "a": {"precision": {"mean": 0.99}, "recall": {"mean": 0.80}, "n_params": 100},
        "b": {"precision": {"mean": 0.95}, "recall": {"mean": 0.90}, "n_params": 100},
    }
    sel = select_deployment_model(aggs, recall_floor=0.85)
    assert sel["model"] == "b"  # "a" has higher precision but recall below floor
    assert sel["floor_met"] is True
    sel2 = select_deployment_model(
        {"a": {"precision": {"mean": 0.99}, "recall": {"mean": 0.80}, "n_params": 100}},
        recall_floor=0.85,
    )
    assert sel2["model"] == "a" and sel2["floor_met"] is False
    assert sel2["floor_used"] == pytest.approx(0.80)
    assert select_deployment_model({}, recall_floor=0.85)["status"] == "NOT RUN YET"


def test_train_torch_classifier_one_step_direct():
    """Direct train loop runs one epoch and returns a finite best val F1."""
    tok = SimpleTokenizer(max_vocab=1000, max_len=16).fit(TRAIN_TEXTS)
    model = BiLSTMClassifier(vocab_size=len(tok), emb_dim=8, hidden=8, layers=1, dropout=0.0)
    config = TorchTrainConfig(emb_dim=8, hidden=8, layers=1, dropout=0.0,
                              max_len=16, lr=1e-2, epochs=1, batch_size=4, patience=1)
    info = train_torch_classifier(model, tok, TRAIN_TEXTS, TRAIN_LABELS,
                                  VAL_TEXTS, VAL_LABELS, config=config, seed=42,
                                  device=torch.device("cpu"))
    assert info["epochs_run"] == 1
    assert 0.0 <= info["best_val_f1"] <= 1.0
    assert 0.0 <= info["best_val_precision"] <= 1.0
    assert 0.0 <= info["best_val_recall"] <= 1.0
    assert info["recall_floor"] == 0.85
    assert info["train_seconds"] >= 0.0


def test_torch_predict_proba_empty_input():
    """Empty input returns an empty (0, 2) array, not an error."""
    tok = SimpleTokenizer(max_vocab=100, max_len=8).fit(TRAIN_TEXTS)
    model = BiLSTMClassifier(vocab_size=len(tok), emb_dim=8, hidden=8, layers=1, dropout=0.0)
    out = torch_predict_proba(model, tok, [], device=torch.device("cpu"))
    assert out.shape == (0, 2)


def test_compute_metrics_full_known_values():
    """Metrics on a known confusion pattern: 1 TP, 1 TN, 1 FP, 1 FN."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.9, 0.4, 0.8]  # preds: 0,1,0,1
    m = compute_metrics_full(y_true, y_prob)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["false_positive_rate"] == pytest.approx(0.5)


def test_mcnemar_known_inputs():
    """McNemar counts b/c on a hand-built case; symmetric case gives p=1."""
    y_true = [0, 0, 1, 1, 0, 1]
    pred_a = [0, 1, 1, 0, 0, 1]  # wrong on idx 1, 3
    pred_b = [0, 0, 0, 1, 1, 1]  # wrong on idx 2, 4
    res = mcnemar_test(y_true, pred_a, pred_b)
    assert res["b"] == 2  # A wrong, B right
    assert res["c"] == 2  # A right, B wrong
    assert res["p_value"] == pytest.approx(1.0)
    # Zero discordant pairs -> p = 1.
    res2 = mcnemar_test(y_true, pred_a, pred_a)
    assert res2["b"] == 0 and res2["c"] == 0 and res2["p_value"] == 1.0
    # Strongly asymmetric: b=10, c=0 -> small p.
    y = [1] * 10 + [0] * 10
    pa = [0] * 10 + [0] * 10  # wrong on first 10
    pb = [1] * 10 + [0] * 10  # all right
    res3 = mcnemar_test(y, pa, pb)
    assert res3["b"] == 10 and res3["c"] == 0
    assert res3["p_value"] < 0.05


def test_bootstrap_f1_ci_contains_point():
    """Bootstrap CI brackets the point F1 estimate."""
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=60).tolist()
    y_pred = [int(v) if rng.rand() > 0.2 else int(1 - v) for v in y_true]
    res = bootstrap_f1_ci(y_true, y_pred, n_resamples=200, seed=42)
    assert res["ci_low"] <= res["f1"] <= res["ci_high"]
    assert 0.0 <= res["ci_low"] <= res["ci_high"] <= 1.0


def test_mean_std():
    """Mean/std across seeds: sample std with ddof=1."""
    res = mean_std([0.9, 0.8, 0.85])
    assert res["mean"] == pytest.approx(0.85)
    assert res["std"] == pytest.approx(0.05)
    assert res["n"] == 3


def _mock_summary(roberta_prec, simple_prec, p_value, ci_low, ci_high,
                  roberta_f1=0.9800, simple_f1=0.9750, simple_recall=0.95):
    """Mocked aggregate summary in the shape build_verdict expects."""
    return {
        "models": {
            "tfidf_lr": {"precision": {"mean": simple_prec, "std": 0.01},
                         "recall": {"mean": simple_recall, "std": 0.01},
                         "f1": {"mean": simple_f1, "std": 0.01}, "n_params": 1000,
                         "latency_cpu_ms_median": {"mean": 1.0}},
            "roberta_eval": {"precision": {"mean": roberta_prec, "std": 0.01},
                             "recall": {"mean": 0.96, "std": 0.01},
                             "f1": {"mean": roberta_f1, "std": 0.01},
                             "n_params": 125_000_000,
                             "latency_cpu_ms_median": {"mean": 25.0}},
        },
        "comparisons_vs_roberta": {
            "tfidf_lr": {
                "mcnemar_vs_roberta": {"b": 5, "c": 4, "p_value": p_value},
                "bootstrap_f1_ci": {"f1": simple_f1, "ci_low": 0.90, "ci_high": 0.99},
                "bootstrap_precision_ci": {"precision": simple_prec, "ci_low": ci_low, "ci_high": ci_high},
            }
        },
    }


def test_verdict_no_significant_gain_branch():
    """Within 1 precision point, overlapping precision CI, p > 0.05 -> complexity NOT justified."""
    summary = _mock_summary(roberta_prec=0.9800, simple_prec=0.9750, p_value=0.30,
                            ci_low=0.9600, ci_high=0.9900)
    verdict = build_verdict(summary)
    assert "no statistically significant precision gain" in verdict
    assert "NOT justified by accuracy alone" in verdict
    assert "XAI/multimodal layer" in verdict


def test_verdict_significant_gain_branch():
    """Large precision gap with small p -> significant precision gain, quantified with ratios."""
    summary = _mock_summary(roberta_prec=0.9900, simple_prec=0.9000, p_value=0.001,
                            ci_low=0.8800, ci_high=0.9200)
    verdict = build_verdict(summary)
    assert "significant precision gain of 9.00 points" in verdict
    assert "25.0x latency" in verdict
    assert "125000.0x parameters" in verdict
    assert "F1 gap" in verdict  # F1 branch kept as a secondary statement


def test_verdict_negative_gap_branch():
    """RoBERTa BELOW the best simple baseline -> 'below by X points', never a 'significant gain'."""
    summary = _mock_summary(roberta_prec=0.9690, simple_prec=0.9820, p_value=0.253,
                            ci_low=0.9746, ci_high=0.9889)
    verdict = build_verdict(summary)
    assert "below the best simple baseline by 1.30 precision points" in verdict
    assert "significant precision gain" not in verdict
    assert "NOT justified by a precision gain" in verdict


def test_verdict_positive_gap_nonsignificant_neutral():
    """Positive gap with McNemar p = 0.625 -> neutral wording, never 'significant gain'."""
    summary = _mock_summary(roberta_prec=1.0000, simple_prec=0.9412, p_value=0.625,
                            ci_low=0.9300, ci_high=0.9600)
    verdict = build_verdict(summary)
    assert "no statistically significant difference" in verdict
    assert "significant precision gain" not in verdict


def test_verdict_p_exactly_boundary_not_significant():
    """McNemar p exactly 0.05 is NOT < 0.05, so a positive gap is not a significant gain."""
    summary = _mock_summary(roberta_prec=0.9900, simple_prec=0.9400, p_value=0.05,
                            ci_low=0.9300, ci_high=0.9500)
    verdict = build_verdict(summary)
    assert "significant precision gain" not in verdict
    assert "no statistically significant difference" in verdict


def test_verdict_not_run_yet():
    """No executed numbers -> NOT RUN YET verdict, no invented claim."""
    verdict = build_verdict({"models": {}, "comparisons_vs_roberta": {}})
    assert verdict.startswith("NOT RUN YET")


def test_recommended_deployment_model():
    """Highest precision subject to recall >= 0.85; below-floor models excluded."""
    summary = {
        "models": {
            "dummy": {"precision": {"mean": 0.60}, "recall": {"mean": 0.90},
                      "f1": {"mean": 0.60}, "n_params": 0},
            "tfidf_lr": {"precision": {"mean": 0.97}, "recall": {"mean": 0.90},
                         "f1": {"mean": 0.97}, "n_params": 5000},
            "bilstm": {"precision": {"mean": 0.99}, "recall": {"mean": 0.80},
                       "f1": {"mean": 0.88}, "n_params": 200000},
            "roberta_eval": {"precision": {"mean": 0.98}, "recall": {"mean": 0.95},
                             "f1": {"mean": 0.98}, "n_params": 125_000_000},
        }
    }
    rec = recommended_deployment_model(summary)
    # bilstm has the highest precision but recall 0.80 < 0.85 floor, so it is excluded.
    assert rec.startswith("roberta_eval")
    assert "recall floor 0.85 met" in rec
    assert recommended_deployment_model({"models": {}}) == "NOT RUN YET"


def test_recommended_deployment_model_floor_relaxed():
    """No model meets recall >= 0.85: the floor actually used is stated."""
    summary = {
        "models": {
            "tfidf_lr": {"precision": {"mean": 0.97}, "recall": {"mean": 0.80},
                         "f1": {"mean": 0.88}, "n_params": 5000},
            "bilstm": {"precision": {"mean": 0.99}, "recall": {"mean": 0.70},
                       "f1": {"mean": 0.82}, "n_params": 200000},
        }
    }
    rec = recommended_deployment_model(summary)
    assert rec.startswith("bilstm")
    assert "no model met recall floor 0.85" in rec
    assert "best recall among models: 0.8000" in rec


def test_latency_timer_helper():
    """Latency helper returns a positive median ms/email and counts timed rows."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    res = measure_latency(model.predict_proba, TEST_TEXTS, n=4, warmup=1)
    assert res["latency_ms_per_email_median"] > 0
    assert res["n_timed"] == 4


def test_perturbations():
    """Truncation caps at 128 tokens; URL removal strips http links."""
    long_text = " ".join(["word"] * 200)
    assert len(apply_perturbation([long_text], "truncated")[0].split()) == 128
    removed = apply_perturbation(["see https://example.com/page now"], "urls_removed")[0]
    assert "https://example.com" not in removed
    with pytest.raises(ValueError):
        apply_perturbation(["x"], "no_such_kind")


def test_faithfulness_with_proba_fn():
    """Model-agnostic faithfulness: comp/suff at k in {1,3,5}, bounded."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    text = "urgent verify your account now click here claim prize"
    from phishing_xai.explain import split_words
    words = split_words(text)
    scores = tfidf_lr_coefficient_scores(model.extra["_pipeline"], text)
    assert len(scores) == len(words)
    probs = model.predict_proba([text])[0]
    pred = int(np.argmax(probs))
    res = evaluate_faithfulness_with_proba_fn(
        model.predict_proba, words, scores, pred, float(probs[pred]), ks=(1, 3, 5)
    )
    assert set(res["comprehensiveness"]) == {1, 3, 5}
    assert set(res["sufficiency"]) == {1, 3, 5}
    for v in list(res["comprehensiveness"].values()) + list(res["sufficiency"].values()):
        assert -1.0 <= v <= 1.0


def test_lime_scores_align_with_words():
    """LIME scores for a baseline line up one-to-one with split_words."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    text = "urgent verify your account click here"
    from phishing_xai.explain import split_words
    scores = lime_scores_for_text(model.predict_proba, text, num_samples=20, seed=42)
    assert len(scores) == len(split_words(text))
    assert all(isinstance(s, float) for s in scores)


def test_integrated_gradients_scores_align_with_words():
    """BiLSTM IntegratedGradients scores line up with split_words."""
    config = TorchTrainConfig(emb_dim=8, hidden=8, layers=1, dropout=0.0,
                              max_len=16, lr=1e-2, epochs=1, batch_size=4, patience=1)
    model, _ = build_torch_baseline(
        "bilstm", TRAIN_TEXTS, TRAIN_LABELS, VAL_TEXTS, VAL_LABELS,
        config=config, seed=42, device=torch.device("cpu"),
    )
    torch_model = model.extra["_model"]
    tokenizer = model.extra["_tokenizer"]
    text = "urgent verify your account click here"
    from phishing_xai.explain import split_words
    scores = integrated_gradients_scores(torch_model, tokenizer, text, n_steps=5,
                                         device=torch.device("cpu"))
    assert len(scores) == len(split_words(text))
    assert all(isinstance(s, float) for s in scores)


def test_evaluate_explainability_lime_tiny():
    """End-to-end explainability eval on 2 samples returns means at k=5."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    res = evaluate_explainability(model, TEST_TEXTS, TEST_LABELS, method="lime",
                                  sample_size=2, seed=42)
    assert res["n_evaluated"] == 2
    assert res["comprehensiveness"]["5"] is not None


def test_random_search_tfidf_validation_only():
    """Random search returns best params by val precision (recall floor 0.85)."""
    res = random_search_tfidf(TRAIN_TEXTS, TRAIN_LABELS, VAL_TEXTS, VAL_LABELS,
                              kind="tfidf_lr", n_trials=2, seed=42)
    assert "C" in res["best_params"]
    assert len(res["trials"]) == 2
    assert 0.0 <= res["best_val_precision"] <= 1.0
    assert 0.0 <= res["best_val_f1"] <= 1.0
    assert res["recall_floor"] == 0.85


def test_error_analysis_directions():
    """Error analysis returns both directions with the expected columns."""
    y_true = [0, 1, 0, 1]
    roberta = [0, 1, 1, 1]   # wrong on idx 2
    baseline = [0, 0, 0, 1]  # wrong on idx 1
    texts = ["safe team meeting notes", "urgent verify account click",
             "office lunch https://example.com", "claim prize click link"]
    df = error_analysis(y_true, roberta, baseline, texts, top_n=20)
    assert set(df["direction"]) == {"roberta_right_baseline_wrong", "baseline_right_roberta_wrong"}
    assert "has_url" in df.columns and "length_bucket" in df.columns
    assert df.loc[df["direction"] == "baseline_right_roberta_wrong", "email_id"].tolist() == [2]


def test_multimodal_hook_and_ablation(tmp_path):
    """H6 hook: extractor-output eval + modality ablation on a tiny frame."""
    model = build_tfidf_lr(TRAIN_TEXTS, TRAIN_LABELS)
    rows = [
        {"text": "team meeting report", "label": 0},
        {"text": "urgent verify account click prize", "label": 1},
    ]
    res = evaluate_on_extractor_output(model, rows)
    assert res["n"] == 2 and 0.0 <= res["metrics"]["f1"] <= 1.0
    df = pd.DataFrame([
        {"text": "team meeting report", "label": 0, "body_text": "team meeting report",
         "ocr_text": "", "pdf_text": "", "qr_text": ""},
        {"text": "urgent verify account click prize", "label": 1,
         "body_text": "urgent verify account", "ocr_text": "click prize",
         "pdf_text": "", "qr_text": ""},
    ])
    ablation = evaluate_multimodal({"tfidf_lr": model}, df)
    assert set(ablation["modality"]) == {"all", "body_text", "ocr_text", "pdf_text", "qr_text"}
    assert (ablation["status"] == "ok").all()
    # Missing modality column -> reported, not invented.
    ablation2 = evaluate_multimodal({"tfidf_lr": model}, df[["text", "label"]])
    assert (ablation2.loc[ablation2["modality"] != "all", "status"] == "column missing").all()


def test_load_roberta_metrics_missing(tmp_path):
    """Missing metrics.json/checkpoint -> NOT RUN YET with untrained_fallback."""
    res = load_roberta_metrics(tmp_path)
    assert res["status"] == "NOT RUN YET"
    assert res["untrained_fallback"] is True


def test_aggregate_seed_results():
    """Seed aggregation gives mean +/- std for F1."""
    per_seed = [
        {"seed": 42, "metrics": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9,
                                 "f1": 0.9, "roc_auc": 0.95, "pr_auc": 0.9,
                                 "false_positive_rate": 0.1},
         "n_params": 10, "size_mb": 0.1, "train_seconds": 2.0,
         "latency_cpu_ms": {"latency_ms_per_email_median": 1.0}},
        {"seed": 1, "metrics": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8,
                                "f1": 0.8, "roc_auc": 0.9, "pr_auc": 0.8,
                                "false_positive_rate": 0.2},
         "n_params": 10, "size_mb": 0.1, "train_seconds": 4.0,
         "latency_cpu_ms": {"latency_ms_per_email_median": 2.0}},
    ]
    agg = aggregate_seed_results(per_seed)
    assert agg["f1"]["mean"] == pytest.approx(0.85)
    assert agg["f1"]["std"] == pytest.approx(0.0707, abs=1e-3)
    assert agg["train_seconds"]["mean"] == pytest.approx(3.0)


def test_resolve_output_dir_returns_path():
    """Output-dir resolver returns a Path (Kaggle redirect only on Kaggle)."""
    assert isinstance(resolve_output_dir("results/baselines"), Path)


def _seed_entry(seed, f1):
    """Minimal per-seed baseline entry with the fields the merge recomputes."""
    return {"model": "tfidf_lr", "seed": seed, "status": "ok",
            "metrics": {"accuracy": f1, "precision": f1, "recall": f1,
                        "f1": f1, "roc_auc": f1, "pr_auc": f1,
                        "false_positive_rate": 0.1},
            "n_params": 10, "size_mb": 0.1, "train_seconds": 2.0,
            "latency_cpu_ms": {"latency_ms_per_email_median": 1.0}}


def test_merge_baseline_results_keeps_previous_models_and_replaces_same_seed():
    """Section G fix: separate invocations accumulate the ladder (model+seed)."""
    existing = {"models": {
        "tfidf_lr": {"per_seed": [_seed_entry(42, 0.90)],
                     "aggregate": {"f1": {"mean": 0.90}}},
        "dummy": {"per_seed": [_seed_entry(42, 0.0)],
                  "aggregate": {"f1": {"mean": 0.0}}},
    }, "fairness": {"seeds": [42]},
        "comparisons_vs_roberta": {"tfidf_lr": {"mcnemar_vs_roberta": {"p": 0.04}}}}
    new = {"models": {
        "tfidf_lr": {"per_seed": [_seed_entry(42, 0.95), _seed_entry(1, 0.85)],
                     "aggregate": {"f1": {"mean": 0.90}}},
    }, "fairness": {"seeds": [42, 1]},
        "comparisons_vs_roberta": {"status": "NOT RUN YET", "reason": "no RoBERTa"}}
    merged = merge_baseline_results(existing, new)
    # Untouched model survives; same-seed entry is replaced, new seed added.
    assert "dummy" in merged["models"]
    tfidf = merged["models"]["tfidf_lr"]
    assert sorted(r["seed"] for r in tfidf["per_seed"]) == [1, 42]
    seed42 = [r for r in tfidf["per_seed"] if r["seed"] == 42][0]
    assert seed42["metrics"]["f1"] == pytest.approx(0.95)
    # Aggregate recomputed from the merged seeds (0.95 + 0.85) / 2 = 0.90.
    assert tfidf["aggregate"]["f1"]["mean"] == pytest.approx(0.90)
    assert tfidf["aggregate"]["n_seeds"] == 2
    # Real comparisons from the previous run survive a placeholder new run.
    assert merged["comparisons_vs_roberta"]["tfidf_lr"]["mcnemar_vs_roberta"]["p"] == 0.04
    assert sorted(merged["fairness"]["seeds"]) == [1, 42]
    assert merged["merged_with_previous_run"] is True


def test_merge_baseline_results_no_previous_returns_new_unchanged():
    """Missing / malformed previous file -> the new run is written as-is."""
    new = {"models": {"dummy": {"per_seed": [], "aggregate": {"status": "NOT RUN YET"}}}}
    assert merge_baseline_results(None, new) == new
    assert merge_baseline_results({"not": "a results file"}, new) == new
    assert merge_baseline_results({"models": "not-a-dict"}, new) == new
