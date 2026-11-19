"""Tests for Section E (multimodal eval set + eval script + faithfulness).

No GPU, no internet, no real checkpoint, no tesseract/zbar needed:
generation uses Pillow/qrcode/PyMuPDF only, and every metric test runs
on toy predictions. The eval script's model path (``evaluate`` /
``score_email_roberta``) is intentionally NOT exercised here — it needs
a RoBERTa model and is covered by the smoke run in the Section E log.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_multimodal_eval_set import CATEGORIES, generate_eval_set
from scripts.run_multimodal_eval import (
    ABLATION_KEYS,
    ABLATION_LABELS,
    build_failure_table,
    compute_metrics,
    filtered_extraction,
    fit_threshold_precision_first,
    modality_texts,
    per_modality_recall,
    precision_first_summary,
    render_markdown,
    roc_auc,
)


def _read_labels(out_dir: Path):
    with open(out_dir / "labels.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# E1 — eval-set generator.
# ---------------------------------------------------------------------------
def test_generator_full_set_balance_and_categories(tmp_path):
    """Full generation: 120 files, 60 phishing / 60 benign, all categories."""
    out_dir = tmp_path / "eval"
    manifest = generate_eval_set(out_dir, seed=42)
    rows = _read_labels(out_dir)
    assert len(rows) == 120
    assert manifest["n_total"] == 120
    assert manifest["n_phishing"] == 60
    assert manifest["n_benign"] == 60
    assert sum(1 for r in rows if int(r["label"]) == 1) == 60
    assert sum(1 for r in rows if int(r["label"]) == 0) == 60
    from collections import Counter

    counts = Counter(r["category"] for r in rows)
    for category, (label, expected) in CATEGORIES.items():
        assert counts[category] == expected, category
        assert all(int(r["label"]) == label for r in rows if r["category"] == category)
    eml_files = sorted(p.name for p in out_dir.glob("*.eml"))
    assert len(eml_files) == 120
    assert {r["filename"] for r in rows} == set(eml_files)
    assert (out_dir / "manifest.json").exists()
    assert "SYNTHETIC" in manifest["synthetic_caveat"]


def test_generator_labels_csv_schema_and_split(tmp_path):
    """labels.csv has filename/label/category/split; split is 24 val / 96 test."""
    out_dir = tmp_path / "eval"
    generate_eval_set(out_dir, seed=42)
    rows = _read_labels(out_dir)
    assert {"filename", "label", "category", "split"} <= set(rows[0].keys())
    assert {r["label"] for r in rows} <= {"0", "1"}
    assert {r["split"] for r in rows} <= {"validation", "test"}
    assert sum(1 for r in rows if r["split"] == "validation") == 24
    assert sum(1 for r in rows if r["split"] == "test") == 96
    # Stratified: 12 phishing + 12 benign in validation.
    assert sum(1 for r in rows if r["split"] == "validation" and r["label"] == "1") == 12
    assert sum(1 for r in rows if r["split"] == "validation" and r["label"] == "0") == 12


def test_generator_determinism_same_seed(tmp_path):
    """Same seed -> identical labels.csv content and identical file list."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_eval_set(dir_a, seed=42)
    generate_eval_set(dir_b, seed=42)
    assert (dir_a / "labels.csv").read_bytes() == (dir_b / "labels.csv").read_bytes()
    assert sorted(p.name for p in dir_a.glob("*.eml")) == sorted(
        p.name for p in dir_b.glob("*.eml")
    )
    assert json.loads((dir_a / "manifest.json").read_text()) == json.loads(
        (dir_b / "manifest.json").read_text()
    )


def test_generator_different_seed_changes_labels(tmp_path):
    """A different seed changes the assignment (labels.csv differs)."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    generate_eval_set(dir_a, seed=42)
    generate_eval_set(dir_b, seed=7)
    assert (dir_a / "labels.csv").read_bytes() != (dir_b / "labels.csv").read_bytes()


def test_generator_debug_subset_balanced(tmp_path):
    """--debug subset: 12 files, 6 phishing / 6 benign, same categories subset."""
    out_dir = tmp_path / "debug"
    manifest = generate_eval_set(out_dir, seed=42, debug=True)
    rows = _read_labels(out_dir)
    assert len(rows) == 12
    assert manifest["debug"] is True
    assert sum(1 for r in rows if int(r["label"]) == 1) == 6
    assert sum(1 for r in rows if int(r["label"]) == 0) == 6


# ---------------------------------------------------------------------------
# E2 — metric computation on known toy predictions.
# ---------------------------------------------------------------------------
def test_compute_metrics_known_values():
    """Hand-computed case: 3 TP, 1 FP, 1 FN, 2 TN at threshold 0.5."""
    y_true = [1, 1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.6, 0.4, 0.7, 0.2, 0.1]
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["confusion_matrix"] == [[2, 1], [1, 3]]
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(0.75)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["n"] == 7


def test_compute_metrics_zero_predicted_positives():
    """Edge case: no predicted positives -> precision 0.0, not a crash/NaN."""
    m = compute_metrics([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4], threshold=0.5)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["confusion_matrix"] == [[2, 0], [2, 0]]
    assert m["false_positive_rate"] == 0.0


def test_compute_metrics_no_true_positives():
    """Edge case: no phishing in truth -> recall 0.0, ROC-AUC None."""
    m = compute_metrics([0, 0, 0], [0.9, 0.1, 0.2], threshold=0.5)
    assert m["recall"] == 0.0
    assert m["roc_auc"] is None
    assert m["false_positive_rate"] == pytest.approx(1 / 3)


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_roc_auc_known_values():
    """Perfect ranking -> 1.0; reversed -> 0.0; chance -> 0.5; one class -> None."""
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.8, 0.9, 0.1, 0.2]) == pytest.approx(0.0)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert roc_auc([1, 1, 1], [0.1, 0.9, 0.5]) is None


def test_fit_threshold_precision_first_meets_floor():
    """Threshold is chosen for max precision with recall >= floor."""
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.95, 0.9, 0.85, 0.6, 0.55, 0.4, 0.3, 0.2]
    fit = fit_threshold_precision_first(y_true, y_prob, recall_floor=0.75)
    assert fit["floor_met"] is True
    assert fit["recall"] >= 0.75
    # At threshold 0.6: precision 4/4 = 1.0 at recall 1.0 — nothing beats it.
    assert fit["precision"] == pytest.approx(1.0)
    assert fit["threshold"] == pytest.approx(0.6)


def test_fit_threshold_floor_not_met_is_reported():
    """When no threshold reaches the recall floor, floor_met is False.

    Note: with at least one true positive the floor is always reachable
    (threshold at/below the lowest score predicts everything positive,
    recall 1.0), so the unmet case only arises for a degenerate
    validation slice with no positives — recall is then 0.0 everywhere
    and that is reported, not hidden.
    """
    y_true = [0, 0]
    y_prob = [0.4, 0.3]
    fit = fit_threshold_precision_first(y_true, y_prob, recall_floor=0.9)
    assert fit["floor_met"] is False
    assert fit["recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# E2 — failure table, per-modality recall, ablation labelling.
# ---------------------------------------------------------------------------
def _toy_rows():
    return [
        {"filename": "a.eml", "label": 1, "category": "qr_body_image",
         "pred_d_weighted": 0, "score_a": 0.1, "score_b": 0.2, "score_c": 0.3,
         "score_d_weighted": 0.4, "score_d_max": 0.6, "threshold_d_weighted": 0.5,
         "has_ocr": True, "has_qr": True, "top_reasons": ["QR scored 0.20."]},
        {"filename": "b.eml", "label": 0, "category": "newsletter_logo_qr",
         "pred_d_weighted": 1, "score_a": 0.6, "score_b": 0.6, "score_c": 0.7,
         "score_d_weighted": 0.8, "score_d_max": 0.9, "threshold_d_weighted": 0.5,
         "has_ocr": True, "has_qr": True, "top_reasons": ["Benign QR present."]},
        {"filename": "c.eml", "label": 1, "category": "scanned_pdf",
         "pred_d_weighted": 1, "score_a": 0.2, "score_b": 0.8, "score_c": 0.9,
         "score_d_weighted": 0.9, "score_d_max": 0.9, "threshold_d_weighted": 0.5,
         "has_ocr": True, "has_qr": False, "top_reasons": []},
    ]


def test_failure_table_builder_only_errors():
    """Only misclassified rows appear, with true/predicted labels and scores."""
    failures = build_failure_table(_toy_rows())
    assert [f["filename"] for f in failures] == ["a.eml", "b.eml"]
    missed = failures[0]
    assert missed["true_label"] == 1 and missed["predicted_label"] == 0
    assert missed["score_d_weighted"] == pytest.approx(0.4)
    assert "missed phishing" in missed["reason"]
    false_alarm = failures[1]
    assert false_alarm["true_label"] == 0 and false_alarm["predicted_label"] == 1
    assert "false alarm" in false_alarm["reason"]


def test_failure_table_empty_when_all_correct():
    rows = [dict(r, pred_d_weighted=r["label"]) for r in _toy_rows()]
    assert build_failure_table(rows) == []


def test_per_modality_recall_counts_only_carriers():
    """Recall is computed over phishing emails carrying that modality."""
    rows = [
        {"label": 1, "has_body": True, "has_ocr": True, "has_pdf": False,
         "has_qr": True, "pred_d_weighted": 1},
        {"label": 1, "has_body": True, "has_ocr": True, "has_pdf": False,
         "has_qr": False, "pred_d_weighted": 0},
        {"label": 0, "has_body": True, "has_ocr": True, "has_pdf": False,
         "has_qr": True, "pred_d_weighted": 1},  # benign: never counted
        {"label": 1, "has_body": True, "has_ocr": False, "has_pdf": True,
         "has_qr": False, "pred_d_weighted": 1},
    ]
    out = per_modality_recall(rows)
    assert out["ocr"]["n_phishing_with_modality"] == 2
    assert out["ocr"]["recall"] == pytest.approx(0.5)
    assert out["pdf"]["n_phishing_with_modality"] == 1
    assert out["pdf"]["recall"] == pytest.approx(1.0)
    assert out["qr"]["n_phishing_with_modality"] == 1
    assert out["qr"]["recall"] == pytest.approx(1.0)


def test_ablation_labelling_all_present():
    """Ablations (a)-(d) are all labelled, in order, for the JSON output."""
    assert ABLATION_KEYS == ("a", "b", "c", "d_weighted", "d_max")
    assert set(ABLATION_LABELS) == set(ABLATION_KEYS)
    for key, label in ABLATION_LABELS.items():
        assert label.startswith(f"({key[0]})"), (key, label)


def test_precision_first_summary_names_best_precision():
    """Summary line leads with the best-precision ablation and its recall."""
    test_metrics = {
        "a": {"precision": 0.5, "recall": 0.9, "f1": 0.6,
              "false_positive_rate": 0.4, "accuracy": 0.6, "n": 10,
              "recall_floor": 0.8},
        "d_weighted": {"precision": 0.9, "recall": 0.7, "f1": 0.78,
                       "false_positive_rate": 0.05, "accuracy": 0.85, "n": 10,
                       "recall_floor": 0.8},
    }
    line = precision_first_summary(test_metrics)
    assert "Best precision" in line
    assert "(d) full fusion" in line
    assert "0.90" in line and "0.70" in line  # precision and recall both visible


def test_render_markdown_has_synthetic_warning_and_ablation_rows():
    """The .md report leads with the synthetic warning and lists (a)-(d)."""
    results = {
        "untrained_fallback": True,
        "precision_first_summary": "Best precision at recall floor 0.80: (d) full fusion — weighted.",
        "metadata": {"eval_dir": "data/multimodal_eval", "seed": 42, "n_total": 120,
                     "n_phishing": 60, "n_benign": 60, "n_validation": 24, "n_test": 96,
                     "fusion": "weighted", "checkpoint": "results/best_model.pt",
                     "models": ["roberta"], "recall_floor": 0.8,
                     "threshold_note": "fitted on validation only."},
        "test_metrics": {
            k: {"precision": 0.8, "recall": 0.7, "f1": 0.74, "accuracy": 0.75,
                "false_positive_rate": 0.1, "roc_auc": 0.85, "threshold": 0.5, "n": 96,
                "confusion_matrix": [[43, 5], [14, 34]]}
            for k in ABLATION_KEYS
        },
        "test_metrics_at_0_5": {},
        "per_modality_recall": {"ocr": {"n_phishing_with_modality": 10, "caught": 7, "recall": 0.7}},
        "baselines": {},
        "failures": [],
        "figures": ["confusion_matrix_weighted.png"],
    }
    md = render_markdown(results)
    assert "SYNTHETIC" in md
    assert "UNTRAINED MODEL" in md
    for label in ABLATION_LABELS.values():
        assert label in md
    assert "validation slice only" in md


# ---------------------------------------------------------------------------
# E2 — extraction helpers (pure, no model).
# ---------------------------------------------------------------------------
def test_modality_texts_and_filtered_extraction():
    """Modality columns split parts correctly; filtering keeps only kinds."""
    extraction = {
        "body_text": "hello body",
        "parts": [
            {"kind": "subject", "text": "Subject line", "filename": None},
            {"kind": "body_text", "text": "hello body", "filename": None},
            {"kind": "image", "text": "OCR words", "filename": "img.png"},
            {"kind": "pdf_text", "text": "invoice text", "filename": "inv.pdf",
             "qr_urls": ["https://billing.company.org/pay"]},
            {"kind": "qr", "text": "", "filename": "q.png",
             "qr_urls": ["http://secure-login-example.com/qr-verify"]},
        ],
        "links": [], "flags": [], "headers": {}, "skipped": [],
    }
    texts = modality_texts(extraction)
    assert "Subject line" in texts["body_text"]
    assert "OCR words" in texts["ocr_text"]
    assert "invoice text" in texts["pdf_text"]
    assert "http://secure-login-example.com/qr-verify" in texts["qr_text"]
    assert "[SUBJECT]" in texts["text"] and "[QR_URL]" in texts["text"]
    body_only = filtered_extraction(extraction, ("subject", "body_text"))
    assert {p["kind"] for p in body_only["parts"]} <= {"subject", "body_text"}
    assert len(body_only["parts"]) == 2


# ---------------------------------------------------------------------------
# E3 — faithfulness script wiring (static check: the random baseline and
# the untrained-fallback flag must be present in the script).
# ---------------------------------------------------------------------------
def test_faithfulness_script_has_random_baseline_and_fallback_flag():
    """run_multimodal_faithfulness.py calls evaluate_random_baseline and
    records untrained_fallback per row (Section A added the flag; E3 adds
    the baseline — this test pins both)."""
    source = (REPO_ROOT / "scripts" / "run_multimodal_faithfulness.py").read_text()
    assert "from phishing_xai.faithfulness import" in source
    assert "evaluate_random_baseline" in source
    assert "random_comprehensiveness_k5" in source
    assert "random_sufficiency_k5" in source
    assert '"untrained_fallback": untrained_fallback' in source
    assert "no_text_parts" in source  # empty-extraction guard
