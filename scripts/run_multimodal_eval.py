"""Multimodal evaluation (Section E): ablations (a)-(d) + cheap baselines.

Evaluates the Section D fusion pipeline on the labelled SYNTHETIC eval
set produced by ``scripts/make_multimodal_eval_set.py`` (default
``data/multimodal_eval/``) and writes ``multimodal_eval.json``,
``multimodal_eval.md`` and two plots into the output directory
(default ``results/multimodal_eval``).

Ablations (precision is the priority metric throughout — user priority
2026-11-19 — so every table leads with precision + FPR, with recall
reported alongside so the trade-off stays visible)::

    (a) text-body only            — RoBERTa on subject + body text only
    (b) text + OCR                — (a) plus image / scanned-PDF OCR text
    (c) text + OCR + QR + PDF     — (b) plus QR payloads + PDF text layer
    (d) full fusion               — (c) fused with URL risk + header flags
                                    via ``multimodal.fuse_email``;
                                    BOTH ``weighted`` (precision-first)
                                    and ``max`` (recall-heavy baseline)
                                    are reported for every email

Thresholds are fitted on the VALIDATION slice only (``labels.csv``
``split == "validation"``), choosing the threshold that maximises
phishing-class precision subject to a recall floor (default 0.80).
Thresholds are never fitted on test. When no validation rows are
available (e.g. a tiny ``--max_samples`` run), the placeholder 0.5 is
used and the report says so explicitly. The placeholder-0.5 metrics
are also reported alongside the fitted-threshold metrics.

Cheap baselines (Section H hook, ``--models``): ``tfidf_lr``, ``tfidf_svm``
and ``bilstm`` are fitted on the REAL-data train split (``results/splits.json``,
identical splits/cleaning as Section H — vocab/TF-IDF fit on train
only) and evaluated on the SAME extractor output as RoBERTa
(subject+body+OCR+PDF+QR combined text), via
``baselines.evaluate_on_extractor_output`` and the per-modality
``baselines.evaluate_multimodal`` table. If the real dataset or the
splits manifest is missing, the baselines are reported as NOT RUN YET
with the reason — never fitted on the synthetic eval set itself (that
would be train/test leakage: the eval set has no train split).

RoBERTa (``--models roberta``, the default) scores parts and contexts
through the existing ``run_multimodal_predict.score_text`` /
``load_trained_roberta`` path — no retraining, no model change.

Untrained fallback: when no trained checkpoint is present the script
refuses cleanly UNLESS ``--allow_pretrained_fallback`` is given, in
which case it runs end-to-end on base ``roberta-base`` with a LOUD
warning, ``"untrained_fallback": true`` in the JSON, and every number
labelled NOT detection evidence (the .md header says so in bold).

SYNTHETIC-DATA WARNING: the eval set is generated from a small set of
templates with simulated OCR stressors. It measures pipeline behaviour
on those templates only. It does not generalise to real phishing, and
no number in the outputs should be quoted as real-world detection
performance. This warning is written into the JSON metadata and the
top of the .md report.

Kaggle path auto-detection: when ``/kaggle`` exists, the checkpoint
defaults to the first ``/kaggle/input/*/best_model.pt`` (else
``/kaggle/working/results/best_model.pt``) and outputs default to
``/kaggle/working/results/multimodal_eval``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.baselines import RobertaScorer, load_scorer
from phishing_xai.config import FusionConfig, ModelConfig, TrainConfig
from phishing_xai.extract import extract_eml
from phishing_xai.multimodal import build_concatenated_context, fuse_email, part_id, score_texts
from phishing_xai.utils import get_logger, set_seed

logger = get_logger("phishing_xai.run_multimodal_eval")

#: Default decision threshold. PLACEHOLDER (0.5) — fitted thresholds come
#: from the validation slice; see the module docstring.
DEFAULT_THRESHOLD: float = FusionConfig.threshold
#: Recall floor for precision-first threshold fitting on validation.
DEFAULT_RECALL_FLOOR: float = 0.80

SYNTHETIC_WARNING = (
    "SYNTHETIC evaluation set: generated images/PDFs from a small set of "
    "templates with simulated OCR stressors (font/size variation, low "
    "contrast, rotation, noise, JPEG compression). No real-world adversary "
    "variation and no real emails. Results measure pipeline behaviour on "
    "these templates only and do NOT generalise to real phishing."
)
UNTRAINED_WARNING = (
    "UNTRAINED MODEL — results are NOT detection evidence: the fine-tuned "
    "checkpoint was missing, so every score below comes from untrained base "
    "roberta-base (pipeline-demonstration output only)."
)

# ---------------------------------------------------------------------------
# Ablation definitions: key -> (label, part kinds whose text feeds it).
# (d) is the fusion step and is handled separately (fuse_email).
# ---------------------------------------------------------------------------
ABLATION_KINDS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "a": ("(a) text-body only", ("subject", "body_text", "html_text")),
    "b": ("(b) text + OCR",
          ("subject", "body_text", "html_text", "image", "pdf_ocr", "pdf_image")),
    "c": ("(c) text + OCR + QR + PDF",
          ("subject", "body_text", "html_text", "image", "pdf_ocr", "pdf_image",
           "qr", "pdf_text")),
}
ABLATION_KEYS: Tuple[str, ...] = ("a", "b", "c", "d_weighted", "d_max")
ABLATION_LABELS: Dict[str, str] = {
    "a": "(a) text-body only",
    "b": "(b) text + OCR",
    "c": "(c) text + OCR + QR + PDF",
    "d_weighted": "(d) full fusion — weighted",
    "d_max": "(d) full fusion — max",
}
#: Part kinds feeding each modality column (Section H hook).
_MODALITY_KINDS: Dict[str, Tuple[str, ...]] = {
    "body_text": ("subject", "body_text", "html_text"),
    "ocr_text": ("image", "pdf_ocr", "pdf_image"),
    "pdf_text": ("pdf_text",),
    "qr_text": ("qr",),
}


# ---------------------------------------------------------------------------
# Pure metric helpers (no model, no GPU — unit-tested in
# tests/test_multimodal_eval.py).
# ---------------------------------------------------------------------------
def roc_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> Optional[float]:
    """ROC-AUC via the Mann-Whitney rank statistic.

    Returns ``None`` when only one class is present (AUC undefined).
    Ties get average ranks, matching the standard definition.
    """
    y_true = [int(v) for v in y_true]
    y_prob = [float(v) for v in y_prob]
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(y_prob)), key=lambda i: y_prob[i])
    ranks = [0.0] * len(y_prob)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_prob[order[j + 1]] == y_prob[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, Any]:
    """Precision-led metric set for phishing (label 1).

    Returns accuracy, precision, recall, F1, false-positive rate, the
    confusion matrix as ``[[tn, fp], [fn, tp]]``, ROC-AUC (``None`` when
    only one class is present) and ``n``. Edge cases are handled, not
    hidden: zero predicted positives gives precision 0.0, zero true
    positives gives recall 0.0, and an empty input raises ``ValueError``.
    """
    y_true = [int(v) for v in y_true]
    y_prob = [float(v) for v in y_prob]
    if not y_true:
        raise ValueError("compute_metrics: empty input")
    if len(y_true) != len(y_prob):
        raise ValueError("compute_metrics: y_true and y_prob lengths differ")
    tp = fp = tn = fn = 0
    for y, p in zip(y_true, y_prob):
        pred = 1 if p >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_auc": roc_auc(y_true, y_prob),
        "n": len(y_true),
        "threshold": float(threshold),
    }


def fit_threshold_precision_first(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    recall_floor: float = DEFAULT_RECALL_FLOOR,
) -> Dict[str, Any]:
    """Fit a decision threshold on VALIDATION data only.

    Scans candidate thresholds (the observed scores plus a 0.05 grid)
    and picks the one maximising phishing-class precision subject to
    recall >= ``recall_floor`` (ties: higher recall, then higher F1,
    then the lower threshold). When no candidate meets the floor,
    ``floor_met`` is False and the threshold maximising recall (then
    precision) is returned instead — the shortfall is reported, never
    hidden. Returns ``{"threshold", "precision", "recall", "f1",
    "floor_met", "recall_floor", "n"}``.
    """
    y_true = [int(v) for v in y_true]
    y_prob = [float(v) for v in y_prob]
    if not y_true:
        raise ValueError("fit_threshold_precision_first: empty input")
    candidates = sorted({round(float(p), 4) for p in y_prob} | {round(0.05 * i, 2) for i in range(1, 20)})
    scored = []
    for thr in candidates:
        m = compute_metrics(y_true, y_prob, threshold=thr)
        scored.append((thr, m))
    meeting = [(thr, m) for thr, m in scored if m["recall"] >= recall_floor]
    if meeting:
        thr, m = min(meeting, key=lambda tm: (-tm[1]["precision"], -tm[1]["recall"], -tm[1]["f1"], tm[0]))
        floor_met = True
    else:
        thr, m = min(scored, key=lambda tm: (-tm[1]["recall"], -tm[1]["precision"], -tm[1]["f1"], tm[0]))
        floor_met = False
    return {
        "threshold": float(thr),
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "floor_met": floor_met,
        "recall_floor": float(recall_floor),
        "n": len(y_true),
    }


def filtered_extraction(
    extraction: Dict[str, Any], kinds: Tuple[str, ...]
) -> Dict[str, Any]:
    """Copy of ``extraction`` keeping only parts whose kind is in ``kinds``.

    Used to build the per-ablation concatenated contexts (a)-(c) with the
    same builder Section D uses, so an ablation context is exactly the
    subset of the full context its modalities contribute.
    """
    keep = set(kinds)
    out = dict(extraction)
    out["parts"] = [p for p in extraction.get("parts") or [] if p.get("kind") in keep]
    return out


def modality_texts(extraction: Dict[str, Any]) -> Dict[str, str]:
    """Per-modality text columns from an extraction (Section H hook).

    ``body_text`` = subject + body + HTML text; ``ocr_text`` = image /
    scanned-PDF / embedded-image OCR; ``pdf_text`` = PDF text layer;
    ``qr_text`` = QR payloads (``qr_urls`` plus any QR part text);
    ``text`` = the full concatenated context (same builder as Section D).
    Empty modalities are empty strings, never missing keys.
    """
    cols: Dict[str, List[str]] = {k: [] for k in _MODALITY_KINDS}
    for part in extraction.get("parts") or []:
        text = (part.get("text") or "").strip()
        for col, kinds in _MODALITY_KINDS.items():
            if part.get("kind") in kinds and text:
                cols[col].append(text)
        if part.get("kind") == "qr":
            for url in part.get("qr_urls") or []:
                if url and url not in cols["qr_text"]:
                    cols["qr_text"].append(str(url))
    body_top = (extraction.get("body_text") or "").strip()
    if body_top and body_top not in cols["body_text"]:
        cols["body_text"].append(body_top)
    out = {col: " ".join(texts) for col, texts in cols.items()}
    out["text"] = build_concatenated_context(extraction)
    return out


def per_modality_recall(
    rows: Sequence[Dict[str, Any]], *, pred_key: str = "pred_d_weighted"
) -> Dict[str, Dict[str, Any]]:
    """Recall on the phishing emails that actually carry each modality.

    A phishing email counts for a modality when that modality's text is
    non-empty in the extraction (``has_<modality>``). Recall is the
    fraction of those emails the (weighted-fusion) verdict caught. This
    is a per-modality recall, not a modality-ablation metric: a caught
    email may have been caught on other evidence too.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for modality in ("body", "ocr", "pdf", "qr"):
        subset = [r for r in rows if r.get("label") == 1 and r.get(f"has_{modality}")]
        caught = sum(1 for r in subset if r.get(pred_key) == 1)
        out[modality] = {
            "n_phishing_with_modality": len(subset),
            "caught": caught,
            "recall": (caught / len(subset)) if subset else None,
        }
    return out


def build_failure_table(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Misclassified emails, with the evidence that was (not) there.

    One row per error: filename, true label, predicted label (weighted
    fusion), the (a)-(d) scores, category, and a short reason built from
    the fusion's top reasons plus the decisive score — e.g. a false
    negative whose weighted score reached the threshold but lacked
    corroboration, or whose OCR text came back empty.
    """
    failures: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("pred_d_weighted") == r.get("label"):
            continue
        reasons = list(r.get("top_reasons") or [])[:2]
        if r.get("label") == 1:
            cause = "missed phishing"
            if not r.get("has_ocr") and r.get("category") in (
                "image_only_phishing", "scanned_pdf",
            ):
                cause += "; OCR text empty for an image-led email"
            elif r.get("score_d_weighted", 0.0) >= r.get("threshold_d_weighted", 0.5):
                cause += "; fused score reached threshold but lacked corroboration"
            else:
                cause += f"; context P(phish)={r.get('score_c', 0.0):.2f}"
        else:
            cause = "false alarm on benign mail"
            if r.get("has_qr"):
                cause += "; benign QR/URL present"
            cause += f"; context P(phish)={r.get('score_c', 0.0):.2f}"
        failures.append({
            "filename": r.get("filename"),
            "category": r.get("category"),
            "true_label": int(r.get("label", 0)),
            "predicted_label": int(r.get("pred_d_weighted", 0)),
            "score_a": round(float(r.get("score_a", 0.0)), 4),
            "score_b": round(float(r.get("score_b", 0.0)), 4),
            "score_c": round(float(r.get("score_c", 0.0)), 4),
            "score_d_weighted": round(float(r.get("score_d_weighted", 0.0)), 4),
            "score_d_max": round(float(r.get("score_d_max", 0.0)), 4),
            "reason": cause + ("; " + "; ".join(reasons) if reasons else ""),
        })
    return failures


def precision_first_summary(
    test_metrics: Dict[str, Dict[str, Any]],
) -> str:
    """One precision-first summary line over the test slice.

    Names the best-precision ablation at the fitted thresholds and the
    recall it paid for, e.g. ``"Best precision at recall floor 0.80: (d)
    full fusion — weighted, precision 0.92 at recall 0.75 (FPR 0.04)."``.
    When nothing was run, the line says so instead of inventing one.
    """
    if not test_metrics:
        return "No ablation was run, so no precision summary is available."
    best_key = max(
        test_metrics,
        key=lambda k: (
            test_metrics[k]["precision"],
            test_metrics[k]["recall"],
            -test_metrics[k]["false_positive_rate"],
        ),
    )
    best = test_metrics[best_key]
    floor = best.get("recall_floor", DEFAULT_RECALL_FLOOR)
    return (
        f"Best precision at recall floor {floor:.2f}: "
        f"{ABLATION_LABELS.get(best_key, best_key)}, precision "
        f"{best['precision']:.2f} at recall {best['recall']:.2f} "
        f"(F1 {best['f1']:.2f}, FPR {best['false_positive_rate']:.2f}, "
        f"accuracy {best['accuracy']:.2f}, n={best['n']})."
    )


# ---------------------------------------------------------------------------
# Paths / checkpoint resolution (mirrors run_multimodal_predict.py).
# ---------------------------------------------------------------------------
def resolve_default_checkpoint() -> Path:
    """Default checkpoint path, with Kaggle auto-detection."""
    if os.path.exists("/kaggle"):
        kaggle_inputs = sorted(Path("/kaggle/input").glob("*/best_model.pt"))
        if kaggle_inputs:
            return kaggle_inputs[0]
        return Path("/kaggle/working/results/best_model.pt")
    return REPO_ROOT / "results" / "best_model.pt"


def resolve_default_output_dir() -> Path:
    """Default output directory, with Kaggle auto-detection."""
    if os.path.exists("/kaggle"):
        return Path("/kaggle/working/results/multimodal_eval")
    return REPO_ROOT / "results" / "multimodal_eval"


def resolve_default_eval_dir() -> Path:
    """Default eval-set directory (Kaggle keeps the repo-relative one)."""
    return REPO_ROOT / "data" / "multimodal_eval"


# ---------------------------------------------------------------------------
# Scorer-based scoring (6B.1): any TextScorer drives the (a)-(d) ablations.
# ---------------------------------------------------------------------------
def score_email(
    extraction: Dict[str, Any],
    scorer: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Score one extracted email under every ablation (a)-(d) with any scorer.

    Each non-empty part is scored once (``scorer.predict_proba``); the (a)-(c)
    scores are the scorer's P(phishing) on the filtered concatenated
    contexts; (d) fuses the part scores + full context score + URL/flag
    evidence with ``fuse_email`` for both fusions. Returns the scores,
    the fused verdict dicts and the modality-presence flags.
    """
    part_scores: Dict[str, float] = {}
    for part in extraction.get("parts") or []:
        text = (part.get("text") or "").strip()
        if not text:
            continue
        part_scores[part_id(part)] = score_texts(scorer, [text])[0]

    ablation_scores: Dict[str, float] = {}
    for key, (_label, kinds) in ABLATION_KINDS.items():
        context = build_concatenated_context(
            filtered_extraction(extraction, kinds),
            scorer=scorer, max_length=ModelConfig.max_length,
        )
        ablation_scores[key] = score_texts(scorer, [context])[0] if context.strip() else 0.0

    full_context = build_concatenated_context(
        extraction, scorer=scorer, max_length=ModelConfig.max_length
    )
    context_prob = score_texts(scorer, [full_context])[0] if full_context.strip() else 0.0
    fused_weighted = fuse_email(
        extraction, part_scores, fusion="weighted",
        concatenated_score=context_prob, threshold=threshold,
        scorer=scorer, max_length=ModelConfig.max_length,
    )
    fused_max = fuse_email(
        extraction, part_scores, fusion="max",
        concatenated_score=context_prob, threshold=threshold,
        scorer=scorer, max_length=ModelConfig.max_length,
    )
    texts = modality_texts(extraction)
    return {
        "score_a": ablation_scores["a"],
        "score_b": ablation_scores["b"],
        "score_c": ablation_scores["c"],
        "context_prob_phishing": float(context_prob),
        "score_d_weighted": fused_weighted["final_score"],
        "score_d_max": fused_max["final_score"],
        "fusion_weighted": fused_weighted,
        "fusion_max": fused_max,
        "top_reasons": fused_weighted.get("top_reasons") or [],
        "has_body": bool(texts["body_text"].strip()),
        "has_ocr": bool(texts["ocr_text"].strip()),
        "has_pdf": bool(texts["pdf_text"].strip()),
        "has_qr": bool(texts["qr_text"].strip()),
        "modality_texts": texts,
    }


def score_email_roberta(
    extraction: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Score one extracted email under every ablation (a)-(d) with RoBERTa.

    Thin wrapper over :func:`score_email` with a ``RobertaScorer`` around
    the ``(model, tokenizer, device)`` triple, kept so existing callers
    and tests do not break. Behaviour is identical to the previous
    implementation (same ``score_text`` calls, same truncation).
    """
    scorer = RobertaScorer(model, tokenizer, device, max_length=ModelConfig.max_length)
    return score_email(extraction, scorer, threshold=threshold)


# ---------------------------------------------------------------------------
# Cheap baselines on the same extractor output (Section H hook).
# ---------------------------------------------------------------------------
def run_cheap_baselines(
    rows: List[Dict[str, Any]],
    models: Sequence[str],
    *,
    bilstm_epochs: int = 3,
) -> Dict[str, Any]:
    """Fit TF-IDF+LR / BiLSTM on the REAL-data train split and evaluate
    them on the synthetic eval rows' extractor text (H6 hook).

    Returns ``{"tfidf_lr": {...} | {"status": "NOT RUN YET", "reason": ...},
    ...}`` per requested model. The baselines are never fitted on the
    synthetic set: fitting on the evaluated set would be leakage, and
    the set has no train split. When the real CSV or the splits manifest
    is missing, every requested baseline is NOT RUN YET with the reason.
    """
    from phishing_xai import baselines as bl

    out: Dict[str, Any] = {}
    wanted = [m for m in models if m in ("tfidf_lr", "tfidf_svm", "bilstm")]
    if not wanted:
        return out
    try:
        splits = bl.load_splits(
            data_path=REPO_ROOT / "data" / "Phishing_Email.csv",
            splits_path=REPO_ROOT / "results" / "splits.json",
        )
    except (FileNotFoundError, ValueError) as exc:
        for name in wanted:
            out[name] = {"status": "NOT RUN YET", "reason": str(exc)}
        return out
    train = splits["train"]
    train_texts = train["clean_text"].astype(str).tolist()
    train_labels = train["label"].astype(int).tolist()
    val = splits["val"]
    val_texts = val["clean_text"].astype(str).tolist()
    val_labels = val["label"].astype(int).tolist()

    fitted: Dict[str, Any] = {}
    if "tfidf_lr" in wanted:
        try:
            fitted["tfidf_lr"] = bl.build_tfidf_lr(train_texts, train_labels)
        except Exception as exc:  # noqa: BLE001 — report, do not crash the eval
            out["tfidf_lr"] = {"status": "NOT RUN YET", "reason": f"fit failed: {exc}"}
    if "tfidf_svm" in wanted:
        try:
            fitted["tfidf_svm"] = bl.build_tfidf_svm(train_texts, train_labels)
        except Exception as exc:  # noqa: BLE001 — report, do not crash the eval
            out["tfidf_svm"] = {"status": "NOT RUN YET", "reason": f"fit failed: {exc}"}
    if "bilstm" in wanted:
        try:
            config = bl.TorchTrainConfig(epochs=int(bilstm_epochs))
            fitted["bilstm"], _info = bl.build_torch_baseline(
                "bilstm", train_texts, train_labels, val_texts, val_labels,
                config=config, seed=TrainConfig.seed,
            )
        except Exception as exc:  # noqa: BLE001 — report, do not crash the eval
            out["bilstm"] = {"status": "NOT RUN YET", "reason": f"fit failed: {exc}"}

    if fitted:
        hook_rows = [
            {"text": r["modality_texts"]["text"], "label": int(r["label"])} for r in rows
        ]
        for name, model in fitted.items():
            try:
                result = bl.evaluate_on_extractor_output(model, hook_rows)
                m = result["metrics"]
                out[name] = {
                    "status": "ok",
                    "n": result["n"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "accuracy": m["accuracy"],
                    "roc_auc": m["roc_auc"],
                    "false_positive_rate": m["false_positive_rate"],
                    "confusion_matrix": m["confusion_matrix"],
                    "trained_on": "real-data train split (results/splits.json), "
                                  "evaluated on synthetic extractor text",
                }
            except Exception as exc:  # noqa: BLE001
                out[name] = {"status": "NOT RUN YET", "reason": f"eval failed: {exc}"}
        try:
            import pandas as pd

            eval_df = pd.DataFrame([
                {
                    "text": r["modality_texts"]["text"],
                    "label": int(r["label"]),
                    "body_text": r["modality_texts"]["body_text"],
                    "ocr_text": r["modality_texts"]["ocr_text"],
                    "pdf_text": r["modality_texts"]["pdf_text"],
                    "qr_text": r["modality_texts"]["qr_text"],
                }
                for r in rows
            ])
            table = bl.evaluate_multimodal(fitted, eval_df)
            out["modality_ablation_per_model"] = table.to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001
            out["modality_ablation_per_model"] = {
                "status": "NOT RUN YET", "reason": f"ablation table failed: {exc}"
            }
    return out


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def _plot_confusion_matrix(metrics: Dict[str, Any], path: Path, title: str) -> Optional[Path]:
    """Confusion-matrix figure (test slice) for the primary fusion."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Skipping confusion-matrix plot (matplotlib missing: {exc}).")
        return None
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow([[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]], cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred Safe", "pred Phishing"])
    ax.set_yticklabels(["true Safe", "true Phishing"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=16)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_ablation_bars(
    test_metrics: Dict[str, Dict[str, Any]], path: Path
) -> Optional[Path]:
    """Grouped precision / recall / FPR bar chart, one group per ablation."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        logger.warning(f"Skipping ablation bar chart (matplotlib missing: {exc}).")
        return None
    keys = [k for k in ABLATION_KEYS if k in test_metrics]
    labels = ["(a)\nbody", "(b)\n+OCR", "(c)\n+QR+PDF", "(d)\nweighted", "(d)\nmax"]
    labels = labels[: len(keys)]
    precision = [test_metrics[k]["precision"] for k in keys]
    recall = [test_metrics[k]["recall"] for k in keys]
    fpr = [test_metrics[k]["false_positive_rate"] for k in keys]
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, precision, width, label="precision")
    ax.bar(x, recall, width, label="recall")
    ax.bar(x + width, fpr, width, label="FPR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Precision / recall / FPR by ablation (test slice, fitted thresholds)")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Report rendering.
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(results: Dict[str, Any]) -> str:
    """Render the .md report from the results dict (precision-led tables)."""
    lines: List[str] = []
    lines.append("# Multimodal evaluation (Section E)")
    lines.append("")
    lines.append("> " + SYNTHETIC_WARNING)
    lines.append("")
    if results.get("untrained_fallback"):
        lines.append("> **" + UNTRAINED_WARNING + "**")
        lines.append("")
    lines.append(results.get("precision_first_summary", ""))
    lines.append("")
    meta = results.get("metadata", {})
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Eval set: `{meta.get('eval_dir', '')}` (seed {meta.get('seed')}, "
                 f"{meta.get('n_total')} emails: {meta.get('n_phishing')} phishing / "
                 f"{meta.get('n_benign')} benign; validation {meta.get('n_validation')} / "
                 f"test {meta.get('n_test')})")
    lines.append(f"- Thresholds: fitted on the validation slice only "
                 f"(precision-first, recall floor {meta.get('recall_floor')}); "
                 f"placeholder 0.5 metrics are shown alongside. Thresholds were "
                 f"never fitted on test. {meta.get('threshold_note', '')}")
    lines.append(f"- Primary fusion: `{meta.get('fusion')}`; checkpoint: "
                 f"`{meta.get('checkpoint')}`; untrained fallback: "
                 f"{results.get('untrained_fallback')}")
    lines.append(f"- Models requested: {', '.join(meta.get('models', []))}")
    lines.append("")
    lines.append("## Ablation table (test slice, fitted thresholds) — precision first")
    lines.append("")
    lines.append("| ablation | precision | FPR | recall | F1 | accuracy | ROC-AUC | threshold | n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in ABLATION_KEYS:
        m = results.get("test_metrics", {}).get(key)
        if not m:
            continue
        lines.append(
            f"| {ABLATION_LABELS[key]} | {m['precision']:.4f} | "
            f"{m['false_positive_rate']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['accuracy']:.4f} | {_fmt(m['roc_auc'])} | {m['threshold']:.2f} | {m['n']} |"
        )
    lines.append("")
    lines.append("Same table at the placeholder threshold 0.5 (for reference only):")
    lines.append("")
    lines.append("| ablation | precision | FPR | recall | F1 | accuracy |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in ABLATION_KEYS:
        m = results.get("test_metrics_at_0_5", {}).get(key)
        if not m:
            continue
        lines.append(
            f"| {ABLATION_LABELS[key]} | {m['precision']:.4f} | "
            f"{m['false_positive_rate']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['accuracy']:.4f} |"
        )
    lines.append("")
    lines.append("## Confusion matrices (test slice, fitted thresholds)")
    lines.append("")
    for key in ("d_weighted", "d_max"):
        m = results.get("test_metrics", {}).get(key)
        if m:
            lines.append(f"- {ABLATION_LABELS[key]}: `[[tn, fp], [fn, tp]]` = {m['confusion_matrix']}")
    lines.append("")
    lines.append("## Per-modality recall (phishing emails carrying that modality, weighted fusion)")
    lines.append("")
    lines.append("| modality | phishing emails with it | caught | recall |")
    lines.append("|---|---:|---:|---:|")
    for modality, vals in results.get("per_modality_recall", {}).items():
        lines.append(
            f"| {modality} | {vals['n_phishing_with_modality']} | {vals['caught']} | "
            f"{_fmt(vals['recall'])} |"
        )
    lines.append("")
    baselines = results.get("baselines", {})
    if baselines:
        lines.append("## Cheap baselines on the same extractor output (Section H hook)")
        lines.append("")
        lines.append("Fitted on the real-data train split; evaluated on the synthetic "
                     "eval set's combined extractor text (subject+body+OCR+PDF+QR).")
        lines.append("")
        lines.append("| model | precision | FPR | recall | F1 | accuracy | ROC-AUC | status |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for name, entry in baselines.items():
            if name == "modality_ablation_per_model" or not isinstance(entry, dict):
                continue
            if entry.get("status") == "ok":
                lines.append(
                    f"| {name} | {entry['precision']:.4f} | {entry['false_positive_rate']:.4f} | "
                    f"{entry['recall']:.4f} | {entry['f1']:.4f} | {entry['accuracy']:.4f} | "
                    f"{_fmt(entry['roc_auc'])} | ok |"
                )
            else:
                lines.append(f"| {name} | — | — | — | — | — | — | "
                             f"NOT RUN YET ({entry.get('reason', '')}) |")
        lines.append("")
        ablation_table = baselines.get("modality_ablation_per_model")
        if isinstance(ablation_table, list):
            lines.append("Per-model modality ablation (F1 on that modality's text only):")
            lines.append("")
            lines.append("| model | modality | F1 | accuracy | status |")
            lines.append("|---|---|---:|---:|---|")
            for row in ablation_table:
                lines.append(
                    f"| {row['model']} | {row['modality']} | {_fmt(row.get('f1'))} | "
                    f"{_fmt(row.get('accuracy'))} | {row.get('status')} |"
                )
            lines.append("")
    failures = results.get("failures", [])
    lines.append(f"## Failures ({len(failures)} misclassified on the full set)")
    lines.append("")
    if failures:
        lines.append("| filename | category | true | predicted | scores a/b/c/d-w/d-max | reason |")
        lines.append("|---|---|---|---|---|---|")
        for f in failures:
            scores = (f"{f['score_a']:.2f}/{f['score_b']:.2f}/{f['score_c']:.2f}/"
                      f"{f['score_d_weighted']:.2f}/{f['score_d_max']:.2f}")
            lines.append(
                f"| `{f['filename']}` | {f['category']} | {f['true_label']} | "
                f"{f['predicted_label']} | {scores} | {f['reason']} |"
            )
    else:
        lines.append("No misclassifications on the evaluated set.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for fig in results.get("figures", []):
        lines.append(f"- `{fig}`")
    lines.append("")
    lines.append("## What this does and does not show")
    lines.append("")
    lines.append("- The set is synthetic (see the warning at the top): template "
                 "images/PDFs, simulated OCR stressors, no real adversary "
                 "variation. Treat every number as a pipeline check on these "
                 "templates, not as real-world detection performance.")
    lines.append("- Thresholds were fitted on 24 validation emails only; with "
                 "so few points the fitted threshold is coarse and should be "
                 "re-fitted on a larger real validation set before any "
                 "deployment claim.")
    lines.append("- The 0.5 threshold is a placeholder; fitted-threshold rows "
                 "are the reported ones.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main evaluation driver.
# ---------------------------------------------------------------------------
def load_labels(labels_path: Path) -> List[Dict[str, Any]]:
    """Read ``labels.csv`` (filename, label, category, split)."""
    with open(labels_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"filename", "label", "category"}
    if not required.issubset(set(rows[0].keys()) if rows else set()):
        raise ValueError(f"labels.csv missing required columns {sorted(required)}: {labels_path}")
    for row in rows:
        row["label"] = int(row["label"])
        row.setdefault("split", "test")
    return rows


def evaluate(
    eval_dir: Path,
    labels: List[Dict[str, Any]],
    scorer: Any,
    *,
    recall_floor: float = DEFAULT_RECALL_FLOOR,
    max_samples: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score every labelled email and fit thresholds on validation only.

    ``scorer`` is a :class:`TextScorer` (RoBERTa, TF-IDF+LR or BiLSTM).
    Returns ``(per-email rows, summary)`` where the summary holds the
    fitted thresholds (one per ablation score), the test-slice metrics at
    the fitted thresholds and at 0.5, per-modality recall and the failure
    table. Threshold fitting uses validation rows only; test rows are
    never used to choose a threshold.
    """
    rows: List[Dict[str, Any]] = []
    for entry in labels[: max_samples] if max_samples is not None else labels:
        eml_path = eval_dir / entry["filename"]
        if not eml_path.exists():
            logger.warning(f"Eval file missing, skipping: {eml_path}")
            continue
        extraction = extract_eml(eml_path)
        scored = score_email(extraction, scorer)
        rows.append({
            "filename": entry["filename"],
            "label": int(entry["label"]),
            "category": entry["category"],
            "split": entry.get("split", "test"),
            **{k: v for k, v in scored.items() if k != "modality_texts"},
            "modality_texts": scored["modality_texts"],
        })
    if not rows:
        raise ValueError("No eval emails could be scored (empty set or all files missing).")

    score_keys = {
        "a": "score_a", "b": "score_b", "c": "score_c",
        "d_weighted": "score_d_weighted", "d_max": "score_d_max",
    }
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]
    fitted: Dict[str, Dict[str, Any]] = {}
    for key, skey in score_keys.items():
        if val_rows:
            fitted[key] = fit_threshold_precision_first(
                [r["label"] for r in val_rows],
                [r[skey] for r in val_rows],
                recall_floor=recall_floor,
            )
        else:
            fitted[key] = {
                "threshold": DEFAULT_THRESHOLD, "precision": None, "recall": None,
                "f1": None, "floor_met": False, "recall_floor": recall_floor,
                "n": 0, "note": "no validation rows; placeholder 0.5 used",
            }
    for key, skey in score_keys.items():
        thr = float(fitted[key]["threshold"])
        for r in rows:
            r[f"pred_{key}"] = 1 if r[skey] >= thr else 0

    def _metrics(subset: List[Dict[str, Any]], key: str, thr: float) -> Dict[str, Any]:
        m = compute_metrics([r["label"] for r in subset],
                            [r[score_keys[key]] for r in subset], threshold=thr)
        m["recall_floor"] = recall_floor
        return m

    test_metrics: Dict[str, Dict[str, Any]] = {}
    test_metrics_at_0_5: Dict[str, Dict[str, Any]] = {}
    for key in ABLATION_KEYS:
        test_metrics[key] = _metrics(test_rows, key, float(fitted[key]["threshold"]))
        test_metrics_at_0_5[key] = _metrics(test_rows, key, DEFAULT_THRESHOLD)
    summary = {
        "fitted_thresholds": fitted,
        "test_metrics": test_metrics,
        "test_metrics_at_0_5": test_metrics_at_0_5,
        "per_modality_recall": per_modality_recall(test_rows),
        "failures": build_failure_table(rows),
        "n_scored": len(rows),
        "n_validation": len(val_rows),
        "n_test": len(test_rows),
    }
    return rows, summary


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate the multimodal fusion pipeline on the synthetic eval set (Section E)."
    )
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Directory with .eml files + labels.csv (default: data/multimodal_eval)")
    parser.add_argument("--labels", type=str, default=None,
                        help="labels.csv path (default: <eval_dir>/labels.csv)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Trained checkpoint (.pt). For --model roberta: the RoBERTa checkpoint. For --model distilbert: a saved DistilBERT artefact (best_distilbert.pt, best-validation seed) — a missing/wrong-kind checkpoint raises, never refits. For --model tfidf_lr/bilstm: a saved baseline artefact (best_<model>.pt from run_baselines.py) is loaded instead of fitting; a RoBERTa .pt or a missing file means fit on the real train split (default: Kaggle auto-detect, else results/best_model.pt)")
    parser.add_argument("--model", choices=["roberta", "distilbert", "tfidf_lr", "tfidf_svm", "bilstm", "rule_based"], default="roberta",
                        help="Scorer model driving the (a)-(d) ablations (default: roberta). distilbert loads a saved best_distilbert.pt artefact (no refit, no untrained fallback); tfidf_lr/tfidf_svm/bilstm fit on the real train split unless --checkpoint points at a saved baseline artefact; rule_based needs no model at all (worst static URL-risk score in the text, NOT a model probability); no checkpoint needed for any of the three.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: Kaggle /kaggle/working/results/multimodal_eval, else results/multimodal_eval)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of labelled emails to score (default: all)")
    parser.add_argument("--debug", action="store_true", help="Verbose (DEBUG) logging")
    parser.add_argument("--fusion", choices=["weighted", "max"], default="weighted",
                        help="Primary fusion for the headline verdict (default: weighted; both are always reported)")
    parser.add_argument("--models", nargs="+", choices=["tfidf_lr", "tfidf_svm", "bilstm", "roberta"],
                        default=["roberta"],
                        help="Cheap Section H baselines for the side table only (default: roberta, i.e. none); the (a)-(d) ablations are driven by --model, not by this flag")
    parser.add_argument("--allow_pretrained_fallback", action="store_true",
                        help="Allow running with base roberta-base when the checkpoint is missing (loud warning; NOT detection evidence)")
    parser.add_argument("--recall_floor", type=float, default=DEFAULT_RECALL_FLOOR,
                        help="Recall floor for precision-first threshold fitting on validation (default: 0.80)")
    parser.add_argument("--bilstm_epochs", type=int, default=3,
                        help="Epochs for the BiLSTM baseline when --models includes bilstm (default: 3)")
    args = parser.parse_args(argv)

    if args.debug:
        import logging

        logging.getLogger("phishing_xai").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    set_seed(TrainConfig.seed)
    eval_dir = Path(args.eval_dir) if args.eval_dir else resolve_default_eval_dir()
    labels_path = Path(args.labels) if args.labels else eval_dir / "labels.csv"
    out_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_dir()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_default_checkpoint()
    if not labels_path.exists():
        print(f"ERROR: labels.csv not found: {labels_path}\n"
              "Generate the eval set first: python scripts/make_multimodal_eval_set.py",
              file=sys.stderr)
        return 2
    labels = load_labels(labels_path)
    if args.max_samples is not None:
        labels = labels[: max(0, args.max_samples)]

    untrained_fallback = False
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    if args.model == "roberta":
        if not ckpt_path.exists() and not args.allow_pretrained_fallback:
            print(
                f"ERROR: Trained checkpoint is not present locally! Missing file: {ckpt_path.as_posix()}\n"
                "Per project rules, do NOT retrain locally. Pass --allow_pretrained_fallback "
                "to run the pipeline smoke test on untrained base roberta-base instead "
                "(numbers will be labelled NOT detection evidence).",
                file=sys.stderr,
            )
            return 2
        from scripts.run_multimodal_predict import load_trained_roberta

        untrained_fallback = not ckpt_path.exists()
        tokenizer, model, device = load_trained_roberta(
            ckpt_path, allow_pretrained_fallback=args.allow_pretrained_fallback
        )
        scorer = RobertaScorer(model, tokenizer, device, max_length=ModelConfig.max_length)
        rows, summary = evaluate(
            eval_dir, labels, scorer,
            recall_floor=args.recall_floor, max_samples=None,
        )
    else:
        # Baseline / DistilBERT / rule-based scorer: distilbert loads a
        # saved best_distilbert.pt artefact (a missing/wrong-kind
        # checkpoint raises — never refits, never falls back to base
        # weights); tfidf_lr/bilstm load a saved baseline artefact when
        # --checkpoint points at one, else fit on the real train split;
        # rule_based needs no fit at all; the checkpoint gate does not
        # apply and its absence is not an error for any of them except
        # distilbert, which requires its artefact.
        scorer = load_scorer(
            args.model, checkpoint=ckpt_path, bilstm_epochs=args.bilstm_epochs,
        )
        rows, summary = evaluate(
            eval_dir, labels, scorer,
            recall_floor=args.recall_floor, max_samples=None,
        )

    baselines_out: Dict[str, Any] = {}
    cheap = [m for m in args.models if m in ("tfidf_lr", "tfidf_svm", "bilstm")]
    if cheap and rows:
        logger.info("Running cheap baselines on the same extractor output: %s", cheap)
        baselines_out = run_cheap_baselines(rows, cheap, bilstm_epochs=args.bilstm_epochs)

    test_metrics = summary.get("test_metrics", {})
    precision_line = precision_first_summary(test_metrics) if test_metrics else (
        "Ablation NOT RUN YET; no precision summary is available."
    )

    results: Dict[str, Any] = {
        "synthetic_warning": SYNTHETIC_WARNING,
        "untrained_fallback": untrained_fallback,
        "untrained_warning": UNTRAINED_WARNING if untrained_fallback else None,
        "precision_first_summary": precision_line,
        "metadata": {
            "eval_dir": str(eval_dir),
            "labels": str(labels_path),
            "seed": 42,
            "n_total": len(labels),
            "n_phishing": sum(1 for r in labels if r["label"] == 1),
            "n_benign": sum(1 for r in labels if r["label"] == 0),
            "n_validation": summary.get("n_validation"),
            "n_test": summary.get("n_test"),
            "fusion": args.fusion,
            "checkpoint": str(ckpt_path),
            "model": args.model,
            "models": [args.model] + [m for m in cheap if m != args.model],
            "recall_floor": args.recall_floor,
            "threshold_note": (
                "Threshold 0.5 is a PLACEHOLDER. Reported thresholds were fitted "
                "on the validation slice only (precision-first, recall floor "
                f"{args.recall_floor}); never fitted on test."
            ),
        },
        "fitted_thresholds": summary.get("fitted_thresholds", {}),
        "test_metrics": test_metrics,
        "test_metrics_at_0_5": summary.get("test_metrics_at_0_5", {}),
        "per_modality_recall": summary.get("per_modality_recall", {}),
        "failures": summary.get("failures", []),
        "baselines": baselines_out,
        "rows": [
            {k: v for k, v in r.items()
             if k not in ("fusion_weighted", "fusion_max")}
            for r in rows
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    figures: List[str] = []
    if test_metrics:
        cm_path = _plot_confusion_matrix(
            test_metrics.get("d_weighted") or test_metrics.get("d_max"),
            out_dir / "confusion_matrix_weighted.png",
            "Confusion matrix — (d) full fusion, weighted (test slice)",
        )
        if cm_path is not None:
            figures.append(cm_path.name)
        bar_path = _plot_ablation_bars(test_metrics, out_dir / "ablation_precision_recall_fpr.png")
        if bar_path is not None:
            figures.append(bar_path.name)
    results["figures"] = figures

    json_path = out_dir / "multimodal_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    md_path = out_dir / "multimodal_eval.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(results))
    print(precision_line)
    print(f"Saved {json_path} and {md_path}")
    if untrained_fallback:
        print("WARNING: UNTRAINED MODEL - numbers above are NOT detection evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
