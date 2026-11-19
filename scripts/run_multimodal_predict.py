"""Multimodal phishing prediction & LITA explanation across .eml body text, images, PDFs, and QR codes.

Loads the existing trained RoBERTa via ``phishing_xai.model.load_tokenizer_and_model``
(never retrains, never modifies ``model.py``). Scores ``body_text`` plus each extracted attachment
part, scores the concatenated context (subject + body + OCR + PDF + QR URL in one
marked string), scores every URL with Section C static risk, and fuses everything
with ``phishing_xai.multimodal.fuse_email`` (Section D).

Fusion (default ``--fusion weighted``, precision-first — see ``multimodal.py``):
the weighted fusion combines the concatenated-context score (weight 0.45), the
single best part (0.20), the worst URL score (0.20) and the header/flag score
(0.15), and additionally requires corroboration across channels, so one weak
part alone cannot flip the verdict. ``--fusion max`` keeps the old baseline:
the verdict is Phishing (1) if ANY non-empty part scores Phishing (prob >=
threshold); it is recall-heavy / precision-poor and ignores URL/flag/context
evidence. Both fusions are computed and reported for every email so Section E
can compare precision / recall / FPR side by side.

``--threshold`` defaults to 0.5, which is a PLACEHOLDER: it must be tuned on
VALIDATION only (never test) on Kaggle with the trained checkpoint, choosing
the threshold that maximises phishing-class precision subject to a recall
floor — never by maximising F1 alone. Until that validation run exists, do
not claim the threshold was tuned.

With ``--explain``, runs the LITA hybrid explainer (``explain.py`` + ``lita.py``)
on the concatenated context, prints the top 5 words, and draws overlays:
image parts via ``draw_lita_boxes_on_image`` (as before), scanned-PDF pages
via ``multimodal.overlay_scanned_pdf_page`` (page rendered with PyMuPDF),
and QR images via ``multimodal.overlay_qr_image`` (decoded-QR region +
annotated URL). Overlay helpers skip with a warning when their rendering
dependency is missing; they never crash the run.

Kaggle path auto-detection: when ``/kaggle`` exists, outputs default to
``/kaggle/working/results`` and the checkpoint defaults to the first
``/kaggle/input/*/best_model.pt`` found, else
``/kaggle/working/results/best_model.pt``. Locally the defaults are
``results/best_model.pt`` and ``results/multimodal``.
"""

import argparse
import email
from email import policy
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phishing_xai.config import FusionConfig, ModelConfig, TrainConfig, XAIConfig
from phishing_xai.data import truncate_text_to_model_window
from phishing_xai.explain import (
    CLASS_NAMES,
    LimeExplainer,
    TransformerExplainer,
    aggregate_subword_attributions_to_words,
    draw_lita_boxes_on_image,
)
from phishing_xai.extract import extract_eml
from phishing_xai.lita import build_hybrid_word_attributions
from phishing_xai.model import load_tokenizer_and_model
from phishing_xai.multimodal import (
    build_concatenated_context,
    fuse_email,
    overlay_qr_image,
    overlay_scanned_pdf_page,
    part_id,
)
from phishing_xai.utils import get_device, get_logger, set_seed

logger = get_logger("run_multimodal_predict")

#: Default decision threshold. PLACEHOLDER (0.5) — tune on validation only;
#: see the module docstring. Never claim it was tuned until that run exists.
DEFAULT_THRESHOLD: float = FusionConfig.threshold


def resolve_default_checkpoint() -> Path:
    """Default checkpoint path, with Kaggle auto-detection.

    On Kaggle (``/kaggle`` exists): first ``/kaggle/input/*/best_model.pt``
    (sorted), else ``/kaggle/working/results/best_model.pt``. Locally:
    ``results/best_model.pt`` under the repo root.
    """
    if os.path.exists("/kaggle"):
        kaggle_inputs = sorted(Path("/kaggle/input").glob("*/best_model.pt"))
        if kaggle_inputs:
            return kaggle_inputs[0]
        return Path("/kaggle/working/results/best_model.pt")
    return REPO_ROOT / "results" / "best_model.pt"


def resolve_default_output_dir() -> Path:
    """Default output directory, with Kaggle auto-detection."""
    if os.path.exists("/kaggle"):
        return Path("/kaggle/working/results")
    return REPO_ROOT / "results" / "multimodal"


def load_trained_roberta(
    checkpoint_path: Path,
    model_name: str = ModelConfig.model_name,
    device: Optional[torch.device] = None,
    allow_pretrained_fallback: bool = False,
) -> Tuple[Any, Any, torch.device]:
    """Load tokenizer and RoBERTa architecture via model.py and restore trained weights."""
    if not checkpoint_path.exists() and not allow_pretrained_fallback:
        raise FileNotFoundError(
            f"Trained checkpoint not found locally: {checkpoint_path.as_posix()} "
            "(Hard rule: do NOT retrain. Provide an existing checkpoint or pass --allow_pretrained_fallback.)"
        )
    device = device or get_device()
    tokenizer, model = load_tokenizer_and_model(
        model_name=model_name,
        num_labels=ModelConfig.num_labels,
        device=device,
    )
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            f"Checkpoint {checkpoint_path.as_posix()} missing locally; using base {model_name} (--allow_pretrained_fallback). "
            "WARNING: UNTRAINED MODEL - results are NOT detection evidence."
        )
        print("!" * 88)
        print("WARNING: UNTRAINED MODEL - results are NOT detection evidence")
        print(f"Checkpoint {checkpoint_path.as_posix()} is missing; running with untrained base {model_name}.")
        print("Any verdicts / probabilities / faithfulness numbers below are pipeline-demonstration")
        print("output only. Do NOT cite them as phishing-detection evidence.")
        print("!" * 88)
    model.eval()
    return tokenizer, model, device


def score_text(
    model: Any,
    tokenizer: Any,
    text: str,
    device: torch.device,
    max_length: int = ModelConfig.max_length,
) -> Tuple[int, float, float]:
    """Return (pred_class, prob_safe, prob_phishing) for a single text string."""
    clean = (text or "").strip()
    if not clean:
        return 0, 1.0, 0.0
    truncated = truncate_text_to_model_window(clean, tokenizer, max_length=max_length)
    encoded = tokenizer(
        truncated,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    prob_safe = float(probs[0])
    prob_phish = float(probs[1])
    pred_class = 1 if prob_phish >= 0.5 else 0
    return pred_class, prob_safe, prob_phish


def explain_part_lita(
    model: Any,
    tokenizer: Any,
    text: str,
    device: torch.device,
    num_samples: int = 200,
    max_length: int = ModelConfig.max_length,
    alpha: float = XAIConfig.alpha,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Run LIME + TransformerExplainer + LITA on a text part and return top-k word dicts."""
    truncated = truncate_text_to_model_window(text, tokenizer, max_length=max_length)
    lime_exp = LimeExplainer(model=model, tokenizer=tokenizer, device=device, max_length=max_length)
    ti_exp = TransformerExplainer(model=model, tokenizer=tokenizer, device=device)

    lime_attrs, pred_class, _ = lime_exp.explain(truncated, num_samples=num_samples)
    raw_ti = ti_exp.explain_raw(truncated, target_class=pred_class)
    word_ti = aggregate_subword_attributions_to_words(truncated, raw_ti, tokenizer, max_length=max_length)
    hybrid = build_hybrid_word_attributions(
        word_attributions_ti=word_ti,
        word_attributions_lime=lime_attrs,
        alpha=alpha,
        variant="magnitude_preserving",
    )
    ranked = sorted(hybrid, key=lambda d: d["lita_score"], reverse=True)
    return ranked[:top_k]


def _extract_first_image_bytes(eml_path: Path, target_filename: Optional[str] = None) -> Optional[bytes]:
    """Pull raw image payload bytes from an .eml file for visual bounding-box rendering."""
    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_type().startswith("image/"):
            fname = part.get_filename() or part.get("Content-ID")
            if target_filename is None or fname == target_filename:
                payload = part.get_payload(decode=True)
                if payload:
                    return payload
    return None


def _extract_pdf_bytes(eml_path: Path, target_filename: Optional[str] = None) -> Optional[bytes]:
    """Pull raw PDF payload bytes from an .eml file (filename may carry a '#img<xref>' suffix)."""
    wanted = (target_filename or "").split("#")[0] or None
    with open(eml_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_type() == "application/pdf" or (part.get_filename() or "").lower().endswith(".pdf"):
            fname = part.get_filename()
            if wanted is None or fname == wanted:
                payload = part.get_payload(decode=True)
                if payload:
                    return payload
    return None


def predict_eml(
    eml_path: Path,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    explain: bool = False,
    lime_samples: int = 200,
    fusion: str = "weighted",
    threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extract parts from an .eml file, score each part + the concatenated context, and fuse.

    Both fusions are computed and returned (``fusion_weighted`` and
    ``fusion_max``); the primary verdict (``final_class`` / ``final_verdict``)
    comes from ``fusion`` (default ``"weighted"``). With ``explain``, LITA
    runs on the concatenated context and overlays are drawn for the winning
    part (image / scanned-PDF page / QR image) into ``output_dir`` when
    given, else next to the .eml file.
    """
    if fusion not in ("weighted", "max"):
        raise ValueError(f"fusion must be 'weighted' or 'max', got {fusion!r}")
    thr = float(DEFAULT_THRESHOLD if threshold is None else threshold)
    out_dir = Path(output_dir) if output_dir is not None else None

    extracted = extract_eml(eml_path)
    part_rows: List[Dict[str, Any]] = []
    part_scores: Dict[str, float] = {}

    for idx, part in enumerate(extracted["parts"]):
        text = (part.get("text") or "").strip()
        label_name = part_id(part)
        if not text:
            part_rows.append({
                "index": idx,
                "part_id": label_name,
                "kind": part["kind"],
                "filename": part.get("filename"),
                "text": "",
                "pred_class": None,
                "verdict": "EMPTY (skipped)",
                "prob_safe": None,
                "prob_phishing": -1.0,
                "boxes": part.get("boxes", []),
            })
            continue

        pred_cls, p_safe, p_phish = score_text(model, tokenizer, text, device=device)
        part_scores[label_name] = p_phish
        part_rows.append({
            "index": idx,
            "part_id": label_name,
            "kind": part["kind"],
            "filename": part.get("filename"),
            "text": text,
            "pred_class": pred_cls,
            "verdict": CLASS_NAMES[pred_cls],
            "prob_safe": p_safe,
            "prob_phishing": p_phish,
            "boxes": part.get("boxes", []),
        })

    context = build_concatenated_context(
        extracted, tokenizer=tokenizer, max_length=ModelConfig.max_length
    )
    _, _, context_prob_phish = score_text(model, tokenizer, context, device=device)

    fused_weighted = fuse_email(
        extracted, part_scores, fusion="weighted",
        concatenated_score=context_prob_phish, threshold=thr,
        tokenizer=tokenizer, max_length=ModelConfig.max_length,
    )
    fused_max = fuse_email(
        extracted, part_scores, fusion="max",
        concatenated_score=context_prob_phish, threshold=thr,
        tokenizer=tokenizer, max_length=ModelConfig.max_length,
    )
    primary = fused_weighted if fusion == "weighted" else fused_max
    final_class = int(primary["final_class"])

    non_empty = [r for r in part_rows if r["pred_class"] is not None]
    winning_part = max(non_empty, key=lambda r: r["prob_phishing"]) if non_empty else part_rows[0]

    top_words: List[Dict[str, Any]] = []
    explained_image_path: Optional[str] = None
    if explain and context.strip():
        top_words = explain_part_lita(
            model=model,
            tokenizer=tokenizer,
            text=context,
            device=device,
            num_samples=lime_samples,
            top_k=5,
        )
        img_dir = out_dir if out_dir is not None else eml_path.parent
        img_dir.mkdir(parents=True, exist_ok=True)
        out_img = img_dir / f"{eml_path.stem}_explained.png"
        kind = winning_part["kind"]
        if kind == "image" and winning_part.get("boxes"):
            img_bytes = _extract_first_image_bytes(eml_path, winning_part.get("filename"))
            if img_bytes:
                draw_lita_boxes_on_image(
                    image_input=img_bytes,
                    ocr_boxes=winning_part["boxes"],
                    lita_top_words=top_words,
                    output_path=out_img,
                    top_k=5,
                )
                explained_image_path = str(out_img)
        elif kind in ("pdf_ocr", "pdf_text", "pdf_image"):
            pdf_bytes = _extract_pdf_bytes(eml_path, winning_part.get("filename"))
            page = winning_part.get("page") if isinstance(winning_part, dict) else None
            # winning_part rows do not carry "page"; recover it from the part.
            if page is None:
                for part in extracted["parts"]:
                    if part_id(part) == winning_part["part_id"]:
                        page = part.get("page") or 1
                        break
            if pdf_bytes and page is not None:
                boxes = winning_part.get("boxes", [])
                result_path = overlay_scanned_pdf_page(
                    pdf_bytes, int(page), boxes, top_words, out_img,
                )
                if result_path is not None:
                    explained_image_path = str(result_path)
        elif kind == "qr":
            img_bytes = _extract_first_image_bytes(eml_path, winning_part.get("filename"))
            qr_urls = []
            for part in extracted["parts"]:
                if part_id(part) == winning_part["part_id"]:
                    qr_urls = part.get("qr_urls") or []
                    break
            if img_bytes:
                result_path = overlay_qr_image(img_bytes, qr_urls, out_img)
                if result_path is not None:
                    explained_image_path = str(result_path)

    return {
        "file": eml_path.name,
        "parts": part_rows,
        "fusion": fusion,
        "threshold": thr,
        "threshold_note": (
            "Threshold 0.5 is a PLACEHOLDER — tune on VALIDATION only on Kaggle "
            "with the trained checkpoint; never claimed as tuned."
            if thr == DEFAULT_THRESHOLD else "Threshold supplied via --threshold."
        ),
        "final_class": final_class,
        "final_verdict": CLASS_NAMES[final_class],
        "final_score": primary["final_score"],
        "winning_part": winning_part["part_id"],
        "winning_prob_phishing": float(winning_part["prob_phishing"]),
        "concatenated_context": context,
        "context_prob_phishing": float(context_prob_phish),
        "fusion_weighted": fused_weighted,
        "fusion_max": fused_max,
        "most_contributing_modality": primary["most_contributing_modality"],
        "top_reasons": primary["top_reasons"],
        "top_5_lita_words": top_words,
        "explained_image": explained_image_path,
    }


def print_prediction_report(result: Dict[str, Any]) -> None:
    print("=" * 88)
    print(f"FILE: {result['file']}")
    print("-" * 88)
    print(f"{'Part':<36} | {'Verdict':<16} | {'P(Phish)':<10} | {'Text Preview'}")
    print("-" * 88)
    for r in result["parts"]:
        prob_str = f"{r['prob_phishing']:.4f}" if r["pred_class"] is not None else "N/A"
        preview = (r["text"][:32] + "...") if len(r["text"]) > 35 else r["text"]
        print(f"{r['part_id']:<36} | {r['verdict']:<16} | {prob_str:<10} | {preview}")
    print("-" * 88)
    if result.get("untrained_fallback"):
        print("WARNING: UNTRAINED MODEL - results are NOT detection evidence (base roberta-base).")
    fw = result.get("fusion_weighted") or {}
    fm = result.get("fusion_max") or {}
    print(
        f"CONTEXT P(Phish)={result.get('context_prob_phishing', float('nan')):.4f} | "
        f"URL score={fw.get('url_score', float('nan')):.2f} | "
        f"Flag score={fw.get('flag_score', float('nan')):.2f} | "
        f"Most contributing modality: {result.get('most_contributing_modality')}"
    )
    print(
        f"WEIGHTED (precision-first): {fw.get('verdict')} "
        f"(score={fw.get('final_score', float('nan')):.4f}, threshold={result.get('threshold')}, "
        f"corroborated={fw.get('corroborated')})"
    )
    print(
        f"MAX (recall-heavy baseline): {fm.get('verdict')} "
        f"(score={fm.get('final_score', float('nan')):.4f})"
    )
    print(
        f"FINAL VERDICT ({result.get('fusion')} fusion): {result['final_verdict']} "
        f"| Winning Part: {result['winning_part']} (P(Phish)={result['winning_prob_phishing']:.4f})"
    )
    for reason in result.get("top_reasons") or []:
        print(f"  - {reason}")
    if result.get("top_5_lita_words"):
        words_fmt = ", ".join(f"{w['word']} ({w['lita_score']:+.3f})" for w in result["top_5_lita_words"])
        print(f"TOP 5 LITA WORDS: {words_fmt}")
    if result.get("explained_image"):
        print(f"VISUAL EXPLANATION SAVED: {result['explained_image']}")
    print("=" * 88)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run multimodal phishing prediction on .eml files")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(REPO_ROOT / "samples" / "multimodal")],
        help=".eml file(s) or directory of .eml files (default: samples/multimodal)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained RoBERTa checkpoint (default: Kaggle auto-detect, else results/best_model.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for JSON summaries and explained images (default: Kaggle /kaggle/working/results, else results/multimodal)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of .eml files to process (default: all found)",
    )
    parser.add_argument(
        "--fusion",
        choices=["weighted", "max"],
        default="weighted",
        help="Fusion rule for the primary verdict (default: weighted, precision-first; max = recall-heavy baseline)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold on the fused score (default 0.5 PLACEHOLDER — tune on validation only, never test)",
    )
    parser.add_argument("--explain", action="store_true", help="Run LITA explainer on the concatenated context")
    parser.add_argument("--lime_samples", type=int, default=200, help="Number of LIME perturbation samples")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON summary")
    parser.add_argument(
        "--allow_pretrained_fallback",
        action="store_true",
        help="Allow running with base roberta-base when results/best_model.pt is missing locally",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose (DEBUG) logging")
    args = parser.parse_args(argv)

    if args.debug:
        import logging

        logging.getLogger("phishing_xai").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    set_seed(TrainConfig.seed)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_default_checkpoint()
    out_dir = Path(args.output_dir) if args.output_dir else resolve_default_output_dir()
    if not ckpt_path.exists() and not args.allow_pretrained_fallback:
        print(
            f"ERROR: Trained checkpoint is not present locally! Missing file: {ckpt_path.as_posix()}\n"
            "Per project rules, do NOT retrain locally.",
            file=sys.stderr,
        )
        return 2

    eml_files: List[Path] = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            eml_files.extend(sorted(p.glob("*.eml")))
        elif p.is_file():
            eml_files.append(p)
    eml_files = sorted(set(eml_files))

    if args.max_samples is not None:
        eml_files = eml_files[: max(0, args.max_samples)]

    if not eml_files:
        print("No .eml files found to score.", file=sys.stderr)
        return 1

    tokenizer, model, device = load_trained_roberta(
        ckpt_path, allow_pretrained_fallback=args.allow_pretrained_fallback
    )
    # True when the fine-tuned checkpoint is absent and base roberta-base is used instead.
    untrained_fallback = not ckpt_path.exists()
    all_results: List[Dict[str, Any]] = []
    for eml_path in eml_files:
        res = predict_eml(
            eml_path=eml_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            explain=args.explain,
            lime_samples=args.lime_samples,
            fusion=args.fusion,
            threshold=args.threshold,
            output_dir=out_dir,
        )
        res["untrained_fallback"] = untrained_fallback
        print_prediction_report(res)
        all_results.append(res)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = Path(args.output_json) if args.output_json else out_dir / "multimodal_predictions.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} prediction(s) to {out_p.as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
