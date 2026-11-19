"""Multimodal phishing prediction & LITA explanation across .eml body text, images, PDFs, and QR codes.

Loads the existing trained RoBERTa via ``phishing_xai.model.load_tokenizer_and_model``
(never retrains, never modifies ``model.py``). Scores ``body_text`` plus each extracted attachment
part, prints a per-part table, and applies the **max rule**:
  - Final verdict is Phishing (1) if ANY non-empty part scores Phishing (prob >= 0.5).
  - Winning part is the part with the highest phishing probability (or highest predicted-class
    probability when all parts are Safe).
With ``--explain``, runs the LITA hybrid explainer (``explain.py`` + ``lita.py``) on the winning
part, prints the top 5 words, and if the part is an image with OCR bounding boxes, draws red
boxes around the top 5 LITA words and saves ``<name>_explained.png``.
"""

import argparse
import email
from email import policy
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phishing_xai.config import ModelConfig, TrainConfig, XAIConfig
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
from phishing_xai.utils import get_device, get_logger, set_seed

logger = get_logger("run_multimodal_predict")


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
            f"Checkpoint {checkpoint_path.as_posix()} missing locally; using base {model_name} (--allow_pretrained_fallback)."
        )
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


def predict_eml(
    eml_path: Path,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    explain: bool = False,
    lime_samples: int = 200,
) -> Dict[str, Any]:
    """Extract parts from an .eml file, score each part, apply max rule, and optionally explain."""
    extracted = extract_eml(eml_path)
    part_rows: List[Dict[str, Any]] = []

    for idx, part in enumerate(extracted["parts"]):
        text = (part.get("text") or "").strip()
        label_name = part["kind"]
        if part.get("filename"):
            label_name += f":{part['filename']}"
        if part.get("page") is not None:
            label_name += f"#p{part['page']}"

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

    non_empty = [r for r in part_rows if r["pred_class"] is not None]
    if non_empty:
        winning_part = max(non_empty, key=lambda r: r["prob_phishing"])
        any_phishing = any(r["pred_class"] == 1 for r in non_empty)
        final_class = 1 if any_phishing else 0
    else:
        winning_part = part_rows[0]
        final_class = 0

    top_words: List[Dict[str, Any]] = []
    explained_image_path: Optional[str] = None
    if explain and winning_part.get("text"):
        top_words = explain_part_lita(
            model=model,
            tokenizer=tokenizer,
            text=winning_part["text"],
            device=device,
            num_samples=lime_samples,
            top_k=5,
        )
        if winning_part["kind"] == "image" and winning_part.get("boxes"):
            img_bytes = _extract_first_image_bytes(eml_path, winning_part.get("filename"))
            if img_bytes:
                out_img = eml_path.with_name(f"{eml_path.stem}_explained.png")
                draw_lita_boxes_on_image(
                    image_input=img_bytes,
                    ocr_boxes=winning_part["boxes"],
                    lita_top_words=top_words,
                    output_path=out_img,
                    top_k=5,
                )
                explained_image_path = str(out_img)

    return {
        "file": eml_path.name,
        "parts": part_rows,
        "final_class": final_class,
        "final_verdict": CLASS_NAMES[final_class],
        "winning_part": winning_part["part_id"],
        "winning_prob_phishing": float(winning_part["prob_phishing"]),
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
    print(
        f"FINAL VERDICT (max rule): {result['final_verdict']} "
        f"| Winning Part: {result['winning_part']} (P(Phish)={result['winning_prob_phishing']:.4f})"
    )
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
        default=str(REPO_ROOT / "results" / "best_model.pt"),
        help="Path to trained RoBERTa checkpoint (default: results/best_model.pt)",
    )
    parser.add_argument("--explain", action="store_true", help="Run LITA explainer on winning part")
    parser.add_argument("--lime_samples", type=int, default=200, help="Number of LIME perturbation samples")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save JSON summary")
    parser.add_argument(
        "--allow_pretrained_fallback",
        action="store_true",
        help="Allow running with base roberta-base when results/best_model.pt is missing locally",
    )
    args = parser.parse_args(argv)

    set_seed(TrainConfig.seed)
    ckpt_path = Path(args.checkpoint)
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

    if not eml_files:
        print("No .eml files found to score.", file=sys.stderr)
        return 1

    tokenizer, model, device = load_trained_roberta(
        ckpt_path, allow_pretrained_fallback=args.allow_pretrained_fallback
    )
    all_results: List[Dict[str, Any]] = []
    for eml_path in eml_files:
        res = predict_eml(
            eml_path=eml_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            explain=args.explain,
            lime_samples=args.lime_samples,
        )
        print_prediction_report(res)
        all_results.append(res)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
