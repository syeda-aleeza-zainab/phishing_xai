"""Multimodal faithfulness evaluation (Comprehensiveness & Sufficiency at top-k=5) on the 6 .eml samples.

Reuses ``phishing_xai.faithfulness`` (no modifications to ``faithfulness.py``) and
``phishing_xai.lita`` / ``phishing_xai.explain`` on the extracted text of the 6 multimodal samples.

Each row reports LITA faithfulness AND the random baseline
(``faithfulness.evaluate_random_baseline``, same k, 5 seeds): an
explanation is only meaningful if its comprehensiveness/sufficiency
beat dropping/keeping k RANDOM words. Rows also carry
``untrained_fallback`` (true when the fine-tuned checkpoint is absent
and base roberta-base was used — numbers are then pipeline
demonstration only, not detection evidence).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.config import ModelConfig, TrainConfig, XAIConfig
from phishing_xai.data import truncate_text_to_model_window
from phishing_xai.explain import (
    LimeExplainer,
    TransformerExplainer,
    aggregate_subword_attributions_to_words,
)
from phishing_xai.extract import extract_eml
from phishing_xai.faithfulness import (
    evaluate_faithfulness_single_method,
    evaluate_random_baseline,
    predict_probability,
)
from phishing_xai.lita import build_hybrid_word_attributions
from phishing_xai.utils import set_seed
from scripts.run_multimodal_predict import load_trained_roberta, score_text


def evaluate_multimodal_samples_faithfulness(
    samples_dir: Path,
    checkpoint_path: Path,
    k: int = 5,
    lime_samples: int = 200,
    allow_pretrained_fallback: bool = False,
) -> List[Dict[str, Any]]:
    set_seed(TrainConfig.seed)
    # True when the fine-tuned checkpoint is absent and base roberta-base is used instead.
    untrained_fallback = not checkpoint_path.exists()
    tokenizer, model, device = load_trained_roberta(
        checkpoint_path, allow_pretrained_fallback=allow_pretrained_fallback
    )
    lime_exp = LimeExplainer(model=model, tokenizer=tokenizer, device=device, max_length=ModelConfig.max_length)
    ti_exp = TransformerExplainer(model=model, tokenizer=tokenizer, device=device)

    rows: List[Dict[str, Any]] = []
    for eml_path in sorted(samples_dir.glob("*.eml")):
        extracted = extract_eml(eml_path)
        # Prefer OCR/PDF/QR extracted attachment text when available, or winning part by score
        candidate_parts = []
        for p in extracted["parts"]:
            txt = (p.get("text") or "").strip()
            if not txt:
                continue
            pred_cls, _, p_phish = score_text(model, tokenizer, txt, device=device)
            candidate_parts.append((p, pred_cls, p_phish))

        if not candidate_parts:
            # No scorable text in this email (e.g. OCR returned nothing):
            # record the skip instead of crashing on max() of an empty list.
            logger_warning = f"No scorable text parts in {eml_path.name}; skipping faithfulness."
            print(f"WARNING: {logger_warning}")
            rows.append({
                "sample": eml_path.name,
                "part_kind": None,
                "pred_class": None,
                "p_orig": None,
                "comprehensiveness_k5": None,
                "sufficiency_k5": None,
                "random_comprehensiveness_k5": None,
                "random_sufficiency_k5": None,
                "top_5_words": [],
                "status": "no_text_parts",
                "untrained_fallback": untrained_fallback,
            })
            continue

        # Pick winning part (max phishing prob, preferring non-body attachment when tied)
        winning_part, pred_cls, _ = max(
            candidate_parts,
            key=lambda item: (item[0]["kind"] != "body_text", item[2]),
        )
        raw_text = winning_part["text"]
        truncated = truncate_text_to_model_window(raw_text, tokenizer, max_length=ModelConfig.max_length)

        lime_attrs, target_class, _ = lime_exp.explain(truncated, num_samples=lime_samples)
        raw_ti = ti_exp.explain_raw(truncated, target_class=target_class)
        word_ti = aggregate_subword_attributions_to_words(
            truncated, raw_ti, tokenizer, max_length=ModelConfig.max_length
        )
        hybrid = build_hybrid_word_attributions(
            word_attributions_ti=word_ti,
            word_attributions_lime=lime_attrs,
            alpha=XAIConfig.alpha,
            variant="magnitude_preserving",
        )
        words = [item["word"] for item in hybrid]
        lita_scores = [item["lita_score"] for item in hybrid]

        p_orig = predict_probability(
            model=model,
            tokenizer=tokenizer,
            text=" ".join(words),
            target_class=target_class,
            device=device,
            max_length=ModelConfig.max_length,
        )
        faith = evaluate_faithfulness_single_method(
            model=model,
            tokenizer=tokenizer,
            words=words,
            scores=lita_scores,
            target_class=target_class,
            p_orig=p_orig,
            ks=(k,),
            device=device,
            max_length=ModelConfig.max_length,
        )
        top_words = [
            d["word"] for d in sorted(hybrid, key=lambda x: x["lita_score"], reverse=True)[:k]
        ]
        # Random baseline (Section E): the same comp/suff maths on k RANDOM
        # words, averaged over 5 seeds. LITA is only meaningful if it beats
        # this baseline; without it the numbers have no reference point.
        random_faith = evaluate_random_baseline(
            model=model,
            tokenizer=tokenizer,
            words=words,
            target_class=target_class,
            p_orig=p_orig,
            ks=(k,),
            num_seeds=5,
            device=device,
            max_length=ModelConfig.max_length,
        )
        rows.append({
            "sample": eml_path.name,
            "part_kind": winning_part["kind"],
            "pred_class": int(target_class),
            "p_orig": float(p_orig),
            "comprehensiveness_k5": float(faith["comprehensiveness"][k]),
            "sufficiency_k5": float(faith["sufficiency"][k]),
            "random_comprehensiveness_k5": float(random_faith["comprehensiveness"][k]),
            "random_sufficiency_k5": float(random_faith["sufficiency"][k]),
            "top_5_words": top_words,
            "status": "ok",
            "untrained_fallback": untrained_fallback,
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LITA faithfulness (top-k=5) on the 6 multimodal samples")
    parser.add_argument(
        "--samples_dir",
        type=str,
        default=str(REPO_ROOT / "samples" / "multimodal"),
        help="Directory containing sample .eml files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(REPO_ROOT / "results" / "best_model.pt"),
        help="Trained RoBERTa checkpoint path",
    )
    parser.add_argument("--lime_samples", type=int, default=200, help="LIME perturbation samples")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--allow_pretrained_fallback",
        action="store_true",
        help="Allow running with base roberta-base when results/best_model.pt is missing locally",
    )
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() and not args.allow_pretrained_fallback:
        print(
            f"ERROR: Trained checkpoint is not present locally! Missing file: {ckpt.as_posix()}\n"
            "Per project rules, do NOT retrain locally.",
            file=sys.stderr,
        )
        return 2

    rows = evaluate_multimodal_samples_faithfulness(
        samples_dir=Path(args.samples_dir),
        checkpoint_path=ckpt,
        k=5,
        lime_samples=args.lime_samples,
        allow_pretrained_fallback=args.allow_pretrained_fallback,
    )
    def _fmt(value: Any) -> str:
        return f"{value:+.4f}" if isinstance(value, (int, float)) else "n/a"

    print("| Sample | Part Kind | Pred Class | P(orig) | Comprehensiveness (k=5) | Sufficiency (k=5) | Random Comp (k=5) | Random Suff (k=5) | Top-5 LITA Words |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        top_w = ", ".join(r["top_5_words"])
        p_orig = f"{r['p_orig']:.4f}" if isinstance(r.get("p_orig"), (int, float)) else "n/a"
        print(
            f"| `{r['sample']}` | `{r['part_kind']}` | {r['pred_class']} | "
            f"{p_orig} | {_fmt(r['comprehensiveness_k5'])} | {_fmt(r['sufficiency_k5'])} | "
            f"{_fmt(r['random_comprehensiveness_k5'])} | {_fmt(r['random_sufficiency_k5'])} | {top_w} |"
        )
    if any(r.get("untrained_fallback") for r in rows):
        print("WARNING: UNTRAINED MODEL - numbers above are NOT detection evidence (base roberta-base).")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
