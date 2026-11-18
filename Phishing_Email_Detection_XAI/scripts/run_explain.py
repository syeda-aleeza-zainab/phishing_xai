"""CLI driver for computing LIME, Transformers-Interpret / Captum, and LITA explanations."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.data import deduplicate_data, load_data
from phishing_xai.explain import (
    LimeExplainer,
    TransformerExplainer,
    aggregate_subword_attributions_to_words,
)
from phishing_xai.lita import build_hybrid_word_attributions
from phishing_xai.model import load_tokenizer_and_model
from phishing_xai.utils import ensure_dir, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("run_explain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XAI explanations: LIME, TI, and LITA")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Model backbone")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to fine-tuned weights (or None for base)")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv", help="Path to raw dataset")
    parser.add_argument("--splits_path", type=str, default="results/splits.json", help="Path to split manifest")
    parser.add_argument("--sample_text", type=str, default=None, help="Custom single text to explain")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test instances to explain")
    parser.add_argument("--lime_samples", type=int, default=1000, help="Perturbation sample count for LIME")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight parameter for LITA")
    parser.add_argument("--variant", type=str, default="magnitude_preserving", choices=["magnitude_preserving", "minmax"])
    parser.add_argument("--output_dir", type=str, default="results/explanations", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode: 2 samples, 100 LIME samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.debug:
        out_dir = Path(args.output_dir)
        if "debug" not in out_dir.parts:
            out_dir = Path("results/debug/explanations")
        args.output_dir = str(out_dir)
        args.num_samples = 2
        args.lime_samples = 30
        logger.info(f"Debug mode enabled: {args.num_samples} samples, {args.lime_samples} LIME samples.")

    out_dir = ensure_dir(args.output_dir)

    # 1. Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(args.model_name, num_labels=2, device=device)
    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        logger.info(f"Loading checkpoint weights from: {args.checkpoint_path}")
        chk = torch.load(args.checkpoint_path, map_location=device)
        state_dict = chk.get("model_state_dict", chk)
        model.load_state_dict(state_dict)

    model.eval()

    # 2. Initialize explainers
    max_len = 64 if args.debug else 512
    lime_explainer = LimeExplainer(model, tokenizer, device=device, max_length=max_len)
    ti_explainer = TransformerExplainer(model, tokenizer, device=device)

    # 3. Determine samples to explain
    texts_to_explain: List[Dict[str, Any]] = []
    if args.sample_text:
        texts_to_explain.append({"id": "custom_0", "text": args.sample_text, "true_label": None})
    else:
        df = load_data(args.data_path)
        df = deduplicate_data(df, text_col="clean_text", label_col="label")
        splits_path = Path(args.splits_path)
        if splits_path.exists():
            splits_data = load_json(splits_path)
            test_indices = splits_data.get("test_indices", [])
            test_df = df.iloc[test_indices]
            if args.debug:
                short_df = test_df[test_df["clean_text"].str.split().str.len() <= 40]
                selected_df = short_df.head(args.num_samples) if len(short_df) >= args.num_samples else test_df.head(args.num_samples)
            else:
                selected_df = test_df.head(args.num_samples)
        else:
            selected_df = df.head(args.num_samples)


        for idx, row in selected_df.iterrows():
            texts_to_explain.append({
                "id": f"email_{idx}",
                "text": str(row["clean_text"]),
                "true_label": int(row["label"]),
            })


    all_explanations: List[Dict[str, Any]] = []

    for item in texts_to_explain:
        text = item["text"]
        logger.info(f"Generating explanations for sample {item['id']} ({len(text.split())} words)...")

        # LIME explanation
        lime_words, pred_class, probs = lime_explainer.explain(
            text,
            num_features=15,
            num_samples=args.lime_samples,
        )

        # TI explanation & aggregation
        ti_raw = ti_explainer.explain_raw(text, target_class=pred_class)
        ti_words = aggregate_subword_attributions_to_words(text, ti_raw, tokenizer)

        # LITA hybrid
        hybrid_words = build_hybrid_word_attributions(
            word_attributions_ti=ti_words,
            word_attributions_lime=lime_words,
            alpha=args.alpha,
            variant=args.variant,
        )

        explanation_record = {
            "id": item["id"],
            "true_label": item["true_label"],
            "pred_label": int(pred_class),
            "probabilities": probs.tolist(),
            "lime_top_words": lime_words,
            "word_level_attributions": hybrid_words,
        }
        all_explanations.append(explanation_record)

    output_file = out_dir / "explanations.json"
    save_json({"explanations": all_explanations, "alpha": args.alpha, "variant": args.variant}, output_file)
    logger.info(f"Saved {len(all_explanations)} explanations to: {output_file}")


if __name__ == "__main__":
    main()
