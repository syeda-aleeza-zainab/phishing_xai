"""Run alpha ablation study on LITA across normalization variants and alpha weighting."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phishing_xai.data import load_data, truncate_text_to_model_window
from phishing_xai.explain import (
    LimeExplainer,
    TransformerExplainer,
    aggregate_subword_attributions_to_words,
    split_words,
)
from phishing_xai.faithfulness import (
    evaluate_faithfulness_single_method,
    predict_probability,
)
from phishing_xai.lita import compute_lita_scores
from phishing_xai.model import load_tokenizer_and_model
from phishing_xai.utils import ensure_dir, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("run_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpha Ablation Study for LITA")
    parser.add_argument("--model_path", type=str, default="results/best_model.pt", help="Path to trained model")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv", help="Path to raw dataset")
    parser.add_argument("--splits_path", type=str, default="results/splits.json", help="Path to split manifest")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save ablation artifacts")
    parser.add_argument("--sample_size", type=int, default=50, help="Number of test instances to evaluate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--k", type=int, default=3, help="Value of k for comprehensiveness and sufficiency")
    parser.add_argument("--debug", action="store_true", help="Debug mode (subsample N=5 emails, fast)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing records")
    return parser.parse_args()


def load_model_and_tokenizer(model_path_str: str, device: torch.device):
    """Load model and tokenizer from checkpoint or fallback."""
    p = Path(model_path_str)
    model_name = "roberta-base"
    tokenizer, model = load_tokenizer_and_model(model_name=model_name, num_labels=2, device=device)

    if p.exists() and p.is_file():
        logger.info(f"Loading model checkpoint from {p}")
        checkpoint = torch.load(p, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    else:
        logger.info(f"Using pretrained {model_name} (checkpoint {p} not found).")

    model.to(device)
    model.eval()
    return model, tokenizer


def run_ablation(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = get_device()

    out_dir = Path("results/debug") if args.debug else Path(args.output_dir)
    ensure_dir(out_dir)

    records_file = out_dir / "ablation_records.jsonl"
    final_output_file = out_dir / "alpha_ablation.json"

    # 1. Load data and test split
    df = load_data(args.data_path)
    if "email_id" not in df.columns:
        df["email_id"] = df.index

    splits_path = Path("results/debug/splits.json") if args.debug and Path("results/debug/splits.json").exists() else Path(args.splits_path)
    if splits_path.exists():
        manifest = load_json(splits_path)
        test_ids = set(manifest["test_indices"])
        test_df = df[df["email_id"].isin(test_ids)].reset_index(drop=True)
    else:
        from phishing_xai.data import create_stratified_splits
        splits = create_stratified_splits(df, seed=args.seed, debug=args.debug)
        test_df = splits.test_df

    sample_count = min(5 if args.debug else args.sample_size, len(test_df))
    eval_df = test_df.sample(n=sample_count, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Running alpha ablation on {len(eval_df)} test samples (debug={args.debug}).")

    # 2. Setup model & explainers
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    lime_explainer = LimeExplainer(model, tokenizer, device=device, max_length=args.max_length, seed=args.seed)
    ti_explainer = TransformerExplainer(model, tokenizer, device=device)

    # 3. Resume support
    completed_ids = set()
    existing_records = []
    if args.resume and records_file.exists():
        with open(records_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed_ids.add(rec["email_id"])
                    existing_records.append(rec)
        logger.info(f"Resuming: found {len(completed_ids)} already completed ablation records.")

    records = list(existing_records)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    variants = ["magnitude_preserving", "minmax"]
    k_eval = args.k
    lime_perturbations = 50 if args.debug else 200

    for idx, row in eval_df.iterrows():
        email_id = int(row["email_id"])
        if email_id in completed_ids:
            continue

        raw_text = str(row["clean_text"])
        true_label = int(row["label"])

        text = truncate_text_to_model_window(raw_text, tokenizer, max_length=args.max_length)
        words = split_words(text)
        if len(words) < 2:
            continue

        p_orig_0 = predict_probability(model, tokenizer, text, 0, device=device, max_length=args.max_length)
        p_orig_1 = predict_probability(model, tokenizer, text, 1, device=device, max_length=args.max_length)
        pred_class = 1 if p_orig_1 >= p_orig_0 else 0
        p_orig = p_orig_1 if pred_class == 1 else p_orig_0

        # Base attributions
        ti_raw = ti_explainer.explain_raw(text, target_class=pred_class)
        ti_agg = aggregate_subword_attributions_to_words(text, ti_raw, tokenizer, max_length=args.max_length)
        ti_scores = [item[2] for item in ti_agg]

        lime_raw, _, _ = lime_explainer.explain(text, num_features=None, num_samples=lime_perturbations)
        lime_map = {w.strip().lower(): s for w, s in lime_raw if w.strip()}
        lime_scores = [float(lime_map.get(w.lower(), 0.0)) for w in words]

        instance_results: Dict[str, Any] = {
            "email_id": email_id,
            "true_label": true_label,
            "pred_class": pred_class,
            "p_orig": p_orig,
            "evaluations": {},
        }

        for variant in variants:
            instance_results["evaluations"][variant] = {}
            for alpha in alphas:
                lita_scores = compute_lita_scores(lime_scores, ti_scores, alpha=alpha, variant=variant).tolist()
                res = evaluate_faithfulness_single_method(
                    model=model,
                    tokenizer=tokenizer,
                    words=words,
                    scores=lita_scores,
                    target_class=pred_class,
                    p_orig=p_orig,
                    ks=[k_eval],
                    device=device,
                    max_length=args.max_length,
                )
                instance_results["evaluations"][variant][str(alpha)] = {
                    "comprehensiveness": res["comprehensiveness"][k_eval],
                    "sufficiency": res["sufficiency"][k_eval],
                }

        records.append(instance_results)
        with open(records_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(instance_results) + "\n")

        logger.info(f"Processed ablation {len(records)}/{len(eval_df)} (id={email_id}).")

    # 4. Summary table
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in variants:
        summary[variant] = {}
        for alpha in alphas:
            alpha_str = str(alpha)
            comp_vals = [r["evaluations"][variant][alpha_str]["comprehensiveness"] for r in records]
            suff_vals = [r["evaluations"][variant][alpha_str]["sufficiency"] for r in records]

            summary[variant][alpha_str] = {
                "comprehensiveness_mean": float(np.mean(comp_vals)) if comp_vals else 0.0,
                "comprehensiveness_std": float(np.std(comp_vals)) if comp_vals else 0.0,
                "sufficiency_mean": float(np.mean(suff_vals)) if suff_vals else 0.0,
                "sufficiency_std": float(np.std(suff_vals)) if suff_vals else 0.0,
            }

    output = {
        "metadata": {
            "debug": args.debug,
            "sample_count": len(records),
            "k": k_eval,
            "alphas": alphas,
            "variants": variants,
            "seed": args.seed,
        },
        "summary": summary,
    }
    save_json(output, final_output_file)
    logger.info(f"Saved alpha ablation summary to: {final_output_file}")
    return output


def main() -> None:
    args = parse_args()
    res = run_ablation(args)
    print("\n" + "=" * 70)
    print(f"ALPHA ABLATION SUMMARY (k={args.k})")
    print("=" * 70)
    for variant, alphas_dict in res["summary"].items():
        print(f"\nVariant: {variant}")
        print(f"  {'Alpha':<8} | {'Comprehensiveness':<25} | {'Sufficiency':<25}")
        print("  " + "-" * 62)
        for alpha_str, m in alphas_dict.items():
            comp = f"{m['comprehensiveness_mean']:.4f} (+/- {m['comprehensiveness_std']:.4f})"
            suff = f"{m['sufficiency_mean']:.4f} (+/- {m['sufficiency_std']:.4f})"
            print(f"  {alpha_str:<8} | {comp:<25} | {suff:<25}")
    print("=" * 70)


if __name__ == "__main__":
    main()
