"""Run faithfulness evaluation (Comprehensiveness, Sufficiency, Random baseline) and global explanations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    evaluate_random_baseline,
    predict_probability,
)
from phishing_xai.lita import compute_lita_scores
from phishing_xai.model import load_tokenizer_and_model
from phishing_xai.utils import ensure_dir, get_device, get_logger, load_json, save_json, set_seed

logger = get_logger("run_faithfulness")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XAI Faithfulness")
    parser.add_argument("--model_path", type=str, default="results/best_model.pt", help="Path to trained model weights or HF model name")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv", help="Path to raw dataset")
    parser.add_argument("--splits_path", type=str, default="results/splits.json", help="Path to split manifest")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save faithfulness artifacts")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of test instances to evaluate")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--num_samples_lime", type=int, default=300, help="Number of LIME perturbation samples")
    parser.add_argument("--debug", action="store_true", help="Debug mode (subsample N=5 emails, fast)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing records if available")
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


def run_faithfulness(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = get_device()

    out_dir = Path("results/debug") if args.debug else Path(args.output_dir)
    ensure_dir(out_dir)

    records_file = out_dir / "faithfulness_records.jsonl"
    final_output_file = out_dir / "faithfulness.json"
    global_exp_file = out_dir / "global_explanations.json"

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
        logger.warning(f"Splits file {splits_path} not found; using test fraction directly.")
        from phishing_xai.data import create_stratified_splits
        splits = create_stratified_splits(df, seed=args.seed, debug=args.debug)
        test_df = splits.test_df

    if args.debug:
        sample_count = min(5, len(test_df))
    else:
        sample_count = min(args.sample_size, len(test_df))

    eval_df = test_df.sample(n=sample_count, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Evaluating faithfulness on {len(eval_df)} test samples (debug={args.debug}).")

    # 2. Load model & initialize explainers
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
        logger.info(f"Resuming: found {len(completed_ids)} already completed evaluations.")

    records = list(existing_records)
    ks = [1, 3, 5]
    lime_perturbations = 50 if args.debug else args.num_samples_lime

    # Global explanations accumulator: class -> {word: [scores]}
    global_word_scores: Dict[int, Dict[str, List[float]]] = {0: {}, 1: {}}

    for idx, row in eval_df.iterrows():
        email_id = int(row["email_id"])
        if email_id in completed_ids:
            continue

        raw_text = str(row["clean_text"])
        true_label = int(row["label"])

        # Truncate text strictly to model window
        text = truncate_text_to_model_window(raw_text, tokenizer, max_length=args.max_length)
        words = split_words(text)
        if len(words) < 2:
            logger.info(f"Skipping email {email_id} (too short after truncation: {len(words)} words).")
            continue

        # Model prediction
        p_orig_0 = predict_probability(model, tokenizer, text, 0, device=device, max_length=args.max_length)
        p_orig_1 = predict_probability(model, tokenizer, text, 1, device=device, max_length=args.max_length)
        pred_class = 1 if p_orig_1 >= p_orig_0 else 0
        p_orig = p_orig_1 if pred_class == 1 else p_orig_0

        # Method 1: TI (or Captum fallback)
        ti_raw = ti_explainer.explain_raw(text, target_class=pred_class)
        ti_agg = aggregate_subword_attributions_to_words(text, ti_raw, tokenizer, max_length=args.max_length)
        ti_scores = [item[2] for item in ti_agg]

        # Method 2: LIME
        lime_raw, _, _ = lime_explainer.explain(text, num_features=None, num_samples=lime_perturbations)
        lime_map = {w.strip().lower(): s for w, s in lime_raw if w.strip()}
        lime_scores = [float(lime_map.get(w.lower(), 0.0)) for w in words]

        # Method 3 & 4: LITA variants
        lita_mag_scores = compute_lita_scores(lime_scores, ti_scores, alpha=0.5, variant="magnitude_preserving").tolist()
        lita_minmax_scores = compute_lita_scores(lime_scores, ti_scores, alpha=0.5, variant="minmax").tolist()

        # Evaluate faithfulness for each method
        eval_methods = {
            "lime": lime_scores,
            "transformers_interpret": ti_scores,
            "lita_magnitude": lita_mag_scores,
            "lita_minmax": lita_minmax_scores,
        }

        instance_results: Dict[str, Any] = {
            "email_id": email_id,
            "true_label": true_label,
            "pred_class": pred_class,
            "p_orig": p_orig,
            "num_words": len(words),
            "metrics": {},
        }

        for method_name, scores in eval_methods.items():
            res = evaluate_faithfulness_single_method(
                model=model,
                tokenizer=tokenizer,
                words=words,
                scores=scores,
                target_class=pred_class,
                p_orig=p_orig,
                ks=ks,
                device=device,
                max_length=args.max_length,
            )
            instance_results["metrics"][method_name] = res

        # Random baseline (5 seeds)
        random_res = evaluate_random_baseline(
            model=model,
            tokenizer=tokenizer,
            words=words,
            target_class=pred_class,
            p_orig=p_orig,
            ks=ks,
            num_seeds=5,
            device=device,
            max_length=args.max_length,
        )
        instance_results["metrics"]["random_baseline"] = random_res

        # Append to records
        records.append(instance_results)
        with open(records_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(instance_results) + "\n")

        # Accumulate global word scores for the predicted class
        for w, s in zip(words, lita_mag_scores):
            w_clean = w.lower()
            if len(w_clean) > 1:
                if w_clean not in global_word_scores[pred_class]:
                    global_word_scores[pred_class][w_clean] = []
                global_word_scores[pred_class][w_clean].append(s)

        logger.info(f"Processed email {len(records)}/{len(eval_df)} (id={email_id}).")

    # 4. Compute aggregate statistics across test instances
    methods = ["lime", "transformers_interpret", "lita_magnitude", "lita_minmax", "random_baseline"]
    summary_table: Dict[str, Dict[str, Dict[str, float]]] = {}

    for method in methods:
        summary_table[method] = {
            "comprehensiveness_mean": {},
            "comprehensiveness_std": {},
            "sufficiency_mean": {},
            "sufficiency_std": {},
        }
        for k in ks:
            comp_vals = [r["metrics"][method]["comprehensiveness"][str(k)] if str(k) in r["metrics"][method]["comprehensiveness"] else r["metrics"][method]["comprehensiveness"][k] for r in records]
            suff_vals = [r["metrics"][method]["sufficiency"][str(k)] if str(k) in r["metrics"][method]["sufficiency"] else r["metrics"][method]["sufficiency"][k] for r in records]

            summary_table[method]["comprehensiveness_mean"][str(k)] = float(np.mean(comp_vals)) if comp_vals else 0.0
            summary_table[method]["comprehensiveness_std"][str(k)] = float(np.std(comp_vals)) if comp_vals else 0.0
            summary_table[method]["sufficiency_mean"][str(k)] = float(np.mean(suff_vals)) if suff_vals else 0.0
            summary_table[method]["sufficiency_std"][str(k)] = float(np.std(suff_vals)) if suff_vals else 0.0

    faithfulness_output = {
        "metadata": {
            "debug": args.debug,
            "sample_count": len(records),
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "summary": summary_table,
    }
    save_json(faithfulness_output, final_output_file)
    logger.info(f"Saved faithfulness summary to: {final_output_file}")

    # 5. Compute global top-20 words for Safe and Phishing
    min_count = 1 if args.debug else 5
    global_rankings: Dict[str, List[Dict[str, Any]]] = {"Safe Email": [], "Phishing Email": []}
    class_map = {0: "Safe Email", 1: "Phishing Email"}

    for c_id, c_name in class_map.items():
        word_aggs = []
        for word, score_list in global_word_scores[c_id].items():
            if len(score_list) >= min_count:
                mean_score = float(np.mean(score_list))
                total_score = float(np.sum(score_list))
                word_aggs.append({
                    "word": word,
                    "mean_score": round(mean_score, 4),
                    "total_score": round(total_score, 4),
                    "count": len(score_list),
                })
        word_aggs.sort(key=lambda item: -item["total_score"])
        global_rankings[c_name] = word_aggs[:20]

    save_json(global_rankings, global_exp_file)
    logger.info(f"Saved global explanations to: {global_exp_file}")

    return faithfulness_output


def main() -> None:
    args = parse_args()
    res = run_faithfulness(args)
    print("\n" + "=" * 70)
    print("FAITHFULNESS EVALUATION SUMMARY")
    print("=" * 70)
    for method, metrics in res["summary"].items():
        print(f"\nMethod: {method}")
        for k in ["1", "3", "5"]:
            c_mean = metrics["comprehensiveness_mean"].get(k, 0.0)
            c_std = metrics["comprehensiveness_std"].get(k, 0.0)
            s_mean = metrics["sufficiency_mean"].get(k, 0.0)
            s_std = metrics["sufficiency_std"].get(k, 0.0)
            print(f"  k={k}: Comp = {c_mean:.4f} (+/- {c_std:.4f}) | Suff = {s_mean:.4f} (+/- {s_std:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
