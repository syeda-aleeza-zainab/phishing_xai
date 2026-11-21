"""Latency + explanation time (never report a CPU number as GPU).

Measures, on the SAME canonical test split and the SAME trained
checkpoints used everywhere else (no retraining):

* Inference latency, median of 200 emails at batch size 1, for RoBERTa,
  the tuned BiLSTM and DistilBERT — separately on CPU and on GPU, each
  labelled with the device it was actually measured on. TF-IDF latency is NOT
  re-measured here: the Stage D ladder file already holds it
  (``results/baselines/baseline_results.json``: TF-IDF+LR CPU 1.08 ms,
  TF-IDF+SVC CPU 8.11 ms, 3-seed means; sklearn has no GPU path).
* Explanation time per email for RoBERTa (LITA: LIME + Transformers-
  Interpret, ``alpha=0.5``), DistilBERT (LITA, same path via the
  ``DistilBertScorer``) and BiLSTM (LITA-IG: LIME + integrated
  gradients) via the existing ``baselines.explain_word_scores`` path —
  the same path ``scripts/run_explain.py`` uses, with the same
  ``lime_samples=1000`` default and the same 10 test emails Stage C
  explained (``test_df.head(10)`` in canonical split order:
  email_1/17/18/25/27/29/30/37/42/44). Median + mean/std over the
  explained emails, plus the per-email seconds.

Writes ``latency_explain_time.json`` into the output directory and
returns 0 even when a section is NOT RUN YET (the reason is in the
JSON); the kernel wrapper checks the JSON instead of the exit code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phishing_xai.baselines import (  # noqa: E402
    build_distilbert_wrapper,
    explain_word_scores,
    load_baseline_model,
    load_distilbert_model,
    load_scorer,
    load_splits,
    measure_latency,
)
from phishing_xai.data import deduplicate_data, load_data  # noqa: E402
from phishing_xai.utils import ensure_dir, get_logger, load_json, set_seed  # noqa: E402

logger = get_logger("phishing_xai.run_latency_explain_time")


def _summarise_seconds(values: Sequence[float]) -> Dict[str, Any]:
    """Median / mean / std / n for a list of per-email seconds."""
    vals = [float(v) for v in values]
    if not vals:
        return {"median": None, "mean": None, "std": None, "n": 0, "per_email": []}
    return {
        "median": float(statistics.median(vals)),
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "per_email": [round(v, 4) for v in vals],
    }


def time_explanations(
    scorer: Any,
    texts: Sequence[Dict[str, Any]],
    *,
    lime_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Time ``explain_word_scores`` once per email; return the summary.

    ``texts`` are ``{"id", "text"}`` dicts (the same 10 Stage C emails).
    The timed call is exactly the explain path ``run_explain.py`` uses
    (LIME + the scorer's second method), so the seconds below are the
    LITA / LITA-IG explanation cost, not a LIME-only cost.
    """
    seconds: List[float] = []
    method_name: Optional[str] = None
    for item in texts:
        start = time.perf_counter()
        _words, _lime, _second, method_name = explain_word_scores(
            scorer, str(item["text"]), lime_samples=lime_samples, seed=seed,
        )
        seconds.append(time.perf_counter() - start)
        logger.info(f"Explained {item['id']} in {seconds[-1]:.2f}s (method={method_name})")
    out = _summarise_seconds(seconds)
    out["method"] = method_name
    out["lime_samples"] = int(lime_samples)
    out["email_ids"] = [item["id"] for item in texts]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Measure RoBERTa / tuned-BiLSTM latency (CPU + GPU, median of 200, batch 1) and explanation time per email."
    )
    parser.add_argument("--roberta_checkpoint", type=str, default="results/best_model.pt")
    parser.add_argument("--bilstm_checkpoint", type=str, default="results/bilstm/best_bilstm.pt")
    parser.add_argument("--distilbert_checkpoint", type=str,
                        default="results/distilbert/seed_42/best_distilbert.pt",
                        help="Saved DistilBERT artefact (best_distilbert.pt, best-validation seed)")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--output_dir", type=str, default="results/rerun_latency")
    parser.add_argument("--n_latency", type=int, default=200,
                        help="Emails timed for latency (batch size 1; default 200, same as baselines.measure_latency)")
    parser.add_argument("--n_explain", type=int, default=10,
                        help="Test emails explained for explanation time (default 10, the Stage C set)")
    parser.add_argument("--lime_samples", type=int, default=1000,
                        help="LIME perturbation samples per explanation (default 1000, same as run_explain.py)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    results: Dict[str, Any] = {
        "seed": args.seed,
        "n_latency": args.n_latency,
        "batch_size": 1,
        "cuda_available": bool(cuda),
        "gpu": gpu_name,
        "protocol": "Latency: baselines.measure_latency (median of n_latency "
                    "single-email predict_proba calls after warmup), CPU and "
                    "GPU measured separately on the same model and labelled "
                    "by device. Explanation time: baselines.explain_word_scores "
                    "timed once per email (LIME + second method), same 10 "
                    "test emails Stage C explained, lime_samples=1000.",
        "latency_ms_per_email": {},
        "explanation_seconds_per_email": {},
    }

    try:
        splits = load_splits(args.data_path, args.splits_path, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: cannot load splits: {exc}", file=sys.stderr)
        return 2
    test_texts = splits["test"]["clean_text"].astype(str).tolist()
    results["n_test"] = len(test_texts)
    latency_texts = test_texts[: args.n_latency]

    # --- RoBERTa latency: GPU scorer + CPU scorer (separate loads) ---
    for device_label, device in (("gpu", "cuda"), ("cpu", "cpu")):
        if device_label == "gpu" and not cuda:
            results["latency_ms_per_email"]["roberta_gpu"] = {
                "status": "NOT RUN YET", "reason": "No CUDA device in this environment.",
            }
            continue
        try:
            scorer = load_scorer(
                "roberta", checkpoint=args.roberta_checkpoint,
                device=torch.device(device),
            )
            timed = measure_latency(scorer.predict_proba, latency_texts, n=args.n_latency)
            results["latency_ms_per_email"][f"roberta_{device_label}"] = {
                "status": "ok", "device": device_label,
                "checkpoint": args.roberta_checkpoint, **timed,
            }
            logger.info(f"RoBERTa latency ({device_label}): median "
                        f"{timed['latency_ms_per_email_median']:.2f} ms/email")
        except Exception as exc:  # noqa: BLE001 — record, do not abort the other sections
            logger.warning(f"RoBERTa latency ({device_label}) NOT RUN YET: {exc}")
            results["latency_ms_per_email"][f"roberta_{device_label}"] = {
                "status": "NOT RUN YET", "reason": str(exc), "device": device_label,
            }

    # --- Tuned BiLSTM latency: one load, set_device to switch ---
    try:
        bilstm = load_baseline_model(args.bilstm_checkpoint)
        for device_label, device in (("gpu", "cuda"), ("cpu", "cpu")):
            if device_label == "gpu" and not cuda:
                results["latency_ms_per_email"]["bilstm_gpu"] = {
                    "status": "NOT RUN YET", "reason": "No CUDA device in this environment.",
                }
                continue
            try:
                bilstm.set_device(torch.device(device))
                timed = measure_latency(bilstm.predict_proba, latency_texts, n=args.n_latency)
                results["latency_ms_per_email"][f"bilstm_{device_label}"] = {
                    "status": "ok", "device": device_label,
                    "checkpoint": args.bilstm_checkpoint, **timed,
                }
                logger.info(f"BiLSTM latency ({device_label}): median "
                            f"{timed['latency_ms_per_email_median']:.2f} ms/email")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"BiLSTM latency ({device_label}) NOT RUN YET: {exc}")
                results["latency_ms_per_email"][f"bilstm_{device_label}"] = {
                    "status": "NOT RUN YET", "reason": str(exc), "device": device_label,
                }
        bilstm.set_device(torch.device("cpu"))
    except Exception as exc:  # noqa: BLE001 — artefact missing/unreadable
        logger.warning(f"BiLSTM latency NOT RUN YET: {exc}")
        for device_label in ("gpu", "cpu"):
            results["latency_ms_per_email"][f"bilstm_{device_label}"] = {
                "status": "NOT RUN YET", "reason": str(exc), "device": device_label,
            }

    # --- DistilBERT latency: one load, set_device to switch (as BiLSTM) ---
    try:
        _d_model, _d_tokenizer, _d_payload = load_distilbert_model(args.distilbert_checkpoint)
        distilbert = build_distilbert_wrapper(_d_model, _d_tokenizer)
        for device_label, device in (("gpu", "cuda"), ("cpu", "cpu")):
            if device_label == "gpu" and not cuda:
                results["latency_ms_per_email"]["distilbert_gpu"] = {
                    "status": "NOT RUN YET", "reason": "No CUDA device in this environment.",
                }
                continue
            try:
                distilbert.set_device(torch.device(device))
                timed = measure_latency(distilbert.predict_proba, latency_texts, n=args.n_latency)
                results["latency_ms_per_email"][f"distilbert_{device_label}"] = {
                    "status": "ok", "device": device_label,
                    "checkpoint": args.distilbert_checkpoint, **timed,
                }
                logger.info(f"DistilBERT latency ({device_label}): median "
                            f"{timed['latency_ms_per_email_median']:.2f} ms/email")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"DistilBERT latency ({device_label}) NOT RUN YET: {exc}")
                results["latency_ms_per_email"][f"distilbert_{device_label}"] = {
                    "status": "NOT RUN YET", "reason": str(exc), "device": device_label,
                }
        distilbert.set_device(torch.device("cpu"))
    except Exception as exc:  # noqa: BLE001 — artefact missing/unreadable
        logger.warning(f"DistilBERT latency NOT RUN YET: {exc}")
        for device_label in ("gpu", "cpu"):
            results["latency_ms_per_email"][f"distilbert_{device_label}"] = {
                "status": "NOT RUN YET", "reason": str(exc), "device": device_label,
            }

    # --- Explanation time: same 10 test emails Stage C explained ---
    try:
        df = load_data(args.data_path)
        df = deduplicate_data(df, text_col="clean_text", label_col="label")
        splits_data = load_json(args.splits_path)
        test_df = df.iloc[splits_data.get("test_indices", [])]
        selected = test_df.head(args.n_explain)
        explain_items = [
            {"id": f"email_{idx}", "text": str(row["clean_text"])}
            for idx, row in selected.iterrows()
        ]
        results["explain_email_ids"] = [item["id"] for item in explain_items]
        if cuda:
            roberta_scorer = load_scorer(
                "roberta", checkpoint=args.roberta_checkpoint,
                device=torch.device("cuda"),
            )
            results["explanation_seconds_per_email"]["roberta_lita"] = time_explanations(
                roberta_scorer, explain_items,
                lime_samples=args.lime_samples, seed=args.seed,
            )
        else:
            results["explanation_seconds_per_email"]["roberta_lita"] = {
                "status": "NOT RUN YET", "reason": "No CUDA device in this environment.",
            }
        bilstm_scorer = load_scorer("bilstm", checkpoint=args.bilstm_checkpoint)
        results["explanation_seconds_per_email"]["bilstm_lita_ig"] = time_explanations(
            bilstm_scorer, explain_items,
            lime_samples=args.lime_samples, seed=args.seed,
        )
        # DistilBERT LITA (LIME + Transformers-Interpret / Captum) via the
        # DistilBertScorer: explain_word_scores takes the LITA branch for
        # it, exactly as for RoBERTa.
        try:
            distilbert_scorer = load_scorer(
                "distilbert", checkpoint=args.distilbert_checkpoint,
            )
            results["explanation_seconds_per_email"]["distilbert_lita"] = time_explanations(
                distilbert_scorer, explain_items,
                lime_samples=args.lime_samples, seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 — artefact missing/unreadable
            logger.warning(f"DistilBERT explanation timing NOT RUN YET: {exc}")
            results["explanation_seconds_per_email"]["distilbert_lita"] = {
                "status": "NOT RUN YET", "reason": str(exc),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Explanation timing NOT RUN YET: {exc}")
        results["explanation_seconds_per_email"]["error"] = {
            "status": "NOT RUN YET", "reason": str(exc),
        }

    out_path = out_dir / "latency_explain_time.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
    for key, entry in results["latency_ms_per_email"].items():
        if entry.get("status") == "ok":
            print(f"{key}: median {entry['latency_ms_per_email_median']:.2f} ms/email "
                  f"(mean {entry['latency_ms_per_email_mean']:.2f}, n={entry['n_timed']})")
        else:
            print(f"{key}: NOT RUN YET ({entry.get('reason')})")
    for key, entry in results["explanation_seconds_per_email"].items():
        if isinstance(entry, dict) and entry.get("median") is not None:
            print(f"{key}: median {entry['median']:.2f} s/email "
                  f"(mean {entry['mean']:.2f}, std {entry['std']:.2f}, n={entry['n']}, method={entry.get('method')})")
        else:
            print(f"{key}: NOT RUN YET ({entry.get('reason') if isinstance(entry, dict) else entry})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
