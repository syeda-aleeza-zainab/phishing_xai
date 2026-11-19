"""Robustness of artefact-backed scorers (no retraining).

Evaluates the three H4 perturbations (``truncated`` / ``urls_removed`` /
``typo_synonym``) on the canonical TEST split for models that can be
loaded without training:

* ``roberta`` — the fine-tuned checkpoint, via ``baselines.load_scorer``
  (``RobertaScorer`` wraps the existing ``score_text`` path, so this is
  the ``predict_proba`` wrapper ``run_baselines.py --robustness`` lacks
  for ``roberta_eval``).
* ``bilstm`` — a saved tuned-BiLSTM artefact (``best_bilstm.pt`` from
  ``run_baselines.py``), via ``baselines.load_baseline_model``.
* ``tfidf_lr`` / ``tfidf_svm`` — fitted on the REAL train split via
  ``load_scorer`` (same no-leakage protocol as the multimodal evals;
  no saved TF-IDF artefact exists in the results dataset).

The test set is touched exactly once per model here (the perturbed
evaluations reuse the same fitted/loaded model; thresholds are the
fixed 0.5 used by ``baselines.evaluate_robustness``). The clean
(unperturbed) F1 at 0.5 is reported alongside so each perturbation
reads as a drop from a known baseline.

NOT covered here (recorded as NOT RUN YET in the output when relevant):
DistilBERT, CNN and the bigger BiLSTM variant — no saved checkpoint
for any of them exists in the results dataset, so each would need a
full refit, which this script never does. ``run_baselines.py
--models tfidf_lr tfidf_svm cnn --robustness`` is the refit route for
the cheap models; DistilBERT would need a fresh fine-tune.

Writes ``robustness_loaded.json`` into the output directory and
returns 0 even when a model is NOT RUN YET (the reason is in the
JSON); the kernel wrapper checks the JSON instead of the exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phishing_xai.baselines import (  # noqa: E402
    compute_metrics_full,
    evaluate_robustness,
    load_scorer,
    load_splits,
)
from phishing_xai.utils import ensure_dir, get_logger, set_seed  # noqa: E402

logger = get_logger("phishing_xai.run_robustness_loaded")

MODEL_CHOICES = ("roberta", "bilstm", "tfidf_lr", "tfidf_svm")

#: Models this script deliberately does NOT attempt, and why (no saved
#: checkpoint exists in the results dataset; each would need a refit).
NOT_ATTEMPTED: Dict[str, str] = {
    "distilbert": "No saved DistilBERT checkpoint exists in the results "
                  "dataset (Stage D ran 1 seed but saved no artefact); "
                  "robustness would need a full fine-tune refit.",
    "cnn": "No saved CNN artefact exists in the results dataset (Stage D "
           "ran 1 seed but saved no artefact); use run_baselines.py "
           "--models cnn --robustness for the refit route.",
    "bilstm_big": "The bigger BiLSTM variant (hidden 256) was deliberately "
                  "NOT saved into the results dataset, so no artefact "
                  "exists to load; it would need a refit.",
}


def run_robustness_for_scorer(
    scorer: Any,
    texts: Sequence[str],
    labels: Sequence[int],
    seed: int = 42,
) -> Dict[str, Any]:
    """Clean + perturbed F1/accuracy/ROC-AUC for one loaded scorer.

    ``scorer`` is any :class:`TextScorer` (``predict_proba`` returning an
    ``(n, 2)`` array). Pure function over the scorer — unit-tested with a
    stub scorer, no model or data needed.
    """
    texts = [str(t) for t in texts]
    labels_arr = np.asarray(labels, dtype=int)
    y_prob = np.asarray(scorer.predict_proba(texts), dtype=float)[:, 1]
    clean = compute_metrics_full(labels_arr, y_prob)
    perturbed = evaluate_robustness(scorer, texts, labels, seed=seed)
    return {
        "clean_at_0_5": {
            "f1": clean["f1"],
            "accuracy": clean["accuracy"],
            "precision": clean["precision"],
            "recall": clean["recall"],
            "roc_auc": clean["roc_auc"],
            "n": clean["n"],
        },
        "robustness": perturbed,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Robustness (3 perturbations, test split) for loaded scorers; never retrains."
    )
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CHOICES),
                        default=["roberta", "bilstm"],
                        help="Scorers to evaluate (default: roberta bilstm)")
    parser.add_argument("--roberta_checkpoint", type=str, default="results/best_model.pt",
                        help="RoBERTa checkpoint (.pt)")
    parser.add_argument("--bilstm_checkpoint", type=str, default="results/bilstm/best_bilstm.pt",
                        help="Saved tuned-BiLSTM artefact (best_bilstm.pt)")
    parser.add_argument("--tfidf_checkpoint", type=str, default=None,
                        help="Optional saved TF-IDF artefact (best_tfidf_*.pt); without it, TF-IDF models are fitted on the real train split")
    parser.add_argument("--data_path", type=str, default="data/Phishing_Email.csv")
    parser.add_argument("--splits_path", type=str, default="results/splits.json")
    parser.add_argument("--output_dir", type=str, default="results/robustness_rerun")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    out_dir = ensure_dir(args.output_dir)
    results: Dict[str, Any] = {
        "seed": args.seed,
        "protocol": "Perturbations truncated / urls_removed / typo_synonym "
                    "on the canonical test split (results/splits.json); "
                    "threshold fixed at 0.5; no retraining; no threshold "
                    "fitting. Clean F1 at 0.5 reported alongside.",
        "models": {},
        "not_attempted": dict(NOT_ATTEMPTED),
    }

    try:
        splits = load_splits(args.data_path, args.splits_path, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: cannot load splits: {exc}", file=sys.stderr)
        return 2
    test_texts = splits["test"]["clean_text"].astype(str).tolist()
    test_labels = splits["test"]["label"].astype(int).tolist()
    results["n_test"] = len(test_texts)
    logger.info(f"Test split loaded: n={len(test_texts)}")

    for name in args.models:
        ckpt: Optional[str] = None
        source = "fitted on the real train split (no artefact)"
        if name == "roberta":
            ckpt = args.roberta_checkpoint
            source = f"loaded checkpoint {ckpt} (no retraining)"
        elif name == "bilstm":
            ckpt = args.bilstm_checkpoint
            source = f"loaded saved artefact {ckpt} (no retraining)"
        elif name in ("tfidf_lr", "tfidf_svm") and args.tfidf_checkpoint:
            ckpt = args.tfidf_checkpoint
            source = f"loaded saved artefact {ckpt}"
        try:
            start = time.perf_counter()
            scorer = load_scorer(name, checkpoint=ckpt)
            res = run_robustness_for_scorer(scorer, test_texts, test_labels, seed=args.seed)
            res["status"] = "ok"
            res["source"] = source
            res["seconds"] = float(time.perf_counter() - start)
            results["models"][name] = res
            logger.info(f"{name}: clean F1={res['clean_at_0_5']['f1']:.4f} "
                        f"robustness={ {k: round(v['f1'], 4) for k, v in res['robustness'].items()} }")
        except Exception as exc:  # noqa: BLE001 — record, do not abort the other models
            logger.warning(f"{name} robustness NOT RUN YET: {exc}")
            results["models"][name] = {"status": "NOT RUN YET", "reason": str(exc), "source": source}

    out_path = out_dir / "robustness_loaded.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")
    for name, entry in results["models"].items():
        if entry.get("status") == "ok":
            r = entry["robustness"]
            print(f"{name}: clean F1 {entry['clean_at_0_5']['f1']:.4f} | "
                  f"truncated {r['truncated']['f1']:.4f} | "
                  f"urls_removed {r['urls_removed']['f1']:.4f} | "
                  f"typo_synonym {r['typo_synonym']['f1']:.4f}")
        else:
            print(f"{name}: NOT RUN YET ({entry.get('reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
