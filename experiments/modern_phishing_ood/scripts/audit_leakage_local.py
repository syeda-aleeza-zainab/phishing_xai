"""Local (CPU-only) leakage audit for the OOD sets.

Reuses ``run_ood_eval.load_ood_set`` / ``load_training_texts`` /
``run_leakage_audit`` — the same code the Kaggle kernel runs — so the
numbers here and on Kaggle are produced by one code path. The only
difference: locally the training texts come from the pandas fallback
(``phishing_xai.data`` needs torch, which is not installed here), while
on Kaggle the repo ``load_data`` + ``deduplicate_data`` path is used.
Both feed the same hash functions, and the Kaggle run re-reports the
audit with the repo path.

Writes ``results/leakage_audit_local.json`` and prints the table.
Usage (repo root)::
    python3 experiments/modern_phishing_ood/scripts/audit_leakage_local.py \
        [--minhash_train_sample N] [--sets phishnchips phishfuzzer ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "modern_phishing_ood"
sys.path.insert(0, str(EXP_DIR / "scripts"))

import run_ood_eval as roe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Local CPU leakage audit.")
    parser.add_argument("--sets", nargs="+", default=list(roe.SET_LOADERS))
    parser.add_argument("--minhash_train_sample", type=int, default=0,
                        help="0 = full training set (slow); else seed-42 subsample")
    parser.add_argument("--out", type=str,
                        default=str(EXP_DIR / "results" / "leakage_audit_local.json"))
    args = parser.parse_args()

    ood_dir = EXP_DIR / "datasets"
    frames = {}
    for name in args.sets:
        frames[name] = roe.load_ood_set(name, ood_dir)
        f = frames[name]
        print(f"[data] {name}: n={len(f)} phish={(f['label'] == 1).sum()} "
              f"legit={(f['label'] == 0).sum()} other={(f['label'] < 0).sum()}", flush=True)
    t0 = time.time()
    train_texts, train_phish, _ = roe.load_training_texts(
        REPO_ROOT / "data" / "Phishing_Email.csv")
    print(f"[leakage] training texts loaded in {time.time() - t0:.1f}s: "
          f"n={len(train_texts)}", flush=True)
    t0 = time.time()
    audit = roe.run_leakage_audit(frames, train_texts, train_phish,
                                  minhash_train_sample=args.minhash_train_sample)
    audit["runtime_seconds"] = time.time() - t0
    audit["minhash_train_sample_arg"] = args.minhash_train_sample
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[done] wrote {out} in {audit['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
