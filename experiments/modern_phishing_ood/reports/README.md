# Modern-phishing OOD experiment — README

RoBERTa (fine-tuned) vs BiLSTM (tuned) on modern phishing emails.
Kaggle GPU (T4) execution, November 2026. **Read the caveat first.**

## The caveat (travels with every result)

The primary set (PhishNChips v5.2) is a **modern URL-grounded
synthetic-stimuli OOD set, NOT real-world inbox mail**: phishing
labels are verified via PhishTank / OpenPhish / GitHub phishing DB
URLs, but the email prose is synthetic stimulus text. Report results
as "modern URL-grounded synthetic-stimuli OOD", never as "real-world
2026 phishing". No verified real-world 2023+ phishing + legitimate
email-body corpus was found (see `reports/dataset_provenance.md`).

## Headline result

* **Primary (PhishNChips, n=2,000, threshold 0.5):** RoBERTa P 0.9333 /
  R 0.0280 / F1 0.0544 / ROC-AUC 0.7787; BiLSTM P 0.8594 / R 0.0550 /
  F1 0.1034 / ROC-AUC 0.6801. Both collapse on recall; RoBERTa ranks
  better, BiLSTM has the better F1 at 0.5. H1 rejected here.
* **Synthetic track:** RoBERTa decisively better everywhere — Frontiers
  LLM R 0.9394 vs 0.4787; human R 0.8402 vs 0.7226; BEC R 0.3772 vs
  0.2102 (all McNemar p≈0, paired CIs exclude 0). H1 holds here.
* **Thresholds fitted on validation (≈0.99) detect ~nothing OOD.**
* **Final verdict:** MIXED — H1 rejected on the primary set,
  supported on the synthetic track. See
  `reports/final_comparative_report.md`.

## Layout

```
experiments/modern_phishing_ood/
├── datasets/            # OOD sets (accepted only; rejected sets are NOT here)
│   ├── phishnchips_core_emails.csv   # primary, 2,000
│   ├── phishfuzzer_seeds.json        # secondary, Manual-300
│   ├── zenodo_frontiers/data/{llm,human}_corpus_sampled.csv
│   └── yoadjei_bec/synthetic_emails{,_poisoned}.csv
├── scripts/
│   ├── run_ood_eval.py        # the eval (Kaggle GPU)
│   ├── ood_common.py          # import-light helpers (hashing, MinHash, thresholds, loaders)
│   ├── compute_ood_stats.py   # McNemar + bootstrap CIs + Tables A–F (from saved predictions)
│   ├── make_ood_plots.py      # the 9 plots
│   └── audit_leakage_local.py # CPU-only leakage audit driver
├── results/             # executed JSONs + per-set metrics (Kaggle runs)
├── predictions/         # per-email predictions, one CSV per set per model
├── plots/               # 9 PNGs
├── logs/                # inspection.md, kernel logs (v1/v2 errors, K2, K3)
└── reports/             # methodology, leakage_audit, evaluation_results (Tables A–F),
                         # statistical_report, robustness_report, efficiency_report,
                         # final_comparative_report, dataset_provenance, this README
```

## Reproduce — Kaggle (full, GPU)

Prerequisites: Kaggle CLI with the project account; datasets (all
private, already created):
`syedaaleezazainab/phishing-xai-repo-ah-6b1` (v13),
`syedaaleezazainab/phishing-xai-checkpoint` (v1),
`syedaaleezazainab/phishing-xai-results` (v7),
`syedaaleezazainab/phishing-xai-ood-modern` (v2).
Kernels (in `~/workspace/kaggle_kernels/`):

```bash
cd ~/workspace/kaggle_kernels/ood_primary
kaggle kernels push -p .        # phishing-xai-ood-primary-eval (v3 ran clean)
cd ../ood_synthetic
kaggle kernels push -p .        # phishing-xai-ood-synthetic-track
cd ../ood_robustness
kaggle kernels push -p .        # phishing-xai-ood-robustness
kaggle kernels output <slug> -p <dir>   # per kernel, then copy
# experiments/modern_phishing_ood/{results,predictions} into the repo tree
```

Each kernel verifies the canonical splits (11,215/2,803/3,495), runs
the 23 OOD unit tests, then `run_ood_eval.py`. Kernel 1 needed 3
versions (OOD mount path, then a `.parent` bug — see
`logs/kernel1_v1_error.log`, `logs/kernel1_v2_error.log`).

Direct eval command (inside a kernel or any machine with the
checkpoints + torch + GPU):

```bash
python experiments/modern_phishing_ood/scripts/run_ood_eval.py \
  --sets phishnchips phishfuzzer --models roberta bilstm \
  --ood_dir <ood-dataset-root> \
  --data_path data/Phishing_Email.csv --splits_path results/splits.json \
  --roberta_checkpoint results/best_model.pt \
  --bilstm_checkpoint results/bilstm/best_bilstm.pt \
  --output_dir experiments/modern_phishing_ood/results \
  --predictions_dir experiments/modern_phishing_ood/predictions \
  --floors 0.85 0.90 0.95 --latency_n_gpu 200 --latency_n_cpu 100
# add --robustness for Kernel 3; use --sets frontiers_llm frontiers_human bec bec_poisoned for Kernel 2
```

## Reproduce — local (CPU-only, no checkpoints)

```bash
# 23 OOD unit tests (need scikit-learn + pytest; torch NOT needed)
python3 -m pytest tests/test_ood_eval.py tests/test_ood_stats.py -q
# Leakage audit only (no GPU, ~4 min; training texts via pandas fallback)
python3 experiments/modern_phishing_ood/scripts/audit_leakage_local.py
# Stats + tables from saved predictions (needs scikit-learn + scipy)
python3 experiments/modern_phishing_ood/scripts/compute_ood_stats.py \
  --predictions_dir experiments/modern_phishing_ood/predictions \
  --results_dir experiments/modern_phishing_ood/results \
  --out_dir experiments/modern_phishing_ood/results
# Plots (needs matplotlib)
python3 experiments/modern_phishing_ood/scripts/make_ood_plots.py \
  --results_dir experiments/modern_phishing_ood/results \
  --predictions_dir experiments/modern_phishing_ood/predictions \
  --out_dir experiments/modern_phishing_ood/plots
```

The full eval cannot run locally: both checkpoints are Kaggle-only
(RoBERTa 450 MB, BiLSTM 25.6 MB) and the eval needs torch + a GPU for
sane runtimes. Never retrain to fill the gap; never use
`--allow_pretrained_fallback` for evidence.

## Hard rules (never bypassed)

* Never modify `src/phishing_xai/model.py` or `src/phishing_xai/faithfulness.py`.
* Never overwrite existing results/checkpoints. New outputs only.
* Never retrain. A missing BiLSTM artefact is a loud failure, never a
  silent refit (`load_scorer` would silently refit — the script guards
  with `_is_baseline_artefact` + `load_baseline_model`).
* Splits from `results/splits.json` via `load_splits`; never regenerate.
* Thresholds fitted on the original validation split only; OOD labels
  never used for fitting. Cross-checked against a pure-numpy mirror
  (max abs diff 0.0 on Kaggle).

## Known issues / gaps (recorded, not hidden)

* `thresholds.json` in Kernel 1/2 outputs holds only the last model
  (BiLSTM) — overwritten per model. Both models' thresholds are in
  `ood_metrics_combined_kernel{1,2}.json`. Fixed in the working copy
  (per-model `thresholds_<model>.json`).
* BEC contrast unchanged-row check used a 1e-9 tolerance; GPU-scored
  BiLSTM shows float nondeterminism up to ~4.4e-08 on identical texts.
  Tolerance is 1e-6 in the fixed copy; probabilities unchanged.
* Kernel 1 v3's log file could not be downloaded (Kaggle read timeout
  ×2); all v3 numbers are in the JSONs. Kernel 2/3 logs are saved.
* BiLSTM GPU latency is labelled `"gpu?"` in the JSONs (no `.device`
  attribute on the scorer); the scorer was CUDA-loaded — see
  `efficiency_report.md`.
* Table C (temporal) is NOT RUN YET — no per-email timestamps exist.
* Stats used local mirrors of the repo helpers (torch unavailable
  locally); same formulas, unit-tested. A Kaggle-side re-run of
  `compute_ood_stats.py` with the repo helpers would be a clean
  confirmation.
* No TF-IDF / DistilBERT / CNN OOD comparison was run (natural next
  experiment — TF-IDF+SVC beat RoBERTa on in-distribution precision).
