# Methodology — modern-phishing OOD experiment

## Question and hypotheses

**Question:** on modern phishing emails, does the fine-tuned RoBERTa
generalise better than the tuned BiLSTM, or is there no significant
advantage?

* **H1:** RoBERTa generalises better than BiLSTM on modern phishing
  (higher precision at the recall floors, or better ranking).
* **H0:** no significant advantage of RoBERTa over BiLSTM.

H1 is rejectable: if the evidence does not support it, the report says
so (see `final_comparative_report.md`).

## Models (loaded, never retrained)

| Model | Artefact | Params | Size (this run) | Provenance |
|---|---|---|---|---|
| RoBERTa | `best_model.pt` from Kaggle dataset `syedaaleezazainab/phishing-xai-checkpoint` v1 | 124,647,170 | 498.7 MB | Stage B tuned retrain (batch 8 / lr 1e-5 / epochs 3, best val F1 0.9807); test P 0.9690 / R 0.9932 / F1 0.9809 / acc 0.9854 / ROC-AUC 0.9986 (n=3,495) |
| BiLSTM | `best_bilstm.pt` from Kaggle dataset `syedaaleezazainab/phishing-xai-results` v7 (`results/bilstm/`) | 6,236,034 | 25.6 MB | Tuned config verified from the artefact itself: emb_dim 200, hidden 64, layers 2, dropout 0.5, max_len 128, max_vocab 30,000; test P 0.9719 ± 0.0015 / R 0.9428 / F1 0.9571 (3 seeds) |

Loading (see `logs/inspection.md` §10.2):
* RoBERTa via `phishing_xai.baselines.load_scorer("roberta", checkpoint=...)`,
  which wraps `run_multimodal_predict.load_trained_roberta` and scores
  with `score_text` (truncate to a 512-token window; empty text → Safe
  with P(phishing) = 0.0). No `--allow_pretrained_fallback`: a missing
  checkpoint is a loud failure, never an untrained run.
* BiLSTM via `load_baseline_model` after an explicit
  `_is_baseline_artefact` guard. This guard exists because
  `load_scorer("bilstm", ...)` silently REFITS a fresh BiLSTM on the
  train split when the artefact path is missing or is not a baseline
  artefact — the script refuses instead. The artefact's stored config
  and a smoke prediction are logged at load time.
* Both models receive text cleaned with `phishing_xai.data.clean_text`
  (the same cleaning applied at training time). `score_text` itself does
  not clean, so the script cleans before scoring.
* `--ood_csv` in `run_baselines.py` was NOT used: it skips RoBERTa.

## Data

Primary OOD set: **PhishNChips v5.2** (2,000; 1,000 phish / 1,000
legit). Secondary: **PhishFuzzer Manual-300** (binary slice 103 phish /
102 valid; 95 spam rows scored and reported separately, never in binary
metrics). Synthetic track (separate, never mixed): **Frontiers 2026**
LLM phishing 4,986 + human phishing 5,000 (old corpora) and **Adversarial
BEC** 4,181 clean (+ poisoned file), each phishing-only set paired with
the same PhishNChips legit 1,000 — the pairing is recorded in the
outputs, never hidden. Full provenance, licences and rejections:
`reports/dataset_provenance.md`.

**Mandatory caveat:** PhishNChips is a modern URL-grounded
synthetic-stimuli OOD set (phishing labels verified via PhishTank /
OpenPhish / GitHub phishing DB URLs; email prose is synthetic stimulus
text, not real inbox mail). Results are reported as such, never as
"real-world 2026 phishing". No verified real-world 2023+ phishing +
legitimate email-body corpus was found.

## Leakage controls

Before scoring, every OOD set is checked against the training texts
(see `reports/leakage_audit.md` for the numbers):
exact normalized-MD5 overlap, body-only exact overlap, first-200-char
near-dup-hash overlap, MinHash near-duplicates (word 5-shingles,
64 permutations, seed 42; per-OOD-text maximum estimated Jaccard vs any
training text; full pairwise comparison), and URL overlap vs training
phishing URLs. The primary set is clean on every check; the two
flagged rows (PhishFuzzer Manual No. 37 body-only; 6 Frontiers-human
MinHash ≥ 0.9) are kept and flagged, not silently dropped.

## Splits and thresholds

* Splits are loaded from `results/splits.json` via
  `baselines.load_splits` (train 11,215 / val 2,803 / test 3,495);
  never regenerated (sklearn-version split instability).
* Thresholds are fitted on the ORIGINAL VALIDATION split only, with
  `baselines.fit_threshold_for_recall` at recall floors 0.85 / 0.90 /
  0.95, plus the fixed 0.5 default. OOD labels are never used for
  fitting. The pure-numpy mirror in `ood_common.fit_thresholds_for_floors`
  cross-checks the repo fitter on Kaggle (max abs threshold difference
  0.0 in Kernel 1). Unit tests (`tests/test_ood_eval.py`) prove OOD
  labels cannot change the fitted thresholds.
* No threshold is fitted on OOD data at any point. The fitted
  thresholds are applied unchanged to every OOD set — including the
  finding that they transfer catastrophically badly to PhishNChips
  (see `evaluation_results.md`).

## Metrics and statistics

* Per model per set at 0.5 and at each fitted threshold: accuracy,
  precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, FPR, FNR
  (`baselines.compute_metrics_full`); cross-checked against the
  pure-numpy `ood_common.metrics_lite` on the same probabilities
  (max abs diff 0.0 except PR-AUC ≤ 8e-05, from average-precision
  implementation detail).
* Per-category breakdowns at 0.5 (PhishNChips datasource / strategy /
  URL category; Frontiers model; PhishFuzzer type).
* Latency: median ms/email at batch 1 via `baselines.measure_latency`
  (n=200 GPU, n=100 CPU for RoBERTa; n=200 GPU, n=100 CPU for BiLSTM —
  see `efficiency_report.md` for the device caveat on the BiLSTM GPU
  number), plus params and checkpoint file size.
* Robustness (Kernel 3, primary set only): the 3 H4 perturbations
  (`truncated` / `urls_removed` / `typo_synonym`) via
  `baselines.evaluate_robustness`, seed 42, no retraining.
* Statistics (`scripts/compute_ood_stats.py`, from saved predictions
  only): paired McNemar on identical examples (ERRORS ONLY — not a
  precision test), paired bootstrap 95% CIs for precision/recall/F1
  and for ROC/PR-AUC (1,000 resamples, seed 42), paired-difference CIs
  (RoBERTa − BiLSTM). Repo helpers are used when importable; on
  CPU-only machines local mirrors of the same formulas are used and
  labelled as mirrors. Checkpoints are single, so no multi-seed
  variability is claimed (n_seeds = 1, stated in every table).
* Expanded statistics (`scripts/compute_ood_stats_expanded.py`, run
  locally 2026-11-20): all 6 model pairs × 6 sets — paired confusion
  matrices, prediction-agreement and error tables, McNemar (errors
  only), paired bootstrap 95% CIs for P/R/F1 and for ROC-AUC/PR-AUC
  differences (1,000 resamples, seed 42). Multiple-comparison
  correction: **Holm-Bonferroni within each set** (primary family),
  with a **global Holm across all sets** reported as a sensitivity
  check and a Bonferroni reference. Significance = Holm-adjusted
  p < 0.05 or a paired CI excluding 0; numerical improvement without
  either is reported as such, never as significant. The primary-set
  McNemar (p=0.0359) is significant within-set but not under the
  global Holm (adjusted p=0.0718) — reported both ways.
* Hypotheses (Task 8, evaluated independently): **H1a (ranking)** —
  supported iff the paired ROC-AUC AND PR-AUC difference CIs both
  exclude 0 in RoBERTa's favour (each reported separately; partial
  support possible). **H1b (operational detection)** — supported iff
  RoBERTa meets the predefined recall floor 0.85 where BiLSTM does
  not, at threshold 0.5 and at validation-fitted thresholds (floors
  0.85/0.90/0.95); rejected if neither meets it. ROC-AUC/PR-AUC are
  never used as deployment-readiness evidence.
* Verdict rule (repo, fixed 2026-11-19): a precision gain is
  "significant" only if the gap is positive AND McNemar p < 0.05.
  A negative gap is never a gain; a positive gap with p ≥ 0.05 is never
  called significant.

## Execution

* Script kernels on Kaggle (GPU T4), code in
  `~/workspace/kaggle_kernels/ood_primary|ood_synthetic|ood_robustness/`:
  Kernel 1 = primary + secondary; Kernel 2 = synthetic track;
  Kernel 3 = robustness (primary only). Each kernel copies the repo
  from dataset `syedaaleezazainab/phishing-xai-repo-ah-6b1` (v13:
  adds `run_ood_eval.py`, `ood_common.py`, `audit_leakage_local.py`,
  `tests/test_ood_eval.py`), attaches the checkpoint, results and OOD
  datasets (`syedaaleezazainab/phishing-xai-ood-modern` v2), verifies
  the canonical splits, runs the 23 OOD unit tests, then the eval.
* Kernel 1 needed 3 versions: v1 failed on the OOD mount path
  (datasets live under `/kaggle/input/datasets/<owner>/<slug>`);
  v2 failed on a `.parent` bug in the same lookup (took the parent of
  the dataset directory); v3 COMPLETED. Logs:
  `logs/kernel1_v1_error.log`, `logs/kernel1_v2_error.log`.
* Known script bugs found during/after the kernels (fixed in the working
  copy afterwards; the executed numbers are unaffected because the full
  values survive in `ood_metrics_combined.json`):
  - `thresholds.json` is overwritten per model, so it holds only the
    last model (BiLSTM); both models' thresholds are preserved in
    `ood_metrics_combined.json`. Fixed to per-model
    `thresholds_<model>.json`.
  - The BEC contrast's unchanged-row check used a 1e-9 tolerance;
    GPU-scored BiLSTM shows float nondeterminism up to ~4.4e-08 on
    identical texts, so the check falsely reported "not identical".
    Tolerance is 1e-6 in the fixed copy; the underlying probabilities
    are unchanged.
* Kernel 1 v3's log file could not be downloaded (Kaggle read timeout
  after 300 s, twice); v1/v2 error logs are saved, and all v3 numbers
  are in the JSONs. Kernel 2 and 3 logs are saved.
* Kernel 4 — TF-IDF OOD (CPU-only, 2026-11-20, COMPLETED): `syedaaleezazainab/phishing-xai-ood-tf-idf-baselines`, script
  in `~/workspace/kaggle_kernels/ood_tfidf/`, repo dataset v14
  (uploaded with `--dir-mode zip`; two earlier root-only versions were
  superseded). TF-IDF+SVC and TF-IDF+LR on all six sets with the same
  examples/labels as the neural models. No saved TF-IDF artefact
  exists, so both models are refit on the canonical TRAIN split only,
  seed-42 Stage A configs (LR: C=10.0, word (1,2), max_features=None;
  SVC: C=5.0, char (3,6), max_features=None) via the existing
  `build_tfidf_lr` / `build_tfidf_svm` pipeline; thresholds fitted on
  validation only; never fitted or tuned on OOD. New prediction files
  (`<set>_tfidf_svm_predictions.csv`, `<set>_tfidf_lr_predictions.csv`);
  the 12 originals are never overwritten. GPU latency NOT MEASURED for
  TF-IDF (no GPU implementation). Reproduction: `README.md` in this
  directory gives the exact commands; `run_ood_eval.py --help` lists every flag. Local runs
  need the checkpoints (Kaggle-only) and torch/sklearn; the leakage
  audit and the unit tests run CPU-only without them.
