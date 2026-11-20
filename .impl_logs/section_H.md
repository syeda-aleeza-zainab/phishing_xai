# Section H — Baseline & Complexity-Justification Study: Change Log

Author: Section H subagent. Base commit: `5d86e95`. Work done 2026-11-19.
Do NOT edit `decision.md` or `docs/PROJECT_LOG.md` directly — the coordinator merges this log.

## User decisions recorded

1. **Precision is the priority focus** (user, 2026-11-19, "mera priority focus precision hai").
   Primary metric is phishing-class PRECISION (minimise false positives / low FPR).
   F1/accuracy/recall are reported everywhere but never used alone for selection.
   Applied as: models ranked by precision first in `run_baselines.py` outputs and the
   complexity-report table; verdict headline judges the PRECISION gain (F1 branch kept
   as a secondary statement); recommended deployment model = highest-precision model
   subject to recall floor 0.85 on validation aggregates; precision@recall-0.90 view
   with thresholds fit on VALIDATION only; bootstrap precision CIs alongside F1 CIs;
   random search and BiLSTM/CNN early stopping select on validation precision subject
   to recall >= 0.85. Recall is reported everywhere; nothing about recall was dropped.
   If no model meets recall >= 0.85, the report states that explicitly (best recall among
   models + chosen model's recall) instead of pretending the floor held.
2. **Fairness constraints (from the brief, recorded here per H10):** baselines use
   identical splits (`results/splits.json`) and identical `clean_text` cleaning;
   vocab/TF-IDF/tokenizer fit on TRAIN only; train-only synonym augmentation via
   `augment.py`; hyperparameter search ~8 trials scored on VALIDATION only; never tune
   on test; single final test evaluation per model.

## Files created (no existing file was modified)

| File | What it is |
|---|---|
| `src/phishing_xai/baselines.py` | Ladder + metrics/stats/faithfulness/multimodal-hook library (see below) |
| `scripts/run_baselines.py` | CLI runner for the ladder |
| `scripts/run_complexity_report.py` | Report generator (md + json + 2 plots), code-written verdict |
| `tests/test_baselines.py` | 34 tests, fake tiny fixtures only |

`src/phishing_xai/model.py` and `src/phishing_xai/faithfulness.py` were NOT modified
(read-only, as specified). Nothing was pushed. No RoBERTa retraining anywhere.

## Decisions and reasons (per change)

- **Splits semantics.** `results/splits.json` indices are `email_id` = positions in the
  deduplicated frame (`load_data` + `deduplicate_data`, the order `scripts/make_splits.py`
  uses). `load_splits()` rebuilds splits that way and validates sizes + label
  distributions against the manifest summary, raising `ValueError` on mismatch instead
  of silently evaluating on different splits. (Note: `scripts/run_faithfulness.py` selects
  by `email_id` on the pre-dedup frame, which is a different mapping; baselines follow
  the `make_splits.py` semantics.)
- **Real data IS present locally.** The brief said the CSV "may NOT be present"; it is
  present (`data/Phishing_Email.csv`, 50 MB). Rebuilt splits validated exactly:
  train 11215 / val 2803 / test 3495 with the manifest label distributions.
  Full (non-debug) baseline runs are still NOT RUN YET — only `--debug` smoke runs
  were executed locally (see Test results).
- **Faithfulness compatibility.** `faithfulness.evaluate_faithfulness_single_method`
  takes a `(PreTrainedModel, PreTrainedTokenizerFast)` pair and calls its own
  `predict_probability` internally, so it cannot be reused for non-transformer models.
  `evaluate_faithfulness_with_proba_fn()` implements the identical maths
  (comp = p_orig − p_drop, suff = p_orig − p_keep, k in {1,3,5}) over any
  `predict_proba(texts)` callable, which every baseline exposes. Documented in the
  module docstring.
- **DistilBERT test-prediction capture.** `train_model()` returns only summary metrics
  (no predictions, no pr_auc). After it returns, the best checkpoint is already loaded,
  so `_fit_distilbert()` re-runs `evaluate_dataloader` on the test loader once solely
  to capture predictions + the full H3 metric set. No model selection uses that second
  pass; it is inference only, but it is a second test pass and is recorded here honestly.
- **RoBERTa robustness not implemented for `roberta_eval`.** `evaluate_roberta_checkpoint`
  returns metrics/predictions but no fitted wrapper, so `--robustness` does not apply to
  it. Perturbation F1 for RoBERTa needs a Kaggle run with the checkpoint wrapped in a
  `predict_proba` callable; recorded as a follow-up, not faked.
- **DistilBERT has no validation-threshold view.** `threshold_val` is computed for
  sklearn/torch baselines only (they expose a wrapper); DistilBERT has no wrapper.
  Follow-up, not faked.
- **`subsample_splits` bug found by the smoke test and fixed.** pandas 2.x
  `groupby.apply` drops the grouping column, so the debug subsample lost the `label`
  column (`KeyError: 'label'`). Rewrote with an explicit per-label sample + concat.
- **Kaggle auto-detection.** `resolve_output_dir()` redirects a relative default output
  under `/kaggle/working/results/baselines` when `/kaggle` exists; absolute paths are
  honoured as given.
- **`untrained_fallback: true`** is written into every `baseline_results.json` when
  `results/best_model.pt` / `results/metrics.json` are missing (they are missing here).
- **Multimodal hook (H6).** Section E's `run_multimodal_eval.py` does not exist yet.
  Implemented as standalone: `evaluate_on_extractor_output(model, rows)` (rows are dicts
  with combined extractor text + label) and `evaluate_multimodal(models, eval_df)` with
  per-modality ablation over `body_text` / `ocr_text` / `pdf_text` / `qr_text` columns;
  CLI flags `--eval_csv` / `--multimodal_eval_dir`. Missing eval set -> status
  NOT RUN YET with the exact re-run command in the output. The interface is documented
  in the `baselines.py` H6 section and in this log for the coordinator to wire to
  `run_multimodal_eval.py --models`.
- **Dataset caveat** is a constant (`DATASET_CAVEAT`) printed in the report and JSON:
  old Enron-style, highly separable dataset; high simple-model scores do not imply
  holding on modern/AI-written or multimodal phishing; harder OOD test needed.

## Status per item (H1–H10)

- H1 Done — 4 new files listed above.
- H2 Done (code) / NOT RUN YET (full execution) — ladder implemented: dummy, TF-IDF+LR,
  TF-IDF char-SVM calibrated, BiLSTM (from-scratch embedding, configurable, AMP on CUDA,
  early stopping on val precision subject to recall >= 0.85), 1D-CNN behind `--with_cnn`,
  DistilBERT behind `--with_distilbert` (existing training-loop pattern, downloads only
  when the flag is used and weights are not cached), RoBERTa eval-only (metrics.json or
  checkpoint; missing -> NOT RUN YET, never retrained). Random search ~8 trials on
  validation, precision-first with recall floor.
- H3 Done (code) / NOT RUN YET (full execution) — accuracy, precision/recall/F1
  (phishing class), ROC-AUC, PR-AUC, confusion matrix, FPR, params, size MB, train
  wall-clock, latency ms/email (median of 200, batch size 1, CPU + GPU when available),
  peak memory; 3 seeds 42/1/7, mean +/- std.
- H4 Done (code) / NOT RUN YET (full execution) — McNemar (b/c/p, exact binomial when
  b+c<25), bootstrap 95% CI for F1 AND precision (1000 resamples), error analysis to
  `results/baselines/error_analysis.csv` (top 20 per direction, categorised by length /
  URL / HTML / non-English), robustness on truncated-128 / URLs-removed / typo-synonym
  (via `augment.py`, typo fallback offline).
- H5 Done (code) / NOT RUN YET (full execution) — TF-IDF+LR native coefficients AND
  LIME; BiLSTM LIME + Captum IntegratedGradients on the embedding layer (word-level,
  shared `split_words`); model-agnostic `predict_proba` wrapper compatible with the
  faithfulness maths (see decision above); comp/suff at k in {1,3,5}. Whether RoBERTa
  explanations are more faithful: NOT RUN YET (no executed numbers).
- H6 Done (code, standalone hook) / NOT RUN YET (execution) — Section E eval set does
  not exist yet; hook + CLI flag implemented and documented, not blocked on the other agent.
- H7 Done (code) / NOT RUN YET (real verdict) — report generator with precision-first
  table (precision/recall/F1/precision@recall-0.90 first), 2 plots, code-written verdict
  with BOTH branches implemented and unit-tested, recommended-deployment rule
  (highest precision, recall floor 0.85, floor actually used stated when unmet),
  dataset caveat, `--ood_csv` argument (never auto-downloads).
- H8 Done (code) — all CLI commands from the brief are implemented, incl. `--debug`,
  `--output_dir`, Kaggle auto-detection, `untrained_fallback` flag. Verified by smoke
  tests below (debug subset only). `run_multimodal_eval.py --models` belongs to Section E;
  this section provides the hook it should call.
- H9 Done — 34 tests, all passing (see below). Fake tiny fixtures only; no GPU, no
  internet, no real checkpoint; BiLSTM tests tiny (1 epoch, CPU).
- H10 Done — recorded above (identical splits/cleaning; never tune on test; ~8 trials
  on validation) and in the `fairness` block of every `baseline_results.json`.

## Test results

- `.venv/bin/python -m pytest tests/test_baselines.py -q` — **34 passed in 25.10s**
  (venv: torch 2.15.0+cpu; the shared venv took ~9 min to become ready).
  Coverage: no-leakage vocab/TF-IDF (test-only token `zxqvtestonly` is OOV), BiLSTM
  forward shape + 1-epoch training, proba rows sum to 1 (dummy/TF-IDF-LR/calibrated-SVM/
  BiLSTM), McNemar known inputs (b/c counts, p=1 symmetric, p<0.05 asymmetric),
  bootstrap F1 + precision CIs, verdict BOTH branches (precision headline + F1
  secondary), recommended model incl. recall-floor exclusion and relaxed-floor wording,
  latency timer, perturbations, faithfulness-with-proba-fn, LIME/IG alignment with
  `split_words`, random search precision-first selection, error-analysis directions,
  multimodal hook + ablation (incl. missing-column handling), `load_roberta_metrics`
  missing -> NOT RUN YET, seed aggregation, threshold fit (validation only).
- One failure found and fixed during the run: the relaxed-floor wording test exposed
  that `floor_used` (best recall among models, 0.80) was described as the chosen
  model's recall (0.70). Fixed the wording in `baselines.py`/`run_complexity_report.py`
  and the test; re-ran to 34/34.
- Full suite: `.venv/bin/python -m pytest tests/ -q` — **167 passed, 3 failed**.
  The 3 failures are NOT from Section H (none of those files import `baselines` or
  `complexity`): `test_sanity.py::test_environment_sanity` (expects Python 3.10, venv
  is 3.12 — pre-existing), `test_explain_and_resumability.py::test_captum_vs_transformers_interpret_tiny_model`,
  `test_multimodal_fusion.py::test_url_score_contribution_benign_vs_malicious`
  (other sections' tests, other agents active in the repo).

## Smoke tests executed (debug subset only — NOT real results)

Real data present, so smoke runs used the real CSV in `--debug` mode
(train 200 / val 100 / test 100 stratified subsample). These numbers are DEBUG-SUBSET
only and must never be quoted as real baseline results.

1. `.venv/bin/python scripts/run_baselines.py --debug --models dummy tfidf_lr --seeds 42 --random_search_trials 2 --output_dir /tmp/baselines_smoke`
   Exit 0. TF-IDF+LR debug: precision 1.0000, recall 0.8947, F1 0.9444 (n=100 test);
   dummy precision 0.0 (never predicts phishing). Validation threshold view worked
   (tfidf_lr threshold 0.3466, val precision 0.7708 at val recall 1.0). Random search
   honestly reported `floor_met=False` on the tiny subset (val recall 0.8378 < 0.85).
   `untrained_fallback: true`; comparisons NOT RUN YET (no RoBERTa predictions).
2. `.venv/bin/python scripts/run_baselines.py --debug --models bilstm --seeds 42 --output_dir /tmp/baselines_smoke_bilstm`
   Exit 0. 1 epoch, train 10.1 s, val precision 0.75 / recall 0.24 / F1 0.367;
   test precision 1.0 / recall 0.395 / F1 0.566 (debug subset).
3. `.venv/bin/python scripts/run_baselines.py --debug --models roberta_eval --seeds 42 --output_dir /tmp/baselines_smoke_roberta`
   Exit 0. Per-seed entry: `status NOT RUN YET, untrained_fallback true, reason:
   Checkpoint not found: results/best_model.pt`.
4. `.venv/bin/python scripts/run_baselines.py --debug --models tfidf_lr --seeds 42 --random_search_trials 0 --faithfulness --sample_size 2 --output_dir /tmp/baselines_smoke_faith`
   Exit 0. TF-IDF+LR coefficients faithfulness on 2 samples: comp@5 -0.0184,
   suff@5 0.3204 (real computed values on 2 debug emails, not a finding).
5. `scripts/run_complexity_report.py --results_dir /tmp/rpt` (with the smoke
   `baseline_results.json` placed at `/tmp/rpt/baselines/baseline_results.json`)
   Exit 0. Produced `complexity_report.md` + `.json` + both plots
   (`complexity_f1_vs_latency.png`, `complexity_f1_vs_train_time.png`); precision-first
   table sorted tfidf_lr above dummy; recommended model tfidf_lr (precision 1.0000,
   recall 0.8947, floor 0.85 met); verdict correctly NOT RUN YET (no RoBERTa numbers).

## Exact commands (full runs — NOT RUN YET, for Kaggle with GPU + checkpoint)

```bash
# Full ladder, 3 seeds (dummy/TF-IDF cheap on CPU; BiLSTM/CNN/DistilBERT/RoBERTa need GPU)
.venv/bin/python scripts/run_baselines.py --models dummy tfidf_lr tfidf_svm --seeds 42 1 7 --analysis --robustness
.venv/bin/python scripts/run_baselines.py --models bilstm --seeds 42 1 7 --epochs 8 --emb_dim 200 --hidden 128 --layers 2 --max_len 256 --analysis --robustness
.venv/bin/python scripts/run_baselines.py --models cnn distilbert --seeds 42
.venv/bin/python scripts/run_baselines.py --models roberta_eval --checkpoint results/best_model.pt
.venv/bin/python scripts/run_baselines.py --models tfidf_lr bilstm --faithfulness --sample_size 100
# Multimodal, once Section E produces its eval CSV:
.venv/bin/python scripts/run_baselines.py --models tfidf_lr bilstm --eval_csv <path-to-section-E-eval.csv>
# Report (reads results/baselines/baseline_results.json):
.venv/bin/python scripts/run_complexity_report.py --results_dir results
# Tests:
.venv/bin/python -m pytest tests/test_baselines.py -q
```

## Verdict paragraph (auto-generated rule)

No real numbers exist, so the verdict is **NOT RUN YET**: "no executed baseline or
RoBERTa precision numbers are available yet, so no verdict can be drawn. Run
scripts/run_baselines.py on the full splits (Kaggle GPU for RoBERTa/BiLSTM) and
re-run this report." The verdict-rule logic (unit-tested, both branches): headline on
PRECISION gap — if the best simple baseline (highest precision, recall >= 0.85) is
within 1.0 precision point of RoBERTa, its bootstrap precision CI overlaps RoBERTa's
precision, and McNemar p > 0.05, the verdict is "Transformer gives no statistically
significant precision gain... complexity is NOT justified by accuracy alone. Value
lies in the XAI/multimodal layer." (plus the F1 gap as a secondary statement);
otherwise "Transformer gives a significant precision gain of X points ... at Yx latency
and Zx parameters." (plus the F1 gap as a secondary statement).

## Comparison table (real numbers only)

All cells NOT RUN YET — no full baseline run has been executed (only `--debug`
smoke runs on 200/100/100 subsets; those numbers are not real results and are not
tabulated here).

| model | precision | recall | F1 | ROC-AUC | McNemar p vs RoBERTa | robustness F1 | faithfulness comp/suff @5 |
|---|---|---|---|---|---|---|---|
| dummy | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET |
| tfidf_lr | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET |
| tfidf_svm | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET |
| bilstm | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET |
| cnn | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET |
| distilbert | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET |
| roberta_eval | NOT RUN YET | NOT RUN YET | NOT RUN YET | NOT RUN YET | — | NOT RUN YET | NOT RUN YET |

## Only verifiable on Kaggle (GPU / checkpoint)

- Full 3-seed runs of BiLSTM/CNN (8 epochs, 11k train rows) and DistilBERT fine-tuning
  (needs `distilbert-base-uncased` weights; downloads only if not cached).
- RoBERTa eval-only: `results/best_model.pt` is absent locally, so `roberta_eval`,
  McNemar vs RoBERTa, bootstrap CIs vs RoBERTa, error analysis, and the verdict all
  need the checkpoint (or `results/metrics.json` with test predictions).
- GPU latency column (CPU latency is measured locally; GPU latency is measured only
  when CUDA is available).
- Faithfulness at `--sample_size 100` for BiLSTM (LIME + Captum IG) — slow on CPU.
- Multimodal comparison — needs Section E's eval set / `run_multimodal_eval.py`.
- RoBERTa robustness perturbations (see decision above).
