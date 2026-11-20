# DistilBERT execution plan (Kaggle)

Phase 2 of the DistilBERT work: tuning → 3-seed final training → text-set
evals → OOD → multimodal → checkpoint-dataset publish. Nothing here runs
itself: each kernel needs the user's explicit go-ahead for that stage plus
a quota check first (standing rule, 2026-11-19). All commands run from the
repo copy at `/kaggle/working/repo` (or the local repo root for the local
checks). Do NOT commit or push from a kernel.

## 0. What was missing (fixed locally 2026-11-20, uncommitted)

The single-seed DistilBERT run (text test P 0.9731 / R 0.9849 / F1 0.9789)
left NO loadable checkpoint. `train_model` did write a `best_model.pt`
per seed run dir, but it holds only `{epoch, model_state_dict, val_f1,
val_metrics}` — no tokenizer, no config, no model name — and nothing
could rebuild the model from it. Fixed in `src/phishing_xai/baselines.py`:

- `save_distilbert_model` / `load_distilbert_model`:
  `best_distilbert.pt` (`{format, kind, model_name, config, state_dict,
  n_params, extra}`) + `tokenizer/` + `config.json` in the same run dir.
  The `.pt` alone loads (model rebuilt from the saved config, no Hub).
  A RoBERTa `.pt` or a baseline artefact raises instead of misreading.
- `train_distilbert`: AdamW + AMP-on-CUDA loop, precision-first best epoch
  (validation precision subject to validation recall >= 0.85, best-F1
  fallback), per-epoch `latest_checkpoint.pt` (resumable), test evaluated
  exactly once per seed on the frozen best state (`test_texts=None`
  never touches test — used by tuning).
- `DistilBertScorer` + `load_scorer("distilbert", checkpoint=...)`:
  loads the artefact, never refits, never falls back to base weights.
  `explain_word_scores` takes the LITA branch for it (LIME +
  Transformers-Interpret/Captum, alpha 0.5).
- `scripts/run_baselines.py --models distilbert` now uses this path
  (configurable `--distilbert_lr`, default 2e-5) and wires the fitted
  wrapper into the per-seed robustness / OOD / val-threshold / latency /
  `.npy` views.

Unit tests: `tests/test_distilbert_checkpoint.py` (13 tests: save/load
round-trip exact, wrong-kind refusals, no-refit `load_scorer`,
`train_distilbert` smoke incl. tuning mode, LITA branch, robustness +
OOD loader smoke). Locally verified: 13/13 new tests pass, plus 84
passed across the directly affected suites
(`test_distilbert_checkpoint`, `test_baseline_save_load`,
`test_baselines`, `test_scorer_adapter`), 52 passed (OOD, multimodal,
explain/LITA) and 25 passed (faithfulness/ablation, model/train,
complexity report). Full local suite, coordinator's count 2026-11-20:
322 passed / 7 failed, all 7 pre-existing env-only failures (incl.
the `transformers-interpret`-not-installed assertion and the Python
3.10-vs-3.13 env sanity check). Real-model debug tuning verified
locally on CPU (2 trials, 1 epoch, 200 train / 100 val, real
`distilbert-base-uncased`): trial 0 (batch 16, lr 5e-05) val P 0.7500 /
R 0.8919 / F1 0.8148, floor met; trial 1 (batch 8, lr 3e-05) val
P 0.8710 / R 0.7297 / F1 0.7941, floor not met; precision-first
selection correctly picked trial 0. Final-driver debug also verified
end to end (1 seed, 200 train / 100 val / 100 test, batch 16 /
lr 5e-05 / 1 epoch): all per-seed files written,
`best_distilbert.pt` 255.45 MB, `latest_checkpoint.pt` 767 MB (AdamW
optimizer + scaler state), and the saved checkpoint reloaded
standalone with sane probabilities. Debug-subset numbers only
(val/test n=100) — NOT model results, NOT citable latencies (CPU
median 579 ms/email on this 1.5-CPU VM is a debug artefact, not a
measurement).
NOTE: never point trial/seed output dirs at `/tmp`: the per-epoch
`latest_checkpoint.pt` carries AdamW optimizer + scaler state (~767 MB)
and does not fit the 512 MB tmpfs (a debug run failed exactly that
way, `torch.save` zip `unexpected pos`, before the fix of running it
under `~/workspace/`).

## Inputs (Kaggle datasets — attach all of these to every kernel)

| Dataset (owner `syedaaleezazainab`) | Version | Contents used |
|---|---|---|
| `phishing-xai-repo-ah-6b1` (repo) | v14 | code + `data/Phishing_Email.csv` + `results/splits.json` |
| `phishing-xai-results` (results) | v7 | RoBERTa `results/best_model.pt` + `metrics.json`; BiLSTM `results/bilstm/best_bilstm.pt`; TF-IDF test `.npy` files; `results/paired_mcnemar_seed42.json` |
| `phishing-xai-checkpoint` (checkpoint) | v1 | RoBERTa `best_model.pt` (same file as in results v7) |
| `phishing-xai-ood-modern` (OOD) | — | OOD sets (PhishNChips, PhishFuzzer, Frontiers human/LLM, BEC clean/poisoned) — needed only for Kernel 4 |
| (new, Kernel 6) `phishing-xai-distilbert-checkpoint` | v1 | DistilBERT `best_distilbert.pt` + `tokenizer/` + `config.json` per seed (see Kernel 6) |

Splits (canonical, `results/splits.json`): train 11,215 / val 2,803 /
test 3,495. Environment: Tesla T4, torch 2.11, transformers 5.20.2,
Internet ON (first `distilbert-base-uncased` download only; all loads
after that use the saved artefacts).

## Kernel order

### Kernel 1 — Tuning (GPU, ~6 trials)

```bash
python scripts/tune_distilbert.py --n_trials 6
# --seed 42 --max_length 256 --subsample_train 4000 --recall_floor 0.85
```

- Search: batch [8, 16, 32] x lr [1e-5, 2e-5, 3e-5, 5e-5] x epochs
  [3, 4], 6 unique configs, seed 42, balanced-train subsample 4000.
- Selection: highest validation precision among trials with validation
  recall >= 0.85 (else highest recall, `floor_met: false`).
- Expected outputs (in `/kaggle/working/results/distilbert/tuning/`):
  `results.json` (all trials + best config, written after EACH trial),
  `best_params.json`.

### Kernel 2 — Final training, seeds 42, 1, 7 (GPU, the long kernel)

```bash
python scripts/run_distilbert_final.py --seeds 42 1 7
# --config results/distilbert/tuning/results.json (best params from Kernel 1)
# add --resume to resume a seed from its latest_checkpoint.pt
```

- Full balanced train split per seed; mixed precision (AMP on CUDA);
  test evaluated ONCE per seed on the frozen best state.
- Expected outputs, written after EACH seed (a preempted kernel keeps
  completed seeds):
  `results/distilbert/seed_<n>/best_distilbert.pt`,
  `seed_<n>/tokenizer/`, `seed_<n>/config.json`,
  `seed_<n>/latest_checkpoint.pt`, `seed_<n>/metrics.json`
  (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, FPR, confusion
  matrix, n_params, size MB, train seconds, best epoch, validation
  metrics, val-threshold view, CPU+GPU latency, peak memory),
  `seed_<n>/test_pred.npy`, `seed_<n>/test_prob.npy`,
  and `results/distilbert/summary.json` (aggregate + best-validation
  seed, updated after each seed).
- Best-validation seed = highest `best_val_f1` across the 3 seeds
  (ties: lowest seed). That seed's checkpoint is the one used by
  Kernels 3–5 (pass it explicitly with `--distilbert_checkpoint`).

### Kernel 3 — Text-set evals on the best-validation seed (GPU + CPU)

```bash
# (a) McNemar / bootstrap vs RoBERTa + per-seed robustness (uses the
#     test_pred/test_prob .npy files from Kernel 2 and RoBERTa test preds):
python scripts/run_baselines.py --models distilbert roberta_eval --seeds 42 1 7 \
    --robustness --distilbert_lr <best lr> --epochs <best epochs> \
    --batch_size <best batch> --checkpoint results/best_model.pt
# NOTE: this refits DistilBERT (it is the refit route for the ladder).
# For McNemar against the Kernel-2 checkpoints WITHOUT refitting, use the
# saved seed .npy files + results/paired_mcnemar_seed42.json with
# baselines.mcnemar_test / bootstrap_f1_ci / bootstrap_precision_ci
# (generic over prediction arrays — no code change needed).

# (b) Robustness of the loaded checkpoint (no retraining):
python scripts/run_robustness_loaded.py --models distilbert \
    --distilbert_checkpoint results/distilbert/seed_<best>/best_distilbert.pt

# (c) Latency (median of 200, batch 1, CPU+GPU) + LITA explanation time:
python scripts/run_latency_explain_time.py \
    --distilbert_checkpoint results/distilbert/seed_<best>/best_distilbert.pt

# (d) Faithfulness (LITA, comp/suff @ k in {1,3,5}, n=94 like Stage C)
#     + alpha ablation (alphas 0, 0.25, 0.5, 0.75, 1.0, n=47 like Stage C):
python scripts/run_faithfulness.py --model distilbert \
    --model_path results/distilbert/seed_<best>/best_distilbert.pt --sample_size 94
python scripts/run_ablation.py --model distilbert \
    --model_path results/distilbert/seed_<best>/best_distilbert.pt --sample_size 47
```

- Expected outputs: `results/baselines/baseline_results.json`
  (distilbert aggregate + `comparisons_vs_roberta`),
  `results/robustness_rerun/robustness_loaded.json`,
  `results/rerun_latency/latency_explain_time.json`
  (`distilbert_cpu/gpu` latency + `distilbert_lita` explanation time),
  `results/faithfulness.json`, `results/alpha_ablation.json`.
- Caveat: `evaluate_explainability(method="integrated_gradients")` does
  NOT support DistilBERT (it assumes a word-level `SimpleTokenizer`);
  the supported second method is Transformers-Interpret/Captum via
  `explain_word_scores` (LITA) — which is what (c)/(d) use.

### Kernel 4 — OOD eval (GPU)

```bash
python experiments/modern_phishing_ood/scripts/run_ood_eval.py \
    --sets phishnchips phishfuzzer frontiers_llm frontiers_human bec bec_poisoned \
    --models distilbert \
    --ood_dir /kaggle/input/.../phishing-xai-ood-modern \
    --data_path data/Phishing_Email.csv --splits_path results/splits.json \
    --distilbert_checkpoint results/distilbert/seed_<best>/best_distilbert.pt \
    --output_dir experiments/modern_phishing_ood/results \
    --predictions_dir experiments/modern_phishing_ood/predictions \
    --floors 0.85 0.90 0.95 --latency_n_gpu 200 --latency_n_cpu 100 --robustness
```

- Expected outputs: `{set}_distilbert_metrics.json` (6 sets),
  `{set}_distilbert_predictions.csv` in the exact schema
  (`email_id, true_label, predicted_label_at_0_5, prob_phishing, source,
  category, model, set`), `thresholds_distilbert.json`,
  `robustness_distilbert.json`, `ood_metrics_combined.json` (updated).

### Kernel 5 — Multimodal eval (GPU)

```bash
python scripts/run_multimodal_eval.py --model distilbert \
    --checkpoint results/distilbert/seed_<best>/best_distilbert.pt
# --model distilbert only drives the (a)-(d) ablations; do NOT add
# distilbert to --models (that side table fits TF-IDF/BiLSTM on the train
# split and has no DistilBERT fit path).
```

- Expected outputs: `results/multimodal_eval/multimodal_eval.json` +
  `.md` (DistilBERT (a)-(d) ablation table). The scorer is the
  `DistilBertScorer` (`TextScorer` protocol: `predict_proba` /
  `tokenize_words` / `truncate`), so the fusion/ablation code is
  unchanged.

### Kernel 6 — Publish the checkpoint dataset (manual, Kaggle UI / CLI)

New PRIVATE dataset `phishing-xai-distilbert-checkpoint` containing,
per seed (42, 1, 7):

```text
seed_42/best_distilbert.pt  seed_42/tokenizer/  seed_42/config.json
seed_1/best_distilbert.pt   seed_1/tokenizer/   seed_1/config.json
seed_7/best_distilbert.pt   seed_7/tokenizer/   seed_7/config.json
metrics_summary.json        (copy of results/distilbert/summary.json)
```

- Upload from `/kaggle/working/results/distilbert/` after Kernel 2.
- Flag the best-validation seed in the dataset description (from
  `summary.json`: `best_validation_seed`).
- Record the new dataset version (v1) here and in the Kernel 4/5
  commands of any later rerun. Do NOT publish the `latest_checkpoint.pt`
  files (resumability state, not needed for eval) or the RoBERTa
  checkpoint in this dataset.

## After the kernels (local, coordinator-owned — NOT this plan's kernels)

- Calibration + threshold-transfer + paired-stats reports for
  DistilBERT: `experiments/modern_phishing_ood/scripts/compute_task2_completeness.py`
  reads the saved `{set}_distilbert_predictions.csv` files generically
  (same schema as the other models). The ONLY change it needs is adding
  `"distilbert"` to its `MODELS` / `MODEL_LABEL` lists — left to the
  coordinator (that script and its 5 report files are theirs, in flight).
- Complexity report (`run_system_complexity_report.py`): picks up the
  DistilBERT aggregate from `baseline_results.json` (S3) automatically
  once Kernel 3 has run; the multimodal S3 cell needs Kernel 5.
- In-distribution ECE for DistilBERT (if wanted): use the Kernel-2
  seed-42 `test_prob.npy` + test labels rebuilt from
  `data/Phishing_Email.csv` + `results/splits.json` — the completeness
  script currently does this only for TF-IDF (same pattern applies).

## What is NOT done / cannot be verified locally

- No Kaggle run has been executed from this plan: tuning best config,
  3-seed metrics, OOD/multimodal numbers and latencies are all
  NOT RUN YET until Kernels 1–5 execute. No number in this plan is a
  result.
- GPU latency / peak CUDA memory can only be measured on the T4
  (Kernel 2/3); local CPU runs cannot verify them.
- `transformers-interpret` is NOT installed in the local venv (the
  local LITA test exercises the Captum fallback); on Kaggle the TI
  primary path is used when the package is present, Captum otherwise.
  Either way `explain_word_scores` returns `method_name="lita"`.
