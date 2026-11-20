# Repo inspection — phishing_xai (for the modern-phishing OOD experiment)

- **Working copy:** `~/workspace/phishing_xai_impl`
- **Branch:** `feature/multimodal-baselines` — **HEAD:** `ca4c78a` ("Add rerun cross-check numbers…", 2026-11-19)
- **Inspected:** 2026-11-20 (read-only). No original file was modified. No change was made to `src/phishing_xai/model.py` or `src/phishing_xai/faithfulness.py`.
- **Pre-existing working-tree state (not mine):** `git status` showed modified `samples/multimodal/sample4_text_pdf_invoice.eml`, `sample5_scanned_pdf.eml`, `sample6_clean_benign.eml`, plus untracked `.impl_logs/`, `results/multimodal_eval/`, `results/system_complexity_plots/`.
- **New directory tree created (only write in this task):** `experiments/modern_phishing_ood/{datasets,scripts,checkpoints,predictions,results,plots,logs,reports}`; this file is `experiments/modern_phishing_ood/logs/inspection.md`.
- **Convention:** every number below is either quoted from a file I read (with `file:line`) or marked **UNVERIFIED** where I could not confirm it in this copy. Stage files that live only in the Kaggle results dataset are flagged as such — they are **not** in this working copy.

---

## 1. Full repo map

```
phishing_xai_impl/
├── README.md / README_SUBMISSION.md        # README.md:14, :115 carry the Kaggle-round headline numbers + leakage note
├── decision.md (103,998 B)                 # running decision log; §§ on splits, 796-figure retraction, rerun round
├── docs/  MULTIMODAL_NOTE.md  PROJECT_LOG.md
├── pyproject.toml  requirements.txt  requirements-kaggle.txt  requirements-colab.txt
├── data/
│   ├── Phishing_Email.csv                  # 50 MB; see §5 (git-ignored via data/*.csv)
│   └── multimodal_eval/                    # 120 synthetic .eml + labels.csv + manifest.json; see §5
├── notebooks/  colab_run.ipynb  kaggle_run.ipynb  legacy/ (2 original notebooks)
├── samples/multimodal/                     # 6 demo .eml + explained PNGs
├── src/phishing_xai/  (16 modules, ~760 KB)
│   ├── __init__.py  config.py  data.py  model.py  train.py  tune.py
│   ├── evaluate.py  augment.py  explain.py  lita.py  faithfulness.py
│   ├── baselines.py (2,370 lines — ladder, stats, scorers, artefact save/load)
│   ├── extract.py (990 lines — .eml/PDF/image/QR extraction)
│   ├── multimodal.py (756 lines — fusion)  url_risk.py (524 lines — static URL risk)
│   └── report.py  utils.py
├── scripts/  (21 scripts; see §7)
│   ├── run_train.py  run_tune.py  tune_bilstm.py  make_splits.py  download_data.py
│   ├── run_baselines.py  run_robustness_loaded.py  run_latency_explain_time.py
│   ├── run_multimodal_eval.py  run_multimodal_predict.py  run_multimodal_faithfulness.py
│   ├── run_explain.py  run_faithfulness.py  run_ablation.py
│   ├── run_system_complexity_report.py  run_complexity_report.py
│   ├── regen_rerun_verdicts.py  make_report.py
│   └── make_multimodal_eval_set.py  make_multimodal_samples.py
├── tests/  (21 files, 263 `def test_` functions counted 2026-11-20)
├── results/  (3.2 MB — see §7; NO checkpoints, NO stage_a/stage_b/baselines dirs locally)
│   ├── splits.json  summary.json  RESULTS.md
│   ├── system_complexity_report.json/.md  system_complexity_plots/
│   ├── multimodal_eval/  rerun_mm/ (floor_0.80 + floor_0.85 × 5 models + verdicts.json)
│   ├── robustness_rerun/  rerun_latency/
│   └── (absent: best_model.pt, bilstm/, tuning/, baselines/, metrics.json)
└── experiments/modern_phishing_ood/        # NEW, this task only
    └── {datasets,scripts,checkpoints,predictions,results,plots,logs,reports}/
```

Key docs: `results/RESULTS.md` (Kaggle-round record, §§1–10), `docs/PROJECT_LOG.md` (§F Kaggle notebook, Kaggle-round + rerun rows), `docs/MULTIMODAL_NOTE.md` (checkpoint status §106, untrained-fallback rule).

---

## 2. RoBERTa implementation + checkpoint provenance

### 2.1 Code

| Piece | Location |
|---|---|
| `EmailDataset` (raw text + optional labels) | `src/phishing_xai/model.py:19` |
| `DynamicPaddingCollator` (dynamic padding, `max_length` default 512) | `src/phishing_xai/model.py:36` (`__init__` :39, `__call__` :47) |
| `load_tokenizer_and_model(model_name="roberta-base", num_labels=2, device=None)` — `AutoTokenizer.from_pretrained(..., use_fast=True)` + `AutoModelForSequenceClassification.from_pretrained(...)`; sets `id2label {0: "Safe Email", 1: "Phishing Email"}` | `src/phishing_xai/model.py:69` |
| `create_dataloader(texts, labels, tokenizer, batch_size=32, shuffle=False, max_length=512)` | `src/phishing_xai/model.py:93` |
| `train_model(...)` — AdamW, AMP (CUDA only), val-F1 checkpointing, early stopping, single final test eval | `src/phishing_xai/train.py:68`; `evaluate_dataloader` :32 |
| `evaluate_dataloader` — softmax P(phishing) = `logits[:,1]`, threshold 0.5 | `src/phishing_xai/train.py:32` |
| `run_random_search` / `train_and_eval_trial` (RoBERTa tuning) | `src/phishing_xai/tune.py:118` / `:46`; search space `DEFAULT_PARAM_DISTRIBUTIONS` `tune.py:29-33` (batch [8,16,32] × lr [1e-5,2e-5,3e-5] × epochs [3,4,5]) |

### 2.2 Config — defaults vs the config actually used

Defaults in code (`src/phishing_xai/config.py`):

- `ModelConfig` (:21): `model_name="roberta-base"`, `num_labels=2`, `max_length=512` (`debug_max_length=128`).
- `TrainConfig` (:30): `batch_size=32`, `eval_batch_size=64`, `learning_rate=2e-5`, `epochs=5`, `weight_decay=0.01`, `early_stopping_patience=3`, `seed=42`, `checkpoint_name="best_model.pt"`, `output_dir=results`.
- `DataConfig` (:9): `test_size=0.2`, `val_size=0.2`, `seed=42`, `replace_ratio=0.2`.
- `XAIConfig` (:73): LITA `alpha=0.5`.

**Actually-used (tuned) config — from `results/RESULTS.md:30` (Stage B, recorded in Kaggle results dataset v7 `stage_b/tuning/results.json`, NOT in this repo):** random search, 10 iterations, `max_length=256`, `subsample_train=4000`, seed 42; **best: batch 8 / lr 1e-5 / epochs 3, best val F1 0.9807 (val P 0.9722)**. The Stage B retrain used exactly those params (best epoch 3). **Do not use the code defaults (batch 32 / lr 2e-5 / epochs 5 / max_len 512) as "the trained config"** — they are the untuned defaults; `scripts/run_train.py` in production mode *requires* `--params` (`run_train.py:133-143` raises otherwise) and overrides lr/batch/epochs from it.

### 2.3 Training script

`scripts/run_train.py` — `parse_args` :20 (defaults mirror `config.py`: `--model_name roberta-base`, `--epochs 5`, `--batch_size 32`, `--learning_rate 2e-5`, `--max_length 512`, `--seed 42`); `resolve_splits` :44; `main` :102. Pipeline: `load_data` → `deduplicate_data` → `resolve_splits` → `balance_training_data` (train-only augmentation) → `load_tokenizer_and_model` → 3 dataloaders → `train_model`.

**Split-stability trap (`run_train.py:44-106`, docstring):** full runs reuse `results/splits.json` when present via `baselines.load_splits`, because regenerating with `StratifiedGroupKFold` is NOT stable across scikit-learn versions — the canonical manifest (train 11,215 / val 2,803 / test 3,495, sklearn 1.9.1) regenerates as 11,227 / 2,786 / 3,500 on Kaggle (sklearn 1.6.1). An OOD script must read `splits.json`, never regenerate splits.

**Checkpoint format** (`src/phishing_xai/train.py`, `train_model`): best checkpoint is a `torch.save` dict `{"epoch", "model_state_dict", "val_f1", "val_metrics"}` written to `<output_dir>/best_model.pt` (selected on **validation F1**); a separate `latest_checkpoint.pt` (model + optimizer + scaler + history) is deleted afterwards unless `--keep_latest`. Final test eval reloads the best checkpoint exactly once.

### 2.4 Checkpoint provenance — the headline finding

- **Local copy: ABSENT.** `results/best_model.pt` does **not** exist in this working copy (verified by `ls` + `find` for `*.pt/*.pth/*.bin/*.safetensors` — zero hits anywhere in the repo). It is git-ignored (`.gitignore`: `*.pt`, `*.bin`, `*.safetensors`, `checkpoints/`, explicit `results/best_model.pt`).
- **Only home:** private Kaggle dataset **`syedaaleezazainab/phishing-xai-checkpoint` v1** — 450 MB, best epoch 3, val F1 0.9834 (`docs/MULTIMODAL_NOTE.md:106`; `results/RESULTS.md:33`). Recorded contents: 124,647,170 params, 201 tensors (`results/RESULTS.md:33`).
- **Test result produced from it (Stage B, Kaggle Tesla T4, seed 42, n=3,495):** precision **0.9690**, recall 0.9932, F1 0.9809, accuracy 0.9854, ROC-AUC 0.9986, FPR 0.0193 (`results/RESULTS.md:33`; `README.md:14`).
- **Local consequence:** any script that loads RoBERTa locally either raises `FileNotFoundError` or — only with an explicit `--allow_pretrained_fallback` — runs untrained `roberta-base` (chance ~0.43–0.44) and must flag `untrained_fallback: true` (`scripts/run_multimodal_predict.py:108-144`; `docs/MULTIMODAL_NOTE.md:38`). The in-repo `results/multimodal_eval/multimodal_eval.json` is such a fallback run (`"untrained_fallback": true`, checkpoint path recorded as the absent local path). **The OOD experiment cannot run RoBERTa until the Kaggle checkpoint is downloaded into `experiments/modern_phishing_ood/checkpoints/` (or `results/`).**
- **Kaggle fetch path (from `notebooks/kaggle_run.ipynb` cells 3/6 and `README.md:64-67`):** attach the checkpoint dataset as input; the notebook copies the first `/kaggle/input/*/best_model.pt` to `/kaggle/working/results/best_model.pt`.

---

## 3. BiLSTM implementation + checkpoint provenance

### 3.1 Code

| Piece | Location |
|---|---|
| `BiLSTMClassifier(nn.Module)` — from-scratch embedding → 1–2-layer BiLSTM → max+mean pooling → linear head (`Linear(hidden*2*2, 2)`) | `src/phishing_xai/baselines.py:612` (`__init__` :615, `forward` :635) |
| `TorchTrainConfig` dataclass (defaults: `emb_dim=200, hidden=128, layers=2, dropout=0.3, max_len=256, max_vocab=30000, lr=1e-3, epochs=8, batch_size=32, patience=3`) | `src/phishing_xai/baselines.py:686` |
| `SimpleTokenizer` (word-level, train-only vocab) | `src/phishing_xai/baselines.py:564` — see §4 |
| `train_torch_classifier` (early stopping on **validation F1**; AMP only on CUDA) | `src/phishing_xai/baselines.py:717` |
| `build_torch_baseline(kind, ...)` — builds tokenizer + model, trains, returns `BaselineModel` wrapper | `src/phishing_xai/baselines.py:1020` |
| `BaselineModel` — uniform `predict_proba(texts) -> (n,2)`, `predict` at 0.5 | `src/phishing_xai/baselines.py:870` (`predict_proba` :898) |
| `torch_predict_proba` (batched inference helper) | `src/phishing_xai/baselines.py:841` |
| `select_best_bilstm_trial` (precision-first trial selection) | `src/phishing_xai/baselines.py:1395` |

### 3.2 Tuned config (parent-supplied values CONFIRMED in-repo)

`results/RESULTS.md:30`: BiLSTM tuning ran (6B-1, 10 trials × 3 epochs, seed 42); **best: emb_dim 200, hidden 64, layers 2, dropout 0.5, lr 0.002, max_len 128, batch_size 16; val precision 0.9798, recall 0.9364, F1 0.9576, recall floor 0.85 met.** Test (3 seeds): precision **0.9719 ± 0.0015**, recall 0.9428, F1 0.9571 (`results/RESULTS.md:46`; McNemar vs RoBERTa p=0.0002). A bigger hidden-256 variant exists (1 seed: P 0.9711 / R 0.9652 / F1 0.9681) but was **deliberately not saved** (`scripts/run_robustness_loaded.py` `NOT_ATTEMPTED`).

Note the tuned config differs from every default: `run_baselines.py` CLI defaults are emb 200 / hidden 128 / layers 2 / max_len 256 / dropout 0.3 / lr 1e-3 / batch 32 / epochs 8 (`run_baselines.py:75-128` area), and `TorchTrainConfig` defaults match those. **Pass the tuned values explicitly; do not rely on defaults.**

### 3.3 Tuning + training scripts

- `scripts/tune_bilstm.py` — `SEARCH_SPACE` (emb [100,200], hidden [64,128,256], layers [1,2], dropout [0.2,0.3,0.5], lr [5e-4,1e-3,2e-3], max_len [128,256], batch [16,32]); `tune_bilstm(...)` trains each trial 1 seed via `build_torch_baseline`, scores on **validation at 0.5**; `select_best_bilstm_trial` picks highest val precision with recall ≥ floor (0.85), else highest recall then precision with `floor_met: false`. Writes `results/bilstm/tuning.json`. Usage: `python scripts/tune_bilstm.py --n_trials 10 --epochs 3`.
- `scripts/run_baselines.py --models bilstm --seeds 42 1 7 --epochs 8 --emb_dim 200 --hidden 128 --layers 2 --max_len 256` (docstring example, :8-9) — refit route. Per-seed test predictions saved as `<out>/<model>_seed<seed>_test_pred.npy` / `_test_prob.npy`; best-seed fitted model (best **validation F1**, ties → lowest seed) saved via `save_baseline_model` to `<out_dir>/best_<model>.pt` (`run_baselines.py:500-513`).

### 3.4 Artefact format + provenance

- `save_baseline_model` (`baselines.py:1127`): torch models saved with `torch.save` as `{"format": "phishing_xai.baseline.v1", "kind", "config", "state_dict", "tokenizer": {word2idx, max_vocab, max_len}, "n_params", "extra"}`; TF-IDF models pickled as `{"format", "kind", "pipeline", ...}`. Format marker `BASELINE_ARTEFACT_FORMAT = "phishing_xai.baseline.v1"` (`baselines.py:1113`); recognised kinds `("bilstm", "cnn", "tfidf_lr", "tfidf_svm")`.
- `load_baseline_model(path, device=None)` (`baselines.py:1229`): rebuilds architecture from the saved config + tokenizer, loads `state_dict`, returns a `BaselineModel`. `_is_baseline_artefact` (:1292) distinguishes these from a RoBERTa `.pt` (which has no format marker).
- **Local copy: ABSENT.** `results/bilstm/best_bilstm.pt` does **not** exist in this working copy (no `results/bilstm/` dir at all). The tuned BiLSTM artefact lives only in the Kaggle results dataset v7 (`stage_6b2/` per `docs/PROJECT_LOG.md` Kaggle-round row; `results/RESULTS.md:7` says stage outputs are in v7 under `stage_a/`, `stage_b/`, `stage_c/`, `stage_e2/`, `stage_6b2/`, `baselines/`). **The OOD experiment needs that artefact downloaded before BiLSTM can be loaded without refitting.** The tuning JSON itself (`results/bilstm/tuning.json`) is likewise Kaggle-only, not in-repo.

---

## 4. Tokenization + preprocessing — both models

### 4.1 Shared cleaning (applies to BOTH — it happens before either tokenizer)

`phishing_xai.data.clean_text` (`src/phishing_xai/data.py:45`): **lowercases** (`text.lower()`), strips HTML via `BeautifulSoup(text, "html.parser").get_text()` (with `MarkupResemblesLocatorWarning` suppressed), replaces `\r` with space, collapses whitespace (`re.sub(r"\s+", " ", ...)`), strips. Called in `load_data` (:108) → `df["clean_text"]`. Both models train/evaluate on `clean_text`, never on raw `Email Text`.

**URL handling in cleaning: none.** `clean_text` does NOT remove, mask, or replace URLs — URLs survive as lowercased tokens inside `clean_text`. URL-specific handling lives elsewhere and does NOT feed the text scorers: `phishing_xai.url_risk.extract_urls` / `score_url` (`src/phishing_xai/url_risk.py:209` / `:272`) and the multimodal fusion URL channel (`src/phishing_xai/multimodal.py:277` `score_urls_for_email`). The robustness perturbation `urls_removed` (`baselines.py:522`, `URL_REGEX.sub(" ", ...)`) is the only place URLs are stripped, and only for that perturbation test.

### 4.2 RoBERTa pipeline

- Tokenizer: Hugging Face **RoBERTa BPE** (`AutoTokenizer.from_pretrained("roberta-base", use_fast=True)`, `model.py:69-76`). Byte-level BPE, subword; no separate lowercasing step in the tokenizer (text is already lowercased by `clean_text`).
- `max_length`: **512** in production (`ModelConfig.max_length`, `config.py:21`; `DynamicPaddingCollator` default 512, `model.py:39`; `create_dataloader` default 512, `model.py:93`). Tuning used max_length 256 and the tuned retrain used the tuned config — see §2.2; the exact max_length of the saved checkpoint's retrain is **UNVERIFIED** in-repo beyond "tuned config" (the Stage B tuning file is Kaggle-only). Single-text scoring (`scripts/run_multimodal_predict.py:145` `score_text`) truncates with `truncate_text_to_model_window(clean, tokenizer, max_length=512)` (`data.py:71`: caps pre-tokenization chars at 50,000, then cuts at the last content token's end offset inside the window).
- Padding: **dynamic per-batch** to the longest sequence in the batch (`model.py:47-58`); no fixed padding to 512.
- Empty text: `score_text` returns `(0, 1.0, 0.0)` without calling the model (`run_multimodal_predict.py:145-153`).

### 4.3 BiLSTM pipeline

- Tokenizer: `SimpleTokenizer` (`baselines.py:564`) — **whitespace split** (`str(text).split()`), word-level, vocab capped at `max_vocab=30000`, fit on **TRAIN texts only** (`fit` :581, frequency order; `build_torch_baseline` :1020 fits it on train). Unseen words → `<unk>` (id 1); `<pad>` id 0. Empty text → single `<unk>` (`encode` :594).
- `max_len`: **128 in the tuned config** (vs 256 default in `TorchTrainConfig` and `run_baselines.py`). Truncation = keep first `max_len` whitespace tokens (`encode`: `str(text).split()[:self.max_len]`).
- Batching: `_collate_torch` (`baselines.py:703`) pads each batch to its own max length; `pack_padded_sequence` in `BiLSTMClassifier.forward` (:635) uses the attention mask lengths.
- Embeddings trained **from scratch** (no pretrained vectors); `nn.Embedding(vocab_size, emb_dim, padding_idx=0)`.

### 4.4 Differences table (what the OOD script must respect)

| Aspect | RoBERTa | BiLSTM |
|---|---|---|
| Tokenizer | HF RoBERTa BPE, subword | `SimpleTokenizer`, whitespace word-level, train-only vocab 30k |
| Vocab source | Pretrained `roberta-base` (never fit on this data) | Fit on train split only; OOD-only words → `<unk>` |
| `max_length` | 512 (code default; tuned retrain **UNVERIFIED**, tuning used 256) | **128 (tuned)**; 256 is only the untuned default |
| Truncation | Token-window cut at exact char boundary (`truncate_text_to_model_window`, 50k char cap) | First-N whitespace tokens |
| Lowercasing | Via shared `clean_text` (both) | Via shared `clean_text` (both) |
| URL handling | URLs kept as text tokens (both); URL-risk is a separate fusion channel, not a text-scorer input | Same |
| Padding | Dynamic per-batch (HF) | Dynamic per-batch (`_collate_torch`) |
| Empty input | `(0, 1.0, 0.0)` in `score_text` | Single `<unk>` id, still scored |
| Augmentation | WordNet synonym replacement, train-only, +2,863 → balanced train 14,078 (`augment.py:67` `balance_training_data`, `replace_ratio=0.2`, seed 42) | Same train-only augmentation is applied in `run_baselines.py` (`_balanced_train`, :182) for the refit route; a **loaded artefact needs no augmentation** |
| Inference batch | `evaluate_dataloader` (batch 64 eval default) / `score_text` one-by-one | `torch_predict_proba` batched; `BaselineModel.predict_proba` one call for a list |

**OOD implication:** the two models must each receive text through their own path — RoBERTa via its HF tokenizer + `clean_text`, BiLSTM via the artefact's stored `word2idx` (never rebuild the vocab on OOD text). `load_scorer` / `RobertaScorer` / `BaselineScorer` already encapsulate this (§10).

---

## 5. Dataset files + provenance

### 5.1 Main dataset

- `data/Phishing_Email.csv` — 50 MB on disk, **175,088 lines including header** (`wc -l`; i.e. 175,087 data rows by line count — the file contains embedded newlines/quoted fields, so this is a line count, not a row count). **Row count verified from the manifest, not from `wc`:** raw rows **18,650** (`results/splits.json`, see §6). Columns: index (unnamed), `Email Text`, `Email Type` (`"Safe Email"` / `"Phishing Email"`).
- Provenance: Kaggle "Phishing Emails" by `subhajournal` (kaggle.com/datasets/subhajournal/phishingemails), mirrored on Hugging Face at `zefang-liu/phishing-email-dataset` (`README.md:130`; `docs/PROJECT_LOG.md:17`). `scripts/download_data.py` is the download route; the CSV is git-ignored (`data/*.csv`). A plain `csv` module read fails on this file (field larger than the 128 KB default limit) — use pandas as `load_data` does.
- **UNVERIFIED in-repo:** per-class row counts of the raw file (the manifest only records split-level label distributions after cleaning; a pandas count was not run for this inspection).

### 5.2 Splits manifest — verified

`results/splits.json` (read directly 2026-11-20):

| Field | Value |
|---|---|
| `raw_rows` | 18,650 |
| `nulls_dropped` | 16 |
| `conflicting_rows_dropped` | 533 |
| `conflicting_unique_texts` | `["empty"]` (one text — the literal string `"empty"` — carried both labels across 533 rows) |
| `exact_duplicates_dropped` | 585 |
| `total_unique_retained` | **17,513** (= 18,650 − 16 − 533 − 585) ✓ |
| `augmentation_samples_added_to_train` | 2,863 |
| `final_balanced_train_size` | 14,078 (= 11,215 + 2,863) ✓ |
| `summary.train` | **11,215** (Safe 7,039 / Phishing 4,176) ✓ |
| `summary.val` | **2,803** (Safe 1,765 / Phishing 1,038) ✓ |
| `summary.test` | **3,495** (Safe 2,173 / Phishing 1,322) ✓ |
| `train_indices` / `val_indices` / `test_indices` | lists of 11,215 / 2,803 / 3,495 `email_id`s (positions in the deduplicated frame) |

Cross-check: `results/summary.json` and `results/RESULTS.md:19` agree with these numbers. (Stale figures 17,532 / 11,220 / 2,806 / 3,506 were corrected in `decision.md:42` — do not use them.)

### 5.3 Synthetic multimodal eval set

`data/multimodal_eval/` — **120 `.eml` files** (verified count), 60 phishing / 60 benign, split 24 validation / 96 test (`manifest.json`):

- Categories (from `manifest.json`): `image_only_phishing` 10, `text_button_image` 8, `qr_body_image` 8, `qr_in_text_pdf` 7, `scanned_pdf` 7, `html_link_mismatch` 7, `html_attachment_phishing` 6, `fake_invoice_pdf` 7, `newsletter_logo_qr` 20, `real_invoice` 20, `meeting_agenda` 20.
- `labels.csv` — 121 lines (header + 120 rows); columns `filename,label,category,split`.
- `manifest.json` — `seed: 42`, `validation_fraction: 0.2`, split rule: "Stratified deterministic split (seed+1 stream): 20% of each label to validation; thresholds are fitted on validation only, never test." Synthetic caveat is explicit in the manifest: generated images/PDFs from a small template set with simulated OCR stressors; "no real-world adversary variation and no real emails."
- Generator: `scripts/make_multimodal_eval_set.py` (661 lines); demo samples in `samples/multimodal/` (6 `.eml`).
- This set is **not** an OOD set for the text models in the usual sense (synthetic, template-based) — it is the prior multimodal eval; reuse its *protocol* (val-fitted thresholds, per-row predictions) rather than its numbers.

---

## 6. Train/val/test splits — how they were built

- Builder: `scripts/make_splits.py` — `make_splits` :32 (CLI `parse_args` :22; defaults `--test_size 0.2 --val_size 0.2 --seed 42`); calls `load_data` → `deduplicate_data` → `create_stratified_splits` → writes the rich manifest (raw/null/conflicting/exact-duplicate counts + augmentation count + `train/val/test_indices`).
- Core: `phishing_xai.data.create_stratified_splits` (`src/phishing_xai/data.py:210`, `group_by_near_dup=True` by default):
  1. `load_data` (:108): drop nulls in `Email Text`/`Email Type`, drop `Unnamed: 0`, map labels (`LABEL_MAP` :20: Safe 0 / Phishing 1), `clean_text`, drop rows empty after cleaning.
  2. `deduplicate_data` (:152): **(a)** conflicting duplicates — same `clean_text` with >1 distinct label → drop **all** copies (533 rows, the `"empty"` text); **(b)** remaining exact duplicates on `clean_text` → keep first (585 rows).
  3. Near-dup grouping: `compute_near_dup_hash` (:61) = **MD5 of the first 200 alphanumeric chars** (lowercased, non-word chars stripped; falls back to the whole text if the prefix is empty). Stored as `near_dup_group`.
  4. Split: if groups exist, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` — first fold = test (20%), then a second 5-fold split of train+val gives val (20% of train+val = 16% overall). Otherwise plain stratified `train_test_split` (same fractions, seed 42).
  5. Hard assertions (:286-307): zero overlap of `clean_text` AND zero overlap of `near_dup_group` across all three split pairs — a violation raises `RuntimeError`.
- **Seed:** 42 everywhere (`DataConfig.seed`, `TrainConfig.seed`, `make_splits.py`, `create_stratified_splits` default). Stratification is on `label` in both the grouped and fallback paths.
- **Reproducibility warning (matters for OOD):** `StratifiedGroupKFold` output is sklearn-version-dependent (`run_train.py:44-106` docstring). The canonical `splits.json` is the source of truth; `baselines.load_splits` (`baselines.py:133`) rebuilds splits from the manifest's `email_id`s and **raises `ValueError` if sizes or label distributions disagree** with the manifest summary. Never regenerate splits for comparison work.

---

## 7. Saved predictions + evaluation scripts

### 7.1 Scripts (what each produces)

| Script | Produces / notes |
|---|---|
| `scripts/run_train.py` | Trains RoBERTa; `results/metrics.json` + `results/best_model.pt` + `loss_and_f1_curves.png`, `confusion_matrix.png`, `roc_auc_curve.png`; appends runs log + consolidated report (production only). Requires `--params` (tuned) unless `--debug`. |
| `scripts/run_tune.py` | RoBERTa random search driver (wraps `tune.py:118`). |
| `scripts/tune_bilstm.py` | BiLSTM random search; writes `results/bilstm/tuning.json`. |
| `scripts/make_splits.py` | `results/splits.json`. |
| `scripts/run_baselines.py` | Baseline ladder. `--models dummy tfidf_lr tfidf_svm bilstm cnn distilbert roberta_eval`; `--seeds 42 1 7`; `--output_dir` (default `results/baselines`); writes `baseline_results.json` + per-seed `<model>_seed<seed>_test_pred.npy` / `_test_prob.npy` + `best_<model>.pt` artefacts + optional `error_analysis.csv`, robustness, faithfulness, multimodal blocks. Has `--ood_csv` (see §10.3 caveat). Merges with an existing `baseline_results.json` by default (`merge_baseline_results`, :129). |
| `scripts/run_multimodal_eval.py` | Multimodal eval on an `.eml` dir; `--model/--models roberta tfidf_lr tfidf_svm bilstm rule_based`; `--fusion weighted|max`; `--recall_floor` (default 0.80, :92); `--checkpoint`; `--eval_dir` (default `data/multimodal_eval`); `--allow_pretrained_fallback`; writes `multimodal_eval.json` (+ `.md`, confusion-matrix + ablation PNGs) into the output dir. Thresholds fitted on the **validation slice only** via `fit_threshold_precision_first` (:211). |
| `scripts/run_multimodal_predict.py` | Single-`.eml`/directory prediction + LITA explanations; `load_trained_roberta` (:108), `score_text` (:145). |
| `scripts/run_multimodal_faithfulness.py` | Multimodal faithfulness (default checkpoint `results/best_model.pt`, :294). |
| `scripts/run_robustness_loaded.py` | Artefact-loaded robustness; `--models roberta bilstm tfidf_lr tfidf_svm`; `--roberta_checkpoint` (default `results/best_model.pt`), `--bilstm_checkpoint` (default `results/bilstm/best_bilstm.pt`); `--output_dir` default `results/robustness_rerun`; writes `robustness_loaded.json`. **Never retrains.** Template for the OOD loader pattern (§10). |
| `scripts/run_latency_explain_time.py` | Latency (median of 200, batch 1, CPU+GPU separately) + explanation time; defaults `--roberta_checkpoint results/best_model.pt`, `--bilstm_checkpoint results/bilstm/best_bilstm.pt`, `--output_dir results/rerun_latency`; writes `latency_explain_time.json`. `main` :99. |
| `scripts/run_system_complexity_report.py` | Reads already-produced JSONs in `--results_dir` (default `results`), writes `system_complexity_report.json` + `.md` + plots to `--output_dir` (default = results dir). `main` :1050; verdict builder §8. `--ood_csv` is **noted only, never evaluated here** (`parse_args` :96). |
| `scripts/run_complexity_report.py` | Earlier complexity report (500 lines; superseded in practice by `run_system_complexity_report.py` — **UNVERIFIED** which one produced the in-repo `system_complexity_report.json`; the rerun verdicts import from `run_system_complexity_report`). |
| `scripts/regen_rerun_verdicts.py` | Regenerates verdicts from saved rerun numbers only (no GPU/retraining); reads `results/rerun_mm/floor_0.80|0.85/*/multimodal_eval.json` + `results/rerun_latency/latency_explain_time.json`; writes `results/rerun_mm/verdicts.json`. |
| `scripts/run_explain.py` / `run_faithfulness.py` / `run_ablation.py` | LIME/TI/LITA explanations, faithfulness (comp/suff @ k∈{1,3,5}), α ablation; `--model_path`/`--checkpoint_path` default `results/best_model.pt`. |
| `scripts/make_multimodal_eval_set.py` / `make_multimodal_samples.py` | Generate the 120-email synthetic set / the 6 demo samples. |
| `scripts/download_data.py` (80 lines) | Dataset download route. |

### 7.2 Saved prediction / metric JSONs present IN THIS REPO

| File | Contents (verified by reading keys) |
|---|---|
| `results/splits.json` | §5.2 |
| `results/summary.json` | Run summary; `tuning`/`final_training_and_test` fields are literally `"NOT RUN YET"` (stale local placeholder — the Kaggle run is recorded in `RESULTS.md`, not here) |
| `results/RESULTS.md` | Narrative record of the Kaggle round + rerun round |
| `results/system_complexity_report.json` (+ `.md`, `system_complexity_plots/multimodal_ablation.png`) | Systems table, multimodal-by-system, verdict, recommended deployment |
| `results/multimodal_eval/multimodal_eval.json` (+ `.md`, 2 PNGs) | **Untrained-fallback run** (`untrained_fallback: true`); do not quote as detection evidence |
| `results/rerun_mm/floor_0.80/` + `floor_0.85/` × 5 models (`roberta`, `bilstm`, `tfidf_lr`, `tfidf_svm`, `rule_based`) | `multimodal_eval.json` + `.md` each; **these contain per-row test predictions** (`rows` with `label`, `pred_d_weighted`, split=test, n=96) — the files `regen_rerun_verdicts.py` recomputes McNemar/CI from |
| `results/rerun_mm/verdicts.json` | `floor_0.80` + `floor_0.85` verdicts + stats (McNemar p, bootstrap CI) |
| `results/robustness_rerun/robustness_loaded.json` | Keys: `seed, protocol, models, not_attempted, n_test` (n_test=3,495); per-model `clean_at_0_5` + `robustness{truncated, urls_removed, typo_synonym}` |
| `results/robustness_rerun/baseline_results.json` | Refit-route robustness for TF-IDF+LR / TF-IDF+SVC / CNN |
| `results/rerun_latency/latency_explain_time.json` | Keys: `seed, n_latency, batch_size, cuda_available, gpu, protocol, latency_ms_per_email, explanation_seconds_per_email, n_test, explain_email_ids`; RoBERTa GPU median 15.50 ms / CPU 242.50 ms; BiLSTM GPU 8.37 / CPU 19.90 ms; RoBERTa LITA median 30.87 s/email vs BiLSTM LITA-IG 1.87 s/email |

### 7.3 Files the task brief names that are NOT in this repo (Kaggle dataset v7 only)

- `stage_a/baseline_results.json`, `stage_b/metrics.json`, `stage_b/tuning/results.json` — **absent locally** (no `results/stage_*/`, `results/baselines/`, `results/tuning/`, `results/metrics.json`). They live in Kaggle dataset `syedaaleezazainab/phishing-xai-results` **v7** (`results/RESULTS.md:7`). Any OOD script that needs Stage B metrics must either read `results/RESULTS.md` (narrative) or download v7.
- No `*_test_pred.npy` / `*_test_prob.npy` files exist anywhere in this working copy (the only `baseline_results.json` in-repo is the robustness-refit one above).

---

## 8. Existing metrics + stats implementations

### 8.1 Metric functions

- `phishing_xai.evaluate.compute_metrics(y_true, y_prob, threshold=0.5)` (`src/phishing_xai/evaluate.py:29`) → accuracy / precision / recall / F1 (phishing class, `zero_division=0`) / ROC-AUC (0.5 fallback if single-class). Also `get_classification_report_str` (:58) and plot helpers (:69, :107, :137).
- `phishing_xai.baselines.compute_metrics_full(y_true, y_prob, threshold=0.5)` (`src/phishing_xai/baselines.py:218`) → the H3 set: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, `precision_at_recall_0.90` (+ threshold), confusion matrix `[[tn,fp],[fn,tp]]`, FPR, n. `precision_at_recall` :256; `fit_threshold_for_recall` :283; `select_deployment_model` :305 (highest precision subject to recall ≥ floor, default 0.85).

### 8.2 Statistical tests

- `mcnemar_test(y_true, pred_a, pred_b)` (`src/phishing_xai/baselines.py:356`): b = A wrong/B right, c = A right/B wrong; χ² with continuity correction; **exact binomial p when b+c < 25**, else χ² survival; returns `{b, c, statistic, p_value}`; `{b:0, c:0} → p=1.0`.
- `bootstrap_f1_ci(y_true, y_pred, n_resamples=1000, seed=42, ci=0.95)` (`baselines.py:389`): percentile CI, resamples with a single class in `y_true` skipped; returns `{f1, ci_low, ci_high, n_resamples}`.
- `bootstrap_precision_ci(...)` (`baselines.py:425`): same scheme for precision; returns `{precision, ci_low, ci_high, n_resamples}`.
- `mean_std(values)` (`baselines.py:461`); `measure_latency(predict_proba_fn, texts, n=200, warmup=5)` (`baselines.py:470`) — median of single-email (batch-1) calls.
- Multimodal-side threshold fitting: `fit_threshold_precision_first` (`scripts/run_multimodal_eval.py:211`) — highest precision with recall ≥ floor (default floor 0.80 in that script), ties → higher recall then F1; returns `floor_met` flag.

### 8.3 Verdict generator rule (FIXED 2026-11-19)

- `_check_set(simple_prec, s4_prec, ci, p_value)` — `scripts/run_system_complexity_report.py:536`. **Significant ⟺ gap > 0 AND McNemar p < 0.05** (:560-567, verbatim in the code comment: *"Significant means a POSITIVE precision gap AND McNemar p < 0.05. A negative gap is never a gain, and a positive gap with p >= 0.05 (or a missing p) is never called significant"*). The old rule (gap > 1.0 pt alone, or p ≤ 0.05 alone, or non-overlapping CI alone) mislabelled a −1.30 pt gap and a +5.88 pt gap with p=0.625 as "significant" — both withdrawn (commit `fd90bb0`, "Fix verdict generator: significant only if positive gap and McNemar p < 0.05").
- `build_system_verdict(rows, mm)` (:581) applies `_check_set` to the text set and the multimodal set; `regen_rerun_verdicts.py` reuses it. Corrected verdict text (`results/rerun_mm/verdicts.json`, floor_0.85): *"Transformer is below the best simple baseline by 0.11 precision points on the text test set (RoBERTa 0.9690 vs tfidf_lr 0.9701; McNemar p = 0.3491) at 223.7x latency and 3116.1x parameters."*
- **Reuse for OOD:** call `mcnemar_test` + `bootstrap_precision_ci` (1000 resamples, seed 42, as `regen_rerun_verdicts.py:66-70` does) and `_check_set`/`build_system_verdict` for any "is the gap significant" claim. Do not invent a new rule.

---

## 9. Leakage audit of the ORIGINAL data

### 9.1 Verified figures (from `results/splits.json`, §5.2)

Raw 18,650 → **16 nulls** dropped → **533 conflicting-label rows** dropped (single text `"empty"`, both labels) → **585 exact duplicates** dropped → **17,513 retained**. Splits 11,215 / 2,803 / 3,495 with the hard zero-overlap assertions in `create_stratified_splits` (`data.py:286-307`).

### 9.2 Paper-protocol leakage 288/3,727 = 7.7% — UNVERIFIED IN THIS REPO

The task brief states the paper-protocol (non-deduplicated) replication leaks 288/3,727 (7.7%) of test emails. **I could not verify this figure from any file in this working copy:** a repo-wide grep for `288` returns only line numbers inside `splits.json` index lists; `3727` appears only in `README_SUBMISSION.md:22` ("on the paper's own test set (3,727 emails)") and `:82-84` (paper protocol: train/val/test split with validation 2,982 / test 3,727; WordNet balancing +2,565 → 14,490 — note this is a *different* balancing count from the clean protocol's +2,863 → 14,078). No script, JSON, or log in this copy computes or stores a 288 overlap count. Treat 288/3,727 as **brief-supplied, not repo-verified**; the OOD write-up must either cite its original source (Phase 3 replication log, Kaggle results dataset v7) or recompute it — do not present it as established by this inspection. (The Phase 3 replication itself — 3-seed mean test accuracy 97.80 ± 0.07% — is recorded in `README_SUBMISSION.md:22` and memory, likewise not recomputable from files in this copy.)

### 9.3 Near-duplicate detection — what exists vs what does not

**Exists in code:**

- `compute_near_dup_hash` (`data.py:61`): MD5 of first 200 alphanumeric chars (lowercased). Used as the `StratifiedGroupKFold` group key in `create_stratified_splits` (`data.py:233-234, 250-259`), with a hard assertion of zero group overlap across splits (`data.py:298-307`). `decision.md:11` records this as "0 near-duplicate groups leak across train/val/test" — that is a *consequence of the grouping + assertion*, not a counted measurement.
- `README.md:115` states the same honestly: "Near-duplicate grouping uses the MD5 hash of the first 200 alphanumeric characters … but no script in the repo logs a near-duplicate group count, so no group figure is claimed here."

**Does NOT exist:**

- **No group-count logging anywhere.** No script counts or logs how many near-duplicate groups exist, how many are multi-member, or their size distribution (`make_splits.py` / `data.py` group by the hash but never count groups — `decision.md:42`).
- **The old "796 near-dup groups" figure was unverified and removed** (`decision.md:11`, `:42`, `:204`). Do not use 796 anywhere.
- **No similarity-based near-dup detection** (no TF-IDF cosine, no MinHash/LSH, no edit-distance, no embedding similarity) — the only near-dup notion in the repo is the exact first-200-char MD5 hash. Template variants that differ inside the first 200 chars, or differ only after it, are **not** grouped. This is a genuine gap the OOD experiment should note: an OOD set that shares templates with the training data beyond the first 200 chars would not be caught by this check.
- **No cross-dataset decontamination helper** — there is no function that takes an external CSV and checks overlap (exact or near-dup) against the training set. The OOD script will need to write one (reusing `clean_text` + `compute_near_dup_hash` from `data.py` is the consistent choice).

---

## 10. What a new OOD eval needs from this repo

### 10.1 Prerequisite (blocking): fetch the two checkpoints first

Both checkpoints are **Kaggle-only** (§2.4, §3.4). Nothing below runs locally until:

1. RoBERTa: download `best_model.pt` from Kaggle dataset `syedaaleezazainab/phishing-xai-checkpoint` v1 → place at `experiments/modern_phishing_ood/checkpoints/best_model.pt` (and verify ~450 MB / 201 tensors; `torch.load` dict must contain `model_state_dict`).
2. BiLSTM: download `best_bilstm.pt` from Kaggle results dataset `syedaaleezazainab/phishing-xai-results` v7 (`stage_6b2/`) → `experiments/modern_phishing_ood/checkpoints/best_bilstm.pt` (verify `format == "phishing_xai.baseline.v1"`, `kind == "bilstm"`).
3. Optional but recommended: download v7 `stage_b/metrics.json` + `stage_b/tuning/results.json` and `stage_a/baseline_results.json` for the in-distribution reference numbers (they are not in this repo).

Do **not** retrain to fill the gap (hard project rule, e.g. `run_multimodal_predict.py:108-124` refuses to train; `run_robustness_loaded.py` never retrains).

### 10.2 Exact load-and-predict entry points

**RoBERTa — `TextScorer` protocol (preferred; identical to the robustness script):**

```python
import sys; sys.path.insert(0, "src")
from phishing_xai.baselines import load_scorer
scorer = load_scorer("roberta", checkpoint="experiments/modern_phishing_ood/checkpoints/best_model.pt")
probs = scorer.predict_proba(texts)          # (n, 2) ndarray, rows sum to 1, col 1 = P(phishing)
# scorer.tokenize_words(text) -> List[str]; scorer.truncate(text, max_length=None) -> str
```

Under the hood (`baselines.py:2146` `load_scorer`, `"roberta"` branch): delegates to `scripts/run_multimodal_predict.py::load_trained_roberta` (`:108` — builds tokenizer+model via `model.py:69 load_tokenizer_and_model`, `torch.load`s the checkpoint, takes `ckpt["model_state_dict"]` if present, `model.eval()`), wrapped in `RobertaScorer` (`baselines.py:2016`), whose `predict_proba` (:2040) calls `score_text` per text (`run_multimodal_predict.py:145`: empty → `(0, 1.0, 0.0)`; else `truncate_text_to_model_window(..., max_length=512)` + single forward pass, threshold 0.5 for the returned class). Direct eval-only alternative: `baselines.evaluate_roberta_checkpoint(checkpoint, test_texts, test_labels, model_name="roberta-base", max_length=512, batch_size=64)` (`baselines.py:1832`) — returns metrics + `y_pred`/`y_prob` lists; loads the checkpoint once and evaluates on a labelled set.

**BiLSTM — same protocol, artefact load (no fitting):**

```python
import sys; sys.path.insert(0, "src")
from phishing_xai.baselines import load_scorer, load_baseline_model
scorer = load_scorer("bilstm", checkpoint="experiments/modern_phishing_ood/checkpoints/best_bilstm.pt")
probs = scorer.predict_proba(texts)          # (n, 2); BaselineScorer wraps BaselineModel
# — or lower level: —
model = load_baseline_model("experiments/modern_phishing_ood/checkpoints/best_bilstm.pt")  # baselines.py:1229
probs = model.predict_proba(texts)           # BaselineModel.predict_proba, baselines.py:898
```

`load_scorer("bilstm", checkpoint=...)` (`baselines.py:2146`): if the path exists **and** `_is_baseline_artefact` (:1292) recognises it, loads via `load_baseline_model` (rebuilds `BiLSTMClassifier` from the saved config + stored `word2idx`, loads `state_dict`). **Trap:** if the checkpoint path is missing, or points at a RoBERTa `.pt`, `load_scorer` silently **fits a fresh BiLSTM on the real train split** (3 epochs, `TorchTrainConfig` defaults — not the tuned config) instead of failing. The OOD script must assert the artefact exists and `_is_baseline_artefact(path)` is True *before* calling `load_scorer`, or call `load_baseline_model` directly (it raises on a bad path). A corrupt artefact also raises rather than falling back — good; check for that explicitly in logs.

**CLI equivalents (no Python needed):**

```bash
# RoBERTa eval-only on a labelled set (needs checkpoint):
python scripts/run_baselines.py --models roberta_eval --checkpoint experiments/modern_phishing_ood/checkpoints/best_model.pt
# BiLSTM refit route (NOT artefact load; only if a refit is ever intended):
python scripts/run_baselines.py --models bilstm --seeds 42 --epochs 8 --emb_dim 200 --hidden 64 --layers 2 --dropout 0.5 --lr 0.002 --max_len 128 --batch_size 16
# Artefact-loaded robustness (the pattern to copy for OOD):
python scripts/run_robustness_loaded.py --models roberta bilstm \
  --roberta_checkpoint experiments/modern_phishing_ood/checkpoints/best_model.pt \
  --bilstm_checkpoint experiments/modern_phishing_ood/checkpoints/best_bilstm.pt
```

### 10.3 Input format the eval scripts expect

- **Text OOD via `run_baselines.py --ood_csv`:** CSV with columns **`Email Text`, `Email Type`** (`_load_ood`, `run_baselines.py:319`, via `data.load_data` → `clean_text` + label map). **Caveat — RoBERTa is NOT covered:** in `run_baselines.py` main loop, `roberta_eval` (:403-410) and `distilbert` `continue` before the OOD block, so `ood_metrics` is only computed for `dummy`/`tfidf_lr`/`tfidf_svm`/`bilstm`/`cnn` (`run_baselines.py:476-478`: `current_model.predict_proba(ood_df["clean_text"])`, `compute_metrics_full`). **Do not use `--ood_csv` for the RoBERTa OOD number** — use the `load_scorer` path in §10.2 (or `evaluate_roberta_checkpoint`) and compute metrics with `compute_metrics_full`.
- **Multimodal-style eval via `run_multimodal_eval.py`:** a directory of `.eml` files + `labels.csv` (`filename,label,category,split`) + `manifest.json` (`load_labels`, :838; defaults `--eval_dir data/multimodal_eval`, `--checkpoint results/best_model.pt`). Thresholds are fitted on the `validation` split only. Only relevant if the OOD set is email-shaped (`.eml`); for plain-text OOD, the `run_baselines`/`load_scorer` path is the right one.
- **Plain-text OOD (recommended):** a CSV with `Email Text` / `Email Type` (same as `--ood_csv`), or a JSONL of `{"text", "label"}` — but note no existing script reads JSONL; the OOD script should read the CSV through `load_data` so `clean_text` is applied identically, then call `scorer.predict_proba(clean_texts)` and `compute_metrics_full` / `mcnemar_test` / `bootstrap_precision_ci`.
- **Threshold rule:** report at the fixed 0.5 threshold for comparability with all in-distribution numbers (every headline number in `RESULTS.md` is at 0.5); if a threshold is fitted, fit it on a validation slice of the OOD set (or the original val split) only, never on the OOD test rows — the repo's standing rule (`config.py` `FusionConfig` docstring; `run_multimodal_eval.py:22`).

### 10.4 Reusable entry points (checklist for the OOD script)

| Need | Reuse (do not reimplement) |
|---|---|
| Load RoBERTa checkpoint + `predict_proba` on arbitrary texts | `baselines.load_scorer("roberta", checkpoint=...)` → `RobertaScorer` (`baselines.py:2146`, `:2016`); or `run_multimodal_predict.load_trained_roberta` + `score_text` (`:108`, `:145`) |
| Load BiLSTM artefact + `predict_proba` | `baselines.load_baseline_model(path)` (`:1229`) or `load_scorer("bilstm", checkpoint=...)` with the artefact-exists guard (§10.2) |
| Identical text cleaning | `data.clean_text` (`data.py:45`); full-frame load via `data.load_data` (`:108`) |
| Near-dup / exact-overlap decontamination check vs train | `data.compute_near_dup_hash` (`data.py:61`) + `data.load_data`/`deduplicate_data`; splits via `baselines.load_splits(data_path, splits_path)` (`baselines.py:133`, validates against `splits.json`) |
| Metrics | `baselines.compute_metrics_full` (`:218`); `evaluate.compute_metrics` (`evaluate.py:29`) |
| Significance | `baselines.mcnemar_test` (`:356`), `bootstrap_f1_ci` (`:389`), `bootstrap_precision_ci` (`:425`); verdict rule `_check_set` (`run_system_complexity_report.py:536`) |
| Perturbation-style stress (if wanted) | `baselines.apply_perturbation` / `evaluate_robustness` (`:522`, `:1474`) — truncated / urls_removed / typo_synonym |
| Latency (if wanted) | `baselines.measure_latency` (`:470`), as used by `run_latency_explain_time.py` |
| Artefact-loaded eval pattern to copy | `scripts/run_robustness_loaded.py` (`run_robustness_for_scorer` :77; `main` :107) — cleanest template: load scorer → `predict_proba` → `compute_metrics_full`, no retraining, NOT RUN YET recorded per model |
| What NOT to reuse for OOD | `run_baselines.py --ood_csv` for RoBERTa (skips it, §10.3); `run_system_complexity_report.py --ood_csv` (noted only, never evaluated); `load_scorer` without an artefact-exists guard for BiLSTM (silent refit trap, §10.2); the in-repo `results/multimodal_eval/` numbers (untrained fallback) |

### 10.5 Gaps / risks the OOD run must handle explicitly

1. **No local checkpoints** — the single biggest blocker (§2.4, §3.4). Download from Kaggle before any OOD run; verify sizes/formats.
2. **RoBERTa tuned `max_length` for the retrain is UNVERIFIED in-repo** (tuning used 256; code default 512; `score_text` uses 512). The checkpoint itself carries no max-length metadata. OOD scoring should use 512 (the `score_text`/`RobertaScorer` default) and state it; do not claim the retrain used 512.
3. **BiLSTM silent-refit trap** in `load_scorer` (§10.2) — guard with an explicit artefact check.
4. **`run_baselines.py --ood_csv` skips RoBERTa/DistilBERT** (§10.3) — a naive `--ood_csv` run would report OOD for the baselines and silently nothing for RoBERTa.
5. **Near-dup detection is first-200-chars MD5 only** (§9.3) — OOD decontamination against train needs a stronger check than the repo's own if the OOD set shares templates; at minimum report both exact-`clean_text` overlap and `near_dup_group` overlap, and state the limitation.
6. **Paper-protocol leakage 288/3,727 is UNVERIFIED in-repo** (§9.2) — do not cite it as established without its source.
7. **Sklearn-version split instability** (§6) — always load `splits.json`; never regenerate.
8. **Stage A/B artefacts are Kaggle-only** (§7.3) — in-distribution reference numbers for the OOD comparison should come from `results/RESULTS.md` (narrative) or a fresh v7 download, not from files assumed to be in-repo.

---

## Appendix — headline in-distribution numbers (for the OOD comparison baseline)

All from `results/RESULTS.md` (Kaggle Tesla T4, 2026-11-19, seed 42, test n=3,495, threshold 0.5):

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| RoBERTa (Stage B) | 0.9690 | 0.9932 | 0.9809 | 0.9854 | 0.9986 |
| Tuned BiLSTM (3 seeds) | 0.9719 ± 0.0015 | 0.9428 | 0.9571 | — | — |
| TF-IDF+SVC (3 seeds) | 0.9820 ± 0.0007 | 0.9892 | 0.9856 | — | 0.9993 |
| TF-IDF+LR (3 seeds) | 0.9702 | 0.9834 | 0.9767 | — | — |
| DistilBERT (1 seed) | 0.9731 | 0.9849 | 0.9789 | — | — |
| CNN (1 seed) | 0.9732 | 0.9614 | 0.9673 | — | — |

McNemar vs RoBERTa: TF-IDF+SVC p=0.253, TF-IDF+LR p=0.349, DistilBERT p=0.603 (not significant); CNN p=0.0006, tuned BiLSTM p=0.0002 (RoBERTa significantly better on overall errors). Bootstrap 95% precision CIs: TF-IDF+SVC [0.9746, 0.9889], TF-IDF+LR [0.9607, 0.9793], tuned BiLSTM [0.9613, 0.9789] — all contain RoBERTa's 0.9690. Robustness (test, thr 0.5): RoBERTa F1 clean 0.9809 / truncated 0.9773 / urls_removed 0.9755 / typo_synonym 0.9638; TF-IDF+SVC most robust. Latency (median, batch 1, n=200): RoBERTa GPU 15.50 / CPU 242.50 ms; BiLSTM GPU 8.37 / CPU 19.90 ms.
