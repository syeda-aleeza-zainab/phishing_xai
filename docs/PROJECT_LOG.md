# Project Decisions Log & Technical History

*Project: Phishing Email Detection with Explainable AI (XAI)*  
*Repository: Phishing_Email_Detection_XAI*  
*Last Updated: 2026-10-20*

---

## 1. Project Overview

### Goal
To scientifically refactor, debug, elevate, and benchmark an NLP and Explainable AI (XAI) research pipeline for phishing email detection. The codebase fine-tunes a RoBERTa language model on email text and explains predictions using LIME, Transformers-Interpret (Integrated Gradients), and LITA (a hybrid attribution metric), supported by rigorous faithfulness benchmarking and ablation studies.

### Upstream Base
- **Upstream Repository**: [https://github.com/Amazuddin/Phishing_Email_Detection_XAI](https://github.com/Amazuddin/Phishing_Email_Detection_XAI)  
  Credited as the original research base (originally comprised two unstructured Jupyter notebooks: `Phishing_Email_Detection_XAI_(LIME_TI_LITA).ipynb` and `RANDOM_SEARCH_HYPERPARAMETER_TUNING_(RoBERTa).ipynb`).
- **Dataset Source**: Kaggle "Phishing Emails" dataset by `subhajournal` ([Kaggle Link](https://www.kaggle.com/datasets/subhajournal/phishingemails)), mirrored on Hugging Face at `zefang-liu/phishing-email-dataset` (`Phishing_Email.csv`).

### Environment & Constraints
- **Python**: Strictly Python 3.10 within a dedicated virtual environment (`.venv`).
- **Operating System**: `Windows-10-10.0.26300-SP0` (verified via `platform.platform()`; requiring `pathlib.Path` throughout; no hardcoded POSIX paths).
- **Hardware**: Local environment is CPU-only (CUDA not available locally; verified via `torch.cuda.is_available() == False`).
- **Execution Architecture**: Fast CPU debug mode (`--debug`) for rapid local verification (~500 emails, max length 128, 1 epoch; measured times: debug tuning ~2.5 min, debug training ~14 min on CPU) combined with a production Google Colab notebook (`notebooks/colab_run.ipynb`) for full GPU training (5 epochs, max length 512, 14,078 balanced samples).

---

## 2. Original Repo Review Findings (Legacy Bugs Identified)

A systematic code audit of the upstream Jupyter notebooks revealed critical bugs, data hygiene issues, and methodological flaws:

1. **LIME Probability Bug**: In the legacy notebook, `classifier_fn` passed to LIME returned raw, unbounded model logits instead of probabilities. LIME requires true probability distributions. Fixed by applying `torch.softmax(logits, dim=1)` and explicit sample bounding (`num_samples=1000`).
2. **Missing Deduplication (Risk of Leakage)**: Duplicates could leak across splits. The Kaggle dataset contained 585 exact duplicate rows and 533 conflicting rows with the text `'empty'` (335 labeled Phishing, 198 labeled Safe) before deduplication (source: `scripts/make_splits.py` execution on `data/Phishing_Email.csv`).
3. **Test Set Evaluated Every Epoch**: Evaluated test every epoch; risk of selecting on test, though no best-checkpoint selection existed (the legacy code simply retained the final epoch).
4. **Broken Figure Generation**: `plt.savefig()` was invoked *after* `plt.show()` (resulting in blank canvas saves) and without file extensions.
5. **Incomplete Seeding**: Only Python's built-in `random.seed()` was set; `torch`, `torch.cuda`, and `numpy` random number generators were left unseeded, causing non-deterministic splits and training trajectories.
6. **LITA Normalization Distortion**: The original LITA score min-max normalized token attributions to $[-1, 1]$ before linear combination ($\alpha \cdot \text{LIME} + (1 - \alpha) \cdot \text{TI}$). This artificially forced the smallest score to $-1$ and largest to $+1$ regardless of original magnitude.
7. **Token vs. Word Aggregation Mismatch**: RoBERTa uses Byte-Pair Encoding (BPE) where subwords are prefixed with `Ġ` (e.g. `Ġaccount`). The legacy notebook attempted string equality matching between raw text words and tokens, failing to align subword attributions.
8. **Meaningless Hyperparameter in Search Space**: `test_batch_size` was included in the random search grid despite having zero impact on model parameters or evaluation metrics.
9. **Inconsistent Class Naming**: Code alternated between `"Legitimate"` vs `"Safe Email"` and `"Phishing"`, leading to indexing confusion.
10. **Engineering Deficits**: No `requirements.txt`, no package structure, no unit tests, no software license, and a one-paragraph README without reproducibility instructions.

---

## 3. Decisions Table

| Decision | Reason | Phase | Status |
|---|---|---|---|
| **Label Map: Safe Email = 0, Phishing Email = 1** | Establishes a uniform binary contract across all data ingestion, model heads, metrics, and XAI explainers. | Phase 0 | Active |
| **Deduplicate on cleaned text before any split** | Eliminates data leakage between partitions. Cleaned text removes HTML tags, punctuation noise, and whitespace before hashing. | Phase 1 | Active |
| **Drop all copies of conflicting-label rows** | 533 rows with the text `'empty'` had conflicting labels (335 Phishing, 198 Safe). Keeping any copy would bias the model; dropping all copies prevents label noise. | Phase 1 | Active |
| **Near-duplicate grouping via normalized 200-char hash** | Groups template-level near-duplicates using `StratifiedGroupKFold` so template variations never cross train/val/test splits (0% near-dup leakage). | Phase 5 | Active |
| **Double stratified split (80/20 train+val/test, 80/20 train/val)** | Preserves exact positive class ratio (~37.3%) across all partitions. Split manifest saved to `results/splits.json` to lock partitions across machines. | Phase 1 | Active |
| **Augment training split ONLY (WordNet synonym replacement)** | Balances phishing class in training (4,176 -> 7,039) while keeping validation and test splits pristine and strictly real. | Phase 1 | Active |
| **Model-window text truncation** | Truncates text to the exact character boundary corresponding to `max_length` tokenization, with a pre-tokenization cap (50k chars) to prevent pathological regex overhead. Ensures LIME, TI, and faithfulness see the exact text the model consumes. | Phase 5 | Active |
| **Shared word splitter regex** | Uniform regex (`r"https?://[^\s,;!?\"'>)]+\|\b\w+(?:['’]\w+)*\|[^\w\s]"`) used for LIME's `split_expression` and subword aggregation, ensuring exact token-to-word alignment for contractions, URLs, and punctuation. | Phase 5 | Active |
| **Validation F1 checkpointing & early stopping** | Prevents overfitting and eliminates test-set leakage. Test split is evaluated strictly once on the best validation checkpoint. | Phase 2 | Active |
| **Resumable training and hyperparameter tuning** | Saves optimizer, model, scaler, epoch, and metric states to `latest_checkpoint.pt` (`--resume`); tuning saves iteratively to `results.json` and skips completed trials. | Phase 2 | Active |
| **Strict debug isolation under `results/debug/`** | Debug artifacts must never overwrite production files or pollute `results/runs_log.csv` or `results/RESULTS.md`. | Phase 2 & 3 | Active |
| **Zero fabricated metrics policy** | All entries in `RESULTS.md` and `summary.json` must originate from real artifacts produced by executed code; unrun stages display `*NOT RUN YET*`. | Phase 3 | Active |
| **Cheaper proxy tuning budget** | CPU/modest GPU tuning uses proxy settings (`max_length=256`, `subsample_train=4000`), with full production training executed at `max_length=512`. Limitation is explicitly documented. | Phase 2 & 3 | Active |
| **Omit LICENSE file** | Upstream repository had no license; creating one unilaterally risks license misattribution. Full credit given to upstream repo and dataset author in README. | Phase 0 | Active |
| **Transformers-Interpret with Captum Fallback** | Maintains upstream compatibility with `SequenceClassificationExplainer` while providing a direct `LayerIntegratedGradients` fallback under an identical interface (baseline `[<s>, <pad>..., </s>]`, softmax target, L2 norm; Spearman rank correlation > 0.9). | Phase 2 & 5 | Active |
| **LITA variants: min-max + magnitude-preserving** | Retains original min-max formulation for experimental fidelity while introducing a magnitude-preserving variant ($\alpha=0.0 \implies \text{TI}, \alpha=1.0 \implies \text{LIME}$). | Phase 4 | Active |
| **Faithfulness evaluation protocol & Alpha Ablation** | Measures comprehensiveness and sufficiency at $k \in \{1, 3, 5\}$ across LIME, TI, LITA (both variants), and a 5-seed random baseline; evaluates $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$. | Phase 5 | Active |
| **Colab stage-by-stage Google Drive backup** | Backs up artifacts to Google Drive after EACH pipeline stage to eliminate data loss from unexpected disconnection. | Phase 5 | Active |
| **OneDrive path & checkpoint storage safeguards** | Repository resides inside OneDrive. Added `--ckpt_dir` to store checkpoints outside OneDrive and auto-delete intermediate `latest_checkpoint.pt` after successful runs (unless `--keep_latest`). | Phase 3 | Active |

---

## 4. Work Completed Per Phase

> Historical record: the `file:///c:/Users/admin/OneDrive/...` links in this section point at the original author's Windows machine (October 2026). They are kept as written because this is a log of work done then, not a set of current paths; nothing in the repo today lives at those locations. Section B (B7, November 2026) removed hard-coded user paths from `extract.py`.

### Phase 0: Foundations & Project Setup (2026-10-18)
- Initialized production directory hierarchy: `src/phishing_xai/`, `tests/`, `scripts/`, `notebooks/`, `docs/`, `results/`.
- Verified execution environment: Python 3.10.0, `Windows-10-10.0.26300-SP0`, no local GPU (`torch.cuda.is_available() == False`).
- Created pinned dependency manifest [`requirements.txt`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/requirements.txt) for Python 3.10 on Windows.
- Created `pyproject.toml` and verified local editable install (`pip install -e .`).

### Phase 1: Data Pipeline, Deduplication, & Augmentation (2026-10-18)
- Created [`src/phishing_xai/data.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/data.py): HTML stripping via BeautifulSoup, lowercase normalization, whitespace collapse, deduplication, conflict resolution, and deterministic stratified partitioning.
- Created [`src/phishing_xai/augment.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/augment.py): WordNet synonym replacement balancing the minority phishing class strictly on the training set.
- Added tests in [`tests/test_data_and_augment.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/tests/test_data_and_augment.py) covering data cleaning, deduplication, conflict dropping, split manifests, and leakage prevention.

### Phase 2: Modeling, Training, Evaluation, Tuning, & Colab Setup (2026-10-18)
- Created [`requirements-colab.txt`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/requirements-colab.txt) for flexible cloud GPU execution.
- Created [`src/phishing_xai/model.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/model.py): Dynamic padding collator and model initialization.
- Created [`src/phishing_xai/evaluate.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/evaluate.py): Metrics calculation (Accuracy, Precision, Recall, F1, ROC-AUC) and visualization utilities with pre-show saving.
- Created [`src/phishing_xai/train.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/train.py): Mixed-precision training (`torch.amp`), gradient accumulation, validation F1 checkpointing, early stopping, test-split isolation, and resumability checkpointing (`latest_checkpoint.pt`).
- Created [`src/phishing_xai/tune.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/tune.py): Random search hyperparameter search over batch size, learning rate, and epochs with trial resumability.
- Created [`notebooks/colab_run.ipynb`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/notebooks/colab_run.ipynb): End-to-end cloud GPU orchestration notebook.

### Phase 3: Remediation, Production Splits, & Reporting (2026-10-18)
- Created [`scripts/download_data.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/scripts/download_data.py): Strict verification of dataset schema and row count (~18,650 rows).
- Created [`scripts/make_splits.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/scripts/make_splits.py): Executed full-dataset preprocessing without model training, producing [`results/splits.json`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/results/splits.json).
- Implemented [`src/phishing_xai/report.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/report.py) and [`scripts/make_report.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/scripts/make_report.py): Idempotent report generation with strict debug rejection.
- Implemented `--ckpt_dir` and automatic deletion of intermediate `latest_checkpoint.pt` after successful completion (unless `--keep_latest`).

### Phase 4: Explainability & LITA Formulation (2026-10-18)
- Implemented [`src/phishing_xai/lita.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/lita.py): Both legacy min-max normalization and magnitude-preserving signed max-absolute scaling.
- Implemented [`src/phishing_xai/explain.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/explain.py): Batched LIME probability wrapper, primary Transformers-Interpret explainer, and LayerIntegratedGradients fallback.
- Created [`scripts/run_explain.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/scripts/run_explain.py): Local explanation generator saving side-by-side LIME, TI, and LITA attributions.

### Phase 5: Faithfulness, Alpha Ablation, & Robust Alignment (2026-10-19 / 2026-10-20)
- Implemented [`src/phishing_xai/faithfulness.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/faithfulness.py): Comprehensiveness ($P(y|x) - P(y|x \setminus \text{top-}k)$) and Sufficiency ($P(y|x) - P(y|\text{top-}k)$) at $k \in \{1, 3, 5\}$ and 5-seed random baseline.
- Implemented [`scripts/run_faithfulness.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/scripts/run_faithfulness.py) with incremental per-email JSONL logging and global explanation extraction (top-20 words for Safe and Phishing with minimum frequency filtering).
- Implemented [`scripts/run_ablation.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/scripts/run_ablation.py): Alpha ablation across $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\} \times \{\text{magnitude\_preserving}, \text{minmax}\}$ at $k=3$.
- Updated [`src/phishing_xai/data.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/data.py) with `truncate_text_to_model_window` and near-duplicate template grouping via `StratifiedGroupKFold` (0% near-duplicate leakage).
- Updated [`src/phishing_xai/explain.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/explain.py) with shared regex word splitter, LIME feature cap configuration, and exact Captum parity.
- Updated [`notebooks/colab_run.ipynb`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/notebooks/colab_run.ipynb) with stage-by-stage Google Drive backup and complete cell ordering.
- Updated [`src/phishing_xai/report.py`](file:///c:/Users/admin/OneDrive/Desktop/phishig%20llm/Phishing_Email_Detection_XAI/src/phishing_xai/report.py) to format sections 5.1 (faithfulness table), 5.2 (alpha ablation table), and 5.3 (global explanations).
- Total test suite expanded to **41 passing tests** with 100% pass rate.

---

## 5. Current Status Table

| Pipeline Stage | Status | Local Execution Notes |
|---|---|---|
| **Data Ingestion & Cleaning** | **COMPLETE** | Full dataset loaded, nulls removed, HTML stripped, text normalized. |
| **Deduplication & Near-Dup Grouping** | **COMPLETE** | Exact duplicates (585) and conflicting-label rows (533) dropped; template near-duplicates grouped into single partitions with 0% cross-split leakage. |
| **Stratified Splitting** | **COMPLETE** | 17,513 emails partitioned into 11,215 train, 2,803 val, 3,495 test. Manifest saved to `results/splits.json`. |
| **Augmentation** | **COMPLETE** | WordNet synonym replacement balancing minority phishing class strictly on the training set (+2,863 samples -> 14,078 balanced train). |
| **Hyperparameter Tuning** | **COMPLETE (code & debug verified)** | Random search driver (`run_tune.py`) tested with trial resumability; production run configured for Colab GPU. |
| **Model Fine-Tuning** | **COMPLETE (code & debug verified)** | Training loop (`train.py`, `run_train.py`) verified with validation-F1 checkpointing, mixed precision, and single-pass test evaluation. |
| **Explainability (LIME / TI / Captum / LITA)** | **COMPLETE** | Shared word regex, LIME probability wrapper, Captum fallback parity (correlation > 0.9), and both LITA variants fully tested. |
| **Faithfulness Benchmarks** | **COMPLETE (code & debug verified)** | Comprehensiveness and sufficiency at $k \in \{1, 3, 5\}$ and 5-seed random baseline implemented and verified. |
| **Alpha Ablation Study** | **COMPLETE (code & debug verified)** | LITA parameter sweep ($\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\} \times 2\text{ variants}$) implemented and verified. |
| **Colab Orchestration** | **COMPLETE** | `notebooks/colab_run.ipynb` configured with GPU assertion, full workflow sequence, and stage-by-stage Google Drive backup. |
| **Consolidated Reporting** | **COMPLETE** | `report.py` dynamically builds `RESULTS.md` and `summary.json` from genuine artifacts without hardcoded metrics. |

---

## 6. How to Reproduce

### Local Smoke-Test Execution (Fast CPU Debug Mode)
1. **Activate Virtual Environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **Generate Leakage-Free Dataset Splits**:
   ```powershell
   python scripts/make_splits.py
   ```
3. **Run Debug Hyperparameter Tuning**:
   ```powershell
   python scripts/run_tune.py --debug
   ```
4. **Run Debug Training with Optimal Parameters**:
   ```powershell
   python scripts/run_train.py --debug --params results/tuning/best_params.json
   ```
5. **Run Local Explanations (LIME, TI, LITA)**:
   ```powershell
   python scripts/run_explain.py --debug
   ```
6. **Run Debug Faithfulness Benchmarks**:
   ```powershell
   python scripts/run_faithfulness.py --debug
   ```
7. **Run Debug Alpha Ablation Study**:
   ```powershell
   python scripts/run_ablation.py --debug
   ```
8. **Generate Consolidated Evaluation Report**:
   ```powershell
   python scripts/make_report.py
   ```
9. **Run Full Test Suite**:
   ```powershell
   pytest -v
   ```

### Google Colab GPU Production Execution
1. Open `notebooks/colab_run.ipynb` in Google Colab.
2. Select a GPU runtime (`Runtime -> Change runtime type -> T4 / V100 / A100`).
3. Run cells in sequential order:
   - **Cell 1**: Asserts CUDA GPU availability.
   - **Cell 2**: Mounts Google Drive and prepares `/content/checkpoints` and `/content/drive/MyDrive/phishing_xai_results`.
   - **Cell 3**: Installs dependencies via `requirements-colab.txt` and `pip install -e .`.
   - **Cell 4**: Downloads and strictly verifies `data/Phishing_Email.csv`.
   - **Cell 5**: Runs `make_splits.py` to generate `results/splits.json`.
   - **Cell 6**: Runs `run_tune.py` (skips automatically if `best_params.json` exists).
   - **Cell 7**: Runs `run_train.py` consuming optimal hyperparameters with `--resume` and `--ckpt_dir /content/checkpoints`.
   - **Cell 8**: Runs `run_explain.py` for local explanation visualizations.
   - **Cell 9**: Runs `run_faithfulness.py` ($N=100$ test emails).
   - **Cell 10**: Runs `run_ablation.py` ($N=50$ test emails across $\alpha$ values).
   - **Cell 11**: Generates `results/RESULTS.md` and synchronizes all artifacts to Google Drive.

---

## 7. Changelog

- **2026-10-18 (Phase 0)**: Initialized repository structure, pinned dependencies in `requirements.txt`, verified Python 3.10 on `Windows-10-10.0.26300-SP0` and CPU constraints.
- **2026-10-18 (Phase 1)**: Built data ingestion, text cleaning, deduplication, conflict removal (533 `'empty'` rows dropped), double stratified splitting, and WordNet augmentation.
- **2026-10-18 (Phase 2)**: Added `requirements-colab.txt`, model initialization, dynamic padding, mixed-precision training loop, validation F1 checkpointing, single-pass test evaluation, plotting, and random search tuning.
- **2026-10-18 (Phase 3)**: Added `scripts/download_data.py` and `scripts/make_splits.py` (generated `results/splits.json`). Implemented consolidated reporting (`report.py`, `RESULTS.md`), strict debug isolation under `results/debug/`, `--ckpt_dir` support, `latest_checkpoint.pt` auto-cleanup, Captum fallback parity, and `word_ids` subword aggregation.
- **2026-10-18 (Phase 4)**: Implemented `src/phishing_xai/lita.py` (supporting both min-max and magnitude-preserving scaling), LIME probability/batching wrapper in `explain.py`, and `scripts/run_explain.py`.
- **2026-10-19 / 2026-10-20 (Phase 5)**:
  - Fixed pre-tokenization window truncation (`truncate_text_to_model_window`) and pre-tokenization char cap.
  - Implemented shared regex word splitter (`WORD_SPLIT_REGEX`, `split_words`) used across LIME and token aggregation.
  - Added near-duplicate template grouping via `StratifiedGroupKFold` on normalized first-200-char hashes (0% near-dup cross-split leakage).
  - Enhanced Captum fallback with exact parity to Transformers-Interpret (Spearman rank correlation > 0.9 and top-5 tokens match).
  - Implemented faithfulness evaluation framework (`faithfulness.py`, `run_faithfulness.py`) measuring Comprehensiveness and Sufficiency at $k \in \{1, 3, 5\}$ and 5-seed random baseline.
  - Implemented alpha parameter ablation study (`run_ablation.py`) across $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
  - Implemented global explanations extraction for Safe and Phishing email classes.
  - Configured `colab_run.ipynb` with stage-by-stage Google Drive backup and complete execution workflow.
  - Enforced strict `--params` validation in `run_train.py` outside debug mode.
  - Expanded unit test suite to **41 tests** with 100% pass rate.

---

## 8. Multimodal Extension Decisions (2026-11-12)

| Decision | Reason | Status |
|---|---|---|
| Add `pytesseract==0.3.13`, `pdfplumber==0.11.10`, `PyMuPDF==1.28.2`, `pyzbar==0.1.9`, `Pillow==12.3.0`, `qrcode==8.2` to `requirements.txt` and `requirements-colab.txt` | Needed for OCR + bounding boxes, PDF text/links, scanned-PDF rendering, and QR code generation/decoding. | Complete |
| Install Tesseract `v5.4.0.20240606` in user space (`C:\Users\admin\AppData\Local\Programs\Tesseract-OCR`) and auto-detect in `extract.py` | Avoids UAC prompt hang on Windows while providing full native Tesseract OCR support. (Historical record: that path is the original author's Windows machine, 2026-11-12, not a path in this repo; Section B (B7) removed hard-coded user paths from `extract.py`.) | Complete |
| Create `src/phishing_xai/extract.py` with CLI `python -m phishing_xai.extract <file.eml>` | Walks MIME parts, OCRs images (`image_to_data`), extracts PDF text/links (`pdfplumber`), renders scanned PDF pages (`PyMuPDF`) for OCR fallback, and decodes QR codes (`pyzbar`). | Complete |
| Create `scripts/make_multimodal_samples.py` and `tests/test_extract.py` | Deterministically generates 6 synthetic `.eml` files in `samples/multimodal/` and verifies extraction, OCR keywords, PDF links, scanned-PDF OCR fallback, QR decoding, and visual bounding boxes. | Complete |
| No retraining; `model.py` and `faithfulness.py` stay unchanged; existing 49 tests stay green | Hard project constraints; verified with `pytest -v` (49/49 passing). | Complete |
| `results/best_model.pt` missing locally check + `--allow_pretrained_fallback` | Full training is deferred to Colab (`results/best_model.pt` not present locally). By default, `run_multimodal_predict.py` and `run_multimodal_faithfulness.py` stop and report `results/best_model.pt` is missing; `--allow_pretrained_fallback` runs with base `roberta-base` via `model.py` for end-to-end demonstration. (Full training is now configured for Kaggle, not Colab — see Section F and Section 10; the fallback behaviour itself is unchanged, and Section A (A2) added the loud UNTRAINED MODEL banner and the `"untrained_fallback": true` JSON flag.) | Complete |
| Verdict rule = max over parts (phishing if ANY part scores phishing) | Implemented in `scripts/run_multimodal_predict.py`, naming the winning part and explaining it with LITA (`--explain`). (Superseded 2026-11-19 by Section D: weighted fusion is now the default verdict and the max rule survives only as a selectable `--fusion max` baseline — see Section 10.) | Complete |
| Append `draw_lita_boxes_on_image` to `src/phishing_xai/explain.py` | Draws red bounding boxes around top-5 LITA words on image parts using OCR boxes and saves `<name>_explained.png` without altering any existing functions. | Complete |
| Reuse `faithfulness.py` in `scripts/run_multimodal_faithfulness.py` | Evaluates Comprehensiveness and Sufficiency at $k=5$ of LITA on the extracted text of all 6 multimodal samples. | Complete |

---

## 9. Section A — Correctness Fixes (2026-11-19)

| Decision | Reason | Status |
|---|---|---|
| **A1. LITA alpha corrected to paper Eq. 2** (`alpha*LIME + (1 - alpha)*TI`) | `combine_scores` had the direction inverted (`(1 - alpha)*LIME + alpha*TI`); alpha=0 produced pure LIME and alpha=1 pure TI, opposite to the paper (and to the legacy notebooks, which already use `alpha * lime + (1 - alpha) * TI`). Fixed the formula and all docstrings (`lita.py`, `config.py`, `run_explain.py` help, `report.py` ablation header); updated the two tests that encoded the inverted direction. `alpha=0.5` results are unaffected (both formulas give `0.5*LIME + 0.5*TI`); pre-fix ablation endpoint labels must be swapped before citation. | Complete |
| **A2. Loud untrained-fallback warning + `untrained_fallback` flag in output JSON** | `--allow_pretrained_fallback` previously only logged a quiet warning while producing verdicts from untrained base `roberta-base`. `load_trained_roberta` now prints a banner ("WARNING: UNTRAINED MODEL - results are NOT detection evidence") and every output JSON carries `"untrained_fallback": true` (`false` with the real checkpoint). | Complete |
| **A3. Docs corrected to `results/splits.json` as source of truth** | `decision.md` said 17,532 rows / splits 11,220-2,806-3,506 / SHA-1 / 100,000-char cap; code and `splits.json` say 17,513 / 11,215-2,803-3,495 / MD5 (first 200 alphanumerics) / 50,000-char cap. Fixed in `decision.md`, `PROJECT_LOG.md`, `README.md`, `docs/MULTIMODAL_NOTE.md`; `RESULTS.md`/`summary.json` already matched. The `796` near-duplicate-groups figure is unverified (no script logs a group count) and was removed. Test count corrected to **61** at end of Section A work (49 Section A baseline + 12 Section B tests that landed mid-session; 41 was the Phase 5 count, +8 `test_extract.py` tests = 49; count will change again as sections B/C/D/E/H add tests). `results/debug/multimodal_predictions.json` is NOT in the repo (`results/debug/` is gitignored). | Complete |
| **A4. Consolidated report merges `results/metrics.json` when present** | Verified, no code change needed: `generate_consolidated_report` already merges `metrics.json` (skipping `"debug": true` files); verified with a throwaway `/tmp` copy with and without a dummy `metrics.json` (dummy deleted; no fake `results/metrics.json` in the repo). `NOT RUN YET` in committed `RESULTS.md`/`summary.json` for unrun stages is genuine and left in place. | Complete |
| **A5. Precision is the priority focus (user decision, recorded 2026-11-19)** | User: "mera priority focus precision hai." Primary metric is phishing-class precision (minimise false positives / low FPR); F1/accuracy/recall are secondary — reported, never optimised/selected on alone. Thresholds fit on validation only, never test. Governs Sections D/E/H (fusion threshold, eval tables, baseline ranking/selection), which implement it. Section A records it here and makes precision prominent in reporting: `report.py` Section 4 now leads with phishing-class precision + FPR (FPR shown only when the metrics artifact records it). | Complete |




---

## 10. Sections B–H — Multimodal and Baseline Work (2026-11-19)

Full change logs live in `.impl_logs/section_B.md` through `.impl_logs/section_H.md` and `.impl_logs/integration.md`. This section is the merged record (decision, reason, status). Nothing here was pushed, nothing was retrained, and `src/phishing_xai/model.py` and `src/phishing_xai/faithfulness.py` were not modified by any of these sections. Anything not executed is **NOT RUN YET**. The same merge is recorded in `decision.md` Section 4.

| Decision | Reason | Status |
|---|---|---|
| **B1. QR decode + OCR on embedded images in every PDF page** (`pdf_image` parts, `"<pdf>#img<xref>"`; near-duplicate embedded images skipped unless they carry a QR code) | QR codes embedded in text PDFs were previously invisible (QR ran only on scanned-page renders). | Complete; 12 new tests in `tests/test_extract.py` (20 total there). |
| **B2. HTML bodies parsed structurally** (anchor/href pairs, mismatch flag, base64 images, hidden text in its own part, form actions, `cid:` images; remote images recorded as links only, never fetched) | `get_text()` flattened links away and silently dropped hidden text. | Complete |
| **B3. Headers + subject as a scored part** (Subject, From, Reply-To, Return-Path, Message-ID, Authentication-Results, SPF/DKIM/DMARC verdicts; subject is a `subject` part, `body_text` unchanged) | Header and subject signals were invisible to scoring. | Complete |
| **B4. More attachment types** (`.html`, `.docx`, nested `message/rfc822` to depth 3, `.zip` member listings only with risky-extension flags, `.svg`, plain-text attachments; unsupported types get a `skipped` entry) | Those attachments were silently dropped. Known limit: a nested message attached as raw base64 `.eml` bytes is not recovered (the stdlib parser itself does not interpret it as a message). | Complete |
| **B5. Safety limits as module constants** (15 MB attachments, 20 PDF pages, 50 parts per mail, Pillow pixel limit, 500 zip members, 100 MB uncompressed zip, nested depth 3; oversize items skipped, never a crash; no signal-based timeout) | Unbounded inputs could hang or exhaust memory on adversarial mail. | Complete |
| **B6. Image preprocessing before OCR/QR** (deterministic variants, original tried first; OpenCV `QRCodeDetector` as a guarded fallback) | Clean-image behaviour stays unchanged while stressed images get a second chance. | Complete |
| **B7. Tesseract discovery without hard-coded user paths** (`$TESSERACT_CMD`, system path, platform defaults only; missing binary degrades to empty OCR text) | Hard-coded user paths break on any other machine. | Complete |
| **B8. Backward-compatible extraction schema** (`body_text` and original part keys unchanged; new top-level `headers`, `links`, `flags`, `skipped`; new part kinds `subject`, `pdf_image`, `html_text`, `hidden_text`, `docx`, `zip_listing`, `svg_text`, `attachment_text`) | Downstream scripts keep working; the predict script now also scores subject and hidden-text parts, which is intended. | Complete |
| **C. New pure-static lexical URL risk module** (`src/phishing_xai/url_risk.py`: `score_url` / `score_urls` / `extract_urls` / `registered_domain`, `UrlRiskConfig`; no fetching, no DNS; `tldextract` only via a guarded import with a bundled-suffix fallback) | URL evidence must be independent of the text model and must never fetch a link. | Complete; 39 tests in `tests/test_url_risk.py`, all passing. |
| **D. Precision-first fusion (user decision 2026-11-19: "mera priority focus precision hai")** — primary metric phishing-class precision, governs D/E/H; weighted fusion is the default (0.45 context, 0.20 best part, 0.20 URL, 0.15 flags) with a corroboration gate (two channels >= 0.5, or one channel + flag score >= 0.30, or fused score >= 0.75); threshold 0.5 is a PLACEHOLDER to be tuned on validation only; max rule kept only as a selectable `--fusion max` baseline; LITA runs on the concatenated context with PDF/QR overlay helpers; Kaggle auto-detection added to the predict script. | The old max rule flips on any part and is recall-heavy / precision-poor by construction; a single noisy part must not flip a verdict. | Complete; 37 tests in `tests/test_multimodal_fusion.py`, all passing. D smoke run 6/6 rows, all untrained fallback, all Safe — pipeline demonstration only. End-to-end overlay for a PDF/QR winning part and threshold tuning are NOT RUN YET (need the trained checkpoint). |
| **E. Synthetic eval set + precision-first ablation** — 120 mails (60 phishing / 60 benign, 24 validation / 96 test) in `data/multimodal_eval/`; thresholds fitted on validation only (recall floor 0.80); ablations (a)–(d). Smoke run (untrained fallback, NOT detection evidence): (d) weighted precision 1.00 at recall 0.75 (F1 0.86, FPR 0.00, accuracy 0.88, n=96), confusion `[[48, 0], [12, 36]]`, separation from URL/flag channels, not the model. TF-IDF+LR fitted on the real-data train split (11,215 rows) and evaluated on the 120 synthetic mails: precision 0.95, recall 0.95, F1 0.95, ROC-AUC 0.9875, FPR 0.05, `[[57, 3], [3, 57]]`. Faithfulness script gained a random baseline per sample and a `no_text_parts` guard; LITA ≈ random in the smoke run, the expected untrained artefact. | No real labelled multimodal set exists locally; the synthetic caveat is written into the manifest, the JSON metadata and the report. | Complete (code + smoke runs); real detection numbers and the BiLSTM baseline are NOT RUN YET. |
| **F. Kaggle notebook + packaging** — `notebooks/kaggle_run.ipynb` (23 cells) is now the primary run path (GPU T4 + Internet ON, repo attached as a dataset, outputs under `/kaggle/working/results/`); the Colab notebook is kept but no longer primary. `requirements-kaggle.txt` has no torch line; `requirements.txt` rewritten from a 78-line frozen dump to 23 direct lines with `pyproject.toml` canonical (added `transformers-interpret`, `scipy`, `Pillow`, new `[multimodal]` extra); `.gitignore` gained `data/*.csv` and `results/best_model.pt`. | Colab is no longer the execution environment; a frozen `pip freeze` dump is not a dependency declaration. | Complete (code); Kaggle execution NOT RUN YET. |
| **F finding: `run_baselines.py` rewrote `baseline_results.json` on every invocation** (verified in code; the complexity report therefore only saw the most recent run). | A silent overwrite would have made the accumulated ladder unrecoverable on Kaggle. | Found in F; fixed in G (see below). |
| **H. Baseline & complexity-justification study** — ladder (dummy, TF-IDF+LR, TF-IDF char-SVM, BiLSTM, 1D-CNN, DistilBERT, RoBERTa eval-only) on identical splits, ranked by precision first, recommended model = highest precision subject to recall floor 0.85; McNemar, bootstrap precision CIs, error analysis, robustness, faithfulness at k in {1,3,5}; code-written verdict rule (precision-gap headline, F1 secondary). H5 note: `evaluate_faithfulness_single_method` cannot be reused for non-transformers (transformer-only signature), so `evaluate_faithfulness_with_proba_fn()` implements the identical maths over any `predict_proba` callable. RoBERTa robustness and the DistilBERT validation-threshold view are recorded as missing, not faked. | The complexity question — is the transformer justified? — needs a fair ladder, not a single comparison. | Complete (code); full execution NOT RUN YET (only `--debug` smoke runs on 200/100/100 subsets; those numbers are never quoted as results). No committed `results/baselines/baseline_results.json` or `results/complexity_report.md` exists yet. |
| **G. Documentation + integration fix (this section)** — README rewritten around Kaggle; `docs/MULTIMODAL_NOTE.md` rewritten with a mermaid architecture diagram and a results section filled only from real runs; threat model & limitations section added to the README and cross-linked; `run_baselines.py` now merges `baseline_results.json` per model+seed by default (`merge_baseline_results`, `--no_merge` to overwrite), covered by 2 new tests (36 in `tests/test_baselines.py`). Test functions recounted at the end of the work: **193**. | The old README still described Colab as primary and the max rule as the verdict; the merge fix stops separate Kaggle baseline invocations from destroying each other. | Complete (docs + fix + unit tests); accumulating a real ladder across Kaggle invocations is not yet exercised. |
| **Pre-existing test failures (environment, not code)** — `tests/test_sanity.py::test_environment_sanity` asserts Python 3.10 while the local venv runs Python 3.12.3; `tests/test_explain_and_resumability.py::test_captum_vs_transformers_interpret_tiny_model` fails because `transformers_interpret` is not installed locally (it would force a transformers downgrade from 5.23; Kaggle requirements pin transformers 4.x, where it is expected to work). Both predate this work; neither is caused by any section change. Full-suite result at the end of Section G is recorded in `.impl_logs/section_G.md`. | Fixing them by changing project code locally would be the wrong repair; they are to be verified on Kaggle. | Recorded, not fixed. |
