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
6. **LITA Normalization Distortion**: The original LITA score min-max normalized token attributions to $[-1, 1]$ before linear combination ($(1 - \alpha) \cdot \text{LIME} + \alpha \cdot \text{TI}$). This artificially forced the smallest score to $-1$ and largest to $+1$ regardless of original magnitude.
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
| **LITA variants: min-max + magnitude-preserving** | Retains original min-max formulation for experimental fidelity while introducing a magnitude-preserving variant ($\alpha=0.0 \implies \text{LIME}, \alpha=1.0 \implies \text{TI}$). | Phase 4 | Active |
| **Faithfulness evaluation protocol & Alpha Ablation** | Measures comprehensiveness and sufficiency at $k \in \{1, 3, 5\}$ across LIME, TI, LITA (both variants), and a 5-seed random baseline; evaluates $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$. | Phase 5 | Active |
| **Colab stage-by-stage Google Drive backup** | Backs up artifacts to Google Drive after EACH pipeline stage to eliminate data loss from unexpected disconnection. | Phase 5 | Active |
| **OneDrive path & checkpoint storage safeguards** | Repository resides inside OneDrive. Added `--ckpt_dir` to store checkpoints outside OneDrive and auto-delete intermediate `latest_checkpoint.pt` after successful runs (unless `--keep_latest`). | Phase 3 | Active |

---

## 4. Work Completed Per Phase

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
