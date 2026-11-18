# Phishing_Email_Detection_XAI

A production-grade, reproducible refactoring and scientific benchmarking of RoBERTa-based phishing email detection with Explainable AI (LIME, Transformers-Interpret / Captum Integrated Gradients, and LITA hybrid attributions).

---

## Status & Experimental Results

- **Production Status**: Production model fine-tuning (5 epochs $\times$ 14,078 balanced samples), full hyperparameter search (10 trials $\times$ 4,000 samples), and XAI faithfulness benchmarks are configured for Google Colab GPU execution ([`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb)) with automated stage-by-stage Google Drive backups.
- **Debug Isolation**: All local metrics from fast CPU `--debug` runs are strictly isolated to `results/debug/` and are never reported as production findings.
- **Reports**: See [`results/RESULTS.md`](results/RESULTS.md) for live dataset hygiene counts and production results (unexecuted stages are explicitly marked `*NOT RUN YET*`).
- **Decisions Log**: For complete architecture decisions, legacy bug analysis, and technical changelogs, see [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).
- **Test Suite**: 41 comprehensive unit and integration tests passing (`pytest -v`).

---

## Quickstart

### Environment Setup (Python 3.10)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Fast CPU Smoke-Test Pipeline
```powershell
# 1. Download and strictly verify dataset
python scripts/download_data.py

# 2. Preprocess, deduplicate, group near-duplicates, and generate split manifest
python scripts/make_splits.py

# 3. Quick CPU hyperparameter tuning (~2.5 minutes)
python scripts/run_tune.py --debug

# 4. Quick CPU fine-tuning (~14 minutes)
python scripts/run_train.py --debug --params results/tuning/best_params.json

# 5. Local explainability visualizations (LIME, TI, LITA)
python scripts/run_explain.py --debug

# 6. Faithfulness evaluation (Comprehensiveness, Sufficiency, Random Baseline)
python scripts/run_faithfulness.py --debug

# 7. Alpha parameter ablation study on LITA
python scripts/run_ablation.py --debug

# 8. Generate consolidated report
python scripts/make_report.py

# 9. Run full test suite (41 tests)
pytest -v
```

### Full Production Training (Google Colab GPU)
For full-budget model fine-tuning, hyperparameter optimization, and faithfulness benchmarking on GPU, execute [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) following the sequential cell order:
1. **GPU Assertion**: Asserts CUDA GPU availability (`!nvidia-smi`).
2. **Drive Mount**: Mounts Google Drive and creates `/content/drive/MyDrive/phishing_xai_results`.
3. **Dependencies**: Installs `requirements-colab.txt` and editable package.
4. **Data Download & Splits**: Executes `download_data.py` and `make_splits.py`.
5. **Hyperparameter Tuning**: Runs `run_tune.py` (skips if tuning results exist) and backs up to Drive.
6. **Final Training**: Runs `run_train.py --params results/tuning/best_params.json --resume --ckpt_dir /content/checkpoints`.
7. **Explainability**: Runs `run_explain.py` for local explanation visualizations.
8. **Faithfulness Benchmarking**: Runs `run_faithfulness.py` evaluating Comprehensiveness and Sufficiency at $k \in \{1, 3, 5\}$.
9. **Alpha Ablation**: Runs `run_ablation.py` across $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
10. **Report Generation**: Runs `make_report.py` and synchronizes all artifacts to Google Drive.

### Multimodal extension (image/PDF phishing)
Extends the trained RoBERTa detector and LITA explainer to handle image-only, QR-code, text-PDF, and scanned-PDF phishing emails without retraining or modifying `model.py` or `faithfulness.py`:

- [`src/phishing_xai/extract.py`](src/phishing_xai/extract.py): Parses `.eml` MIME parts (`stdlib email`), runs `pytesseract` OCR with word-level bounding boxes (`image_to_data`), extracts PDF text/links via `pdfplumber` with a `PyMuPDF` (`fitz`) page-render + OCR fallback for scanned PDFs, and decodes QR payloads via `pyzbar`.
- [`scripts/make_multimodal_samples.py`](scripts/make_multimodal_samples.py): Deterministically generates 6 synthetic `.eml` samples in `samples/multimodal/` covering image-only phishing, inline button images, QR phishing, text PDF invoices, scanned image-only PDFs, and benign attachments.
- [`scripts/run_multimodal_predict.py`](scripts/run_multimodal_predict.py): Scores `body_text` and each extracted attachment part with RoBERTa, applies the **max rule** (phishing if any part scores phishing), and with `--explain` computes top-5 LITA words and saves `<name>_explained.png` with red bounding boxes via `draw_lita_boxes_on_image` in [`src/phishing_xai/explain.py`](src/phishing_xai/explain.py).
- [`scripts/run_multimodal_faithfulness.py`](scripts/run_multimodal_faithfulness.py): Reuses [`src/phishing_xai/faithfulness.py`](src/phishing_xai/faithfulness.py) to compute Comprehensiveness and Sufficiency ($k=5$) of LITA on the extracted text of the 6 samples.
- [`docs/MULTIMODAL_NOTE.md`](docs/MULTIMODAL_NOTE.md): One-page summary of the multimodal extension, 6-sample results table, and limitations.

```powershell
# 1. Generate the 6 multimodal .eml samples
python scripts/make_multimodal_samples.py

# 2. Extract structured JSON from a single .eml file
python -m phishing_xai.extract samples/multimodal/sample1_image_only_phish.eml --no_boxes

# 3. Score all 6 samples and generate LITA explanations + bounding-box overlays
python scripts/run_multimodal_predict.py --explain

# 4. Evaluate LITA faithfulness (top-k=5) on the 6 multimodal samples
python scripts/run_multimodal_faithfulness.py
```

---

## Acknowledgements

- **Original Base Repository**: This project builds upon and scientifically refactors the initial research codebase from [Amazuddin/Phishing_Email_Detection_XAI](https://github.com/Amazuddin/Phishing_Email_Detection_XAI).
- **Dataset**: Kaggle "Phishing Emails" dataset created by `subhajournal` ([Kaggle Dataset Link](https://www.kaggle.com/datasets/subhajournal/phishingemails)), mirrored on Hugging Face at `zefang-liu/phishing-email-dataset`.
