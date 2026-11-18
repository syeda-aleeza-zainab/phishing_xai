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

---

## Acknowledgements

- **Original Base Repository**: This project builds upon and scientifically refactors the initial research codebase from [Amazuddin/Phishing_Email_Detection_XAI](https://github.com/Amazuddin/Phishing_Email_Detection_XAI).
- **Dataset**: Kaggle "Phishing Emails" dataset created by `subhajournal` ([Kaggle Dataset Link](https://www.kaggle.com/datasets/subhajournal/phishingemails)), mirrored on Hugging Face at `zefang-liu/phishing-email-dataset`.
