# Phishing Email Detection with Explainable AI (RoBERTa + LIME / Transformers-Interpret / LITA)

Working code submitted with the project report. This is the implementation of the selected paper:

> M. A. Uddin, M. Mahiuddin, I. H. Sarker, "An explainable transformer-based model for phishing email
> detection: A large language model approach", *Computer Networks* 277 (2026) 112061.
> DOI: 10.1016/j.comnet.2026.112061

The implementation fine-tunes RoBERTa on the Kaggle "Phishing Emails" dataset (18,650 emails) and explains
its predictions with LIME, Transformers-Interpret (Integrated Gradients) and the paper's hybrid LITA
method (Eq. (2): H = alpha * LIME_norm + (1 - alpha) * TI_norm, alpha = 0.5).

## Results obtained with this code (executed on Kaggle, Tesla T4, 22 October 2026)

| Run | Protocol | Test accuracy | Phishing F1 | ROC-AUC |
|---|---|---|---|---|
| Paper (reported) | paper's protocol | 98.45% | 0.98 | curves only |
| Phase 3 (3 seeds: 42/1/7) | paper's protocol, exact replication | **97.80 ± 0.07%** | 0.9724 | 0.9981 |
| Phase 4 | clean protocol, paper hyperparameters (batch 32) | 98.57% | 0.9804 | 0.9993 |
| Phase 1 | clean protocol, tuned (batch 16) | 99.09% | 0.9876 | 0.9992 |

The headline implementation result is the Phase 3 replication: on the paper's own test set (3,727 emails)
the implementation scores 97.80 ± 0.07%, slightly below the paper's reported 98.45%.

## Project structure

- `src/phishing_xai/` — the package: `data.py` (loading, cleaning, deduplication, splits),
  `augment.py` (WordNet synonym augmentation / class balancing), `model.py`, `train.py`
  (validation-F1 checkpointing, single final test evaluation), `tune.py` (random search),
  `evaluate.py`, `explain.py` (LIME + Transformers-Interpret), `lita.py` (LITA hybrid),
  `faithfulness.py` (comprehensiveness / sufficiency), `report.py`, `config.py`, `utils.py`.
- `scripts/` — command-line entry points (see run order below).
- `tests/` — 41 unit tests (`pytest`).
- `data/Phishing_Email.csv` — the dataset (Kaggle "Phishing Emails", subhajournal).
- `notebooks/`, top-level `.ipynb` files — the original Colab notebooks (tuning + LIME/TI/LITA).
- `results/` — data-stage summary (`summary.json`, `RESULTS.md`).
- `README_SUBMISSION.md` — this file. (The archive's original `README.md` is kept unchanged.)

## Requirements

- Python 3.10+ and a CUDA GPU for training (a free Kaggle/Colab T4 is sufficient).
- Install dependencies:

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"
```

Key versions used for the executed runs: PyTorch 2.x (CUDA), transformers 4.57.0, captum >= 0.7,
lime 0.2.0.1, transformers-interpret 0.10.0, scikit-learn 1.6.1.

## How to run (in order)

```bash
cd Phishing_Email_Detection_XAI

# 1. Build the deduplicated, leakage-controlled splits
python scripts/make_splits.py

# 2. (Optional) hyperparameter random search on the proxy budget
python scripts/run_tune.py

# 3. Train RoBERTa (writes results/metrics.json, results/best_model.pt, plots)
python scripts/run_train.py --params <best_params.json>

# 4. Explain test emails with LIME / Transformers-Interpret / LITA
python scripts/run_explain.py --checkpoint_path results/best_model.pt --num_samples 10

# 5. Faithfulness of the explainers (comprehensiveness / sufficiency)
python scripts/run_faithfulness.py --checkpoint_path results/best_model.pt --sample_size 100

# 6. LITA alpha ablation
python scripts/run_ablation.py --checkpoint_path results/best_model.pt --sample_size 50

# 7. Regenerate results/RESULTS.md and results/summary.json
python scripts/make_report.py --results_dir results

# Tests
pytest tests/
```

To reproduce the paper-protocol replication (Phase 3, 97.80 ± 0.07%): use the paper's own protocol —
null-drop and cleaning only (no deduplication), a plain stratified split (seed 42; train 11,925 /
validation 2,982 / test 3,727), WordNet balancing of the training split only (+2,565 -> 14,490), and the
paper's Table 1 hyperparameters (batch 32, AdamW, lr 2e-5, 5 epochs, max length 512).

## Notes for the marker

- Two deviations of the submitted code from the paper are documented in the accompanying report:
  D1 — the LITA alpha weighting in `lita.py` is mirrored relative to Eq. (2) (identical at
  alpha = 0.5); the executed runs patched it in-kernel. D2 — the default LITA variant is
  magnitude-preserving; the paper's min-max variant is available via `--variant minmax` and was used
  for the headline explanations.
- `scripts/run_explain.py` does not truncate long emails to RoBERTa's 512-token window; the executed
  Kaggle run applied a truncation hotfix in the notebook (documented in the full comparison report,
  Appendix C).
