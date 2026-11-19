# Phishing_Email_Detection_XAI

A production-grade, reproducible refactoring and scientific benchmarking of RoBERTa-based phishing email detection with Explainable AI (LIME, Transformers-Interpret / Captum Integrated Gradients, and LITA hybrid attributions). The detector is now extended, without retraining, to image, QR-code and PDF phishing in `.eml` mail (see the multimodal section below).

---

## Status & Experimental Results

- **Production Status**: Production model fine-tuning (5 epochs $\times$ 14,078 balanced samples), full hyperparameter search (10 trials $\times$ 4,000 samples), and XAI faithfulness benchmarks are configured for Kaggle GPU execution ([`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb), GPU T4, Internet ON). The older Colab notebook ([`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb)) is kept in the repo but is no longer the primary path.
- **Primary metric**: phishing-class **precision** (user decision, 2026-11-19: minimise false positives and the false-positive rate). F1, accuracy and recall are reported alongside precision everywhere, but no model, threshold or fusion is selected on F1 alone. Every threshold is fitted on **validation only, never test**.
- **Debug Isolation**: All local metrics from fast CPU `--debug` runs are strictly isolated to `results/debug/` and are never reported as production findings.
- **Reports**: See [`results/RESULTS.md`](results/RESULTS.md) for live dataset hygiene counts and production results (unexecuted stages are explicitly marked `*NOT RUN YET*`). The dataset behind those counts is locked in [`results/splits.json`](results/splits.json): 17,513 retained emails, split 11,215 train / 2,803 validation / 3,495 test, with train-only WordNet augmentation adding 2,863 rows to reach 14,078.
- **Decisions Log**: For complete architecture decisions, legacy bug analysis, and technical changelogs, see [`decision.md`](decision.md) and [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).
- **Kaggle run results (2026-11-19, all executed on Tesla T4)**: RoBERTa test precision **0.9690** / recall 0.9932 / F1 0.9809 / accuracy 0.9854 (n=3,495, best epoch 3); checkpoint in private dataset `syedaaleezazainab/phishing-xai-checkpoint`. Ladder: TF-IDF+SVC **0.9820** precision / 0.9856 F1 beats RoBERTa on precision and F1 (McNemar p=0.253, not significant); tuned BiLSTM 0.9719 / 0.9571. Multimodal (synthetic 120, test n=96): full weighted fusion **P 1.00 / R 0.9792 / F1 0.9895 / FPR 0.00**; BiLSTM backbone P 1.00 / R 1.00. Code-generated 6B.5 verdict: "significant precision gain of 5.88 points on the multimodal held-out set (RoBERTa 1.0000 vs TF-IDF+LR 0.9412; McNemar p = 0.6250) at 3116.1x parameters"; text-only verdict -1.30 points vs TF-IDF+SVC (p=0.253). Full numbers in [`results/RESULTS.md`](results/RESULTS.md); all stage outputs in results dataset `syedaaleezazainab/phishing-xai-results` v7.
- **Test Suite**: 193 test functions (`grep -r "def test_" tests/ | wc -l`; run with `.venv/bin/python -m pytest -q`). Two pre-existing environment failures are documented in the limitations below and are not caused by any project change.

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

# 9. Run full test suite (193 tests)
pytest -v
```

### Full Production Training (Kaggle GPU — the primary path)
Run [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb) top to bottom, cells in order, with these settings:

1. **Accelerator: GPU** — Settings -> Accelerator -> GPU (T4 or P100).
2. **Internet: ON** — Settings -> Internet -> ON (needed for apt-get, pip, and the dataset download in the notebook).
3. **Attach the repo as a dataset** — Add Input -> upload a zip or folder of this repo (it must contain `scripts/run_train.py` and `requirements-kaggle.txt`). It appears read-only under `/kaggle/input/<dataset-name>/`; the notebook copies it to writable `/kaggle/working` and works from there.
4. **(Optional) Attach the data and checkpoint** — if `data/Phishing_Email.csv` is inside the repo dataset the notebook uses it, otherwise attach the CSV as a second dataset or let the notebook download it from the Hugging Face mirror. Attach a dataset containing `best_model.pt` to skip training (the notebook copies it to `/kaggle/working/results/best_model.pt`).
5. **Run the cells in order** — GPU check, install (`requirements-kaggle.txt`, plus `apt-get install -y tesseract-ocr libzbar0` for OCR and QR decoding), data and splits (train 11,215 / validation 2,803 / test 3,495), optional tuning, training with `--resume`, explainability, faithfulness, alpha ablation, the multimodal pipeline, the baseline ladder, and the complexity report. Tuning, the CNN/DistilBERT baselines and the consolidated single-invocation baseline run are optional cells; training can be skipped only if a checkpoint was attached.
6. **Save output** — Save Version with output saved, so `/kaggle/working/results/` persists. To reuse the checkpoint later, either attach that saved run output as an input dataset in the next notebook, or download `results/best_model.pt` and upload it via Datasets -> New Dataset.

The notebook depends on `requirements-kaggle.txt` (unpinned minimums matching Kaggle's preinstalled stack, with no torch line because Kaggle ships a CUDA torch build). The older Colab notebook (`notebooks/colab_run.ipynb`, Google Drive backups) is kept for reference but is not the path this README documents.

### Multimodal extension (image / QR / PDF phishing)
The trained RoBERTa detector and LITA explainer are extended to image-only, QR-code, text-PDF and scanned-PDF phishing emails without retraining and without modifying `src/phishing_xai/model.py` or `src/phishing_xai/faithfulness.py`. The pipeline runs in three stages, extract, then URL risk, then fusion:

- **Extract** ([`src/phishing_xai/extract.py`](src/phishing_xai/extract.py)): parses `.eml` MIME parts, runs `pytesseract` OCR with word-level bounding boxes, extracts PDF text and links via `pdfplumber` with a `PyMuPDF` page-render and OCR fallback for scanned PDFs, decodes QR payloads via `pyzbar`, and now also parses HTML bodies structurally (anchor versus href mismatch, hidden text kept separate from body text, base64 images, forms, `cid:` images), reads headers and the subject as a scored part, and handles `.docx`, nested messages, `.zip` listings and `.svg` attachments under documented safety limits. It returns `body_text`, `parts`, `headers`, `links`, `flags` and `skipped`.
- **URL risk** ([`src/phishing_xai/url_risk.py`](src/phishing_xai/url_risk.py)): a purely static lexical scorer (no fetching, no DNS). It scores each URL 0 to 1 from signals such as dangerous schemes, IP hosts, punycode, suspicious TLDs, shorteners, brand-keyword mismatch, high-entropy tokens and QR-only origin.
- **Fusion** ([`src/phishing_xai/multimodal.py`](src/phishing_xai/multimodal.py)): builds one concatenated context (`[SUBJECT] [BODY] [HIDDEN] [OCR] [PDF] [ATTACHMENT] [QR_URL]`) scored by RoBERTa, scores the worst URL and the structural flags, and fuses them. The default `--fusion weighted` uses weights 0.45 context, 0.20 best part, 0.20 URL, 0.15 flags, and a phishing verdict additionally needs corroboration across channels (at least two channels at 0.5 or more, or one channel plus a flag score of 0.30 or more, or a fused score of 0.75 or more), so a single noisy part cannot flip the verdict on its own. `--fusion max` keeps the old any-part rule as a recall-heavy baseline for comparison. The decision threshold is **0.5 as a placeholder**; it must be tuned on validation only, choosing the value that maximises phishing-class precision subject to a recall floor, and no tuned value is claimed anywhere until that run exists.
- **Evaluation** ([`scripts/make_multimodal_eval_set.py`](scripts/make_multimodal_eval_set.py), [`scripts/run_multimodal_eval.py`](scripts/run_multimodal_eval.py)): a deterministic synthetic eval set of 120 `.eml` files (60 phishing / 60 benign, 24 validation / 96 test) in `data/multimodal_eval/`, evaluated with precision-first ablation tables for text-only, text plus OCR, text plus OCR plus QR plus PDF, and full fusion (weighted and max). See [`docs/MULTIMODAL_NOTE.md`](docs/MULTIMODAL_NOTE.md) for the architecture diagram, the results that come from real runs, and what is still not run.
- **Baselines and complexity** ([`src/phishing_xai/baselines.py`](src/phishing_xai/baselines.py), [`scripts/run_baselines.py`](scripts/run_baselines.py), [`scripts/run_complexity_report.py`](scripts/run_complexity_report.py)): a baseline ladder (dummy, TF-IDF logistic regression and SVM, BiLSTM, 1D-CNN, DistilBERT, RoBERTa eval-only) on the identical splits, ranked by precision first, with McNemar tests, bootstrap precision intervals, robustness and faithfulness, and a code-written verdict on whether the transformer is justified. Full runs are **NOT RUN YET**; only `--debug` smoke runs on small subsets exist locally.
- **Samples** ([`scripts/make_multimodal_samples.py`](scripts/make_multimodal_samples.py)): deterministically generates 6 synthetic `.eml` samples in `samples/multimodal/` covering image-only phishing, an inline button image, QR phishing, a text PDF invoice, a scanned image-only PDF, and a benign mail with harmless image and PDF.

```powershell
# 1. Generate the 6 multimodal .eml samples
python scripts/make_multimodal_samples.py

# 2. Extract structured JSON from a single .eml file
python -m phishing_xai.extract samples/multimodal/sample1_image_only_phish.eml --no_boxes

# 3. Score all 6 samples with weighted fusion (default) and LITA explanations + overlays
python scripts/run_multimodal_predict.py --explain

# 4. Evaluate LITA faithfulness (top-k=5, with a random baseline) on the 6 samples
python scripts/run_multimodal_faithfulness.py

# 5. Regenerate the 120-email synthetic eval set and evaluate the full pipeline on it
python scripts/make_multimodal_eval_set.py
python scripts/run_multimodal_eval.py --allow_pretrained_fallback --output_dir results/multimodal_eval
```

Step 5 needs the warning in the limitations below: without the trained checkpoint it runs on untrained base `roberta-base` and its numbers are pipeline demonstration only, never detection evidence.

---

## Threat model & limitations

This section states what the system is built to catch, what it cannot see, and what has not been measured yet. The same points are repeated in [`docs/MULTIMODAL_NOTE.md`](docs/MULTIMODAL_NOTE.md) and [`decision.md`](decision.md); this README copy is the visible one.

**What it is built for.** Ordinary phishing mail whose signal lives in text, links, attachments or structure: deceptive wording, mismatched anchor text, suspicious URLs, QR codes hiding a link, fake invoices in PDF, and scanned images carrying the lure as pixels. The detector reads what the mail contains; it never fetches a URL, resolves DNS, or opens an attachment, so a link is judged only by its spelling and shape.

**What it cannot see.** The RoBERTa model is text-only, so it cannot inspect brand logos, page layout, or visual spoofing directly; a pixel-perfect clone of a login page that OCR reads as clean text looks benign to the model unless its words or links betray it. OCR itself fails on adversarial images: heavy compression, skew, CAPTCHA-style noise, decorative fonts and colour-on-colour banners corrupt tokens, and a missed token means the model and the LITA word match never see it. QR decoding has the same exposure, since a damaged or stylised code yields no payload at all.

**What the evaluation does not prove.** The 120-email eval set is synthetic, built from a small set of templates with simulated OCR stressors, and contains no real emails and no real adversary variation, so its numbers measure pipeline behaviour on those templates only. The 6 sample mails are likewise synthetic demonstrations. The full baseline ladder and the complexity verdict have not been run (only `--debug` smoke runs on 200/100/100 subsets), the production tuning, training and faithfulness stages in `results/RESULTS.md` are marked NOT RUN YET, and the fusion threshold of 0.5 is an untuned placeholder.

**Precision and recall trade-off.** Precision is the priority, so the weighted fusion and its corroboration gate deliberately sacrifice recall to avoid false alarms: a single suspicious part is not enough to call a mail phishing. That choice is right for a detector whose false positives cost trust, but it means some real phishing will be missed, and the size of that miss is unknown until a trained-checkpoint run on labelled data measures it.

**Data age and leakage notes.** The training data is an older Enron-style corpus, so high scores on it do not imply the same performance on modern or AI-written phishing; a harder out-of-distribution test is still needed. Near-duplicate grouping uses the MD5 hash of the first 200 alphanumeric characters to keep template variants in one split, but no script in the repo logs a near-duplicate group count, so no group figure is claimed here.

**Latency.** No latency numbers have been measured for the full pipeline yet. OCR, PDF rendering and per-part scoring add hundreds of milliseconds to several seconds per mail over plain-text inference, but that is an expectation, not a measurement; the baseline ladder is designed to measure latency per email (median of 200, batch size 1) when it runs.

**Untrained-fallback labelling.** Any run made with `--allow_pretrained_fallback` scores with untrained base `roberta-base` (about 0.43 to 0.44 phishing probability on every part) and prints a loud UNTRAINED MODEL warning while writing `"untrained_fallback": true` into every output JSON row. Those numbers demonstrate that the pipeline runs; they are not detection results and must always be quoted with that label.

**Known pre-existing test failures (environment, not code).** The full suite currently has 2 failures that predate this work: `tests/test_sanity.py::test_environment_sanity` asserts Python 3.10 while the local virtual environment runs Python 3.12, and `tests/test_explain_and_resumability.py::test_captum_vs_transformers_interpret_tiny_model` fails because the `transformers_interpret` package is not installed in the local environment (it would force a transformers downgrade here; the Kaggle requirements pin transformers 4.x, where it is expected to work). Both are to be verified on Kaggle, not fixed by changing project code locally.

**Known open issue (fixed in Section G, verify on Kaggle).** Separate `run_baselines.py` invocations used to rewrite `results/baselines/baseline_results.json` from scratch, so the complexity report only ever saw the most recent run. The runner now merges per-model results keyed by model and seed by default (use `--no_merge` for a clean overwrite); the merge is covered by unit tests, but accumulating a real ladder across several Kaggle invocations has not been exercised yet.

---

## Acknowledgements

- **Original Base Repository**: This project builds upon and scientifically refactors the initial research codebase from [Amazuddin/Phishing_Email_Detection_XAI](https://github.com/Amazuddin/Phishing_Email_Detection_XAI).
- **Dataset**: Kaggle "Phishing Emails" dataset created by `subhajournal` ([Kaggle Dataset Link](https://www.kaggle.com/datasets/subhajournal/phishingemails)), mirrored on Hugging Face at `zefang-liu/phishing-email-dataset`.
