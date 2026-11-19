# Project & Multimodal Extension Decisions Log (`decision.md`)

All architectural, experimental, and multimodal design decisions for `Phishing_Email_Detection_XAI` are recorded here and synchronized with [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

## 1. Core Pipeline Decisions (Phases 0–5)

| Decision | Reason | Phase | Status |
|---|---|---|---|
| **Label Map**: `Safe Email = 0`, `Phishing Email = 1` | Enforces consistent binary classification targets across training, tuning, evaluation, and XAI. | Phase 1 | Complete |
| **Exact Deduplication & Conflict Removal** | Dropped 585 exact duplicates and all 533 conflicting `'empty'` placeholder rows before splitting (`17,532` clean rows remaining). | Phase 1 | Complete |
| **Near-Duplicate Grouping (`StratifiedGroupKFold`)** | Grouped emails sharing a normalized first-200-char SHA-1 hash (`796` multi-member groups) into the same split so `0` near-duplicate groups leak across train/val/test. | Phase 5 | Complete |
| **Double Stratified Split (80/20 then 80/20)** | Saved deterministic split indices in `results/splits.json` (`11,220` train, `2,806` val, `3,506` test); test split is evaluated only once at the end of training. | Phase 1 / 3 | Complete |
| **Training-Only Augmentation** | WordNet synonym replacement (`replace_ratio=0.2`, `seed=42`) is applied strictly to the training split (`11,220` $\rightarrow$ `14,078`). | Phase 1 | Complete |
| **Model Window Pre-Truncation** | Truncate raw text to the exact character span covered by `tokenizer(..., max_length=max_length, return_offsets_mapping=True)` with a 100,000-char pre-cap so RoBERTa, LIME, TI/Captum, and faithfulness see the exact same text. | Phase 5 | Complete |
| **Shared Regex Word Splitter (`WORD_SPLIT_REGEX`)** | Used by both `LimeTextExplainer(split_expression=split_words)` and `aggregate_subword_attributions_to_words` so URLs, contractions, numbers, and repeated word occurrences align 1-to-1. | Phase 5 | Complete |
| **Captum `LayerIntegratedGradients` Parity Fallback** | Uses `[<s>, <pad>..., </s>]` baseline, `roberta.embeddings.word_embeddings`, and L2 normalization to match `transformers-interpret` (Spearman $\rho > 0.9$). | Phase 3 / 5 | Complete |
| **LITA Magnitude-Preserving Normalization** | Added `scores / max(|scores|)` alongside legacy min-max scaling to preserve zero-baseline, sign polarity, and relative importance ratios. | Phase 4 | Complete |
| **Strict Debug Isolation** | All `--debug` runs write strictly to `results/debug/` and are blocked from `results/RESULTS.md` and `results/runs_log.csv`. | Phase 3 | Complete |

---

## 2. Multimodal Extension Decisions (Steps 1–7)

| Decision | Reason | Step | Status |
|---|---|---|---|
| **Add Multimodal Dependencies (`pytesseract`, `pdfplumber`, `PyMuPDF`, `pyzbar`, `Pillow`, `qrcode`)** | Required for image OCR + word boxes, PDF text/link extraction, scanned-PDF page rendering, and QR code generation/decoding. Pinned exact versions in `requirements.txt` and added to `requirements-colab.txt`. | Step 1 | Complete |
| **User-Space Tesseract 5.4.0 Installation** | Installed UB-Mannheim Tesseract (`v5.4.0.20240606`) into `C:\Users\admin\AppData\Local\Programs\Tesseract-OCR` via `RunAsInvoker` (avoiding headless UAC hang) and configured `_configure_tesseract()` in `extract.py` to auto-detect both system and user-space paths. | Step 1 | Complete |
| **Unified `.eml` Extraction Schema (`src/phishing_xai/extract.py`)** | `extract_eml()` walks MIME parts via Python's `stdlib email`, returning `{"body_text": str, "parts": [{"kind", "filename", "text", "boxes", "qr_urls", "page"}, ...]}` and exposing CLI `python -m phishing_xai.extract <file.eml>`. | Step 2 | Complete |
| **Scanned PDF Automatic OCR Fallback** | In `extract_pdf_bytes()`, if `pdfplumber` returns empty text on a PDF page, the page is rendered at 200 DPI via `PyMuPDF` (`fitz`) and passed to `ocr_image()` and `decode_qr()`, tagged as `kind="pdf_ocr"`. | Step 2 | Complete |
| **6 Synthetic Multimodal `.eml` Samples (`scripts/make_multimodal_samples.py`)** | Deterministically builds (a) image-only phishing body, (b) normal text + inline `'Verify Account'` button image, (c) QR code phishing email, (d) text PDF fake invoice with URI link, (e) scanned image-only PDF, and (f) clean benign email with harmless image & PDF. | Step 3 | Complete |
| **Zero Retraining & Unmodified `model.py` / `faithfulness.py`** | Hard constraint: never retrain the model and never modify `model.py`, `faithfulness.py`, or the 41 existing tests. | Step 4–6 | Complete |
| **Missing Checkpoint Reporting & Pretrained Fallback Flag** | Because full GPU training is deferred to Google Colab, `results/best_model.pt` is not present locally. `run_multimodal_predict.py` and `run_multimodal_faithfulness.py` stop by default and report the exact missing path (`results/best_model.pt`), while supporting `--allow_pretrained_fallback` to demonstrate pipeline execution with base `roberta-base` via `model.py`. | Step 4 | Complete |
| **Max-Rule Multimodal Verdict** | An email is classified as Phishing (`1`) if **any** extracted part (body text, image OCR, QR URL, PDF text, or scanned PDF OCR) scores Phishing, naming the highest-scoring part as the winning part. | Step 4 | Complete |
| **Visual LITA Bounding-Box Overlay (`draw_lita_boxes_on_image`)** | Appended one new function `draw_lita_boxes_on_image` to `src/phishing_xai/explain.py` without changing any existing functions; draws red rectangles (`outline="red"`, `width=3`) around OCR word boxes matching top-5 LITA words and saves `<name>_explained.png`. | Step 5 | Complete |
| **Multimodal Faithfulness (`scripts/run_multimodal_faithfulness.py`)** | Reuses `evaluate_faithfulness_single_method` and `predict_probability` from `phishing_xai.faithfulness` unchanged to compute Comprehensiveness and Sufficiency at $k=5$ on the extracted text of the 6 samples. | Step 6 | Complete |
| **Pyrefly Config in `pyproject.toml`** | Added `[tool.pyrefly]` excluding legacy notebooks and ignoring `missing-source-for-stubs` for unused legacy `tensorflow` stubs. | Maintenance | Complete |
