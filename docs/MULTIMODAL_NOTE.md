# Multimodal Extension Note (Image, QR & PDF Phishing)

## 1. What Was Added

Without retraining the model or modifying `src/phishing_xai/model.py`, `src/phishing_xai/faithfulness.py`, or the 41 existing unit tests, the repository was extended to detect and explain multimodal phishing emails (`.eml` messages containing image-only bodies, inline button images, QR codes, text PDFs, and scanned image-only PDFs):

1. **`src/phishing_xai/extract.py` (CLI: `python -m phishing_xai.extract <file.eml>`)**:
   - Parses `.eml` messages with Python's `stdlib email` library and walks all MIME parts.
   - Extracts image text and word-level bounding boxes (`left`, `top`, `width`, `height`, `conf`) via `pytesseract.image_to_data` (Tesseract `v5.4.0.20240606`).
   - Extracts per-page PDF text and embedded/regex hyperlinks via `pdfplumber`; when a PDF page contains no extractable text (`scanned` PDF), renders the page to a PNG at 200 DPI via `PyMuPDF` (`fitz`) and runs Tesseract OCR (`kind="pdf_ocr"`).
   - Decodes QR code payloads and URLs from images and rendered PDF pages via `pyzbar`.
2. **`scripts/make_multimodal_samples.py` & `tests/test_extract.py`**:
   - Generates 6 deterministic `.eml` samples in `samples/multimodal/` and verifies MIME parsing, OCR keywords, bounding boxes, PDF text/links, scanned-PDF OCR fallback, QR decoding, and bounding-box visualization in `tests/test_extract.py` (8 new tests; **49 total tests passing**).
3. **`scripts/run_multimodal_predict.py`**:
   - Loads RoBERTa via `phishing_xai.model.load_tokenizer_and_model`, scores `body_text` plus each extracted part, and applies the **max rule** (verdict is Phishing if *any* part scores Phishing, naming the highest-scoring part as the winning part).
   - Supports `--explain` to compute LITA hybrid attributions (`alpha=0.5`, magnitude-preserving) on the winning part and print the top 5 words.
   - Checks for `results/best_model.pt` locally and stops with an explicit missing-file error by default rather than retraining; supports `--allow_pretrained_fallback` to demonstrate execution locally before the Colab GPU checkpoint is downloaded.
4. **Visual Explanation (`draw_lita_boxes_on_image` in `src/phishing_xai/explain.py`)**:
   - Appended a single new function to `src/phishing_xai/explain.py` that matches top-5 LITA words against OCR bounding boxes and draws red rectangles (`outline="red"`, `width=3`), saving `<name>_explained.png` (verified on `sample1_image_only_phish_explained.png`).
5. **`scripts/run_multimodal_faithfulness.py`**:
   - Reuses `phishing_xai.faithfulness` (`evaluate_faithfulness_single_method`, `predict_probability`) unchanged to measure Comprehensiveness and Sufficiency at $k=5$ on the extracted text of the 6 samples.

---

## 2. Six-Sample Results Table

> **Checkpoint Status**: `results/best_model.pt` is not present locally (full fine-tuning is configured for Google Colab GPU in `notebooks/colab_run.ipynb`). Per project rules, no local retraining was performed. Default invocation of `run_multimodal_predict.py` stops and reports `results/best_model.pt` as missing; the table below records the pipeline output with `--allow_pretrained_fallback` (`roberta-base` via `model.py`).

| Sample (`.eml`) | Modality / Parts Extracted | Winning Part (Max Rule) | Extracted Text / Payload (Winning Part) | Top-5 LITA Words ($\alpha=0.5$) | Comp ($k=5$) | Suff ($k=5$) |
|---|---|---|---|---|---|---|
| `sample1_image_only_phish.eml` | `body_text` (empty), `image:account_notice.png` | `image:account_notice.png` | `"Your account will be closed. Verify now at http://secure-login-example.com"` | `at`, `account`, `Your`, `http://secure-login-example.com`, `be` | `+0.0032` | `-0.0013` |
| `sample2_text_with_button.eml` | `body_text`, `image:verify_button.png` | `image:verify_button.png` | `"Verify Account"` | `Account`, `Verify` | `-0.0041` | `+0.0000` |
| `sample3_qr_phish.eml` | `body_text`, `image:auth_qr.png`, `qr:auth_qr.png` | `qr:auth_qr.png` | `"http://secure-login-example.com/qr-verify"` | `http://secure-login-example.com/qr-verify` | `-0.0017` | `+0.0000` |
| `sample4_text_pdf_invoice.eml` | `body_text`, `pdf_text:invoice.pdf#p1` | `pdf_text:invoice.pdf#p1` | `"INVOICE #INV-90821 Payment due. Click https://billing-example.com/pay ..."` | `account`, `your`, `.`, `Click` | `-0.0005` | `-0.0027` |
| `sample5_scanned_pdf.eml` | `body_text`, `pdf_ocr:scanned_notice.pdf#p1` | `pdf_ocr:scanned_notice.pdf#p1` | `"URGENT WIRE TRANSFER REQUIRED Your account will be suspended ..."` | `confirmed`, `account`, `WIRE`, `Your`, `.` | `+0.0015` | `-0.0004` |
| `sample6_clean_benign.eml` | `body_text`, `image:agenda_banner.png`, `pdf_text:roadmap_notes.pdf#p1` | `pdf_text:roadmap_notes.pdf#p1` | `"Quarterly Engineering Roadmap Notes All sprint milestones are on track ..."` | `Quarterly`, `next`, `the`, `release`, `sprint` | `+0.0003` | `-0.0025` |

---

## 3. Honest Limitations

1. **OCR Errors on Low-Quality or Adversarial Images**: `pytesseract` relies on clean character segmentation and contrast. Heavy compression artifacts, skewed scans, CAPTCHA-style noise, decorative fonts, or color-on-color phishing banners can corrupt OCR tokens, causing downstream tokenization and LITA word matching to miss key phishing terms.
2. **Additional Latency from OCR, PDF Rendering, and Per-Part XAI**: Rendering scanned PDF pages via `PyMuPDF`, running Tesseract `image_to_data`, and scoring/explaining multiple MIME parts sequentially adds hundreds of milliseconds to several seconds per email compared to plain-text inference.
3. **Only 6 Synthetic Samples**: The evaluation set in `samples/multimodal/` consists of 6 clean, synthetically generated `.eml` files designed to test each modality path (`image`, `qr`, `pdf_text`, `pdf_ocr`). It does not measure recall or false-positive rates on wild multimodal phishing campaigns.
4. **No Multimodal Retraining & Missing Local Checkpoint**: The classifier is a text-only RoBERTa model applied zero-shot to extracted OCR/PDF/QR strings (it cannot inspect visual brand logos or layout spoofing directly), and requires the Colab-trained `results/best_model.pt` checkpoint for calibrated phishing probabilities.
