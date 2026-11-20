# Section B — Multimodal extraction gaps (extract.py) — change log

Date: 2026-11-19. Scope: `src/phishing_xai/extract.py` + `tests/test_extract.py` only.
Did NOT touch: `src/phishing_xai/model.py`, `src/phishing_xai/faithfulness.py`, `decision.md`, `docs/PROJECT_LOG.md`. No push, no retrain, no fabricated metrics. No URL in any email/QR/PDF was fetched (static analysis only).

## Changes

### B1 — QR decode + OCR on embedded images in EVERY PDF page — Done
- Decision: `extract_pdf_bytes` now opens the PDF with PyMuPDF once and, for every page, extracts embedded images via `page.get_images(full=True)` + `doc.extract_image(xref)`, running OCR + QR decode on each even when the page has a text layer.
- New part kind `pdf_image` with `page` set and filename `"<pdf>#img<xref>"` (so several images on one page stay distinguishable). A QR found in an embedded image still emits a `qr` part. An embedded image whose OCR text is a near-duplicate of the page's own text (token Jaccard >= 0.8) is skipped as a duplicate and recorded in `skipped`, unless it carries a QR code.
- Reason: previously QR codes embedded in text PDFs were invisible (QR ran only on scanned-page renders).
- Status: Done. Covered by new test `test_b1_qr_in_text_pdf_embedded_image`.

### B2 — HTML bodies parsed structurally — Done
- Decision: new `_parse_html` (BeautifulSoup, with a regex tag-strip fallback if bs4 is missing): extracts all `<a href>` + anchor text into `links` (source `html_anchor`); flags anchor-text-vs-href domain mismatch as `{"type": "anchor_href_mismatch", "anchor": ..., "href": ...}` (only when the anchor itself looks like a domain/URL); decodes `data:` base64 `<img>` and OCRs + QR-decodes them (image parts named `html-embedded-image-<i>.png`); collects hidden text (`display:none`, `visibility:hidden`, `font-size:0`, white-on-white inline style, `hidden` attribute) into a `hidden_text` part + flag, never into `body_text`; records `<form action>` in `links` (source `form_action`) + a `form_action` flag; resolves `cid:` images from the matching MIME image part (deduped so the image is not OCR'd twice) and records a `missing_cid_image` flag + `skipped` entry when absent. Remote `<img src="http...">` is recorded as a link only, never fetched.
- Reason: `get_text()` flattened links away and silently dropped hidden text.
- Status: Done. Covered by `test_b2_html_anchor_href_mismatch`, `test_b2_html_base64_image_part`, `test_b2_html_hidden_text_not_in_body`, `test_b2_html_form_action_and_missing_cid`.

### B3 — Email headers + Subject as a scored part — Done
- Decision: `_extract_headers` returns `headers` = Subject, From, Reply-To, Return-Path, Message-ID, Message-ID domain, raw Authentication-Results, plus loosely parsed `spf`/`dkim`/`dmarc` verdicts (None when absent). Subject is added as a part with kind `subject` (right after the `body_text` part); `body_text` itself is unchanged for backward compatibility. Module docstring documents that consumers should score subject + body.
- Status: Done. Covered by `test_b3_headers_and_subject_part`.

### B4 — More attachment types — Done
- `.html`/`.htm` attachments: parsed with the same `_parse_html` (visible text as an `html_text` part + links/flags/image parts).
- `.docx`: `python-docx` text (paragraphs + tables) + hyperlink targets and in-text URLs into `links`; guarded import — if python-docx is missing or the file is corrupt, a `skipped` entry is recorded, no crash. Part kind `docx`.
- `message/rfc822`: recursed into with the same extraction, capped at `MAX_NESTED_DEPTH = 3`; nested body text joins the parent body, nested parts/links/flags/skipped merge into the parent result. Implementation note: `msg.walk()` descends into nested messages and yields their inner parts raw, so a custom `_iter_parts` yields `message/rfc822` as one leaf (never double-walked) and the rfc822 check runs before the multipart skip. Known limit: a nested message attached as raw `.eml` bytes (base64 CTE) is a degenerate construction the stdlib parser itself does not interpret as a message; it is not recovered. Standard construction (attaching the Message object, as real clients do) and real forwarded `.eml` files work.
- `.zip`: member names listed ONLY (never extracted) as a `zip_listing` part; flags `suspicious_zip_member` for `.exe .js .scr .iso .img .lnk .hta` and `double_extension` for names like `invoice.pdf.exe`. Member count capped at 500, total uncompressed size capped at 100 MB.
- `.svg`: text content via `xml.etree.ElementTree` (script/style element text dropped, never executed); link hrefs into `links`. Part kind `svg_text`.
- Also: plain-text attachments now yield an `attachment_text` part instead of being dropped; genuinely unsupported types get a `skipped` entry instead of silent drop.
- Status: Done. Covered by `test_b4_docx_attachment`, `test_b4_nested_message_rfc822`, `test_b4_zip_listing_and_extension_flags`, `test_b4_svg_attachment`.

### B5 — Safety limits as module constants — Done
- Decision: `MAX_ATTACHMENT_BYTES = 15 MB`, `MAX_PDF_PAGES = 20`, `MAX_PARTS_PER_EMAIL = 50`, `MAX_IMAGE_PIXELS = 178_956_970` (Pillow's default), `MAX_ZIP_MEMBERS = 500`, `MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 100 MB`, `MAX_NESTED_DEPTH = 3`, `OCR_TIMEOUT_SECONDS = 30`. Oversize images are skipped via a pixel-count check before load plus catching Pillow's `DecompressionBombError`; oversize attachments / extra PDF pages / extra parts / oversize zips are skipped with a `skipped` entry (never a crash).
- Timeout choice: no signal-based timeout — `signal` timeouts are not portable (Unix main thread only) and this module may run in threads or on Windows. Instead the size/page/part limits bound the work and every OCR/QR call is wrapped so failures are recorded as skipped. `OCR_TIMEOUT_SECONDS` is kept as the documented per-part budget a caller may enforce with its own mechanism (e.g. a worker timeout). Documented in the module docstring.
- Status: Done. Covered by `test_b5_oversize_attachment_skipped` (monkeypatched limit).

### B6 — Image preprocessing before OCR/QR — Done
- Decision: deterministic variants, original tried first, fallbacks only when the original yields no text/QR (so behaviour on clean images is unchanged): OCR variants = original, grayscaled + upscaled 2x if small, autocontrast + thresholded, inverted. QR variants = original, grayscale, inverted, rotations 90/180/270, 2x upscale if small / 0.5x downscale if very large. OpenCV `QRCodeDetector` (`detectAndDecodeMulti`, falling back to `detectAndDecode`) is a guarded-import fallback when pyzbar finds nothing; absence of cv2 never crashes.
- Status: Done (no dedicated test — exercised implicitly by the OCR/QR sample tests; kept simple/deterministic per spec).

### B7 — Tesseract discovery without hard-coded user paths — Done
- Decision: removed `_CANDIDATE_TESSERACT_PATHS` (including `C:\Users\admin\...`). `_configure_tesseract` now uses, in order: `$TESSERACT_CMD` (if the path exists), `shutil.which("tesseract")`, then platform defaults only (`C:\Program Files\Tesseract-OCR\...` on Windows, `/opt/homebrew/bin` + `/usr/local/bin` on macOS, `/usr/bin` + `/usr/local/bin` on Linux). `ocr_image` checks `_tesseract_available()` at call time and, when no binary is found (or pytesseract raises), logs a clear warning and returns `{"text": "", "boxes": []}` — extraction never crashes.
- Status: Done. Covered by `test_b7_missing_tesseract_graceful_skip` (monkeypatches `shutil.which`/pytesseract to simulate absence).

### B8 — Schema + tests — Done
- Decision: backward compatible — `body_text` and the original part keys (`kind`, `filename`, `text`, `boxes`, `qr_urls`, `page`) are unchanged, and the original part kinds keep their meaning. Added top-level keys: `headers`, `links` (list of `{url, source, anchor_text?}`), `flags` (list of dicts), `skipped` (list of `{part, reason}`). New part kinds: `subject`, `pdf_image`, `html_text`, `hidden_text`, `docx`, `zip_listing`, `svg_text`, `attachment_text`.
- Tests: 12 new tests appended to `tests/test_extract.py` (all hermetic — no tesseract/zbar/GPU/internet/checkpoint; QR/OCR in the B1 test are monkeypatched stubs). All 8 original tests untouched and still required to pass.
- Status: Done (code); test run — see below.

## Test results

- Environment notes (2026-11-19, this VM): the shared `.venv` build had stalled/misbuilt — site-packages contained only pip when first checked (its first `pip install` step had failed silently behind `| tail -5`). Re-installed into the same venv: `pillow pytest numpy beautifulsoup4 pdfplumber PyMuPDF pytesseract python-docx qrcode matplotlib pyzbar`, plus `pip install -e . --no-deps` (package was not importable), plus CPU `torch` (first download failed: network truncation at ~75/209 MB; retry with `--resume-retries 10` succeeded, torch 2.15.0+cpu) and `scikit-learn` was NOT installed (see below), plus `captum lime transformers` for the pre-existing LITA test. System packages installed via apt for the existing OCR/QR tests: `tesseract-ocr`, `libzbar0` (neither was present on the VM). Note: `scikit-learn` was never installed — `pip install torch scikit-learn` in one command failed on the torch download, and the retry installed torch only; nothing in `tests/test_extract.py` imports sklearn, so the suite passes without it. The coordinator's venv build should be checked: its first step fails silently and it never installed `pyzbar` (needed by sample3's QR test) or `torch`/`captum`/`lime`/`transformers` (needed by the LITA test via `explain.py`).
- `python -m py_compile src/phishing_xai/extract.py tests/test_extract.py`: OK.
- `pytest tests/test_extract.py -q`: **20 passed, 0 failed** (8 original + 12 new Section B tests), 34.8 s, 1 warning (PyMuPDF `fitz` deprecation, pre-existing). First run was 16 passed / 4 failed; fixes applied: (a) SVG branch moved before the generic `image/` branch (`image/svg+xml` was being OCR'd as a raster image); (b) trailing punctuation stripped from regex-extracted URLs (`_clean_url`) so a docx URL ending in `.` is recorded cleanly; (c) nested-message walk replaced with `_iter_parts` — `msg.walk()` descends into `message/rfc822` and yields its inner parts raw (base64 blob), so the rfc822 check now runs before the multipart skip and nested parts are never double-walked; (d) LITA test needed `captum`/`lime`/`transformers` installed (pre-existing `explain.py` imports, were missing from this venv). The nested-message test itself was also fixed to use the standard construction (`add_attachment(inner Message)`); attaching raw `.eml` bytes produces a degenerate base64 blob that even the stdlib parser does not interpret as a message — that construction is not recovered (documented in the test comment).
- `python scripts/make_multimodal_samples.py`: regenerated all 6 samples OK.
- `python -m phishing_xai.extract <sample> --no_boxes` per sample (2026-11-19, this VM, tesseract + zbar apt-installed for the run):
  - sample1_image_only_phish: OK — body empty, `image` OCR "Your account will be closed. Verify now…", + `subject` part "Account Notice".
  - sample2_text_with_button: OK — body + `image` OCR "Verify Account", + `subject`.
  - sample3_qr_phish: OK — `image` + `qr` parts, URL `http://secure-login-example.com/qr-verify` decoded, + `subject`.
  - sample4_text_pdf_invoice: OK — `pdf_text` p1 with billing URL `https://billing-example.com/pay`, + `subject`.
  - sample5_scanned_pdf: OK — `pdf_ocr` p1 "URGENT WIRE TRANSFER REQUIRED…"; embedded page image correctly recognised as a duplicate of the page text and recorded in `skipped` (`scanned_notice.pdf#img5`, "embedded image text duplicates the page text"), + `subject`.
  - sample6_clean_benign: OK — body + `image` + `pdf_text`, + `subject`; no flags, no skipped.

## Behaviour changes downstream scripts must know about

- `run_multimodal_predict.py` scores every part in `extracted["parts"]`: it will now also score the new `subject` part (intended — subject is scored separately from body), `hidden_text` parts (hidden text is scored as its own part, never mixed into body), and `pdf_image`/`docx`/`zip_listing`/`svg_text`/`html_text`/`attachment_text` parts. No code change is needed in that script, but its per-part table will have more rows and the max-rule verdict can now be driven by e.g. a subject line or hidden text — that is the intended behaviour, flag it in the report if a subject-only hit looks odd.
- `extract_eml` now returns 6 top-level keys (`body_text`, `parts`, `headers`, `links`, `flags`, `skipped`) instead of 2; readers that assumed exactly 2 keys should be checked (both known scripts only read `parts`/`body_text`, so they are fine).
- `run_multimodal_faithfulness.py` reads `parts` the same way; new parts flow through unchanged.
- Filenames for PDF embedded images are `"<pdf>#img<xref>"`, not the plain PDF filename.
- Emails with oversize/unsupported attachments now produce `skipped` entries instead of silently ignoring them; empty OCR (tesseract missing) yields empty-text image parts instead of raising.

## Needs Kaggle / checkpoint verification

- Nothing in Section B needs a checkpoint or GPU: all new tests are hermetic. Real-world OCR/QR quality (B6 variants, B1 embedded-image QR on genuine phishing PDFs, OpenCV fallback) should be spot-checked on Kaggle or any machine with tesseract + zbar + a real phishing corpus, since this VM had neither until apt-installed for testing.
- The existing sample tests (sample1 OCR, sample3 QR, sample5 scanned PDF) depend on system tesseract + zbar; if the Kaggle/CI environment lacks them, those tests fail for environmental reasons, not code reasons — the new B7 graceful path is what keeps extraction itself working there.
