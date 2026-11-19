# Project & Multimodal Extension Decisions Log (`decision.md`)

All architectural, experimental, and multimodal design decisions for `Phishing_Email_Detection_XAI` are recorded here and synchronized with [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).

## 1. Core Pipeline Decisions (Phases 0–5)

| Decision | Reason | Phase | Status |
|---|---|---|---|
| **Label Map**: `Safe Email = 0`, `Phishing Email = 1` | Enforces consistent binary classification targets across training, tuning, evaluation, and XAI. | Phase 1 | Complete |
| **Exact Deduplication & Conflict Removal** | Dropped 585 exact duplicates and all 533 conflicting `'empty'` placeholder rows before splitting (`17,513` clean rows remaining). | Phase 1 | Complete |
| **Near-Duplicate Grouping (`StratifiedGroupKFold`)** | Grouped emails sharing a normalized first-200-char MD5 hash into the same split so `0` near-duplicate groups leak across train/val/test. (Note: the `796` multi-member-groups figure previously stated here is unverified — no script in the repo logs a near-duplicate group count; `make_splits.py` / `data.py` group by the hash but never count groups.) | Phase 5 | Complete |
| **Double Stratified Split (80/20 then 80/20)** | Saved deterministic split indices in `results/splits.json` (`11,215` train, `2,803` val, `3,495` test); test split is evaluated only once at the end of training. | Phase 1 / 3 | Complete |
| **Training-Only Augmentation** | WordNet synonym replacement (`replace_ratio=0.2`, `seed=42`) is applied strictly to the training split (`11,215` $\rightarrow$ `14,078`). | Phase 1 | Complete |
| **Model Window Pre-Truncation** | Truncate raw text to the exact character span covered by `tokenizer(..., max_length=max_length, return_offsets_mapping=True)` with a 50,000-char pre-cap so RoBERTa, LIME, TI/Captum, and faithfulness see the exact same text. | Phase 5 | Complete |
| **Shared Regex Word Splitter (`WORD_SPLIT_REGEX`)** | Used by both `LimeTextExplainer(split_expression=split_words)` and `aggregate_subword_attributions_to_words` so URLs, contractions, numbers, and repeated word occurrences align 1-to-1. | Phase 5 | Complete |
| **Captum `LayerIntegratedGradients` Parity Fallback** | Uses `[<s>, <pad>..., </s>]` baseline, `roberta.embeddings.word_embeddings`, and L2 normalization to match `transformers-interpret` (Spearman $\rho > 0.9$). | Phase 3 / 5 | Complete |
| **LITA Magnitude-Preserving Normalization** | Added `scores / max(|scores|)` alongside legacy min-max scaling to preserve zero-baseline, sign polarity, and relative importance ratios. | Phase 4 | Complete |
| **Strict Debug Isolation** | All `--debug` runs write strictly to `results/debug/` and are blocked from `results/RESULTS.md` and `results/runs_log.csv`. | Phase 3 | Complete |

---

## 2. Section A — Correctness Fixes (2026-11-19)

### A1. LITA alpha was inverted relative to paper Eq. 2 (PROMINENT — read before using any alpha result)

**What was wrong.** `src/phishing_xai/lita.py::combine_scores` computed `(1 - alpha)*LIME + alpha*TI`, i.e. alpha was the **TI** weight. Paper Eq. 2 is `alpha*LIME + (1 - alpha)*TI`, i.e. alpha is the **LIME** weight. The implementation therefore had the alpha direction exactly backwards: a run labelled `alpha=0` produced pure LIME when the paper means pure TI, and `alpha=1` produced pure TI when the paper means pure LIME. Corroboration that the paper direction is the intended one: the upstream legacy notebooks in this repo already implement `alpha * lime_norm + (1 - alpha) * trans_norm` (notebooks left untouched per scope).

**What changed.**
- Code: `combine_scores` now returns `alpha*LIME + (1 - alpha)*TI`. Docstrings fixed in `combine_scores`, `compute_lita_scores`, and `build_hybrid_word_attributions` (all three previously described alpha as the LIME weight while the code did the opposite, or stated the inverted endpoints). `src/phishing_xai/config.py` (`XAIConfig.alpha`) and the `--alpha` help text in `scripts/run_explain.py` now state the direction explicitly. `src/phishing_xai/report.py` ablation table header corrected from "Alpha (TI weight)" to "Alpha (LIME weight)". `scripts/run_ablation.py` prints an explicit direction line ("Alpha is the LIME weight (paper Eq. 2): 0.0 = pure TI, 1.0 = pure LIME."); it had no mislabelled alpha=0/alpha=1 label to fix, only the bare "Alpha" column header.
- Tests (deliberately updated, not silently): `tests/test_explain_and_lita.py::test_lita_combine_scores_alpha_weighting` and `tests/test_faithfulness_and_ablation.py::test_alpha_ablation_endpoints` previously encoded the inverted direction (alpha=0 asserted pure LIME); both now assert alpha=0 is pure TI and alpha=1 is pure LIME.
- Docs: `docs/PROJECT_LOG.md` finding 6 formula and the LITA-variants decision row corrected to the paper direction.
- Other places checked: `src/phishing_xai/explain.py` has no alpha references; `scripts/run_faithfulness.py` and the multimodal scripts only use `alpha=0.5` (see below); `docs/MULTIMODAL_NOTE.md` and `README.md` only cite `alpha=0.5`, which is unaffected. No other file assumed the old direction.

**What is unaffected.** At `alpha=0.5` the old and new formulas are identical (`0.5*LIME + 0.5*TI`), so every result produced at the default `alpha=0.5` — including all multimodal LITA explanations and faithfulness numbers — is unchanged. Only the `alpha != 0.5` ablation endpoints swap meaning: previously reported `alpha=0` rows are in fact pure TI and `alpha=1` rows are in fact pure LIME, so any ablation table produced before this fix must have its endpoint labels swapped before it is cited. No retraining was involved; this is a pure scoring/doc fix.

### A2. Untrained-fallback runs are now loud and flagged in output JSON

`scripts/run_multimodal_predict.py` and `scripts/run_multimodal_faithfulness.py` previously only logged a quiet `logger.warning` when `--allow_pretrained_fallback` ran on untrained base `roberta-base`. Now `load_trained_roberta` (shared by both scripts) prints a banner — "WARNING: UNTRAINED MODEL - results are NOT detection evidence" — plus the same message via `logger.warning`, and every output JSON carries `"untrained_fallback": true` (per-file result objects in the predict script, per-sample rows in the faithfulness script). When the real checkpoint is used the flag is written as `false`. The printed per-file report also shows the warning line when the flag is set. Default behaviour is unchanged: without the flag the scripts still stop with an error when `results/best_model.pt` is missing.

### A3. Docs/code inconsistencies fixed — `results/splits.json` is the source of truth

Verified in code: near-duplicate hash is **MD5** of the first 200 alphanumeric characters (`data.py::compute_near_dup_hash`, `n_chars=200` default); pre-tokenization char cap is **50,000** (`data.py::truncate_text_to_model_window`, `pre_tokenization_char_cap=50_000` default) — not 100,000 and not SHA-1 as `decision.md` previously said. `results/splits.json` is the source of truth: raw 18,650, nulls 16, conflicting 533 (`'empty'`), exact dups 585, retained **17,513**, splits train **11,215** (7,039 safe / 4,176 phish), val **2,803**, test **3,495**, augmentation +2,863 train-only to **14,078**. Fixed everywhere the wrong numbers appeared: `decision.md` (17,532 → 17,513; splits 11,220/2,806/3,506 → 11,215/2,803/3,495; augmentation 11,220 → 11,215; 100,000-char cap → 50,000; SHA-1 → MD5), `docs/PROJECT_LOG.md`, `README.md`, `docs/MULTIMODAL_NOTE.md`. `results/RESULTS.md` and `results/summary.json` already matched `splits.json` and needed no change. The `796` multi-member near-duplicate-groups figure previously stated in `decision.md` is **unverified**: no script in the repo logs a group count (`make_splits.py`/`data.py` group by the hash but never count groups), so the number was removed and replaced with that note rather than re-asserted. Test count: the tracked repo has **61** test functions at the end of this work (`grep "def test_"` over tracked `tests/` files) — 49 at Section A completion plus 12 added to `tests/test_extract.py` by Section B, which landed in this working copy during this session. (41 was the correct count at the end of Phase 5; the 8 `test_extract.py` tests added by the multimodal extension brought it to 49.) This count will change again as sections B/C/D/E/H add tests; the 61 figure is the count true at the end of Section A work, not a frozen total. Untracked files from other sections' in-flight work (`src/phishing_xai/url_risk.py`, `src/phishing_xai/baselines.py`, `tests/test_url_risk.py`, `tests/test_baselines.py`, `scripts/run_baselines.py`, `scripts/run_complexity_report.py`) were present in this working copy, were not touched, and are NOT included in the 61 count (a plain on-disk `grep` returns more because of them). Note: `results/debug/multimodal_predictions.json` is NOT in this repo (`results/debug/` is gitignored, confirmed via `git check-ignore`); its contents were not inspected here.

### A4. Consolidated report merges `results/metrics.json` when present

Verified: `src/phishing_xai/report.py::generate_consolidated_report` already loads `results/metrics.json` when present (and already skips it when it carries `"debug": true`), and `tests/test_report.py::test_summary_numbers_match_source_artifacts` already covers the merge, so the merge itself needed no code change. Two reporting changes were still made here: (a) per the precision-first decision (A5), the Section 4 test-metrics list now leads with phishing-class precision + FPR, with F1/accuracy/recall kept and listed after — nothing re-ranked or dropped; FPR is shown only when the metrics artifact records it, otherwise honestly shown as "not recorded in `metrics.json`" (it is not derived from other numbers); (b) the alpha-ablation table header was corrected from "Alpha (TI weight)" to "Alpha (LIME weight)" (see A1). Verified empirically 2026-11-19 by running `scripts/make_report.py` against a throwaway copy in `/tmp` (never in `results/`): without `metrics.json` the training section stays `NOT RUN YET` with dataset numbers from `splits.json`; with a dummy `metrics.json` in the `train.py` shape the test metrics appear in `RESULTS.md`/`summary.json`, precision first. The dummy was deleted afterwards; no fake `results/metrics.json` exists in the repo. The `NOT RUN YET` wording in the committed `results/RESULTS.md` / `results/summary.json` for tuning, training, and explainability is genuine (those stages have not run in production) and was left in place.

---
### A5. Precision is the priority focus (user decision, recorded 2026-11-19)

User's words: "mera priority focus precision hai." Primary metric is **phishing-class precision** (minimise false positives / low FPR). F1, accuracy, and recall are secondary: reported, but never optimised or selected on alone. Thresholds are fit on **validation only, never test**. This decision governs Sections D/E/H (fusion threshold, eval tables, baseline ranking/selection), which implement it; Section A only records it here and makes precision prominent where reporting surfaces a headline metric (see A4 above: `report.py` now leads the test-metrics list with phishing-class precision + FPR). `evaluate.py::compute_metrics` already records phishing-class precision (sklearn binary `precision_score`, positive label 1).


## 4. Sections B–H — Multimodal and Baseline Work (2026-11-19)

Full change logs live in `.impl_logs/section_B.md` through `.impl_logs/section_H.md` and `.impl_logs/integration.md`. This section is the merged record: decision, reason, status per section. Nothing here was pushed, nothing was retrained, and `src/phishing_xai/model.py` and `src/phishing_xai/faithfulness.py` were not modified by any of these sections (verified via `git status` by each section agent). Anything not executed is **NOT RUN YET**.

### B. Multimodal extraction gaps (`extract.py`) — Done

| Decision | Reason | Status |
|---|---|---|
| **B1. QR decode + OCR on embedded images in every PDF page** — new part kind `pdf_image` (`"<pdf>#img<xref>"`); a QR in an embedded image still emits a `qr` part; an embedded image whose OCR text near-duplicates the page text (token Jaccard >= 0.8) is skipped unless it carries a QR code. | QR codes embedded in text PDFs were previously invisible (QR ran only on scanned-page renders). | Done; 12 new tests in `tests/test_extract.py` (20 total there). |
| **B2. HTML bodies parsed structurally** — anchor/href pairs into `links` (source `html_anchor`), anchor-vs-href domain mismatch flag, base64 `<img>` OCR + QR, hidden text into its own `hidden_text` part (never into `body_text`), `<form action>` links + flag, `cid:` images resolved from the matching MIME part with a `missing_cid_image` flag when absent; remote images recorded as links only, never fetched. | `get_text()` flattened links away and silently dropped hidden text. | Done. |
| **B3. Headers + subject as a scored part** — `headers` holds Subject, From, Reply-To, Return-Path, Message-ID, Authentication-Results and loosely parsed SPF/DKIM/DMARC verdicts (None when absent); subject is a `subject` part after `body_text`, which is unchanged. | Header and subject signals were invisible to scoring. | Done. |
| **B4. More attachment types** — `.html` via the same HTML parser; `.docx` text + hyperlinks (guarded import, corrupt files recorded as skipped); `message/rfc822` recursed to depth 3; `.zip` member names listed only, never extracted, with `suspicious_zip_member` and `double_extension` flags; `.svg` text via ElementTree; plain-text attachments now yield `attachment_text`; unsupported types get a `skipped` entry instead of a silent drop. | Those attachments were silently dropped. Known limit: a nested message attached as raw base64 `.eml` bytes is a degenerate construction the stdlib parser itself does not interpret as a message; it is not recovered. | Done. |
| **B5. Safety limits as module constants** — 15 MB attachments, 20 PDF pages, 50 parts per mail, Pillow pixel limit, 500 zip members, 100 MB total uncompressed zip, nested depth 3, documented 30 s OCR budget a caller may enforce; oversize items are skipped with a `skipped` entry, never a crash. No signal-based timeout (not portable to threads or Windows). | Unbounded inputs could hang or exhaust memory on adversarial mail. | Done. |
| **B6. Image preprocessing before OCR/QR** — deterministic variants, original tried first, fallbacks only when the original yields nothing: OCR (grayscale + 2x upscale, autocontrast + threshold, inverted), QR (grayscale, inverted, rotations, rescale), OpenCV `QRCodeDetector` as a guarded fallback when pyzbar finds nothing. | Clean-image behaviour stays unchanged while stressed images get a second chance. | Done (no dedicated test; exercised by the sample tests). |
| **B7. Tesseract discovery without hard-coded user paths** — removed `_CANDIDATE_TESSERACT_PATHS` (including the historical `C:\Users\admin\...` path); order is `$TESSERACT_CMD`, `shutil.which("tesseract")`, then platform defaults only; a missing binary logs a warning and returns empty OCR text instead of raising. | Hard-coded user paths break on any other machine. | Done. |
| **B8. Backward-compatible schema** — `body_text` and the original part keys are unchanged; new top-level keys `headers`, `links`, `flags`, `skipped`; new part kinds `subject`, `pdf_image`, `html_text`, `hidden_text`, `docx`, `zip_listing`, `svg_text`, `attachment_text`. | Downstream scripts read `parts`/`body_text` only and keep working; the predict script now also scores the new subject and hidden-text parts, which is intended. | Done; `pytest tests/test_extract.py` 20 passed. |

### C. URL / structural risk module (`url_risk.py`) — Done

| Decision | Reason | Status |
|---|---|---|
| **New pure-static lexical URL risk module** — `score_url` / `score_urls` / `extract_urls` / `registered_domain` plus `UrlRiskConfig`; no network calls, no DNS, no fetching, no new hard dependencies. Scores 0–1 from weighted signals: dangerous scheme 0.55, IP host 0.30, `@` 0.20, punycode/non-ASCII host 0.25, suspicious TLD 0.20 (heuristic list, not a blocklist), shortener 0.25, http 0.10, subdomains 0.08–0.15, hyphens 0.10–0.15, brand keyword in subdomain/path with a different registered domain 0.25 (real `paypal.com` / `login.microsoft.com` never flagged), high-entropy token 0.15, long-URL tiers, QR-only +0.10. | URL evidence must be independent of the text model and must never fetch a link. | Done; 39 tests in `tests/test_url_risk.py`, all passing. |
| **`tldextract` only via a guarded import** with a bundled snapshot (`suffix_list_urls=[]`); it is not in `pyproject.toml` and is not installed in the venv, so the minimal fallback runs in practice (last two labels, plus a small multi-label suffix set). | No download at import time; the fallback shifts a subdomain/brand signal by at most one weight, never a scheme/IP/punycode verdict. | Done; documented in the module docstring. |

### D. Multimodal fusion + verdict (`multimodal.py`) — Done

| Decision | Reason | Status |
|---|---|---|
| **Precision-first fusion (user decision 2026-11-19, governs D/E/H)** — primary metric is phishing-class precision; F1/accuracy/recall are reported, never the sole selection criterion. | User: "mera priority focus precision hai." False positives cost trust. | Done; implemented in D/E/H. |
| **Weighted fusion is the default** (`--fusion weighted`): weights 0.45 concatenated context, 0.20 best part, 0.20 URL, 0.15 flags, in `FusionConfig` (`config.py`). The context weight is largest because it sees the whole mail at once; the single-part weight is deliberately small so one noisy OCR fragment cannot flip the verdict. | The old max rule flips on any part and is recall-heavy / precision-poor by construction. | Done; 37 tests in `tests/test_multimodal_fusion.py`, all passing. |
| **Corroboration gate (weighted only)** — besides `final_score >= threshold`, at least two of the three channels (context, max part, URL) must be >= 0.5, or one channel plus a flag score >= 0.30, or the fused score must reach 0.75. Otherwise the verdict stays Safe with a recorded reason. | A single channel, however confident, is not enough evidence for a phishing call. | Done; the exact divergence (one part at 0.85 with context 0.30 stays Safe under weighted, Phishing under max) is asserted in tests. |
| **Threshold 0.5 is a PLACEHOLDER** — it must be tuned on validation only, on Kaggle with the trained checkpoint, maximising phishing-class precision subject to a recall floor, never by maximising F1 alone. No claim that it was tuned is made anywhere. | An untuned threshold quoted as tuned would be a fabricated result. | Done (code + docs); tuning itself NOT RUN YET. |
| **Max rule kept only as a selectable baseline** (`--fusion max`); both fusions are computed and stored for every mail so Section E can compare precision/recall/FPR side by side. | Needed for the honest comparison, not as the verdict. | Done. |
| **LITA on the concatenated context + overlays** — `--explain` runs LITA on the full context; new helpers overlay LITA boxes on scanned-PDF pages (rendered at the same 200 DPI the extractor OCR'd at) and on QR images (decoded-QR region + URL annotation, never fetched). | Explaining one winning part hid evidence spread across the mail. | Done in code and unit tests; end-to-end overlay for a PDF/QR winning part needs a trained checkpoint (NOT RUN YET — in the smoke run the winning part was the subject on most samples because untrained scores are all chance-level). |
| **Predict-script fixes found in this section** — Section B stores PDF hyperlinks in the `qr_urls` slot of `pdf_text`/`pdf_ocr` parts, so the first version scored sample 4's billing link as QR-only (+0.10); only `qr`/`image`/`pdf_image` parts now count as QR sources (fixed in `multimodal.py`, not by editing `extract.py`). Kaggle auto-detection added (checkpoint defaults to the first `/kaggle/input/*/best_model.pt`, outputs to `/kaggle/working/results`); the JSON summary is now always written. The A2 untrained-fallback banner and `"untrained_fallback": true` per row are kept verbatim. | The QR bump mislabelled a plain PDF link as QR evidence. | Done; D smoke run 6/6 rows, all untrained fallback, all Safe under both fusions (chance-level scores 0.43–0.44) — pipeline demonstration only. |
| **Accidental `XAIConfig` header drop during the `config.py` edit** — caught by the test run (`ImportError`) and restored verbatim the same day; verified `XAIConfig().alpha == 0.5`. | Editing accident, fixed before it shipped. | Done. |

### E. Evaluation set + ablation (`make_multimodal_eval_set.py`, `run_multimodal_eval.py`) — Done (code); smoke runs only, untrained

| Decision | Reason | Status |
|---|---|---|
| **Synthetic eval set, 120 mails** — deterministic (seed 42), 60 phishing / 60 benign in `data/multimodal_eval/` with `labels.csv` and `manifest.json`; 8 phishing categories (image-only 10, text-button 8, QR-body 8, QR-in-text-PDF 7, scanned-PDF 7, HTML link mismatch 7, HTML attachment 6, fake invoice 7) and 3 benign look-alike categories (newsletter-with-QR 20, real invoice 20, meeting agenda 20); stratified split 24 validation / 96 test; seeded OCR stressors (font/size, low contrast, rotation, noise, JPEG compression). Same seed gives identical labels/file list/manifest, but `.eml` byte equality is not guaranteed (PyMuPDF embeds a fresh document ID per write). | No real labelled multimodal set exists locally; a synthetic set measures pipeline behaviour on templates, and the caveat is written into the manifest, the JSON metadata and the top of the report. | Done; 20 tests in `tests/test_multimodal_eval.py`, all passing. |
| **Precision-first eval tables** — every table leads with phishing-class precision + FPR; thresholds fitted on validation only (recall floor 0.80 default), never on test; the placeholder-0.5 metrics are shown alongside. Ablations (a) text-body only, (b) text + OCR, (c) text + OCR + QR + PDF, (d) full fusion weighted and max. | The user's precision priority governs how results are read. | Done. |
| **Smoke results are UNTRAINED FALLBACK, not detection evidence** — full-set run with `--allow_pretrained_fallback` (checkpoint missing, untrained `roberta-base`): best precision at recall floor 0.80 is (d) weighted, precision 1.00 at recall 0.75 (F1 0.86, FPR 0.00, accuracy 0.88, n=96), confusion `[[48, 0], [12, 36]]`; the separation comes from the static URL/flag channels, not the model. Ablations (a)–(c) collapse to precision 0.50 / FPR 1.00 at fitted threshold 0.05. Full numbers in `results/multimodal_eval/multimodal_eval.json`/`.md`. | Chance-level model scores force the validation fit to "predict everything phishing" for (a)–(c); quoting any of it as detection performance would be false. | Done (smoke run executed); real detection numbers NOT RUN YET. |
| **TF-IDF+LR cheap baseline on the same extractor output** — fitted on the real-data train split (11,215 rows, `results/splits.json`), evaluated on the 120 synthetic mails: precision 0.95, recall 0.95, F1 0.95, accuracy 0.95, ROC-AUC 0.9875, FPR 0.05, confusion `[[57, 3], [3, 57]]`, n=120, threshold 0.5; per-modality F1 all 0.95, body 0.79, OCR 0.83, PDF 0.79, QR 0.73. Never fitted on the synthetic set (that would be leakage). BiLSTM was not run locally (CPU cost). | The baseline hook (Section H) evaluated on the same extractor output. The number says the synthetic templates are easy for a linear model, nothing about real multimodal phishing. | Done (TF-IDF only); BiLSTM NOT RUN YET (Kaggle). |
| **Faithfulness script additions** — random baseline (`evaluate_random_baseline`, same k=5, 5 seeds) now computed per sample and reported as `random_comprehensiveness_k5` / `random_sufficiency_k5`, plus an empty-extraction guard (`status: "no_text_parts"` instead of crashing). Smoke run: LITA ≈ random on all 6 samples, the expected untrained-model artefact. | Without a random baseline, a faithfulness number cannot be told apart from chance. | Done. |

### F. Kaggle notebook + packaging (`kaggle_run.ipynb`, `requirements-kaggle.txt`) — Done (code); NOT RUN YET (execution)

| Decision | Reason | Status |
|---|---|---|
| **Kaggle is the primary run path; the Colab notebook is kept but no longer primary** — `notebooks/kaggle_run.ipynb` (23 cells, nbformat-validated) mirrors the Colab cell order with Kaggle specifics: GPU T4 + Internet ON, repo attached as a dataset and copied to writable `/kaggle/working`, `requirements-kaggle.txt` installed, `apt-get install -y tesseract-ocr libzbar0`, checkpoint copied from an attached dataset when present, outputs under `/kaggle/working/results/`. Every CLI flag was verified against `--help` or the argparse code; no flag was guessed. | Colab is no longer the execution environment; the notebook is the only way the full pipeline runs. | Done (code); execution NOT RUN YET — cannot be done from this VM. |
| **`requirements-kaggle.txt` has no torch line** (Kaggle ships a CUDA torch build; reinstalling it wastes time and can break the stack); `transformers>=4.36.0` minimum only, no upper bound (the code falls back to Captum if the Kaggle image ships transformers 5.x). `tldextract` is explicitly not included (optional in `url_risk.py`). | Reinstalling Kaggle's torch is the classic way to break a Kaggle notebook. | Done; dry-run install exit 0. |
| **`requirements.txt` rewritten + `pyproject.toml` aligned** — the old file was a 78-line frozen `pip freeze` dump; it is now 23 direct-dependency lines, with `pyproject.toml` canonical; added the three deps the code imports but pyproject was missing (`transformers-interpret`, `scipy`, `Pillow`) plus a new `[multimodal]` extra. Cross-check: every pyproject dep appears in `requirements.txt` with the identical version floor and vice versa, no mismatches. | A frozen dump is not a dependency declaration. | Done; both dry-runs exit 0; editable install still imports. |
| **`.gitignore` additions** — `data/*.csv` and an explicit `results/best_model.pt` line. Caveat: `data/Phishing_Email.csv` is already tracked, so the new rule does not untrack it; untracking was not done (coordinator decision). `results/debug/` was verified absent from the repo — there were no absolute paths to replace there and nothing to regenerate. | Belt-and-braces against committing data and checkpoints. | Done. |
| **Finding: `run_baselines.py` rewrites `baseline_results.json` on every invocation** — verified in code (`all_results` built fresh, saved with `save_json`, no merge), so the complexity report only sees the most recent run. The notebook states this in the H8 intro and adds an optional consolidated single-invocation cell. **Fixed in Section G (see below, G5)** by merging per-model results keyed by model+seed; the notebook caveat now describes the pre-fix behaviour. | A silent overwrite would have made the accumulated ladder unrecoverable on Kaggle. | Done (found in F; fixed in G). |

### H. Baseline & complexity-justification study (`baselines.py`, `run_baselines.py`, `run_complexity_report.py`) — Done (code); NOT RUN YET (full execution)

| Decision | Reason | Status |
|---|---|---|
| **Precision-first ladder** — primary metric phishing-class precision (user decision 2026-11-19); models ranked by precision first; recommended deployment model is the highest-precision model subject to recall floor 0.85 on validation aggregates; precision@recall-0.90 view with thresholds fitted on validation only; bootstrap precision CIs alongside F1 CIs; random search (~8 trials) and BiLSTM/CNN early stopping select on validation precision subject to recall >= 0.85. If no model meets the floor, the report says so explicitly instead of pretending it held. | The user's precision priority governs selection, not just reporting. | Done (code); full execution NOT RUN YET. |
| **Fairness constraints** — identical splits (`results/splits.json`) and identical `clean_text` cleaning for every baseline; vocab/TF-IDF/tokenizer fit on TRAIN only; train-only synonym augmentation; hyperparameter search on validation only; never tune on test; a single final test evaluation per model. | A baseline that trains on test data proves nothing. | Done; recorded in the `fairness` block of every `baseline_results.json`. |
| **Ladder implemented** — dummy, TF-IDF+LR, TF-IDF char-SVM (calibrated), BiLSTM (from-scratch embedding, AMP on CUDA, early stopping), 1D-CNN behind `--with_cnn`, DistilBERT behind `--with_distilbert` (downloads weights only when the flag is used), RoBERTa eval-only (metrics.json or checkpoint; missing means NOT RUN YET, never retrained). H3 metrics (accuracy, precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix, FPR, params, size, train wall-clock, latency ms/email median of 200 batch-size-1, peak memory), 3 seeds 42/1/7, mean ± std. H4 McNemar, bootstrap 95% CIs for F1 and precision, error analysis, robustness on 3 perturbations. H5 TF-IDF+LR coefficients and LIME, BiLSTM LIME + Captum IG, model-agnostic `predict_proba` faithfulness at k in {1,3,5}. H7 report generator with a code-written verdict (both branches unit-tested) and the dataset caveat (old Enron-style, highly separable; high simple-model scores do not imply holding on modern/AI-written or multimodal phishing). | The complexity question — is the transformer justified? — needs a fair ladder, not a single comparison. | Done (code); full execution NOT RUN YET. Only `--debug` smoke runs on 200/100/100 subsets were executed locally (TF-IDF+LR debug precision 1.0000 / recall 0.8947 on n=100; BiLSTM debug 1 epoch precision 1.0 / recall 0.395; those are subset numbers and are never quoted as real results). |
| **H5 note: `evaluate_faithfulness_single_method` cannot be reused for non-transformers** — it takes a `(PreTrainedModel, PreTrainedTokenizerFast)` pair and calls its own `predict_probability` internally. `evaluate_faithfulness_with_proba_fn()` implements the identical maths (comp = p_orig − p_drop, suff = p_orig − p_keep, k in {1,3,5}) over any `predict_proba(texts)` callable, which every baseline exposes. | Reusing the transformer-only function for a TF-IDF or BiLSTM model would have been a type error dressed up as a result. | Done; documented in the module docstring. |
| **RoBERTa gaps recorded honestly, not faked** — `roberta_eval` returns metrics/predictions but no fitted wrapper, so `--robustness` does not apply to it (perturbation F1 for RoBERTa needs a Kaggle run with the checkpoint wrapped in a `predict_proba` callable); DistilBERT has no validation-threshold view (no wrapper); DistilBERT test predictions are captured by a second inference-only pass over the test loader after `train_model` returns (recorded here because it is a second test pass, though no selection uses it). | A missing capability reported as a number would be fabrication. | Done (recorded); follow-ups for Kaggle. |
| **Multimodal hook (H6)** — `evaluate_on_extractor_output(model, rows)` and `evaluate_multimodal(models, eval_df)` with per-modality ablation over `body_text` / `ocr_text` / `pdf_text` / `qr_text`; CLI flags `--eval_csv` / `--multimodal_eval_dir`; a missing eval set reports NOT RUN YET with the exact re-run command. Section E's eval script did not exist when H was written, so the hook was implemented standalone and Section E wired it in. | The ladder must be comparable on the same extractor output, not on different text. | Done (code); execution on the Section E set NOT RUN YET as a `run_baselines.py` invocation (Section E ran TF-IDF through its own script instead). |
| **Verdict rule (no real numbers yet)** — headline on the PRECISION gap: if the best simple baseline (highest precision, recall >= 0.85) is within 1.0 precision point of RoBERTa, its bootstrap precision CI overlaps RoBERTa's precision, and McNemar p > 0.05, the verdict is "Transformer gives no statistically significant precision gain... complexity is NOT justified by accuracy alone. Value lies in the XAI/multimodal layer." Otherwise "Transformer gives a significant precision gain of X points ... at Yx latency and Zx parameters." (F1 gap stated as a secondary statement in both branches.) With no executed numbers the verdict is NOT RUN YET. | The verdict must be written before the numbers arrive, or the numbers will write the verdict. | Done (rule + unit tests); verdict itself NOT RUN YET. No committed `results/baselines/baseline_results.json` or `results/complexity_report.md` exists in the repo yet. |

### 6B.1. Scorer adapter (`TextScorer`) — Done (2026-11-19)

| Decision | Reason | Status |
|---|---|---|
| **Minimal `TextScorer` protocol in `baselines.py`** — `name`, `predict_proba(texts) -> (n, 2)` (rows sum to 1, col 1 = P(phishing)), `tokenize_words`, `truncate`. `RobertaScorer` wraps (model, tokenizer, device) and calls the EXISTING `score_text` exactly (lazy import, so the RoBERTa default path is byte-for-byte unchanged, incl. empty-text `(0, 1.0, 0.0)`); `BaselineScorer` wraps `BaselineModel` (word-cap truncation). `load_scorer(model, checkpoint, device, bilstm_epochs=3, seed=42)`: `"roberta"` delegates to `load_trained_roberta` + `resolve_default_checkpoint`; `"tfidf_lr"` / `"bilstm"` fit on the REAL train split via `load_splits` + `build_tfidf_lr` / `build_torch_baseline` exactly as `run_cheap_baselines` does (checkpoint ignored, synthetic eval set never fitted on); `ValueError` on unknown kind. `evaluate_random_baseline_with_proba_fn` mirrors `faithfulness.evaluate_random_baseline` (same RNG, same perturb helpers). `explain_word_scores` dispatches: RoBERTa = LIME + TI (`"lita"`), tfidf_lr = LIME + coefficients (`"lime_coefficients"`), bilstm = LIME + Captum IG (`"lime_ig"`), anything else = LIME only (`"lime_only"`, second scores = LIME copy, caller records `ti_available=False`, `alpha_effective=1.0` — TI is never fabricated). | The six scripts each hard-coded the (model, tokenizer) RoBERTa pair, so no baseline could drive the multimodal / faithfulness / ablation paths. | Done (code + 20 new tests); no Kaggle run, no metrics invented. |
| **`multimodal.py`: `score_texts(scorer, texts)`** (empty/whitespace -> 0.0) and **`build_concatenated_context(..., scorer=None)`** (scorer truncation takes precedence over tokenizer, then the existing char cap; default behaviour unchanged). `fuse_email` gains an optional `scorer=` kwarg passed through; docstrings reworded "RoBERTa P(phishing)" -> "scorer P(phishing)". | Fusion must score parts/context through any scorer without changing the default call. | Done. |
| **All six scripts gain `--model {roberta,tfidf_lr,bilstm}` (default `roberta`)**; `--checkpoint` help now says it is a RoBERTa `.pt`, ignored for tfidf_lr/bilstm. `run_multimodal_predict`: non-RoBERTa uses `load_scorer` + `predict_eml(..., scorer=...)`, no checkpoint gate, explanation via `explain_word_scores` + `compute_lita_scores`. `run_multimodal_eval`: new `score_email(scorer)` generalises the ablations; `score_email_roberta` is a thin wrapper over it; `evaluate()` now takes a scorer; `--model` drives (a)-(d), `--models` stays only for the cheap-baseline side table; checkpoint gate skipped when `--model != roberta`. `run_multimodal_faithfulness`: non-RoBERTa branch uses `scorer.predict_proba`, `evaluate_faithfulness_with_proba_fn`, `evaluate_random_baseline_with_proba_fn`, `explain_word_scores`; rows gain `model` / `explanation_method`. `run_explain`: non-RoBERTa skips the HF load entirely, output gains `model` / `explanation_method` (`--model_name` kept as the HF identifier). `run_faithfulness` / `run_ablation`: non-RoBERTa replaces `predict_probability` / `evaluate_faithfulness_single_method` / `evaluate_random_baseline` with the `*_with_proba_fn` versions and `explain_word_scores`; the alpha loop itself is unchanged; the dead `transformers` imports at line 12 of both files were deleted. | Each script needed the same branch without duplicating the RoBERTa path. | Done; RoBERTa paths untouched (verified by the unchanged-behaviour tests + full suite). |
| **New test file `tests/test_scorer_adapter.py` (20 tests)** — fake scorer defined in the file only (deterministic 0.9/0.1 trigger rule, empty text -> `[1.0, 0.0]`); RobertaScorer parity vs monkeypatched `score_text`; `load_scorer` delegation + train-split-only fit; `predict_eml` with the fake scorer (hand-computed weighted 0.585 / max 0.9); `score_email` ranges; multimodal faithfulness `lime_only`; dispatch for tfidf_lr / bilstm / fake; context-truncation precedence (scorer / char-cap / tokenizer); `--model` default `roberta` in all six `--help` outputs and `--model bilstm` skipping the missing-checkpoint error. | The adapter must be proven not to change RoBERTa behaviour and not to leak the eval set into baseline fits. | Done; all 20 pass. Full suite: **2 failed, 211 passed** (baseline 191 passed + 20 new; the 2 failures are the pre-existing environment ones below). |
| **Environment note (this run)** — the first full-suite run after the adapter showed 5 extra `test_extract` failures (OCR/QR samples) because a service restart had wiped the apt-installed `tesseract-ocr` + `libzbar0` from the ephemeral system layer; the suite log showed "Tesseract OCR binary not found" / "pyzbar unavailable". Both packages were reinstalled (`apt-get update` + `install`, slow mirror ~9 min) and all 20 extract tests pass again. No adapter code was changed for this. | Those 5 tests depend on system tesseract + zbar (Section B log says so); their failure was environmental, not a regression. | Done; verified by re-run. |

### 6B.2. Canonical-split fix in `run_train.py` (2026-11-19) — Done

**Bug (verified):** the documented canonical split is train 11,215 / val 2,803 / test 3,495 (`results/splits.json`, generated locally with scikit-learn 1.9.1), and the fairness contract (`scripts/run_baselines.py` header, `src/phishing_xai/baselines.py` docstring) says every model uses that manifest — but `scripts/run_train.py` ignored it and regenerated splits via `create_stratified_splits()` on every run. On Kaggle (scikit-learn 1.6.1, numpy 2.1.3) `StratifiedGroupKFold` produces a different split from the same 17,513 retained rows of the same CSV (dataframes verified identical): train 11,227 / val 2,786 / test 3,500. RoBERTa would therefore have trained and been evaluated on a different test set (n=3,500) than the documented one (n=3,495) and than the baselines, which do read `results/splits.json` via `baselines.load_splits()`.

**Fix (minimal, Option C — use the canonical manifest):** new `resolve_splits(df, data_path, output_dir, seed=42, debug=False)` in `scripts/run_train.py`. For full (non-debug) runs, if `<output_dir>/splits.json` already exists it rebuilds the exact splits from that manifest by reusing `baselines.load_splits()` (lazy import inside the function, to avoid import cycles; returns a dict of DataFrames, adapted into the `DatasetSplits` the rest of `run_train` expects). `load_splits` already raises `ValueError` on size or label-distribution mismatch against the manifest summary, and that behaviour is kept. If no manifest exists, behaviour is exactly as before (generate via `create_stratified_splits`, which saves the manifest). Debug mode (`--debug`) is exactly as before (subsample via `create_stratified_splits(debug=True)`) and never loads the manifest. The outcome (which path was taken + split sizes) is logged via the existing logger. No change to `make_splits.py`, `data.py`, or `baselines.py`; `model.py` and `faithfulness.py` untouched; no existing test modified, only `tests/test_run_train_splits.py` added (4 tests: manifest present returns the manifest's exact train/val/test `email_id`s and sizes 11,215/2,803/3,495 without rewriting the manifest; no manifest generates and saves one and a second call reuses the identical ids; debug never loads the manifest; a tampered manifest summary raises `ValueError`).

**Kaggle rule — do NOT run `make_splits.py` on Kaggle.** It regenerates the manifest with the Kaggle-environment split (11,227/2,786/3,500) and would overwrite the canonical manifest. Kaggle stage kernels must ship the committed `results/splits.json` as-is and verify it instead (check the summary sizes are 11,215/2,803/3,495 before training). Follow-up CLOSED 2026-11-19: `notebooks/kaggle_run.ipynb` Cell 4 no longer runs `make_splits.py`; it now verifies the shipped `results/splits.json` summary equals (11215, 2803, 3495) and asserts otherwise.

Full suite (this run): **2 failed, 215 passed** (211 baseline + 4 new; the 2 failures are the pre-existing environment ones). Note: the first full-suite run in this session showed 5 extra `test_extract` OCR/QR failures that did not reproduce — a pristine-tree re-run (changes stashed) gave the exact 211/2 baseline and a second with-changes run gave 215/2, so they were flaky/pre-existing, not a regression from this change (no extract test imports `run_train`, and the new test file runs after `test_extract` alphabetically).

### 6B.3a. GPU-only cuDNN fix in `integrated_gradients_scores` (2026-11-19) — Done (code); GPU verification PENDING

**Bug (verified on Kaggle GPU, Tesla T4, torch 2.11.0+cu130, captum 0.9.0):** in the full suite, `tests/test_scorer_adapter.py::test_explain_word_scores_dispatch_bilstm` failed with exactly one error line: `RuntimeError: cudnn RNN backward can only be called in training mode`. It passed standalone on Kaggle CPU (torch 2.11.0+cpu) and locally (CPU torch 2.15.0), so it is GPU/cuDNN-specific, not order-dependent. Call chain: test (`build_torch_baseline "bilstm"`, 1 epoch, tiny) -> `explain_word_scores` (`BaselineScorer`, method `lime_ig`) -> `baselines.py integrated_gradients_scores()` -> Captum `LayerIntegratedGradients(_forward, model.embedding).attribute(...)`.

**Why only on GPU:** `integrated_gradients_scores()` calls `model.eval()` and Captum then needs gradients through the BiLSTM. cuDNN's RNN backward requires the module to have run forward in training mode; in eval mode cuDNN takes a path that cannot backward. The CPU native RNN implementation supports backward in eval mode, so the same code passes on CPU.

**Fix (minimal, `src/phishing_xai/baselines.py`, `integrated_gradients_scores` only):** the model stays in eval mode (dropout `p=0.3` stays off, so attributions are deterministic) and the `lig.attribute(...)` call is wrapped in `with torch.backends.cudnn.flags(enabled=False):`, so the LSTM runs on the native implementation, which supports backward in eval mode. The context manager exists in torch 2.x and restores the previous flag values on exit (verified locally on torch 2.15.0+cpu: flag `True` before, `False` inside, `True` after). The `model.train()` + set-dropout-to-eval alternative was not used: it changes model state more broadly. No signature, return-value, or LIME-path change. The fix covers all callers, because every IG path goes through this one function (verified by grep): `evaluate_explainability` (method `"integrated_gradients"`) and `explain_word_scores` (bilstm, `"lime_ig"` — the 6B.3 LITA-IG path via `run_ablation.py` / `run_faithfulness.py` / `run_multimodal_predict.py` / `run_multimodal_faithfulness.py` / `run_explain.py`, all of which call `explain_word_scores`). No other place in `src/` or `scripts/` calls `LayerIntegratedGradients` on an RNN model in eval mode: the only other use is `explain.py TransformerExplainer._explain_with_captum`, a Captum fallback for transformer models (RoBERTa/BERT/DistilBERT, no RNN), so it is unaffected by this bug.

**Tests:** the existing `test_explain_word_scores_dispatch_bilstm` is the regression test (fails on Kaggle GPU, passes on CPU). One new CPU-observable test added in a new file `tests/test_ig_cudnn_eval.py` (no existing test modified): `test_integrated_gradients_deterministic_in_eval_mode` builds a tiny dropout=0.3 BiLSTM, calls `integrated_gradients_scores` twice, and asserts bit-identical scores, the model still in eval mode (dropout off), and the cuDNN flag restored afterwards. It passes locally. `model.py` and `faithfulness.py` untouched; nothing committed.

**Full suite (local, this run): `2 failed, 216 passed, 1 warning in 206.72s (0:03:26)`** — 216 passed is the previous 215 plus the 1 new test; the 2 failures are the same pre-existing environment ones (`test_environment_sanity` Python 3.10 assert; `test_captum_vs_transformers_interpret_tiny_model` missing `transformers-interpret`). **GPU verification is PENDING: the fix cannot be proved locally (no GPU here); the next Kaggle pytest run must confirm `test_explain_word_scores_dispatch_bilstm` passes in the full suite on the T4.**

### 6B.2-prep. BiLSTM tuning driver, rule-based scorer, bigger BiLSTM (2026-11-19) — Done (code); Kaggle execution NOT RUN YET

**1. BiLSTM random-search tuning driver (`scripts/tune_bilstm.py`, new, 209 lines):** samples `--n_trials` configs (default 10) over emb_dim {100,200}, hidden {64,128,256}, layers {1,2}, dropout {0.2,0.3,0.5}, lr {5e-4,1e-3,2e-3}, max_len {128,256}, batch_size {16,32}; trains each for `--epochs` (default 3, seed 42, one seed) via the EXISTING `build_torch_baseline`; selects with new `select_best_bilstm_trial` in `baselines.py` on VALIDATION precision subject to recall >= 0.85 (same rule as RoBERTa tuning). Fallback when NO trial meets the floor: highest recall, then precision, with `floor_met: false` in `results/bilstm/tuning.json` and a printed `WARNING: NO trial met recall >= ...` line — this differs deliberately from `random_search_tfidf`, whose no-floor fallback is highest precision overall (documented in the function's docstring). Tuning only: no test use, no full-data retrain; fitting on the train split, selection on validation only. Smoke-verified locally (`--debug`, 2 trials, 1 epoch, 200/100 subset): ran end to end and exercised the no-floor fallback path honestly (1-epoch smoke models predicted all-Safe, recall 0.0 — a smoke artefact, never a result).

**2. Rule-based-only multimodal config (`"rule_based"`):** new `RuleBasedScorer` in `baselines.py` (+ `load_scorer("rule_based")`, no data / fit / checkpoint) and `--model` gains `rule_based` in `run_multimodal_eval.py` and `run_multimodal_predict.py` (both already branched generically on `load_scorer` for non-RoBERTa, so only the choices list + help + a comment changed). RULE-BASED SCORE DEFINITION: P(phishing) for a text = the WORST (max) static `url_risk.score_url` score among URLs extracted from that text, 0.0 when the text has no URL; every body word is ignored. These are NOT model probabilities and must never be presented as such. Header/structural flags do not enter this score (they are not in the text); they reach the verdict through `fuse_email`'s separate flag channel, and links/QR URLs through its URL channel, so the (a)-(d) table goes through the same `score_email`/`fuse_email` path as the other backbones and stays comparable. Consequence, stated in the class docstring: weighted fusion double-counts URL evidence by construction (context/part scores ARE URL risk, plus the fusion's own 0.20 URL channel). A fixed neutral prior was NOT chosen: it would have made (a)-(c) identical constants and thrown away the only non-text evidence the pipeline has. Smoke-verified locally: `run_multimodal_eval.py --model rule_based --max_samples 12` ran end to end (12-sample smoke numbers only, never results). The other four `--model` scripts were deliberately NOT given `rule_based` (spec scope was eval + predict).

**3. Bigger BiLSTM variant — NO code change, no attention pooling:** attention pooling was NOT added (out of scope for this prep). Verified the existing `build_torch_baseline` accepts the bigger config via the existing CLI flags: `python scripts/run_baselines.py --models bilstm --seeds 42 --hidden 256 --layers 2 --emb_dim 200 --debug` ran cleanly (extra records emb_dim 200, hidden 256, layers 2; max+mean pooling unchanged). Kaggle command uses the same flags without `--debug`.

**Tests:** new file `tests/test_6b_prep.py` (9 tests, all passing, 43 s): selection rule on fabricated rows (precision-among-floor-meeters; no-floor fallback = highest recall then precision; empty raises); rule-based shape/row-sums, no-URL = 0.0, body-word independence at scorer level AND at `score_email` level, `load_scorer("rule_based")` with `load_splits` monkeypatched to raise; `--help` contains `rule_based` for both scripts; bigger BiLSTM builds + predicts (emb_dim 200, hidden 256, layers 2, 1 epoch, tiny frames). No existing test modified; `model.py` and `faithfulness.py` untouched; nothing committed.

Full suite (local, this run): **2 failed, 225 passed, 1 warning in 241.44s (0:04:01)** — 225 = previous 216 + 9 new; the 2 failures are the same pre-existing environment ones (`test_environment_sanity` Python 3.10 assert; `test_captum_vs_transformers_interpret_tiny_model` missing `transformers-interpret`).

### 6B.2-save-load. Fitted-baseline save / load so Step 6B.2 can save `results/bilstm/best_bilstm.pt` (2026-11-19) — Done (code); Kaggle execution NOT RUN YET

**What was added.** `src/phishing_xai/baselines.py` gains `save_baseline_model(model, path)` and `load_baseline_model(path, device=None)` (one file per model, one loader function) plus a byte-only artefact check `_is_baseline_artefact(path)`. `scripts/run_baselines.py` saves the best-seed fitted model per savable kind to `<output_dir>/best_<model>.pt` (`best_bilstm.pt`, `best_cnn.pt`, `best_tfidf_lr.pt`, `best_tfidf_svm.pt`). `load_scorer` in `baselines.py` now loads a saved baseline artefact when `checkpoint` points at one, and keeps fitting on the real train split otherwise. No metrics / `baseline_results.json` schema change; `model.py` and `faithfulness.py` untouched; no existing test modified; nothing committed.

**Artefact format.** Marker `BASELINE_ARTEFACT_FORMAT = "phishing_xai.baseline.v1"` in every file. Torch (`bilstm`/`cnn`): `torch.save` dict with `format`, `kind`, `config`, `state_dict` (CPU tensors), `tokenizer` (full `word2idx` + `max_vocab` + `max_len`), `n_params`, `extra` (public, `_`-free keys only). The rebuild config is recovered by introspecting the fitted module itself (embedding dim, LSTM hidden/layers, CNN filters/kernels, dropout), so no `TorchTrainConfig` is carried; loading rebuilds the SAME architecture via the existing `BiLSTMClassifier` / `CNNClassifier` classes (no duplicated module definition), restores `state_dict`, rebuilds the tokenizer exactly, and re-wraps in `BaselineModel` with the same name/kind and re-attached `_model` / `_tokenizer` / `_device_holder`, so `predict_proba` is identical. TF-IDF (`tfidf_lr`/`tfidf_svm`): `pickle` dict with `format`, `kind`, `pipeline` (the fitted sklearn pipeline / calibrated classifier), `n_params`, `extra`. The dummy baseline has no fitted artefact and raises `ValueError` on save.

**`load_scorer` checkpoint semantics (for `tfidf_lr` / `bilstm` / `cnn`; `cnn` also added to `SCORER_MODELS` and to the fit path).** When `checkpoint` is a path to an EXISTING file: if the file holds a baseline artefact (marker found by a byte-only scan — `data.pkl` inside the torch-save zip, chunk scan for a pickle), it is LOADED via `load_baseline_model` and no fitting happens; if it is a torch checkpoint without the marker (e.g. a RoBERTa `.pt`, dict with `model_state_dict`), the old behaviour holds exactly (fit on the real train split). When `checkpoint` is None or the path does not exist, fit as before. A checkpoint that exists but is neither a baseline artefact nor a torch checkpoint (e.g. a truncated pickle, garbage bytes) raises `RuntimeError`, and a baseline artefact that fails to load raises — never a silent fall back to fitting. Detection detail worth knowing: torch 2.15 names the pickle entry `<stem>/data.pkl`, not `archive/data.pkl` as older torch did — the peek accepts any `*/data.pkl` entry (found the hard way: the first version hardcoded `archive/data.pkl` and misclassified a valid torch-saved artefact as a RoBERTa checkpoint).

**`run_baselines.py` best-seed rule.** Per savable kind, each completed seed's fitted model is scored on VALIDATION (never test) with `compute_metrics_full` F1 at threshold 0.5; the best is kept, ties go to the lowest seed (explicit comparison, not seed order, since the default seed order is 42, 1, 7). Saving happens after the per-model loop, logs `Best <model> model (seed <s>, val F1 <f>) saved -> <path>`, and a save failure warns instead of failing the run (metrics already computed stay the primary output). Kaggle use: `--output_dir results/bilstm` makes the artefact land at `results/bilstm/best_bilstm.pt`.

**Wiring.** `run_multimodal_eval.py` and `run_multimodal_predict.py` already passed `--checkpoint` through to `load_scorer` for baselines, so no call-site change was needed there — only their `--checkpoint` / `--model` help text and comments, which wrongly said the checkpoint is "ignored for --model tfidf_lr/bilstm". The same one-line help fix was applied to `run_multimodal_faithfulness.py`, `run_ablation.py`, `run_explain.py`, `run_faithfulness.py`, which also pass their checkpoint through to `load_scorer`. `cnn` is loadable via `load_scorer("cnn", checkpoint=...)` but was deliberately NOT added to the two multimodal scripts' `--model` choices (spec scope was `bilstm`; the 6B.2 Kaggle flow only needs it).

**Tests:** new file `tests/test_baseline_save_load.py` (12 tests, all passing, ~25 s, CPU only): bilstm + cnn save -> load -> `predict_proba` exact equality (`assert_array_equal`); missing file raises `FileNotFoundError`; loader rejects a RoBERTa-style `.pt` with `ValueError`; tfidf_lr round-trip exact equality; `load_scorer` with a saved bilstm checkpoint returns matching predictions with `load_splits` / `build_torch_baseline` / `build_tfidf_lr` all monkeypatched to raise (proves no refit); `load_scorer` with a RoBERTa-style `.pt` fits on monkeypatched tiny frames instead; garbage-bytes checkpoint raises `RuntimeError`; truncated tfidf pickle raises `RuntimeError`; `load_scorer(checkpoint=None)` still fits for `tfidf_lr` / `bilstm` / `cnn` on tiny frames.

**Smoke verification (local, `--debug`, NOT results):** `run_baselines.py --models bilstm tfidf_lr --seeds 42 1 --epochs 1 --output_dir /tmp/smoke_baselines --debug --no_merge` wrote `best_bilstm.pt` (13.48 MB; best seed 1, val F1 0.5785) and `best_tfidf_lr.pt` (0.14 MB; best seed 42, val F1 0.8611); the saved bilstm artefact reloaded via `load_scorer("bilstm", checkpoint=...)` with predictions identical to a direct `load_baseline_model`. Those are 1-epoch debug-smoke numbers on a 200/100/100 subset, never results.

Full suite (local, this run): **2 failed, 245 passed, 1 warning in 404.68s (0:06:44)** — 245 = previous 225 + 12 new here + 8 from the concurrently-running 6B.5 system-complexity work; the 2 failures are the same pre-existing environment ones (`test_environment_sanity` Python 3.10 assert; `test_captum_vs_transformers_interpret_tiny_model` missing `transformers-interpret`). Note: the first full-suite attempt in this run reported `12 failed, 235 passed` because `/tmp` (512 MB tmpfs) filled to 100% mid-run; the other 10 failures were all `OSError: [Errno 28] No space left on device` artefacts — all 10 passed on re-run with `--basetemp` on the home disk, and the clean re-run above confirms it. No previously-passing test was broken.

### 6B.5. System-complexity report script (2026-11-19) — Done (code); Kaggle execution NOT RUN YET

**New `scripts/run_system_complexity_report.py` (1,056 lines, pure JSON analysis, no GPU, no training, no model loading):** reads only already-produced JSONs and writes `results/system_complexity_report.md` + `results/system_complexity_report.json` + 4 plots under `results/system_complexity_plots/` (precision vs latency, faithfulness @5, multimodal ablation bars, LSTM vs RoBERTa top-word agreement). Builds the S1-S4 table (S1 TF-IDF+LR, S2 BiLSTM, S3 DistilBERT, S4 RoBERTa) with precision/recall/FPR/F1 (mean±std), ROC-AUC, McNemar p vs S4, bootstrap 95% CI for precision, params, size MB, train time, CPU/GPU latency (median of 200), peak memory, explanation time/email, faithfulness @5, robustness F1, multimodal held-out precision/recall. Any column with no source data is `NOT RUN YET` with the reason; exit 0 with missing inputs, non-zero only on unreadable/corrupt JSON. S3 reads "skipped" unless DistilBERT was run. Explanation time/email has no source anywhere (no script records it) — permanently `NOT RUN YET` until a script starts timing explanations. CLI: `python scripts/run_system_complexity_report.py --results_dir results [--ood_csv PATH]` (Kaggle 6B-2 kernel: same command with `--results_dir /kaggle/working/results`, run after the 6B baselines/faithfulness/ablation/multimodal stages have written their JSONs; OOD test is `NOT RUN` unless an `--ood_csv` result is present in the inputs).

**Verdict rule (code-generated, copied verbatim into the .md):** the three conditions from `run_complexity_report.build_verdict` (gap ≤ 1.0 precision point, overlapping bootstrap precision CI, McNemar p > 0.05) must hold on BOTH the text test set and the multimodal held-out set for the "not statistically justified" branch — "The Transformer is not statistically justified by accuracy on this data; the project's value is the XAI + multimodal layer, and <S1/S2> is the recommended lightweight deployment." Otherwise the significant-gain branch reports the gain size and the cost multiples (latency x, params x). The text-only `build_verdict` string is also reproduced in the JSON for reference (reused by lazy import, not duplicated). Third branch, deliberate deviation from a strict two-branch spec: when a needed statistic is missing the verdict says so explicitly (`NOT RUN YET`, listing what is missing) instead of claiming a gain that was never measured. Recommended deployment line: highest precision subject to recall ≥ 0.85, tie-break lowest CPU latency (mirrors `baselines.select_deployment_model` except its tie-break is fewest parameters). Guard added after a real-data smoke run: multimodal numbers flagged `untrained_fallback` can never drive a verdict (the local `results/multimodal_eval/multimodal_eval.json` is untrained-fallback, so the smoke-run verdict is honestly `NOT RUN YET`, not a "gain" from base-model scores). Multimodal McNemar p / bootstrap CIs are not stored by `run_multimodal_eval.py` today, so the multimodal leg of the rule can only be decided once a comparison step records them (supported input shapes: a `multimodal` block in `baseline_results.json`, or top-level `mcnemar_vs_roberta` / `bootstrap_precision_ci` in a per-backbone `multimodal_eval.json`).

**Agreement (Jaccard + Spearman on top-5 words per email):** FEASIBLE from the explainability outputs — `run_explain.py` explanations.json stores per-email top words (`word_level_attributions` with `lita_score` per word, plus `lime_top_words` top 15, keyed by `id`), and the script computes mean Jaccard and mean Spearman (union ranked, absent word = rank 6) over matched email ids. NO code change to any explain script was needed or made. Two CLI-discipline requirements, recorded in the report's Sampling note: both models' explain runs must use the same `--num_samples` AND different `--output_dir` values (the default `results/explanations/explanations.json` would be overwritten by the second run). Faithfulness/ablation outputs do NOT store per-email top words (summary + global top-20 only; `faithfulness_records.jsonl` stores metrics per email_id but no words), so agreement cannot come from them.

**Sampling comparability (checked against the scripts):** `run_faithfulness.py` / `run_ablation.py` draw `test_df.sample(n=sample_size, random_state=seed)` from the same test split, so the SAME `--seed` + SAME `--sample_size` give the SAME emails for RoBERTa and BiLSTM — no fix needed, provided the splits are identical (verifiable post-hoc via `email_id` in `faithfulness_records.jsonl`). `run_explain.py` is different: it takes `test_df.head(num_samples)` (first N test rows, seed-independent), so its emails are NOT the faithfulness sample; agreement is still valid because both models explain the same head(N) emails.

**Tests:** new file `tests/test_system_complexity_report.py` (8 tests, all passing, synthetic JSONs in `tmp_path` only): all-inputs-present (S1-S4 table, no-gain verdict, recommended line, all 4 plots, Jaccard 4/6 on the fixture); no-gain case names S1; significant-gain case reports "9.00 points" + "25.0x latency" + "250.0x parameters"; missing inputs → `NOT RUN YET`, exit 0; corrupt JSON → non-zero exit; agreement missing → `NOT RUN YET`; multimodal-only gain with no text S4 (regression for a `TypeError` found by the real-data smoke run); untrained-fallback multimodal numbers never drive a verdict. `model.py` and `faithfulness.py` untouched; no existing test modified; nothing committed.

### G. Documentation + integration fix (this section, 2026-11-19) — Done

| Decision | Reason | Status |
|---|---|---|
| **G1. README rewritten around Kaggle** — Kaggle quickstart (GPU T4 + Internet ON, repo attached as a dataset, `notebooks/kaggle_run.ipynb` cells in order) replaces Colab as the primary path; the Colab notebook file is kept and named. Multimodal section rewritten to the real pipeline (extract → url_risk → fusion, `--fusion max\|weighted`, default weighted, threshold 0.5 placeholder to be tuned on validation). Test count corrected to the end-of-work recount (193). Limitations rewritten (precision-first, untrained-fallback labelling, synthetic eval set, NOT RUN YET items). | The old README still described Colab as primary and the max rule as the verdict, both superseded. | Done. |
| **G2. `docs/MULTIMODAL_NOTE.md` rewritten** — new mermaid architecture diagram (extract → url_risk → fusion → verdict/LITA overlay, with the eval-set and baselines branches), results section filled only from real runs (Section E smoke eval quoted only as untrained fallback, TF-IDF+LR synthetic numbers quoted as a baseline fitted on the real-data train split and evaluated on 120 synthetic mails, everything else NOT RUN YET), honest limitations. | The old note described the pre-B/C/D pipeline and quoted untrained six-sample numbers without a strong enough label. | Done. |
| **G3. This merge** — every section's log merged into `decision.md` and `docs/PROJECT_LOG.md` (A was already there; B, C, D, E, F, H added here). | One decisions record, or the logs rot in `.impl_logs/`. | Done. |
| **G4. Threat model & limitations section** — added to the README (most visible place) and cross-linked from `docs/MULTIMODAL_NOTE.md` and this log: OCR failure on adversarial images, text-only RoBERTa cannot see logos/layout spoofing, synthetic eval-set limits, dataset age (Enron-style), possible train/test leakage notes (near-dup grouping via MD5 of the first 200 alphanumerics; the 796-groups figure is unverified and is not re-asserted), latency with no measured numbers yet (NOT RUN YET), no URL fetching (static only), precision/recall trade-off. | The threat model was scattered across section logs; it belongs where a reader lands first. | Done. |
| **G5. Integration fix: `run_baselines.py` now merges `baseline_results.json`** — `merge_baseline_results(existing, new)` in `scripts/run_baselines.py`: loads the existing JSON when present, merges per-model results keyed by model+seed (a new per-seed entry replaces the old entry for the same model and seed, other seeds and untouched models are kept), recomputes each aggregate from the merged seeds with `aggregate_seed_results`, keeps real `comparisons_vs_roberta` / error-analysis / faithfulness / multimodal blocks when the new run has only placeholders, unions the fairness seed lists, and marks the output `merged_with_previous_run: true`. `--no_merge` opts out for a clean overwrite; a missing, unreadable or corrupt previous file falls back to writing the new run only (with a warning). Two new tests in `tests/test_baselines.py` (36 tests there now, all passing): same-seed replacement + untouched-model retention + aggregate recomputation, and no-previous-file passthrough. | Section F found that every invocation rewrote the file from scratch, so the complexity report only ever saw the most recent run. | Done (code + unit tests); accumulating a real ladder across several Kaggle invocations has not been exercised yet — verify on Kaggle. |
| **G6. Final recount + full pytest** — test functions recounted with `grep -r "def test_" tests/ | wc -l` at the end of the work: **193** (34 baselines + 12 data/augment + 7 explain/LITA + 7 explain/resumability + 20 extract + 5 faithfulness/ablation + 4 model/train + 20 multimodal eval + 37 multimodal fusion + 5 report + 1 sanity + 39 url_risk). Full suite run at the end of this section; exact result recorded in `.impl_logs/section_G.md`. The 2 pre-existing environment failures are recorded below, not fixed by changing project code. | Done; see "Pre-existing test failures" below. |

### Pre-existing test failures (environment, not code — recorded 2026-11-19)

At the integration run (`.impl_logs/integration.md`, after A+B+C landed, H+D still running) the full suite was **132 passed, 2 failed**: `tests/test_sanity.py::test_environment_sanity` asserts Python 3.10 while the local virtual environment runs Python 3.12.3, and `tests/test_explain_and_resumability.py::test_captum_vs_transformers_interpret_tiny_model` fails because the `transformers_interpret` package is not installed in the local venv (`ModuleNotFoundError`; it was not installed locally because it would force a transformers downgrade from 5.23 while other agents were using the venv; the Kaggle requirements pin transformers 4.x, where it is expected to work). Neither failure is caused by any section change. A third failure seen once in Section H's full-suite run (`tests/test_multimodal_fusion.py::test_url_score_contribution_benign_vs_malicious`) did not reproduce in Section D's or the integration run and is not attributed to any section; the final Section G full-suite result is the authoritative count.

## 5. Multimodal Extension Decisions (Steps 1–7)

| Decision | Reason | Step | Status |
|---|---|---|---|
| **Add Multimodal Dependencies (`pytesseract`, `pdfplumber`, `PyMuPDF`, `pyzbar`, `Pillow`, `qrcode`)** | Required for image OCR + word boxes, PDF text/link extraction, scanned-PDF page rendering, and QR code generation/decoding. Pinned exact versions in `requirements.txt` and added to `requirements-colab.txt`. | Step 1 | Complete |
| **User-Space Tesseract 5.4.0 Installation** | Installed UB-Mannheim Tesseract (`v5.4.0.20240606`) into `C:\Users\admin\AppData\Local\Programs\Tesseract-OCR` via `RunAsInvoker` (avoiding headless UAC hang) and configured `_configure_tesseract()` in `extract.py` to auto-detect both system and user-space paths. (Historical record: that path is the original author's Windows machine, 2026-11-12, not a path in this repo. Section B (B7) removed all hard-coded user paths from `extract.py`; discovery is now `$TESSERACT_CMD`, the system path, or platform defaults only.) | Step 1 | Complete |
| **Unified `.eml` Extraction Schema (`src/phishing_xai/extract.py`)** | `extract_eml()` walks MIME parts via Python's `stdlib email`, returning `{"body_text": str, "parts": [{"kind", "filename", "text", "boxes", "qr_urls", "page"}, ...]}` and exposing CLI `python -m phishing_xai.extract <file.eml>`. | Step 2 | Complete |
| **Scanned PDF Automatic OCR Fallback** | In `extract_pdf_bytes()`, if `pdfplumber` returns empty text on a PDF page, the page is rendered at 200 DPI via `PyMuPDF` (`fitz`) and passed to `ocr_image()` and `decode_qr()`, tagged as `kind="pdf_ocr"`. | Step 2 | Complete |
| **6 Synthetic Multimodal `.eml` Samples (`scripts/make_multimodal_samples.py`)** | Deterministically builds (a) image-only phishing body, (b) normal text + inline `'Verify Account'` button image, (c) QR code phishing email, (d) text PDF fake invoice with URI link, (e) scanned image-only PDF, and (f) clean benign email with harmless image & PDF. | Step 3 | Complete |
| **Zero Retraining & Unmodified `model.py` / `faithfulness.py`** | Hard constraint: never retrain the model and never modify `model.py`, `faithfulness.py`, or the 61 existing tests. | Step 4–6 | Complete |
| **Missing Checkpoint Reporting & Pretrained Fallback Flag** | Because full GPU training is deferred to Google Colab, `results/best_model.pt` is not present locally. `run_multimodal_predict.py` and `run_multimodal_faithfulness.py` stop by default and report the exact missing path (`results/best_model.pt`), while supporting `--allow_pretrained_fallback` to demonstrate pipeline execution with base `roberta-base` via `model.py`. (Section A, A2: fallback runs now also print a loud UNTRAINED MODEL banner and write `"untrained_fallback": true` into every output JSON; `false` when the real checkpoint is used. Note: `results/debug/multimodal_predictions.json` is NOT in this repo — `results/debug/` is gitignored — so its contents were not inspected here.) | Step 4 | Complete |
| **Max-Rule Multimodal Verdict** | An email is classified as Phishing (`1`) if **any** extracted part (body text, image OCR, QR URL, PDF text, or scanned PDF OCR) scores Phishing, naming the highest-scoring part as the winning part. (Superseded 2026-11-19 by Section D: weighted fusion is now the default verdict and the max rule survives only as a selectable `--fusion max` baseline for comparison — see Section 4, D.) | Step 4 | Complete |
| **Visual LITA Bounding-Box Overlay (`draw_lita_boxes_on_image`)** | Appended one new function `draw_lita_boxes_on_image` to `src/phishing_xai/explain.py` without changing any existing functions; draws red rectangles (`outline="red"`, `width=3`) around OCR word boxes matching top-5 LITA words and saves `<name>_explained.png`. | Step 5 | Complete |
| **Multimodal Faithfulness (`scripts/run_multimodal_faithfulness.py`)** | Reuses `evaluate_faithfulness_single_method` and `predict_probability` from `phishing_xai.faithfulness` unchanged to compute Comprehensiveness and Sufficiency at $k=5$ on the extracted text of the 6 samples. | Step 6 | Complete |
| **Pyrefly Config in `pyproject.toml`** | Added `[tool.pyrefly]` excluding legacy notebooks and ignoring `missing-source-for-stubs` for unused legacy `tensorflow` stubs. | Maintenance | Complete |

## 2026-11-19 ~18:00 PKT — Stage B "ERROR" post-mortem (training was COMPLETE)
- Kernel `phishing-xai-stage-b-tune-train-roberta` ended ERROR, but the ONLY kernel-side
  failure was the verify step in `stage_b.py` line 110:
  `assert (REPO / "results/best_model.pt").exists()` — run_train.py saves the
  checkpoint to `--ckpt_dir /kaggle/working/checkpoints`, NOT to results/.
  Epoch 3/3 finished 100% (1760/1760 batches), metrics.json complete:
  test (n=3495) accuracy 0.9854, precision 0.9690, recall 0.9932, F1 0.9809,
  ROC-AUC 0.9986; best_epoch 3, best_val_f1 0.9834; tuned params batch 8, lr 1e-5, 3 epochs.
  NO retraining was needed.
- Second, independent failure: the parent's local download of the kernel output to
  /tmp/stageb_out hit OSError Errno 28 (No space left on device) because /tmp is a
  512 MB tmpfs and best_model.pt is 476 MB. Lesson: NEVER download kernel outputs
  to /tmp; always use ~/workspace (141 GB free). The partial 331 MB copy in /tmp was
  deleted; the complete file was re-downloaded to persistent disk.
- Checkpoint VALIDATED locally (torch.load, CPU): 201 tensors, 124,647,170 params,
  keys epoch/model_state_dict/val_f1/val_metrics; epoch 3, val_f1 0.9834 — matches
  metrics.json exactly.
- Fix in `stage_b.py`: verify step now checks /kaggle/working/checkpoints/best_model.pt
  and copies it into results/best_model.pt. (No re-run; training already done.)
- Checkpoint dataset created: `syedaaleezazainab/phishing-xai-checkpoint` v1
  (best_model.pt 498,672,794 B + metrics.json). Results dataset created:
  `syedaaleezazainab/phishing-xai-results` — v1 held only STAGE_MANIFEST.json because
  `kaggle datasets create` defaults to --dir-mode skip (drops subdirectories, same bug
  as the repo dataset); v2 uploaded as results.zip (31 entries) and extracted under
  results/ — all stage C/D/E/6B-2 kernels got a merge fix (prefer the manifest whose
  parent holds stage_* dirs; flatten a nested results/ dir after copytree).
- Stage A COMPLETE (verified): dummy acc 0.6217/prec 0.0; tfidf_lr precision
  0.9702±0.0006, recall 0.9834, F1 0.9767, acc 0.9823, AUC 0.9981, FPR 0.0184;
  tfidf_svm precision 0.9820±0.0007, recall 0.9892, F1 0.9856, acc 0.9890,
  AUC 0.9992, FPR 0.0110 (3 seeds each, n=3495).
- Stage C pushed 18:01 PKT (RUNNING), Stage D pushed same time (QUEUED).

## 2026-11-19 ~18:40 PKT — Stage C v1 ERROR: CUDA assert on long email (NOT disk)
- Kernel `phishing-xai-stage-c-explainability` v1 ended ERROR ~6 min in. The
  kernel-side failure was NOT "No space left on device" — that OSError was
  again the local /tmp download (512 MB tmpfs; the parent's /tmp/stagec_out
  holds the 459 MB partial best_model.pt). The kernel log tail shows:
  `torch.AcceleratorError: CUDA error: device-side assert triggered`
  (ScatterGatherKernel index out of bounds) in `run_explain.py` on the 5th
  sampled email, email_27 (896 words). Reproduced locally with the exact
  sampling (test_df.head(10)): email_27 = 1029 tokens, email_30 = 1290
  tokens; the first four (91-238 words, <=280 tokens) succeeded.
- Root cause (real bug): `TransformerExplainer.explain_raw` passed raw text
  to transformers-interpret, which tokenizes with NO truncation, and
  `_explain_with_captum` encoded with no truncation either, while RoBERTa
  position_embeddings has 514 rows. Position ids are cumsum(non-pad mask) +
  padding_idx, so a >512-token email exceeds the table. Training truncates
  to 512, so truncating explanations to the first 512 tokens matches what
  the model actually sees. Fix in `src/phishing_xai/explain.py` ONLY
  (model.py / faithfulness.py untouched): `TransformerExplainer` gains
  `max_length=512`, capped at `position_embeddings.num_embeddings - 2`
  (roberta-base 514 -> 512; tiny test model 512 -> 510, matching its
  tokenizer model_max_length); `explain_raw` truncates via
  encode(truncation=True) + decode before the TI call; `_explain_with_captum`
  encodes with truncation. All TI paths go through `explain_raw`
  (run_explain, run_faithfulness, run_ablation, explain_word_scores), so
  one fix point covers Stage C, Stage D --faithfulness, and 6B-2.
- New regression test `test_transformer_explainer_truncates_long_text`
  (120x repeated sentence, 1563 tokens): both TI and Captum paths return
  <=512 attributions, finite scores, no crash. Full suite local after the
  fix + both 6B workers' code: **2 failed, 246 passed** (the 2 failures are
  the pre-existing environment ones).
- Also fixed in Stage C v2 script: v1 passed `--checkpoint_path` to
  run_faithfulness.py / run_ablation.py, which take `--model_path` (would
  have failed after explain); v2 uses the right flags, points all three
  scripts at the checkpoint in the INPUT mount (never copied into
  /kaggle/working), and at the end keeps only small outputs in
  /kaggle/working/outputs/ and removes the repo copy (code + 50 MB CSV).
- Risk flagged, not yet resolved: Stage D (RUNNING, dataset v5/v6 code)
  runs `--faithfulness` on 100 sampled test emails, which will almost
  certainly include a >512-token email, so it will likely ERROR with the
  same CUDA assert. If it does, re-push D with dataset v7 (fix included).

## 2026-11-19 ~19:00 PKT — Stage C v2 ERROR: CUDA out of memory (NOT disk, NOT the assert)
- Kernel `phishing-xai-stage-c-explainability` v2 ended ERROR ~3 min in.
  Kernel log (downloaded to ~/workspace/kaggle_downloads/stage_c_v2_log/):
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 300.00 MiB.
  GPU 0 has a total capacity of 14.56 GiB of which 148.81 MiB is free.
  ... 13.43 GiB is allocated by PyTorch, and 873.74 MiB is reserved by
  PyTorch but unallocated.` Failed in `run_explain.py` on email_27
  (896 words, 512 tokens after the v7 truncation fix): the
  transformers-interpret call OOM'd (logged as a warning), then the
  Captum fallback OOM'd on the same memory and the command failed.
- Root cause: Captum's `LayerIntegratedGradients.attribute` defaults to
  `internal_batch_size=None`, which runs ALL 50 IG steps in ONE forward
  (a 50 x 512-token batch for a long email). transformers-interpret uses
  the same default internally, so both paths blow the T4's 14.56 GiB.
- Fix in `src/phishing_xai/explain.py` ONLY for the RoBERTa path
  (model.py / faithfulness.py untouched): `_explain_with_captum` now
  passes `n_steps=50, internal_batch_size=5` — same 50 steps, same math,
  chunked forwards, so attributions are unchanged and only peak memory
  drops. `explain_raw` calls `torch.cuda.empty_cache()` after a
  successful TI call and before the Captum fallback (TI's cached blocks
  were what the fallback tripped over). Same bound added to
  `integrated_gradients_scores` in `baselines.py` (BiLSTM LITA-IG path,
  used in 6B) plus `empty_cache` after it. Existing regression test
  `test_transformer_explainer_truncates_long_text` already exercises
  `_explain_with_captum` on a >512-token text, so it now also covers the
  bounded call (CPU, tiny model).
- Stage C v3 script also sets `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` (fragmentation was 873 MB reserved-unallocated).
- Stage D (RUNNING since 18:05 PKT on dataset v5/v6 code) will hit the
  same OOM in its `--faithfulness` step (unbounded Captum, pre-fix code)
  — if it errors, re-push D with the v8 dataset (this fix included).
  Do NOT stop D while it runs.

## 2026-11-19 ~19:12 PKT — Stage C v3 COMPLETE (OOM fix worked)
- Kernel `phishing-xai-stage-c-explainability` v3 COMPLETE in ~16 min.
  Outputs downloaded to ~/workspace/kaggle_downloads/stage_c_v3/outputs/:
  10 explanations (email_1/17/18/25/27/29/30/37/42/44 — incl. the two
  long emails that crashed v1/v2), faithfulness n=97 of 100 requested,
  alpha ablation n=49 of 50 requested, no errors/warnings in the log.
- Faithfulness (comp/suff, higher comp + lower suff = better; random
  baseline comp@5 0.0165 / suff@5 0.4258): LITA-magnitude @5 comp 0.1055
  / suff 0.0809; LIME @5 comp 0.1115 / suff 0.1640; TI @5 comp 0.0337 /
  suff 0.2058. All explainers beat random on comprehensiveness; LITA
  has the best sufficiency at every k.
- Alpha ablation (k=3, magnitude variant): alpha 0.0 (pure TI) comp
  0.0703 / suff 0.1743; 0.25 comp 0.0947 / suff 0.1074; 0.5 comp 0.1296 /
  suff 0.0665 (best on both); 0.75 comp 0.1424 / suff 0.1169; 1.0 (pure
  LIME) comp 0.1272 / suff 0.1900. Alpha=0.5 is the best balance on
  this data; the paper's unjustified alpha=0.5 now has measured support.
- Results dataset updated to v3 (stage_c/ added; manifest
  stages_completed A, B, C; repo v8, commit dde5d50).
- 6B-1 pushed as `phishing-xai-stage-6b-1-bilstm-training` (kernel v1;
  first push attempt hit Kaggle's 2-concurrent-GPU-session cap while C
  and D ran, second attempt 409'd on a slug mismatch — title slug is
  `...-6b-1-bilstm-training`, metadata id corrected to match).

## 2026-11-19 ~20:00 PKT — Stage D COMPLETE (full ladder, all executed)
- Kernel `phishing-xai-stage-d` COMPLETE. Ladder (test n=3495, 3 seeds
  unless noted; precision primary):
  TF-IDF+SVC P 0.9820±0.0007 / R 0.9892 / F1 0.9856 / AUC 0.9992;
  TF-IDF+LR P 0.9702±0.0006 / R 0.9834 / F1 0.9767 / AUC 0.9981;
  DistilBERT (1 seed) P 0.9731 / R 0.9849 / F1 0.9789;
  CNN (1 seed) P 0.9732 / R 0.9614 / F1 0.9673;
  BiLSTM untuned config P 0.9632±0.006 / R 0.9607 / F1 0.9618;
  RoBERTa (eval-only, same checkpoint 3x) P 0.969 / R 0.9932 /
  F1 0.9809 / AUC 0.9986 / FPR 0.0193; dummy acc 0.6217.
  Headline: TF-IDF+SVC beats RoBERTa on precision AND F1; RoBERTa
  wins on recall only. (McNemar/bootstrap still missing — see below.)
- Gaps in Stage D's own outputs (recorded, not papered over):
  (a) `comparisons_vs_roberta` is EMPTY and the top-level `roberta`
  block says NOT RUN YET ("metrics file not found:
  results/metrics.json") even though `models.roberta_eval` holds the
  real per-seed numbers — the comparison code reads the stale
  top-level block, so no McNemar p or bootstrap CI was computed;
  (b) no `robustness` block in baseline_results.json despite the
  --analysis --robustness step; (c) `error_analysis` NOT RUN YET
  ("RoBERTa test predictions unavailable") even though y_pred arrays
  ARE saved for every model incl. roberta_eval; (d) `faithfulness`
  block holds only tfidf_lr (coefficients, n=100, comp@5 0.0272 /
  suff@5 0.5531) — the other models' faithfulness did not land.
  These gaps are for Step 9/10 to close from the saved predictions
  (code-generated, same formulas), not by re-running the ladder.
- Stage D did NOT hit the predicted OOM: its --faithfulness step ran
  (only tfidf_lr recorded) and the kernel completed. The v8 OOM fix
  was in the repo dataset for D? NO — D ran on dataset v5/v6 code
  (pushed 18:05, before v8 existed); it completed anyway, likely
  because its faithfulness path uses the baseline explainers, not the
  RoBERTa TI path. Either way, D is done and its numbers stand.
- 6B-1 v1 ERRORed AFTER tuning completed (best BiLSTM config emb 200 /
  hidden 64 / layers 2 / dropout 0.5 / lr 0.002 / max_len 128 / batch
  16; val precision 0.9798, recall 0.9364, F1 0.9576, floor MET):
  FileNotFoundError reading results/bilstm/tuning.json — the script
  read the repo-relative path while tune_bilstm.py's Kaggle redirect
  wrote to /kaggle/working/results/bilstm/. Fixed in the kernel script
  only (results_path() helper, Kaggle path first); v2 re-pushed.
  Tuning re-runs in v2 (~7 min; BiLSTM epochs are ~8 s each on GPU).
- Stage E pushed 20:0x PKT (kernel v1, QUEUED) with the v3-style
  hardening (input-mount checkpoint, expandable_segments).

## 2026-11-19 evening — Kaggle run, recovery coordinator (Stages C/D/E, 6B-1, 6B-2)

- Stage C v3 COMPLETE ~19:12 PKT (kernel phishing-xai-stage-c-explainability
  v3, ~16 min). v1 failed on CUDA ScatterGather assert (email_27, 1029
  tokens > RoBERTa 514 position rows); v2 failed on CUDA OOM in Captum IG
  (default internal_batch_size=None runs all 50 steps in one 50x512 forward).
  Fixes (commit dde5d50): truncate to 512 tokens in explain.py; Captum
  internal_batch_size=5 + empty_cache between explainers;
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. Faithfulness n=97/100:
  LITA-magnitude @5 comp 0.1055/suff 0.0809; LIME @5 comp 0.1115/suff 0.1640;
  TI @5 comp 0.0337/suff 0.2058; random @5 comp 0.0165/suff 0.4258 (all beat
  random on comprehensiveness; LITA best sufficiency at every k). Alpha
  ablation n=49/50, k=3: alpha 0.5 best on both (comp 0.1296/suff 0.0665);
  alpha 0.0 pure TI (0.0703/0.1743) and 1.0 pure LIME (0.1272/0.1900)
  verified as endpoints.
- Stage D COMPLETE ~20:00 PKT. Ladder test n=3495: TF-IDF+SVC P 0.9820+-0.0007 /
  R 0.9892 / F1 0.9856 / AUC 0.9993; TF-IDF+LR P 0.9702 / R 0.9834 / F1 0.9767;
  DistilBERT P 0.9731 / R 0.9849 / F1 0.9789; CNN P 0.9732 / R 0.9614 /
  F1 0.9673; BiLSTM untuned P 0.9632+-0.006 / R 0.9607 / F1 0.9618; RoBERTa
  eval-only P 0.969 / R 0.9932 / F1 0.9809. Headline: TF-IDF+SVC beats RoBERTa
  on precision AND F1; RoBERTa wins on recall only. Gaps in D's own JSON
  (see entry above) closed LOCALLY from the saved predictions with the
  repo's own stats functions (seed 42, n_resamples 1000; script
  ~/workspace/kaggle_downloads/compute_stage_d_comparisons.py): McNemar
  tfidf_svm vs RoBERTa p=0.253 (not significant), tfidf_lr p=0.3491,
  distilbert p=0.6025, cnn p=0.00056, bilstm(untuned) p=0.00066,
  dummy p~0; precision CIs: tfidf_svm [0.9746, 0.9889], tfidf_lr
  [0.9607, 0.9793]. Error analysis (40 rows) computed the same way;
  deployment selection: tfidf_svm (highest precision, recall floor 0.85
  met). Results dataset v4 carries the patched file + error_analysis.csv.
  Robustness stays NOT RUN YET (no rerun; RoBERTa robustness needs the
  checkpoint on GPU). Stage D's complexity report is all-NOT-RUN-YET
  (path-redirect bug: it ran with cwd=REPO so it looked under the repo
  copy instead of /kaggle/working/results) — superseded by 6B.5.
- 6B-1 v2 ERRORed on KeyError 'best_config' (tuning.json schema has
  best_params, not best_config) after tuning re-ran fine. Fixed in the
  kernel script only (read tuning["best_params"], fallback
  selection.best.params); v3 re-pushed and COMPLETE ~20:22 PKT. Tuned
  BiLSTM 3-seed test: P 0.9719+-0.0015 / R 0.9428+-0.0164 / F1 0.9571+-0.0077
  / AUC 0.9942 / FPR 0.0166, 6.24M params, 23.79 MB, train ~90-160 s/seed.
  Bigger variant (hidden 256, 1 seed): P 0.9711 / R 0.9652 / F1 0.9681 —
  capacity was not the gap. McNemar vs RoBERTa (computed locally from saved
  predictions, same method): tuned p=0.00018, big p=0.00091 — RoBERTa still
  significantly better on overall errors; precision CIs overlap RoBERTa's
  0.969 ([0.9613, 0.9789] tuned, [0.9615, 0.9796] big). Merged into the
  ladder file (bilstm entry REPLACED, bilstm_big added; script
  ~/workspace/kaggle_downloads/merge_tuned_bilstm.py). Results dataset v5.
- Stage E v1: eval ran fully and VALID on the trained checkpoint
  (untrained_fallback:false everywhere; 6 predictions all false; sensible
  high-confidence probabilities), but the kernel status was ERROR on the
  post-run guard: the guard scanned REPO/results, which holds MERGED PRIOR
  results whose untrained_fallback:true is legitimate history (the Stage A
  file is true because Stage A never ran RoBERTa). The roberta-base
  "MISSING classifier" lines in the log are the intermediate
  from_pretrained report, not the final state — the checkpoint state dict
  loads after it. Fixed in stage_e.py v2 (guard scoped to new Stage E
  outputs only; checkpoint size check >100 MB; faithfulness --output_json);
  v2 COMPLETE ~20:49 PKT with test_metrics IDENTICAL to v1 (deterministic
  reproduction). 4-ablation, test n=96, thresholds fitted on validation
  n=24 only: (a) body only P 0.8605/R 0.7708/F1 0.8132/FPR 0.125;
  (b) +OCR P 0.8776/R 0.8958/F1 0.8866/FPR 0.125; (c) +QR+PDF P 0.9792/
  R 0.9792/F1 0.9792/FPR 0.0208; (d) full fusion weighted P 1.00/R 0.9792/
  F1 0.9895/FPR 0.00 (threshold 0.6); (d) max rule P 0.9130/R 0.875/
  F1 0.8936/FPR 0.0833 (threshold 0.9962). Weighted fusion beats max.
  1 failure: scanned_pdf_001.eml (corroboration gate; max rule would have
  caught it at 0.9998, weighted fused 0.5449 < 0.6). Baselines on the same
  120 synthetic: tfidf_lr P/R/F1 0.95; bilstm untuned 3-epoch P 0.71/R 0.82/
  F1 0.76. Multimodal faithfulness (6 samples, LITA k=5, saved as JSON in
  v2): comp 0.004-0.065 on phishing samples (model saturated at P~0.999),
  suff ~0.0 (beats random suff 0.18-0.23 on samples 4-5); benign sample
  comp 0.9418. OCR (tesseract) + QR (zbar) verified working on the 6 samples.
- 6B-2 fixes before push (kernel script only, no repo code change):
  (a) run_faithfulness.py/run_ablation.py take --model_path, not
  --checkpoint_path (v1 of the script had the wrong flag); (b) --output_dir
  on all 6B.3 calls (explain -> results/bilstm/explanations, faithfulness
  and ablation -> results/bilstm); (c) RoBERTa multimodal eval passes
  --output_dir results/multimodal_eval (its default is the Kaggle-redirected
  /kaggle/working/results, which the 6B.5 report never reads — same redirect
  trap as Stage D's complexity report); (d) BiLSTM multimodal eval to
  results/bilstm/multimodal_eval (the report's S2 location); (e) flatten
  stage_c/{explanations,faithfulness.json,alpha_ablation.json} and
  stage_b/metrics.json to results/ root after the merge (the report reads
  them at root, the merge leaves them under stage_c//stage_b/); (f) size
  checks on both checkpoints; (g) metadata title fixed to resolve to the
  kernel id (first push failed on title/slug mismatch). Results dataset v5
  attached (bilstm/best_bilstm.pt + tuning + merged ladder + Stage E
  outputs); the bigger variant's .pt is deliberately NOT in v5 so
  find_one("best_bilstm.pt") cannot pick the wrong checkpoint.
- Multimodal McNemar/CI note for Step 9: run_system_complexity_report.py
  reads mcnemar_vs_roberta/bootstrap_precision_ci from the multimodal eval
  files, but no eval writes them ("Absent in current outputs"). After 6B-2,
  compute them locally from the per-row predictions in the 6B-2 eval files
  (roberta vs tfidf_lr/bilstm/rule_based on the synthetic test n=96) and
  re-run the report locally (CPU, JSON-only) so the verdict is fully
  evidenced instead of "undecided" on the multimodal set.

- 6B-2 v1 ERRORed at the 6B.4 RoBERTa multimodal eval (ValueError: No eval
  emails could be scored). Root cause: the eval set is NOT regenerated by
  run_multimodal_eval.py (the 6B-2 docstring claim was wrong) and the repo
  dataset ships labels.csv but no .eml files (excluded from dataset v8),
  so labels loaded but every file was missing. Fix in stage_6b2.py v2: run
  make_multimodal_samples.py + make_multimodal_eval_set.py before the 6B.4
  evals (seed 42, identical set to Stage E); v2 re-pushed 20:56 PKT.
  6B.3 COMPLETED in v1 and its outputs are verified (downloaded to
  kaggle_downloads/stage_6b2_v1/repo/results/bilstm/): 10 explanations on
  the SAME email ids as Stage C (email_1/17/18/25/27/29/30/37/42/44);
  faithfulness n=97/100 (3 skipped: too short after truncation to 128),
  LIME @5 comp 0.0814/suff 0.0002, LITA-IG (lime_ig) @5 comp 0.0752/
  suff -0.0044, LITA-magnitude @5 comp 0.0950/suff -0.0027, LITA-minmax
  @5 comp 0.0909/suff -0.0027, random @5 comp 0.0145/suff 0.1826 — all
  BiLSTM explainers beat random on both metrics; alpha ablation n=49/50,
  k=3: minmax alpha 0.5 best on both (comp 0.0873/suff 0.0126),
  magnitude alpha 0.5 best comp (0.0829) with alpha 0.25 best suff
  (0.0150) — alpha 0.5 is the best balance on both models (RoBERTa Stage C:
  0.1296/0.0665). Quirk noted: BiLSTM LIME top words contain duplicate
  tokens (e.g. "spanish" 3x, "which" 4x in email_1) from subword pieces
  mapping to the same word — cosmetic, affects top-5 Jaccard slightly.

- 6B-2 v1 ERRORed at the 6B.4 RoBERTa multimodal eval (ValueError: No eval
  emails could be scored). Root cause: the eval set is NOT regenerated by
  run_multimodal_eval.py (the 6B-2 docstring claim was wrong) and the repo
  dataset ships labels.csv but no .eml files (excluded from dataset v8).
  Fix in stage_6b2.py v2: run make_multimodal_samples.py +
  make_multimodal_eval_set.py before the 6B.4 evals (seed 42, identical set
  to Stage E); v2 COMPLETE ~21:21 PKT. 6B.4 per-backbone full fusion (d)
  weighted, test n=96: RoBERTa P 1.00/R 0.9792/F1 0.9895 (IDENTICAL
  test_metrics to Stage E — deterministic reproduction); BiLSTM P 1.00/
  R 1.00/F1 1.00, 0 failures; TF-IDF+LR P 0.9412/R 1.00/F1 0.9697,
  3 failures; rule-based P 1.00/R 0.8333/F1 0.9091, 10 failures. 6B.5
  report generated by the kernel (system_complexity_report.md/.json + 4
  plots): verdict (text set) -0.29 points vs tuned BiLSTM (McNemar
  p=0.0002, 20.0x params); agreement Jaccard 0.1004 / Spearman -0.4974
  (n=10); recommended deployment (S1-S4 only): distilbert.
- 6B.5 local re-run (2026-11-19 ~21:30 PKT, CPU, JSON-only) closed the
  report's multimodal-statistics gaps: the kernel's S4 eval file had an
  empty baselines block (no --models flag) so S1 multimodal was NOT RUN
  YET, and no eval file stores McNemar/CI. Computed locally from the saved
  per-row predictions (repo's own mcnemar_test / bootstrap_precision_ci,
  seed 42, 1000 resamples, test n=96, pred_d_weighted; script
  ~/workspace/kaggle_downloads/patch_mm_stats.py): BiLSTM vs RoBERTa
  McNemar p=1.0, CI [1.0, 1.0]; TF-IDF+LR p=0.625, CI [0.8718, 1.0];
  rule-based p=0.0156, CI [1.0, 1.0]. Patched the eval files, restored
  Stage E's baselines block into the S4 file (identical deterministic
  run), added a "multimodal" block to the ladder file, and re-ran
  run_system_complexity_report.py locally (relative --results_dir so no
  local absolute paths leak into the output). Final verdict: "Transformer
  gives a significant precision gain of 5.88 points on the multimodal
  held-out set (RoBERTa 1.0000 vs best simple baseline tfidf_lr 0.9412;
  McNemar p = 0.6250) at unknown latency and 3116.1x parameters."
  Text-only verdict: -1.30 points vs TF-IDF+SVC (p=0.2530). S2 passes the
  multimodal triple check (gap 0.0, CI contains 1.00, p=1.0) but fails the
  text check (McNemar p=0.0002), so the "Transformer is not statistically
  justified" verdict did NOT trigger on either set. Plain reading:
  RoBERTa's precision is BELOW the best simple baseline on text (-1.30pt
  vs TF-IDF+SVC) and its multimodal "gain" (5.88pt vs TF-IDF+LR) is not
  significant by McNemar (p=0.625, n=96); its real advantage is recall.
  Results dataset v6/v7 carry the final report + patched files.
- Step 9 fixes (2026-11-19): the ladder file's top-level `roberta` block
  said NOT RUN YET with untrained_fallback:true (stale Stage A skeleton —
  results/metrics.json was absent when Stage A ran) even though
  models.roberta_eval holds the real numbers; corrected to the executed
  Stage B single evaluation (P 0.9690, best epoch 3, test n=3495) with a
  provenance note, and top-level untrained_fallback set false. No number
  invented. Step 9 checks all PASS: no untrained_fallback:true in any new
  Stage E/6B-2 output; RoBERTa test n=3495 (classification-report support
  2173+1322); McNemar/bootstrap/error_analysis present; no local absolute
  paths in outputs; seeds 42/1/7 present; canonical splits verified.

- Verdict-generator fix (2026-11-19, user item 1): `scripts/run_complexity_report.py::build_verdict` and `scripts/run_system_complexity_report.py::build_system_verdict`/`_check_set` no longer say "significant precision gain" unless the precision gap is POSITIVE and McNemar p < 0.05. The old rule flagged gap > 1.0pt alone, p <= 0.05 alone, or a non-overlapping CI alone, so it labelled a -1.30-point text gap (RoBERTa 0.9690 vs TF-IDF+SVC 0.9820, p=0.2530) and a +5.88-point multimodal gap (RoBERTa 1.0000 vs TF-IDF+LR 0.9412, p=0.6250) both "significant". New branches: (a) positive gap + p < 0.05 -> "significant precision gain of X points" with latency/param multiples; (b) negative gap -> "Transformer is below the best simple baseline by X precision points" (with p, ratios, F1 gap); (c) otherwise -> "no statistically significant difference in precision". The "no significant gain" branch (gap <= 1.0pt, CI overlap, p > 0.05 -> complexity NOT justified) is unchanged. A check whose multimodal numbers are flagged untrained_fallback is excluded from every verdict branch (it previously fell through to the undecided branch, which still happens). One existing test (`test_mm_only_significant_gain_no_text_s4`) encoded the old rule (a 6.00-point gap with NO McNemar p stored was called significant); updated to expect the neutral wording, keeping its no-crash regression purpose. New tests: 3 in tests/test_baselines.py (negative gap, positive non-significant, p exactly 0.05) and 5 in tests/test_system_complexity_report.py (direct _check_set definition + 4 build_system_verdict branches). Full local suite after the fix: 254 passed, 2 failed (the 2 pre-existing environment failures). Regenerated verdicts from the already-measured results-dataset v7 numbers with the fixed generator (local re-run, no new GPU): system verdict "Transformer is below the best simple baseline by 0.11 precision points on the text test set (RoBERTa 0.9690 vs tfidf_lr 0.9701; McNemar p = 0.3491)"; text-only verdict "below the best simple baseline by 1.30 precision points (RoBERTa 0.9690 vs tfidf_svm 0.9820; McNemar p = 0.2530)". Recommended deployment unchanged: distilbert (S3, P 0.9731, R 0.9849). Regenerated files: results/system_complexity_report.md/.json (replacing the stale NOT RUN YET placeholders). model.py and faithfulness.py untouched.

- Rerun items 2/3/5/6 (2026-11-19, fix-and-rerun round; model.py and faithfulness.py untouched): (a) tfidf_svm scorer adapter — `baselines.py` `SCORER_MODELS` + `load_scorer` now accept `"tfidf_svm"` (saved artefact via `load_baseline_model`, else fit via the existing `build_tfidf_svm` on the real train split, same as tfidf_lr; a RoBERTa `.pt` falls through to fitting, a corrupt artefact raises); `run_multimodal_eval.py` `--model` / `--models` choices + `run_cheap_baselines` accept it too. 9 new tests in `tests/test_tfidf_svm_scorer.py` (scorer proba sums to 1, dispatch monkeypatched, artefact round-trip through `load_scorer`, `--help` lists the choice, robustness/latency helpers on stubs). Full local suite after the adapter: 263 passed, 2 failed (the same 2 pre-existing environment failures). (b) New `scripts/run_robustness_loaded.py`: robustness (truncated / urls_removed / typo_synonym, test split, threshold 0.5) for artefact-backed scorers only — RoBERTa via the `RobertaScorer` predict_proba wrapper (the wrapper `run_baselines.py --robustness` lacks for `roberta_eval`) and the tuned BiLSTM via its saved artefact; TF-IDF models fit on the real train split; DistilBERT / CNN / bigger-BiLSTM are recorded NOT RUN YET in the output (no saved checkpoint exists for any of them; each would need a refit). Never retrains. (c) New `scripts/run_latency_explain_time.py`: RoBERTa + tuned-BiLSTM latency, median of 200 emails at batch size 1, measured separately on CPU and on GPU and labelled by device (TF-IDF latency is NOT re-measured — the Stage D ladder file already holds it: TF-IDF+LR CPU 1.08 ms, TF-IDF+SVC CPU 8.11 ms), plus explanation time per email (RoBERTa LITA, BiLSTM LITA-IG) via the existing `explain_word_scores` path, same 10 test emails Stage C explained (email_1/17/18/25/27/29/30/37/42/44), lime_samples=1000; median + mean/std + per-email seconds. Kernels `phishing-xai-rerun-multimodal` (items 2+5: 5 backbones x floors 0.80+0.85), `phishing-xai-rerun-robustness` (item 6: refit route tfidf_lr/tfidf_svm/cnn seed 42 + artefact route roberta/bilstm), `phishing-xai-rerun-latency` (item 3) in `~/workspace/kaggle_kernels/rerun_*/`. (d) Item 4 (RoBERTa tuning) NOT re-run: it already executed in Stage B — `results/stage_b/tuning/results.json` in results dataset v7 holds the full 10-trial random search (same budget item 4 specifies: n_iterations 10, max_length 256, subsample_train 4000, seed 42; search space batch [8,16,32] x lr [1e-5,2e-5,3e-5] x epochs [3,4,5]; best batch 8 / lr 1e-5 / epochs 3, best val F1 0.9807, val P 0.9722) and the Stage B retrain used exactly those params (summary.json config lr 1e-5, epochs 3, best epoch 3; test P 0.9690/R 0.9932/F1 0.9809). The "fixed config" IS the tuned config, so a re-run would spend GPU quota to reproduce the same search. Note: the same trial params (16/1e-5/3) scored val F1 0.9796 in iteration 3 but 0.9661 in iteration 6 — GPU nondeterminism at the tenth-of-a-point level, so the 0.9807 best is within noise of the other trials. The PROJECT_LOG Kaggle-round row and the task brief both said tuning had not run; that is corrected here.
