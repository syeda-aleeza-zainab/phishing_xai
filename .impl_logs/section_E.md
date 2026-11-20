# Section E — Evaluation set + ablation (eval set, eval script, faithfulness) — change log

Date: 2026-11-19. Scope: `scripts/make_multimodal_eval_set.py` (new),
`scripts/run_multimodal_eval.py` (new), `tests/test_multimodal_eval.py` (new),
`scripts/run_multimodal_faithfulness.py` (random baseline + guard only),
plus generated outputs `data/multimodal_eval/` and `results/multimodal_eval/`.
Did NOT touch: `src/phishing_xai/model.py`, `src/phishing_xai/faithfulness.py`,
`decision.md`, `docs/PROJECT_LOG.md`. No push, no retrain, no fabricated
metrics. No URL in any email/QR/PDF was fetched or visited (static analysis
only). No `.venv` content is included in any output.

## Precision-first design (user priority 2026-11-19: "mera priority focus precision hai")

- Every table in the eval report leads with phishing-class PRECISION + FPR;
  recall is always shown alongside so the trade-off stays visible.
- The headline is a precision-first summary line naming the best-precision
  ablation at the fitted thresholds, e.g. "Best precision at recall floor
  0.80: (d) full fusion — weighted, precision 1.00 at recall 0.75 (F1 0.86,
  FPR 0.00, accuracy 0.88, n=96)." (Untrained-fallback smoke run — see below;
  NOT detection evidence.)
- Thresholds are fitted on the VALIDATION slice only (24 emails: 12 phishing
  / 12 benign), maximising precision subject to a recall floor (default
  0.80); they are never fitted on test. The placeholder-0.5 metrics are
  reported alongside for reference. When no validation rows exist (tiny
  `--max_samples` run) the placeholder 0.5 is used and the report says so.

## Changes

### E1 — `scripts/make_multimodal_eval_set.py` (new) + `data/multimodal_eval/` — Done

- Decision: deterministic (seeded, `--seed` default 42) generator writing
  120 `.eml` files (60 phishing / 60 benign) under `data/multimodal_eval/`
  (small text+image files, kept in the repo next to the existing `samples/`
  tree), plus `labels.csv` (`filename,label,category,split`) and a
  `manifest.json` (seed, per-category counts, split sizes, synthetic caveat,
  threshold note). `--debug` writes a 12-file balanced subset (6/6).
- Categories (phishing 60): `image_only_phishing` 10, `text_button_image` 8,
  `qr_body_image` 8, `qr_in_text_pdf` 7, `scanned_pdf` 7,
  `html_link_mismatch` 7, `html_attachment_phishing` 6, `fake_invoice_pdf` 7.
  Benign look-alikes (60): `newsletter_logo_qr` 20 (logo/banner image + QR
  for a benign event page), `real_invoice` 20, `meeting_agenda` 20.
- Split: stratified deterministic (separate `seed+1` RNG stream), 20% of each
  label to validation (12+12 = 24) and the rest to test (48+48 = 96).
- Content: template slots (company, invoice number, amount, sender, URL)
  drawn from a seeded RNG; phishing headers carry mismatched Reply-To and
  usually failing SPF/DKIM/DMARC (60% fail) so Section D flag scoring has
  something to see; benign headers are consistent with SPF/DKIM/DMARC pass.
  Message-IDs and MIME boundaries are deterministic per file.
- OCR stressors (seeded, each fires only sometimes so the set mixes clean
  and stressed images): font family/size variation (DejaVu Sans/Serif/Mono),
  low contrast, small rotations (-4..+4 deg), sparse pixel noise, JPEG
  compression (quality 35-70) on some images.
- Generation reuses the Section A sample patterns (`render_text_png`,
  text/scanned PDFs via PyMuPDF, QR via `qrcode`) with seeded variation;
  `scripts/make_multimodal_samples.py` was NOT modified.
- Determinism limit (documented in the module docstring and here): the same
  seed gives identical `labels.csv` content, file list and manifest, but
  `.eml` BYTE equality is not guaranteed — PyMuPDF embeds a fresh document
  ID in every PDF it writes. Tests assert on `labels.csv` + file list, not
  raw bytes. (Verified empirically: two identical `fitz` writes differ.)
- Status: Done. Verified by extraction spot-check on the debug subset:
  QR-in-text-PDF decoded, HTML anchor/href mismatch flagged, scanned-PDF
  OCR text recovered, image OCR recovered, benign QR decoded as benign URL.

### E2 — `scripts/run_multimodal_eval.py` (new) — Done

- Decision: evaluates the Section D pipeline on the eval set and writes
  `results/multimodal_eval/multimodal_eval.json`,
  `multimodal_eval.md`, `confusion_matrix_weighted.png` and
  `ablation_precision_recall_fpr.png` (matplotlib Agg; no seaborn).
- Ablations: (a) text-body only, (b) text + OCR, (c) text + OCR + QR + PDF
  (each = RoBERTa P(phishing) on the filtered concatenated context, built
  with the same `build_concatenated_context` Section D uses, so an ablation
  context is exactly the subset of the full context its modalities
  contribute), (d) full fusion via `fuse_email`, reporting BOTH `weighted`
  and `max` for every email. `--fusion` selects the primary verdict only.
- Scoring reuses the predict script's path (`load_trained_roberta`,
  `score_text` from `scripts/run_multimodal_predict.py`); each non-empty
  part is scored once, then the (a)-(c) contexts and the full context.
- Metrics (pure-Python helpers in the script, unit-tested): accuracy,
  precision, recall, F1, FPR, confusion matrix `[[tn, fp], [fn, tp]]`,
  ROC-AUC (Mann-Whitney; `None` when one class is present), per-modality
  recall (phishing emails actually carrying that modality, weighted
  fusion), and a failure table (filename, true/predicted label, all five
  scores, category, reason built from the fusion's top reasons + the
  decisive evidence, e.g. "fused score reached threshold but lacked
  corroboration").
- Threshold fitting (`fit_threshold_precision_first`): scans observed scores
  + a 0.05 grid on VALIDATION only, picks max precision with recall >=
  floor (ties: higher recall, then F1, then lower threshold). If no
  candidate meets the floor, `floor_met: false` is reported with the
  best-recall threshold. Note found while testing: with at least one true
  positive the floor is always reachable (threshold at the lowest score
  predicts everything positive, recall 1.0), so the unmet branch only
  fires for a degenerate no-positive validation slice — the test pins
  that case.
- Cheap baselines (Section H hook, `--models`): `tfidf_lr` and `bilstm`
  are fitted on the REAL-data train split (`results/splits.json`,
  identical splits/cleaning as Section H) and evaluated on the SAME
  extractor output via `baselines.evaluate_on_extractor_output` plus the
  per-modality `baselines.evaluate_multimodal` table. They are NEVER fitted
  on the synthetic set (no train split there; fitting on the evaluated set
  would be leakage). Missing CSV/splits manifest -> NOT RUN YET with the
  reason. `bilstm` uses `--bilstm_epochs` (default 3, not Section H's 8,
  to keep the eval script usable on CPU; documented). Default
  `--models` is `roberta` only.
- Fallback behaviour chosen: run end-to-end with `--allow_pretrained_fallback`
  (same as the predict script) — LOUD banner, `"untrained_fallback": true`
  in the JSON, bold UNTRAINED warning at the top of the .md, and every
  number labelled NOT detection evidence. Without the flag and without a
  checkpoint it refuses cleanly (exit 2, clear message). Kaggle
  auto-detection mirrors the predict script (`/kaggle/input/*/best_model.pt`,
  outputs under `/kaggle/working/results/multimodal_eval`).
- CLI: `--eval_dir`, `--labels`, `--checkpoint`, `--output_dir`,
  `--max_samples`, `--debug`, `--fusion`, `--models tfidf_lr bilstm roberta`,
  `--allow_pretrained_fallback`, `--recall_floor` (default 0.80),
  `--bilstm_epochs` (default 3).
- Status: Done (code + smoke runs below). One post-run merge: the delivered
  `results/multimodal_eval/` combines the RoBERTa run (`--models roberta`)
  with a separate TF-IDF run (`--models tfidf_lr`, same 120 files, same
  extractor output) — the `baselines` block from the TF-IDF run's JSON was
  merged into the main JSON and the .md re-rendered with the script's own
  `render_markdown`; `metadata.models` is `["roberta", "tfidf_lr"]` and a
  `baseline_note` records the merge. A single combined invocation
  (`--models roberta tfidf_lr`) would re-score RoBERTa from scratch and
  produces the same file directly on Kaggle.

### E3 — `scripts/run_multimodal_faithfulness.py` — Done (added what was missing)

- What Section A already had: `untrained_fallback` per row and the
  `--allow_pretrained_fallback` flag — verified present, kept as-is.
- What was missing and is now added: the RANDOM baseline —
  `evaluate_random_baseline` (same k=5, 5 seeds) is now computed per sample
  and reported as `random_comprehensiveness_k5` /
  `random_sufficiency_k5` in the JSON rows and as extra columns in the
  printed table, so LITA faithfulness can be compared against dropping /
  keeping k random words. Also added an empty-extraction guard: a sample
  with no scorable text parts now records a row with
  `status: "no_text_parts"` and null metrics instead of crashing on
  `max()` of an empty list. Normal rows carry `status: "ok"`.
- Status: Done (code + wiring test; end-to-end smoke run below).

### E4 — Synthetic-data documentation — Done

- The caveat is written in three places: the `manifest.json`
  `synthetic_caveat`, the JSON `synthetic_warning` metadata, and the top of
  `multimodal_eval.md` (plus a "What this does and does not show" section
  noting the 24-email validation slice is coarse and thresholds must be
  re-fitted on a larger real validation set before any deployment claim).
  Wording used everywhere: generated images/PDFs from a small set of
  templates, simulated OCR stressors, no real-world adversary variation,
  no real emails — results measure pipeline behaviour on these templates
  only and do NOT generalise to real phishing.

## Test results

- `.venv/bin/python -m pytest tests/test_multimodal_eval.py -q` —
  **20 passed** (2026-11-19). Covers: full-set balance 60/60 + exact
  per-category counts + labels.csv schema + 24 validation / 96 test split
  (12+12 stratified); determinism (same seed -> identical labels.csv,
  file list and manifest; different seed -> labels.csv differs); debug
  subset 12 files 6/6; `compute_metrics` on a hand-computed case
  (TP 3 / FP 1 / FN 1 / TN 2) and edge cases (zero predicted positives ->
  precision 0.0; no true positives -> recall 0.0, ROC-AUC None; empty
  input raises); `roc_auc` known values (1.0 / 0.0 / 0.5 / None);
  threshold fitting (floor met -> precision 1.0 at threshold 0.6; floor
  unmet -> `floor_met: false`); failure-table builder (only errors listed,
  correct labels/scores/reasons; empty when all correct);
  per-modality recall (counts only carriers; benign never counted);
  ablation labelling (a)-(d) keys + labels; precision-first summary line
  names the best-precision ablation with its recall; `.md` rendering
  contains the synthetic + untrained warnings and all (a)-(d) rows;
  `modality_texts` / `filtered_extraction` on a fake extraction;
  faithfulness-script wiring (random baseline import + call + output keys,
  `untrained_fallback` flag, `no_text_parts` guard).
  Two failures found and fixed in the TESTS (not the code): the
  floor-not-met case used a set where the floor was in fact reachable
  (see E2 note), and the markdown fixture lacked `confusion_matrix`.
  Re-run: 20/20.
- Neighbour suites re-run after the change (same day):
  `.venv/bin/python -m pytest tests/test_multimodal_eval.py
  tests/test_multimodal_fusion.py tests/test_url_risk.py -q` —
  **96 passed** (20 + 37 + 39). Section D's 37 fusion tests are untouched
  and still pass.
- `model.py` and `faithfulness.py` untouched (verified via `git status` /
  `git diff --stat`: no output for either file).

## Smoke tests executed (untrained fallback — NOT detection evidence)

All runs used `--allow_pretrained_fallback` because
`results/best_model.pt` is absent locally. Every number below is
pipeline-demonstration output from untrained base `roberta-base`
(P(phish) ~0.43-0.44 on every part/context) plus static URL/flag
evidence. Do NOT cite any of it as phishing-detection evidence.

1. Generator debug:
   `.venv/bin/python scripts/make_multimodal_eval_set.py --debug --out_dir /tmp/eval_debug`
   Exit 0. 12 files (6/6), manifest + labels.csv written.
2. Generator full:
   `.venv/bin/python scripts/make_multimodal_eval_set.py`
   Exit 0. 120 files (60/60; 24 validation / 96 test) in
   `data/multimodal_eval/`; per-category counts verified against
   CATEGORIES (see E1 above).
3. Eval script, debug set (12 emails):
   `.venv/bin/python scripts/run_multimodal_eval.py --eval_dir /tmp/eval_debug
   --allow_pretrained_fallback --output_dir /tmp/smoke_E --debug`
   Exit 0 (~62 s incl. model load). Precision-first line: "(d) full fusion
   — weighted, precision 0.71 at recall 1.00 (F1 0.83, FPR 0.40, accuracy
   0.80, n=10)". JSON + .md + both plots written to `/tmp/smoke_E/`.
4. Eval script, FULL set (120 emails, RoBERTa):
   `.venv/bin/python scripts/run_multimodal_eval.py --allow_pretrained_fallback
   --output_dir results/multimodal_eval`
   Exit 0 (~6.5 min). Precision-first line: **"Best precision at recall
   floor 0.80: (d) full fusion — weighted, precision 1.00 at recall 0.75
   (F1 0.86, FPR 0.00, accuracy 0.88, n=96)."** Test slice (fitted
   thresholds): (a) precision 0.50 / FPR 1.00 / recall 1.00 (threshold
   0.05 — the untrained context scores cluster at chance, so the
   validation fit collapses to "predict everything phishing");
   (b) same 0.50/1.00/1.00; (c) same 0.50/1.00/1.00; (d) weighted
   precision 1.00 / FPR 0.00 / recall 0.75 / F1 0.86, threshold 0.34,
   confusion `[[48, 0], [12, 36]]` — the weighted fusion's separation
   here comes from the static URL/flag channels, NOT the model, since the
   model scores everything at chance; (d) max precision 0.56 / FPR 0.52 /
   recall 0.67, confusion `[[23, 25], [16, 32]]` — the recall-heavy
   baseline is precision-poor exactly as Section D predicted. At the
   placeholder 0.5 every ablation predicts nothing (all scores < 0.5).
   Per-modality recall (weighted, test phishing): body 36/48 (0.75),
   ocr 11/19 (0.58), pdf 11/14 (0.79), qr 12/14 (0.86). Failures: 14
   (all false negatives on the full set; 0 false positives at the fitted
   weighted threshold). Outputs in `results/multimodal_eval/` (JSON, .md,
   both plots — verified openable with Pillow).
5. Cheap baseline TF-IDF+LR (same extractor output, full 120):
   `.venv/bin/python scripts/run_multimodal_eval.py --models tfidf_lr
   --output_dir /tmp/smoke_E_tfidf`
   Exit 0 (~85 s). TF-IDF+LR fitted on the real-data train split
   (11,215 rows, splits validated exactly) and evaluated on the synthetic
   extractor text: **precision 0.95, recall 0.95, F1 0.95, accuracy 0.95,
   ROC-AUC 0.9875, FPR 0.05, confusion `[[57, 3], [3, 57]]`, n=120** at
   threshold 0.5. Per-modality F1: all 0.95, body 0.79, ocr 0.83, pdf
   0.79, qr 0.73. Merged into the delivered `results/multimodal_eval/`
   (see E2 merge note). BiLSTM was NOT run locally (CPU cost); it is a
   `--models bilstm` run for Kaggle.
6. Faithfulness script smoke (new random-baseline path):
   `.venv/bin/python scripts/run_multimodal_faithfulness.py
   --allow_pretrained_fallback --lime_samples 10
   --output_json /tmp/smoke_E_faith.json`
   Exit 0 (~74 s). All 6 samples reported with LITA comp/suff AND the new
   random-baseline columns (e.g. sample1: comp -0.0031 vs random -0.0031;
   sample4: suff -0.0018 vs random -0.0003). LITA ≈ random here is the
   expected untrained-model artefact (chance-level probabilities), not a
   finding. The printed table now has the Random Comp/Suff columns and the
   UNTRAINED warning footer; `/tmp/smoke_E_faith.json` holds the rows.

## Exact commands (re-run / Kaggle with trained checkpoint)

```bash
# Regenerate the eval set (seeded, deterministic):
.venv/bin/python scripts/make_multimodal_eval_set.py [--seed 42] [--debug]
# Full eval with the trained checkpoint (Kaggle GPU):
.venv/bin/python scripts/run_multimodal_eval.py --checkpoint results/best_model.pt
# Full eval + cheap baselines in one invocation (Kaggle):
.venv/bin/python scripts/run_multimodal_eval.py --models roberta tfidf_lr bilstm
# Cheap baselines only (no checkpoint needed):
.venv/bin/python scripts/run_multimodal_eval.py --models tfidf_lr [--models bilstm]
# Tests:
.venv/bin/python -m pytest tests/test_multimodal_eval.py -q
```

## Files created / changed

| File | Change |
|---|---|
| `scripts/make_multimodal_eval_set.py` | NEW (~430 lines): generator, E1 |
| `scripts/run_multimodal_eval.py` | NEW (~700 lines): eval script, E2 |
| `tests/test_multimodal_eval.py` | NEW (20 tests) |
| `scripts/run_multimodal_faithfulness.py` | EDIT: random baseline per row + empty-parts guard (E3) |
| `data/multimodal_eval/` | NEW: 120 .eml + labels.csv + manifest.json (E1) |
| `results/multimodal_eval/` | NEW: multimodal_eval.json/.md + 2 plots (E2 smoke outputs) |

## Only verifiable on Kaggle (trained checkpoint / GPU)

- Everything about real detection: threshold fitting on validation with a
  trained model (the local fitted thresholds are artefacts of a
  chance-level untrained model — (a)-(c) collapse to 0.05), precision /
  recall / FPR for weighted vs max, whether the corroboration gate costs
  too much recall, and per-modality recall with real scores.
- BiLSTM baseline on the same extractor output (`--models bilstm`).
- End-to-end faithfulness with the trained checkpoint (LITA + random
  baseline on real scores) if the local smoke run did not complete.
- Kaggle path auto-detection (`/kaggle/input/*/best_model.pt`,
  `/kaggle/working/results/multimodal_eval`) — code mirrors the predict
  script and is untested on this VM, which has no `/kaggle` directory.
