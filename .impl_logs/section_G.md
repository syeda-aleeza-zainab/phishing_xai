# Section G — Documentation + integration fix — change log

Date: 2026-11-19. Base commit: `5d86e95` (fresh clone, working copy shared with sections A–F/H).
Did NOT touch: `src/phishing_xai/model.py`, `src/phishing_xai/faithfulness.py`
(verified: `git status`/`git diff --stat` empty for both). No push, no retrain,
no fabricated metrics — anything not executed is "NOT RUN YET". No `.venv`
content in any output.

## Sources read (all, before writing)

- `.impl_logs/section_B.md`, `section_C.md`, `section_D.md`, `section_E.md`,
  `section_F.md`, `section_H.md`, `.impl_logs/integration.md`; Section A entries
  in `decision.md` §2 and `docs/PROJECT_LOG.md` §9.
- `results/splits.json` (17,513 retained; 11,215/2,803/3,495; +2,863 → 14,078),
  `results/multimodal_eval/multimodal_eval.md` + `.json`,
  `results/RESULTS.md`, `results/summary.json`.
- `README.md`, `docs/MULTIMODAL_NOTE.md`, `decision.md`, `docs/PROJECT_LOG.md`
  (read fully).
- Verified present: `notebooks/kaggle_run.ipynb` (23 cells), `requirements-kaggle.txt`,
  `src/phishing_xai/url_risk.py`, `src/phishing_xai/multimodal.py`,
  `src/phishing_xai/baselines.py`, `scripts/make_multimodal_eval_set.py`,
  `scripts/run_multimodal_eval.py`, `scripts/run_baselines.py`,
  `scripts/run_complexity_report.py`, `data/multimodal_eval/` (120 `.eml` +
  `labels.csv` + `manifest.json`, counts verified).
- Verified absent: `results/baselines/baseline_results.json` and any
  `results/complexity_report.*` (Section H only ran `--debug` smoke runs under
  `/tmp`; no committed baseline or complexity outputs exist in the repo).

## Changes by item

### G1 — README.md rewritten — Done
- Kaggle quickstart is now the primary path (GPU T4 + Internet ON, repo attached
  as a dataset, `notebooks/kaggle_run.ipynb` cells in order, checkpoint-as-dataset
  options, Save Version). The Colab notebook file is kept and named as no longer
  primary. Local CPU smoke-test pipeline kept.
- Multimodal section rewritten to the real pipeline: extract → url_risk → fusion,
  `--fusion max|weighted`, default weighted (0.45/0.20/0.20/0.15 + corroboration
  gate), threshold 0.5 placeholder to be tuned on validation only; the old
  max-rule-as-verdict wording removed.
- Test count corrected to the end-of-work recount (193). Status line now records
  precision-first, the `splits.json` counts, and the 2 pre-existing env failures.
- Limitations rewritten (precision-first, untrained-fallback labelling, synthetic
  eval set, NOT RUN YET items).

### G2 — docs/MULTIMODAL_NOTE.md rewritten — Done
- New mermaid architecture diagram (extract → url_risk → fusion → verdict/LITA
  overlay, with the eval-set and baselines branches).
- Module descriptions updated to the B/C/D state (6-key extraction schema,
  url_risk, weighted fusion, random-baseline faithfulness).
- Results section filled ONLY from real runs: §3.1 Section E smoke eval quoted
  only as untrained fallback (weighted precision 1.00 at recall 0.75, n=96, with
  the explicit note that separation comes from URL/flag channels, not the model);
  §3.2 TF-IDF+LR quoted as a baseline fitted on the real-data train split and
  evaluated on 120 synthetic mails (precision/recall/F1 0.95, ROC-AUC 0.9875);
  §3.3 lists everything else as NOT RUN YET. The six-sample table is kept and
  labelled prominently as untrained-fallback pipeline demonstration, with the
  LITA ≈ random artefact explained.
- Honest limitations (§4, 8 items) including the threat-model cross-link.

### G3 — decision.md + docs/PROJECT_LOG.md merged — Done
- `decision.md`: new §4 "Sections B–H" with per-section decision/reason/status
  tables (B1–B8, C, D incl. precision-first + threshold placeholder + the PDF
  `qr_urls` bug fix, E incl. smoke results labelled untrained + TF-IDF numbers,
  F incl. the baseline-overwrite finding, H incl. verdict rule + H5
  `evaluate_faithfulness_single_method` note + RoBERTa/DistilBERT gaps, G);
  old §3 renumbered §5; Max-Rule row annotated as superseded by D; Tesseract
  row annotated as a historical path. Pre-existing test failures recorded with
  causes. The 2026-11-19 precision-first user decision, the LITA A1 inversion
  (already prominent in §2), and the A2 untrained-fallback flag are all carried
  in the merged record.
- `docs/PROJECT_LOG.md`: new §10 with the same B–H + G merge in table form;
  §4 header annotated that the `file:///c:/Users/admin/...` links are historical
  records from the original author's Windows machine; §8 Tesseract row and the
  max-rule row annotated (superseded / Kaggle now primary).
- No wrong numbers reintroduced: swept README/docs/decision.md for
  17,532 / 11,220-2,806-3,506 / SHA-1 / 100,000 — only Section A's own historical
  correction mentions remain. The 796 near-dup-groups figure is mentioned only
  as unverified, never asserted.

### G4 — "Threat model & limitations" — Done
- Added to the README (most visible place) and cross-linked from
  `docs/MULTIMODAL_NOTE.md` (§4 item 2) and `decision.md` (§4, G4 row):
  OCR failure on adversarial images, text-only RoBERTa cannot see logos/layout
  spoofing, synthetic eval-set limits, dataset age (Enron-style), leakage notes
  (MD5 of first 200 alphanumerics; 796-groups figure unverified), latency with
  no measured numbers yet (NOT RUN YET), no URL fetching (static only),
  precision/recall trade-off.

### G5 — INTEGRATION FIX: run_baselines.py merges baseline_results.json — Done (fix, not just documented)
- `merge_baseline_results(existing, new)` added to `scripts/run_baselines.py`:
  loads the existing JSON when present, merges per-model results keyed by
  model+seed (new per-seed entry replaces the same model+seed, other seeds and
  untouched models kept), recomputes each aggregate from the merged seeds via
  `aggregate_seed_results`, keeps real `comparisons_vs_roberta` / error-analysis
  / faithfulness / multimodal blocks when the new run has only placeholders,
  unions fairness seed lists, marks `merged_with_previous_run: true`.
- `main()` calls it by default before saving; `--no_merge` opts out for a clean
  overwrite; a missing/unreadable/corrupt previous file falls back to writing
  the new run only (with a warning).
- 2 new tests in `tests/test_baselines.py` (36 there now): same-seed replacement
  + untouched-model retention + aggregate recomputation, and no-previous-file
  passthrough. `pytest tests/test_baselines.py -q`: **36 passed**.
- Not yet exercised end to end: accumulating a real ladder across several Kaggle
  invocations (only unit tests here) — verify on Kaggle.

### G6 — Final recount + verification — Done
- `grep -r "def test_" tests/ | wc -l` at the end of the work: **193**
  (36 baselines, 12 data/augment, 7 explain/LITA, 7 explain/resumability,
  20 extract, 5 faithfulness/ablation, 4 model/train, 20 multimodal eval,
  37 multimodal fusion, 5 report, 1 sanity, 39 url_risk).
- Full suite `.venv/bin/python -m pytest -q`: **191 passed, 2 failed in 80.45 s**
  (2026-11-19). The 2 failures are exactly the pre-existing environment ones:
  `tests/test_sanity.py::test_environment_sanity` (asserts Python 3.10; venv is
  Python 3.12.3) and
  `tests/test_explain_and_resumability.py::test_captum_vs_transformers_interpret_tiny_model`
  (`transformers_interpret` not installed locally). No NEW failures; the third
  failure seen once in Section H's full run
  (`test_url_score_contribution_benign_vs_malicious`) did not reproduce here.
- Absolute/Windows paths: `grep` for `C:\` and `/home/` in README.md,
  docs/MULTIMODAL_NOTE.md, decision.md — README and MULTIMODAL_NOTE have none;
  decision.md has two `C:\Users\admin...` mentions, both explicitly marked as
  historical records (B7 row + Step 1 Tesseract row). `docs/PROJECT_LOG.md`
  still contains the historical `file:///c:/Users/admin/...` links in §4 and
  one `C:\Users\admin` path in §8; both are now annotated as historical records
  of the original author's Windows machine rather than presented as current
  paths (rewriting the log's history was not done). The generated
  results/multimodal_eval/multimodal_eval.md contained an absolute home-directory path (fixed to relative)
  absolute paths from the local smoke run; it is generated output, not a doc
  touched here — regenerating it on Kaggle will replace those paths.
- `model.py` / `faithfulness.py` untouched (verified via git status/diff).

## Files created / changed by this section

| File | Change |
|---|---|
| `README.md` | Rewritten (Kaggle primary, multimodal pipeline, threat model, 193 tests) |
| `docs/MULTIMODAL_NOTE.md` | Rewritten (mermaid diagram, results only from real runs, limitations) |
| `decision.md` | New §4 (B–H + G merge), §3→§5 renumber, superseded/historical annotations |
| `docs/PROJECT_LOG.md` | New §10 (B–H + G merge), historical-path annotations in §4/§8 |
| `scripts/run_baselines.py` | `merge_baseline_results` + `--no_merge` + merge call in `main()` |
| `tests/test_baselines.py` | 2 new merge tests (34 → 36) + `run_baselines` import |
| `.impl_logs/section_G.md` | This file |

Other `M`/`??` entries in `git status` belong to sections A–F/H and the
coordinator's working-tree state, not to Section G.

## Only verifiable on Kaggle (trained checkpoint / GPU)

- Kaggle execution of `notebooks/kaggle_run.ipynb` end to end (the notebook is
  validated, flag-checked and dry-run only; nothing in it has run on Kaggle).
- Threshold tuning on validation with the trained checkpoint; real precision /
  recall / FPR for weighted vs max fusion; whether the corroboration gate costs
  too much recall; end-to-end LITA overlays for a PDF/QR winning part.
- The full baseline ladder (3 seeds) and the complexity-report verdict; the
  `run_baselines.py` merge accumulating a real ladder across several separate
  invocations (unit-tested only here).
- BiLSTM baseline on the synthetic eval set; faithfulness with the trained
  checkpoint; Kaggle path auto-detection (`/kaggle/input/*/best_model.pt`).
- The 2 pre-existing test failures: Python 3.10 assertion (Kaggle env version
  to be confirmed) and `transformers_interpret` under transformers 4.x.

## Doc claims backed by real executed results vs NOT RUN YET

Backed by executed runs (cite with their labels):
- Dataset hygiene + splits: 17,513 retained, 11,215/2,803/3,495, +2,863 → 14,078
  (`results/splits.json`; executed by `make_splits.py`).
- Section B extraction behaviour: 20 `test_extract.py` tests pass; 6 samples
  extracted on this VM (Section B log).
- Section C url_risk: 39 tests pass; calibration values in the C log.
- Section D fusion: 37 tests pass (incl. exact 0.66 weighted arithmetic and the
  weighted-vs-max divergence); 6-sample smoke run, all untrained fallback.
- Section E synthetic eval (UNTRAINED FALLBACK, not detection evidence): (d)
  weighted precision 1.00 / recall 0.75 / F1 0.86 / FPR 0.00, n=96;
  (a)–(c) precision 0.50 / FPR 1.00 at threshold 0.05; max precision 0.56 /
  FPR 0.52; per-modality recall body 0.75 / OCR 0.58 / PDF 0.79 / QR 0.86.
- TF-IDF+LR on the synthetic set (fitted on real train split, n=120):
  precision/recall/F1/accuracy 0.95, ROC-AUC 0.9875, FPR 0.05, `[[57, 3],
  [3, 57]]`; per-modality F1 all 0.95, body 0.79, OCR 0.83, PDF 0.79, QR 0.73.
- Section F packaging: notebook nbformat-valid (23 cells), 15 script paths all
  present, both requirements files parse and dry-run exit 0, pyproject/
  requirements cross-check exact.
- Section H: 36 `test_baselines.py` tests pass (incl. 2 new G merge tests);
  `--debug` smoke runs only (subset numbers, never quoted as results).
- Full suite: 191 passed, 2 failed (the 2 pre-existing env failures above).

NOT RUN YET (must not be cited as results):
- Production tuning, training, test evaluation, explainability/faithfulness,
  alpha ablation (`results/RESULTS.md` §§3–5).
- Fusion threshold tuning (0.5 is a placeholder); any trained-checkpoint
  precision/recall/FPR for weighted vs max; corroboration-gate recall cost.
- Full baseline ladder (all 7 models, 3 seeds), McNemar vs RoBERTa, bootstrap
  precision CIs, error analysis, robustness, baseline faithfulness at n=100,
  complexity-report verdict, recommended deployment model.
- BiLSTM on the synthetic eval set; trained-checkpoint faithfulness and
  PDF/QR LITA overlays end to end; latency (any number); GPU latency.
- Kaggle notebook execution; Kaggle path auto-detection; multi-invocation
  baseline merge on real runs.
