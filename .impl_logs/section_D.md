# Section D — Multimodal fusion + verdict (multimodal.py, predict script) — change log

Date: 2026-11-19. Scope: `src/phishing_xai/multimodal.py` (new),
`src/phishing_xai/config.py` (added `FusionConfig` only),
`scripts/run_multimodal_predict.py` (rewritten to use fusion),
`tests/test_multimodal_fusion.py` (new).
Did NOT touch: `src/phishing_xai/model.py`, `src/phishing_xai/faithfulness.py`,
`decision.md`, `docs/PROJECT_LOG.md`. No push, no retrain, no fabricated
metrics. No URL in any email/QR/PDF was fetched (static analysis only).
No `.venv` content is included in any output.

## Precision-first design (user priority 2026-11-19: "mera priority focus precision hai")

- Primary metric is phishing-class PRECISION (minimise false positives / FPR).
  F1 / accuracy / recall are secondary: they are reported, never the sole
  selection criterion.
- Weighted fusion (default) gives the concatenated-context score the largest
  weight (0.45) and a single part only 0.20, so one weak part scoring high on
  its own cannot flip the verdict. A phishing verdict needs corroboration
  across channels (context + a part, context + URL, part + URL + flags, ...).
- Corroboration gate (weighted only): besides `final_score >= threshold`,
  at least two of the three channels (context, max part, URL) must be
  >= 0.5, OR one channel plus a flag score >= 0.30, OR the fused score must
  reach the high-confidence threshold 0.75. Otherwise the verdict stays Safe
  and a reason is recorded saying corroboration was lacking.
- Threshold: 0.5 is a PLACEHOLDER. It must be tuned on VALIDATION only
  (never test) on Kaggle with the trained checkpoint, choosing the threshold
  that maximises phishing-class precision subject to a recall floor — never
  by maximising F1 alone. Until that validation run exists, no claim that
  the threshold was tuned is made anywhere (code, docstrings, CLI help,
  JSON `threshold_note`, or this log).
- Max rule is kept only as a selectable baseline (`--fusion max`): it flips
  on ANY part >= threshold and ignores URL/flag/context evidence, so it is
  recall-heavy / precision-poor by construction. Both fusions are computed
  and stored for every email so Section E can compare precision / recall /
  FPR side by side.

## Changes

### D1 — `src/phishing_xai/multimodal.py` (new) + `FusionConfig` in `config.py` — Done

- Decision: fusion weights live in a `FusionConfig` dataclass in
  `config.py` (alongside the other configs), not in `multimodal.py` —
  re-exported from `multimodal.py` so Section E can import everything from
  one place.
- Default weights (sum to 1.0): `w_context = 0.45` (concatenated-context
  RoBERTa score), `w_max_part = 0.20` (single best part, deliberately
  small), `w_url = 0.20` (worst static URL score, Section C),
  `w_flags = 0.15` (header/structural flag score). `threshold = 0.5`
  (PLACEHOLDER, see above), `high_confidence_threshold = 0.75`,
  `max_length = 512`.
- `final = w_context*context + w_max_part*max_part + w_url*url + w_flags*flags`.
  Worked check: context 0.80, max part 0.60, URL 0.40, flags 0.50 with
  weights 0.5/0.25/0.15/0.10 gives exactly 0.66 (covered by test).
- `build_concatenated_context(extraction, *, tokenizer=None, max_length=512)`:
  one string with sections in order `[SUBJECT] [BODY] [HIDDEN] [OCR] [PDF]
  [ATTACHMENT] [QR_URL]` (empty sections omitted; `body_text` used for
  `[BODY]` when no `body_text` part carries text; QR payloads appended to
  `[QR_URL]`). Hidden text gets its own section, never mixed into `[BODY]`.
  Truncation uses the existing `truncate_text_to_model_window` when a
  tokenizer is passed; without one it caps at `max_length * 4` chars with a
  warning (documented approximation).
- `score_urls_for_email(extraction, *, url_config=None)`: collects URLs from
  `links`, part `qr_urls`, and `extract_urls` over body/part text;
  de-duplicated (a URL seen in text and QR is scored once, as QR). Returns
  `{"urls", "url_score" (worst/max), "max_url", "n_urls", "reasons"}`.
  BUG FOUND AND FIXED IN THIS SECTION: Section B stores PDF hyperlinks in
  the `qr_urls` slot of `pdf_text` / `pdf_ocr` parts, so the first version
  scored `https://billing-example.com/pay` (sample4) as QR-only (+0.10).
  Only `qr` / `image` / `pdf_image` parts now count as QR sources; PDF
  links are scored with `in_qr=False`. Fixed in the module (not by editing
  `extract.py`), covered by a dedicated test.
- `score_flags(extraction)`: flag contributions — `anchor_href_mismatch`
  0.15, `double_extension` 0.20, `suspicious_zip_member` 0.15,
  `hidden_text` 0.10, `form_action` 0.05, `missing_cid_image` 0.02, unknown
  flag type 0.05; each failed SPF / DKIM / DMARC verdict +0.10; Reply-To
  domain != From domain +0.10. `flag_score` = sum clamped to [0, 1].
  Returns `{"flag_score", "contributions", "reasons"}`.
- `fuse_email(extraction, part_scores, *, fusion="weighted", config=None,
  concatenated_score=None, threshold=None, url_config=None, tokenizer=None,
  max_length=None)`: see "Exact public API" below. `part_scores` keys are
  matched per part in order: canonical part id
  `"<kind>[:<filename>][#p<page>]"` (see `part_id`), then bare `kind`, then
  part index as a string. When `concatenated_score` is `None` it falls back
  to the max part score, recorded as
  `context_score_source == "fallback_max_part"`.
- Most-contributing modality: per-channel weighted contributions; the
  context contribution is credited to the winning part's modality (the
  context is built from the same parts — documented as an attribution
  heuristic, not a per-section model score). Labels: `body`, `ocr`, `qr`,
  `pdf`, `attachment`, `url`, `headers`, `none`.
- Status: Done. Note: my first edit to `config.py` accidentally dropped
  the `XAIConfig` class header (the old text I replaced included it);
  caught by the test run (`ImportError: cannot import name 'XAIConfig'`)
  and restored verbatim the same day; verified `XAIConfig().alpha == 0.5`
  and the neighbouring suites still pass (see test results).

### D2 — Per-email output — Done

- Every `fuse_email` result carries: `fusion`, `verdict`
  (`"Phishing Email"` / `"Safe Email"`), `final_class`, `final_score`,
  `threshold`, `corroborated`, `context_score` + `context_score_source`,
  `max_part_score`, `winning_part` / `winning_part_kind`, `url_score` /
  `url_details`, `flag_score` / `flag_details`, resolved `part_scores`,
  `contributions` (per-channel), `most_contributing_modality`, `reasons`
  and `top_reasons` (first 5, never empty), `concatenated_context`.
- Reasons are human-readable, e.g. "Highest-scoring part 'qr:auth_qr.png'
  (kind qr) scored P(phishing)=0.44.", "Riskiest URL
  'http://secure-login-example.com/qr-verify' scored 0.20: plain http, not
  https (no transport encryption).", plus one line per flag contribution.
- The predict script stores both `fusion_weighted` and `fusion_max` in
  every output row plus the primary `final_verdict` / `final_score` /
  `most_contributing_modality` / `top_reasons` at the top level.

### D3 — LITA on concatenated text + overlays — Done

- LITA now runs on the concatenated context (not just the winning part)
  when `--explain` is given; top-5 words printed and stored as before.
  `draw_lita_boxes_on_image` is kept as-is and still used for image parts.
- New `overlay_scanned_pdf_page(pdf_bytes, page, ocr_boxes, lita_top_words,
  output_path, *, dpi=200, top_k=5)`: renders the page with PyMuPDF at the
  same DPI the extractor OCR'd at (200 by default, so box coordinates line
  up) and draws the LITA boxes via `draw_lita_boxes_on_image`. Returns the
  path or `None` with a warning when PyMuPDF/Pillow is missing, the page is
  out of range, or rendering fails — never raises.
- New `overlay_qr_image(image_input, qr_urls, output_path, *,
  qr_boxes=None, top_k=5)`: draws the decoded-QR region from pyzbar's
  polygon/rect when pyzbar is available (red border otherwise), draws any
  supplied `qr_boxes`, and annotates the decoded URL(s) as text at the top
  (white strip behind the label). Returns the path or `None` with a warning
  when Pillow is missing or drawing fails — never raises. The URL is drawn
  as text, never fetched.
- Status: Done in code and unit tests. End-to-end overlay in the predict
  script is wired for winning kinds `image` / `pdf_ocr` / `pdf_text` /
  `pdf_image` / `qr`; in the smoke run the winning part was the subject on
  most samples (untrained model scores ~0.44 everywhere), so no overlay
  file was produced there — the helpers themselves are covered by direct
  tests (real PyMuPDF render + real PIL draw). The pipeline branch for a
  PDF/QR winning part still needs a trained-checkpoint run to be observed
  end to end (see "Needs Kaggle").

### D4 — `scripts/run_multimodal_predict.py` updated — Done

- Uses `multimodal.py` for context building, URL/flag scoring and fusion;
  per-part scoring, `load_trained_roberta`, `score_text` and
  `explain_part_lita` are unchanged (same names/signatures, so
  `run_multimodal_faithfulness.py` still imports from here).
- CLI (all old flags still work): `--checkpoint`, `--output_dir`,
  `--max_samples`, plus `--fusion max|weighted` (default `weighted`),
  `--threshold` (default `None` = 0.5 placeholder, documented as such in
  help text and in each JSON row's `threshold_note`), `--explain`,
  `--lime_samples`, `--output_json`, `--allow_pretrained_fallback`,
  `--debug` (verbose logging only; does not cap samples).
- Kaggle auto-detection (new helpers `resolve_default_checkpoint`,
  `resolve_default_output_dir`): if `/kaggle` exists, outputs default to
  `/kaggle/working/results` and the checkpoint defaults to the first
  `/kaggle/input/*/best_model.pt` found, else
  `/kaggle/working/results/best_model.pt`. Locally: `results/best_model.pt`
  and `results/multimodal`. Explicit `--checkpoint` / `--output_dir` always
  win. Default JSON summary is now always written
  (`<output_dir>/multimodal_predictions.json`, or `--output_json` when
  given); previously it was only written with `--output_json`.
- A2 untrained-fallback behaviour was already present and is kept verbatim:
  the LOUD `WARNING: UNTRAINED MODEL` banner on stdout, the per-row
  warning line in the report, and `"untrained_fallback": true` in every
  output JSON row (verified in the smoke-test JSON: 6/6 rows).
- The per-email report now prints context P(phish), URL score, flag score,
  most-contributing modality, the weighted verdict (with corroboration) and
  the max baseline verdict side by side, then the top reasons.

### D5 — CLI contract on touched scripts — Done

- `run_multimodal_predict.py` accepts `--checkpoint`, `--output_dir`,
  `--max_samples` and Kaggle auto-detection (see D4). No other script was
  touched (`run_multimodal_faithfulness.py` was only read).

## Exact public API (for the Section E agent)

```python
from phishing_xai.multimodal import (
    FusionConfig,               # dataclass in phishing_xai.config, re-exported here
    build_concatenated_context, # (extraction, *, tokenizer=None, max_length=512) -> str
    score_urls_for_email,       # (extraction, *, url_config=None) -> dict
    score_flags,                # (extraction) -> dict
    fuse_email,                 # (extraction, part_scores, *, fusion="weighted",
                                #  config=None, concatenated_score=None, threshold=None,
                                #  url_config=None, tokenizer=None, max_length=None) -> dict
    overlay_scanned_pdf_page,   # (pdf_bytes, page, ocr_boxes, lita_top_words,
                                #  output_path, *, dpi=200, top_k=5) -> Path | None
    overlay_qr_image,           # (image_input, qr_urls, output_path, *,
                                #  qr_boxes=None, top_k=5) -> Path | None
    part_id,                    # (part) -> "<kind>[:<filename>][#p<page>]"
    FLAG_CONTRIBUTIONS,         # dict flag type -> numeric contribution
    PART_KIND_TO_MODALITY,      # dict part kind -> modality label
)
```

Minimal Section E usage:

```python
from phishing_xai.multimodal import fuse_email
result = fuse_email(extraction, part_scores, fusion="weighted",
                    concatenated_score=context_prob)
# result["verdict"], result["final_score"], result["final_class"],
# result["most_contributing_modality"], result["top_reasons"],
# result["fusion_weighted"] / result["fusion_max"] when called per fusion.
# part_scores keys: part_id(part) preferred, bare kind also accepted.
```

Default weights / threshold and why (also in `multimodal.py` docstring and
`FusionConfig` docstring):

| Setting | Value | Why |
|---|---|---|
| `w_context` | 0.45 | Largest weight: the concatenated context sees the whole email at once, so it is the most reliable single signal. |
| `w_max_part` | 0.20 | Deliberately small: one part (often a noisy OCR fragment) must not flip the verdict alone — this is the precision guard. |
| `w_url` | 0.20 | Static URL risk is independent evidence from the text model; a clearly malicious link corroborates a suspicious text. |
| `w_flags` | 0.15 | Structural flags (anchor mismatch, auth failures, risky attachments) are supporting evidence, never proof on their own. |
| `threshold` | 0.5 | PLACEHOLDER — tune on validation only on Kaggle (precision-first, recall floor); never claimed as tuned. |
| `high_confidence_threshold` | 0.75 | A fused score this high is phishing even with a single active channel. |

## Precision-vs-max comparison as implemented

- Weighted: final is a corroborated weighted sum; a single part at 0.85
  with context 0.30 and no URL/flags gives 0.305 and stays Safe
  (`corroborated=False`); the same email under max is Phishing (0.85).
  This exact divergence is asserted in
  `test_single_weak_part_high_alone_is_safe_weighted_but_phishing_max`.
- Max: final = max part score; verdict flips on any part >= threshold;
  URL/flag/context evidence is computed and reported but ignored.
- Both verdicts/scores are stored per email (`fusion_weighted`,
  `fusion_max`), so Section E can report precision / recall / FPR for both
  from the same rows without re-running the model.

## Test results

- `.venv/bin/python -m pytest tests/test_multimodal_fusion.py -q`:
  **37 passed, 0 failed** (2026-11-19, venv torch 2.15.0+cpu). Covers:
  context markers/order, body fallback, empty extraction, char-cap and
  fake-tokenizer truncation, weighted arithmetic (0.66 exact), context
  fallback to max part, max-rule firing/ignoring URL+flags, unknown fusion
  error, threshold flips for both fusions, the precision-aware
  weighted-vs-max divergence, strong corroboration flipping weighted,
  empty-email safe under both, URL benign vs malicious (incl. the
  schemeless-www duplicate quirk of Section C `extract_urls`, see below),
  QR bump, PDF links NOT treated as QR, flag contributions (known types,
  auth failures + Reply-To mismatch, all-pass zero, unknown type),
  top-reasons non-empty/human-readable, most-contributing modality
  (qr / url / headers / pdf / kind-fallback), overlay draw + graceful skip
  when Pillow/PyMuPDF are missing (monkeypatched), scanned-PDF render,
  page-out-of-range, and the predict-script `untrained_fallback`
  passthrough + missing-checkpoint error (monkeypatched model loading).
- First run was 33 passed / 3 failed; fixes: (a) restored the `XAIConfig`
  class header dropped by my `config.py` edit (see D1 note); (b) corrected
  the benign-URL test to expect `n_urls == 2` (Section C `extract_urls`
  returns both the full URL and its schemeless `www.` form); (c) the PDF
  `qr_urls` QR-bump bug above, found by the smoke test and fixed in the
  module with a new test. Re-run: 37/37 pass.
- Neighbour suites re-run after the change (same day):
  `pytest tests/test_extract.py tests/test_url_risk.py -q` — **59 passed**.
- `model.py` and `faithfulness.py` untouched (verified via `git status`).

## Smoke test (CPU, untrained fallback — NOT detection evidence)

- `.venv/bin/python scripts/run_multimodal_predict.py samples/multimodal
  --debug --allow_pretrained_fallback --max_samples 6 --output_dir
  /tmp/smoke_D --fusion weighted` — executed cleanly; 6/6 rows written to
  `/tmp/smoke_D/multimodal_predictions.json`, all with
  `"untrained_fallback": true` and the LOUD warning printed.
- All 6 samples verdict Safe under both fusions (expected: untrained base
  `roberta-base` scores every part/context at chance, P(phish) 0.43–0.44,
  e.g. sample3 QR 0.4382, sample6 benign 0.4405). Weighted scores
  0.28–0.32, max scores 0.44. These numbers are pipeline-demonstration
  output only — do NOT cite them as phishing-detection evidence.
- Post-fix check: sample4 URL score is now 0.00 (was 0.10 with the wrong
  QR bump); sample3 QR URL correctly scores 0.20 (http 0.10 + QR-only
  0.10; no generic-lure signal because `secure-login-example` is a single
  hyphenated label, so the registered domain is the host itself).
- `--explain` run on sample5 (`--lime_samples 20`): LITA ran on the
  concatenated context, top words printed (confirmed +0.845, account
  +0.582, ...); no overlay file was produced because the winning part was
  the subject (untrained scores are all chance-level), which is expected
  behaviour, not a failure.

## Known quirks / things Section E should know

- Section C `extract_urls` returns both `https://www.x/...` and the
  schemeless `www.x/...` form, so `n_urls` can count one link twice. The
  max (`url_score`) is unaffected; do not treat `n_urls` as a distinct-URL
  count without de-duplicating on registered domain.
- Section B stores PDF hyperlinks in the `qr_urls` slot of `pdf_text` /
  `pdf_ocr` parts (see D1 fix). If Section B later moves them to `links`,
  `score_urls_for_email` still works (it reads both), but the
  `pdf_text`-as-QR-source exception in the module can then be dropped.
- `fuse_email` credits the context contribution to the winning part's
  modality (documented heuristic); it is not a per-section model score.
- The predict script's explained images go to `--output_dir` when given
  (else next to the .eml file); the JSON summary is always written now.

## Needs Kaggle / trained checkpoint verification (NOT RUN YET)

- Everything about real detection: threshold tuning on VALIDATION only
  (precision-first, recall floor — the 0.5 default is a placeholder),
  precision / recall / FPR for weighted vs max on labelled data, and
  whether the corroboration gate costs too much recall on real phishing
  (if it does, tune the weights/threshold on validation — do not weaken
  the gate ad hoc).
- End-to-end overlay for a PDF/QR winning part (needs a trained model so a
  PDF/QR part actually wins), and LITA on the concatenated context with
  the trained checkpoint.
- Kaggle path auto-detection itself (`/kaggle/input/*/best_model.pt`,
  `/kaggle/working/results`) — cannot be exercised on this VM, which has
  no `/kaggle` directory; the code paths are unit-covered only via the
  monkeypatched `main()` test, which does not touch the Kaggle branches.
