# Section C — URL / structural risk module (url_risk.py)

## Decision
- New pure-static lexical URL risk module: `src/phishing_xai/url_risk.py`.
  No network calls, no DNS, no fetching, no new hard dependencies.
- `tldextract` is used ONLY via a guarded import with `TLDExtract(suffix_list_urls=[])`
  (bundled snapshot, no downloads). It is NOT in `pyproject.toml` and is not
  installed in the venv, so the minimal fallback runs in practice: last two
  labels, with a small multi-label public-suffix set (`co.uk`, `com.au`,
  `ac.uk`, `co.jp`, ... -> last three labels). Fallback limitation documented
  in the module docstring: an uncovered multi-label suffix is under-counted
  by one label, shifting subdomain count / brand check by at most one signal
  weight, never a scheme/IP/punycode verdict.
- Public API (kept small):
  - `score_url(url: str, *, in_qr: bool = False, config: UrlRiskConfig | None = None) -> dict`
    returns `{"url", "url_score" (0-1, clamped), "reasons": [str], "signals": {...},
    "registered_domain", "host", "scheme"}`.
  - `score_urls(urls, *, in_qr=False, config=None) -> list[dict]` batch helper.
  - `extract_urls(text) -> list[str]` (http(s), `data:`/`javascript:`, schemeless
    `www.`, de-duplicated, order-preserving).
  - `registered_domain(host) -> str` helper + `UrlRiskConfig` dataclass for
    weights/thresholds (tune there, not in `score_url`).
  - Exported constants: `SUSPICIOUS_TLDS`, `URL_SHORTENERS`, `BRAND_DOMAINS`.
- Signals/weights (see `UrlRiskConfig`): dangerous scheme `data:`/`javascript:`
  0.55; IP host 0.30; `@` 0.20; punycode/non-ASCII host 0.25; suspicious TLD
  0.20 (list: tk ml ga cf gq xyz top click link buzz work live icu cyou quest
  sbs — documented as heuristic, not a blocklist); shortener 0.25; http 0.10;
  3+ subdomains 0.15 / 2 subdomains 0.08; 5+ hyphens 0.15 / 3-4 hyphens 0.10;
  brand keyword in subdomain/path with a different registered domain 0.25
  (real paypal.com / login.microsoft.com never flagged); generic lure word in
  subdomain only 0.08; high-entropy token (segment >=16 chars, Shannon
  entropy >=4.0) 0.15; long URL tiers 0.05/0.10/0.15 (>75/>120/>200 chars);
  QR-only (`in_qr=True`) 0.10.
- Style: typed, docstrings, `get_logger("phishing_xai.url_risk")` from
  `phishing_xai.utils`. Read `extract.py` (URL_REGEX/QR payloads),
  `utils.py`, `config.py` (dataclass style), `tests/test_extract.py`,
  `pyproject.toml` (confirmed: no `tldextract` dependency) before writing.
- Reason: Section C scope only. Did NOT touch `model.py`, `faithfulness.py`,
  `decision.md`, `docs/PROJECT_LOG.md`; did NOT retrain; did NOT push.

## Status per change
- `src/phishing_xai/url_risk.py` (new, ~370 lines) — Done.
- `tests/test_url_risk.py` (new, 35 tests) — Done. Covers every signal:
  benign google/wikipedia/amazon/paypal.com/login.microsoft.com (<0.35);
  long URL; many/two subdomains; `@`; IPv4 + IPv6; punycode; non-ASCII host;
  suspicious TLD (.xyz, .tk); shorteners (bit.ly, t.co); http vs https;
  hyphens; brand mismatch in subdomain and in path; real-domain non-flag;
  high-entropy token + normal-word non-flag; `data:` + `javascript:` schemes;
  QR-only bump (+0.10); two composite malicious URLs (>0.7); score bounds
  0-1; empty/None/unparseable input; `score_urls` helper; `extract_urls`
  (incl. dedupe/empty); registered-domain correctness incl. forced-fallback
  test (monkeypatched `_TLDEXTRACT_EXTRACTOR = None`) and the documented
  unknown-suffix limitation. No GPU/internet/checkpoint required.
- Calibration (verified pre-pytest with stubbed torch/numpy, system python):
  benign google/wikipedia/paypal.com = 0.0, login.microsoft.com = 0.08;
  IP+`@`+http+long+entropy composite = 0.80; punycode+brand+.tk+http
  composite = 1.0 (clamped); `data:`/`javascript:` = 0.55 each.

## Test results
- `.venv/bin/python -m pytest tests/test_url_risk.py -q` — **39 passed in 6.63s**
  (2026-11-19, venv torch 2.15.0+cpu). First run: 38 passed, 1 failed
  (`test_invalid_input_no_host_handled`: schemeless garbage `"not a url at
  all"` was parsed as a bare hostname instead of unparseable). Fixed in the
  module, not the test: added `_valid_host_chars()` so hosts with whitespace
  or URL punctuation are rejected as unparseable (score 0.0, reason
  "could not parse a host from the URL"). Re-run: 39/39 pass.
- No other test files run (other sections own them); `model.py` and
  `faithfulness.py` untouched; nothing pushed; no retraining.
