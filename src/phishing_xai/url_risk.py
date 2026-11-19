"""Static lexical URL risk scoring for phishing detection (Section C).

Purely static analysis of a URL string: no network calls, no DNS lookups,
no fetching, no downloads. Used for URLs found in an email body, HTML
``href`` attributes, PDF links, QR-code payloads and OCR text.

Each signal contributes a weight to a ``url_score`` in ``[0, 1]`` and adds a
human-readable reason. Weights are heuristic and calibrated so that everyday
benign URLs (e.g. ``https://www.google.com/search?q=test``) score below
0.35 while URLs stacking several phishing signals score above 0.7. See
``UrlRiskConfig`` for the weights and thresholds; tune them there, not in
``score_url``.

Registered-domain handling: ``tldextract`` is used only when it is
importable and works offline (guarded import, constructed with
``suffix_list_urls=[]`` so it uses its bundled snapshot and never downloads
a suffix list). ``tldextract`` is NOT a hard dependency (it is not in
``pyproject.toml``). When it is unavailable the fallback in
``registered_domain`` takes the last two labels, except for a small set of
known multi-label public suffixes (``co.uk``, ``com.au``, ``ac.uk``, ...),
where it takes the last three labels. Limitation of the fallback: for a
public suffix not in that small set (e.g. ``co.jp`` is covered, but an
obscure one such as ``pvt.k12.ma.us`` is not) the registered domain may be
under- or over-counted by one label, which can shift the subdomain count
and the brand-mismatch check by one label. That only moves the score by at
most one signal weight and never changes a scheme/IP/punycode verdict.
"""

import ipaddress
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.url_risk")

__all__ = [
    "UrlRiskConfig",
    "SUSPICIOUS_TLDS",
    "URL_SHORTENERS",
    "BRAND_DOMAINS",
    "registered_domain",
    "extract_urls",
    "score_url",
    "score_urls",
]

# ---------------------------------------------------------------------------
# Optional tldextract (offline only, never a hard dependency)
# ---------------------------------------------------------------------------

try:
    import tldextract as _tldextract
    # Empty suffix_list_urls -> bundled snapshot only, no network fetch.
    _TLDEXTRACT_EXTRACTOR = _tldextract.TLDExtract(suffix_list_urls=[])
except Exception:  # pragma: no cover - tldextract simply not installed
    _tldextract = None
    _TLDEXTRACT_EXTRACTOR = None

# Small set of known multi-label public suffixes used by the fallback.
# Deliberately short: it only needs to cover the common cases; anything
# else falls back to "last two labels" (see module docstring).
MULTI_LABEL_SUFFIXES = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk", "net.uk",
    "com.au", "net.au", "org.au",
    "co.nz", "org.nz",
    "co.jp", "co.kr", "co.za", "com.br", "com.mx", "com.ar",
    "co.in", "com.sg", "com.hk", "com.tw",
})

#: TLDs frequently abused in phishing. Heuristic list, not a blocklist:
#: plenty of legitimate sites use these TLDs, so this is only +0.20.
SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq",       # free Freenom TLDs, heavily abused
    "xyz", "top", "click", "link", "buzz", "work", "live",
    "icu", "cyou", "quest", "sbs",
})

#: Known URL-shortener registered domains. A shortened link hides the real
#: destination, so it is a risk signal on its own, not proof of phishing.
URL_SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "shorturl.at", "tiny.cc", "rb.gy", "cutt.ly",
    "rebrand.ly", "shrtco.de", "bitly.com", "s.id", "t.ly", "bl.ink",
})

#: Brand keyword -> the brand's real registered domain. A brand keyword in a
#: subdomain/path only counts as a mismatch when the URL's registered domain
#: is different from this real domain, so https://www.paypal.com/... and
#: https://login.microsoft.com/... are never flagged.
BRAND_DOMAINS = {
    "paypal": "paypal.com",
    "microsoft": "microsoft.com",
    "apple": "apple.com",
    "amazon": "amazon.com",
    "google": "google.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "netflix": "netflix.com",
    "whatsapp": "whatsapp.com",
    "linkedin": "linkedin.com",
    "twitter": "twitter.com",
    "chase": "chase.com",
    "hsbc": "hsbc.com",
    "barclays": "barclays.co.uk",
    "visa": "visa.com",
    "mastercard": "mastercard.com",
    "binance": "binance.com",
    "coinbase": "coinbase.com",
    "dhl": "dhl.com",
    "fedex": "fedex.com",
    "adobe": "adobe.com",
    "dropbox": "dropbox.com",
}

#: Generic lure words. Weaker than a brand mismatch (+0.08, subdomain only):
#: real sites use "support.example.com" and "login.microsoft.com" too, so a
#: generic word alone must never push a benign URL over the threshold.
GENERIC_LURE_WORDS = frozenset({
    "login", "verify", "secure", "account", "update", "support",
    "bank", "signin", "confirm", "unlock",
})

#: Schemes that carry executable / hidden content instead of a web page.
DANGEROUS_SCHEMES = frozenset({"data", "javascript"})

_URL_REGEX = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_SCHEMELESS_URL_REGEX = re.compile(r"\bwww\.[^\s<>\"')\]]+", re.IGNORECASE)
_DATA_JS_REGEX = re.compile(r"\b(?:data|javascript):[^\s<>\"')\]]+", re.IGNORECASE)


@dataclass
class UrlRiskConfig:
    """Weights and thresholds for URL risk scoring.

    Tune here (and re-run ``tests/test_url_risk.py``) rather than editing
    ``score_url``. All weights are additive and the total is clamped to
    ``[0, 1]``.
    """
    # Per-signal weights.
    w_dangerous_scheme: float = 0.55
    w_ip_host: float = 0.30
    w_at_sign: float = 0.20
    w_punycode: float = 0.25
    w_suspicious_tld: float = 0.20
    w_shortener: float = 0.25
    w_http: float = 0.10
    w_many_subdomains: float = 0.15   # 3+ subdomain labels
    w_two_subdomains: float = 0.08    # exactly 2 subdomain labels
    w_many_hyphens: float = 0.15      # 5+ hyphens in host
    w_some_hyphens: float = 0.10      # 3-4 hyphens in host
    w_brand_mismatch: float = 0.25
    w_generic_lure_subdomain: float = 0.08
    w_high_entropy_token: float = 0.15
    w_qr_only: float = 0.10
    # Length tiers (longest matching tier wins).
    long_url_len: int = 75
    w_long_url: float = 0.05
    very_long_url_len: int = 120
    w_very_long_url: float = 0.10
    extremely_long_url_len: int = 200
    w_extremely_long_url: float = 0.15
    # High-entropy token: a single host/path segment this long whose
    # Shannon entropy (bits/char over its own character distribution)
    # reaches this threshold looks machine-generated, not a word.
    entropy_min_segment_len: int = 16
    entropy_threshold: float = 4.0
    # Subdomain / hyphen count thresholds.
    many_subdomains: int = 3
    many_hyphens: int = 5
    some_hyphens: int = 3


def _fallback_registered_domain(host: str) -> str:
    """Last-two-labels registered domain, with multi-label suffix handling."""
    labels = [label for label in host.lower().split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    last_two = ".".join(labels[-2:])
    if last_two in MULTI_LABEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def registered_domain(host: str) -> str:
    """Return the registered domain (e.g. ``example.co.uk``) for a host.

    Uses ``tldextract`` when available (offline, bundled snapshot); otherwise
    the minimal fallback documented in the module docstring. Returns ``""``
    for an empty host and the host itself when it has no dot (e.g. an IP
    address or ``localhost``).
    """
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return ""
    if _TLDEXTRACT_EXTRACTOR is not None:
        try:
            ext = _TLDEXTRACT_EXTRACTOR(host)
            if ext.domain and ext.suffix:
                return f"{ext.domain}.{ext.suffix}"
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"tldextract failed for host {host!r}: {exc}")
    return _fallback_registered_domain(host)


def extract_urls(text: str) -> List[str]:
    """Extract URLs from free text (body, HTML text, PDF text, OCR text).

    Finds ``http(s)://`` URLs, schemeless ``www.`` URLs and ``data:`` /
    ``javascript:`` URIs, de-duplicated in order of first appearance.
    Trailing sentence punctuation is stripped. Never fetches anything.
    """
    if not text:
        return []
    found: List[str] = []
    seen = set()
    for pattern in (_URL_REGEX, _DATA_JS_REGEX, _SCHEMELESS_URL_REGEX):
        for match in pattern.findall(text):
            url = match.rstrip(".,;:!?")
            if url and url not in seen:
                seen.add(url)
                found.append(url)
    # Preserve order of first appearance in the text, not pattern order.
    found.sort(key=lambda u: text.find(u))
    return found


def _shannon_entropy(segment: str) -> float:
    """Shannon entropy (bits per character) of one string's characters."""
    if not segment:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in segment:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(segment)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _valid_host_chars(host: str) -> bool:
    """True when a host contains only plausible hostname characters.

    Allows letters/digits, ``- . _ : [ ]`` and non-ASCII letters (IDN hosts,
    handled by the punycode signal). Rejects whitespace and URL punctuation,
    so schemeless garbage such as ``"not a url at all"`` is unparseable
    rather than scored as a bare hostname.
    """
    for ch in host:
        if ch.isspace():
            return False
        if ord(ch) > 127:
            continue
        if ch.isalnum() or ch in "-._:[]":
            continue
        return False
    return True


def _subdomain_labels(host: str, reg_domain: str) -> List[str]:
    """Host labels that sit left of the registered domain."""
    if not host or not reg_domain:
        return []
    labels = host.lower().split(".")
    reg_labels = reg_domain.split(".")
    if len(labels) <= len(reg_labels):
        return []
    return labels[: len(labels) - len(reg_labels)]


def score_url(
    url: str,
    *,
    in_qr: bool = False,
    config: Optional[UrlRiskConfig] = None,
) -> Dict[str, Any]:
    """Score one URL for phishing risk using only static lexical signals.

    Args:
        url: The URL string (body, HTML href, PDF link, QR payload, OCR text).
        in_qr: Pass ``True`` when this URL came only from a QR code: a QR
            code hides the link from the reader, so it adds a small bump.
        config: Optional :class:`UrlRiskConfig` overriding weights/thresholds.

    Returns:
        ``{"url", "url_score", "reasons", "signals", "registered_domain",
        "host", "scheme"}`` where ``url_score`` is a float clamped to
        ``[0, 1]`` and ``signals`` maps each signal name to its detail
        (bool / number / string) so Section D fusion can use more than the
        single score.
    """
    cfg = config or UrlRiskConfig()
    raw = url if isinstance(url, str) else ""
    url_str = raw.strip()
    signals: Dict[str, Any] = {}
    reasons: List[str] = []
    score = 0.0

    def add(weight: float, signal: str, detail: Any, reason: str) -> None:
        nonlocal score
        signals[signal] = detail
        if weight:
            score += weight
        reasons.append(reason)

    if not url_str:
        signals["empty"] = True
        return {
            "url": raw,
            "url_score": 0.0,
            "reasons": ["empty or missing URL"],
            "signals": signals,
            "registered_domain": "",
            "host": "",
            "scheme": "",
        }

    parsed = urlparse(url_str)
    scheme = (parsed.scheme or "").lower()
    signals["scheme"] = scheme

    # Dangerous schemes carry no host to analyse; score them on their own.
    if scheme in DANGEROUS_SCHEMES:
        add(
            cfg.w_dangerous_scheme,
            "dangerous_scheme",
            scheme,
            f"'{scheme}:' scheme can hide executable or misleading content",
        )
        if in_qr:
            add(cfg.w_qr_only, "qr_only", True,
                "URL appeared only inside a QR code, hiding the link")
        signals["length"] = len(url_str)
        return _result(raw, url_str, score, reasons, signals, "", "", scheme)

    # Schemeless input such as "www.example.com/path": re-parse with "//".
    if not parsed.netloc and not scheme:
        parsed = urlparse("//" + url_str)
    host = (parsed.hostname or "").lower()
    signals["host"] = host
    if not host or not _valid_host_chars(host):
        signals["unparseable"] = True
        return {
            "url": raw,
            "url_score": 0.0,
            "reasons": ["could not parse a host from the URL"],
            "signals": signals,
            "registered_domain": "",
            "host": "",
            "scheme": scheme,
        }

    reg_domain = registered_domain(host)
    signals["registered_domain"] = reg_domain

    # --- Length -----------------------------------------------------------
    signals["length"] = len(url_str)
    if len(url_str) > cfg.extremely_long_url_len:
        add(cfg.w_extremely_long_url, "long_url", len(url_str),
            f"very long URL ({len(url_str)} chars)")
    elif len(url_str) > cfg.very_long_url_len:
        add(cfg.w_very_long_url, "long_url", len(url_str),
            f"long URL ({len(url_str)} chars)")
    elif len(url_str) > cfg.long_url_len:
        add(cfg.w_long_url, "long_url", len(url_str),
            f"long URL ({len(url_str)} chars)")

    # --- '@' (userinfo can mask the real host) ----------------------------
    if "@" in (parsed.netloc or ""):
        add(cfg.w_at_sign, "at_sign", True,
            "'@' in URL can hide the real host behind a fake userinfo part")
    signals["at_sign"] = "@" in (parsed.netloc or "")

    # --- IP-address host --------------------------------------------------
    ip_version = None
    try:
        ip_version = ipaddress.ip_address(host).version
    except ValueError:
        ip_version = None
    signals["ip_host"] = ip_version
    if ip_version is not None:
        add(cfg.w_ip_host, "ip_host", ip_version,
            f"host is an IP address (IPv{ip_version}) instead of a domain name")

    # --- Punycode / non-ASCII host (homograph risk) -----------------------
    non_ascii = any(ord(ch) > 127 for ch in host)
    punycode = any(label.startswith("xn--") for label in host.split("."))
    signals["punycode"] = punycode
    signals["non_ascii_host"] = non_ascii
    if punycode or non_ascii:
        detail = "punycode (xn--)" if punycode else "non-ASCII characters"
        add(cfg.w_punycode, "punycode_or_non_ascii", detail,
            f"host uses {detail}, which can mimic a trusted domain (homograph)")

    # --- Suspicious TLD ---------------------------------------------------
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    signals["tld"] = tld
    if tld in SUSPICIOUS_TLDS:
        add(cfg.w_suspicious_tld, "suspicious_tld", tld,
            f"suspicious top-level domain '.{tld}' (frequently abused)")

    # --- URL shortener ----------------------------------------------------
    is_shortener = reg_domain in URL_SHORTENERS
    signals["shortener"] = reg_domain if is_shortener else None
    if is_shortener:
        add(cfg.w_shortener, "shortener", reg_domain,
            f"URL shortener ({reg_domain}) hides the real destination")

    # --- http vs https -----------------------------------------------------
    if scheme == "http":
        add(cfg.w_http, "http_not_https", True,
            "plain http, not https (no transport encryption)")
    signals["http_not_https"] = scheme == "http"

    # --- Subdomain count ---------------------------------------------------
    sub_labels = [] if ip_version is not None else _subdomain_labels(host, reg_domain)
    signals["subdomain_count"] = len(sub_labels)
    if len(sub_labels) >= cfg.many_subdomains:
        add(cfg.w_many_subdomains, "many_subdomains", len(sub_labels),
            f"many subdomains ({len(sub_labels)}) can bury the real domain")
    elif len(sub_labels) == 2:
        add(cfg.w_two_subdomains, "many_subdomains", len(sub_labels),
            "two subdomains before the registered domain")

    # --- Hyphens in host ----------------------------------------------------
    hyphen_count = host.count("-")
    signals["hyphen_count"] = hyphen_count
    if hyphen_count >= cfg.many_hyphens:
        add(cfg.w_many_hyphens, "many_hyphens", hyphen_count,
            f"many hyphens in host ({hyphen_count})")
    elif hyphen_count >= cfg.some_hyphens:
        add(cfg.w_some_hyphens, "many_hyphens", hyphen_count,
            f"several hyphens in host ({hyphen_count})")

    # --- Brand keyword with a different registered domain ------------------
    haystack_sub = ".".join(sub_labels)
    path = (parsed.path or "").lower()
    brand_hit = None
    for brand, real_domain in BRAND_DOMAINS.items():
        if brand in haystack_sub or brand in path:
            if reg_domain != real_domain:
                brand_hit = brand
                break
    signals["brand_mismatch"] = brand_hit
    if brand_hit is not None:
        add(
            cfg.w_brand_mismatch,
            "brand_mismatch",
            brand_hit,
            f"brand name '{brand_hit}' in subdomain/path but the registered "
            f"domain is '{reg_domain}', not the real '{BRAND_DOMAINS[brand_hit]}'",
        )

    # --- Generic lure word in subdomain --------------------------------------
    generic_hit = next((w for w in GENERIC_LURE_WORDS if w in haystack_sub), None)
    signals["generic_lure_subdomain"] = generic_hit
    if generic_hit is not None:
        add(cfg.w_generic_lure_subdomain, "generic_lure_subdomain", generic_hit,
            f"lure word '{generic_hit}' in subdomain")

    # --- Long random-looking (high-entropy) token -----------------------------
    segments = [s for s in re.split(r"[^A-Za-z0-9]+", host + " " + (parsed.path or "")) if s]
    worst_segment, worst_entropy = "", 0.0
    for segment in segments:
        if len(segment) >= cfg.entropy_min_segment_len:
            entropy = _shannon_entropy(segment)
            if entropy > worst_entropy:
                worst_segment, worst_entropy = segment, entropy
    signals["max_token_entropy"] = round(worst_entropy, 2)
    signals["max_token"] = worst_segment
    if worst_entropy >= cfg.entropy_threshold:
        add(
            cfg.w_high_entropy_token,
            "high_entropy_token",
            {"token": worst_segment, "entropy": round(worst_entropy, 2)},
            f"long random-looking token (entropy {worst_entropy:.2f})",
        )

    # --- QR-only -------------------------------------------------------------
    signals["qr_only"] = bool(in_qr)
    if in_qr:
        add(cfg.w_qr_only, "qr_only", True,
            "URL appeared only inside a QR code, hiding the link from the reader")

    return _result(raw, url_str, score, reasons, signals, reg_domain, host, scheme)


def _result(
    raw: str,
    url_str: str,
    score: float,
    reasons: List[str],
    signals: Dict[str, Any],
    reg_domain: str,
    host: str,
    scheme: str,
) -> Dict[str, Any]:
    """Clamp the score to [0, 1] and build the standard result dict."""
    clamped = max(0.0, min(1.0, float(score)))
    logger.debug(f"score_url {url_str[:80]!r} -> {clamped:.3f} ({len(reasons)} reasons)")
    return {
        "url": raw,
        "url_score": round(clamped, 3),
        "reasons": reasons,
        "signals": signals,
        "registered_domain": reg_domain,
        "host": host,
        "scheme": scheme,
    }


def score_urls(
    urls: Iterable[str],
    *,
    in_qr: bool = False,
    config: Optional[UrlRiskConfig] = None,
) -> List[Dict[str, Any]]:
    """Score each URL in ``urls`` with :func:`score_url`.

    ``in_qr`` applies to every URL (use it when the whole batch came from QR
    codes); for mixed batches call :func:`score_url` per URL instead.
    """
    return [score_url(url, in_qr=in_qr, config=config) for url in urls]
