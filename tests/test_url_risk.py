"""Tests for static lexical URL risk scoring (src/phishing_xai/url_risk.py).

No GPU, no internet, no checkpoint needed: scoring is purely lexical.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phishing_xai import url_risk
from phishing_xai.url_risk import (
    extract_urls,
    registered_domain,
    score_url,
    score_urls,
)

RANDOM_TOKEN = "aBcDeFgHiJkLmNoPqRsTuVwXyZ1234567890abcdefGh"


def reasons_text(result):
    return " ".join(result["reasons"]).lower()


# ---------------------------------------------------------------------------
# Benign URLs must score low
# ---------------------------------------------------------------------------

def test_benign_google_search_scores_low():
    res = score_url("https://www.google.com/search?q=test")
    assert res["url_score"] < 0.35
    assert res["reasons"] == []


def test_benign_wikipedia_scores_low():
    res = score_url("https://en.wikipedia.org/wiki/Phishing")
    assert res["url_score"] < 0.35
    assert res["registered_domain"] == "wikipedia.org"


def test_benign_amazon_product_scores_low():
    res = score_url("https://www.amazon.com/dp/B08N5WRWNW")
    assert res["url_score"] < 0.35


def test_real_paypal_domain_not_brand_mismatch():
    res = score_url("https://www.paypal.com/signin")
    assert res["url_score"] < 0.35
    assert res["signals"].get("brand_mismatch") is None
    assert "brand" not in reasons_text(res)


def test_real_microsoft_login_subdomain_not_brand_mismatch():
    res = score_url("https://login.microsoft.com/common/oauth2/authorize")
    assert res["url_score"] < 0.35
    assert res["signals"].get("brand_mismatch") is None


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------

def test_long_url_flagged():
    short = score_url("https://example.com/a")
    long_url = "https://example.com/" + "a" * 220
    res = score_url(long_url)
    assert "long" in reasons_text(res)
    assert res["url_score"] > short["url_score"]


def test_many_subdomains_flagged():
    res = score_url("https://a.b.c.d.example.com/login")
    assert res["signals"]["subdomain_count"] == 4
    assert "subdomain" in reasons_text(res)
    assert res["url_score"] >= 0.15


def test_two_subdomains_small_bump():
    res = score_url("https://a.b.example.com/")
    assert res["signals"]["subdomain_count"] == 2
    assert res["url_score"] == pytest.approx(0.08)


def test_at_sign_flagged():
    res = score_url("https://www.google.com@evil-example.com/login")
    assert "@" in reasons_text(res)
    assert res["signals"]["at_sign"] is True
    assert res["url_score"] >= 0.20


def test_ipv4_host_flagged():
    res = score_url("http://203.0.113.45/login")
    assert "ip address" in reasons_text(res)
    assert res["signals"]["ip_host"] == 4
    assert res["url_score"] >= 0.40  # 0.30 IP + 0.10 http


def test_ipv6_host_flagged():
    res = score_url("http://[2001:db8::1]/secure")
    assert "ip address" in reasons_text(res)
    assert res["signals"]["ip_host"] == 6
    assert res["url_score"] >= 0.30


def test_punycode_host_flagged():
    res = score_url("https://xn--pypal-4y8d.example.com/login")
    assert "punycode" in reasons_text(res)
    assert res["signals"]["punycode"] is True
    assert res["url_score"] >= 0.25


def test_non_ascii_host_flagged():
    # Cyrillic 'а' (U+0430) in place of Latin 'a' in "paypal".
    res = score_url("https://p\u0430ypal.com/login")
    assert "non-ascii" in reasons_text(res)
    assert res["signals"]["non_ascii_host"] is True
    assert res["url_score"] >= 0.25


def test_suspicious_tld_xyz_flagged():
    res = score_url("https://example.xyz/login")
    assert "suspicious" in reasons_text(res)
    assert res["signals"]["tld"] == "xyz"
    assert res["url_score"] >= 0.20


def test_suspicious_tld_tk_flagged():
    res = score_url("https://free-prize.tk/claim")
    assert res["signals"]["tld"] == "tk"
    assert res["url_score"] >= 0.20


def test_shortener_bitly_flagged():
    res = score_url("https://bit.ly/3AbCdEf")
    assert "shortener" in reasons_text(res)
    assert res["signals"]["shortener"] == "bit.ly"
    assert res["url_score"] >= 0.25


def test_shortener_tco_flagged():
    res = score_url("https://t.co/AbCdEfGh")
    assert res["signals"]["shortener"] == "t.co"
    assert res["url_score"] >= 0.25


def test_http_riskier_than_https():
    http_res = score_url("http://example.com/page")
    https_res = score_url("https://example.com/page")
    assert http_res["url_score"] > https_res["url_score"]
    assert "http" in reasons_text(http_res)
    assert https_res["url_score"] == 0.0


def test_many_hyphens_in_host_flagged():
    res = score_url("https://paypal-secure-login-verify-account.com/")
    assert "hyphen" in reasons_text(res)
    assert res["signals"]["hyphen_count"] >= 3
    assert res["url_score"] >= 0.10


def test_brand_in_subdomain_vs_other_domain_flagged():
    res = score_url("https://paypal-login.example.xyz/secure")
    assert "paypal" in reasons_text(res)
    assert res["signals"]["brand_mismatch"] == "paypal"
    # brand mismatch 0.25 + suspicious TLD .xyz 0.20 + generic lure 0.08
    assert res["url_score"] >= 0.45


def test_brand_in_path_vs_other_domain_flagged():
    res = score_url("https://example.com/microsoft-verify/login")
    assert res["signals"]["brand_mismatch"] == "microsoft"
    assert res["url_score"] >= 0.25


def test_high_entropy_token_flagged():
    res = score_url(f"https://example.com/reset/{RANDOM_TOKEN}")
    assert "random-looking" in reasons_text(res)
    assert res["signals"]["max_token_entropy"] >= 4.0
    assert res["url_score"] >= 0.15


def test_normal_word_not_high_entropy():
    res = score_url("https://example.com/internationalization")
    assert res["signals"].get("high_entropy_token") is None


def test_data_scheme_high_risk():
    res = score_url("data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==")
    assert res["scheme"] == "data"
    assert "data:" in reasons_text(res)
    assert res["url_score"] >= 0.50


def test_javascript_scheme_high_risk():
    res = score_url("javascript:alert(document.cookie)")
    assert res["scheme"] == "javascript"
    assert "javascript:" in reasons_text(res)
    assert res["url_score"] >= 0.50


def test_qr_only_flag_increases_score_and_adds_reason():
    url = "https://example.com/qr-verify"
    plain = score_url(url)
    qr = score_url(url, in_qr=True)
    assert qr["url_score"] > plain["url_score"]
    assert qr["url_score"] == pytest.approx(plain["url_score"] + 0.10)
    assert "qr code" in reasons_text(qr)
    assert qr["signals"]["qr_only"] is True


# ---------------------------------------------------------------------------
# Composite malicious URLs must score high
# ---------------------------------------------------------------------------

def test_composite_ip_at_http_entropy_scores_high():
    url = f"http://user@203.0.113.45/{RANDOM_TOKEN}extra9876543210"
    res = score_url(url)
    assert res["url_score"] > 0.7
    assert "ip address" in reasons_text(res)
    assert "@" in reasons_text(res)


def test_composite_punycode_brand_tld_scores_high():
    url = f"http://paypal-verify.xn--pypal-9d0b.tk/{RANDOM_TOKEN}"
    res = score_url(url)
    assert res["url_score"] > 0.7
    assert res["signals"]["brand_mismatch"] == "paypal"
    assert res["signals"]["punycode"] is True


# ---------------------------------------------------------------------------
# Score bounds, input handling, helpers
# ---------------------------------------------------------------------------

def test_score_bounds_zero_to_one():
    urls = [
        "https://www.google.com/",
        "http://user@203.0.113.45/" + RANDOM_TOKEN,
        f"http://paypal-verify.xn--pypal-9d0b.tk/{RANDOM_TOKEN}",
        "data:text/html,hello",
        "",
        "not a url at all",
    ]
    for res in score_urls(urls):
        assert 0.0 <= res["url_score"] <= 1.0


def test_empty_input_handled():
    res = score_url("")
    assert res["url_score"] == 0.0
    assert "empty" in reasons_text(res)
    assert res["registered_domain"] == ""


def test_none_input_handled():
    res = score_url(None)
    assert res["url_score"] == 0.0


def test_invalid_input_no_host_handled():
    res = score_url("not a url at all")
    assert res["url_score"] == 0.0
    assert res["host"] == ""


def test_score_urls_helper_matches_score_url():
    urls = ["https://www.google.com/", "https://bit.ly/abc123"]
    batch = score_urls(urls)
    assert len(batch) == 2
    assert batch[0]["url_score"] == score_url(urls[0])["url_score"]
    assert batch[1]["url_score"] == score_url(urls[1])["url_score"]


def test_score_urls_empty_list():
    assert score_urls([]) == []


def test_extract_urls_finds_urls_in_text():
    text = "Visit https://example.com/a and http://bit.ly/x, also see www.test.org/p."
    found = extract_urls(text)
    assert "https://example.com/a" in found
    assert "http://bit.ly/x" in found
    assert "www.test.org/p" in found


def test_extract_urls_dedupes_and_empty():
    assert extract_urls("https://a.com/ https://a.com/") == ["https://a.com/"]
    assert extract_urls("") == []
    assert extract_urls("no links here") == []


# ---------------------------------------------------------------------------
# Registered-domain handling (incl. fallback without tldextract)
# ---------------------------------------------------------------------------

def test_registered_domain_simple():
    assert registered_domain("www.example.com") == "example.com"
    assert registered_domain("sub.example.co.uk") == "example.co.uk"
    assert registered_domain("a.b.example.com.au") == "example.com.au"
    assert registered_domain("example.com") == "example.com"
    assert registered_domain("localhost") == "localhost"
    assert registered_domain("") == ""


def test_registered_domain_fallback_without_tldextract(monkeypatch):
    monkeypatch.setattr(url_risk, "_TLDEXTRACT_EXTRACTOR", None)
    assert registered_domain("sub.example.co.uk") == "example.co.uk"
    assert registered_domain("www.example.com") == "example.com"
    assert registered_domain("a.b.c.example.com") == "example.com"


def test_registered_domain_fallback_unknown_suffix_limitation():
    # Fallback takes the last two labels, so an uncovered multi-label
    # suffix is under-counted by one label; documented limitation.
    assert registered_domain("www.example.pvt.k12.ma.us") == "ma.us"
