"""Tests for Section D multimodal fusion (src/phishing_xai/multimodal.py).

No GPU, no internet, no checkpoint needed: fusion is pure arithmetic over
caller-supplied scores, and the model is never loaded here. The one test
that exercises scripts/run_multimodal_predict.py monkeypatches model
loading and per-email prediction, so it also needs no checkpoint.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.config import FusionConfig
from phishing_xai.multimodal import (
    build_concatenated_context,
    fuse_email,
    overlay_qr_image,
    overlay_scanned_pdf_page,
    part_id,
    score_flags,
    score_urls_for_email,
)


def make_extraction(parts=None, body_text="", links=None, flags=None, headers=None):
    return {
        "body_text": body_text,
        "parts": parts or [],
        "links": links or [],
        "flags": flags or [],
        "headers": headers or {},
        "skipped": [],
    }


def P(kind, text, filename=None, page=None, qr_urls=None, boxes=None):
    return {
        "kind": kind,
        "filename": filename,
        "text": text,
        "boxes": boxes or [],
        "qr_urls": qr_urls or [],
        "page": page,
    }


# ---------------------------------------------------------------------------
# part_id
# ---------------------------------------------------------------------------

def test_part_id_includes_kind_filename_page():
    assert part_id(P("image", "x", filename="a.png", page=2)) == "image:a.png#p2"
    assert part_id(P("body_text", "x")) == "body_text"


# ---------------------------------------------------------------------------
# build_concatenated_context
# ---------------------------------------------------------------------------

def test_context_markers_present_in_order():
    ext = make_extraction(
        parts=[
            P("subject", "Urgent: verify"),
            P("body_text", "Hello, click the link."),
            P("image", "Verify now", filename="shot.png"),
            P("pdf_text", "Invoice total $10", filename="inv.pdf", page=1),
            P("qr", "http://secure-login-example.com/qr-verify",
              qr_urls=["http://secure-login-example.com/qr-verify"]),
        ],
        body_text="Hello, click the link.",
    )
    ctx = build_concatenated_context(ext)
    for marker in ("[SUBJECT]", "[BODY]", "[OCR]", "[PDF]", "[QR_URL]"):
        assert marker in ctx
    assert ctx.index("[SUBJECT]") < ctx.index("[BODY]") < ctx.index("[OCR]")
    assert ctx.index("[OCR]") < ctx.index("[PDF]") < ctx.index("[QR_URL]")
    assert "Urgent: verify" in ctx and "Verify now" in ctx
    assert "http://secure-login-example.com/qr-verify" in ctx


def test_context_body_text_fallback_when_no_body_part():
    ext = make_extraction(body_text="fallback body words")
    assert "[BODY] fallback body words" in build_concatenated_context(ext)


def test_context_empty_extraction_is_empty():
    assert build_concatenated_context(make_extraction()) == ""


def test_context_char_cap_truncation_without_tokenizer():
    long_body = "word " * 2000  # 10,000 chars
    ext = make_extraction(parts=[P("body_text", long_body)])
    ctx = build_concatenated_context(ext, max_length=128)
    assert len(ctx) <= 128 * 4
    assert ctx.startswith("[BODY] word")


class FakeTokenizer:
    """Mimics truncate_text_to_model_window's tokenizer call: keeps 10 chars."""

    def __call__(self, text, max_length=None, truncation=None, return_offsets_mapping=None):
        offsets = [(i, i + 1) for i in range(10)] + [(0, 0)] * 5
        return {"offset_mapping": offsets}


def test_context_tokenizer_truncation_applied():
    ext = make_extraction(parts=[P("body_text", "abcdefghijklmnopqrstuvwxyz")])
    ctx = build_concatenated_context(ext, tokenizer=FakeTokenizer(), max_length=512)
    assert ctx == "[BODY] abc"  # first 10 chars of "[BODY] abcdef..."


# ---------------------------------------------------------------------------
# Weighted fusion arithmetic
# ---------------------------------------------------------------------------

def test_weighted_fusion_arithmetic_known_weights():
    cfg = FusionConfig(w_context=0.5, w_max_part=0.25, w_url=0.15, w_flags=0.10,
                       threshold=0.5)
    ext = make_extraction(
        parts=[P("body_text", "click here to verify your account")],
        links=[{"url": "http://192.0.2.10/login", "source": "html_text"}],
        # flags summing to 0.50: anchor 0.15 + double ext 0.20 + hidden 0.10 + form 0.05
        flags=[
            {"type": "anchor_href_mismatch", "anchor": "paypal.com",
             "href": "http://192.0.2.10/login"},
            {"type": "double_extension", "member": "invoice.pdf.exe"},
            {"type": "hidden_text", "text": "secret"},
            {"type": "form_action", "action": "http://192.0.2.10/submit"},
        ],
    )
    # IP host (0.30) + http (0.10) = 0.40 for http://192.0.2.10/login.
    res = fuse_email(
        ext, {"body_text": 0.6}, fusion="weighted", config=cfg,
        concatenated_score=0.8,
    )
    # 0.5*0.8 + 0.25*0.6 + 0.15*0.40 + 0.10*0.50 = 0.40+0.15+0.06+0.05 = 0.66
    assert res["final_score"] == pytest.approx(0.66)
    assert res["context_score"] == pytest.approx(0.8)
    assert res["max_part_score"] == pytest.approx(0.6)
    assert res["url_score"] == pytest.approx(0.40)
    assert res["flag_score"] == pytest.approx(0.50)
    assert res["final_class"] == 1
    assert res["corroborated"] is True  # context + part both >= 0.5


def test_weighted_fusion_context_fallback_is_max_part():
    ext = make_extraction(parts=[P("body_text", "hello")])
    res = fuse_email(ext, {"body_text": 0.4}, fusion="weighted")
    assert res["context_score_source"] == "fallback_max_part"
    assert res["context_score"] == pytest.approx(0.4)
    # 0.45*0.4 + 0.20*0.4 = 0.26 < 0.5 -> Safe
    assert res["final_score"] == pytest.approx(0.26)
    assert res["final_class"] == 0


# ---------------------------------------------------------------------------
# Max rule baseline
# ---------------------------------------------------------------------------

def test_max_rule_baseline_fires_on_any_part():
    ext = make_extraction(parts=[P("body_text", "hello"), P("image", "verify now")])
    res = fuse_email(ext, {"body_text": 0.2, "image": 0.9}, fusion="max")
    assert res["final_score"] == pytest.approx(0.9)
    assert res["final_class"] == 1
    assert res["winning_part"] == "image"
    assert res["most_contributing_modality"] == "ocr"


def test_max_rule_ignores_url_and_flags():
    ext = make_extraction(
        parts=[P("body_text", "hello")],
        links=[{"url": "http://192.0.2.10/login", "source": "html_text"}],
        flags=[{"type": "anchor_href_mismatch", "anchor": "a", "href": "b"}],
    )
    res = fuse_email(ext, {"body_text": 0.1}, fusion="max")
    assert res["final_class"] == 0  # URL/flags cannot flip the max rule
    assert res["url_score"] > 0  # ...but they are still computed and reported


def test_unknown_fusion_raises():
    with pytest.raises(ValueError):
        fuse_email(make_extraction(), {}, fusion="average")


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------

def test_threshold_flips_weighted_verdict():
    ext = make_extraction(parts=[P("body_text", "verify your account now")])
    scores = {"body_text": 0.9}
    low = fuse_email(ext, scores, fusion="weighted", concatenated_score=0.9,
                     threshold=0.5)
    high = fuse_email(ext, scores, fusion="weighted", concatenated_score=0.9,
                      threshold=0.95)
    assert low["final_class"] == 1
    assert high["final_class"] == 0
    assert low["threshold"] == pytest.approx(0.5)
    assert high["threshold"] == pytest.approx(0.95)


def test_threshold_flips_max_verdict():
    ext = make_extraction(parts=[P("body_text", "hello")])
    assert fuse_email(ext, {"body_text": 0.6}, fusion="max", threshold=0.5)["final_class"] == 1
    assert fuse_email(ext, {"body_text": 0.6}, fusion="max", threshold=0.7)["final_class"] == 0


# ---------------------------------------------------------------------------
# Precision-aware behaviour: the documented weighted-vs-max difference
# ---------------------------------------------------------------------------

def test_single_weak_part_high_alone_is_safe_weighted_but_phishing_max():
    # One part scores high (0.85), everything else is low, no URL, no flags.
    ext = make_extraction(parts=[P("image", "verify now"), P("body_text", "hello")])
    scores = {"image": 0.85, "body_text": 0.2}
    weighted = fuse_email(ext, scores, fusion="weighted", concatenated_score=0.30)
    maximum = fuse_email(ext, scores, fusion="max", concatenated_score=0.30)
    assert maximum["final_class"] == 1  # max rule fires on the single part
    assert weighted["final_class"] == 0  # weighted needs corroboration
    # 0.45*0.30 + 0.20*0.85 = 0.135 + 0.17 = 0.305
    assert weighted["final_score"] == pytest.approx(0.305)
    assert weighted["corroborated"] is False


def test_strong_corroboration_flips_weighted():
    ext = make_extraction(
        parts=[P("image", "verify now"), P("body_text", "verify your account")],
        links=[{"url": "http://192.0.2.10/login", "source": "html_text"}],
    )
    scores = {"image": 0.85, "body_text": 0.8}
    res = fuse_email(ext, scores, fusion="weighted", concatenated_score=0.85)
    assert res["final_class"] == 1
    assert res["corroborated"] is True


def test_empty_email_is_safe_under_both():
    ext = make_extraction()
    for fusion in ("weighted", "max"):
        res = fuse_email(ext, {}, fusion=fusion)
        assert res["final_class"] == 0
        assert res["final_score"] == pytest.approx(0.0)
        assert res["top_reasons"]


# ---------------------------------------------------------------------------
# URL contribution
# ---------------------------------------------------------------------------

def test_url_score_contribution_benign_vs_malicious():
    benign = make_extraction(
        parts=[P("body_text", "see https://www.google.com/search?q=test")],
        links=[{"url": "https://www.google.com/search?q=test", "source": "html_text"}],
    )
    nasty = make_extraction(
        parts=[P("body_text", "click http://192.0.2.10/login")],
        links=[{"url": "http://192.0.2.10/login", "source": "html_text"}],
    )
    b = score_urls_for_email(benign)
    n = score_urls_for_email(nasty)
    assert b["url_score"] == pytest.approx(0.0)
    # Section C's extract_urls returns both the full URL and its schemeless
    # www. form, so the benign URL is counted twice; both score 0.0.
    assert b["n_urls"] == 2
    assert all(u["url_score"] == pytest.approx(0.0) for u in b["urls"])
    assert n["url_score"] == pytest.approx(0.40)  # IP 0.30 + http 0.10
    assert n["max_url"]["url"] == "http://192.0.2.10/login"


def test_url_score_pdf_links_not_treated_as_qr():
    # Section B stores PDF hyperlinks in the qr_urls slot of pdf_text parts;
    # they are plain links, so no QR-only bump (+0.10) applies.
    ext = make_extraction(parts=[P("pdf_text", "Pay at https://billing-example.com/pay",
                                   filename="inv.pdf", page=1,
                                   qr_urls=["https://billing-example.com/pay"])])
    res = score_urls_for_email(ext)
    assert res["url_score"] == pytest.approx(0.0)
    assert res["max_url"]["signals"]["qr_only"] is False


def test_url_score_qr_bump():
    ext = make_extraction(parts=[P("qr", "https://bit.ly/abc123",
                                   qr_urls=["https://bit.ly/abc123"])])
    res = score_urls_for_email(ext)
    # shortener 0.25 + QR-only 0.10 = 0.35
    assert res["url_score"] == pytest.approx(0.35)
    assert res["max_url"]["signals"]["qr_only"] is True


def test_url_score_no_urls_is_zero():
    res = score_urls_for_email(make_extraction(parts=[P("body_text", "no links here")]))
    assert res == {"urls": [], "url_score": 0.0, "max_url": None, "n_urls": 0, "reasons": []}


# ---------------------------------------------------------------------------
# Flag contributions
# ---------------------------------------------------------------------------

def test_flag_contributions_known_types():
    ext = make_extraction(flags=[
        {"type": "anchor_href_mismatch", "anchor": "paypal.com", "href": "http://evil.example/"},
        {"type": "double_extension", "member": "invoice.pdf.exe"},
    ])
    res = score_flags(ext)
    assert res["flag_score"] == pytest.approx(0.35)  # 0.15 + 0.20
    assert len(res["contributions"]) == 2
    assert res["reasons"]


def test_flag_auth_failures_and_reply_to_mismatch():
    ext = make_extraction(headers={
        "from": "Support <support@paypal.com>",
        "reply_to": "help@evil-example.com",
        "spf": "fail", "dkim": "pass", "dmarc": "fail",
    })
    res = score_flags(ext)
    # spf fail 0.10 + dmarc fail 0.10 + reply-to mismatch 0.10 = 0.30
    assert res["flag_score"] == pytest.approx(0.30)
    types = {c["type"] for c in res["contributions"]}
    assert {"auth_spf_fail", "auth_dmarc_fail", "reply_to_mismatch"} <= types


def test_flag_auth_pass_and_matching_reply_to_score_zero():
    ext = make_extraction(headers={
        "from": "Support <support@paypal.com>",
        "reply_to": "help@paypal.com",
        "spf": "pass", "dkim": "pass", "dmarc": "pass",
    })
    assert score_flags(ext)["flag_score"] == pytest.approx(0.0)


def test_flag_unknown_type_small_contribution():
    res = score_flags(make_extraction(flags=[{"type": "something_new"}]))
    assert res["flag_score"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Reasons + most-contributing modality
# ---------------------------------------------------------------------------

def test_top_reasons_non_empty_and_human_readable():
    ext = make_extraction(
        parts=[P("body_text", "verify now")],
        links=[{"url": "http://192.0.2.10/login", "source": "html_text"}],
        flags=[{"type": "anchor_href_mismatch", "anchor": "a", "href": "b"}],
    )
    res = fuse_email(ext, {"body_text": 0.7}, fusion="weighted",
                     concatenated_score=0.7)
    assert res["top_reasons"]
    assert len(res["top_reasons"]) <= 5
    joined = " ".join(res["top_reasons"])
    assert "192.0.2.10" in joined
    assert "anchor_href_mismatch" in joined


def test_most_contributing_modality_qr():
    ext = make_extraction(parts=[P("qr", "https://bit.ly/abc123",
                                   qr_urls=["https://bit.ly/abc123"])])
    res = fuse_email(ext, {"qr": 0.9}, fusion="weighted", concatenated_score=0.9)
    assert res["most_contributing_modality"] == "qr"


def test_most_contributing_modality_url_when_url_dominates():
    ext = make_extraction(
        parts=[P("body_text", "hello")],
        links=[{"url": "http://192.0.2.10/login", "source": "html_text"}],
    )
    # part/context tiny, URL 0.40 * 0.20 = 0.08 beats part 0.10*0.20 + ctx 0.10*0.45
    res = fuse_email(ext, {"body_text": 0.1}, fusion="weighted",
                     concatenated_score=0.1)
    assert res["most_contributing_modality"] == "url"


def test_most_contributing_modality_headers_when_only_flags():
    ext = make_extraction(
        parts=[P("body_text", "hello")],
        flags=[{"type": "double_extension", "member": "invoice.pdf.exe"},
               {"type": "suspicious_zip_member", "member": "run.exe"}],
    )
    res = fuse_email(ext, {"body_text": 0.0}, fusion="weighted",
                     concatenated_score=0.0)
    assert res["most_contributing_modality"] == "headers"
    assert res["flag_score"] == pytest.approx(0.35)


def test_most_contributing_modality_pdf():
    ext = make_extraction(parts=[P("pdf_text", "Invoice total", filename="inv.pdf", page=1)])
    res = fuse_email(ext, {"pdf_text:inv.pdf#p1": 0.9}, fusion="weighted",
                     concatenated_score=0.9)
    assert res["most_contributing_modality"] == "pdf"
    assert res["winning_part"] == "pdf_text:inv.pdf#p1"


def test_part_scores_matched_by_kind_fallback():
    ext = make_extraction(parts=[P("body_text", "hello"), P("subject", "hi")])
    res = fuse_email(ext, {"body_text": 0.3, "subject": 0.8}, fusion="max")
    assert res["winning_part"] == "subject"
    assert res["part_scores"]["subject"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Overlays (D3)
# ---------------------------------------------------------------------------

def test_qr_overlay_draws_and_annotates(tmp_path):
    from PIL import Image

    img_path = tmp_path / "qr.png"
    Image.new("RGB", (200, 200), "white").save(img_path)
    out = tmp_path / "qr_explained.png"
    result = overlay_qr_image(img_path, ["https://example.com/verify"], out)
    assert result is not None and Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_qr_overlay_skips_gracefully_without_pillow(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    monkeypatch.setitem(sys.modules, "PIL.ImageDraw", None)
    result = overlay_qr_image(b"not-an-image", ["https://example.com/"], "/tmp/x.png")
    assert result is None


def test_scanned_pdf_overlay_renders_page(tmp_path):
    import fitz

    pdf_path = tmp_path / "scan.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((50, 100), "URGENT WIRE TRANSFER", fontsize=20)
    doc.save(pdf_path)
    doc.close()
    boxes = [{"text": "URGENT", "left": 50, "top": 80, "width": 80, "height": 25, "conf": 95.0}]
    out = tmp_path / "page_explained.png"
    result = overlay_scanned_pdf_page(
        pdf_path.read_bytes(), 1, boxes,
        [{"word": "URGENT", "lita_score": 0.9}], out,
    )
    assert result is not None and Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_scanned_pdf_overlay_skips_gracefully_without_fitz(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "fitz", None)
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    result = overlay_scanned_pdf_page(b"%PDF-1.4 fake", 1, [], [], "/tmp/x.png")
    assert result is None


def test_scanned_pdf_overlay_page_out_of_range(tmp_path):
    import fitz

    doc = fitz.open()
    doc.new_page(width=200, height=200)
    pdf_bytes = doc.tobytes()
    doc.close()
    assert overlay_scanned_pdf_page(pdf_bytes, 99, [], [], tmp_path / "o.png") is None


# ---------------------------------------------------------------------------
# Predict script: untrained_fallback passthrough + CLI flags (D4/D5)
# ---------------------------------------------------------------------------

def test_predict_main_untrained_fallback_flag_and_cli(monkeypatch, tmp_path):
    """main() with --allow_pretrained_fallback writes untrained_fallback=true
    into every output JSON row, without loading any model (monkeypatched)."""
    import json

    import scripts.run_multimodal_predict as predict_mod

    eml = tmp_path / "sample.eml"
    eml.write_text(
        "From: a@example.com\nSubject: hi\n\nhello world\n", encoding="utf-8"
    )
    out_json = tmp_path / "out.json"

    monkeypatch.setattr(
        predict_mod, "load_trained_roberta",
        lambda ckpt, **kw: (None, None, None),
    )
    fake_result = {
        "file": "sample.eml", "parts": [], "final_class": 0,
        "final_verdict": "Safe Email", "winning_part": "body_text",
        "winning_prob_phishing": 0.1, "top_5_lita_words": [],
        "explained_image": None,
    }
    monkeypatch.setattr(
        predict_mod, "predict_eml",
        lambda **kw: dict(fake_result),
    )
    rc = predict_mod.main([
        str(tmp_path),
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--output_dir", str(tmp_path / "results"),
        "--max_samples", "1",
        "--fusion", "weighted",
        "--threshold", "0.5",
        "--allow_pretrained_fallback",
        "--output_json", str(out_json),
    ])
    assert rc == 0
    rows = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["untrained_fallback"] is True


def test_predict_main_missing_checkpoint_without_fallback_errors(tmp_path):
    import scripts.run_multimodal_predict as predict_mod

    rc = predict_mod.main([
        str(tmp_path),
        "--checkpoint", str(tmp_path / "missing.pt"),
    ])
    assert rc == 2
