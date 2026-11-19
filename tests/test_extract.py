"""Tests for multimodal extraction (src/phishing_xai/extract.py) and sample .eml generation."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phishing_xai.extract import extract_eml
from scripts.make_multimodal_samples import SAMPLES_DIR, generate_samples


@pytest.fixture(scope="module", autouse=True)
def ensure_multimodal_samples():
    """Ensure the 6 multimodal sample .eml files exist before running extraction tests."""
    paths = generate_samples(SAMPLES_DIR)
    assert len(paths) == 6
    return paths


def test_sample1_image_only_ocr_and_boxes():
    """(a) Image-only phishing email has empty body_text and OCRs keywords + bounding boxes."""
    res = extract_eml(SAMPLES_DIR / "sample1_image_only_phish.eml")
    assert res["body_text"] == ""
    img_parts = [p for p in res["parts"] if p["kind"] == "image"]
    assert len(img_parts) == 1
    ocr_text = img_parts[0]["text"].lower()
    assert "account" in ocr_text
    assert "closed" in ocr_text
    assert "verify" in ocr_text
    assert "secure-login-example.com" in ocr_text
    assert len(img_parts[0]["boxes"]) >= 5
    first_box = img_parts[0]["boxes"][0]
    for key in ("text", "left", "top", "width", "height", "conf"):
        assert key in first_box


def test_sample2_text_and_inline_button_ocr():
    """(b) Normal text email with inline 'Verify Account' button extracts both body and button OCR."""
    res = extract_eml(SAMPLES_DIR / "sample2_text_with_button.eml")
    assert "unusual sign-in activity" in res["body_text"].lower()
    img_parts = [p for p in res["parts"] if p["kind"] == "image"]
    assert len(img_parts) == 1
    ocr_text = img_parts[0]["text"].lower()
    assert "verify" in ocr_text
    assert "account" in ocr_text


def test_sample3_qr_code_decode():
    """(c) QR code email decodes the embedded phishing URL via pyzbar."""
    res = extract_eml(SAMPLES_DIR / "sample3_qr_phish.eml")
    qr_parts = [p for p in res["parts"] if p["kind"] == "qr"]
    assert len(qr_parts) == 1
    assert "http://secure-login-example.com/qr-verify" in qr_parts[0]["qr_urls"]
    assert "http://secure-login-example.com/qr-verify" in qr_parts[0]["text"]


def test_sample4_text_pdf_extraction_and_links():
    """(d) Text PDF attachment extracts invoice text and billing URL via pdfplumber."""
    res = extract_eml(SAMPLES_DIR / "sample4_text_pdf_invoice.eml")
    pdf_parts = [p for p in res["parts"] if p["kind"] == "pdf_text"]
    assert len(pdf_parts) == 1
    assert "payment due" in pdf_parts[0]["text"].lower()
    assert "https://billing-example.com/pay" in pdf_parts[0]["text"]
    assert "https://billing-example.com/pay" in pdf_parts[0]["qr_urls"]
    assert pdf_parts[0]["page"] == 1


def test_sample5_scanned_pdf_ocr_fallback():
    """(e) Scanned image-only PDF has no text layer and falls back to PyMuPDF render + OCR."""
    res = extract_eml(SAMPLES_DIR / "sample5_scanned_pdf.eml")
    ocr_pdf_parts = [p for p in res["parts"] if p["kind"] == "pdf_ocr"]
    assert len(ocr_pdf_parts) == 1
    text_lower = ocr_pdf_parts[0]["text"].lower()
    assert "urgent" in text_lower or "transfer" in text_lower
    assert "account" in text_lower
    assert len(ocr_pdf_parts[0]["boxes"]) > 0
    assert ocr_pdf_parts[0]["page"] == 1


def test_sample6_clean_benign_email():
    """(f) Clean email parses body text, harmless image OCR, and harmless text PDF."""
    res = extract_eml(SAMPLES_DIR / "sample6_clean_benign.eml")
    assert "sprint planning" in res["body_text"].lower()
    kinds = [p["kind"] for p in res["parts"]]
    assert "body_text" in kinds
    assert "image" in kinds
    assert "pdf_text" in kinds


def test_extract_cli_entrypoint():
    """CLI `python -m phishing_xai.extract <file.eml>` outputs valid JSON."""
    sample_path = SAMPLES_DIR / "sample1_image_only_phish.eml"
    proc = subprocess.run(
        [sys.executable, "-m", "phishing_xai.extract", str(sample_path), "--no_boxes"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"body_text"' in proc.stdout
    assert '"account_notice.png"' in proc.stdout


def test_draw_lita_boxes_on_sample1_image(tmp_path):
    """Step 5: draw_lita_boxes_on_image draws red bounding boxes around top LITA words on sample 1."""
    from PIL import Image
    import numpy as np
    from phishing_xai.explain import draw_lita_boxes_on_image

    res = extract_eml(SAMPLES_DIR / "sample1_image_only_phish.eml")
    img_part = [p for p in res["parts"] if p["kind"] == "image"][0]
    src_img_path = SAMPLES_DIR / "sample1_body.png"
    out_img_path = tmp_path / "sample1_body_explained.png"

    saved_path = draw_lita_boxes_on_image(
        image_input=src_img_path,
        ocr_boxes=img_part["boxes"],
        lita_top_words=[("Verify", 0.95), ("account", 0.88), ("closed", 0.76)],
        output_path=out_img_path,
        top_k=3,
    )
    assert saved_path.exists()
    orig_arr = np.asarray(Image.open(src_img_path).convert("RGB"))
    annotated_arr = np.asarray(Image.open(saved_path).convert("RGB"))
    # Red pixels (R > 200, G < 50, B < 50) must be present in the annotated image
    red_mask = (annotated_arr[:, :, 0] > 200) & (annotated_arr[:, :, 1] < 50) & (annotated_arr[:, :, 2] < 50)
    assert int(red_mask.sum()) > 50
    assert not np.array_equal(orig_arr, annotated_arr)



# ---------------------------------------------------------------------------
# Section B tests (hermetic: no tesseract / zbar / GPU / internet / checkpoint)
# ---------------------------------------------------------------------------
import base64
import io
import zipfile
from email.message import EmailMessage

import phishing_xai.extract as extract_mod
from phishing_xai.extract import _parse_html, decode_qr, ocr_image


def _eml_with_html(tmp_path, name, html, subject="HTML test"):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "sender@example.com"
    msg["To"] = "user@example.org"
    msg.set_content("Plain fallback body.")
    msg.add_alternative(html, subtype="html")
    path = tmp_path / name
    path.write_bytes(msg.as_bytes())
    return path


def test_b1_qr_in_text_pdf_embedded_image(tmp_path, monkeypatch):
    """B1: a QR code embedded as an image in a TEXT pdf is decoded (not only scanned pages)."""
    import fitz
    from scripts.make_multimodal_samples import make_qr_png

    monkeypatch.setattr(extract_mod, "ocr_image",
                        lambda image: {"text": "QR CODE HERE", "boxes": []})
    monkeypatch.setattr(extract_mod, "decode_qr",
                        lambda image: ["http://qr-in-pdf-example.com/x"])
    doc = fitz.open()
    page = doc.new_page(width=595, height=400)
    page.insert_text((50, 72), "INVOICE #INV-1", fontsize=16)
    page.insert_text((50, 100), "Please scan the code below.", fontsize=16)
    page.insert_image(fitz.Rect(50, 130, 250, 330), stream=make_qr_png("http://qr-in-pdf-example.com/x"))
    pdf_bytes = doc.tobytes()
    doc.close()
    msg = EmailMessage()
    msg["Subject"] = "Invoice with QR"
    msg["From"] = "billing@example.com"
    msg["To"] = "user@example.org"
    msg.set_content("See attached invoice.")
    msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename="invoice_qr.pdf")
    path = tmp_path / "qr_text_pdf.eml"
    path.write_bytes(msg.as_bytes())

    res = extract_eml(path)
    assert any(p["kind"] == "pdf_text" for p in res["parts"])
    qr_parts = [p for p in res["parts"] if p["kind"] == "qr"]
    assert qr_parts and "http://qr-in-pdf-example.com/x" in qr_parts[0]["qr_urls"]
    img_parts = [p for p in res["parts"] if p["kind"] == "pdf_image"]
    assert len(img_parts) == 1
    assert img_parts[0]["page"] == 1


def test_b2_html_anchor_href_mismatch(tmp_path):
    """B2: anchor text showing one domain while href points elsewhere is flagged."""
    html = (
        '<html><body><p>Please verify:</p>'
        '<a href="http://evil-example.com/login">https://mybank-example.com/login</a>'
        '<a href="https://mybank-example.com/help">https://mybank-example.com/help</a>'
        "</body></html>"
    )
    res = extract_eml(_eml_with_html(tmp_path, "mismatch.eml", html))
    mismatches = [f for f in res["flags"] if f.get("type") == "anchor_href_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0]["href"] == "http://evil-example.com/login"
    assert any(link["url"] == "http://evil-example.com/login" for link in res["links"])


def test_b2_html_base64_image_part(tmp_path):
    """B2: a data: base64 image in HTML is decoded and yields an image part (OCR may be empty)."""
    from scripts.make_multimodal_samples import render_text_png

    png = render_text_png(["Hello World"], size=(500, 120), font_size=28)
    b64 = base64.b64encode(png).decode()
    html = f'<html><body><p>See image:</p><img src="data:image/png;base64,{b64}"></body></html>'
    res = extract_eml(_eml_with_html(tmp_path, "b64.eml", html))
    img_parts = [p for p in res["parts"]
                 if p["kind"] == "image" and (p["filename"] or "").startswith("html-embedded-image-")]
    assert len(img_parts) == 1


def test_b2_html_hidden_text_not_in_body(tmp_path):
    """B2: display:none text is kept as its own part/flag, never mixed into body_text."""
    html = (
        '<html><body><p>Visible hello.</p>'
        '<span style="display:none">secret hidden phrase</span>'
        '<span style="font-size:0">zero size phrase</span>'
        "</body></html>"
    )
    res = extract_eml(_eml_with_html(tmp_path, "hidden.eml", html))
    assert "secret hidden phrase" not in res["body_text"]
    assert "zero size phrase" not in res["body_text"]
    hidden = [p for p in res["parts"] if p["kind"] == "hidden_text"]
    assert len(hidden) == 1
    assert "secret hidden phrase" in hidden[0]["text"]
    assert any(f.get("type") == "hidden_text" for f in res["flags"])


def test_b2_html_form_action_and_missing_cid(tmp_path):
    """B2: <form action> is recorded; a cid: image with no matching part is flagged missing."""
    html = (
        '<html><body><form action="http://evil-example.com/submit">'
        '<input type="text"></form>'
        '<img src="cid:missing123@example">'
        "</body></html>"
    )
    res = extract_eml(_eml_with_html(tmp_path, "formcid.eml", html))
    assert any(f.get("type") == "form_action" for f in res["flags"])
    assert any(f.get("type") == "missing_cid_image" for f in res["flags"])
    assert any(link["url"] == "http://evil-example.com/submit" for link in res["links"])
    assert any(s["part"] == "cid:missing123@example" for s in res["skipped"])


def test_b3_headers_and_subject_part(tmp_path):
    """B3: headers are returned parsed; Subject is a separate part, body_text unchanged."""
    msg = EmailMessage()
    msg["Subject"] = "Urgent: verify now"
    msg["From"] = "spoofed@example.com"
    msg["Reply-To"] = "other@example.com"
    msg["Return-Path"] = "<bounce@example.com>"
    msg["Message-ID"] = "<abc123@mail.example.com>"
    msg["Authentication-Results"] = "mx.example.com; spf=pass smtp.mailfrom=example.com; dkim=fail header.d=example.com; dmarc=pass"
    msg["To"] = "user@example.org"
    msg.set_content("Body text here.")
    path = tmp_path / "headers.eml"
    path.write_bytes(msg.as_bytes())

    res = extract_eml(path)
    assert res["headers"]["subject"] == "Urgent: verify now"
    assert res["headers"]["from"] == "spoofed@example.com"
    assert res["headers"]["reply_to"] == "other@example.com"
    assert res["headers"]["message_id_domain"] == "mail.example.com"
    assert res["headers"]["spf"] == "pass"
    assert res["headers"]["dkim"] == "fail"
    assert res["headers"]["dmarc"] == "pass"
    subject_parts = [p for p in res["parts"] if p["kind"] == "subject"]
    assert len(subject_parts) == 1
    assert subject_parts[0]["text"] == "Urgent: verify now"
    assert res["body_text"] == "Body text here."


def test_b4_docx_attachment(tmp_path):
    """B4: a .docx attachment yields its text (and a URL in the text is linked)."""
    import docx

    document = docx.Document()
    document.add_paragraph("Please review the attached memo.")
    document.add_paragraph("Details at https://memo-example.com/info.")
    buf = io.BytesIO()
    document.save(buf)
    msg = EmailMessage()
    msg["Subject"] = "Memo"
    msg["From"] = "a@example.com"
    msg["To"] = "user@example.org"
    msg.set_content("See attached memo.")
    msg.add_attachment(buf.getvalue(),
                       maintype="application",
                       subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
                       filename="memo.docx")
    path = tmp_path / "docx.eml"
    path.write_bytes(msg.as_bytes())

    res = extract_eml(path)
    docx_parts = [p for p in res["parts"] if p["kind"] == "docx"]
    assert len(docx_parts) == 1
    assert "attached memo" in docx_parts[0]["text"]
    assert any(link["url"] == "https://memo-example.com/info" for link in res["links"])


def test_b4_nested_message_rfc822(tmp_path):
    """B4: a nested message/rfc822 is recursed into (depth-capped)."""
    inner = EmailMessage()
    inner["Subject"] = "Forwarded notice"
    inner["From"] = "inner@example.com"
    inner.set_content("inner secret phrase for recursion check")
    outer = EmailMessage()
    outer["Subject"] = "FW: notice"
    outer["From"] = "a@example.com"
    outer["To"] = "user@example.org"
    outer.set_content("Outer body.")
    # Standard construction: attach the Message object itself (as real mail
    # clients do). Attaching raw .eml bytes instead produces a degenerate
    # base64 blob that even the stdlib parser does not interpret as a message.
    outer.add_attachment(inner, filename="forwarded.eml")
    path = tmp_path / "nested.eml"
    path.write_bytes(outer.as_bytes())

    res = extract_eml(path)
    assert "inner secret phrase for recursion check" in res["body_text"]


def test_b4_zip_listing_and_extension_flags(tmp_path):
    """B4: a .zip lists member names only and flags risky / double extensions."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("invoice.pdf", "fake")
        zf.writestr("invoice.pdf.exe", "fake")
        zf.writestr("run.exe", "fake")
        zf.writestr("notes.txt", "fake")
    msg = EmailMessage()
    msg["Subject"] = "Files"
    msg["From"] = "a@example.com"
    msg["To"] = "user@example.org"
    msg.set_content("See attached archive.")
    msg.add_attachment(buf.getvalue(), maintype="application", subtype="zip", filename="files.zip")
    path = tmp_path / "zip.eml"
    path.write_bytes(msg.as_bytes())

    res = extract_eml(path)
    listing = [p for p in res["parts"] if p["kind"] == "zip_listing"]
    assert len(listing) == 1
    assert "invoice.pdf.exe" in listing[0]["text"]
    assert any(f.get("type") == "double_extension" and f.get("member") == "invoice.pdf.exe"
               for f in res["flags"])
    assert any(f.get("type") == "suspicious_zip_member" and f.get("member") == "run.exe"
               for f in res["flags"])


def test_b4_svg_attachment(tmp_path):
    """B4: an .svg attachment yields its text content; scripts are never run."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><text>Verify your account now</text>'
           '<script>alert(1)</script></svg>')
    msg = EmailMessage()
    msg["Subject"] = "Image"
    msg["From"] = "a@example.com"
    msg["To"] = "user@example.org"
    msg.set_content("See attached image.")
    msg.add_attachment(svg.encode(), maintype="image", subtype="svg+xml", filename="pic.svg")
    path = tmp_path / "svg.eml"
    path.write_bytes(msg.as_bytes())

    res = extract_eml(path)
    svg_parts = [p for p in res["parts"] if p["kind"] == "svg_text"]
    assert len(svg_parts) == 1
    assert "Verify your account now" in svg_parts[0]["text"]
    assert "alert" not in svg_parts[0]["text"]


def test_b5_oversize_attachment_skipped(tmp_path, monkeypatch):
    """B5: an attachment over MAX_ATTACHMENT_BYTES is skipped with a reason, not a crash."""
    from scripts.make_multimodal_samples import render_text_png

    monkeypatch.setattr(extract_mod, "MAX_ATTACHMENT_BYTES", 10)
    png = render_text_png(["Hello"], size=(500, 120), font_size=28)
    assert len(png) > 10
    msg = EmailMessage()
    msg["Subject"] = "Big image"
    msg["From"] = "a@example.com"
    msg["To"] = "user@example.org"
    msg.set_content("Body.")
    msg.add_attachment(png, maintype="image", subtype="png", filename="big.png")
    path = tmp_path / "big.eml"
    path.write_bytes(msg.as_bytes())

    res = extract_eml(path)
    assert not [p for p in res["parts"] if p["kind"] == "image"]
    assert any(s["part"] == "big.png" and "MAX_ATTACHMENT_BYTES" in s["reason"]
               for s in res["skipped"])


def test_b7_missing_tesseract_graceful_skip(monkeypatch):
    """B7: with no tesseract binary anywhere, OCR returns empty text/boxes, never raises."""
    import pytesseract
    from PIL import Image

    monkeypatch.setattr(extract_mod.shutil, "which", lambda *a, **k: None)
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "/nonexistent/tesseract")
    # Simulate "no binary found": _configure_tesseract must not resurrect one.
    monkeypatch.setattr(extract_mod, "_configure_tesseract",
                        lambda: setattr(pytesseract.pytesseract, "tesseract_cmd",
                                        "/nonexistent/tesseract"))
    result = ocr_image(Image.new("RGB", (100, 40), color="white"))
    assert result == {"text": "", "boxes": []}
    # Extraction of a real image-only email still completes.
    res = extract_eml(SAMPLES_DIR / "sample1_image_only_phish.eml")
    img_parts = [p for p in res["parts"] if p["kind"] == "image"]
    assert len(img_parts) == 1
    assert img_parts[0]["text"] == ""
    assert img_parts[0]["boxes"] == []
    assert "headers" in res and "links" in res and "flags" in res and "skipped" in res
