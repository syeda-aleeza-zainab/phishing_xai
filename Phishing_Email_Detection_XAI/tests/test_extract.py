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

