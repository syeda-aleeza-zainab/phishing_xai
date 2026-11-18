"""Generate 6 synthetic multimodal .eml sample files in samples/multimodal/.

Samples generated:
  1. sample1_image_only_phish.eml : Image-only phishing body ('Your account will be closed. Verify now at http://secure-login-example.com')
  2. sample2_text_with_button.eml : Normal text email with an inline 'Verify Account' button image
  3. sample3_qr_phish.eml         : Email with a QR code image encoding 'http://secure-qr-verify-example.com/login'
  4. sample4_text_pdf_invoice.eml : Email with a text PDF attachment ('Payment due. Click https://billing-example.com/pay')
  5. sample5_scanned_pdf.eml      : Email with a scanned image-only PDF attachment (triggers PyMuPDF render + OCR fallback)
  6. sample6_clean_benign.eml     : Clean benign email with a harmless image and harmless text PDF
"""

from email.message import EmailMessage
import io
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import qrcode


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples" / "multimodal"


def _load_font(size: int = 28) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("arial.ttf", "calibri.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_text_png(
    lines: List[str],
    size: Tuple[int, int] = (1000, 260),
    bg_color: str = "white",
    fg_color: str = "black",
    font_size: int = 28,
) -> bytes:
    """Render lines of text into a clear, high-contrast PNG image for reliable OCR."""
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    y = 40
    for line in lines:
        draw.text((40, y), line, fill=fg_color, font=font)
        y += font_size + 24
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_text_pdf(text_lines: List[str], link_url: str | None = None) -> bytes:
    """Create a real text PDF using PyMuPDF (fitz), optionally adding a clickable URI link."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=400)
    y = 72
    for line in text_lines:
        page.insert_text((50, y), line, fontsize=16)
        y += 28
    if link_url:
        link_rect = fitz.Rect(50, 50, 500, y + 20)
        page.insert_link({"kind": fitz.LINK_URI, "from": link_rect, "uri": link_url})
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_scanned_image_pdf(lines: List[str]) -> bytes:
    """Create an image-only (scanned) PDF with zero extractable text streams."""
    png_bytes = render_text_png(lines, size=(1100, 320), font_size=30)
    doc = fitz.open()
    page = doc.new_page(width=600, height=250)
    page.insert_image(fitz.Rect(20, 20, 580, 220), stream=png_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_qr_png(url: str) -> bytes:
    """Create a PNG image containing a QR code for the given URL."""
    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_samples(out_dir: Path = SAMPLES_DIR) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # (a) Sample 1: Image-only phishing body
    msg1 = EmailMessage()
    msg1["Subject"] = "Account Notice"
    msg1["From"] = "security@alerts-example.com"
    msg1["To"] = "user@example.org"
    png1 = render_text_png([
        "Your account will be closed.",
        "Verify now at http://secure-login-example.com",
    ])
    msg1.add_attachment(png1, maintype="image", subtype="png", filename="account_notice.png")
    p1 = out_dir / "sample1_image_only_phish.eml"
    p1.write_bytes(msg1.as_bytes())
    written.append(p1)
    # Also save the raw PNG for visual explanation testing
    (out_dir / "sample1_body.png").write_bytes(png1)

    # (b) Sample 2: Normal text email with an inline 'Verify Account' button image
    msg2 = EmailMessage()
    msg2["Subject"] = "Security Check Required"
    msg2["From"] = "support@portal-example.com"
    msg2["To"] = "user@example.org"
    msg2.set_content(
        "Hello,\n\nWe noticed unusual sign-in activity on your profile. "
        "Please use the button below to confirm your identity immediately."
    )
    png2 = render_text_png(
        ["Verify Account"],
        size=(500, 140),
        bg_color="white",
        fg_color="black",
        font_size=36,
    )
    msg2.add_attachment(
        png2,
        maintype="image",
        subtype="png",
        filename="verify_button.png",
        disposition="inline",
    )
    p2 = out_dir / "sample2_text_with_button.eml"
    p2.write_bytes(msg2.as_bytes())
    written.append(p2)

    # (c) Sample 3: Email with a QR code image
    msg3 = EmailMessage()
    msg3["Subject"] = "Two-Factor Authentication Re-enrollment"
    msg3["From"] = "it-helpdesk@corp-auth-example.com"
    msg3["To"] = "user@example.org"
    msg3.set_content("Scan the attached QR code with your phone to keep your account active.")
    qr_png = make_qr_png("http://secure-login-example.com/qr-verify")
    msg3.add_attachment(qr_png, maintype="image", subtype="png", filename="auth_qr.png")
    p3 = out_dir / "sample3_qr_phish.eml"
    p3.write_bytes(msg3.as_bytes())
    written.append(p3)

    # (d) Sample 4: Email with a text PDF attachment (fake invoice)
    msg4 = EmailMessage()
    msg4["Subject"] = "Invoice #INV-90821 Due Today"
    msg4["From"] = "billing@billing-example.com"
    msg4["To"] = "user@example.org"
    msg4.set_content("Please find attached your monthly invoice.")
    pdf4 = make_text_pdf(
        [
            "INVOICE #INV-90821",
            "Payment due. Click https://billing-example.com/pay",
            "Failure to pay immediately will suspend your account.",
        ],
        link_url="https://billing-example.com/pay",
    )
    msg4.add_attachment(pdf4, maintype="application", subtype="pdf", filename="invoice.pdf")
    p4 = out_dir / "sample4_text_pdf_invoice.eml"
    p4.write_bytes(msg4.as_bytes())
    written.append(p4)

    # (e) Sample 5: Email with a scanned image-only PDF attachment
    msg5 = EmailMessage()
    msg5["Subject"] = "Scanned Document from Accounting"
    msg5["From"] = "scanner@office-example.com"
    msg5["To"] = "user@example.org"
    msg5.set_content("Attached is the scanned notice from the remittance department.")
    pdf5 = make_scanned_image_pdf([
        "URGENT WIRE TRANSFER REQUIRED",
        "Your account will be suspended unless payment is confirmed.",
        "Login at http://scanned-billing-example.com",
    ])
    msg5.add_attachment(pdf5, maintype="application", subtype="pdf", filename="scanned_notice.pdf")
    p5 = out_dir / "sample5_scanned_pdf.eml"
    p5.write_bytes(msg5.as_bytes())
    written.append(p5)

    # (f) Sample 6: Clean benign email with a harmless image and PDF
    msg6 = EmailMessage()
    msg6["Subject"] = "Weekly Team Lunch & Project Notes"
    msg6["From"] = "colleague@company.org"
    msg6["To"] = "team@company.org"
    msg6.set_content(
        "Hi everyone,\n\nThanks for joining the sprint planning session earlier today. "
        "Attached are the workshop agenda banner and the quarterly roadmap notes for next Tuesday."
    )
    png6 = render_text_png(["Engineering Workshop Agenda Tuesday Morning"], size=(900, 160), font_size=28)
    pdf6 = make_text_pdf([
        "Quarterly Engineering Roadmap Notes",
        "All sprint milestones are on track for the release next month.",
    ])
    msg6.add_attachment(png6, maintype="image", subtype="png", filename="agenda_banner.png")
    msg6.add_attachment(pdf6, maintype="application", subtype="pdf", filename="roadmap_notes.pdf")
    p6 = out_dir / "sample6_clean_benign.eml"
    p6.write_bytes(msg6.as_bytes())
    written.append(p6)

    return written


if __name__ == "__main__":
    paths = generate_samples()
    for p in paths:
        print(f"Generated: {p.relative_to(REPO_ROOT)}")
