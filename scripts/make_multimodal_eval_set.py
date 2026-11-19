"""Generate a labelled SYNTHETIC multimodal evaluation set (Section E).

Deterministically (seeded, ``--seed`` default 42) writes at least 120
``.eml`` files (60 phishing / 60 benign) plus ``labels.csv`` and a
``manifest.json`` recording the seed and the per-category counts.

Output layout (default ``data/multimodal_eval/`` — small text+image
``.eml`` files, kept in the repo next to the existing ``samples/`` tree;
the directory is documented in ``.impl_logs/section_E.md``)::

    data/multimodal_eval/
        <category>_<idx:03d>.eml   (e.g. qr_body_image_003.eml)
        labels.csv                 (filename, label, category, split)
        manifest.json              (seed, counts, split sizes, caveat)

``labels.csv`` columns: ``filename`` (the .eml file name), ``label``
(1 = phishing, 0 = benign), ``category`` (one of CATEGORIES below),
``split`` (``validation`` — the small slice thresholds are fitted on —
or ``test`` — the slice reported on; thresholds are NEVER fitted on
test). The split is stratified by label (12 phishing + 12 benign in
validation, 48 + 48 in test) and deterministic for a given seed.

Categories (8 phishing x 60, 3 benign x 60):

Phishing:
    image_only_phishing  (10)  body is an image only (OCR is the signal)
    text_button_image    (8)   plain body + an inline "Verify" button image
    qr_body_image        (8)   QR code image in the body (hides the URL)
    qr_in_text_pdf       (7)   QR code embedded in a text-layer PDF
    scanned_pdf          (7)   image-only (scanned) PDF attachment
    html_link_mismatch   (7)   HTML body, anchor text shows a trusted
                              domain while href points elsewhere
    html_attachment_phishing (6) phishing .html attachment
    fake_invoice_pdf     (7)   text PDF invoice with a payment link + urgency
Benign look-alikes:
    newsletter_logo_qr   (20)  newsletter with a logo/banner image and a
                              QR code for a (benign) event page
    real_invoice         (20)  genuine invoice PDF, https link, no urgency
    meeting_agenda       (20)  meeting agenda: body + banner image + PDF

OCR stressors (simulated, seeded): font family/size variation, low
contrast, small rotations, pixel noise, and JPEG compression on some
images. Nothing here is a real-world email: the set is SYNTHETIC —
generated images/PDFs from a small set of templates, no real adversary
variation — so results on it measure pipeline behaviour on these
templates only and do NOT generalise to real phishing. That caveat is
also written into the manifest and into every eval report built on it.

Determinism: the same seed produces the same ``labels.csv`` and the
same file list. ``.eml`` byte equality is NOT guaranteed across runs:
PyMuPDF embeds a fresh document ID in every PDF it writes, so PDF
bytes (and therefore .eml bytes) can differ even when every visible
word, image pixel and label is identical. Tests assert on
``labels.csv`` content and the file list, not on raw bytes.

No URL in any generated email is fetched or visited (static content
only); every URL uses an ``-example`` / ``.org`` placeholder domain.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import qrcode

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.make_multimodal_eval_set")

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "multimodal_eval"

#: category -> (label, count). Label 1 = phishing, 0 = benign.
CATEGORIES: Dict[str, Tuple[int, int]] = {
    # Phishing (60).
    "image_only_phishing": (1, 10),
    "text_button_image": (1, 8),
    "qr_body_image": (1, 8),
    "qr_in_text_pdf": (1, 7),
    "scanned_pdf": (1, 7),
    "html_link_mismatch": (1, 7),
    "html_attachment_phishing": (1, 6),
    "fake_invoice_pdf": (1, 7),
    # Benign look-alikes (60).
    "newsletter_logo_qr": (0, 20),
    "real_invoice": (0, 20),
    "meeting_agenda": (0, 20),
}

#: Fraction of each label held out as the validation slice (thresholds
#: are fitted on validation only, never on test).
VALIDATION_FRACTION: float = 0.20

#: Fixed PDF metadata dates so the visible PDF content is stable; the
#: PDF byte stream itself still carries a fresh document ID per write
#: (see the module docstring — byte equality is not guaranteed).
_FIXED_PDF_METADATA: Dict[str, str] = {
    "creationDate": "D:20260101000000Z",
    "modDate": "D:20260101000000Z",
    "creator": "phishing_xai synthetic eval-set generator",
}

_FONT_PATHS: Tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

_PHISH_SENDERS: Tuple[str, ...] = (
    "security@alerts-example.com",
    "support@portal-example.com",
    "billing@billing-example.com",
    "it-helpdesk@corp-auth-example.com",
    "accounts@secure-notice-example.com",
)
_PHISH_REPLY_TO: Tuple[str, ...] = (
    "verify@collect-example.net",
    "support@account-check-example.tk",
    "helpdesk@login-verify-example.xyz",
)
_BENIGN_SENDERS: Tuple[str, ...] = (
    "newsletter@company.org",
    "billing@company.org",
    "team@company.org",
    "events@company.org",
    "hr@company.org",
)

_PHISH_URLS: Tuple[str, ...] = (
    "http://secure-login-example.com/verify",
    "http://paypal-secure-verify-example.tk/login",
    "http://account-check-example.xyz/signin",
    "http://secure-login-example.com/qr-verify",
    "http://billing-example.com/pay-now",
    "http://scanned-billing-example.com/confirm",
)
_BENIGN_URLS: Tuple[str, ...] = (
    "https://events.company.org/spring-fair",
    "https://portal.company.org/invoices",
    "https://intranet.company.org/agenda",
    "https://billing.company.org/pay",
)

_COMPANIES: Tuple[str, ...] = (
    "Acme Corp", "Northwind Traders", "BluePeak Ltd", "Harbor & Co",
    "Summit Retail", "Cedar Logistics",
)


def _load_font(size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    """Pick a seeded font (family + size vary to stress OCR)."""
    path = rng.choice(_FONT_PATHS)
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()


def _apply_stressors(
    img: Image.Image, rng: random.Random
) -> Image.Image:
    """Apply seeded OCR stressors: rotation, noise, low contrast.

    Each stressor fires only sometimes, so the set mixes clean and
    stressed images instead of stressing everything at once.
    """
    if rng.random() < 0.35:
        angle = rng.choice((-4, -3, -2, 2, 3, 4))
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor="white")
    if rng.random() < 0.30:
        # Low contrast: blend the image towards mid-grey.
        grey = Image.new("RGB", img.size, (200, 200, 200))
        img = Image.blend(img, grey, rng.uniform(0.25, 0.45))
    if rng.random() < 0.30:
        # Sparse dark-pixel noise.
        pixels = img.load()
        width, height = img.size
        for _ in range(rng.randint(200, 800)):
            pixels[rng.randrange(width), rng.randrange(height)] = (0, 0, 0)
    return img


def render_text_image(
    lines: Sequence[str],
    rng: random.Random,
    *,
    size: Tuple[int, int] = (1000, 260),
    font_size: int = 28,
    jpeg: bool = False,
) -> bytes:
    """Render text lines into a PNG (or JPEG) image with seeded stressors."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    font = _load_font(rng.choice((font_size - 4, font_size, font_size + 4)), rng)
    fg = (0, 0, 0) if rng.random() >= 0.25 else (90, 90, 90)  # low contrast sometimes
    y = 40
    for line in lines:
        draw.text((40, y), line, fill=fg, font=font)
        y += font_size + 24
    img = _apply_stressors(img, rng)
    buf = io.BytesIO()
    if jpeg:
        img.save(buf, format="JPEG", quality=rng.randint(35, 70))
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


def make_qr_png(url: str) -> bytes:
    """Create a PNG image containing a QR code for the given URL."""
    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_text_pdf(
    text_lines: Sequence[str],
    *,
    link_url: Optional[str] = None,
    qr_png: Optional[bytes] = None,
) -> bytes:
    """Create a text-layer PDF, optionally with a link and an embedded QR image."""
    doc = fitz.open()
    doc.set_metadata(_FIXED_PDF_METADATA)
    page = doc.new_page(width=595, height=500)
    y = 72
    for line in text_lines:
        page.insert_text((50, y), line, fontsize=16)
        y += 28
    if link_url:
        page.insert_link(
            {"kind": fitz.LINK_URI, "from": fitz.Rect(50, 50, 500, y + 20), "uri": link_url}
        )
    if qr_png is not None:
        page.insert_text((50, y + 10), "Scan the code below to continue:", fontsize=14)
        page.insert_image(fitz.Rect(50, y + 30, 250, y + 230), stream=qr_png)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_scanned_pdf(lines: Sequence[str], rng: random.Random) -> bytes:
    """Create an image-only (scanned) PDF with zero extractable text."""
    png_bytes = render_text_image(lines, rng, size=(1100, 320), font_size=30)
    doc = fitz.open()
    doc.set_metadata(_FIXED_PDF_METADATA)
    page = doc.new_page(width=600, height=250)
    page.insert_image(fitz.Rect(20, 20, 580, 220), stream=png_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _base_message(
    *,
    subject: str,
    sender: str,
    seed: int,
    filename: str,
    reply_to: Optional[str] = None,
    auth_results: Optional[str] = None,
) -> EmailMessage:
    """Common headers for every generated email (deterministic IDs)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = "user@example.org"
    msg["Message-ID"] = f"<eval-{seed}-{filename}@example.org>"
    if reply_to:
        msg["Reply-To"] = reply_to
    if auth_results:
        msg["Authentication-Results"] = auth_results
    try:
        msg.set_boundary(f"==eval-{seed}-{filename}==")
    except Exception:
        pass
    return msg


def _phish_headers(rng: random.Random, seed: int, filename: str) -> Dict[str, Any]:
    """Spoofed-looking headers for phishing mail (mismatched Reply-To,
    often failing SPF/DKIM — the extractor turns these into flag scores)."""
    auth = (
        "spf=fail dkim=fail dmarc=fail"
        if rng.random() < 0.6
        else "spf=pass dkim=pass dmarc=pass"
    )
    return {
        "sender": rng.choice(_PHISH_SENDERS),
        "reply_to": rng.choice(_PHISH_REPLY_TO),
        "auth_results": auth,
    }


def _benign_headers(rng: random.Random) -> Dict[str, Any]:
    return {
        "sender": rng.choice(_BENIGN_SENDERS),
        "reply_to": None,
        "auth_results": "spf=pass dkim=pass dmarc=pass",
    }


def build_email(
    category: str, idx: int, rng: random.Random, *, seed: int
) -> Tuple[str, EmailMessage]:
    """Build one labelled email for ``category`` (1-based ``idx``).

    Returns ``(filename, message)``. The caller writes the bytes and
    records the label from :data:`CATEGORIES`.
    """
    filename = f"{category}_{idx:03d}.eml"
    company = rng.choice(_COMPANIES)
    phish_url = rng.choice(_PHISH_URLS)
    benign_url = rng.choice(_BENIGN_URLS)
    invoice_no = f"INV-{rng.randint(10000, 99999)}"
    amount = rng.randint(49, 499)

    if category == "image_only_phishing":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject=rng.choice(["Account Notice", "Urgent: Account Closure", "Verify Your Account"]),
            seed=seed, filename=filename, **hdr,
        )
        png = render_text_image(
            ["Your account will be closed.", f"Verify now at {phish_url}"],
            rng, jpeg=rng.random() < 0.3,
        )
        msg.add_attachment(png, maintype="image",
                           subtype="jpeg" if png[:3] == b"\xff\xd8\xff" else "png",
                           filename="account_notice.png")
        return filename, msg

    if category == "text_button_image":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject="Security Check Required", seed=seed, filename=filename, **hdr,
        )
        msg.set_content(
            "Hello,\n\nWe noticed unusual sign-in activity on your profile. "
            "Please use the button below to confirm your identity immediately. "
            f"Your reference is {invoice_no}."
        )
        png = render_text_image(["Verify Account"], rng, size=(500, 140), font_size=36)
        msg.add_attachment(png, maintype="image", subtype="png",
                           filename="verify_button.png", disposition="inline")
        return filename, msg

    if category == "qr_body_image":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject="Two-Factor Authentication Re-enrollment",
            seed=seed, filename=filename, **hdr,
        )
        msg.set_content(
            "Scan the attached QR code with your phone to keep your account active. "
            "This link expires in 24 hours."
        )
        msg.add_attachment(make_qr_png(phish_url), maintype="image", subtype="png",
                           filename="auth_qr.png")
        return filename, msg

    if category == "qr_in_text_pdf":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject=f"Payment Verification {invoice_no}", seed=seed, filename=filename, **hdr,
        )
        msg.set_content("Please see the attached document and follow the steps inside.")
        pdf = make_text_pdf(
            [f"PAYMENT VERIFICATION {invoice_no}",
             "Your payment could not be processed.",
             "Scan the code in this document to verify your account."],
            qr_png=make_qr_png(phish_url),
        )
        msg.add_attachment(pdf, maintype="application", subtype="pdf",
                           filename="verification.pdf")
        return filename, msg

    if category == "scanned_pdf":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject="Scanned Document from Accounting", seed=seed, filename=filename, **hdr,
        )
        msg.set_content("Attached is the scanned notice from the remittance department.")
        pdf = make_scanned_pdf(
            ["URGENT WIRE TRANSFER REQUIRED",
             "Your account will be suspended unless payment is confirmed.",
             f"Login at {phish_url}"],
            rng,
        )
        msg.add_attachment(pdf, maintype="application", subtype="pdf",
                           filename="scanned_notice.pdf")
        return filename, msg

    if category == "html_link_mismatch":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject="Your Account Needs Attention", seed=seed, filename=filename, **hdr,
        )
        msg.set_content(
            "Hello,\n\nPlease sign in to review your account. "
            "Use the link in this email to continue."
        )
        html = (
            "<html><body><p>Hello,</p>"
            "<p>Please sign in to review your account:</p>"
            f'<p><a href="{phish_url}">https://portal.company.org/login</a></p>'
            "<p>This request expires soon.</p></body></html>"
        )
        msg.add_alternative(html, subtype="html")
        return filename, msg

    if category == "html_attachment_phishing":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject="Action Required: Confirm Your Details", seed=seed, filename=filename, **hdr,
        )
        msg.set_content("Please open the attached file and follow the instructions.")
        html = (
            "<html><body><h1>Account Verification</h1>"
            "<p>Your account will be locked unless you verify now.</p>"
            f'<p><a href="{phish_url}">Click here to verify your account</a></p>'
            "</body></html>"
        )
        msg.add_attachment(html.encode("utf-8"), maintype="text", subtype="html",
                           filename="verify_account.html")
        return filename, msg

    if category == "fake_invoice_pdf":
        hdr = _phish_headers(rng, seed, filename)
        msg = _base_message(
            subject=f"Invoice {invoice_no} Due Today", seed=seed, filename=filename, **hdr,
        )
        msg.set_content("Please find attached your monthly invoice.")
        pdf = make_text_pdf(
            [f"INVOICE {invoice_no} — {company}",
             f"Amount due: ${amount}.00. Payment due today.",
             f"Pay now: {phish_url}",
             "Failure to pay immediately will suspend your account."],
            link_url=phish_url,
        )
        msg.add_attachment(pdf, maintype="application", subtype="pdf", filename="invoice.pdf")
        return filename, msg

    if category == "newsletter_logo_qr":
        hdr = _benign_headers(rng)
        msg = _base_message(
            subject=rng.choice(["Monthly Newsletter — Spring Events", "Community Update: Spring Fair"]),
            seed=seed, filename=filename, **hdr,
        )
        msg.set_content(
            f"Hi everyone,\n\nWelcome to the {company} monthly newsletter. "
            "This month we are hosting a spring fair with workshops and talks. "
            "Scan the code for the event schedule, or visit our events page."
        )
        logo = render_text_image([f"{company} — Spring Fair"], rng, size=(900, 160), font_size=28)
        msg.add_attachment(logo, maintype="image", subtype="png", filename="newsletter_banner.png")
        msg.add_attachment(make_qr_png(benign_url), maintype="image", subtype="png",
                           filename="event_qr.png")
        return filename, msg

    if category == "real_invoice":
        hdr = _benign_headers(rng)
        msg = _base_message(
            subject=f"Invoice {invoice_no} from {company}", seed=seed, filename=filename, **hdr,
        )
        msg.set_content(
            f"Hello,\n\nPlease find attached invoice {invoice_no} for ${amount}.00, "
            "due at the end of the month. Thank you for your business."
        )
        pdf = make_text_pdf(
            [f"INVOICE {invoice_no} — {company}",
             f"Amount due: ${amount}.00. Due at the end of the month.",
             f"Pay online: {benign_url}",
             "Questions? Reply to this email or call our billing team."],
            link_url=benign_url,
        )
        msg.add_attachment(pdf, maintype="application", subtype="pdf", filename="invoice.pdf")
        return filename, msg

    if category == "meeting_agenda":
        hdr = _benign_headers(rng)
        msg = _base_message(
            subject="Weekly Team Meeting Agenda", seed=seed, filename=filename, **hdr,
        )
        msg.set_content(
            "Hi everyone,\n\nAttached is the agenda for Tuesday's planning meeting: "
            "sprint review, roadmap discussion, and action items. See you there."
        )
        banner = render_text_image(["Team Meeting — Tuesday 10:00"], rng,
                                   size=(900, 160), font_size=28)
        msg.add_attachment(banner, maintype="image", subtype="png", filename="agenda_banner.png")
        pdf = make_text_pdf(
            ["Weekly Planning Meeting — Agenda",
             "1. Sprint review and demo.",
             "2. Roadmap discussion for next quarter.",
             "3. Action items and owners."],
        )
        msg.add_attachment(pdf, maintype="application", subtype="pdf", filename="agenda.pdf")
        return filename, msg

    raise ValueError(f"Unknown eval category: {category!r}")


def _assign_splits(
    rows: List[Dict[str, Any]], seed: int
) -> List[Dict[str, Any]]:
    """Stratified deterministic split: 20% of each label -> validation.

    A separate RNG stream (``seed + 1``) shuffles the per-label index
    lists, so the split does not depend on generation order.
    """
    rng = random.Random(seed + 1)
    by_label: Dict[int, List[int]] = {0: [], 1: []}
    for i, row in enumerate(rows):
        by_label[int(row["label"])].append(i)
    val_indices = set()
    for label, indices in by_label.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(indices) * VALIDATION_FRACTION)))
        val_indices.update(shuffled[:n_val])
    for i, row in enumerate(rows):
        row["split"] = "validation" if i in val_indices else "test"
    return rows


def generate_eval_set(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    seed: int = 42,
    debug: bool = False,
) -> Dict[str, Any]:
    """Generate the full eval set (or a 12-file debug subset) into ``out_dir``.

    Writes ``<category>_<idx>.eml`` files, ``labels.csv`` and
    ``manifest.json``. Returns the manifest dict. ``debug=True`` writes
    only the first 12 emails in deterministic order (6 phishing / 6
    benign) with the same seed; the manifest records ``debug: true``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for category, (label, count) in CATEGORIES.items():
        n = count
        if debug:
            # First 12 in category order, balanced 6 phishing / 6 benign:
            # take 1-2 per phishing category (first 6 categories give 6)
            # and 2 per benign category (first 3 give 6).
            if label == 1:
                n = 1 if category in (
                    "image_only_phishing", "text_button_image", "qr_body_image",
                    "qr_in_text_pdf", "scanned_pdf", "html_link_mismatch",
                ) else 0
            else:
                n = 2
        counts[category] = n
        for idx in range(1, n + 1):
            filename, msg = build_email(category, idx, rng, seed=seed)
            (out_dir / filename).write_bytes(msg.as_bytes())
            rows.append({"filename": filename, "label": label, "category": category})

    rows = _assign_splits(rows, seed)
    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "category", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    n_phish = sum(1 for r in rows if r["label"] == 1)
    n_benign = len(rows) - n_phish
    n_val = sum(1 for r in rows if r["split"] == "validation")
    manifest: Dict[str, Any] = {
        "seed": seed,
        "debug": debug,
        "n_total": len(rows),
        "n_phishing": n_phish,
        "n_benign": n_benign,
        "n_validation": n_val,
        "n_test": len(rows) - n_val,
        "validation_fraction": VALIDATION_FRACTION,
        "categories": counts,
        "labels_csv": "labels.csv",
        "split_rule": (
            "Stratified deterministic split (seed+1 stream): 20% of each label "
            "to validation; thresholds are fitted on validation only, never test."
        ),
        "synthetic_caveat": (
            "SYNTHETIC evaluation set: generated images/PDFs from a small set "
            "of templates with simulated OCR stressors (font/size variation, "
            "low contrast, rotation, noise, JPEG compression). No real-world "
            "adversary variation and no real emails. Results measure pipeline "
            "behaviour on these templates only and do not generalise to real "
            "phishing without a real labelled set."
        ),
        "threshold_note": (
            "Threshold 0.5 is a PLACEHOLDER. Fit thresholds on the validation "
            "slice only (see scripts/run_multimodal_eval.py); never tune on test."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Wrote %d eval emails (%d phishing / %d benign; %d validation / %d test) to %s",
        len(rows), n_phish, n_benign, n_val, len(rows) - n_val, out_dir,
    )
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python scripts/make_multimodal_eval_set.py``."""
    parser = argparse.ArgumentParser(
        description="Generate the synthetic labelled multimodal eval set (Section E)."
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
        help="Output directory for .eml files, labels.csv and manifest.json "
             "(default: data/multimodal_eval)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed (default: 42)")
    parser.add_argument(
        "--debug", action="store_true",
        help="Generate a 12-file debug subset (6 phishing / 6 benign) instead of the full 120",
    )
    args = parser.parse_args(argv)
    manifest = generate_eval_set(Path(args.out_dir), seed=args.seed, debug=args.debug)
    print(
        f"Wrote {manifest['n_total']} eval emails "
        f"({manifest['n_phishing']} phishing / {manifest['n_benign']} benign; "
        f"{manifest['n_validation']} validation / {manifest['n_test']} test) "
        f"to {args.out_dir}"
    )
    for category, count in manifest["categories"].items():
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
