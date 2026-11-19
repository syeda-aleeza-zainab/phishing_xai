"""Multimodal extraction: parse .eml files and pull text from bodies, images, PDFs and QR codes.

Returns one dict::

    {"body_text": str,
     "parts": [{"kind", "filename", "text", "boxes", "qr_urls", "page"}, ...],
     "headers": {...}, "links": [...], "flags": [...], "skipped": [...]}

``body_text`` is the visible plain-text/HTML body only, kept as-is for backward
compatibility. The Subject is NOT folded into ``body_text``; it is returned as a
separate part with kind ``"subject"``. Consumers that score an email should score
the ``"subject"`` part plus ``body_text`` (and the other parts), as
``run_multimodal_predict`` does by scoring every part.

``kind`` values: ``body_text``, ``subject``, ``image``, ``pdf_text``, ``pdf_ocr``,
``pdf_image``, ``qr``, ``html_text``, ``hidden_text``, ``docx``, ``zip_listing``,
``svg_text``, ``attachment_text``.

* ``pdf_image`` (B1): one part per embedded image extracted from a PDF page
  (PyMuPDF ``page.get_images``), OCR'd and QR-decoded even when the page already
  has a text layer. The filename is ``"<pdf filename>#img<xref>"`` so several
  images on one page stay distinguishable; ``page`` is the 1-based page number.
  An embedded image whose OCR text is (near-)identical to the page's own text
  is skipped as a duplicate of the page (recorded in ``skipped``), unless it
  carries a QR code, in which case a ``qr`` part is still emitted.
* ``hidden_text`` (B2): text hidden with ``display:none`` / ``visibility:hidden`` /
  ``font-size:0`` / white-on-white inline styles. It is NEVER silently dropped and
  NEVER mixed into the scored ``body_text``; it gets its own part plus a flag.
* ``boxes`` is a list of ``{"text","left","top","width","height","conf"}`` (OCR word
  boxes, in pixels of the source image) or an empty list. ``page`` is a 1-based
  PDF page or ``None``.

``headers`` holds Subject, From, Reply-To, Return-Path, Message-ID (+ its domain)
and loosely parsed SPF/DKIM/DMARC verdicts from Authentication-Results.
``links`` is a list of ``{"url", "source", "anchor_text"?}`` (``source`` is one of
``html_anchor``, ``html_text``, ``html_image``, ``form_action``, ``docx_link``,
``docx_text``, ``svg_link``). ``flags`` is a list of dicts such as
``{"type": "anchor_href_mismatch", "anchor": ..., "href": ...}``.
``skipped`` is a list of ``{"part", "reason"}`` for anything not processed
(oversize attachment, missing tool, unsupported type, ...).

Safety limits (B5) are module constants (``MAX_ATTACHMENT_BYTES`` etc.) and are
enforced by skipping + recording, never by crashing. OCR timeout: no
signal-based timeout is used, because ``signal`` timeouts are not portable
(they only work on Unix main threads) and this module may run in threads or on
Windows; instead the size / page / part limits bound the work and every OCR/QR
call is wrapped so a failure is recorded as skipped. ``OCR_TIMEOUT_SECONDS`` is
kept as the documented per-part budget that a caller may enforce with its own
mechanism (e.g. a worker timeout). Static analysis only: no URL found in an
email, QR code or PDF is ever fetched or visited.
"""

import argparse
import base64
import binascii
import email
import io
import json
import os
import re
import shutil
import sys
import zipfile
from email import policy
from email.message import Message
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.extract")

URL_REGEX = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"(?:https?://)?(?:www\.)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})", re.IGNORECASE)
_HIDDEN_STYLE_RES = [
    re.compile(r"display\s*:\s*none", re.IGNORECASE),
    re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE),
    re.compile(r"font-size\s*:\s*0(?:px|pt|em|rem|%)?\b", re.IGNORECASE),
    re.compile(r"color\s*:\s*(?:white|#fff(?:fff)?)\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Safety limits (B5) — configurable module constants.
# ---------------------------------------------------------------------------
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024  # 15 MB per attachment / embedded blob
MAX_PDF_PAGES = 20  # only the first N pages of a PDF are processed
MAX_PARTS_PER_EMAIL = 50  # cap on extracted parts per email (body/subject excluded)
MAX_IMAGE_PIXELS = 178_956_970  # == Pillow's default Image.MAX_IMAGE_PIXELS (decompression-bomb guard)
MAX_ZIP_MEMBERS = 500
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_NESTED_DEPTH = 3  # max recursion into message/rfc822 attachments
OCR_TIMEOUT_SECONDS = 30  # documented per-part budget; see module docstring (no signal timeout)

SUSPICIOUS_ZIP_EXTENSIONS = {".exe", ".js", ".scr", ".iso", ".img", ".lnk", ".hta"}
_DOUBLE_EXT_RE = re.compile(r"\.[^./\\]+\.(exe|js|scr|iso|img|lnk|hta)$", re.IGNORECASE)


class _Skipped(Exception):
    """Internal: a part was skipped for a recorded reason (never a crash)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _record_skipped(skipped: Optional[List[Dict[str, Any]]], part: str, reason: str) -> None:
    if skipped is not None:
        skipped.append({"part": str(part), "reason": reason})


def _add_part(
    parts: List[Dict[str, Any]],
    skipped: Optional[List[Dict[str, Any]]],
    part: Dict[str, Any],
) -> bool:
    """Append ``part`` unless the per-email part cap is hit (then record skipped)."""
    if len(parts) >= MAX_PARTS_PER_EMAIL:
        _record_skipped(skipped, part.get("filename") or part.get("kind", "?"),
                         f"max parts per email ({MAX_PARTS_PER_EMAIL}) reached; part dropped")
        return False
    parts.append(part)
    return True


def _clean_url(url: str) -> str:
    """Strip trailing punctuation that a regex sweep picks up after a URL."""
    return url.rstrip(".,;:!?)]}\"'") if url else url


def _iter_parts(msg: Message):
    """Yield the leaf parts of a message without descending into message/rfc822.

    ``msg.walk()`` descends into nested messages and yields their inner parts
    raw (base64 blobs when the nested message was attached as bytes), so it
    cannot be used here: a ``message/rfc822`` part is yielded as one leaf and
    handled (recursed into) by the caller.
    """
    if msg.get_content_type() == "message/rfc822":
        yield msg
        return
    if msg.is_multipart():
        for sub in msg.iter_parts():
            yield from _iter_parts(sub)
    else:
        yield msg


def _domain_of(value: str) -> Optional[str]:
    """Lower-cased registrable-ish domain of a URL / bare domain, or None."""
    if not value or " " in value.strip():
        return None
    text = value.strip()
    if "://" not in text:
        text = "//" + text
    try:
        netloc = urlparse(text).netloc or ""
    except Exception:
        return None
    netloc = netloc.split("@")[-1].split(":")[0].lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


# ---------------------------------------------------------------------------
# Tesseract discovery (B7) — no hard-coded user paths.
# ---------------------------------------------------------------------------
def _configure_tesseract() -> None:
    """Point pytesseract at a tesseract binary: $TESSERACT_CMD, PATH, then platform defaults."""
    try:
        import pytesseract
    except Exception:
        return
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).exists():
        pytesseract.pytesseract.tesseract_cmd = env_cmd
        return
    if shutil.which("tesseract") is not None:
        return
    if sys.platform.startswith("win"):
        candidates = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    elif sys.platform == "darwin":
        candidates = [Path("/opt/homebrew/bin/tesseract"), Path("/usr/local/bin/tesseract")]
    else:
        candidates = [Path("/usr/bin/tesseract"), Path("/usr/local/bin/tesseract")]
    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            tessdata_dir = candidate.parent / "tessdata"
            if tessdata_dir.exists() and "TESSDATA_PREFIX" not in os.environ:
                os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
            return


def _tesseract_available() -> bool:
    """True when a tesseract binary can be found (checked at call time)."""
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        return Path(env_cmd).exists()
    if shutil.which("tesseract") is not None:
        return True
    try:
        import pytesseract

        cmd = pytesseract.pytesseract.tesseract_cmd
        return bool(cmd) and Path(str(cmd)).exists()
    except Exception:
        return False


def _make_part(
    kind: str,
    filename: Optional[str] = None,
    text: str = "",
    boxes: Optional[List[Dict[str, Any]]] = None,
    qr_urls: Optional[List[str]] = None,
    page: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "filename": filename,
        "text": text,
        "boxes": boxes or [],
        "qr_urls": qr_urls or [],
        "page": page,
    }


# ---------------------------------------------------------------------------
# Image preprocessing (B6) + OCR / QR
# ---------------------------------------------------------------------------
def _ocr_variants(image: Any) -> List[Any]:
    """Deterministic preprocessing variants for OCR, original first.

    Only used as fallbacks: the original image is tried first and later
    variants run only when it yields no words, so behaviour on clean images
    is unchanged.
    """
    from PIL import Image, ImageOps

    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image))
    original = image.convert("RGB")
    variants: List[Any] = [original]
    try:
        gray = image.convert("L")
        if max(image.size) < 1000:
            gray = gray.resize((gray.width * 2, gray.height * 2), Image.LANCZOS)
        variants.append(gray)
        thresholded = ImageOps.autocontrast(gray).point(lambda p: 255 if p > 140 else 0)
        variants.append(thresholded.convert("RGB"))
        variants.append(ImageOps.invert(original))
    except Exception as exc:
        logger.warning(f"OCR preprocessing failed ({exc}); using original image only.")
    return variants


def _ocr_one(image: Any) -> Dict[str, Any]:
    import pytesseract

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    boxes: List[Dict[str, Any]] = []
    words: List[str] = []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        words.append(word)
        boxes.append({
            "text": word,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": conf,
        })
    return {"text": " ".join(words), "boxes": boxes}


def ocr_image(image: Any) -> Dict[str, Any]:
    """OCR a PIL image (or image bytes). Returns {"text", "boxes"}.

    Fails gracefully when tesseract is missing or errors: logs a warning and
    returns empty text/boxes instead of raising, so extraction continues.
    """
    try:
        _configure_tesseract()
    except Exception as exc:
        logger.warning(f"Could not configure tesseract ({exc}); skipping OCR.")
        return {"text": "", "boxes": []}
    if not _tesseract_available():
        logger.warning("Tesseract OCR binary not found (PATH / $TESSERACT_CMD / platform "
                       "default); skipping OCR for this image.")
        return {"text": "", "boxes": []}
    try:
        variants = _ocr_variants(image)
    except Exception as exc:
        logger.warning(f"Could not open image for OCR ({exc}); skipping OCR.")
        return {"text": "", "boxes": []}
    for variant in variants:
        try:
            result = _ocr_one(variant)
        except Exception as exc:
            logger.warning(f"OCR failed on a variant ({exc}); trying next variant.")
            continue
        if result["text"].strip():
            return result
    return {"text": "", "boxes": []}


def _qr_variants(image: Any) -> List[Any]:
    """Deterministic QR variants: original, grayscale, inverted, rotations, scales."""
    from PIL import Image, ImageOps

    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image))
    original = image.convert("RGB")
    variants: List[Any] = [original]
    try:
        variants.append(image.convert("L"))
        variants.append(ImageOps.invert(original))
        for angle in (90, 180, 270):
            variants.append(original.rotate(angle, expand=True))
        small = max(original.size)
        if small < 600:
            variants.append(original.resize((original.width * 2, original.height * 2), Image.LANCZOS))
        elif small > 2000:
            variants.append(original.resize((original.width // 2, original.height // 2), Image.LANCZOS))
    except Exception as exc:
        logger.warning(f"QR preprocessing failed ({exc}); using original image only.")
    return variants


def _decode_qr_pyzbar(image: Any) -> List[str]:
    try:
        from pyzbar.pyzbar import decode
    except Exception as exc:
        logger.warning(f"pyzbar unavailable ({exc}); skipping pyzbar QR decode.")
        return []
    out: List[str] = []
    for variant in _qr_variants(image):
        try:
            results = decode(variant)
        except Exception:
            continue
        for item in results:
            try:
                payload = item.data.decode("utf-8", errors="replace")
            except Exception:
                continue
            if payload and payload not in out:
                out.append(payload)
        if out:
            break
    return out


def _decode_qr_opencv(image: Any) -> List[str]:
    """OpenCV QRCodeDetector fallback; guarded import, never crashes if cv2 is absent."""
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except Exception:
        return []
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(io.BytesIO(image))
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        detector = cv2.QRCodeDetector()
        try:
            _ok, decoded, _pts = detector.detectAndDecodeMulti(gray)
            candidates = list(decoded) if decoded is not None else []
        except Exception:
            data, _pts, _straight = detector.detectAndDecode(gray)
            candidates = [data] if data else []
        return [c for c in candidates if c]
    except Exception as exc:
        logger.warning(f"OpenCV QR decode failed ({exc}).")
        return []


def decode_qr(image: Any) -> List[str]:
    """Decode QR codes in a PIL image (or bytes); return decoded strings (URLs/payloads).

    Tries pyzbar over several deterministic variants first, then OpenCV's
    QRCodeDetector as a fallback when pyzbar finds nothing. Returns [] when
    no decoder is available or nothing decodes — never raises.
    """
    out = _decode_qr_pyzbar(image)
    if out:
        return out
    for payload in _decode_qr_opencv(image):
        if payload not in out:
            out.append(payload)
    return out


def _open_image_checked(data: bytes, filename: Optional[str],
                        skipped: Optional[List[Dict[str, Any]]]) -> Optional[Any]:
    """Open image bytes with the decompression-bomb pixel check (B5).

    Returns a loaded PIL image, or None (with a ``skipped`` entry) when the
    image is oversized or unreadable.
    """
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data))
    except Exception as exc:
        logger.warning(f"Could not open image {filename}: {exc}")
        _record_skipped(skipped, filename or "image", f"unreadable image: {exc}")
        return None
    try:
        width, height = image.size
    except Exception as exc:
        logger.warning(f"Could not read size of image {filename}: {exc}")
        return None
    if width * height > MAX_IMAGE_PIXELS:
        reason = f"image is {width}x{height} pixels, over MAX_IMAGE_PIXELS ({MAX_IMAGE_PIXELS})"
        logger.warning(f"Skipping oversized image {filename}: {reason}")
        _record_skipped(skipped, filename or "image", reason)
        return None
    try:
        image.load()
    except Image.DecompressionBombError as exc:
        logger.warning(f"Skipping decompression-bomb image {filename}: {exc}")
        _record_skipped(skipped, filename or "image", f"decompression bomb: {exc}")
        return None
    except Exception as exc:
        logger.warning(f"Could not load image {filename}: {exc}")
        _record_skipped(skipped, filename or "image", f"unreadable image: {exc}")
        return None
    return image


def extract_image_bytes(data: bytes, filename: Optional[str] = None,
                        skipped: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """OCR an image attachment/inline image and decode any QR codes it contains."""
    parts: List[Dict[str, Any]] = []
    image = _open_image_checked(data, filename, skipped)
    if image is None:
        return parts
    ocr = ocr_image(image)
    qr_urls = decode_qr(image)
    parts.append(_make_part("image", filename, ocr["text"], ocr["boxes"], qr_urls))
    if qr_urls:
        parts.append(_make_part("qr", filename, " ".join(qr_urls), [], qr_urls))
    return parts


def _token_jaccard(a: str, b: str) -> float:
    set_a = set(re.findall(r"\w+", (a or "").lower()))
    set_b = set(re.findall(r"\w+", (b or "").lower()))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_pdf_bytes(data: bytes, filename: Optional[str] = None, ocr_dpi: int = 200,
                      skipped: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Extract text/links per page with pdfplumber; OCR pages lacking text.

    B1: embedded images are extracted from EVERY page (PyMuPDF
    ``page.get_images``) and OCR'd + QR-decoded even when the page has a text
    layer; see the module docstring for the ``pdf_image`` part shape.
    Only the first ``MAX_PDF_PAGES`` pages are processed.
    """
    import pdfplumber
    from PIL import Image

    parts: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        total_pages = len(pdf.pages)
        if total_pages > MAX_PDF_PAGES:
            _record_skipped(skipped, filename or "pdf",
                            f"PDF has {total_pages} pages; only first {MAX_PDF_PAGES} processed")
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for idx, page in enumerate(pdf.pages[:MAX_PDF_PAGES], start=1):
                text = (page.extract_text() or "").strip()
                links = [h.get("uri") for h in (page.hyperlinks or []) if h.get("uri")]
                for url in URL_REGEX.findall(text):
                    url = _clean_url(url)
                    if url not in links:
                        links.append(url)

                if text:
                    parts.append(_make_part("pdf_text", filename, text, [], links, page=idx))
                else:
                    # No extractable text -> render with PyMuPDF and OCR
                    pix = doc[idx - 1].get_pixmap(dpi=ocr_dpi)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr = ocr_image(image)
                    qr_urls = decode_qr(image)
                    all_links = links + [u for u in qr_urls if u not in links]
                    parts.append(_make_part("pdf_ocr", filename, ocr["text"], ocr["boxes"],
                                            all_links, page=idx))
                    text = ocr["text"]

                # B1: embedded images on this page, even when it has text.
                try:
                    fitz_page = doc[idx - 1]
                except Exception:
                    continue
                for img_info in fitz_page.get_images(full=True):
                    xref = img_info[0]
                    try:
                        pix_dict = doc.extract_image(xref)
                        img_bytes = pix_dict["image"]
                    except Exception as exc:
                        logger.warning(f"Could not extract embedded image xref={xref} "
                                       f"from {filename} p{idx}: {exc}")
                        continue
                    img_filename = f"{filename}#img{xref}" if filename else f"pdf-image-{xref}"
                    image = _open_image_checked(img_bytes, img_filename, skipped)
                    if image is None:
                        continue
                    qr_urls = decode_qr(image)
                    ocr = ocr_image(image)
                    if qr_urls:
                        parts.append(_make_part("qr", img_filename, " ".join(qr_urls), [],
                                                qr_urls, page=idx))
                    if not ocr["text"].strip():
                        continue
                    if _token_jaccard(ocr["text"], text) >= 0.8:
                        _record_skipped(skipped, img_filename,
                                        "embedded image text duplicates the page text")
                        continue
                    parts.append(_make_part("pdf_image", img_filename, ocr["text"],
                                            ocr["boxes"], qr_urls, page=idx))
        finally:
            doc.close()
    return parts


# ---------------------------------------------------------------------------
# Headers (B3)
# ---------------------------------------------------------------------------
def _parse_auth_results(raw: Optional[str]) -> Dict[str, Optional[str]]:
    """Loosely parse SPF/DKIM/DMARC verdicts out of an Authentication-Results header."""
    verdicts: Dict[str, Optional[str]] = {"spf": None, "dkim": None, "dmarc": None}
    if not raw:
        return verdicts
    for match in re.finditer(r"\b(spf|dkim|dmarc)\s*=\s*([A-Za-z]+)", raw, re.IGNORECASE):
        key = match.group(1).lower()
        if verdicts.get(key) is None:
            verdicts[key] = match.group(2).lower()
    return verdicts


def _extract_headers(msg: Message) -> Dict[str, Any]:
    subject = msg.get("Subject")
    message_id = msg.get("Message-ID")
    message_id_domain = None
    if message_id:
        match = re.search(r"@([^>\s]+)", message_id)
        if match:
            message_id_domain = match.group(1).strip(">").lower()
    auth_raw = msg.get("Authentication-Results")
    headers: Dict[str, Any] = {
        "subject": str(subject) if subject is not None else None,
        "from": str(msg.get("From")) if msg.get("From") is not None else None,
        "reply_to": str(msg.get("Reply-To")) if msg.get("Reply-To") is not None else None,
        "return_path": str(msg.get("Return-Path")) if msg.get("Return-Path") is not None else None,
        "message_id": str(message_id) if message_id is not None else None,
        "message_id_domain": message_id_domain,
        "authentication_results": str(auth_raw) if auth_raw is not None else None,
    }
    headers.update(_parse_auth_results(auth_raw))
    return headers


# ---------------------------------------------------------------------------
# HTML bodies / attachments (B2)
# ---------------------------------------------------------------------------
def _is_hidden_tag(tag: Any) -> bool:
    if tag.has_attr("hidden"):
        return True
    style = tag.get("style") or ""
    return any(rx.search(style) for rx in _HIDDEN_STYLE_RES)


def _parse_html(html: str, filename: Optional[str] = None,
                cid_images: Optional[Dict[str, Tuple[bytes, Optional[str]]]] = None,
                skipped: Optional[List[Dict[str, Any]]] = None,
                consumed_cids: Optional[set] = None) -> Dict[str, Any]:
    """Parse HTML: visible text, hidden text, links, flags, embedded/cid images.

    Returns {"visible_text", "hidden_text", "links", "flags", "parts"}.
    Hidden text is collected separately and never added to ``visible_text``.
    ``cid:`` images are resolved from ``cid_images`` when present, else flagged
    as missing. No URL in the HTML is ever fetched.
    """
    links: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    parts: List[Dict[str, Any]] = []
    hidden_chunks: List[str] = []
    cid_images = cid_images or {}

    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        logger.warning(f"BeautifulSoup unavailable ({exc}); falling back to plain tag stripping.")
        visible = re.sub(r"<[^>]+>", " ", html or "")
        visible = re.sub(r"\s+", " ", visible).strip()
        for url in URL_REGEX.findall(visible):
            links.append({"url": url, "source": "html_text"})
        return {"visible_text": visible, "hidden_text": "", "links": links,
                "flags": flags, "parts": parts}

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if _is_hidden_tag(tag):
            text = tag.get_text(" ", strip=True)
            if text:
                hidden_chunks.append(text)
            tag.decompose()
    hidden_text = re.sub(r"\s+", " ", " ".join(hidden_chunks)).strip()
    if hidden_text:
        parts.append(_make_part("hidden_text", filename, hidden_text))
        flags.append({"type": "hidden_text", "text": hidden_text[:200]})

    visible_text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()

    for tag in soup.find_all("a", href=True):
        href = (tag.get("href") or "").strip()
        anchor = tag.get_text(" ", strip=True)
        if not href:
            continue
        links.append({"url": href, "source": "html_anchor", "anchor_text": anchor})
        anchor_domain = _domain_of(anchor) if anchor and "." in anchor else None
        href_domain = _domain_of(href)
        if anchor_domain and href_domain and anchor_domain != href_domain:
            flags.append({"type": "anchor_href_mismatch", "anchor": anchor, "href": href})

    for tag in soup.find_all("form", action=True):
        action = (tag.get("action") or "").strip()
        if action:
            links.append({"url": action, "source": "form_action"})
            flags.append({"type": "form_action", "action": action})

    for i, tag in enumerate(soup.find_all("img", src=True)):
        src = (tag.get("src") or "").strip()
        if not src:
            continue
        if src.lower().startswith("data:"):
            try:
                _meta, _, b64 = src.partition(",")
                raw = base64.b64decode(b64, validate=False)
            except (binascii.Error, ValueError) as exc:
                _record_skipped(skipped, f"html-embedded-image-{i}",
                                f"undecodable base64 image: {exc}")
                continue
            if len(raw) > MAX_ATTACHMENT_BYTES:
                _record_skipped(skipped, f"html-embedded-image-{i}",
                                f"embedded image over MAX_ATTACHMENT_BYTES ({MAX_ATTACHMENT_BYTES})")
                continue
            parts.extend(extract_image_bytes(raw, f"html-embedded-image-{i}.png", skipped))
        elif src.lower().startswith("cid:"):
            cid = src[4:].strip().strip("<>")
            hit = cid_images.get(cid) or cid_images.get(f"<{cid}>")
            if hit is None:
                flags.append({"type": "missing_cid_image", "cid": cid})
                _record_skipped(skipped, f"cid:{cid}", "cid image not present in the email")
            else:
                payload, cid_filename = hit
                if consumed_cids is not None:
                    consumed_cids.add(cid)
                    consumed_cids.add(f"<{cid}>")
                parts.extend(extract_image_bytes(payload, cid_filename or f"cid-{cid}.png", skipped))
        elif src.lower().startswith(("http://", "https://")):
            links.append({"url": src, "source": "html_image"})

    for url in URL_REGEX.findall(visible_text):
        url = _clean_url(url)
        if url and not any(link["url"] == url for link in links):
            links.append({"url": url, "source": "html_text"})
    return {"visible_text": visible_text, "hidden_text": hidden_text,
            "links": links, "flags": flags, "parts": parts}


# ---------------------------------------------------------------------------
# More attachment types (B4)
# ---------------------------------------------------------------------------
def _extract_docx_bytes(data: bytes, filename: Optional[str],
                        links: List[Dict[str, Any]],
                        skipped: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract text + links from a .docx (guarded import; skip, don't crash)."""
    try:
        import docx
    except Exception as exc:
        logger.warning(f"python-docx unavailable ({exc}); skipping .docx {filename}.")
        _record_skipped(skipped, filename or "docx", f"python-docx unavailable: {exc}")
        return []
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        logger.warning(f"Could not parse .docx {filename}: {exc}")
        _record_skipped(skipped, filename or "docx", f"unreadable .docx: {exc}")
        return []
    chunks: List[str] = []
    for para in document.paragraphs:
        if para.text.strip():
            chunks.append(para.text.strip())
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    chunks.append(cell.text.strip())
    text = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    try:
        for rel in document.part.rels.values():
            if rel.reltype.endswith("hyperlink") and getattr(rel, "target_ref", None):
                url = str(rel.target_ref)
                if not any(link["url"] == url for link in links):
                    links.append({"url": url, "source": "docx_link"})
    except Exception:
        pass
    for url in URL_REGEX.findall(text):
        url = _clean_url(url)
        if url and not any(link["url"] == url for link in links):
            links.append({"url": url, "source": "docx_text"})
    if not text:
        return []
    return [_make_part("docx", filename, text)]


def _extract_zip_bytes(data: bytes, filename: Optional[str],
                       flags: List[Dict[str, Any]],
                       skipped: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """List .zip member names ONLY (never extract members) and flag risky names."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:
        logger.warning(f"Could not open .zip {filename}: {exc}")
        _record_skipped(skipped, filename or "zip", f"unreadable .zip: {exc}")
        return []
    with zf:
        infos = zf.infolist()
        total_uncompressed = sum(info.file_size for info in infos)
        if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            _record_skipped(skipped, filename or "zip",
                            f"zip uncompressed total {total_uncompressed} bytes over "
                            f"MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES ({MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES})")
            return []
        names = [info.filename for info in infos]
        if len(names) > MAX_ZIP_MEMBERS:
            _record_skipped(skipped, filename or "zip",
                            f"zip has {len(names)} members; only first {MAX_ZIP_MEMBERS} listed")
            names = names[:MAX_ZIP_MEMBERS]
        for name in names:
            lower = name.lower()
            ext = Path(lower).suffix
            if _DOUBLE_EXT_RE.search(lower):
                flags.append({"type": "double_extension", "member": name,
                              "reason": "double extension ending in a risky executable type"})
            elif ext in SUSPICIOUS_ZIP_EXTENSIONS:
                flags.append({"type": "suspicious_zip_member", "member": name,
                              "reason": f"risky extension {ext}"})
        return [_make_part("zip_listing", filename, "\n".join(names))]


def _extract_svg_bytes(data: bytes, filename: Optional[str],
                       links: List[Dict[str, Any]],
                       skipped: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract text (and link hrefs) from an .svg; scripts are dropped, never executed."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(data)
    except Exception as exc:
        logger.warning(f"Could not parse .svg {filename}: {exc}")
        _record_skipped(skipped, filename or "svg", f"unreadable .svg: {exc}")
        return []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1].lower()
        if tag in ("script", "style"):
            elem.clear()
    text = re.sub(r"\s+", " ", " ".join(root.itertext())).strip()
    for elem in root.iter():
        href = elem.attrib.get("href") or elem.attrib.get("{http://www.w3.org/1999/xlink}href")
        if href and not any(link["url"] == href for link in links):
            links.append({"url": href, "source": "svg_link"})
    if not text:
        return []
    return [_make_part("svg_text", filename, text)]


# ---------------------------------------------------------------------------
# .eml driver
# ---------------------------------------------------------------------------
def _extract_message(msg: Message, ctx: Dict[str, Any], depth: int = 0) -> None:
    """Walk one message, filling ctx (body_chunks/parts/links/flags/skipped)."""
    body_chunks: List[str] = ctx["body_chunks"]
    parts: List[Dict[str, Any]] = ctx["parts"]
    links: List[Dict[str, Any]] = ctx["links"]
    flags: List[Dict[str, Any]] = ctx["flags"]
    skipped: List[Dict[str, Any]] = ctx["skipped"]

    if depth == 0:
        ctx["headers"] = _extract_headers(msg)

    # Pre-pass: map Content-ID -> (payload, filename) for cid: resolution (B2).
    cid_images: Dict[str, Tuple[bytes, Optional[str]]] = {}
    for part in _iter_parts(msg):
        cid = part.get("Content-ID")
        if cid and part.get_content_type().startswith("image/"):
            payload = part.get_payload(decode=True)
            if payload:
                cid_images[cid.strip()] = (payload, part.get_filename())
    consumed_cids: set = set()

    def add_all(new_parts: List[Dict[str, Any]]) -> None:
        for p in new_parts:
            _add_part(parts, skipped, p)

    for part in _iter_parts(msg):
        ctype = part.get_content_type()
        filename = part.get_filename()
        disposition = part.get_content_disposition()
        lower_name = (filename or "").lower()

        # Nested message (B4): recurse with the same extraction, capped depth.
        # Checked before the multipart skip: a message/rfc822 part IS multipart.
        if ctype == "message/rfc822":
            if depth >= MAX_NESTED_DEPTH:
                _record_skipped(skipped, filename or "nested message",
                                f"nested message depth over MAX_NESTED_DEPTH ({MAX_NESTED_DEPTH})")
                continue
            nested = None
            try:
                content = part.get_content()
            except Exception:
                content = None
            if isinstance(content, Message):
                nested = content
            elif isinstance(content, list):
                nested = next((c for c in content if isinstance(c, Message)), None)
            if nested is None:
                payload = part.get_payload(decode=True)
                if payload:
                    try:
                        nested = email.message_from_bytes(payload, policy=policy.default)
                    except Exception as exc:
                        _record_skipped(skipped, filename or "nested message",
                                        f"unparseable nested message: {exc}")
                        continue
                else:
                    continue
            _extract_message(nested, ctx, depth + 1)
            continue

        if part.is_multipart():
            continue

        if ctype in ("text/plain", "text/html") and disposition != "attachment":
            try:
                content = part.get_content()
            except Exception:
                content = ""
            if not isinstance(content, str):
                try:
                    payload = part.get_payload(decode=True) or b""
                    content = payload.decode("utf-8", errors="replace")
                except Exception:
                    content = ""
            if ctype == "text/html":
                parsed = _parse_html(content, filename, cid_images, skipped, consumed_cids)
                if parsed["visible_text"]:
                    body_chunks.append(parsed["visible_text"])
                for link in parsed["links"]:
                    if not any(l["url"] == link["url"] and l["source"] == link["source"]
                               for l in links):
                        links.append(link)
                flags.extend(parsed["flags"])
                add_all(parsed["parts"])
            else:
                content = re.sub(r"\s+", " ", content or "").strip()
                if content:
                    body_chunks.append(content)
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if len(payload) > MAX_ATTACHMENT_BYTES:
            _record_skipped(skipped, filename or ctype,
                            f"attachment over MAX_ATTACHMENT_BYTES ({MAX_ATTACHMENT_BYTES})")
            continue
        try:
            if ctype == "image/svg+xml" or lower_name.endswith(".svg"):
                add_all(_extract_svg_bytes(payload, filename, links, skipped))
            elif ctype.startswith("image/"):
                cid = (part.get("Content-ID") or "").strip()
                if cid and (cid in consumed_cids or f"<{cid.strip('<>')}>" in consumed_cids
                            or cid.strip("<>") in {c.strip("<>") for c in consumed_cids}):
                    continue  # already OCR'd via the cid: reference in the HTML body
                add_all(extract_image_bytes(payload, filename or part.get("Content-ID"), skipped))
            elif ctype == "application/pdf" or lower_name.endswith(".pdf"):
                add_all(extract_pdf_bytes(payload, filename, skipped=skipped))
            elif ctype == "text/html" or lower_name.endswith((".html", ".htm")):
                text = payload.decode("utf-8", errors="replace")
                parsed = _parse_html(text, filename, cid_images, skipped, consumed_cids)
                if parsed["visible_text"]:
                    add_all([_make_part("html_text", filename, parsed["visible_text"])])
                for link in parsed["links"]:
                    if not any(l["url"] == link["url"] and l["source"] == link["source"]
                               for l in links):
                        links.append(link)
                flags.extend(parsed["flags"])
                add_all(parsed["parts"])
            elif (ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    or lower_name.endswith(".docx")):
                add_all(_extract_docx_bytes(payload, filename, links, skipped))
            elif ctype == "application/zip" or lower_name.endswith(".zip"):
                add_all(_extract_zip_bytes(payload, filename, flags, skipped))
            elif ctype == "text/plain":
                text = re.sub(r"\s+", " ", payload.decode("utf-8", errors="replace")).strip()
                if text:
                    add_all([_make_part("attachment_text", filename, text)])
            else:
                _record_skipped(skipped, filename or ctype,
                                f"unsupported attachment type {ctype}")
        except _Skipped as exc:
            _record_skipped(skipped, filename or ctype, exc.reason)
        except Exception as exc:
            logger.warning(f"Failed to extract {filename or ctype}: {exc}")
            _record_skipped(skipped, filename or ctype, f"extraction failed: {exc}")


def extract_eml(path: Path | str) -> Dict[str, Any]:
    """Parse an .eml file: walk MIME parts, gather body text and process attachments/inline images.

    Backward compatible: ``body_text`` and the original ``parts`` shape are
    unchanged; the result additionally carries ``headers``, ``links``,
    ``flags`` and ``skipped`` (see the module docstring).
    """
    path = Path(path)
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    ctx: Dict[str, Any] = {
        "body_chunks": [], "parts": [], "links": [], "flags": [], "skipped": [],
        "headers": {},
    }
    _extract_message(msg, ctx, depth=0)

    body_text = " ".join(ctx["body_chunks"]).strip()
    final_parts: List[Dict[str, Any]] = [_make_part("body_text", None, body_text)]
    subject = (ctx["headers"] or {}).get("subject")
    if subject:
        final_parts.append(_make_part("subject", None, subject))
    final_parts.extend(ctx["parts"])
    return {"body_text": body_text, "parts": final_parts,
            "headers": ctx["headers"], "links": ctx["links"],
            "flags": ctx["flags"], "skipped": ctx["skipped"]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract text/OCR/QR/PDF content from an .eml file")
    parser.add_argument("eml", type=str, help="Path to .eml file")
    parser.add_argument("--no_boxes", action="store_true", help="Omit OCR word boxes from printed JSON")
    args = parser.parse_args(argv)

    result = extract_eml(args.eml)
    if args.no_boxes:
        for p in result["parts"]:
            p["boxes"] = f"<{len(p['boxes'])} boxes>"
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
