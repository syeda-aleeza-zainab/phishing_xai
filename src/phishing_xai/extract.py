"""Multimodal extraction: parse .eml files and pull text from bodies, images, PDFs and QR codes.

Returns one dict::

    {"body_text": str,
     "parts": [{"kind", "filename", "text", "boxes", "qr_urls", "page"}, ...]}

``kind`` is one of: ``body_text``, ``image``, ``pdf_text``, ``pdf_ocr``, ``qr``.
``boxes`` is a list of ``{"text","left","top","width","height","conf"}`` (OCR word boxes, in
pixels of the source image) or an empty list. ``page`` is a 1-based PDF page or ``None``.
"""

import argparse
import email
import io
import json
import os
import re
import shutil
import sys
from email import policy
from pathlib import Path
from typing import Any, Dict, List, Optional

from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.extract")

URL_REGEX = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_CANDIDATE_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    Path(r"C:\Users\admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]


def _configure_tesseract() -> None:
    """Point pytesseract at Windows install locations when tesseract is not on PATH."""
    import pytesseract

    if shutil.which("tesseract") is not None:
        return
    for candidate in _CANDIDATE_TESSERACT_PATHS:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            tessdata_dir = candidate.parent / "tessdata"
            if tessdata_dir.exists() and "TESSDATA_PREFIX" not in os.environ:
                os.environ["TESSDATA_PREFIX"] = str(tessdata_dir)
            return


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


def ocr_image(image: Any) -> Dict[str, Any]:
    """OCR a PIL image. Returns {"text", "boxes"} with word-level bounding boxes."""
    import pytesseract
    from PIL import Image

    _configure_tesseract()
    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image))
    image = image.convert("RGB")
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


def decode_qr(image: Any) -> List[str]:
    """Decode QR codes in a PIL image and return decoded strings (URLs or payloads)."""
    from PIL import Image

    try:
        from pyzbar.pyzbar import decode
    except Exception as exc:  # pragma: no cover - depends on native zbar DLL
        logger.warning(f"pyzbar unavailable ({exc}); skipping QR decode.")
        return []
    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image))
    results = decode(image.convert("RGB"))
    out: List[str] = []
    for item in results:
        try:
            payload = item.data.decode("utf-8", errors="replace")
        except Exception:
            continue
        if payload and payload not in out:
            out.append(payload)
    return out


def extract_image_bytes(data: bytes, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """OCR an image attachment/inline image and decode any QR codes it contains."""
    from PIL import Image

    parts: List[Dict[str, Any]] = []
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        logger.warning(f"Could not open image {filename}: {exc}")
        return parts

    ocr = ocr_image(image)
    qr_urls = decode_qr(image)
    parts.append(_make_part("image", filename, ocr["text"], ocr["boxes"], qr_urls))
    if qr_urls:
        parts.append(_make_part("qr", filename, " ".join(qr_urls), [], qr_urls))
    return parts


def extract_pdf_bytes(data: bytes, filename: Optional[str] = None, ocr_dpi: int = 200) -> List[Dict[str, Any]]:
    """Extract text/links per page with pdfplumber; OCR pages (via PyMuPDF render) lacking text."""
    import pdfplumber
    from PIL import Image

    parts: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            links = [h.get("uri") for h in (page.hyperlinks or []) if h.get("uri")]
            for url in URL_REGEX.findall(text):
                if url not in links:
                    links.append(url)

            if text:
                parts.append(_make_part("pdf_text", filename, text, [], links, page=idx))
                continue

            # No extractable text -> render with PyMuPDF and OCR
            import fitz  # PyMuPDF

            doc = fitz.open(stream=data, filetype="pdf")
            try:
                pix = doc[idx - 1].get_pixmap(dpi=ocr_dpi)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
            finally:
                doc.close()
            ocr = ocr_image(image)
            qr_urls = decode_qr(image)
            all_links = links + [u for u in qr_urls if u not in links]
            parts.append(_make_part("pdf_ocr", filename, ocr["text"], ocr["boxes"], all_links, page=idx))
    return parts


def extract_eml(path: Path | str) -> Dict[str, Any]:
    """Parse an .eml file: walk MIME parts, gather body text and process attachments/inline images."""
    path = Path(path)
    with open(path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    body_chunks: List[str] = []
    parts: List[Dict[str, Any]] = []

    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        filename = part.get_filename()
        disposition = part.get_content_disposition()

        if ctype in ("text/plain", "text/html") and disposition != "attachment":
            try:
                content = part.get_content()
            except Exception:
                content = ""
            if ctype == "text/html":
                from bs4 import BeautifulSoup

                content = BeautifulSoup(content, "html.parser").get_text(" ")
            content = re.sub(r"\s+", " ", content or "").strip()
            if content:
                body_chunks.append(content)
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if ctype.startswith("image/"):
            parts.extend(extract_image_bytes(payload, filename or part.get("Content-ID")))
        elif ctype == "application/pdf" or (filename or "").lower().endswith(".pdf"):
            parts.extend(extract_pdf_bytes(payload, filename))

    body_text = " ".join(body_chunks).strip()
    parts.insert(0, _make_part("body_text", None, body_text))
    return {"body_text": body_text, "parts": parts}


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
