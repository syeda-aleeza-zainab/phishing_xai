"""Multimodal fusion (Section D): combine per-part scorer scores, a
concatenated-context score, static URL risk and header/structural flags
into one precision-first verdict.

Public API (what Section E's eval script should import)::

    from phishing_xai.multimodal import (
        FusionConfig,               # re-exported from phishing_xai.config
        build_concatenated_context, # extraction dict -> one marked string
        score_urls_for_email,       # extraction dict -> worst-URL summary
        score_flags,                # extraction dict -> flag score + reasons
        fuse_email,                 # extraction + part scores -> verdict dict
        overlay_scanned_pdf_page,   # render a PDF page + draw LITA boxes
        overlay_qr_image,           # draw decoded-QR region + annotate URL
    )

    result = fuse_email(
        extraction, part_scores,
        fusion="weighted",                 # or "max" (recall-heavy baseline)
        concatenated_score=context_prob,    # scorer P(phish) on the context
        config=FusionConfig(),            # weights / threshold live here
    )

``part_scores`` maps a part identifier to its scorer P(phishing) in [0, 1].
Identifiers are matched in this order per part: the canonical part id
``"<kind>[:<filename>][#p<page>]"`` (see :func:`part_id`), then the bare
``kind``, then the part index as a string. Callers that score parts should
key by :func:`part_id` (that is what ``scripts/run_multimodal_predict.py``
does). Unknown keys are ignored.

Precision-first design (user priority 2026-11-19: "mera priority focus
precision hai" — minimise false positives / FPR; F1 and accuracy are
secondary and must never be the sole selection criterion):

* Weighted fusion (default)::

      final = w_context * context_score
            + w_max_part * max_part_score
            + w_url * url_score
            + w_flags * flag_score

  with the defaults in :class:`FusionConfig` (0.45 / 0.20 / 0.20 / 0.15,
  summing to 1.0). The concatenated context gets the largest weight and a
  single part only 0.20, so one weak part scoring high on its own cannot
  push the final score over the threshold: e.g. max part 0.95 with context
  0.30 and no URL/flag evidence gives 0.45*0.30 + 0.20*0.95 = 0.325, still
  Safe. A phishing verdict needs corroboration: context + a part, or
  context + URL risk, or a part + URL risk + flags.
* Corroboration gate (weighted only): a phishing verdict additionally
  requires at least two of the three channels (context, max part, URL) to
  be at/above 0.5, or one channel plus a flag score >= 0.30, or a fused
  score at/above ``high_confidence_threshold`` (0.75). The gate is reported
  as ``corroborated`` in the result.
* Max rule (``fusion="max"``, selectable baseline): final score is the max
  part score and the verdict flips as soon as ANY part scores >= threshold.
  It is recall-heavy and precision-poor by construction: a single noisy
  OCR part can cause a false positive, and URL/flag evidence is ignored.
  Keep it only as the baseline Section E compares against.
* Threshold: ``FusionConfig.threshold`` defaults to 0.5 as a PLACEHOLDER.
  It must be tuned on VALIDATION only (never test) on Kaggle with the
  trained checkpoint, choosing the threshold that maximises phishing-class
  precision subject to a recall floor — never by maximising F1 alone.
  Until that validation run exists, do not claim the threshold was tuned.

Flag contributions (each flag type maps to a small numeric contribution;
``flag_score`` is the sum clamped to [0, 1]):

* ``anchor_href_mismatch`` 0.15, ``double_extension`` 0.20,
  ``suspicious_zip_member`` 0.15, ``hidden_text`` 0.10,
  ``form_action`` 0.05, ``missing_cid_image`` 0.02, unknown flag 0.05.
* Header checks (from ``extraction["headers"]``): each of SPF / DKIM /
  DMARC == ``"fail"`` adds 0.10; a Reply-To domain different from the From
  domain adds 0.10 (``reply_to_mismatch``).

URL scoring reuses Section C (:func:`phishing_xai.url_risk.score_url`) over
every URL found: ``extraction["links"]``, every part's ``qr_urls`` from a
genuine QR source (``qr`` / ``image`` / ``pdf_image`` parts, scored with
``in_qr=True``), and URLs extracted from body/part text. Section B stores
PDF hyperlinks in the ``qr_urls`` slot of ``pdf_text`` / ``pdf_ocr`` parts;
those are plain links, not QR payloads, so they are scored with
``in_qr=False``. The email's ``url_score`` is the worst (max) single-URL
score. Static analysis only: no URL is ever fetched or visited.

LITA explanation stays on the winning / concatenated text (the caller runs
the explainer on ``result["concatenated_context"]``); the overlay helpers
in this module only draw boxes for scanned-PDF pages and QR images, reusing
``draw_lita_boxes_on_image`` from ``explain.py``. Both helpers use guarded
imports and return ``None`` with a warning (never a crash) when their
rendering dependency is missing.

No GPU, no internet, no checkpoint is needed by anything in this module
except the caller's model scoring, which is passed in as numbers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phishing_xai.config import FusionConfig
from phishing_xai.url_risk import extract_urls as _extract_urls
from phishing_xai.url_risk import score_url
from phishing_xai.utils import get_logger

logger = get_logger("phishing_xai.multimodal")

__all__ = [
    "FusionConfig",
    "FLAG_CONTRIBUTIONS",
    "PART_KIND_TO_MODALITY",
    "part_id",
    "build_concatenated_context",
    "score_texts",
    "score_urls_for_email",
    "score_flags",
    "fuse_email",
    "overlay_scanned_pdf_page",
    "overlay_qr_image",
]

# ---------------------------------------------------------------------------
# Flag -> numeric contribution (documented in the module docstring).
# ---------------------------------------------------------------------------
FLAG_CONTRIBUTIONS: Dict[str, float] = {
    "anchor_href_mismatch": 0.15,
    "double_extension": 0.20,
    "suspicious_zip_member": 0.15,
    "hidden_text": 0.10,
    "form_action": 0.05,
    "missing_cid_image": 0.02,
}
UNKNOWN_FLAG_CONTRIBUTION: float = 0.05
AUTH_FAIL_CONTRIBUTION: float = 0.10   # per failed SPF / DKIM / DMARC check
REPLY_TO_MISMATCH_CONTRIBUTION: float = 0.10
#: A flag score at/above this counts as "strong flag support" for the
#: corroboration gate (one active channel + strong flags is enough).
STRONG_FLAG_SCORE: float = 0.30
#: A channel counts as "active" for the corroboration gate at/above this.
CHANNEL_ACTIVE_THRESHOLD: float = 0.5

#: Part kind -> modality label used for "most contributing modality".
PART_KIND_TO_MODALITY: Dict[str, str] = {
    "body_text": "body",
    "subject": "body",
    "html_text": "body",
    "hidden_text": "body",
    "image": "ocr",
    "pdf_ocr": "ocr",
    "pdf_image": "ocr",
    "qr": "qr",
    "pdf_text": "pdf",
    "docx": "attachment",
    "zip_listing": "attachment",
    "svg_text": "attachment",
    "attachment_text": "attachment",
}

#: Context sections, in order: (marker, part kinds that feed it).
_CONTEXT_SECTIONS: List[Tuple[str, Tuple[str, ...]]] = [
    ("[SUBJECT]", ("subject",)),
    ("[BODY]", ("body_text", "html_text")),
    ("[HIDDEN]", ("hidden_text",)),
    ("[OCR]", ("image", "pdf_ocr", "pdf_image")),
    ("[PDF]", ("pdf_text",)),
    ("[ATTACHMENT]", ("docx", "zip_listing", "svg_text", "attachment_text")),
    ("[QR_URL]", ("qr",)),
]

#: Fallback char cap when no tokenizer is available: ~4 chars per token.
_CHARS_PER_TOKEN_FALLBACK: int = 4


def part_id(part: Dict[str, Any]) -> str:
    """Canonical identifier for an extracted part.

    ``"<kind>[:<filename>][#p<page>]"`` — the same label
    ``scripts/run_multimodal_predict.py`` prints in its per-part table.
    """
    label = str(part.get("kind") or "?")
    if part.get("filename"):
        label += f":{part['filename']}"
    if part.get("page") is not None:
        label += f"#p{part['page']}"
    return label


def _parts_of(extraction: Dict[str, Any]) -> List[Dict[str, Any]]:
    parts = extraction.get("parts") if isinstance(extraction, dict) else None
    return list(parts) if isinstance(parts, list) else []


def score_texts(scorer: Any, texts: List[str]) -> List[float]:
    """P(phishing) for each text under any scorer (``TextScorer`` protocol).

    Empty / whitespace-only texts score 0.0 without calling the scorer.
    """
    out: List[float] = []
    for text in texts:
        if not (text or "").strip():
            out.append(0.0)
            continue
        probs = scorer.predict_proba([text])
        out.append(float(probs[0, 1]))
    return out


def build_concatenated_context(
    extraction: Dict[str, Any],
    *,
    tokenizer: Any = None,
    max_length: int = 512,
    scorer: Any = None,
) -> str:
    """Build ONE scored string from an extraction dict, with part markers.

    Sections in order: ``[SUBJECT]`` (subject part), ``[BODY]``
    (body + HTML text), ``[HIDDEN]`` (hidden text, kept separate so it is
    visible to the model but never silently mixed into the body),
    ``[OCR]`` (image / scanned-PDF / embedded-image OCR text), ``[PDF]``
    (PDF text layer), ``[ATTACHMENT]`` (docx / zip listing / svg / plain
    attachments), ``[QR_URL]`` (QR payloads). Empty sections are omitted.
    The top-level ``body_text`` is used for ``[BODY]`` when no ``body_text``
    part carries text (both hold the same text; the part wins if present).
    QR payloads from ``qr_urls`` are appended to ``[QR_URL]`` even when the
    ``qr`` part text is empty.

    Truncation: with a scorer, ``scorer.truncate`` is used; else with a
    tokenizer, the existing ``truncate_text_to_model_window`` is used so
    the string is exactly what the model sees; without either, the text
    is capped at ``max_length * 4`` characters (roughly 4 chars/token)
    so the call can never blow up the model window silently.
    """
    sections: Dict[str, List[str]] = {marker: [] for marker, _ in _CONTEXT_SECTIONS}
    kind_to_marker = {
        kind: marker for marker, kinds in _CONTEXT_SECTIONS for kind in kinds
    }
    for part in _parts_of(extraction):
        text = (part.get("text") or "").strip()
        marker = kind_to_marker.get(part.get("kind"))
        if marker is None:
            continue
        if text:
            sections[marker].append(text)
        for qr_url in part.get("qr_urls") or []:
            if qr_url and qr_url not in sections["[QR_URL]"]:
                sections["[QR_URL]"].append(str(qr_url))

    if not sections["[BODY]"]:
        body_text = (extraction.get("body_text") or "").strip()
        if body_text:
            sections["[BODY]"].append(body_text)

    chunks: List[str] = []
    for marker, _ in _CONTEXT_SECTIONS:
        texts = sections[marker]
        if texts:
            chunks.append(f"{marker} " + " ".join(texts))
    context = " ".join(chunks).strip()
    if not context:
        return ""

    if scorer is not None:
        return scorer.truncate(context, max_length)
    if tokenizer is not None:
        from phishing_xai.data import truncate_text_to_model_window

        return truncate_text_to_model_window(context, tokenizer, max_length=max_length)
    cap = max_length * _CHARS_PER_TOKEN_FALLBACK
    if len(context) > cap:
        logger.warning(
            f"Concatenated context truncated by char cap ({len(context)} -> {cap} chars); "
            "pass a tokenizer for exact token-window truncation."
        )
        return context[:cap]
    return context


def score_urls_for_email(
    extraction: Dict[str, Any],
    *,
    url_config: Any = None,
) -> Dict[str, Any]:
    """Score every URL in an email with Section C and summarise the worst.

    URL sources: ``extraction["links"]`` (HTML href / form action / docx /
    svg links), every part's ``qr_urls`` (scored with ``in_qr=True`` — a QR
    code hides the link from the reader), and URLs extracted from
    ``body_text`` and every part's text. De-duplicated; a URL seen both in
    text and in a QR code is scored once, as QR (the riskier reading).
    Exception: Section B stores PDF hyperlinks in the ``qr_urls`` slot of
    ``pdf_text`` / ``pdf_ocr`` parts — those are plain links, scored with
    ``in_qr=False`` (only ``qr`` / ``image`` / ``pdf_image`` parts count
    as QR sources).
    Returns ``{"urls", "url_score", "max_url", "n_urls", "reasons"}`` where
    ``url_score`` is the worst (max) single-URL score, ``max_url`` is that
    URL's full :func:`score_url` dict (or ``None`` when there are no URLs),
    and ``reasons`` are the worst URL's reasons. Static analysis only.
    """
    in_qr_by_url: Dict[str, bool] = {}

    def _add(url: str, in_qr: bool) -> None:
        url = (url or "").strip()
        if not url:
            return
        in_qr_by_url[url] = in_qr_by_url.get(url, False) or bool(in_qr)

    for link in extraction.get("links") or []:
        if isinstance(link, dict):
            _add(str(link.get("url") or ""), False)
        elif isinstance(link, str):
            _add(link, False)
    for part in _parts_of(extraction):
        # Section B stores PDF hyperlinks in the ``qr_urls`` slot of
        # ``pdf_text`` / ``pdf_ocr`` parts (they are links, not QR payloads),
        # so only ``qr`` / ``image`` / ``pdf_image`` parts contribute QR URLs.
        part_is_qr_source = part.get("kind") in ("qr", "image", "pdf_image")
        for qr_url in part.get("qr_urls") or []:
            _add(str(qr_url), part_is_qr_source)
        for url in _extract_urls(part.get("text") or ""):
            _add(url, part.get("kind") == "qr")
    for url in _extract_urls(extraction.get("body_text") or ""):
        _add(url, False)

    scored = [
        score_url(url, in_qr=in_qr, config=url_config)
        for url, in_qr in in_qr_by_url.items()
    ]
    if not scored:
        return {"urls": [], "url_score": 0.0, "max_url": None, "n_urls": 0, "reasons": []}
    worst = max(scored, key=lambda r: r["url_score"])
    return {
        "urls": scored,
        "url_score": float(worst["url_score"]),
        "max_url": worst,
        "n_urls": len(scored),
        "reasons": list(worst["reasons"]),
    }


def _domain_of_address(address: Optional[str]) -> Optional[str]:
    """Lower-cased domain of an email address header value, or None."""
    if not address:
        return None
    import re

    match = re.search(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", str(address))
    return match.group(1).lower() if match else None


def score_flags(extraction: Dict[str, Any]) -> Dict[str, Any]:
    """Map an extraction's flags + headers to a numeric flag score.

    Each ``flags`` entry contributes per :data:`FLAG_CONTRIBUTIONS` (unknown
    types contribute 0.05); each failed SPF / DKIM / DMARC verdict in
    ``headers`` contributes 0.10; a Reply-To domain different from the From
    domain contributes 0.10. ``flag_score`` is the sum clamped to [0, 1].
    Returns ``{"flag_score", "contributions", "reasons"}`` where
    ``contributions`` is a list of ``{"type", "contribution", "detail"}``.
    """
    contributions: List[Dict[str, Any]] = []
    reasons: List[str] = []

    for flag in extraction.get("flags") or []:
        if not isinstance(flag, dict):
            continue
        ftype = str(flag.get("type") or "unknown")
        value = FLAG_CONTRIBUTIONS.get(ftype, UNKNOWN_FLAG_CONTRIBUTION)
        detail = (
            flag.get("href")
            or flag.get("member")
            or flag.get("action")
            or flag.get("text")
            or flag.get("reason")
            or ""
        )
        contributions.append({"type": ftype, "contribution": value, "detail": str(detail)[:200]})
        reasons.append(f"Flag '{ftype}'" + (f": {detail}" if detail else "") + f" (+{value:.2f})")

    headers = extraction.get("headers") or {}
    if isinstance(headers, dict):
        for mechanism in ("spf", "dkim", "dmarc"):
            verdict = (headers.get(mechanism) or "")
            if isinstance(verdict, str) and verdict.lower() == "fail":
                contributions.append({
                    "type": f"auth_{mechanism}_fail",
                    "contribution": AUTH_FAIL_CONTRIBUTION,
                    "detail": f"{mechanism.upper()} failed in Authentication-Results",
                })
                reasons.append(
                    f"Authentication {mechanism.upper()} failed (+{AUTH_FAIL_CONTRIBUTION:.2f})"
                )
        from_domain = _domain_of_address(headers.get("from"))
        reply_domain = _domain_of_address(headers.get("reply_to"))
        if from_domain and reply_domain and reply_domain != from_domain:
            contributions.append({
                "type": "reply_to_mismatch",
                "contribution": REPLY_TO_MISMATCH_CONTRIBUTION,
                "detail": f"Reply-To domain {reply_domain} differs from From domain {from_domain}",
            })
            reasons.append(
                f"Reply-To domain ({reply_domain}) differs from From domain ({from_domain}) "
                f"(+{REPLY_TO_MISMATCH_CONTRIBUTION:.2f})"
            )

    total = min(1.0, sum(c["contribution"] for c in contributions))
    return {"flag_score": round(float(total), 3), "contributions": contributions, "reasons": reasons}


def _match_part_scores(
    extraction: Dict[str, Any], part_scores: Dict[str, float]
) -> Dict[str, float]:
    """Resolve a part's score from ``part_scores`` (part id, then kind, then index)."""
    resolved: Dict[str, float] = {}
    for idx, part in enumerate(_parts_of(extraction)):
        score = None
        for key in (part_id(part), str(part.get("kind") or ""), str(idx)):
            if key in part_scores:
                score = part_scores[key]
                break
        if score is not None:
            resolved[part_id(part)] = max(0.0, min(1.0, float(score)))
    return resolved


def fuse_email(
    extraction: Dict[str, Any],
    part_scores: Dict[str, float],
    *,
    fusion: str = "weighted",
    config: Optional[FusionConfig] = None,
    concatenated_score: Optional[float] = None,
    threshold: Optional[float] = None,
    url_config: Any = None,
    tokenizer: Any = None,
    max_length: Optional[int] = None,
    scorer: Any = None,
) -> Dict[str, Any]:
    """Fuse per-part scores, context score, URL risk and flags into a verdict.

    Args:
        extraction: The dict returned by ``extract_eml`` (``parts``,
            ``body_text``, ``links``, ``flags``, ``headers``).
        part_scores: ``{part identifier: scorer P(phishing)}``; see the
            module docstring for the identifier matching order.
        fusion: ``"weighted"`` (default, precision-first) or ``"max"``
            (recall-heavy baseline: any part >= threshold flips the verdict;
            URL/flag/context evidence is computed and reported but ignored).
        config: :class:`FusionConfig` with weights / threshold. Defaults are
            precision-first (see the module docstring); ``threshold`` here is
            a 0.5 PLACEHOLDER until tuned on validation.
        concatenated_score: scorer P(phishing) of
            :func:`build_concatenated_context` for this email. When ``None``
            it falls back to the max part score (recorded as
            ``context_score_source == "fallback_max_part"``) so the weighted
            sum stays on the same scale.
        threshold: Overrides ``config.threshold`` for this call.
        url_config: Optional :class:`UrlRiskConfig` passed to URL scoring.
        tokenizer / max_length: Passed to :func:`build_concatenated_context`
            (only used to truncate the returned context string).
        scorer: Optional scorer passed through to
            :func:`build_concatenated_context` for truncation; takes
            precedence over ``tokenizer`` there.

    Returns a dict with ``fusion``, ``verdict`` (``"Phishing Email"`` /
    ``"Safe Email"``), ``final_class`` (1/0), ``final_score``,
    ``threshold``, ``corroborated`` (weighted only; ``True`` for max),
    ``context_score``, ``context_score_source``, ``max_part_score``,
    ``winning_part`` / ``winning_part_kind`` (highest-scoring part),
    ``url_score`` / ``url_details``, ``flag_score`` / ``flag_details``,
    ``part_scores`` (resolved, keyed by part id), ``contributions``
    (per-channel weighted contributions), ``most_contributing_modality``
    (one of ``"body"`` / ``"ocr"`` / ``"qr"`` / ``"pdf"`` /
    ``"attachment"`` / ``"url"`` / ``"headers"`` / ``"none"``),
    ``reasons`` and ``top_reasons`` (human-readable, first 5),
    ``concatenated_context`` (truncated as documented above).
    """
    if fusion not in ("weighted", "max"):
        raise ValueError(f"fusion must be 'weighted' or 'max', got {fusion!r}")
    cfg = config or FusionConfig()
    thr = float(cfg.threshold if threshold is None else threshold)
    ml = int(cfg.max_length if max_length is None else max_length)

    context = build_concatenated_context(
        extraction, tokenizer=tokenizer, max_length=ml, scorer=scorer
    )
    resolved = _match_part_scores(extraction, part_scores or {})

    parts = _parts_of(extraction)
    winning_part_id: Optional[str] = None
    winning_kind: Optional[str] = None
    max_part_score = 0.0
    if resolved:
        winning_part_id = max(resolved, key=lambda k: resolved[k])
        max_part_score = float(resolved[winning_part_id])
        for part in parts:
            if part_id(part) == winning_part_id:
                winning_kind = str(part.get("kind"))
                break

    if concatenated_score is None:
        context_score = max_part_score
        context_source = "fallback_max_part"
    else:
        context_score = max(0.0, min(1.0, float(concatenated_score)))
        context_source = "scored"

    url_details = score_urls_for_email(extraction, url_config=url_config)
    url_score = float(url_details["url_score"])
    flag_details = score_flags(extraction)
    flag_score = float(flag_details["flag_score"])

    contributions = {
        "context": round(cfg.w_context * context_score, 4),
        "part": round(cfg.w_max_part * max_part_score, 4),
        "url": round(cfg.w_url * url_score, 4),
        "flags": round(cfg.w_flags * flag_score, 4),
    }

    # Most-contributing modality. The context contribution is credited to the
    # winning part's modality: the context is built from the same parts, so
    # this is an attribution heuristic, not a per-section model score, and is
    # documented as such. URL and header-flag evidence keep their own labels.
    modality_scores: Dict[str, float] = {}
    if winning_kind is not None:
        modality = PART_KIND_TO_MODALITY.get(winning_kind, "body")
        modality_scores[modality] = contributions["part"] + contributions["context"]
    if url_score > 0:
        modality_scores["url"] = contributions["url"]
    if flag_score > 0:
        modality_scores["headers"] = contributions["flags"]
    most_modality = (
        max(modality_scores, key=lambda k: modality_scores[k]) if modality_scores else "none"
    )

    reasons: List[str] = []
    if winning_part_id is not None and max_part_score > 0:
        reasons.append(
            f"Highest-scoring part '{winning_part_id}' (kind {winning_kind}) "
            f"scored P(phishing)={max_part_score:.2f}."
        )
    if context:
        reasons.append(
            f"Concatenated context (subject+body+OCR+PDF+QR, {context_source}) "
            f"scored P(phishing)={context_score:.2f}."
        )
    if url_details["max_url"] is not None:
        worst = url_details["max_url"]
        first_reason = (worst["reasons"] or [""])[0]
        reasons.append(
            f"Riskiest URL {worst['url']!r} scored {url_score:.2f}"
            + (f": {first_reason}." if first_reason else ".")
        )
    reasons.extend(flag_details["reasons"])
    if not reasons:
        reasons.append("No part, URL or flag evidence above zero; nothing scored as phishing.")

    if fusion == "max":
        final_score = max_part_score
        final_class = 1 if final_score >= thr else 0
        corroborated = True
        reasons.append(
            "Max rule: the verdict flips as soon as ANY part scores at/above the "
            "threshold (recall-heavy baseline; URL/flag/context evidence ignored)."
        )
    else:
        final_score = round(sum(contributions.values()), 4)
        active = sum(
            s >= CHANNEL_ACTIVE_THRESHOLD
            for s in (context_score, max_part_score, url_score)
        )
        corroborated = (
            active >= 2
            or final_score >= cfg.high_confidence_threshold
            or (active >= 1 and flag_score >= STRONG_FLAG_SCORE)
        )
        if final_score >= thr and not corroborated:
            reasons.append(
                f"Weighted score {final_score:.2f} reached the threshold {thr:.2f} but "
                f"lacked corroboration (only {active} of 3 channels >= "
                f"{CHANNEL_ACTIVE_THRESHOLD:.2f}, flag score {flag_score:.2f}); "
                "verdict kept Safe to protect precision."
            )
        final_class = 1 if (final_score >= thr and corroborated) else 0

    return {
        "fusion": fusion,
        "verdict": "Phishing Email" if final_class == 1 else "Safe Email",
        "final_class": final_class,
        "final_score": round(float(final_score), 4),
        "threshold": thr,
        "corroborated": bool(corroborated),
        "context_score": round(float(context_score), 4),
        "context_score_source": context_source,
        "max_part_score": round(float(max_part_score), 4),
        "winning_part": winning_part_id,
        "winning_part_kind": winning_kind,
        "url_score": round(float(url_score), 4),
        "url_details": url_details,
        "flag_score": round(float(flag_score), 4),
        "flag_details": flag_details,
        "part_scores": resolved,
        "contributions": contributions,
        "most_contributing_modality": most_modality,
        "reasons": reasons,
        "top_reasons": reasons[:5],
        "concatenated_context": context,
    }


# ---------------------------------------------------------------------------
# Overlays (D3) — guarded imports, never crash, return None when deps miss.
# ---------------------------------------------------------------------------
def overlay_scanned_pdf_page(
    pdf_bytes: bytes,
    page: int,
    ocr_boxes: List[Dict[str, Any]],
    lita_top_words: List[Any],
    output_path: Any,
    *,
    dpi: int = 200,
    top_k: int = 5,
) -> Optional[Path]:
    """Render one scanned-PDF page and draw LITA boxes on it.

    Args:
        pdf_bytes: Raw PDF bytes (e.g. the attachment payload from the .eml).
        page: 1-based page number.
        ocr_boxes: OCR word boxes for that page, in pixels of the OCR render.
            The page is re-rendered at the same ``dpi`` the extractor used
            (200 by default) so the box coordinates line up; if the caller
            OCR'd at a different DPI, pass that DPI here.
        lita_top_words: Top LITA words (same shapes ``draw_lita_boxes_on_image``
            accepts).
        output_path: Destination PNG path.
        dpi: Render DPI (must match the OCR render DPI for box alignment).
        top_k: Maximum number of top LITA words to highlight.

    Returns the output path, or ``None`` (with a warning) when PyMuPDF or
    Pillow is missing or rendering fails. Never raises.
    """
    try:
        try:
            import fitz  # PyMuPDF (deprecated alias; works in this venv)
        except ImportError:
            import pymupdf as fitz  # type: ignore[no-redef]
        from PIL import Image  # noqa: F401  (needed by draw_lita_boxes_on_image)
    except Exception as exc:
        logger.warning(f"Skipping scanned-PDF overlay (rendering dependency missing: {exc}).")
        return None
    try:
        from phishing_xai.explain import draw_lita_boxes_on_image

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if page < 1 or page > len(doc):
                logger.warning(f"Skipping scanned-PDF overlay: page {page} out of range (1-{len(doc)}).")
                return None
            pix = doc[page - 1].get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("png")
        finally:
            doc.close()
        return Path(draw_lita_boxes_on_image(
            image_input=image_bytes,
            ocr_boxes=ocr_boxes,
            lita_top_words=lita_top_words,
            output_path=Path(output_path),
            top_k=top_k,
        ))
    except Exception as exc:
        logger.warning(f"Skipping scanned-PDF overlay (rendering failed: {exc}).")
        return None


def overlay_qr_image(
    image_input: Any,
    qr_urls: List[str],
    output_path: Any,
    *,
    qr_boxes: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
) -> Optional[Path]:
    """Draw the decoded-QR region on an image and annotate the decoded URL.

    The QR region is drawn from pyzbar's decoded polygon when pyzbar is
    available; otherwise the whole image gets a red border (the QR location
    is unknown without a decoder). The decoded URL(s) are annotated as text
    at the top of the image. ``qr_boxes`` (optional OCR-style
    ``{"left","top","width","height"}`` dicts) are also drawn when given.
    Returns the output path, or ``None`` (with a warning) when Pillow is
    missing or drawing fails. Never raises. Static only: the URL is drawn
    as text, never fetched.
    """
    try:
        import io
        from pathlib import Path as _Path
        from PIL import Image, ImageDraw
    except Exception as exc:
        logger.warning(f"Skipping QR overlay (Pillow missing: {exc}).")
        return None
    try:
        if isinstance(image_input, (str, _Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        else:
            img = image_input.copy().convert("RGB")
        out_path = _Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        draw = ImageDraw.Draw(img)
        drew_region = False
        try:
            from pyzbar.pyzbar import decode as _pyzbar_decode

            for item in _pyzbar_decode(img):
                polygon = getattr(item, "polygon", None)
                if polygon:
                    points = [(int(p.x), int(p.y)) for p in polygon]
                    if len(points) >= 2:
                        draw.line(points + [points[0]], fill="red", width=4)
                        drew_region = True
                rect = getattr(item, "rect", None)
                if rect is not None:
                    draw.rectangle(
                        [rect.left, rect.top, rect.left + rect.width, rect.top + rect.height],
                        outline="red",
                        width=4,
                    )
                    drew_region = True
        except Exception as exc:
            logger.debug(f"pyzbar QR polygon unavailable ({exc}); drawing border only.")
        for box in qr_boxes or []:
            try:
                draw.rectangle(
                    [int(box["left"]), int(box["top"]),
                     int(box["left"]) + int(box["width"]),
                     int(box["top"]) + int(box["height"])],
                    outline="red",
                    width=3,
                )
                drew_region = True
            except Exception:
                continue
        if not drew_region:
            draw.rectangle([2, 2, img.width - 3, img.height - 3], outline="red", width=4)

        urls = [str(u) for u in (qr_urls or []) if u][:top_k]
        if urls:
            label = "QR: " + " | ".join(urls)
            # White strip behind the label so it stays readable on any image.
            draw.rectangle([0, 0, img.width, 16], fill="white")
            draw.text((4, 2), label[:120], fill="red")
        img.save(out_path, format="PNG")
        return out_path
    except Exception as exc:
        logger.warning(f"Skipping QR overlay (drawing failed: {exc}).")
        return None
