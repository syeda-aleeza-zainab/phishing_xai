"""XAI explainers: LIME (batched probabilities) and Transformers-Interpret with Captum fallback."""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from lime.lime_text import LimeTextExplainer
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from phishing_xai.utils import get_device, get_logger

logger = get_logger("phishing_xai.explain")
CLASS_NAMES = ["Safe Email", "Phishing Email"]

# Shared word-level regex preserving URLs, contractions, words, numbers, and punctuation
WORD_SPLIT_REGEX = r"https?://[^\s,;!?\"'>)]+|\b\w+(?:['’]\w+)*|[^\w\s]"


def split_words(text: str) -> List[str]:
    """Shared word splitter using regex preserving URLs, contractions, numbers, and punctuation."""
    if not isinstance(text, str) or not text:
        return []
    return re.findall(WORD_SPLIT_REGEX, text)


class LimeExplainer:
    """LIME explainer wrapper ensuring batched inference, softmax probabilities, and shared word splitting."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        max_length: int = 512,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names or CLASS_NAMES
        self.device = device or get_device()
        self.batch_size = batch_size
        self.max_length = max_length
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            random_state=seed,
            split_expression=split_words,
        )

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Batched prediction returning true softmax probabilities summing to 1.0."""
        self.model.eval()
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                chunk = texts[i : i + self.batch_size]
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # Fix LIME bug: convert raw unbounded logits to probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def explain(
        self,
        text: str,
        num_features: Optional[int] = None,
        num_samples: int = 1000,
        max_features_cap: int = 500,
    ) -> Tuple[List[Tuple[str, float]], int, np.ndarray]:
        """Explain an instance using LIME.

        Args:
            text: Input text instance.
            num_features: Number of features to return. If None, returns all words in the
                instance (capped by max_features_cap) for complete faithfulness evaluation.
            num_samples: Number of perturbation samples for LIME.
            max_features_cap: Maximum feature count when num_features is None.

        Returns:
            - word_attributions: list of (word, score) tuples for the predicted class.
            - pred_class: predicted class index (0 or 1).
            - probs: prediction probabilities for [Safe, Phishing].
        """
        probs = self.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))

        if num_features is None:
            words = split_words(text)
            n_features = min(len(words), max_features_cap) if words else 15
        else:
            n_features = num_features

        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_proba,
            num_features=n_features,
            num_samples=num_samples,
            labels=[pred_class],
        )

        word_attributions = exp.as_list(label=pred_class)
        return word_attributions, pred_class, probs


class TransformerExplainer:
    """Integrated Gradients explainer with primary Transformers-Interpret and Captum fallback."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerFast,
        device: Optional[torch.device] = None,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or get_device()
        # RoBERTa position ids are cumsum(non-pad mask) + padding_idx, so a
        # sequence of length L needs position ids up to L+1: the position
        # table must have more than L+1 rows. roberta-base has 514 rows
        # (L=512 is safe, matching training); the tiny test model has 512
        # rows (L=510 max, matching its tokenizer model_max_length).
        # Cap max_length at the table size minus two when detectable.
        self.max_length = max_length
        pos_rows = None
        for _attr in ("roberta", "distilbert", "bert"):
            _emb = getattr(getattr(self.model, _attr, None), "embeddings", None)
            _pe = getattr(_emb, "position_embeddings", None)
            if _pe is not None:
                pos_rows = _pe.num_embeddings
                break
        if pos_rows is not None:
            self.max_length = min(self.max_length, pos_rows - 2)
        self.model.to(self.device)
        self.model.eval()

        self.ti_explainer = None
        try:
            from transformers_interpret import SequenceClassificationExplainer
            self.ti_explainer = SequenceClassificationExplainer(self.model, self.tokenizer)
            logger.info("Initialized primary explainer: transformers-interpret.")
        except Exception as e:
            logger.warning(f"Could not initialize transformers-interpret ({e}); using Captum fallback.")

    def explain_raw(self, text: str, target_class: Optional[int] = None) -> List[Tuple[str, float]]:
        """Compute token-level integrated gradient attributions."""
        class_name = None
        if target_class is not None:
            if hasattr(self.model.config, "id2label") and target_class in self.model.config.id2label:
                class_name = self.model.config.id2label[target_class]
            else:
                class_name = target_class

        # Truncate to the first max_length tokens BEFORE any explainer sees
        # the text. RoBERTa position_embeddings has 514 rows, so an
        # untruncated >512-token email makes position_ids exceed the table
        # and raises a CUDA ScatterGather index-out-of-bounds assert (Stage C
        # 2026-11-19: email_27, 896 words / 1029 tokens; email_30, 1290
        # tokens). transformers-interpret tokenizes internally with no
        # truncation, so the truncation must happen here, not downstream.
        # The model itself truncates to 512 in training, so this matches
        # what the model actually sees.
        ids = self.tokenizer.encode(
            text, add_special_tokens=True, truncation=True, max_length=self.max_length
        )
        text = self.tokenizer.decode(ids, skip_special_tokens=True)

        if self.ti_explainer is not None:
            try:
                # transformers-interpret handles target class mapping
                attributions = self.ti_explainer(text, class_name=class_name)
                self._free_cuda()
                return attributions
            except Exception as e:
                logger.warning(f"transformers-interpret call failed ({e}); falling back to Captum.")
                # TI may have left large cached CUDA blocks behind (Stage C
                # v2: TI OOM'd on a 512-token email); free them before the
                # Captum fallback or it OOMs on the same memory.
                self._free_cuda()

        # Captum LayerIntegratedGradients fallback
        return self._explain_with_captum(text, target_class=target_class)

    @staticmethod
    def _free_cuda() -> None:
        """Release cached CUDA blocks (no-op on CPU). Cheap insurance
        between explainer calls so a long email's peak memory does not
        carry over to the next email or the fallback path."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clean_text(self, text: str) -> str:
        """Preprocess text to mirror transformers-interpret tokenization behavior."""
        text = re.sub(r"([.,!?()])", r" \1 ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def _explain_with_captum(self, text: str, target_class: Optional[int] = None) -> List[Tuple[str, float]]:
        """Fallback explainer using Captum LayerIntegratedGradients with exact parity to transformers-interpret."""
        clean_text = self._clean_text(text)
        # Same truncation as explain_raw: without it a >512-token email
        # overflows RoBERTa position_embeddings (514 rows) and raises a
        # CUDA ScatterGather index-out-of-bounds assert.
        input_ids_list = self.tokenizer.encode(
            clean_text, add_special_tokens=True, truncation=True, max_length=self.max_length
        )
        input_ids = torch.tensor([input_ids_list], device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)

        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask=attention_mask).logits
                target_class = int(torch.argmax(logits, dim=1).item())

        # Forward function returning softmax target probability
        def forward_fn(inputs, mask):
            logits = self.model(inputs, attention_mask=mask).logits
            probs = torch.softmax(logits, dim=1)
            return probs[:, target_class]

        # Target exact word_embeddings layer
        if hasattr(self.model, "roberta") and hasattr(self.model.roberta, "embeddings"):
            embedding_layer = self.model.roberta.embeddings.word_embeddings
        elif hasattr(self.model, "distilbert") and hasattr(self.model.distilbert, "embeddings"):
            embedding_layer = self.model.distilbert.embeddings.word_embeddings
        elif hasattr(self.model, "bert") and hasattr(self.model.bert, "embeddings"):
            embedding_layer = self.model.bert.embeddings.word_embeddings
        else:
            embedding_layer = self.model.get_input_embeddings()

        # Construct baseline: [BOS] + [PAD] * (N - 2) + [EOS]
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        bos_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else self.tokenizer.bos_token_id
        eos_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else self.tokenizer.eos_token_id

        baseline_ids = input_ids.clone()
        baseline_ids[0, :] = pad_id
        if bos_id is not None and baseline_ids.shape[1] > 0:
            baseline_ids[0, 0] = bos_id
        if eos_id is not None and baseline_ids.shape[1] > 1:
            baseline_ids[0, -1] = eos_id

        lig = LayerIntegratedGradients(forward_fn, embedding_layer)
        # Stage C v2 OOM (2026-11-19, T4 14.56 GiB): Captum's default
        # internal_batch_size=None runs ALL n_steps interpolated inputs in
        # one forward, so a 512-token email means a 50 x 512-token batch
        # (~13 GiB allocated) and transformers-interpret fails the same way
        # before the fallback. Chunking the interpolation (internal_batch_size)
        # changes nothing numerically — same 50 steps, same math, smaller
        # peak memory. n_steps stays 50 so attributions are unchanged.
        try:
            attributions, _ = lig.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=(attention_mask,),
                target=None,
                n_steps=50,
                internal_batch_size=5,
                return_convergence_delta=True,
            )
        finally:
            self._free_cuda()

        # Sum across embedding dimensions and apply L2 normalization
        token_attrs = attributions.sum(dim=-1).squeeze(0)
        norm = torch.norm(token_attrs)
        if norm > 0:
            token_attrs = token_attrs / norm
        token_attributions = token_attrs.detach().cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        clean_tokens = [t.replace("Ġ", "").replace("##", "") for t in tokens]

        return list(zip(clean_tokens, token_attributions.tolist()))


def aggregate_subword_attributions_to_words(
    text: str,
    token_attributions: List[Tuple[str, float]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 512,
) -> List[Tuple[int, str, float]]:
    """Aggregate token-level attributions to word-level using shared word regex and offset_mapping.

    Resolves the RoBERTa BPE issue where 'Ġ' prefixes broke token-string matching.
    Crucially, repeated word occurrences are NOT merged: each word occurrence in the sequence
    is returned with its distinct word_index, word text, and aggregated attribution score.

    Returns:
        List[Tuple[int, str, float]]: List of (word_index, word_string, attribution_score)
        in sequential sentence order.
    """
    words_with_spans = [(m.group(), m.start(), m.end()) for m in re.finditer(WORD_SPLIT_REGEX, text)]
    if not words_with_spans:
        return []

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoding.get("offset_mapping", [])

    token_scores = [score for _, score in token_attributions]
    word_scores: Dict[int, List[float]] = {i: [] for i in range(len(words_with_spans))}

    for idx, (s, e) in enumerate(offsets):
        if idx >= len(token_scores):
            break
        if s == e:
            continue  # Special token
        score = token_scores[idx]
        for w_idx, (w_str, w_s, w_e) in enumerate(words_with_spans):
            if not (e <= w_s or s >= w_e):
                word_scores[w_idx].append(score)
                break

    aggregated: List[Tuple[int, str, float]] = []
    for w_idx, (w_str, w_s, w_e) in enumerate(words_with_spans):
        scores = word_scores[w_idx]
        avg_score = float(np.mean(scores)) if scores else 0.0
        aggregated.append((w_idx, w_str, avg_score))

    return aggregated


def draw_lita_boxes_on_image(
    image_input: Any,
    ocr_boxes: List[Dict[str, Any]],
    lita_top_words: List[Any],
    output_path: Optional[Any] = None,
    top_k: int = 5,
) -> Any:
    """Draw red bounding boxes around the LITA top-k words on an image using OCR word boxes.

    Args:
        image_input: Path to an image, raw image bytes, or a PIL.Image.Image object.
        ocr_boxes: List of OCR box dicts (``{"text","left","top","width","height","conf"}``)
            returned by ``phishing_xai.extract.ocr_image``.
        lita_top_words: Top words from LITA, passed as strings, ``(word, score)`` tuples,
            or ``{"word": ..., "lita_score": ...}`` dicts.
        output_path: Destination path for ``<name>_explained.png``. If None and ``image_input``
            is a path, defaults to ``<stem>_explained.png`` in the same directory.
        top_k: Maximum number of top LITA words to highlight.

    Returns:
        Path to the saved ``<name>_explained.png`` file.
    """
    import io
    from pathlib import Path
    from PIL import Image, ImageDraw

    if isinstance(image_input, (str, Path)):
        src_path = Path(image_input)
        img = Image.open(src_path).convert("RGB")
        if output_path is None:
            output_path = src_path.with_name(f"{src_path.stem}_explained.png")
    elif isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input)).convert("RGB")
        if output_path is None:
            output_path = Path("image_explained.png")
    else:
        img = image_input.copy().convert("RGB")
        if output_path is None:
            output_path = Path("image_explained.png")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _norm(token: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", token.strip().lower())

    target_tokens: List[str] = []
    for item in list(lita_top_words)[:top_k]:
        if isinstance(item, dict):
            w = str(item.get("word", ""))
        elif isinstance(item, (tuple, list)) and len(item) >= 1:
            w = str(item[0])
        else:
            w = str(item)
        norm_w = _norm(w)
        if norm_w and norm_w not in target_tokens:
            target_tokens.append(norm_w)

    draw = ImageDraw.Draw(img)
    for box in ocr_boxes:
        box_text = _norm(str(box.get("text", "")))
        if not box_text:
            continue
        matched = any(
            box_text == t or (len(t) >= 4 and (t in box_text or box_text in t))
            for t in target_tokens
        )
        if matched:
            x0 = int(box["left"]) - 2
            y0 = int(box["top"]) - 2
            x1 = int(box["left"]) + int(box["width"]) + 2
            y1 = int(box["top"]) + int(box["height"]) + 2
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

    img.save(out_path, format="PNG")
    return out_path

