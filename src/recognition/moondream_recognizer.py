from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

from src.models import CatalogItem
from .base import Recognizer
from .catalog import Catalog


class MoondreamQueryRecognizer(Recognizer):
    """
    Ensemble recognizer: weighted combination of image-to-image CLIP and
    Moondream text-to-catalog-text CLIP.

    Combined score per catalog item:
        img_weight  * cosine(crop_embed, ref_img_embed)
      + text_weight * cosine(moondream_answer_embed, catalog_text_embed)

    Image-to-image is the primary signal (more visually discriminative).
    Moondream text narrows ambiguous cases (e.g. two similar bottles).
    """

    _QUERY = (
        "What brand and product name is shown in this image? "
        "Be specific: e.g. 'Topo Chico mineral water bottle', 'Coca-Cola soda bottle', "
        "'SunChips Harvest Cheddar bag', 'Lay's Baked chips bag', "
        "'Kellogg's Froot Loops box', 'Kellogg's Frosted Mini Wheats box', "
        "'Jarritos Tamarind soda bottle', 'Jarritos Fruit Punch bottle'. "
        "Reply with just the product name."
    )

    def __init__(self, model, catalog: Catalog, fallback: Recognizer,
                 clip_recognizer, min_score: float = 0.15,
                 img_weight: float = 0.65):
        self._model = model
        self._catalog = catalog
        self._fallback = fallback
        self._clip = clip_recognizer
        self._min_score = min_score
        self._img_weight = img_weight
        self._text_weight = 1.0 - img_weight
        self._items = list(catalog)
        self._text_embeddings = self._build_text_embeddings()

    def _build_text_embeddings(self) -> dict[str, np.ndarray]:
        embs = {}
        for item in self._items:
            text = f"{item.name}. {item.description}"
            embs[item.id] = self._clip._embed_text(text)
        return embs

    def identify(self, crop: np.ndarray) -> tuple[CatalogItem | None, float]:
        # ── image embedding (always computed) ────────────────────────────────
        crop_emb = self._clip.embed(crop)

        # ── Moondream text answer (best-effort) ───────────────────────────────
        answer_emb: np.ndarray | None = None
        try:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            enc = self._model.encode_image(pil)
            answer: str = self._model.query(enc, self._QUERY)["answer"].strip()
            if answer and len(answer) >= 3:
                answer_emb = self._clip._embed_text(answer)
        except Exception:
            pass

        # ── ensemble score per catalog item ───────────────────────────────────
        best_item: CatalogItem | None = None
        best_score = -1.0

        for item in self._items:
            score = 0.0

            if item.ref_embedding is not None:
                score += self._img_weight * float(np.dot(crop_emb, item.ref_embedding))

            if answer_emb is not None:
                ref_text = self._text_embeddings.get(item.id)
                if ref_text is not None:
                    score += self._text_weight * float(np.dot(answer_emb, ref_text))

            if score > best_score:
                best_score = score
                best_item = item

        if best_score >= self._min_score:
            return best_item, best_score
        return None, best_score

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return self._clip.embed(crop)
