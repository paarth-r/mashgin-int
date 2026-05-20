from __future__ import annotations
import numpy as np
import cv2
import open_clip
import torch
from PIL import Image

from src.config import Config
from src.models import CatalogItem
from .base import Recognizer
from .catalog import Catalog


class ClipRecognizer(Recognizer):
    """
    Image-to-image CLIP recognizer with text-embedding fallback.

    Priority:
      1. Average of ref crop image embeddings  (best accuracy)
      2. CLIP text embedding from item description  (automatic fallback)
    This means recognition works immediately even with no ref crops.
    """

    def __init__(self, config: Config, catalog: Catalog):
        self._config = config
        self._catalog = catalog
        self._device = config.device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            config.clip_model_name,
            pretrained=config.clip_pretrained,
            device=self._device,
        )
        self._tokenizer = open_clip.get_tokenizer(config.clip_model_name)
        self._model.eval()
        self._build_catalog_embeddings()

    def _build_catalog_embeddings(self) -> None:
        ref_images = self._catalog.load_ref_images(self._config.catalog_refs_dir)
        for item in self._catalog:
            bgr_list = ref_images.get(item.id, [])
            if bgr_list:
                embs = [self._embed_bgr(img) for img in bgr_list]
                avg = np.stack(embs).mean(axis=0)
                item.ref_embedding = avg / (np.linalg.norm(avg) + 1e-8)
            else:
                # fall back to text embedding from description
                item.ref_embedding = self._embed_text(item.description)

    def identify(self, crop: np.ndarray) -> tuple[CatalogItem | None, float]:
        emb = self.embed(crop)
        best_item: CatalogItem | None = None
        best_score = -1.0
        for item in self._catalog:
            if item.ref_embedding is None:
                continue
            score = float(np.dot(emb, item.ref_embedding))
            if score > best_score:
                best_score = score
                best_item = item
        if best_score < self._config.min_recognition_score:
            return None, best_score
        return best_item, best_score

    def embed(self, crop: np.ndarray) -> np.ndarray:
        return self._embed_bgr(crop)

    # ── private ───────────────────────────────────────────────────────────────

    def _embed_bgr(self, bgr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tensor = self._preprocess(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_image(tensor)
        feat = feat.cpu().numpy()[0].astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)

    def _embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._model.encode_text(tokens)
        feat = feat.cpu().numpy()[0].astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)
