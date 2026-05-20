from __future__ import annotations
import json
import os
import numpy as np
import cv2
from PIL import Image

from src.models import CatalogItem


class Catalog:
    """
    Loads catalog.json and reference crops; stores embeddings.
    Embeddings are computed externally (by ClipRecognizer) and stored here.
    """

    def __init__(self, items: list[CatalogItem]):
        self._items = items

    @classmethod
    def load(cls, catalog_json: str) -> "Catalog":
        if not os.path.exists(catalog_json):
            return cls([])
        with open(catalog_json) as f:
            raw = json.load(f)
        items = []
        for entry in raw:
            item = CatalogItem(
                id=entry["id"],
                name=entry["name"],
                description=entry["description"],
                packaging_type=entry["packaging_type"],
                size=entry.get("size"),
                upc=entry.get("upc"),
                ref_image_paths=entry.get("ref_image_paths", []),
            )
            items.append(item)
        return cls(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def load_ref_images(self, refs_dir: str) -> dict[str, list[np.ndarray]]:
        """
        Returns {item_id: [bgr_array, ...]} for all reference images.

        Two sources are merged for each item (duplicates skipped by path):
          1. Folder scan: catalog/refs/<item_id>/ — drop any images here,
             no catalog.json edits needed.
          2. Explicit list: ref_image_paths in catalog.json (paths relative
             to refs_dir), for backward compatibility.
        """
        result: dict[str, list[np.ndarray]] = {}
        for item in self._items:
            seen: set[str] = set()
            imgs: list[np.ndarray] = []

            # 1 — auto-scan per-item subfolder
            folder = os.path.join(refs_dir, item.id)
            if os.path.isdir(folder):
                for fname in sorted(os.listdir(folder)):
                    if os.path.splitext(fname)[1].lower() in self._IMG_EXTS:
                        full = os.path.join(folder, fname)
                        seen.add(full)
                        img = cv2.imread(full)
                        if img is not None:
                            imgs.append(img)

            # 2 — explicit paths from catalog.json
            for p in item.ref_image_paths:
                full = os.path.join(refs_dir, p)
                if full in seen:
                    continue
                seen.add(full)
                if os.path.exists(full):
                    img = cv2.imread(full)
                    if img is not None:
                        imgs.append(img)

            result[item.id] = imgs
        return result
