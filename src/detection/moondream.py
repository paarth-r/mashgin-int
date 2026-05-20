from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

from src.config import Config
from src.models import BBox
from .base import Detector


class MoondreamDetector(Detector):
    """
    Runs Moondream2 detect() for each configured prompt and NMS-merges results.
    Uses AutoModelForCausalLM (HuggingFace, trust_remote_code).
    Model is loaded once and reused across all scenes/cameras.
    """

    def __init__(self, config: Config):
        self._config = config
        self._model, self._tokenizer = _load_moondream(
            config.moondream_model_id, config.device
        )

    def detect(self, image: np.ndarray) -> list[BBox]:
        """Run all detection prompts on one image, return NMS-filtered BBoxes."""
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]

        # encode image once, reuse for all prompts
        encoded = self._model.encode_image(pil)

        all_boxes: list[BBox] = []
        for prompt in self._config.detection_prompts:
            boxes = self._detect_prompt(encoded, prompt, w, h)
            all_boxes.extend(boxes)

        all_boxes = [b for b in all_boxes if b.area >= self._config.min_box_area]
        return _nms(all_boxes, self._config.nms_iou_threshold)

    def _detect_prompt(self, encoded, prompt: str, img_w: int, img_h: int) -> list[BBox]:
        try:
            result = self._model.detect(encoded, prompt)
        except Exception:
            return []

        objects = result.get("objects", []) if isinstance(result, dict) else result
        boxes: list[BBox] = []
        for obj in objects:
            x_min = obj.get("x_min", obj.get("xmin", 0))
            y_min = obj.get("y_min", obj.get("ymin", 0))
            x_max = obj.get("x_max", obj.get("xmax", 1))
            y_max = obj.get("y_max", obj.get("ymax", 1))
            x = x_min * img_w
            y = y_min * img_h
            bw = (x_max - x_min) * img_w
            bh = (y_max - y_min) * img_h
            if bw > 0 and bh > 0:
                boxes.append(BBox(x=x, y=y, w=bw, h=bh))
        return boxes


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_moondream(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer


def _nms(boxes: list[BBox], iou_threshold: float) -> list[BBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b.area, reverse=True)
    kept: list[BBox] = []
    for box in boxes:
        if all(box.iou(k) < iou_threshold for k in kept):
            kept.append(box)
    return kept
