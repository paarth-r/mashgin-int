from __future__ import annotations
import math
from src.config import Config
from src.models import Detection, SceneResult
from src.scene import Scene
from src.detection.base import Detector
from src.recognition.base import Recognizer
from src.matching.base import Matcher


def _cameras_required(confidence: float, min_cams: int,
                      curve_start: float, floor_cams: int) -> int:
    """
    Step-function curve: at low confidence, require min_cams cameras.
    As confidence rises above curve_start, the requirement decreases linearly
    (with ceiling rounding) down to floor_cams.

    Example (min_cams=2, curve_start=0.65, floor_cams=1):
        conf < 0.65  → 2 cameras required
        conf = 0.825 → ceil(2 * 0.5) = 1  camera required
        conf = 1.0   → max(1, ceil(0))  = 1  camera required
    """
    if confidence <= curve_start or min_cams <= floor_cams:
        return min_cams
    t = (confidence - curve_start) / max(1e-9, 1.0 - curve_start)  # 0→1
    return max(floor_cams, math.ceil(min_cams * (1.0 - t)))


class Pipeline:
    """
    Orchestrates the four-stage vision pipeline for a single scene.

    Stages:
        detect (on warped top-down image)
        → canvas→world projection  (linear, no parallax)
        → ensemble CLIP recognition (image-to-image + Moondream text-text)
        → cross-camera match/dedupe
        → confidence-curve camera filter
    """

    def __init__(self, detector: Detector, recognizer: Recognizer,
                 matcher: Matcher, config: Config):
        self._detector   = detector
        self._recognizer = recognizer
        self._matcher    = matcher
        self._config     = config

    def run(self, scene: Scene) -> SceneResult:
        detections: list[Detection] = []

        for pair_id, cam in scene.cameras.items():
            warped = cam.warped_image()
            raw_boxes = self._detector.detect(warped)

            for bbox in raw_boxes:
                world_xy = cam.canvas_to_world(bbox.bottom_center)
                if not cam.in_mat_bounds(world_xy):
                    continue

                crop = cam.crop_warped(bbox)
                catalog_item, rec_score, embedding = None, 0.0, None
                if crop.size > 0:
                    catalog_item, rec_score = self._recognizer.identify(crop)
                    embedding = self._recognizer.embed(crop)

                det = Detection(
                    pair_id=pair_id,
                    bbox=bbox,
                    world_xy=world_xy,
                    crop_embedding=embedding,
                    catalog_item=catalog_item,
                    recognition_score=rec_score,
                )
                detections.append(det)

        products = self._matcher.match(detections)

        cfg = self._config
        products = [
            p for p in products
            if len(p.cameras) >= _cameras_required(
                p.confidence,
                cfg.min_cameras,
                cfg.conf_curve_start,
                cfg.floor_cameras,
            )
        ]
        return SceneResult(scene_id=scene.scene_id, items=products)
