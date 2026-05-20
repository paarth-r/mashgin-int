from __future__ import annotations
import numpy as np
from src.config import Config
from src.models import Detection, MatchedProduct
from .base import Matcher


class GeometryClipMatcher(Matcher):
    """
    Greedy agglomerative clustering of detections across cameras.

    Merge predicate (both must hold):
      1. world_dist(a, b) < dist_threshold_mm
      2. same catalog id  OR  CLIP crop similarity > sim_threshold
    """

    def __init__(self, config: Config):
        self._dist_thr = config.world_dist_threshold_mm
        self._sim_thr = config.clip_sim_threshold

    def match(self, detections: list[Detection]) -> list[MatchedProduct]:
        # only consider detections that have been projected to the mat
        valid = [d for d in detections if d.world_xy is not None]
        clusters: list[list[Detection]] = []

        for det in valid:
            placed = False
            for cluster in clusters:
                if self._should_merge(det, cluster):
                    cluster.append(det)
                    placed = True
                    break
            if not placed:
                clusters.append([det])

        return [self._build_product(c) for c in clusters]

    def _should_merge(self, det: Detection, cluster: list[Detection]) -> bool:
        """True if det matches *any* existing member of cluster."""
        for member in cluster:
            if _world_dist(det, member) < self._dist_thr and self._visual_match(det, member):
                return True
        return False

    def _visual_match(self, a: Detection, b: Detection) -> bool:
        same_catalog = (
            a.catalog_item is not None
            and b.catalog_item is not None
            and a.catalog_item.id == b.catalog_item.id
        )
        if same_catalog:
            return True
        if a.crop_embedding is not None and b.crop_embedding is not None:
            sim = float(np.dot(a.crop_embedding, b.crop_embedding))
            if sim >= self._sim_thr:
                return True
        return False

    def _build_product(self, cluster: list[Detection]) -> MatchedProduct:
        # choose catalog item by majority vote, then highest recognition score
        item = _vote_catalog_item(cluster)
        world_xy = _average_world(cluster)
        cameras = sorted({d.pair_id for d in cluster})
        cam_detections = {d.pair_id: d.bbox for d in cluster}
        # confidence: mean CLIP recognition score, boosted by camera count
        scores = [d.recognition_score for d in cluster if d.recognition_score > 0]
        clip_conf = float(np.mean(scores)) if scores else 0.0
        cam_factor = min(1.0, len(cameras) / 4.0)
        confidence = 0.6 * clip_conf + 0.4 * cam_factor

        return MatchedProduct(
            catalog_item=item,
            cameras=cameras,
            camera_detections=cam_detections,
            world_position=world_xy,
            confidence=round(confidence, 3),
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _world_dist(a: Detection, b: Detection) -> float:
    ax, ay = a.world_xy  # type: ignore[misc]
    bx, by = b.world_xy  # type: ignore[misc]
    return float(np.hypot(ax - bx, ay - by))


def _average_world(cluster: list[Detection]) -> tuple[float, float] | None:
    pts = [d.world_xy for d in cluster if d.world_xy is not None]
    if not pts:
        return None
    xs, ys = zip(*pts)
    return float(np.mean(xs)), float(np.mean(ys))


def _vote_catalog_item(cluster: list[Detection]):
    from collections import Counter
    counts: Counter = Counter()
    best_score: dict = {}
    for d in cluster:
        if d.catalog_item is None:
            continue
        cid = d.catalog_item.id
        counts[cid] += 1
        if d.recognition_score > best_score.get(cid, 0.0):
            best_score[cid] = d.recognition_score
    if not counts:
        return None
    # pick most-voted; break ties by score
    winner_id = max(counts, key=lambda cid: (counts[cid], best_score.get(cid, 0)))
    for d in cluster:
        if d.catalog_item and d.catalog_item.id == winner_id:
            return d.catalog_item
    return None
