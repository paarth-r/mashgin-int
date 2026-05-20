from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BBox:
    """Axis-aligned bounding box in image pixel space."""
    x: float  # left
    y: float  # top
    w: float
    h: float

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Contact point with the mat — near z=0, minimises parallax error."""
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> float:
        return self.w * self.h

    def to_dict(self) -> dict:
        return {"x": round(self.x), "y": round(self.y),
                "w": round(self.w), "h": round(self.h)}

    def iou(self, other: "BBox") -> float:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.w, other.x + other.w)
        y2 = min(self.y + self.h, other.y + other.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class CatalogItem:
    """A single product in the catalog."""
    id: str
    name: str
    description: str
    packaging_type: str
    size: Optional[str] = None
    upc: Optional[str] = None
    ref_image_paths: list[str] = field(default_factory=list)
    # populated at runtime by ClipRecognizer
    ref_embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Detection:
    """One detected item in one camera frame."""
    pair_id: int
    bbox: BBox
    world_xy: Optional[tuple[float, float]] = None   # None if outside mat
    crop_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    catalog_item: Optional[CatalogItem] = None
    recognition_score: float = 0.0


@dataclass
class MatchedProduct:
    """A deduplicated product across cameras."""
    catalog_item: Optional[CatalogItem]
    cameras: list[int]
    camera_detections: dict[int, BBox]  # pair_id -> BBox
    world_position: Optional[tuple[float, float]]
    confidence: float
    # fallback fields when catalog_item is None
    fallback_name: str = "Unknown Item"

    def to_dict(self) -> dict:
        ci = self.catalog_item
        d: dict = {
            "name": ci.name if ci else self.fallback_name,
            "description": ci.description if ci else "",
            "packaging_type": ci.packaging_type if ci else "unknown",
            "cameras": self.cameras,
            "camera_detections": {
                str(k): v.to_dict() for k, v in self.camera_detections.items()
            },
            "confidence": round(self.confidence, 3),
        }
        if ci and ci.size:
            d["size"] = ci.size
        if ci and ci.upc:
            d["upc"] = ci.upc
        if self.world_position:
            d["world_position"] = {
                "x_mm": round(self.world_position[0], 1),
                "y_mm": round(self.world_position[1], 1),
            }
        return d


@dataclass
class SceneResult:
    scene_id: str
    items: list[MatchedProduct]

    def to_dict(self) -> dict:
        return {
            "scene": self.scene_id,
            "items": [item.to_dict() for item in self.items],
        }
