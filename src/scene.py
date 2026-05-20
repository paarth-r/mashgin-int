from __future__ import annotations
import os
import sys
import numpy as np
import cv2
from PIL import Image

# allow running from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_INTERVIEW_ROOT = os.path.dirname(_HERE)
if _INTERVIEW_ROOT not in sys.path:
    sys.path.insert(0, _INTERVIEW_ROOT)

from calibration import (
    load_calibrations,
    compute_homography,
    project_to_world,
    get_ideal_world_points_2d,
    CameraCalibration,
)
from src.config import Config


class CameraView:
    """One camera's view of a scene: image, calibration, and homography."""

    # Warped canvas dimensions (shared across all cameras for a given scene)
    CANVAS_W: int = 800
    CANVAS_H: int = 600

    def __init__(
        self,
        pair_id: int,
        image: np.ndarray,
        calibration: CameraCalibration,
        homography: np.ndarray,
        mat_bounds: tuple[float, float, float, float],  # (xmin, ymin, xmax, ymax) mm
    ):
        self.pair_id = pair_id
        self.image = image                  # HxWx3 BGR
        self.calibration = calibration
        self._H = homography
        self._H_inv = np.linalg.inv(homography)
        self._mat_bounds = mat_bounds       # world mm bounds, with margin
        self._warped: np.ndarray | None = None          # cached warped image
        self._remap_x: np.ndarray | None = None         # cached remap tables
        self._remap_y: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        h, w = self.image.shape[:2]
        return w, h

    # ── raw-image helpers ─────────────────────────────────────────────────────

    def project(self, pixel: tuple[float, float]) -> tuple[float, float]:
        """Map a raw-image pixel to world (x_mm, y_mm) on the mat plane."""
        return project_to_world(self._H, pixel)

    def in_mat_bounds(self, world_xy: tuple[float, float]) -> bool:
        xmin, ymin, xmax, ymax = self._mat_bounds
        return xmin <= world_xy[0] <= xmax and ymin <= world_xy[1] <= ymax

    def crop(self, bbox) -> np.ndarray:
        """Crop raw image to BBox, clamped to image dimensions."""
        h, w = self.image.shape[:2]
        x1 = max(0, int(bbox.x))
        y1 = max(0, int(bbox.y))
        x2 = min(w, int(bbox.x + bbox.w))
        y2 = min(h, int(bbox.y + bbox.h))
        return self.image[y1:y2, x1:x2]

    # ── warped top-down view ──────────────────────────────────────────────────

    def warped_image(self) -> np.ndarray:
        """
        Return (and cache) the image warped to a top-down CANVAS_W×CANVAS_H view.
        Every canvas pixel maps linearly to world mm, eliminating parallax.
        """
        if self._warped is None:
            self._warped = self._build_warped()
        return self._warped

    def canvas_to_world(self, canvas_xy: tuple[float, float]) -> tuple[float, float]:
        """Map a canvas pixel to world mm. Linear: no homography needed."""
        xmin, ymin, xmax, ymax = self._mat_bounds
        wx = canvas_xy[0] * (xmax - xmin) / self.CANVAS_W + xmin
        wy = canvas_xy[1] * (ymax - ymin) / self.CANVAS_H + ymin
        return wx, wy

    def crop_warped(self, bbox) -> np.ndarray:
        """Crop the warped image to a BBox (canvas coordinates)."""
        warped = self.warped_image()
        h, w = warped.shape[:2]
        x1 = max(0, int(bbox.x))
        y1 = max(0, int(bbox.y))
        x2 = min(w, int(bbox.x + bbox.w))
        y2 = min(h, int(bbox.y + bbox.h))
        return warped[y1:y2, x1:x2]

    def pil_image(self) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def pil_warped(self) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(self.warped_image(), cv2.COLOR_BGR2RGB))

    # ── private ───────────────────────────────────────────────────────────────

    def _build_warped(self) -> np.ndarray:
        """Build the remap tables and produce the warped image."""
        xmin, ymin, xmax, ymax = self._mat_bounds
        scale_x = (xmax - xmin) / self.CANVAS_W
        scale_y = (ymax - ymin) / self.CANVAS_H
        map_x = np.zeros((self.CANVAS_H, self.CANVAS_W), dtype=np.float32)
        map_y = np.zeros((self.CANVAS_H, self.CANVAS_W), dtype=np.float32)
        for vc in range(self.CANVAS_H):
            for uc in range(self.CANVAS_W):
                wx = uc * scale_x + xmin
                wy = vc * scale_y + ymin
                pt = self._H_inv @ np.array([wx, wy, 1.0])
                map_x[vc, uc] = pt[0] / pt[2]
                map_y[vc, uc] = pt[1] / pt[2]
        return cv2.remap(
            self.image, map_x, map_y,
            cv2.INTER_LINEAR, borderValue=(128, 128, 128)
        )


class Scene:
    """All 4 camera views for one scene, with calibration loaded."""

    def __init__(self, scene_id: str, cameras: dict[int, CameraView]):
        self.scene_id = scene_id
        self.cameras = cameras  # pair_id -> CameraView

    @classmethod
    def load(cls, scene_dir: str, config: Config) -> "Scene":
        scene_id = os.path.basename(scene_dir.rstrip("/"))
        calib_dir = os.path.join(scene_dir, "calibration")
        images_dir = os.path.join(scene_dir, "images")

        calibrations = load_calibrations(calib_dir)
        mat_bounds = _compute_mat_bounds(config.mat_margin_mm)

        cameras: dict[int, CameraView] = {}
        for pair_id, calib in sorted(calibrations.items()):
            img_path = os.path.join(images_dir, f"pair_{pair_id}.jpg")
            if not os.path.exists(img_path):
                continue
            image = cv2.imread(img_path)
            if image is None:
                raise IOError(f"Cannot read image: {img_path}")
            H = compute_homography(calib)
            cameras[pair_id] = CameraView(pair_id, image, calib, H, mat_bounds)

        return cls(scene_id, cameras)


def _compute_mat_bounds(margin_mm: float) -> tuple[float, float, float, float]:
    pts = get_ideal_world_points_2d()
    xmin, ymin = pts.min(axis=0) - margin_mm
    xmax, ymax = pts.max(axis=0) + margin_mm
    return xmin, ymin, xmax, ymax
