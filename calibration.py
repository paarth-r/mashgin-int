"""Camera calibration utilities.

Each camera has been calibrated using a circle pattern printed on the checkout
mat. The calibration data maps 85 known image points to their corresponding
positions on the mat surface (in millimeters).

This module provides utilities to:
- Load calibration data from JSON files
- Compute a homography matrix that maps image pixels to world coordinates
- Project any pixel coordinate to world (x, y) in millimeters

Usage:
    from calibration import load_calibrations, compute_homography, project_to_world

    calibrations = load_calibrations("scene_01/calibration")
    H = compute_homography(calibrations[0])
    world_x, world_y = project_to_world(H, (500, 400))
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraCalibration:
    """Calibration data for a single camera."""
    pair_id: int
    pattern_points: list[list[float]]  # 85 image-space 2D points [[x, y], ...]
    world_params_yaml: str             # OpenCV YAML string for a 4x4 matrix
    valid: bool = True


def get_ideal_world_points_2d() -> np.ndarray:
    """Compute the 85 ideal circle-pattern world positions at z=0.

    Returns an Nx2 array of (x, y) in millimeters, centered at origin.
    The pattern is a 9x5 grid with 45 offset points, spaced 39mm apart.
    """
    unit = 39.0
    points = []

    for x in range(9):
        for y in range(5):
            points.append([x * unit, y * unit])
        if x < 8:
            for y in range(5):
                points.append([unit / 2 + x * unit, unit / 2 + y * unit])

    pts = np.array(points, dtype=np.float32)
    center = pts.mean(axis=0)
    pts -= center
    return pts


def parse_opencv_matrix(yaml_str: str) -> np.ndarray | None:
    """Parse an OpenCV YAML-serialized matrix into a numpy array.

    Args:
        yaml_str: OpenCV YAML string containing a matrix node named 'mat'.

    Returns:
        Numpy array, or None if parsing fails.
    """
    if not yaml_str:
        return None
    fs = cv2.FileStorage(yaml_str, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    if not fs.isOpened():
        return None
    try:
        return fs.getNode("mat").mat()
    finally:
        fs.release()


def load_calibrations(calibration_dir: str) -> dict[int, CameraCalibration]:
    """Load all calibration files from a directory.

    Args:
        calibration_dir: Path to directory containing pair_*.json files.

    Returns:
        Dict mapping pair_id -> CameraCalibration.
    """
    calibrations = {}

    for fname in sorted(os.listdir(calibration_dir)):
        if not fname.startswith("pair_") or not fname.endswith(".json"):
            continue
        path = os.path.join(calibration_dir, fname)
        with open(path) as f:
            data = json.load(f)

        calib = CameraCalibration(
            pair_id=data["pair_id"],
            pattern_points=data["pattern_points"],
            world_params_yaml=data.get("world_params_yaml", ""),
            valid=data.get("valid", True),
        )
        calibrations[calib.pair_id] = calib

    return calibrations


def compute_homography(calib: CameraCalibration) -> np.ndarray:
    """Compute homography mapping image pixels to world (x, y) on the mat.

    Uses the calibration pattern_points (image coordinates) and ideal world
    points (known positions at z=0) to compute the transformation.

    Args:
        calib: Camera calibration data.

    Returns:
        3x3 homography matrix H such that world_pt = H @ [u, v, 1].
    """
    image_pts = np.array(calib.pattern_points, dtype=np.float32)
    world_pts = get_ideal_world_points_2d()

    if len(image_pts) != len(world_pts):
        raise ValueError(
            f"Pattern points count mismatch for pair {calib.pair_id}: "
            f"{len(image_pts)} image pts vs {len(world_pts)} world pts"
        )

    H, mask = cv2.findHomography(image_pts, world_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError(f"Failed to compute homography for pair {calib.pair_id}")

    return H


def project_to_world(H: np.ndarray, image_point: tuple[int, int]) -> tuple[float, float]:
    """Project an image pixel to world (x, y) coordinates using a homography.

    Args:
        H: 3x3 homography matrix from compute_homography().
        image_point: (u, v) pixel coordinates in the image.

    Returns:
        (world_x, world_y) in millimeters.
    """
    pt = np.array([image_point[0], image_point[1], 1.0], dtype=np.float64)
    world = H @ pt
    return (world[0] / world[2], world[1] / world[2])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python calibration.py <scene_dir>")
        print("Example: python calibration.py data/scene_01")
        sys.exit(1)

    scene_dir = sys.argv[1]
    calib_dir = os.path.join(scene_dir, "calibration")
    calibrations = load_calibrations(calib_dir)
    print(f"Loaded {len(calibrations)} camera calibrations from {calib_dir}")

    for pair_id, calib in sorted(calibrations.items()):
        H = compute_homography(calib)
        # Project the center of a 2048x1536 image as a demo
        center = (1024, 768)
        wx, wy = project_to_world(H, center)
        print(f"  Pair {pair_id}: image center {center} -> world ({wx:.1f}, {wy:.1f}) mm")
