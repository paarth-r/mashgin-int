#!/usr/bin/env python3
"""
Extract reference crop images from scene data using classical CV (no ML).

Strategy:
  - Warp each camera image to a top-down mat view using the homography
  - In that rectified view, threshold to find non-background pixels
  - Extract blobs as item candidates; save crops to catalog/refs/raw/

Run this, then:
  1. Inspect catalog/refs/raw/  — pick the best crop per product
  2. Copy/rename to catalog/refs/<product_id>.jpg
  3. Update catalog.json ref_image_paths to include the filename

Usage:
    python make_ref_crops.py --scenes scene_01 scene_02 scene_03
"""
from __future__ import annotations
import argparse
import os
import sys
import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.config import Config
from src.scene import Scene, _compute_mat_bounds
from calibration import get_ideal_world_points_2d

# output canvas size for the rectified top-down view
CANVAS_W, CANVAS_H = 800, 600
PADDING_MM = 60.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--scenes", nargs="*", default=["scene_01", "scene_02", "scene_03"])
    parser.add_argument("--camera", type=int, default=1,
                        help="Which camera pair_id to use (1 or 3 are most top-down)")
    args = parser.parse_args()

    raw_dir = os.path.join(_ROOT, "catalog", "refs", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    config = Config()
    world_pts = get_ideal_world_points_2d()
    xmin, ymin = world_pts.min(axis=0) - PADDING_MM
    xmax, ymax = world_pts.max(axis=0) + PADDING_MM
    world_w = xmax - xmin
    world_h = ymax - ymin

    # canvas corners in world mm
    dst_pts = np.float32([
        [0, 0], [CANVAS_W, 0], [CANVAS_W, CANVAS_H], [0, CANVAS_H]
    ])
    # corresponding world corners
    src_world = np.float32([
        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
    ])

    saved = 0
    for scene_name in args.scenes:
        scene_dir = os.path.join(_ROOT, args.data, scene_name)
        scene = Scene.load(scene_dir, config)

        cam = scene.cameras.get(args.camera)
        if cam is None:
            print(f"  {scene_name}: camera {args.camera} not found, skipping")
            continue

        H_inv = np.linalg.inv(cam._H)

        # build canvas->image mapping: canvas pixel -> world mm -> image pixel
        # We want an inverse-homography warp: image -> canvas
        # H maps image->world; so H_inv maps world->image
        # We need: canvas pixel -> world mm -> image pixel
        # canvas_to_world: scale + translate
        scale_x = world_w / CANVAS_W
        scale_y = world_h / CANVAS_H

        # Build the combined transform: canvas pixel -> world -> image pixel
        # world = canvas_pt * scale + (xmin, ymin)
        # image = H_inv @ [world_x, world_y, 1]
        # We use remap for this

        map_x = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
        map_y = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
        for vc in range(CANVAS_H):
            for uc in range(CANVAS_W):
                wx = uc * scale_x + xmin
                wy = vc * scale_y + ymin
                pt = H_inv @ np.array([wx, wy, 1.0])
                map_x[vc, uc] = pt[0] / pt[2]
                map_y[vc, uc] = pt[1] / pt[2]

        warped = cv2.remap(cam.image, map_x, map_y,
                           cv2.INTER_LINEAR, borderValue=(128, 128, 128))

        blobs = _find_items(warped)
        print(f"  {scene_name} cam{args.camera}: {len(blobs)} blobs")

        for i, (x, y, w, h) in enumerate(blobs):
            crop = warped[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            fname = f"{scene_name}_cam{args.camera}_{i:02d}.jpg"
            cv2.imwrite(os.path.join(raw_dir, fname), crop)
            saved += 1

        # also save full warped view for reference
        cv2.imwrite(os.path.join(raw_dir, f"{scene_name}_cam{args.camera}_warped.jpg"), warped)

    print(f"\nSaved {saved} crops + warped views to {raw_dir}")
    print("\nNext: inspect the raw/ crops, pick best per product,")
    print("copy to catalog/refs/<product_id>.jpg, update catalog.json ref_image_paths.")


def _find_items(warped: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Segment non-mat pixels in the rectified top-down view."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # the mat is white (~200-255) with small black dots (~0-50)
    # items are colorful packages in the mid range
    # strategy: find pixels that are neither very bright nor very dark
    mask_not_white = gray < 210
    mask_not_black = gray > 40

    # also use color: items are more saturated than the neutral mat
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask_saturated = hsv[:, :, 1] > 50  # saturation > 50 = colorful

    mask = mask_not_white & mask_not_black & mask_saturated
    mask = mask.astype(np.uint8) * 255

    # morphology to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3000:  # skip tiny blobs
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # add small padding
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(CANVAS_W - x, w + 2 * pad)
        h = min(CANVAS_H - y, h + 2 * pad)
        blobs.append((x, y, w, h))

    return blobs


if __name__ == "__main__":
    main()
