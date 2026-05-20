#!/usr/bin/env python3
"""
Bootstrap catalog: run Moondream on a few scenes, save crops to catalog/refs/,
then print a template catalog.json to hand-label.

After running this script:
  1. Inspect the crops in catalog/refs/raw/
  2. For each distinct product, pick the cleanest crop and copy/rename it to
     catalog/refs/<product_id>.jpg
  3. Fill out catalog/catalog.json with the correct fields

Usage:
    python build_catalog.py --scenes scene_01 scene_02 --data data
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import cv2

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.config import Config
from src.scene import Scene
from src.detection.moondream import MoondreamDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--scenes", nargs="*", default=["scene_01", "scene_02"])
    parser.add_argument("--max-per-camera", type=int, default=10)
    args = parser.parse_args()

    config = Config()
    raw_dir = os.path.join(_ROOT, "catalog", "refs", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Loading Moondream2...")
    detector = MoondreamDetector(config)

    saved = 0
    for scene_name in args.scenes:
        scene_dir = os.path.join(_ROOT, args.data, scene_name)
        scene = Scene.load(scene_dir, config)
        print(f"\n── {scene_name}")
        for pair_id, cam in scene.cameras.items():
            boxes = detector.detect(cam.image)
            # filter to on-mat detections
            on_mat = []
            for b in boxes:
                wxy = cam.project(b.bottom_center)
                if cam.in_mat_bounds(wxy):
                    on_mat.append(b)

            for i, bbox in enumerate(on_mat[: args.max_per_camera]):
                crop = cam.crop(bbox)
                if crop.size == 0:
                    continue
                fname = f"{scene_name}_cam{pair_id}_{i:02d}.jpg"
                cv2.imwrite(os.path.join(raw_dir, fname), crop)
                saved += 1
                print(f"  saved {fname}")

    print(f"\nSaved {saved} raw crops to {raw_dir}")
    print("\nNext steps:")
    print("  1. Review crops in catalog/refs/raw/")
    print("  2. Copy best crop per unique product to catalog/refs/<id>.jpg")
    print("  3. Update catalog/catalog.json with entries (see template printed below)")
    print()

    # print a minimal template so the user can fill it in
    template = [
        {
            "id": "example_id",
            "name": "Product Name",
            "description": "Color, shape, branding description",
            "packaging_type": "can|bottle|bag|box|wrapper|cup",
            "size": "12oz",
            "upc": None,
            "ref_image_paths": ["example_id.jpg"],
        }
    ]
    print("── catalog.json template ──")
    print(json.dumps(template, indent=2))


if __name__ == "__main__":
    main()
