#!/usr/bin/env python3
"""
Run the vision pipeline over all scenes and write output.json per scene.

Usage:
    python run.py [--data data] [--out outputs] [--scenes scene_01 ...]
"""
from __future__ import annotations
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.config import Config
from src.scene import Scene
from src.detection.moondream import MoondreamDetector
from src.recognition.catalog import Catalog
from src.recognition.clip_recognizer import ClipRecognizer
from src.recognition.moondream_recognizer import MoondreamQueryRecognizer
from src.matching.spatial_matcher import GeometryClipMatcher
from src.pipeline import Pipeline
from src.io_utils import write_output_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--scenes", nargs="*")
    args = parser.parse_args()

    config = Config()
    data_dir = os.path.join(_ROOT, args.data)
    outputs_dir = os.path.join(_ROOT, args.out)

    all_scenes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("scene_")
    )
    target_scenes = args.scenes or all_scenes
    print(f"Scenes: {target_scenes}")

    print("Loading models...")
    detector = MoondreamDetector(config)

    catalog = Catalog.load(os.path.join(_ROOT, config.catalog_json))
    clip_rec = ClipRecognizer(config, catalog)
    # Use Moondream's own query() for naming — more accurate for bottles
    recognizer = MoondreamQueryRecognizer(
        model=detector._model,
        catalog=catalog,
        fallback=clip_rec,
        clip_recognizer=clip_rec,
        min_score=0.15,
        img_weight=config.recognition_img_weight,
    )

    matcher = GeometryClipMatcher(config)
    pipeline = Pipeline(detector, recognizer, matcher, config)

    for scene_name in target_scenes:
        scene_dir = os.path.join(data_dir, scene_name)
        print(f"\n── {scene_name} ──")
        try:
            scene = Scene.load(scene_dir, config)
            result = pipeline.run(scene)
            print(f"  {len(result.items)} item(s)")
            for item in result.items:
                name = item.catalog_item.name if item.catalog_item else item.fallback_name
                print(f"    [{name}] cameras={item.cameras} conf={item.confidence}")
            write_output_json(result, scene_dir, outputs_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
