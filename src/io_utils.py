from __future__ import annotations
import json
import os
from src.models import SceneResult


def write_output_json(result: SceneResult, scene_dir: str, outputs_dir: str | None = None) -> None:
    """Write output.json into scene_dir and optionally into outputs_dir."""
    data = result.to_dict()
    _write(data, os.path.join(scene_dir, "output.json"))
    if outputs_dir:
        os.makedirs(outputs_dir, exist_ok=True)
        _write(data, os.path.join(outputs_dir, f"{result.scene_id}.json"))


def _write(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")
