from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── model identifiers ──────────────────────────────────────────────────────
    moondream_model_id: str = "vikhyatk/moondream2"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # ── device ────────────────────────────────────────────────────────────────
    device: str = "mps"  # "mps" | "cuda" | "cpu"

    # ── detection ─────────────────────────────────────────────────────────────
    # Prompts passed to Moondream detect; runs one pass per prompt, then NMS
    detection_prompts: list[str] = field(default_factory=lambda: [
        "packaged food item",
        "bottle",
        "bag of chips",
        "cereal box",
        "small cereal box",
        "snack package",
    ])
    nms_iou_threshold: float = 0.4      # IoU above which overlapping boxes are merged
    min_box_area: int = 2000            # drop tiny boxes (px²) — calibration dots etc.

    # ── geometry / mat bounds ─────────────────────────────────────────────────
    mat_margin_mm: float = 60.0         # allow this many mm outside ideal pattern bounds

    # ── recognition ───────────────────────────────────────────────────────────
    min_recognition_score: float = 0.20 # below this CLIP score → unidentified item

    # ── recognition ensemble ──────────────────────────────────────────────────
    # Combined score = img_weight * image_clip + (1-img_weight) * text_clip
    # Higher img_weight → more visually grounded; lower → more semantic/text-driven
    recognition_img_weight: float = 0.65

    # ── matching / dedupe ─────────────────────────────────────────────────────
    world_dist_threshold_mm: float = 80.0  # max world distance to consider merging
    clip_sim_threshold: float = 0.82       # CLIP crop similarity to force merge
    # if catalog ids agree, world dist alone decides; otherwise both must hold.
    min_cameras: int = 2                   # drop items seen by fewer cameras than this
    # Confidence curve: required cameras drops from min_cameras → floor_cameras
    # as confidence rises from conf_curve_start → 1.0 (ceiling step function)
    conf_curve_start: float = 0.65        # confidence at which camera req starts relaxing
    floor_cameras: int = 1                # absolute minimum cameras required

    # ── paths ─────────────────────────────────────────────────────────────────
    catalog_json: str = "catalog/catalog.json"
    catalog_refs_dir: str = "catalog/refs"
