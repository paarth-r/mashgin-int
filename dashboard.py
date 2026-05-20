"""
Tuning dashboard for the multi-camera checkout vision pipeline.

Launch:
    .venv/bin/streamlit run dashboard.py
"""
from __future__ import annotations
import os, sys, time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

st.set_page_config(page_title="Pipeline Tuner", layout="wide")

# ── load models once ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Moondream2 + CLIP…")
def load_models():
    """Returns heavy model objects only — recognizer is built per-run so sliders apply."""
    from src.config import Config
    from src.detection.moondream import _load_moondream
    from src.recognition.catalog import Catalog
    from src.recognition.clip_recognizer import ClipRecognizer
    cfg = Config()
    md_model, _ = _load_moondream(cfg.moondream_model_id, cfg.device)
    catalog = Catalog.load(os.path.join(_ROOT, cfg.catalog_json))
    clip_rec = ClipRecognizer(cfg, catalog)
    return md_model, clip_rec, catalog

# ── sidebar: sliders ──────────────────────────────────────────────────────────
st.sidebar.title("Parameters")

st.sidebar.subheader("Detection")
detection_prompts = st.sidebar.multiselect(
    "Moondream prompts",
    ["packaged food item", "bottle", "bag of chips", "cereal box",
     "small cereal box", "snack package", "can", "box"],
    default=["packaged food item", "bottle", "bag of chips",
             "cereal box", "small cereal box", "snack package"],
)
nms_iou      = st.sidebar.slider("NMS IoU threshold",     0.1, 0.9, 0.4, 0.05)
min_box_area = st.sidebar.slider("Min box area (px²)",    500, 10000, 2000, 500)

st.sidebar.subheader("Geometry")
mat_margin   = st.sidebar.slider("Mat margin (mm)",        0.0, 120.0, 60.0, 5.0)

st.sidebar.subheader("Recognition")
img_weight = st.sidebar.slider(
    "Image vs text weight",
    0.0, 1.0, 0.65, 0.05,
    help="1.0 = pure image-to-image CLIP; 0.0 = pure Moondream text-text CLIP. "
         "Higher = more visually grounded (better for similar-looking bottles).",
)

st.sidebar.subheader("Deduplication")
world_dist   = st.sidebar.slider("World dist threshold (mm)", 20.0, 250.0, 80.0, 5.0)
min_cameras  = st.sidebar.slider("Min cameras to survive",    1, 4, 2, 1)

st.sidebar.subheader("Confidence curve")
conf_curve_start = st.sidebar.slider(
    "Curve start confidence",
    0.30, 0.95, 0.65, 0.05,
    help="Above this confidence, the required camera count starts dropping.",
)
floor_cameras = st.sidebar.slider(
    "Floor cameras",
    1, 4, 1, 1,
    help="Minimum cameras required even at maximum confidence.",
)

# ── curve preview ─────────────────────────────────────────────────────────────
with st.sidebar.expander("Camera curve preview", expanded=False):
    import math as _math
    import plotly.graph_objects as _go

    _confs = [c / 100 for c in range(0, 101)]
    _reqs  = []
    for _c in _confs:
        if _c <= conf_curve_start or min_cameras <= floor_cameras:
            _reqs.append(min_cameras)
        else:
            _t = (_c - conf_curve_start) / max(1e-9, 1.0 - conf_curve_start)
            _reqs.append(max(floor_cameras, _math.ceil(min_cameras * (1.0 - _t))))

    _curve_fig = _go.Figure(_go.Scatter(
        x=_confs, y=_reqs, mode="lines",
        line=dict(color="#4a90e2", width=2, shape="hv"),
    ))
    _curve_fig.update_layout(
        xaxis=dict(title="Confidence", tickformat=".0%", range=[0, 1]),
        yaxis=dict(title="Cameras required", dtick=1,
                   range=[0, min_cameras + 0.5]),
        height=180, margin=dict(l=35, r=10, t=10, b=35),
    )
    st.plotly_chart(_curve_fig, use_container_width=True)

# ── scene picker ──────────────────────────────────────────────────────────────
data_dir = os.path.join(_ROOT, "data")
scenes = sorted(d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("scene_"))

st.title("Mashgin Pipeline Tuner")
selected_scene = st.selectbox("Scene", scenes)
run_btn = st.button("▶ Run pipeline", type="primary")

# ── color palette ─────────────────────────────────────────────────────────────
_PALETTE: dict[str, tuple] = {}
_rng = np.random.RandomState(42)

def color_for(name: str) -> tuple[int, int, int]:
    if name not in _PALETTE:
        _PALETTE[name] = tuple(int(c) for c in _rng.randint(80, 230, 3))
    return _PALETTE[name]  # type: ignore[return-value]

def bgr_to_hex(col: tuple[int, int, int]) -> str:
    return f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"


def draw_boxes(image_bgr: np.ndarray, detections, highlighted_bboxes=None) -> np.ndarray:
    vis = image_bgr.copy()
    # draw non-highlighted first so highlights render on top
    ordered = sorted(detections, key=lambda d: bool(highlighted_bboxes and d.bbox in highlighted_bboxes))
    for det in ordered:
        b = det.bbox
        name = det.catalog_item.name if det.catalog_item else "UNKNOWN"
        col = color_for(name)
        x1, y1, x2, y2 = int(b.x), int(b.y), int(b.x + b.w), int(b.y + b.h)
        is_hi = bool(highlighted_bboxes and det.bbox in highlighted_bboxes)

        if is_hi:
            # bright white halo, then thick colored rect
            cv2.rectangle(vis, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 4)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 5)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

        label = f"{name.split()[0]} {det.recognition_score:.2f}"
        font_scale = 0.65 if is_hi else 0.45
        thickness  = 2    if is_hi else 1
        cv2.putText(vis, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, thickness, cv2.LINE_AA)
    return vis


# ── run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running…"):
        t0 = time.time()
        md_model, clip_rec, catalog = load_models()

        from src.config import Config
        from src.scene import Scene, _compute_mat_bounds
        from src.models import Detection, BBox
        from src.detection.moondream import _nms
        from src.matching.spatial_matcher import GeometryClipMatcher
        from src.recognition.moondream_recognizer import MoondreamQueryRecognizer
        from PIL import Image as PILImage

        cfg = Config()
        cfg.detection_prompts       = detection_prompts or cfg.detection_prompts
        cfg.nms_iou_threshold       = nms_iou
        cfg.min_box_area            = min_box_area
        cfg.mat_margin_mm           = mat_margin
        cfg.world_dist_threshold_mm = world_dist
        cfg.min_cameras             = min_cameras
        cfg.conf_curve_start        = conf_curve_start
        cfg.floor_cameras           = floor_cameras
        cfg.recognition_img_weight  = img_weight

        # Build recognizer here so img_weight slider applies immediately
        query_rec = MoondreamQueryRecognizer(
            model=md_model, catalog=catalog, fallback=clip_rec,
            clip_recognizer=clip_rec, min_score=0.15,
            img_weight=img_weight,
        )

        scene = Scene.load(os.path.join(data_dir, selected_scene), cfg)
        matcher = GeometryClipMatcher(cfg)
        all_dets: list[Detection] = []
        cam_dets: dict[int, list[Detection]] = {}

        for pair_id, cam in sorted(scene.cameras.items()):
            warped = cam.warped_image()
            pil = PILImage.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            enc = md_model.encode_image(pil)
            h, w = warped.shape[:2]

            raw_boxes: list[BBox] = []
            for prompt in cfg.detection_prompts:
                try:
                    result = md_model.detect(enc, prompt)
                    objs = result.get("objects", []) if isinstance(result, dict) else result
                    for obj in objs:
                        x1 = obj.get("x_min", obj.get("xmin", 0))
                        y1 = obj.get("y_min", obj.get("ymin", 0))
                        x2 = obj.get("x_max", obj.get("xmax", 1))
                        y2 = obj.get("y_max", obj.get("ymax", 1))
                        bw, bh = (x2 - x1) * w, (y2 - y1) * h
                        if bw > 0 and bh > 0:
                            raw_boxes.append(BBox(x1 * w, y1 * h, bw, bh))
                except Exception:
                    pass

            raw_boxes = [b for b in raw_boxes if b.area >= cfg.min_box_area]
            boxes = _nms(raw_boxes, cfg.nms_iou_threshold)

            dets: list[Detection] = []
            for bbox in boxes:
                world_xy = cam.canvas_to_world(bbox.bottom_center)
                if not cam.in_mat_bounds(world_xy):
                    continue
                crop = cam.crop_warped(bbox)
                ci, score, emb = None, 0.0, None
                if crop.size > 0:
                    ci, score = query_rec.identify(crop)
                    emb = clip_rec.embed(crop)
                det = Detection(pair_id=pair_id, bbox=bbox, world_xy=world_xy,
                                crop_embedding=emb, catalog_item=ci,
                                recognition_score=score)
                det._crop_bgr = crop  # stash for crop browser
                dets.append(det)
                all_dets.append(det)
            cam_dets[pair_id] = dets

        products = matcher.match(all_dets)
        products = [p for p in products if len(p.cameras) >= cfg.min_cameras]
        mat_bounds = _compute_mat_bounds(cfg.mat_margin_mm)
        elapsed = time.time() - t0

        st.session_state.update({
            "products":       products,
            "cam_dets":       cam_dets,
            "all_dets":       all_dets,
            "scene":          scene,
            "mat_bounds":     mat_bounds,
            "elapsed":        elapsed,
            "run_scene":      selected_scene,
        })

# ── bail if nothing run yet ───────────────────────────────────────────────────
if "products" not in st.session_state or st.session_state.get("run_scene") != selected_scene:
    st.info("Pick a scene and click **▶ Run pipeline**.")
    st.stop()

products   = st.session_state["products"]
cam_dets   = st.session_state["cam_dets"]
all_dets   = st.session_state["all_dets"]
scene      = st.session_state["scene"]
mat_bounds = st.session_state["mat_bounds"]
elapsed    = st.session_state["elapsed"]

st.success(f"Done in {elapsed:.1f}s — **{len(products)} unique item(s)**")

# ── build bbox→product lookup (BBox objects are shared by identity) ────────────
bbox_to_product = {}
for prod in products:
    for pid, bbox in prod.camera_detections.items():
        bbox_to_product[id(bbox)] = prod

# ── world map + item table ────────────────────────────────────────────────────
st.subheader("World map & final items")
col_map, col_items = st.columns([2, 3])

with col_items:
    if not products:
        st.warning("No items after dedup — try lowering **Min cameras**.")
        selected_prod = None
    else:
        rows = []
        for prod in products:
            name = prod.catalog_item.name if prod.catalog_item else "Unknown"
            wp = prod.world_position
            rows.append({
                "Item":       name,
                "Cameras":    str(sorted(prod.cameras)),
                "Conf":       round(prod.confidence, 2),
                "World (mm)": f"({wp[0]:.0f}, {wp[1]:.0f})" if wp else "–",
                "UPC":        prod.catalog_item.upc if prod.catalog_item else "–",
            })
        df = pd.DataFrame(rows)
        st.caption("Click a row to highlight that item in all camera views.")
        sel = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
        )
        sel_rows = sel.selection.rows
        selected_prod = products[sel_rows[0]] if sel_rows else None

with col_map:
    try:
        import plotly.graph_objects as go

        xmin, ymin, xmax, ymax = mat_bounds
        fig = go.Figure()

        # mat outline
        fig.add_shape(type="rect", x0=xmin, y0=ymin, x1=xmax, y1=ymax,
                      line=dict(color="#888", width=1.5),
                      fillcolor="rgba(200,200,200,0.05)")

        seen_names: set[str] = set()
        for prod in products:
            name = prod.catalog_item.name if prod.catalog_item else "Unknown"
            col  = color_for(name)
            hex_col = bgr_to_hex(col)
            wp = prod.world_position

            # per-camera detection dots
            for pid, bbox in prod.camera_detections.items():
                det = next((d for d in all_dets
                            if d.pair_id == pid and d.bbox is bbox), None)
                if det is None or det.world_xy is None:
                    continue
                wx, wy = det.world_xy
                fig.add_trace(go.Scatter(
                    x=[wx], y=[wy],
                    mode="markers",
                    marker=dict(size=9, color=hex_col, opacity=0.6,
                                line=dict(color="white", width=0.5)),
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        f"Camera {pid}<br>"
                        f"Detection: ({wx:.0f}, {wy:.0f}) mm<extra></extra>"
                    ),
                ))

            # centroid star — one legend entry per item name
            if wp:
                cx, cy = wp
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode="markers+text",
                    marker=dict(size=16, color=hex_col, symbol="star",
                                line=dict(color="white", width=1.5)),
                    text=[name.split()[0]],
                    textposition="top center",
                    textfont=dict(size=9, color=hex_col),
                    name=name,
                    legendgroup=name,
                    showlegend=(name not in seen_names),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        f"Centroid: ({cx:.0f}, {cy:.0f}) mm<br>"
                        f"Cameras: {sorted(prod.cameras)}<br>"
                        f"Confidence: {prod.confidence:.2f}<extra></extra>"
                    ),
                ))
                seen_names.add(name)

        fig.update_layout(
            xaxis=dict(title="x (mm)", range=[xmin - 20, xmax + 20]),
            yaxis=dict(title="y (mm)", range=[ymin - 20, ymax + 20],
                       scaleanchor="x", scaleratio=1),
            height=380,
            margin=dict(l=40, r=10, t=30, b=40),
            legend=dict(font=dict(size=9), itemsizing="constant"),
            title=dict(text="World-space detections (★ = centroid)", font=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Install plotly for the interactive map: `.venv/bin/pip install plotly`")

# ── 4 camera views ────────────────────────────────────────────────────────────
if selected_prod:
    name = selected_prod.catalog_item.name if selected_prod.catalog_item else "Unknown"
    st.subheader(f"Camera views — highlighting: **{name}**")
else:
    st.subheader("All 4 camera views (warped top-down)")

highlighted_bboxes: set | None = None
if selected_prod:
    highlighted_bboxes = set(selected_prod.camera_detections.values())

cols = st.columns(4)
for pair_id in range(4):
    cam = scene.cameras.get(pair_id)
    if cam is None:
        continue
    dets = cam_dets.get(pair_id, [])
    vis  = draw_boxes(cam.warped_image(), dets, highlighted_bboxes=highlighted_bboxes)
    with cols[pair_id]:
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 caption=f"Camera {pair_id} — {len(dets)} detection(s)",
                 use_container_width=True)

# ── crop browser ──────────────────────────────────────────────────────────────
st.subheader("Crop browser — save as reference images")
st.caption(
    "Each box below is one detection crop — exactly what CLIP sees. "
    "Correct the label if wrong, then click **Save** to add it to that item's reference folder."
)

catalog_item_ids  = [item.id   for item in catalog]
catalog_item_names = [item.name for item in catalog]
refs_dir = os.path.join(_ROOT, "catalog", "refs")

for det in all_dets:
    crop = getattr(det, "_crop_bgr", None)
    if crop is None or crop.size == 0:
        continue

    guessed_name = det.catalog_item.name if det.catalog_item else "Unknown"
    guessed_id   = det.catalog_item.id   if det.catalog_item else None
    default_idx  = catalog_item_ids.index(guessed_id) if guessed_id in catalog_item_ids else 0

    col_img, col_ctrl = st.columns([1, 3])
    with col_img:
        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_ctrl:
        st.markdown(
            f"**Cam {det.pair_id}** · guessed: *{guessed_name}* · score: `{det.recognition_score:.2f}`"
        )
        chosen_idx = st.selectbox(
            "Save as",
            range(len(catalog_item_ids)),
            index=default_idx,
            format_func=lambda i: catalog_item_names[i],
            key=f"label_{id(det)}",
        )
        if st.button("💾 Save crop", key=f"save_{id(det)}"):
            chosen_id = catalog_item_ids[chosen_idx]
            out_dir   = os.path.join(refs_dir, chosen_id)
            os.makedirs(out_dir, exist_ok=True)
            existing  = [f for f in os.listdir(out_dir)
                         if os.path.splitext(f)[1].lower() in {".jpg", ".png", ".jpeg", ".webp"}]
            fname = f"{selected_scene}_cam{det.pair_id}_{len(existing):02d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), crop)
            st.success(f"Saved → `catalog/refs/{chosen_id}/{fname}`")
