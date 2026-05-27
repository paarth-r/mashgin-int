# Interview Talking Points

## Scenes 3–10 Quality

- Three root causes: edge blur from warp, planar homography breaks for tall items, glass bottles look identical from above
- Jarritos vs Topo Chico failure: labels are on the sides — invisible from top-down
- Fix priority: multi-view reference embeddings (per SKU, all 4 angles), then detect on raw images, then add ground truth + tune thresholds against precision/recall
- Scenes 1–2 perfect because items are flat, spread out, visually distinct

---

## Warped Images

- Have actual examples to show (all 4 cameras, scene_01)
- Key property: same physical item appears at same canvas pixel in all 4 warped views → deduplication is just world distance
- Gray regions = outside camera FOV
- Pair_0 and pair_2 are steep-angle → more edge blurring in warped output
- Trade-off: geometric simplicity vs resolution loss at edges

---

## Warping Before Detection

- Not strictly wrong — gives accurate geometry for flat items
- Main problem: Moondream trained on natural images, not top-down views → distribution shift
- Resolution degrades at warp edges → misses items there
- Right approach: detect on raw images (better quality), project bottom-center through homography for world position, warp only for recognition crops

---

## Bottom-Center & Occlusion

- Heuristic assumes z=0 (item touches mat) — valid only for unoccluded flat items
- Mitigated in practice: 4 cameras give redundant views; if one camera gets a bad bottom estimate, others correct it via averaging
- Breaks for stacked items — bottom of bbox is the visible portion, not the mat contact
- Better approach: flip dedup to appearance-first, geometry-second (see Dedup section)
- Proper fix: triangulation or Z estimation (see below)

---

## Triangulation

- Calibration files already contain everything needed:
  - 85 dot correspondences per camera (pixel → world mm)
  - `world_params_yaml` = camera-to-world 4×4 extrinsic (camera heights: pair_0=326mm, pair_1=529mm, pair_2=322mm, pair_3=521mm above mat)
- Missing piece: intrinsics K — estimable via `cv2.calibrateCamera` on those same 85 points
- Pipeline: invert 4×4 to get [R|t] → build P = K[R|t] → triangulate per detection → get true (X, Y, Z)

**The hard part: what 2D point do you pass to triangulatePoints?**
- Need the *same physical 3D point* visible in two cameras — bounding boxes don't give you that
- Bbox center: different cameras see different faces of the item → different 3D points → garbage result
- Bottom-center: better for Z estimation but breaks with occlusion
- Feature matching (SIFT/SuperPoint) within bbox regions: finds real correspondences but adds complexity, fails on textureless packaging

**Practical alternative — estimate Z without triangulation:**
- Warp already gives correct (X, Y) for z=0 items
- Given (X, Y) and camera extrinsics, find the Z that minimizes reprojection error against observed bbox centers across all cameras
- 1D search per item, no correspondence needed, still gives stacking height
- Simpler than full triangulation, uses same calibration data

---

## Deduplication Ideas

**Current approach:** world distance gates → CLIP similarity confirms. Greedy agglomerative. Assumes z=0.

**Ideas ranked by implementation cost:**

1. **Warped canvas IoU** — bbox overlap in the shared warped coordinate system. Same item → overlapping boxes. ~3 lines of code, geometrically correct for non-stacked items. Start here.

2. **Appearance-first (flip current logic)** — cluster by CLIP similarity first, use world distance as consistency check. Immediately handles stacked items, no z=0 assumption.

3. **Catalog-first grouping** — run recognition on all detections first, group by catalog ID, use geometry only to resolve conflicts (e.g. two Coca-Colas). Semantically correct if recognition is reliable.

4. **Graph matching** — nodes = detections, edge weights = similarity score. Spectral clustering or connected components. Not order-dependent like greedy.

5. **Hungarian algorithm per camera pair** — optimal 1-to-1 assignment for each of the 6 camera pairs, aggregate by vote. Prevents one bad pair from forcing a wrong merge.

6. **Epipolar constraints** — detection in camera A constrains where corresponding detection must appear in camera B (epipolar line). Pure geometry, no appearance. Requires K (same step as triangulation).

7. **Probabilistic / GMM** — model item positions as a Gaussian mixture, run EM. Handles noisy world positions, gives principled confidence scores.

**Would implement first:** warped canvas IoU (#1) + catalog-first (#3).

---

## Speed Budgets

- Current: ~4s, bottleneck is 4 cameras × 6 Moondream prompts = 24 sequential inference calls
- **<1s**: swap Moondream detector for YOLOv8-nano (~15ms/image CPU) + keep CLIP for recognition → ~250ms total, no GPU needed
- **<200ms**: GPU + TensorRT/CoreML (YOLO ~2ms, CLIP ~2ms, batch all 4 cameras) → ~20ms; or CPU-only with fine-tuned MobileNetV3 on 8 SKUs + classical CV detection → ~50ms. Fixed installation = can precompute per-camera, per-SKU appearance templates at startup.

---

## Detector Bugs (if it comes up)

- NMS sorts by area → large false positive boxes suppress correct smaller detections; should prefer tighter boxes
- Missing bbox keys default to 0/1 → full-image false positive box that passes area filter and pollutes NMS
- Exception handling only wraps `model.detect()`, not result parsing — can still throw
- 6 prompts are redundant ("packaged food" subsumes most others) → 6× inference cost for marginal gain
- Unused `self._tokenizer` stored on detector class
