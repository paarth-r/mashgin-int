# Vision Pipeline – Approach & Writeup

## Approach

### Overview

A four-stage pipeline processes each scene independently:

1. **Detect** – Run Moondream2 (local VLM on Apple MPS) on a rectified top-down view of each camera image to produce bounding boxes of items on the tray.
2. **Project** – Map each bbox's bottom-center from canvas coordinates to world millimeters using the provided calibration homography. Drop detections outside the mat bounds.
3. **Recognize** – For each on-mat detection, encode the crop with Moondream's `encode_image`, call `query()` asking for the product name, then match the natural-language answer against catalog entries using CLIP text embeddings.
4. **Match / deduplicate** – Cluster detections across all 4 cameras: two detections merge if their world-space distance is below a threshold and they share visual similarity (CLIP embedding cosine similarity) or the same catalog identity. Each cluster becomes one output item.

### Key Decisions

**Detect on the warped top-down image, not the raw camera image.**
Projecting the bottom of a bounding box in a raw angled-camera image to world space has large parallax error — a tall bottle's lower bbox edge is not at z=0. Warping each camera image to a fixed top-down canvas first makes the `canvas_xy → world_mm` mapping purely linear (no parallax), so the same physical item gets consistent world coordinates across all four cameras. This is the main reason deduplication works.

**Moondream query + CLIP text matching for recognition.**
CLIP image-to-image matching struggled because warped views from oblique cameras produce different crop appearances than our top-down reference images. Asking Moondream to describe each crop in natural language ("Topo Chico mineral water bottle") and then embedding that answer with CLIP's text encoder gives a modality-consistent matching signal — both the query answer and the catalog descriptions live in the same CLIP text space, making similarity meaningful regardless of camera angle.

**Hand-built closed catalog.**
The product universe across all 10 scenes is small (~8 distinct SKUs). Rather than zero-shot recognition, we built a `catalog.json` with reference crop images and rich descriptions. This gives CLIP a strong prior and avoids hallucination on out-of-vocabulary items.

**Geometric deduplication as the primary signal.**
Items at the same world position across different cameras are the same item — this is the core deduplication invariant. CLIP visual similarity is a secondary signal that handles cases where world distance is marginal (items close together on the tray).

### Pipeline Architecture

```
CameraView.warped_image()   ← compute once, cache
       │
MoondreamDetector.detect()  ← runs all prompts, NMS
       │
CameraView.canvas_to_world() ← linear map, exact
       │
MoondreamQueryRecognizer     ← query + CLIP text-text
       │
GeometryClipMatcher          ← cluster by world dist + CLIP sim
       │
SceneResult.to_dict()        ← write output.json
```

## Results

| Scene | Items expected | Items detected | Notes |
|-------|---------------|----------------|-------|
| 01 | 4 | 4 ✓ | Perfect — Coca-Cola, Topo Chico, SunChips, Lay's |
| 02 | 4 | 4 ✓ | Perfect |
| 03 | 6 | 4 | Cereal boxes not detected (occluded in warped views) |
| 04 | 5 | 5 | Froot Loops detected; one Coca-Cola mislabeled as Topo Chico |
| 05 | 5 | 4 | Frosted Mini Wheats missed |
| 06 | 5 | 5 | One false SunChips duplicate; Topo Chico detected |
| 07 | 6 | 3 | Moondream misses SunChips + Coca-Cola in warped views |
| 08 | 4 | 3 | Coca-Cola missed |
| 09 | 7+ | 6 | Jarritos bottles confused with Topo Chico |
| 10 | 8+ | 7 | Jarritos over-split; Lay's Baked missed in one camera |

Simple scenes with 4 items (01, 02) are perfect. Performance degrades with more items on the tray (occlusion, items near mat edges) and with glass bottles that look similar from above (Topo Chico vs Jarritos).

## Tradeoffs

**Local VLM vs. API.**
Moondream2 runs fully locally on Apple MPS (~4s per scene). A cloud VLM (GPT-4o, Claude) would be more accurate — especially for glass bottles with subtle label differences — at the cost of latency, rate limits, and API spend. The detection + recognition architecture is the same; swapping the backend is a one-class change.

**Warped detection eliminates parallax but loses resolution.**
Warping an oblique camera image to top-down stretches and blurs regions away from the mat center. Moondream sometimes misses items near the mat edges in warped views from cameras 0 and 2. The alternative — detecting on raw images and using a more sophisticated depth-aware projection — would be more accurate but significantly more complex.

**Closed catalog vs. open-vocabulary.**
The catalog approach gives strong recognition on known items but fails on anything not in the catalog. Open-vocabulary (ask the VLM to name items freely, then cluster by embedding) would generalize better but requires more careful deduplication logic.

## What I'd Do with More Time

1. **Better glass-bottle recognition.** Extract reference crops from all 4 warped camera views and average their embeddings per catalog item. Jarritos and Topo Chico are easily confused from above; a multi-view embedding average would separate them.

2. **Detection on raw + warped images.** Run Moondream on both the raw and warped image, taking the union. Raw images give better visual quality for recognition; warped images give accurate world positions. Map raw-image crops to world coordinates via the homography.

3. **Learned matcher.** Replace the distance-threshold heuristic with a learned scoring function trained on multi-camera correspondences. This would handle items close together and occlusion more robustly.

4. **Per-scene metrics.** Add a ground-truth annotation file and compute precision/recall per scene. This would let us tune thresholds (`world_dist_threshold_mm`, `min_cameras`, `clip_sim_threshold`) against real data rather than by inspection.

5. **Real-time inference.** The current per-scene warping uses a nested Python loop (slow). Precomputing warp maps once with vectorized NumPy and running Moondream in batch mode would cut latency significantly.
