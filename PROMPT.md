# Vision Pipeline Take-Home

## Background

You have a self-checkout kiosk with 4 cameras, each viewing the checkout tray from a different angle. Each camera has been calibrated using a circle-pattern calibration process. You are given 10 scenes captured from this hardware.

## Your Task

Build a prototype system that takes these multi-camera images as input and outputs a list of products on the tray.

Deliver:
1. **Working code** that runs on the provided data and produces `output.json` for each scene
2. **A short writeup** (~1 page) explaining your approach, key decisions, tradeoffs, and what you'd do differently with more time

## Output Format

For each scene, produce an `output.json`:

```json
{
  "scene": "scene_01",
  "items": [
    {
      "name": "Coca-Cola 12oz Can",
      "description": "Red aluminum can with white Coca-Cola script logo",
      "packaging_type": "can",
      "size": "12oz",
      "upc": "049000006346",
      "cameras": [0, 1, 2, 3],
      "camera_detections": {
        "0": {"x": 450, "y": 320, "w": 200, "h": 280},
        "1": {"x": 610, "y": 290, "w": 180, "h": 260},
        "2": {"x": 380, "y": 350, "w": 210, "h": 290},
        "3": {"x": 520, "y": 310, "w": 190, "h": 270}
      },
      "confidence": 0.92,
      "world_position": {
        "x_mm": 45.2,
        "y_mm": -12.8
      }
    }
  ]
}
```

| Field | Required | Notes |
|-------|----------|-------|
| `name` | Yes | Product name |
| `description` | Yes | Visual description of the product |
| `packaging_type` | Yes | e.g. can, bottle, bag, box, wrapper, cup |
| `cameras` | Expected | Which camera pair_ids this item was visible in |
| `camera_detections` | Expected | Bounding box per camera where the item was found |
| `size` | Optional | Estimated size or volume |
| `upc` | Optional | UPC barcode if determinable |
| `confidence` | Optional | Your system's confidence score, 0-1 |
| `world_position` | Optional | Position in world coordinates (mm), if calibration was used |

## Data Package Contents

```
data/
  scene_01/
    images/
      pair_0.jpg    (2048x1536)
      pair_1.jpg
      pair_2.jpg
      pair_3.jpg
    calibration/
      pair_0.json
      pair_1.json
      pair_2.json
      pair_3.json
  scene_02/
    ...
  scene_10/
    ...
calibration.py      # Helper for parsing calibration data (see docstrings)
```

## Tools & Approach

We expect you to use an AI coding assistant (Cursor, Claude Code, Copilot, etc.) throughout this exercise. This is not a test of whether you can write OpenCV code from memory -- it's a test of how effectively you can combine your ML knowledge with AI-assisted development to build a working CV prototype quickly.

We built an internal version of this pipeline using Claude Code and hit the timelines below. We're interested in seeing how you approach the same problem.

## Time Guidance

Budget as much time as you can. Here's roughly what we'd expect at each level:

- **~2 hours:** Get segmentation, basic cross-camera item matching, and item recognition working end-to-end. Output the JSON for all scenes.
- **~4 hours:** Focus on quality of item de-duplication across cameras. Reduce false duplicates and missed merges.
- **~6 hours:** Try multiple approaches on segmentation, matching, or another part of the pipeline. Have experimental results comparing them (e.g., accuracy, cost, latency).

## Notes

- You may use any tools, libraries, or APIs (including LLM vision APIs, local models, etc.)
- There is no single "right" answer -- we care about your approach, engineering decisions, and how you handle ambiguity
- A `calibration.py` helper is provided for parsing the calibration data (see its docstrings for usage)
