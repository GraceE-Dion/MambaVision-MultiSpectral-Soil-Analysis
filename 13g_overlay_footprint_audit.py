"""
13g_overlay_footprint_audit.py
=================================
Phase 8C Finding 1 prerequisite: determine the real spatial footprint
of the burned-in "Moisture: N / Time:" text overlay across all 7
acquisition sources, BEFORE designing any removal method.

Per peer review: do not assume a single fixed region -- measure it.
Also flags whether the overlay may contain more than moisture class
(this script reports detected regions; a human must still read a
sample to confirm content, since OCR on stylized/small burned-in text
can be unreliable).

Method: the overlay text is consistently yellow (per direct visual QA
transcripts this session -- "Yellow overlay text, top left of each
panel"). Detect yellow-ish pixels via HSV color thresholding rather
than OCR, more robust to small/stretched burned-in text and doesn't
require the text to be machine-readable.

Usage:
    python 13g_overlay_footprint_audit.py --sample_per_source 15

Output: printed summary + overlay_footprint_audit.json with per-image
detected bounding boxes, so the removal-region design is based on
real measured data, not assumption.
"""

import os
import json
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np

VAL_IMAGES_DIR = "/data/Grace/Master_Detection/val/images"
OUT_DIR = "./results/phase8c"

DATASET_PATTERNS = {
    "v4":         lambda f: f.startswith("Soil-Moisture-v4-") and "-IR-" not in f and "-UV-" not in f,
    "v4-IR":      lambda f: f.startswith("Soil-Moisture-v4-IR-"),
    "v4-UV":      lambda f: f.startswith("Soil-Moisture-v4-UV-"),
    "IR":         lambda f: f.startswith("Soil-Moisture-IR-"),
    "5sagf":      lambda f: f.startswith("Soil-Moisture-1_"),
    "September":  lambda f: f.startswith("Soil_Moisture_September-"),
    "Stir-Sept":  lambda f: f.startswith("Soil_Moisture_Stir_September-"),
}


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


def detect_yellow_text_bbox(img):
    """Returns (bbox, quadrant, pixel_count). bbox is
    (x_min, y_min, x_max, y_max) of detected TEXT-LIKE yellow blobs,
    or None if nothing found.

    REVISED after first audit run returned near-full-image bounding
    boxes (0,0)-(639,639) for nearly every source -- the original
    broad HSV threshold was catching warm-toned soil/gravel and
    possibly laser glow, not just the small burned-in text. Fixed by:
    (1) tightening saturation/value thresholds to match pure, bright,
    saturated text rather than generic warm soil tones, and (2) using
    connected-component analysis with a plausible text-character-
    cluster size filter, discarding large diffuse blobs that cannot
    be text."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Tightened: pure bright yellow text is typically near-maximum
    # saturation and value (close to RGB (255,255,0) territory), not
    # merely "warm-toned" like soil/gravel under directional lighting.
    lower_yellow = np.array([22, 150, 180])
    upper_yellow = np.array([32, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Connected-component filtering: keep only components whose size
    # is plausible for a text character/word cluster. Discard both
    # tiny noise specks and large diffuse regions (soil, glare, laser
    # glow) that cannot be burned-in text at this image resolution.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    MIN_COMPONENT_AREA = 8
    MAX_COMPONENT_AREA = 400
    text_mask = np.zeros_like(mask)
    for i in range(1, n_labels):  # skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
            text_mask[labels == i] = 255

    ys, xs = np.where(text_mask > 0)
    if len(xs) == 0:
        return None, "none_found", 0

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    quadrant = "top_left" if (x_max < w * 0.5 and y_max < h * 0.5) else "spread_or_elsewhere"

    return (x_min, y_min, x_max, y_max), quadrant, int(len(xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_per_source", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Overlay Footprint Audit")
    print("=" * 70)

    by_source = defaultdict(list)
    for fname in sorted(os.listdir(VAL_IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        src = get_dataset_name(fname)
        by_source[src].append(fname)

    random.seed(0)
    results = []
    source_bboxes = defaultdict(list)

    for src, files in sorted(by_source.items()):
        sample = random.sample(files, min(args.sample_per_source, len(files)))
        for fname in sample:
            img = cv2.imread(os.path.join(VAL_IMAGES_DIR, fname))
            if img is None:
                continue
            bbox, quadrant, pixel_count = detect_yellow_text_bbox(img)
            results.append({
                "filename": fname, "source": src, "bbox": bbox,
                "quadrant": quadrant, "yellow_pixel_count": pixel_count,
            })
            if bbox is not None:
                source_bboxes[src].append(bbox)

    print(f"\nSampled {len(results)} images across {len(by_source)} sources\n")
    print(f"{'Source':<12} {'N sampled':<10} {'N detected':<11} {'Union bbox':<36} {'Consistent?'}")

    all_union = None
    for src in sorted(by_source.keys()):
        bboxes = source_bboxes[src]
        n_sampled = min(args.sample_per_source, len(by_source[src]))
        n_detected = len(bboxes)
        if bboxes:
            x_min = min(b[0] for b in bboxes)
            y_min = min(b[1] for b in bboxes)
            x_max = max(b[2] for b in bboxes)
            y_max = max(b[3] for b in bboxes)
            widths = [b[2] - b[0] for b in bboxes]
            heights = [b[3] - b[1] for b in bboxes]
            consistent = "yes" if (max(widths) - min(widths) < 40 and max(heights) - min(heights) < 40) else "VARIES"
            union = (x_min, y_min, x_max, y_max)
            all_union = union if all_union is None else (
                min(all_union[0], x_min), min(all_union[1], y_min),
                max(all_union[2], x_max), max(all_union[3], y_max))
            print(f"{src:<12} {n_sampled:<10} {n_detected:<11} {str(union):<36} {consistent}")
        else:
            print(f"{src:<12} {n_sampled:<10} {n_detected:<11} {'NO DETECTION':<36} N/A")

    print(f"\nOVERALL UNION across all sources (candidate fixed removal region): {all_union}")

    non_topleft = [r for r in results if r["bbox"] is not None and r["quadrant"] != "top_left"]
    if non_topleft:
        print(f"\nWARNING: {len(non_topleft)} images detected overlay OUTSIDE top-left quadrant")
        print("-- a single fixed top-left region would NOT cover these. Samples:")
        for r in non_topleft[:5]:
            print(f"  {r['filename']} ({r['source']}): bbox={r['bbox']}")

    no_detection = [r for r in results if r["bbox"] is None]
    if no_detection:
        print(f"\n{len(no_detection)} images had NO yellow-text detection at all.")
        print("May have a different overlay color, no overlay, or threshold needs")
        print("adjustment -- worth manually checking a few samples:")
        for r in no_detection[:5]:
            print(f"  {r['filename']} ({r['source']})")

    out_path = os.path.join(OUT_DIR, "overlay_footprint_audit.json")
    with open(out_path, "w") as f:
        json.dump({
            "per_image_results": results,
            "per_source_union_bbox": {k: (min(b[0] for b in v), min(b[1] for b in v),
                                            max(b[2] for b in v), max(b[3] for b in v)) if v else None
                                        for k, v in source_bboxes.items()},
            "overall_union_bbox": all_union,
        }, f, indent=2)
    print(f"\nFull results saved -> {out_path}")

    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("  1. If 'Consistent?' is 'yes' for all sources and no cross-quadrant")
    print("     warnings: a single conservative fixed region (union bbox above,")
    print("     padded slightly) is likely sufficient.")
    print("  2. If sources vary or warnings appeared: manually inspect flagged")
    print("     images -- may need per-source regions instead of one global one.")
    print("  3. Manually open 3-5 images per source and confirm overlay CONTENT --")
    print("     only moisture class, or also source ID/date/measurement values?")
    print("  4. Only once footprint is confirmed: design the overlay-removal")
    print("     method (same-image texture repair, applied uniformly to clean AND")
    print("     all A/B/C images -- not composites only).")


if __name__ == "__main__":
    main()
