"""
13j_verify_unified_overlay_region.py
=======================================
Before hardcoding a THIRD guess at the overlay region (the first two
attempts, both per-source and audit-derived, failed on real images),
this verifies a PROPOSED region against real detected pixels across
all 5 overlay-bearing sources, using the SAME tightened yellow-
detection logic already validated in the footprint audit (tight
HSV threshold + connected-component size filtering, confirmed to
correctly reject large diffuse regions and correctly find small
text-like blobs).

Does NOT modify any images. Pure verification: for the proposed
region, checks whether any detected overlay pixel falls OUTSIDE it,
across a larger sample than the original audit used, for every
source that has a real overlay.

Usage:
    python 13j_verify_unified_overlay_region.py --sample_per_source 40
"""

import os
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np

VAL_IMAGES_DIR = "/data/Grace/Master_Detection/val/images"

DATASET_PATTERNS = {
    "v4":         lambda f: f.startswith("Soil-Moisture-v4-") and "-IR-" not in f and "-UV-" not in f,
    "v4-IR":      lambda f: f.startswith("Soil-Moisture-v4-IR-"),
    "v4-UV":      lambda f: f.startswith("Soil-Moisture-v4-UV-"),
    "IR":         lambda f: f.startswith("Soil-Moisture-IR-"),
    "5sagf":      lambda f: f.startswith("Soil-Moisture-1_"),
    "September":  lambda f: f.startswith("Soil_Moisture_September-"),
    "Stir-Sept":  lambda f: f.startswith("Soil_Moisture_Stir_September-"),
}

# Sources confirmed (via direct human visual inspection this session)
# to have a real overlay -- 5sagf and IR excluded, confirmed none.
OVERLAY_SOURCES = ["September", "Stir-Sept", "v4", "v4-IR", "v4-UV"]

# Proposed unified region, from direct visual pixel-estimate on real
# images (Stir-Sept: 5,3,100,40 and v4-IR: 3,2,90,38), combined with
# extra safety margin given two prior per-source guesses both failed
# by being too small/mispositioned.
PROPOSED_REGION = (0, 0, 110, 46)  # x_min, y_min, x_max, y_max


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


def detect_yellow_text_pixels(img):
    """Same validated tight-threshold + connected-component approach
    as the footprint audit -- returns list of (x, y) pixel coords of
    detected text-like blobs, or empty list if none found."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # CORRECTED -- the original version of this script mistakenly used
    # the OLD, already-disproven loose threshold ([18,80,120] to
    # [35,255,255]), which catches warm-toned soil/gravel/glow across
    # large diffuse areas, not just small text. This reproduced the
    # exact false-failure signature already seen and fixed once before
    # (huge "violation" spans like (0,0)-(561,639), thousands of
    # pixels -- clearly not text). Using the validated tight threshold
    # from the corrected footprint audit instead.
    lower_yellow = np.array([22, 150, 180])
    upper_yellow = np.array([32, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    MIN_COMPONENT_AREA = 6
    MAX_COMPONENT_AREA = 600
    text_mask = np.zeros_like(mask)
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
            text_mask[labels == i] = 255

    ys, xs = np.where(text_mask > 0)
    return list(zip(xs.tolist(), ys.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_per_source", type=int, default=40)
    args = parser.parse_args()

    print("=" * 70)
    print("  Unified Overlay Region Verification")
    print("=" * 70)
    print(f"  Proposed region: {PROPOSED_REGION}")
    print(f"  Sampling {args.sample_per_source} images per overlay-bearing source\n")

    by_source = defaultdict(list)
    for fname in sorted(os.listdir(VAL_IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        src = get_dataset_name(fname)
        if src in OVERLAY_SOURCES:
            by_source[src].append(fname)

    random.seed(1)
    px_min, py_min, px_max, py_max = PROPOSED_REGION

    overall_outside_count = 0
    overall_checked = 0
    worst_outliers = []

    for src in OVERLAY_SOURCES:
        files = by_source[src]
        sample = random.sample(files, min(args.sample_per_source, len(files)))
        images_with_pixels_outside = 0

        for fname in sample:
            img = cv2.imread(os.path.join(VAL_IMAGES_DIR, fname))
            if img is None:
                continue
            pixels = detect_yellow_text_pixels(img)
            outside_pixels = [(x, y) for x, y in pixels
                               if x < px_min or y < py_min or x >= px_max or y >= py_max]
            if outside_pixels:
                images_with_pixels_outside += 1
                xs = [p[0] for p in outside_pixels]
                ys = [p[1] for p in outside_pixels]
                extent = (min(xs), min(ys), max(xs), max(ys))
                worst_outliers.append((src, fname, extent, len(outside_pixels)))

        overall_checked += len(sample)
        overall_outside_count += images_with_pixels_outside

        status = "PASS" if images_with_pixels_outside == 0 else f"FAIL ({images_with_pixels_outside}/{len(sample)} images have pixels outside proposed region)"
        print(f"  {src:12s}: {len(sample)} checked -- {status}")

    print(f"\n{'=' * 70}")
    if overall_outside_count == 0:
        print(f"  RESULT: PASS -- proposed region {PROPOSED_REGION} contains all detected")
        print(f"  overlay pixels across {overall_checked} sampled images, all 5 sources.")
        print(f"  Safe to use as the unified repair region.")
    else:
        print(f"  RESULT: FAIL -- {overall_outside_count} images had detected overlay pixels")
        print(f"  OUTSIDE the proposed region. Do NOT use this region yet.")
        print(f"\n  Worst violations (source, file, outside-pixel extent, count):")
        worst_outliers.sort(key=lambda w: -w[3])
        for src, fname, extent, count in worst_outliers[:10]:
            print(f"    [{src}] {fname}: outside pixels span {extent}, {count} pixels")
        all_outside = [(w[2]) for w in worst_outliers]
        if all_outside:
            need_x_min = min(px_min, min(e[0] for e in all_outside))
            need_y_min = min(py_min, min(e[1] for e in all_outside))
            need_x_max = max(px_max, max(e[2] for e in all_outside) + 1)
            need_y_max = max(py_max, max(e[3] for e in all_outside) + 1)
            print(f"\n  Region that WOULD cover everything detected in this sample:")
            print(f"    ({need_x_min}, {need_y_min}, {need_x_max}, {need_y_max})")
            print(f"  (still recommend adding a few pixels of margin beyond this)")


if __name__ == "__main__":
    main()
