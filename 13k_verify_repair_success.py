"""
13k_verify_repair_success.py
================================
Final quantitative confirmation, per the other chat's own suggestion:
checks EVERY repaired image (not just spot-checked pairs) in the
sanitized output directory for any residual overlay pixels, using the
same validated tight-threshold detection logic.

This is the closing check before treating overlay removal as done and
moving Phase 8C's composite generator over to the sanitized directory.

Usage:
    python 13k_verify_repair_success.py
"""

import os

import cv2
import numpy as np

SANITIZED_DIR = "/data/Grace/Master_Detection_overlay_sanitized/val/images"

DATASET_PATTERNS = {
    "v4":         lambda f: f.startswith("Soil-Moisture-v4-") and "-IR-" not in f and "-UV-" not in f,
    "v4-IR":      lambda f: f.startswith("Soil-Moisture-v4-IR-"),
    "v4-UV":      lambda f: f.startswith("Soil-Moisture-v4-UV-"),
    "IR":         lambda f: f.startswith("Soil-Moisture-IR-"),
    "5sagf":      lambda f: f.startswith("Soil-Moisture-1_"),
    "September":  lambda f: f.startswith("Soil_Moisture_September-"),
    "Stir-Sept":  lambda f: f.startswith("Soil_Moisture_Stir_September-"),
}

OVERLAY_SOURCES = ["September", "Stir-Sept", "v4", "v4-IR", "v4-UV"]


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


def detect_yellow_text_pixels(img):
    """Same validated tight threshold as the fixed footprint audit /
    region-verification scripts."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
    print("=" * 70)
    print("  Full-Batch Repair Success Verification")
    print("=" * 70)
    print(f"  Checking EVERY repaired image in {SANITIZED_DIR}")
    print(f"  for any residual overlay-like pixels (not sampled -- full batch)\n")

    files_by_source = {src: [] for src in OVERLAY_SOURCES}
    for fname in sorted(os.listdir(SANITIZED_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        src = get_dataset_name(fname)
        if src in OVERLAY_SOURCES:
            files_by_source[src].append(fname)

    total_checked = 0
    total_residual = 0
    residual_files = []

    for src in OVERLAY_SOURCES:
        files = files_by_source[src]
        src_residual = 0
        for fname in files:
            img = cv2.imread(os.path.join(SANITIZED_DIR, fname))
            if img is None:
                continue
            pixels = detect_yellow_text_pixels(img)
            total_checked += 1
            if pixels:
                src_residual += 1
                total_residual += 1
                residual_files.append((src, fname, len(pixels)))

        status = "CLEAN" if src_residual == 0 else f"{src_residual}/{len(files)} STILL SHOW RESIDUAL PIXELS"
        print(f"  {src:12s}: {len(files)} images checked -- {status}")

    print(f"\n{'=' * 70}")
    if total_residual == 0:
        print(f"  RESULT: CLEAN -- all {total_checked} repaired images across all 5 sources")
        print(f"  show zero residual overlay-like pixels. Overlay removal confirmed")
        print(f"  successful across the FULL batch, not just spot-checked samples.")
        print(f"\n  Safe to proceed: point Phase 8C's composite generator at")
        print(f"  {SANITIZED_DIR} instead of the original Master_Detection.")
    else:
        print(f"  RESULT: {total_residual}/{total_checked} images still show residual pixels.")
        print(f"  Do NOT treat overlay removal as complete yet. Affected files:")
        for src, fname, count in sorted(residual_files, key=lambda r: -r[2])[:15]:
            print(f"    [{src}] {fname}: {count} residual pixels")


if __name__ == "__main__":
    main()
