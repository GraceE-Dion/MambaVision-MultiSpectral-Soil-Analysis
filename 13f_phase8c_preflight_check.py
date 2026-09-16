"""
13f_phase8c_preflight_check.py
=================================
Quick checks before trusting 13e's composite generator on the full
sweep:
  1. Does every image in Master_Detection/val/images/ get correctly
     classified by get_dataset_name() (zero "Unknown" results)?
  2. Are all images the same dimensions (so the composite/inpaint
     logic never silently hits the resize fallback)?

Run BEFORE the pilot composite generation, not after.
"""

import os
import cv2
from collections import Counter

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


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


def main():
    print("=" * 70)
    print("  Phase 8C Pre-flight Check")
    print("=" * 70)

    files = sorted(f for f in os.listdir(VAL_IMAGES_DIR)
                    if f.lower().endswith((".png", ".jpg", ".jpeg")))
    print(f"\nTotal images: {len(files)}")

    # ── Check 1: dataset name matching ──────────────────────────────────
    print("\n--- Check 1: Dataset name matching ---")
    source_counts = Counter()
    unknown_files = []
    for f in files:
        src = get_dataset_name(f)
        source_counts[src] += 1
        if src == "Unknown":
            unknown_files.append(f)

    for src, count in source_counts.most_common():
        print(f"  {src:15s}: {count}")

    if unknown_files:
        print(f"\n  FAIL: {len(unknown_files)} files unmatched. Samples:")
        for f in unknown_files[:5]:
            print(f"    {f}")
        print("  -> Fix DATASET_PATTERNS before running the composite generator.")
    else:
        print("\n  PASS: every image matched to a known source.")

    # ── Check 2: image size uniformity ──────────────────────────────────
    print("\n--- Check 2: Image dimensions ---")
    size_counts = Counter()
    for f in files:
        img = cv2.imread(os.path.join(VAL_IMAGES_DIR, f))
        if img is None:
            print(f"  WARNING: could not read {f}")
            continue
        size_counts[(img.shape[1], img.shape[0])] += 1  # (w, h)

    if len(size_counts) == 1:
        (w, h), count = list(size_counts.items())[0]
        print(f"  PASS: all {count} images are {w}x{h}. No resize fallback risk.")
    else:
        print(f"  FAIL: {len(size_counts)} distinct sizes found:")
        for (w, h), count in size_counts.most_common():
            print(f"    {w}x{h}: {count} images")
        print("  -> Composite generator's resize fallback WILL trigger for some")
        print("     pairs. Bbox coordinates must be rescaled accordingly if so —")
        print("     current generator does NOT do this; needs a fix before --full.")


if __name__ == "__main__":
    main()
