"""
13e_phase8c_generate_composites.py
=====================================
Phase 8C, QA Steps 2-4: builds the donor eligibility/pairing map and
generates A/B/C background-swap composite images.

Environment-agnostic (only needs opencv, numpy, PIL) -- does NOT
load any model. Run this once; its output (composite images +
metadata) is then consumed separately by the detector eval script
(mambayolo env) and the classifier eval script (mambavision env).

PILOT MODE (default): generates only a small sample (--n_pilot
images per condition) for visual QA, per the locked sequence's Step 3.
Run this first. Only after visual inspection passes do you re-run
with --full to generate the complete eligible-image sweep (Step 4/5).

Usage:
    python 13e_phase8c_generate_composites.py --n_pilot 15
    # ... visually inspect output/pilot/ ...
    python 13e_phase8c_generate_composites.py --full

Assumptions needing confirmation before trusting the full run:
  - Val split lives at /data/Grace/Master_Detection/val/ with
    images/ and labels/ subfolders (matches Phase 8B's data.yaml).
  - Labels are YOLO-format .txt files, one bbox per image (single
    annotated laser ROI per the locked plan's matching-rule
    assumption).
  - "Source dataset" for eligibility is recovered from filename via
    the SAME get_dataset_name() logic already validated in Script 13
    (13_confound_characterization.py) -- reused here, not
    re-derived, to avoid re-introducing the stem-matching bug. Run
    13f_phase8c_preflight_check.py FIRST to confirm this matches
    every file in Master_Detection/val/images (zero "Unknown"
    results) and that all images share one uniform size (the resize
    fallback below does not rescale bbox coordinates).
  - Modality mapping for Condition C is sourced directly from Paper
    1's dataset table (SOURCE_MODALITY below), not inferred.
"""

import os
import re
import json
import random
import argparse

import cv2
import numpy as np

VAL_DIR = "/data/Grace/Master_Detection/val"
IMAGES_DIR = os.path.join(VAL_DIR, "images")
LABELS_DIR = os.path.join(VAL_DIR, "labels")

OUTPUT_BASE = "./results/phase8c"
PILOT_DIR = os.path.join(OUTPUT_BASE, "pilot")
FULL_DIR = os.path.join(OUTPUT_BASE, "full")

SEED = 0
MAX_DONORS = 3

# ── Dataset detection — reused verbatim from 13_confound_characterization.py ──
# CONFIRM this still matches current filenames before trusting eligibility
# grouping; copied here rather than imported to keep this script
# environment-independent (no dependency on either conda env's package set).
DATASET_PATTERNS = {
    "v4":         lambda f: f.startswith("Soil-Moisture-v4-") and "-IR-" not in f and "-UV-" not in f,
    "v4-IR":      lambda f: f.startswith("Soil-Moisture-v4-IR-"),
    "v4-UV":      lambda f: f.startswith("Soil-Moisture-v4-UV-"),
    "IR":         lambda f: f.startswith("Soil-Moisture-IR-"),
    "5sagf":      lambda f: f.startswith("Soil-Moisture-1_"),
    "September":  lambda f: f.startswith("Soil_Moisture_September-"),
    "Stir-Sept":  lambda f: f.startswith("Soil_Moisture_Stir_September-"),
}

# Real, documented laser modality per source, from Paper 1's dataset
# table (Journal_of_MultiSpectral_Laser_Soil_Moisture draft, Table 1) --
# NOT inferred, sourced directly. Used for Condition C's required
# same-vs-cross-modality metadata.
SOURCE_MODALITY = {
    "v4":         "UV",
    "v4-UV":      "UV",
    "September":  "UV",
    "v4-IR":      "Near-IR",
    "IR":         "Near-IR",
    "5sagf":      "Near-IR",
    "Stir-Sept":  "Mixed-UV-Red",  # unique modality, own category
}


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


def load_yolo_label(label_path, img_w, img_h):
    """Returns (class_id, x_min, y_min, x_max, y_max) in pixel space
    for the single annotated box. Returns None if missing/empty/malformed."""
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        lines = f.readlines()
    if not lines:
        return None
    parts = lines[0].strip().split()
    if len(parts) < 5:
        return None
    cls_id, cx, cy, w, h = map(float, parts[:5])
    x_min = max(0, int((cx - w / 2) * img_w))
    y_min = max(0, int((cy - h / 2) * img_h))
    x_max = min(img_w, int((cx + w / 2) * img_w))
    y_max = min(img_h, int((cy + h / 2) * img_h))
    return int(cls_id), x_min, y_min, x_max, y_max


def build_image_index():
    """Scan the val split, return list of dicts with per-image metadata."""
    index = []
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(IMAGES_DIR, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: could not read {fname}, skipping")
            continue
        h, w = img.shape[:2]

        label = load_yolo_label(label_path, w, h)
        if label is None:
            continue  # matches Script 13's established handling of unmatched/missing labels
        cls_id, x_min, y_min, x_max, y_max = label

        index.append({
            "filename": fname,
            "stem": stem,
            "path": img_path,
            "source": get_dataset_name(fname),
            "class_id": cls_id,
            "bbox": (x_min, y_min, x_max, y_max),
            "img_w": w,
            "img_h": h,
        })
    return index


def get_eligible_donors(target, index, condition):
    """Per the locked plan:
    A: same source + same class (excluding target itself)
    B: same source + different class
    C: different source + different class
    """
    candidates = []
    for img in index:
        if img["stem"] == target["stem"]:
            continue
        if condition == "A":
            if img["source"] == target["source"] and img["class_id"] == target["class_id"]:
                candidates.append(img)
        elif condition == "B":
            if img["source"] == target["source"] and img["class_id"] != target["class_id"]:
                candidates.append(img)
        elif condition == "C":
            if img["source"] != target["source"] and img["class_id"] != target["class_id"]:
                candidates.append(img)
    return candidates


def inpaint_donor_roi(donor_img, donor_bbox):
    """Removes the donor's own laser-spot ROI using OpenCV inpainting
    (Telea algorithm) rather than a flat block, per the locked plan's
    explicit requirement to avoid introducing an artificial rectangular
    feature. Returns the donor image with its ROI region plausibly
    filled in from surrounding context."""
    x_min, y_min, x_max, y_max = donor_bbox
    mask = np.zeros(donor_img.shape[:2], dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    inpainted = cv2.inpaint(donor_img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted


def composite_image(target, donor):
    """Target ROI pixels unchanged; everything outside comes from the
    donor's background, with the donor's own ROI inpainted out first."""
    target_img = cv2.imread(target["path"])
    donor_img = cv2.imread(donor["path"])

    # Resize donor to target's dimensions if they differ (shouldn't for
    # this project's uniformly-sized dataset, but guard against it)
    if donor_img.shape[:2] != target_img.shape[:2]:
        donor_img = cv2.resize(donor_img, (target_img.shape[1], target_img.shape[0]))
        # NOTE: if resizing occurs, donor bbox coordinates below would need
        # rescaling too -- flagging this as an assumption that image sizes
        # are uniform across this dataset; confirm during QA if any resize
        # actually triggers.

    donor_clean = inpaint_donor_roi(donor_img, donor["bbox"])

    composite = donor_clean.copy()
    x_min, y_min, x_max, y_max = target["bbox"]
    composite[y_min:y_max, x_min:x_max] = target_img[y_min:y_max, x_min:x_max]

    return composite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pilot", type=int, default=15,
                         help="Number of target images to use for pilot QA mode")
    parser.add_argument("--full", action="store_true",
                         help="Run the full sweep instead of pilot mode")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 70)
    print("  Phase 8C — Composite Generator")
    print("=" * 70)
    print(f"  Mode: {'FULL SWEEP' if args.full else f'PILOT (n={args.n_pilot})'}")
    print()

    print("Building image index...")
    index = build_image_index()
    print(f"  {len(index)} images with valid labels indexed (of val split)")

    unknown_sources = [img for img in index if img["source"] == "Unknown"]
    if unknown_sources:
        print(f"  WARNING: {len(unknown_sources)} images have unrecognized source "
              f"(dataset-detection pattern may need updating). Sample: "
              f"{unknown_sources[0]['filename']}")

    out_dir = PILOT_DIR if not args.full else FULL_DIR
    os.makedirs(out_dir, exist_ok=True)

    targets = index if args.full else random.sample(index, min(args.n_pilot, len(index)))

    records = []
    eligibility_counts = {"A": 0, "B": 0, "C": 0}
    skipped_counts = {"A": 0, "B": 0, "C": 0}

    for target in targets:
        for condition in ["A", "B", "C"]:
            eligible = get_eligible_donors(target, index, condition)
            if not eligible:
                skipped_counts[condition] += 1
                continue
            eligibility_counts[condition] += 1

            k = min(MAX_DONORS, len(eligible))
            donors = random.sample(eligible, k)

            for donor_idx, donor in enumerate(donors):
                composite = composite_image(target, donor)
                out_name = f"{target['stem']}__cond{condition}__donor{donor_idx}.png"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, composite)

                same_modality = None
                if condition == "C":
                    target_modality = SOURCE_MODALITY.get(target["source"], "Unknown")
                    donor_modality = SOURCE_MODALITY.get(donor["source"], "Unknown")
                    same_modality = (target_modality == donor_modality)

                records.append({
                    "target_id": target["stem"],
                    "condition": condition,
                    "donor_id": donor["stem"],
                    "donor_index": donor_idx,
                    "donor_count_used": k,
                    "donor_count_eligible": len(eligible),
                    "target_source": target["source"],
                    "target_class": target["class_id"],
                    "donor_source": donor["source"],
                    "donor_class": donor["class_id"],
                    "same_modality": same_modality,
                    "composite_path": out_path,
                    "target_bbox": target["bbox"],
                })

    print(f"\nGenerated {len(records)} composites across {len(targets)} target images")
    print(f"Eligibility (targets with >=1 eligible donor): "
          f"A={eligibility_counts['A']}, B={eligibility_counts['B']}, C={eligibility_counts['C']}")
    print(f"Skipped (no eligible donor found): "
          f"A={skipped_counts['A']}, B={skipped_counts['B']}, C={skipped_counts['C']}")

    metadata_path = os.path.join(out_dir, "composite_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nMetadata saved -> {metadata_path}")
    print(f"Composites saved -> {out_dir}/")

    if not args.full:
        print("\n" + "=" * 70)
        print("  NEXT STEP — MANUAL VISUAL QA (locked plan, Step 3)")
        print("=" * 70)
        print("  Open several composites from each condition (A/B/C) and check:")
        print("  - Target ROI region matches the original target image exactly")
        print("  - Donor's own laser spot is NOT visible anywhere in the composite")
        print("  - No obvious rectangular inpainting artifact where donor ROI was")
        print("  - Source/class/donor metadata in composite_metadata.json is correct")
        print("  Only after this passes: rerun with --full")


if __name__ == "__main__":
    main()
