"""
13o_phase8c_bbox_geometry_by_source.py
=========================================
Diagnostic: is the September/Stir-Sept concentration among weak-
candidate-support donors (7/10 in pilot_finding2v2) a structural
property of those sources' annotated ROI geometry, or incidental?

Reads the same val-split index the composite generator builds (image
+ label pairs) and reports, PER SOURCE: mean/median bbox area as a
fraction of image area, and mean/median distance from bbox edge to
nearest image edge (the same two quantities select_pilot_donors()
uses to identify geometrically hard donors).

Pure read-only diagnostic -- no model, no repair, no composite
generation. Does not touch or change the frozen generator.

Usage:
    python 13o_phase8c_bbox_geometry_by_source.py
"""

import os
import cv2

VAL_DIR = "/data/Grace/Master_Detection_overlay_sanitized/val"
IMAGES_DIR = os.path.join(VAL_DIR, "images")
LABELS_DIR = os.path.join(VAL_DIR, "labels")

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


def load_yolo_label(label_path, img_w, img_h):
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
    return x_min, y_min, x_max, y_max


def main():
    by_source = {}
    n_total = 0
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img_path = os.path.join(IMAGES_DIR, fname)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        bbox = load_yolo_label(label_path, w, h)
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox

        source = get_dataset_name(fname)
        area_frac = ((x_max - x_min) * (y_max - y_min)) / (w * h)
        boundary_dist = min(x_min, y_min, w - x_max, h - y_max)
        boundary_dist_frac = boundary_dist / min(w, h)

        by_source.setdefault(source, []).append({
            "area_frac": area_frac,
            "boundary_dist_frac": boundary_dist_frac,
        })
        n_total += 1

    def median(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    print("=" * 90)
    print(f"  Per-source ROI geometry -- {n_total} images total")
    print("=" * 90)
    print(f"{'Source':<12} {'n':>4}  {'area_frac (mean/median)':>26}  {'boundary_dist_frac (mean/median)':>34}")
    print("-" * 90)

    # Sort sources by mean area_frac descending -- puts the most
    # geometrically-constrained sources at the top for easy reading.
    rows = []
    for src, recs in by_source.items():
        areas = [r["area_frac"] for r in recs]
        bdists = [r["boundary_dist_frac"] for r in recs]
        rows.append((src, len(recs), sum(areas)/len(areas), median(areas),
                      sum(bdists)/len(bdists), median(bdists)))
    rows.sort(key=lambda r: r[2], reverse=True)

    for src, n, area_mean, area_med, bd_mean, bd_med in rows:
        print(f"{src:<12} {n:>4}  {area_mean:>10.4f} / {area_med:<10.4f}  "
              f"{bd_mean:>15.4f} / {bd_med:<15.4f}")

    print("\n  area_frac = bbox area / image area (larger -> bigger padded search")
    print("    region -> fewer candidate offsets fit in-bounds -> more likely weak support)")
    print("  boundary_dist_frac = distance from bbox to nearest image edge, as a")
    print("    fraction of the shorter image dimension (smaller -> bbox sits closer")
    print("    to the edge -> also fewer candidate offsets fit)")
    print("\n  If September/Stir-Sept show a substantially higher area_frac and/or")
    print("  lower boundary_dist_frac than the other five sources, that's structural")
    print("  evidence explaining the weak-candidate-support concentration -- not")
    print("  something to fix in the repair algorithm, but something to document as")
    print("  a known, source-linked property when interpreting Phase 8C results.")


if __name__ == "__main__":
    main()
