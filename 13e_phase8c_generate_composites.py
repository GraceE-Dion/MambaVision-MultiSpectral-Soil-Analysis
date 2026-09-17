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


def get_surround_stats(img, x_min, y_min, x_max, y_max, ring_width=20):
    """Stats of the annulus immediately OUTSIDE the destination region --
    per peer review: we don't know what was behind the laser, so match
    against what surrounds the removal region, not its (unknown) interior."""
    img_h, img_w = img.shape[:2]
    ox_min = max(0, x_min - ring_width)
    oy_min = max(0, y_min - ring_width)
    ox_max = min(img_w, x_max + ring_width)
    oy_max = min(img_h, y_max + ring_width)

    ring_mask = np.zeros((img_h, img_w), dtype=bool)
    ring_mask[oy_min:oy_max, ox_min:ox_max] = True
    ring_mask[y_min:y_max, x_min:x_max] = False  # exclude the region itself

    pixels = img[ring_mask]
    if pixels.size == 0:
        pixels = img[oy_min:oy_max, ox_min:ox_max].reshape(-1, 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    ring_grad = grad_mag[ring_mask] if ring_mask.any() else grad_mag[oy_min:oy_max, ox_min:ox_max].flatten()

    gray_ring = gray[ring_mask] if ring_mask.any() else gray[oy_min:oy_max, ox_min:ox_max].flatten()
    hist = cv2.calcHist([gray_ring.astype(np.uint8)], [0], None, [32], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return {
        "mean": pixels.reshape(-1, 3).mean(axis=0),
        "std": pixels.reshape(-1, 3).std(axis=0),
        "grad_mean": float(ring_grad.mean()) if ring_grad.size else 0.0,
        "hist": hist,
    }


def patch_similarity_score(candidate_patch, surround_stats):
    """Lower = more similar. Combines channel mean/std difference,
    gradient-magnitude (texture) difference, and grayscale histogram
    distance -- per peer review's specified factors, kept deliberately
    simple rather than a more sophisticated exemplar-based scheme."""
    cand_mean = candidate_patch.reshape(-1, 3).mean(axis=0)
    cand_std = candidate_patch.reshape(-1, 3).std(axis=0)

    gray = cv2.cvtColor(candidate_patch, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    cand_grad_mean = float(np.sqrt(grad_x**2 + grad_y**2).mean())

    cand_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
    cand_hist = cv2.normalize(cand_hist, cand_hist).flatten()

    mean_dist = float(np.linalg.norm(cand_mean - surround_stats["mean"]))
    std_dist = float(np.linalg.norm(cand_std - surround_stats["std"]))
    grad_dist = abs(cand_grad_mean - surround_stats["grad_mean"])
    hist_dist = float(cv2.compareHist(cand_hist, surround_stats["hist"], cv2.HISTCMP_CHISQR))

    # Simple weighted sum -- not claiming optimality, just a principled,
    # reproducible combination of the four factors peer review specified.
    return mean_dist + std_dist + 0.5 * grad_dist + 0.1 * hist_dist


def generate_candidate_offsets(region_w, region_h):
    """Deterministic local neighborhood grid, not just 8 fixed points --
    multiple radii x 8 compass directions."""
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    radii = [1.0, 1.5, 2.0]
    offsets = []
    for r in radii:
        for dx_sign, dy_sign in directions:
            offsets.append((int(dx_sign * region_w * r), int(dy_sign * region_h * r)))
    return offsets


def repair_donor_roi(donor_img, donor_bbox, pad_fraction=0.6, min_pad_px=15):
    """Same-image texture replacement (per peer review: NOT 'inpainting'
    -- no PDE propagation, no learned/generative model). Selects the
    BEST-MATCHING eligible same-image patch by appearance distance to
    the destination region's immediate surround, rather than the first
    geometrically valid candidate.

    Returns: (repaired_img, diagnostic_info) where diagnostic_info
    contains everything needed for the five-view QA panel and the
    objective QA metrics peer review requires saved (not just visually
    inspected).
    """
    x_min, y_min, x_max, y_max = donor_bbox
    box_w, box_h = x_max - x_min, y_max - y_min

    pad_x = max(min_pad_px, int(box_w * pad_fraction))
    pad_y = max(min_pad_px, int(box_h * pad_fraction))

    img_h, img_w = donor_img.shape[:2]
    px_min = max(0, x_min - pad_x)
    py_min = max(0, y_min - pad_y)
    px_max = min(img_w, x_max + pad_x)
    py_max = min(img_h, y_max + pad_y)

    region_w = px_max - px_min
    region_h = py_max - py_min

    surround_stats = get_surround_stats(donor_img, px_min, py_min, px_max, py_max)

    candidates = []
    for dx, dy in generate_candidate_offsets(region_w, region_h):
        sx_min, sy_min = px_min + dx, py_min + dy
        sx_max, sy_max = px_max + dx, py_max + dy
        if sx_min < 0 or sy_min < 0 or sx_max > img_w or sy_max > img_h:
            continue
        overlaps = not (sx_max <= px_min or sx_min >= px_max or
                         sy_max <= py_min or sy_min >= py_max)
        if overlaps:
            continue
        patch = donor_img[sy_min:sy_max, sx_min:sx_max]
        if patch.shape[0] != region_h or patch.shape[1] != region_w:
            continue
        score = patch_similarity_score(patch, surround_stats)
        candidates.append({"dx": dx, "dy": dy, "sx_min": sx_min, "sy_min": sy_min,
                            "patch": patch, "score": score})

    # TIER 2: the fixed 1.0/1.5/2.0x radii can fail entirely when the
    # padded region is large or sits near an image edge (found via
    # pilot-v3 real-data QA -- two composites silently hit the old
    # fallback_inpaint path and reproduced attempt-2's streaking bug).
    # Before ever falling back to PDE inpainting, retry with smaller
    # radii, which are more likely to fit near edges.
    used_smaller_radii = False
    if not candidates:
        used_smaller_radii = True
        for r in [0.75, 0.5, 0.25]:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx_sign, dy_sign in directions:
                dx, dy = int(dx_sign * region_w * r), int(dy_sign * region_h * r)
                sx_min, sy_min = px_min + dx, py_min + dy
                sx_max, sy_max = px_max + dx, py_max + dy
                if sx_min < 0 or sy_min < 0 or sx_max > img_w or sy_max > img_h:
                    continue
                overlaps = not (sx_max <= px_min or sx_min >= px_max or
                                 sy_max <= py_min or sy_min >= py_max)
                if overlaps:
                    continue
                patch = donor_img[sy_min:sy_max, sx_min:sx_max]
                if patch.shape[0] != region_h or patch.shape[1] != region_w:
                    continue
                score = patch_similarity_score(patch, surround_stats)
                candidates.append({"dx": dx, "dy": dy, "sx_min": sx_min, "sy_min": sy_min,
                                    "patch": patch, "score": score})
            if candidates:
                break

    diagnostic = {
        "candidate_count": len(candidates),
        "used_smaller_radii_tier": used_smaller_radii,
        "pad_x": pad_x, "pad_y": pad_y,
        "region": (px_min, py_min, px_max, py_max),
        # Logged so tier-3/failure cases are diagnosable from metadata
        # alone, without needing to re-derive them from a different
        # image's bbox by mistake (as happened investigating t000_C_1).
        "donor_bbox": donor_bbox,
        "donor_img_shape": (img_w, img_h),
        "region_w": region_w, "region_h": region_h,
    }

    result = donor_img.copy()

    if not candidates:
        # TIER 3 (last resort): take the LARGEST available space adjacent
        # to the region (whichever of top/bottom/left/right has the most
        # room within image bounds) and RESIZE it (interpolation) to fit
        # the destination region -- NOT tiled/repeated, which was the
        # previous implementation and produced a visible mechanical
        # stripe-repetition artifact (found via pilot-v4 real-data QA on
        # t000_C_1). Resizing avoids repeating patterns entirely.
        space_top = py_min
        space_bottom = img_h - py_max
        space_left = px_min
        space_right = img_w - px_max
        best_space = max(space_top, space_bottom, space_left, space_right)

        if best_space == space_top and space_top > 0:
            source = donor_img[0:py_min, px_min:px_max]
        elif best_space == space_bottom and space_bottom > 0:
            source = donor_img[py_max:img_h, px_min:px_max]
        elif best_space == space_left and space_left > 0:
            source = donor_img[py_min:py_max, 0:px_min]
        elif best_space == space_right and space_right > 0:
            source = donor_img[py_min:py_max, px_max:img_w]
        else:
            source = None

        if source is not None and source.shape[0] > 0 and source.shape[1] > 0:
            reflected = cv2.resize(source, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
        else:
            # Genuinely no adjacent space at all (region covers ~whole
            # image) -- flat fill from surround color as absolute last
            # resort; should be exceptionally rare.
            reflected = np.full((region_h, region_w, 3), surround_stats["mean"], dtype=np.uint8)

        result[py_min:py_max, px_min:px_max] = reflected
        diagnostic.update({"method": "reflection_fill_last_resort",
                            "selected_score": None, "selected_source": None})
        return result, diagnostic

    best = min(candidates, key=lambda c: c["score"])
    source_patch = best["patch"]

    # Feather blend, same as before
    alpha = np.ones((region_h, region_w), dtype=np.float32)
    feather_px = max(3, min(region_w, region_h) // 8)
    if region_h > 2 * feather_px and region_w > 2 * feather_px:
        alpha = cv2.copyMakeBorder(
            alpha[feather_px:-feather_px, feather_px:-feather_px],
            feather_px, feather_px, feather_px, feather_px,
            cv2.BORDER_CONSTANT, value=0
        )
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_px / 2)
    alpha = np.clip(alpha, 0, 1)[..., None]

    target_region = result[py_min:py_max, px_min:px_max].astype(np.float32)
    blended = target_region * (1 - alpha) + source_patch.astype(np.float32) * alpha
    result[py_min:py_max, px_min:px_max] = blended.astype(np.uint8)

    diagnostic.update({
        "method": "texture_patch_replacement",
        "selected_score": best["score"],
        "selected_source": (best["sx_min"], best["sy_min"]),
    })

    # Objective QA metrics (saved always, used as flags not auto-exclusion,
    # per peer review point 4)
    repaired_region = result[py_min:py_max, px_min:px_max]
    repaired_stats_mean = repaired_region.reshape(-1, 3).mean(axis=0)
    repaired_stats_std = repaired_region.reshape(-1, 3).std(axis=0)
    gray_repaired = cv2.cvtColor(repaired_region, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray_repaired, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_repaired, cv2.CV_32F, 0, 1, ksize=3)
    repaired_grad_mean = float(np.sqrt(gx**2 + gy**2).mean())

    diagnostic["qa_metrics"] = {
        "delta_mean_luminance": float(np.linalg.norm(repaired_stats_mean - surround_stats["mean"])),
        "delta_contrast": float(np.linalg.norm(repaired_stats_std - surround_stats["std"])),
        "delta_gradient_energy": abs(repaired_grad_mean - surround_stats["grad_mean"]),
    }

    return result, diagnostic


def save_diagnostic_panel(donor_img, donor_bbox, target_img, target_bbox,
                           diagnostic, composite, out_path):
    """Five-view QA panel per peer review point 5: original donor (with
    padded-mask region outlined) -> mask overlay -> selected source patch
    location highlighted -> repaired donor -> final composite."""
    px_min, py_min, px_max, py_max = diagnostic["region"]

    view1 = donor_img.copy()
    cv2.rectangle(view1, (px_min, py_min), (px_max, py_max), (0, 0, 255), 2)

    view2 = donor_img.copy()
    overlay = view2.copy()
    cv2.rectangle(overlay, (px_min, py_min), (px_max, py_max), (0, 0, 255), -1)
    view2 = cv2.addWeighted(overlay, 0.35, view2, 0.65, 0)

    view3 = donor_img.copy()
    if diagnostic.get("selected_source"):
        sx, sy = diagnostic["selected_source"]
        region_w, region_h = px_max - px_min, py_max - py_min
        cv2.rectangle(view3, (sx, sy), (sx + region_w, sy + region_h), (255, 0, 0), 2)
    cv2.rectangle(view3, (px_min, py_min), (px_max, py_max), (0, 0, 255), 1)

    repaired_donor, _ = repair_donor_roi(donor_img, donor_bbox)  # recompute for display consistency
    view4 = repaired_donor

    view5 = composite.copy()
    tx_min, ty_min, tx_max, ty_max = target_bbox
    cv2.rectangle(view5, (tx_min, ty_min), (tx_max, ty_max), (0, 255, 0), 2)

    h = donor_img.shape[0]
    panel = np.hstack([view1, view2, view3, view4, view5])
    cv2.imwrite(out_path, panel)


def composite_image(target, donor):
    """Target ROI pixels unchanged; everything outside comes from the
    donor's background, with the donor's own ROI removed first via
    same-image texture replacement. Returns (composite, diagnostic,
    roi_integrity_ok, roi_max_diff) -- the ROI-integrity check is a
    real pixel-for-pixel assertion per the locked plan's requirement,
    not just assumed."""
    target_img = cv2.imread(target["path"])
    donor_img = cv2.imread(donor["path"])

    if donor_img.shape[:2] != target_img.shape[:2]:
        donor_img = cv2.resize(donor_img, (target_img.shape[1], target_img.shape[0]))

    donor_clean, diagnostic = repair_donor_roi(donor_img, donor["bbox"])

    composite = donor_clean.copy()
    x_min, y_min, x_max, y_max = target["bbox"]
    composite[y_min:y_max, x_min:x_max] = target_img[y_min:y_max, x_min:x_max]

    # ROI-integrity assertion (locked-plan requirement, peer review
    # point 7): target ROI pixels must be EXACTLY unchanged.
    roi_diff = np.abs(
        composite[y_min:y_max, x_min:x_max].astype(np.int16) -
        target_img[y_min:y_max, x_min:x_max].astype(np.int16)
    )
    roi_max_diff = int(roi_diff.max()) if roi_diff.size else 0
    roi_integrity_ok = (roi_max_diff == 0)

    return composite, diagnostic, roi_integrity_ok, roi_max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pilot", type=int, default=15,
                         help="Number of target images to use for pilot QA mode")
    parser.add_argument("--full", action="store_true",
                         help="Run the full sweep instead of pilot mode")
    parser.add_argument("--tag", type=str, default="",
                         help="Optional suffix for pilot output folder (e.g. 'v2') "
                              "so repeated pilot runs don't overwrite each other")
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

    out_dir = FULL_DIR if args.full else PILOT_DIR + (f"_{args.tag}" if args.tag else "")
    os.makedirs(out_dir, exist_ok=True)

    if args.full:
        targets = list(enumerate(index))
    else:
        # Heterogeneous pilot sampling (peer review point 9): stratify
        # across source datasets rather than pure random, so the pilot
        # stress-tests the algorithm across different acquisition
        # modalities/textures instead of whatever the random draw
        # happens to favor.
        by_source = {}
        for img in index:
            by_source.setdefault(img["source"], []).append(img)
        sources = sorted(by_source.keys())
        per_source = max(1, args.n_pilot // len(sources))
        sampled = []
        for src in sources:
            pool = by_source[src]
            sampled.extend(random.sample(pool, min(per_source, len(pool))))
        if len(sampled) < args.n_pilot:
            remaining = [img for img in index if img not in sampled]
            sampled.extend(random.sample(remaining, min(args.n_pilot - len(sampled), len(remaining))))
        targets = list(enumerate(sampled[:max(args.n_pilot, len(sources))]))

    records = []
    eligibility_counts = {"A": 0, "B": 0, "C": 0}
    skipped_counts = {"A": 0, "B": 0, "C": 0}
    roi_integrity_failures = []

    for target_idx, target in targets:
        for condition in ["A", "B", "C"]:
            eligible = get_eligible_donors(target, index, condition)
            if not eligible:
                skipped_counts[condition] += 1
                continue
            eligibility_counts[condition] += 1

            k = min(MAX_DONORS, len(eligible))
            donors = random.sample(eligible, k)

            for donor_idx, donor in enumerate(donors):
                composite, diagnostic, roi_ok, roi_max_diff = composite_image(target, donor)
                if not roi_ok:
                    roi_integrity_failures.append({
                        "target_id": target["stem"], "donor_id": donor["stem"],
                        "condition": condition, "roi_max_diff": roi_max_diff,
                    })

                composite_id = f"t{target_idx:03d}_{condition}_{donor_idx}"
                out_name = f"{composite_id}.png"
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, composite)

                if not args.full:
                    donor_img_full = cv2.imread(donor["path"])
                    target_img_full = cv2.imread(target["path"])
                    panel_path = os.path.join(out_dir, f"{composite_id}_QA_panel.png")
                    save_diagnostic_panel(donor_img_full, donor["bbox"], target_img_full,
                                           target["bbox"], diagnostic, composite, panel_path)

                same_modality = None
                if condition == "C":
                    target_modality = SOURCE_MODALITY.get(target["source"], "Unknown")
                    donor_modality = SOURCE_MODALITY.get(donor["source"], "Unknown")
                    same_modality = (target_modality == donor_modality)

                records.append({
                    "composite_id": composite_id,
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
                    "roi_integrity_ok": roi_ok,
                    "roi_max_diff": roi_max_diff,
                    "repair_method": diagnostic.get("method"),
                    "repair_candidate_count": diagnostic.get("candidate_count"),
                    "repair_selected_score": diagnostic.get("selected_score"),
                    "repair_selected_source": diagnostic.get("selected_source"),
                    "repair_qa_metrics": diagnostic.get("qa_metrics"),
                    "repair_donor_bbox": diagnostic.get("donor_bbox"),
                    "repair_region_w": diagnostic.get("region_w"),
                    "repair_region_h": diagnostic.get("region_h"),
                    "repair_donor_img_shape": diagnostic.get("donor_img_shape"),
                    "repair_used_smaller_radii_tier": diagnostic.get("used_smaller_radii_tier"),
                })

    print(f"\nGenerated {len(records)} composites across {len(targets)} target images")
    print(f"Eligibility (targets with >=1 eligible donor): "
          f"A={eligibility_counts['A']}, B={eligibility_counts['B']}, C={eligibility_counts['C']}")
    print(f"Skipped (no eligible donor found): "
          f"A={skipped_counts['A']}, B={skipped_counts['B']}, C={skipped_counts['C']}")

    print(f"\nROI integrity (target pixels must be EXACTLY unchanged): "
          f"{len(records) - len(roi_integrity_failures)}/{len(records)} passed")
    if roi_integrity_failures:
        print(f"  FAIL: {len(roi_integrity_failures)} composites modified target ROI pixels!")
        for fail in roi_integrity_failures[:5]:
            print(f"    {fail}")
        print("  This is a locked-plan violation -- investigate before trusting any result.")

    scores = [r["repair_selected_score"] for r in records if r["repair_selected_score"] is not None]
    if scores:
        print(f"\nRepair patch-selection scores: min={min(scores):.2f}, "
              f"max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")
        print("  (Lower = better match to destination surround. Large max values")
        print("   may indicate a composite worth visually spot-checking.)")

    metadata_path = os.path.join(out_dir, "composite_metadata.json")
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")
    with open(metadata_path, "w") as f:
        json.dump(records, f, indent=2, default=_json_safe)
    print(f"\nMetadata saved -> {metadata_path}")
    print(f"Composites saved -> {out_dir}/")

    if not args.full:
        print("\n" + "=" * 70)
        print("  NEXT STEP — MANUAL VISUAL QA (locked plan, Step 3)")
        print("=" * 70)
        print("  Each composite has a matching *_QA_panel.png showing five views:")
        print("  original donor (red box) -> mask overlay -> selected source patch")
        print("  (blue box) -> repaired donor -> final composite (green box = target).")
        print("  Open several *_QA_panel.png files across conditions A/B/C and check:")
        print("  - Target ROI region matches the original target image exactly")
        print("  - Donor's own laser spot is NOT visible anywhere in the composite")
        print("  - No obvious rectangular inpainting artifact where donor ROI was")
        print("  - Source/class/donor metadata in composite_metadata.json is correct")
        print("  Only after this passes: rerun with --full")


if __name__ == "__main__":
    main()
