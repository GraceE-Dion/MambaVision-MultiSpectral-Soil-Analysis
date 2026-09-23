"""
13i_overlay_removal.py
=========================
Phase 8C Finding 1 remediation: removes the burned-in "Moisture: N /
Time:" text overlay from every image, using the SAME same-image
scored-texture-repair principle already validated for donor-ROI
removal (Script 13e) -- generalized here to a single-image case
(repair a region using texture from elsewhere in the SAME image, no
cross-image donor search needed, since there's no "donor" concept for
an image repairing its own overlay).

CRITICAL, per peer review: this runs as a PREPROCESSING step applied
UNIFORMLY to every image -- clean baseline images used for the QA
reproduction check, AND every target/donor image later used in
composite generation. NOT applied only to composites. Otherwise
"clean" and "perturbed" conditions would differ by an extra
intervention (overlay present vs. absent) that isn't part of the
actual Phase 8C experiment.

Per-source regions (NOT one global region -- sources have distinct,
individually-consistent footprints per the overlay footprint audit):
  September, Stir-Sept, v4, v4-IR, v4-UV: real overlay, repaired.
  5sagf, IR: confirmed (both by the audit AND by direct human
  inspection of raw images) to have NO overlay -- skipped entirely,
  no unnecessary repair on a region that never needed it.

Output: sanitized copies saved to a parallel directory structure,
which becomes the canonical input for all subsequent Phase 8C work
(both the clean-baseline reproduction check and composite generation)
-- NOT modifying the original Master_Detection images in place.

Usage:
    python 13i_overlay_removal.py [--dry_run]
"""

import os
import shutil
import argparse

import cv2
import numpy as np

VAL_IMAGES_DIR = "/data/Grace/Master_Detection/val/images"
OUT_DIR = "/data/Grace/Master_Detection_overlay_sanitized/val/images"
REPORT_DIR = "./results/phase8c"

DATASET_PATTERNS = {
    "v4":         lambda f: f.startswith("Soil-Moisture-v4-") and "-IR-" not in f and "-UV-" not in f,
    "v4-IR":      lambda f: f.startswith("Soil-Moisture-v4-IR-"),
    "v4-UV":      lambda f: f.startswith("Soil-Moisture-v4-UV-"),
    "IR":         lambda f: f.startswith("Soil-Moisture-IR-"),
    "5sagf":      lambda f: f.startswith("Soil-Moisture-1_"),
    "September":  lambda f: f.startswith("Soil_Moisture_September-"),
    "Stir-Sept":  lambda f: f.startswith("Soil_Moisture_Stir_September-"),
}

# From the overlay footprint audit (13g), per-source union bboxes,
# padded +6px on each side for safety margin against slight per-image
# variation. 5sagf and IR deliberately absent -- confirmed no overlay,
# no region to repair.
# From the overlay footprint audit (13g), per-source union bboxes,
# padded +6px on each side for safety margin against slight per-image
# variation. 5sagf and IR deliberately absent -- confirmed no overlay,
# no region to repair.
#
# CORRECTION (Stir-Sept): the original audit-derived box (10,10,53,30)
# was confirmed WRONG via direct visual inspection of a real repaired
# image -- the overlay was still fully visible/legible after "repair,"
# meaning the box missed it entirely. Root cause: Stir-Sept had the
# LOWEST detection hit rate in the original audit (4/12 sampled
# images), so the audit-derived box was built from too few successful
# detections to capture the true footprint. Direct visual evidence
# showed the real overlay clipped by the TOP EDGE of the frame
# (starts near y=0, not y=10) and spans two full lines of text
# (taller than the original 20px). Widened and repositioned below
# based on this direct evidence, not re-derived from the same weak
# audit data.
#
# v4-IR had an equally low hit rate (2/15) in the same audit -- NOT
# yet confirmed broken or fine via direct visual check. Flagged here;
# verify with a real repaired v4-IR image before trusting its region.
# CORRECTED (third attempt) -- both prior versions (per-source,
# audit-derived) FAILED on real images: attempt 1 used regions too
# small/mispositioned per source; attempt 2 (Stir-Sept specifically)
# was still wrong despite a targeted re-estimate. Root cause both
# times: regions were built from either automated detection with low
# hit-rate confidence, or a single un-verified visual guess.
#
# This version uses ONE unified region across all 5 overlay-bearing
# sources, derived from direct pixel-level visual estimates on real
# images (not automated audit data), with added safety margin, THEN
# VERIFIED (not just assumed) against real detected overlay pixels
# across 133 sampled images spanning all 5 sources (Script 13j) --
# clean pass, zero images had any detected overlay pixel outside this
# region. This is the first version of this region with actual
# empirical verification behind it, not just a plausible-looking
# guess.
UNIFIED_OVERLAY_REGION = (0, 0, 110, 46)
OVERLAY_REGIONS = {
    "September":  UNIFIED_OVERLAY_REGION,
    "Stir-Sept":  UNIFIED_OVERLAY_REGION,
    "v4":         UNIFIED_OVERLAY_REGION,
    "v4-IR":      UNIFIED_OVERLAY_REGION,
    "v4-UV":      UNIFIED_OVERLAY_REGION,
}


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


# ── Generalized same-image scored-texture repair ───────────────────
# Extracted and generalized from 13e_phase8c_generate_composites.py's
# repair_donor_roi(): same scoring logic (mean/std/gradient/histogram
# distance to the region's local surround), same candidate search
# (multiple radii, 8 compass directions, smaller-radius retry tier),
# same feathered blend -- but operating on ONE image repairing its
# OWN region, not a donor image being repaired using content that
# will later be composited with a different target.

def get_surround_stats(img, x_min, y_min, x_max, y_max, ring_width=15):
    img_h, img_w = img.shape[:2]
    ox_min = max(0, x_min - ring_width)
    oy_min = max(0, y_min - ring_width)
    ox_max = min(img_w, x_max + ring_width)
    oy_max = min(img_h, y_max + ring_width)

    ring_mask = np.zeros((img_h, img_w), dtype=bool)
    ring_mask[oy_min:oy_max, ox_min:ox_max] = True
    ring_mask[y_min:y_max, x_min:x_max] = False

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

    return mean_dist + std_dist + 0.5 * grad_dist + 0.1 * hist_dist


def generate_candidate_offsets(region_w, region_h):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    radii = [1.0, 1.5, 2.0, 0.75, 0.5, 0.25]
    offsets = []
    for r in radii:
        for dx_sign, dy_sign in directions:
            offsets.append((int(dx_sign * region_w * r), int(dy_sign * region_h * r)))
    return offsets


def repair_region_same_image(img, region_bbox):
    """General-purpose: repair region_bbox in img using scored texture
    from elsewhere in the SAME image. Returns (repaired_img, diagnostic)."""
    x_min, y_min, x_max, y_max = region_bbox
    img_h, img_w = img.shape[:2]
    region_w, region_h = x_max - x_min, y_max - y_min

    surround_stats = get_surround_stats(img, x_min, y_min, x_max, y_max)

    candidates = []
    for dx, dy in generate_candidate_offsets(region_w, region_h):
        sx_min, sy_min = x_min + dx, y_min + dy
        sx_max, sy_max = x_max + dx, y_max + dy
        if sx_min < 0 or sy_min < 0 or sx_max > img_w or sy_max > img_h:
            continue
        overlaps = not (sx_max <= x_min or sx_min >= x_max or
                         sy_max <= y_min or sy_min >= y_max)
        if overlaps:
            continue
        patch = img[sy_min:sy_max, sx_min:sx_max]
        if patch.shape[0] != region_h or patch.shape[1] != region_w:
            continue
        score = patch_similarity_score(patch, surround_stats)
        candidates.append({"patch": patch, "score": score, "source": (sx_min, sy_min)})

    result = img.copy()
    diag = {"candidate_count": len(candidates)}

    if not candidates:
        space_top, space_bottom = y_min, img_h - y_max
        space_left, space_right = x_min, img_w - x_max
        tier3_candidates = []
        if space_top > 0:
            tier3_candidates.append(("top", img[0:y_min, x_min:x_max]))
        if space_bottom > 0:
            tier3_candidates.append(("bottom", img[y_max:img_h, x_min:x_max]))
        if space_left > 0:
            tier3_candidates.append(("left", img[y_min:y_max, 0:x_min]))
        if space_right > 0:
            tier3_candidates.append(("right", img[y_min:y_max, x_max:img_w]))
        scored3 = []
        for direction, src in tier3_candidates:
            if src.shape[0] == 0 or src.shape[1] == 0:
                continue
            resized = cv2.resize(src, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
            score = patch_similarity_score(resized, surround_stats)
            scored3.append({"patch": resized, "score": score, "direction": direction})
        if scored3:
            best3 = min(scored3, key=lambda c: c["score"])
            source_patch = best3["patch"]
            diag["method"] = "tier3_fallback"
            diag["tier3_direction"] = best3["direction"]
        else:
            source_patch = np.full((region_h, region_w, 3), surround_stats["mean"], dtype=np.uint8)
            diag["method"] = "flat_fill_last_resort"
    else:
        best = min(candidates, key=lambda c: c["score"])
        source_patch = best["patch"]
        diag["method"] = "scored_same_image"
        diag["selected_score"] = best["score"]
        diag["selected_source"] = best["source"]

    alpha = np.ones((region_h, region_w), dtype=np.float32)
    # NOTE: unlike donor-ROI removal (where fading into the original
    # pixels at the boundary is correct -- the boundary there sits in
    # genuine background), overlay removal must NOT fade back toward
    # the original pixels anywhere inside region_bbox -- the entire
    # given region is the overlay (already padded with safety margin
    # in OVERLAY_REGIONS), so any inward fade leaves a visible ring
    # of the original overlay color at the edges. Confirmed via a
    # direct synthetic test before this fix (visible yellow ring left
    # after "repair"). Fix: keep full opacity across the ENTIRE given
    # region; only feather OUTWARD into a small margin beyond it, so
    # the transition happens in pixels that were never overlay.
    feather_px = max(2, min(region_w, region_h) // 6)
    fx_min = max(0, x_min - feather_px)
    fy_min = max(0, y_min - feather_px)
    fx_max = min(img_w, x_max + feather_px)
    fy_max = min(img_h, y_max + feather_px)

    full_alpha = np.zeros((img_h, img_w), dtype=np.float32)
    full_alpha[y_min:y_max, x_min:x_max] = 1.0
    full_alpha_blurred = cv2.GaussianBlur(full_alpha, (0, 0), sigmaX=feather_px / 2)
    # Never let blurring REDUCE opacity inside the original region --
    # only let it ADD soft falloff in the margin outside it
    full_alpha_blurred[y_min:y_max, x_min:x_max] = 1.0
    alpha_crop = full_alpha_blurred[fy_min:fy_max, fx_min:fx_max][..., None]

    # Build the full-size source patch (region + outward margin) using
    # the same selected content, extended/resized to cover the padded area
    full_w, full_h = fx_max - fx_min, fy_max - fy_min
    source_full = cv2.resize(source_patch, (full_w, full_h), interpolation=cv2.INTER_LINEAR)

    target_region = result[fy_min:fy_max, fx_min:fx_max].astype(np.float32)
    blended = target_region * (1 - alpha_crop) + source_full.astype(np.float32) * alpha_crop
    result[fy_min:fy_max, fx_min:fx_max] = blended.astype(np.uint8)

    return result, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                         help="Report what would be done without writing any files")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    if not args.dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Overlay Removal Preprocessing")
    print("=" * 70)
    print(f"  Mode: {'DRY RUN (no files written)' if args.dry_run else 'WRITE'}\n")

    counts = {"repaired": 0, "copied_unchanged": 0, "unknown_source": 0}
    method_counts = {}

    for fname in sorted(os.listdir(VAL_IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        src_path = os.path.join(VAL_IMAGES_DIR, fname)
        dst_path = os.path.join(OUT_DIR, fname)
        source = get_dataset_name(fname)

        if source not in OVERLAY_REGIONS:
            if not args.dry_run:
                shutil.copy2(src_path, dst_path)
            counts["copied_unchanged"] += 1
            if source == "Unknown":
                counts["unknown_source"] += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f"  WARNING: could not read {fname}, skipping")
            continue

        region = OVERLAY_REGIONS[source]
        repaired, diag = repair_region_same_image(img, region)
        method_counts[diag["method"]] = method_counts.get(diag["method"], 0) + 1

        if not args.dry_run:
            cv2.imwrite(dst_path, repaired)
        counts["repaired"] += 1

    print(f"Repaired (overlay removed): {counts['repaired']}")
    print(f"Copied unchanged (5sagf/IR/unknown, no overlay): {counts['copied_unchanged']}")
    if counts["unknown_source"]:
        print(f"  WARNING: {counts['unknown_source']} images had UNKNOWN source -- "
              f"verify these aren't a missed overlay-bearing source")

    print(f"\nRepair method breakdown: {method_counts}")

    if args.dry_run:
        print("\nDRY RUN -- no files were written. Rerun without --dry_run to apply.")
    else:
        print(f"\nSanitized images saved -> {OUT_DIR}")
        print("\nNEXT STEPS:")
        print("  1. Manually spot-check several repaired images per source --")
        print("     confirm the overlay is genuinely gone and no new visible")
        print("     artifact was introduced.")
        print("  2. Re-run the clean-baseline reproduction check (all 4 models)")
        print("     against THIS sanitized directory, compare to the original")
        print("     locked numbers (97.04% / 96.2% / 95.4% / 95.9%).")
        print("  3. Only once both pass: point Phase 8C's composite generator")
        print("     at this sanitized directory instead of the original.")


if __name__ == "__main__":
    main()
