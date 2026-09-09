"""
13d_check_v4_uv_pairing.py
============================
Checks whether v4 and v4-UV share sample indices — evidence for whether
they represent the same physical soil samples photographed under visible
vs UV light, or independently numbered acquisitions.

Filenames inside the raw Roboflow export folders follow the pattern
"<index>_png.rf.<hash>.jpg" with NO dataset-name prefix (confirmed via
direct ls on the cluster) — the prefix only appears later when images
are consolidated into Master_Soil_Moisture.

Run:
    python 13d_check_v4_uv_pairing.py
"""

import os
import re
from collections import defaultdict

LABEL_DIR_BASE = "/data/Grace/soil-moisture-dataset"
V4_PROJECT_DIR    = os.path.join(LABEL_DIR_BASE, "Soil-Moisture-v4-3")
V4UV_PROJECT_DIR  = os.path.join(LABEL_DIR_BASE, "Soil-Moisture-v4-UV-1")

# Confirmed pattern: "<index>_png.rf.<hash>.jpg"
INDEX_PATTERN = re.compile(r'^(\d+)_png\.rf\.')


def collect_indices(project_dir):
    """Walk train/valid/test/images under project_dir, extract sample
    indices, and track which split each index came from."""
    indices = {}  # index -> list of (split, filename)
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(project_dir, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            match = INDEX_PATTERN.match(fname)
            if match:
                idx = int(match.group(1))
                indices.setdefault(idx, []).append((split, fname))
    return indices


def main():
    print("=" * 70)
    print("  v4 / v4-UV co-registration check")
    print("=" * 70)
    print(f"\nv4 project dir:    {V4_PROJECT_DIR}")
    print(f"v4-UV project dir: {V4UV_PROJECT_DIR}\n")

    v4_indices = collect_indices(V4_PROJECT_DIR)
    v4uv_indices = collect_indices(V4UV_PROJECT_DIR)

    print(f"v4 images with a parsed sample index:    {sum(len(v) for v in v4_indices.values())}")
    print(f"v4-UV images with a parsed sample index: {sum(len(v) for v in v4uv_indices.values())}")

    v4_set = set(v4_indices.keys())
    v4uv_set = set(v4uv_indices.keys())
    shared = v4_set & v4uv_set
    only_v4 = v4_set - v4uv_set
    only_v4uv = v4uv_set - v4_set

    print("\n" + "-" * 70)
    print("  INDEX OVERLAP")
    print("-" * 70)
    print(f"  Unique sample indices in v4:    {len(v4_set)}")
    print(f"  Unique sample indices in v4-UV: {len(v4uv_set)}")
    print(f"  Shared indices (both datasets): {len(shared)}")
    if v4_set:
        print(f"  Overlap as % of v4:    {len(shared) / len(v4_set) * 100:.1f}%")
    if v4uv_set:
        print(f"  Overlap as % of v4-UV: {len(shared) / len(v4uv_set) * 100:.1f}%")
    print(f"  Indices only in v4:    {len(only_v4)}")
    print(f"  Indices only in v4-UV: {len(only_v4uv)}")

    # Check for shared gaps — a coincidental shared missing index between
    # two independently-numbered sequences would be a striking coincidence;
    # look for gaps across the full observed range in both.
    if v4_set and v4uv_set:
        full_range = range(min(v4_set | v4uv_set), max(v4_set | v4uv_set) + 1)
        missing_v4 = set(full_range) - v4_set
        missing_v4uv = set(full_range) - v4uv_set
        shared_gaps = missing_v4 & missing_v4uv
        print(f"\n  Shared missing indices (gap in BOTH sequences): {len(shared_gaps)}")
        if shared_gaps:
            sample_gaps = sorted(shared_gaps)[:10]
            print(f"    e.g. {sample_gaps}")
            print("    NOTE: a shared gap in two independently-numbered sequences")
            print("    would be a notable coincidence — this is circumstantial")
            print("    evidence FOR pairing, not proof. Confirm with Dr. Zhang.")

    # Split consistency check — if paired, same index should usually land
    # in the same split (train/valid/test) in both datasets, since Roboflow
    # typically preserves capture-session grouping. Deviations don't
    # disprove pairing but are worth knowing about.
    split_mismatches = 0
    split_matches = 0
    for idx in shared:
        v4_splits = set(s for s, _ in v4_indices[idx])
        v4uv_splits = set(s for s, _ in v4uv_indices[idx])
        if v4_splits == v4uv_splits:
            split_matches += 1
        else:
            split_mismatches += 1
    if shared:
        print(f"\n  Split consistency (train/valid/test) for shared indices:")
        print(f"    Same split in both: {split_matches}/{len(shared)}")
        print(f"    Different split:    {split_mismatches}/{len(shared)}")

    print("\n" + "-" * 70)
    print("  VERDICT")
    print("-" * 70)
    if len(v4_set) == 0 or len(v4uv_set) == 0:
        print("  ERROR: no indices parsed from one or both datasets.")
        print("  The regex did not match — check filenames manually.")
    else:
        overlap_pct_v4 = len(shared) / len(v4_set) * 100
        overlap_pct_v4uv = len(shared) / len(v4uv_set) * 100
        if overlap_pct_v4 > 80 and overlap_pct_v4uv > 80:
            print(f"  STRONG overlap ({overlap_pct_v4:.1f}% of v4, {overlap_pct_v4uv:.1f}% of v4-UV).")
            print("  Filenames are consistent with v4 and v4-UV being the SAME")
            print("  physical samples photographed under visible vs UV light.")
            print("  This is circumstantial (filename-based) evidence, not")
            print("  confirmation — verify with Dr. Zhang before stating this")
            print("  as fact in the paper.")
        elif overlap_pct_v4 > 20 or overlap_pct_v4uv > 20:
            print(f"  PARTIAL overlap ({overlap_pct_v4:.1f}% of v4, {overlap_pct_v4uv:.1f}% of v4-UV).")
            print("  Some shared indices exist but coverage is incomplete —")
            print("  could be partial pairing, or coincidental overlap in a")
            print("  shared numbering convention. Needs manual verification.")
        else:
            print(f"  LOW overlap ({overlap_pct_v4:.1f}% of v4, {overlap_pct_v4uv:.1f}% of v4-UV).")
            print("  Filenames do not indicate paired acquisitions.")


if __name__ == "__main__":
    main()