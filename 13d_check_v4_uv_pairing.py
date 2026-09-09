"""
13d_check_v4_uv_pairing.py
============================
Checks whether v4 and v4-UV are co-registered (paired visible/UV captures
of the same physical samples) or independent acquisitions that happen to
share a naming convention.

Since Roboflow assigns a unique export hash per file (`rf.<hash>`), the
hash itself can never match across v4 and v4-UV. What CAN indicate
pairing is the sample index embedded between the dataset prefix and the
Roboflow suffix — e.g. "Soil-Moisture-v4-23_..." vs
"Soil-Moisture-v4-UV-23_..." sharing index 23.

This is necessary-but-not-sufficient evidence: matching indices are
consistent with pairing, but could also arise from two independently
numbered sequential capture sessions. Where available, file modification
timestamps on disk are checked as a secondary (weaker, since Roboflow
re-exports can reset these) signal. The most conclusive check is visual
side-by-side inspection, which this script sets up (prints exact paths
for matched pairs) but does not perform.

Run:
    python 13d_check_v4_uv_pairing.py
"""

import os
import re
import sys
from collections import defaultdict
from importlib import import_module

sys.path.insert(0, '.')
script13 = import_module("13_confound_characterization")

LABEL_DIR_BASE = script13.LABEL_DIR_BASE  # /data/Grace/soil-moisture-dataset

# Roboflow project folder names under LABEL_DIR_BASE — confirmed via
# `ls /data/Grace/soil-moisture-dataset/` on the cluster (Sept 2026)
V4_PROJECT_CANDIDATES = ["Soil-Moisture-v4-3"]
V4UV_PROJECT_CANDIDATES = ["Soil-Moisture-v4-UV-1"]

# Matches the sample index between the dataset prefix and the Roboflow
# suffix, e.g. "Soil-Moisture-v4-23_jpg.rf.abc123.jpg" -> "23"
# and "Soil-Moisture-v4-UV-23_jpg.rf.def456.jpg" -> "23"
V4_INDEX_RE   = re.compile(r'^Soil-Moisture-v4-(\d+)_')
V4UV_INDEX_RE = re.compile(r'^Soil-Moisture-v4-UV-(\d+)_')


def find_project_dir(candidates):
    for name in candidates:
        path = os.path.join(LABEL_DIR_BASE, name)
        if os.path.isdir(path):
            return path
    # fall back: case-insensitive scan
    for entry in os.listdir(LABEL_DIR_BASE):
        if entry.lower() in [c.lower() for c in candidates]:
            return os.path.join(LABEL_DIR_BASE, entry)
    return None


def collect_index_map(project_dir, index_re):
    """Returns {sample_index: [(full_path, filename, split, mtime), ...]}
    across train/valid/test image dirs."""
    index_map = defaultdict(list)
    if project_dir is None:
        return index_map
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(project_dir, split, "images")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            m = index_re.match(fname)
            if not m:
                continue
            idx = m.group(1)
            full_path = os.path.join(img_dir, fname)
            mtime = os.path.getmtime(full_path)
            index_map[idx].append((full_path, fname, split, mtime))
    return index_map


def main():
    print("=" * 70)
    print("  v4 / v4-UV co-registration check")
    print("=" * 70)

    v4_dir = find_project_dir(V4_PROJECT_CANDIDATES)
    v4uv_dir = find_project_dir(V4UV_PROJECT_CANDIDATES)

    print(f"\nv4 project dir:    {v4_dir}")
    print(f"v4-UV project dir: {v4uv_dir}")

    if v4_dir is None or v4uv_dir is None:
        print("\nERROR: could not locate one or both project directories under "
              f"{LABEL_DIR_BASE}. Update V4_PROJECT_CANDIDATES / "
              "V4UV_PROJECT_CANDIDATES with the exact folder names and rerun.")
        return

    v4_map = collect_index_map(v4_dir, V4_INDEX_RE)
    v4uv_map = collect_index_map(v4uv_dir, V4UV_INDEX_RE)

    print(f"\nv4 images with a parsed sample index:    {sum(len(v) for v in v4_map.values())}")
    print(f"v4-UV images with a parsed sample index: {sum(len(v) for v in v4uv_map.values())}")

    v4_indices = set(v4_map.keys())
    v4uv_indices = set(v4uv_map.keys())
    shared = v4_indices & v4uv_indices
    only_v4 = v4_indices - v4uv_indices
    only_v4uv = v4uv_indices - v4_indices

    print("\n" + "-" * 70)
    print("  INDEX OVERLAP")
    print("-" * 70)
    print(f"  Unique sample indices in v4:    {len(v4_indices)}")
    print(f"  Unique sample indices in v4-UV: {len(v4uv_indices)}")
    print(f"  Shared indices (both datasets): {len(shared)}")
    print(f"  Overlap as % of v4:    {100 * len(shared) / max(1, len(v4_indices)):.1f}%")
    print(f"  Overlap as % of v4-UV: {100 * len(shared) / max(1, len(v4uv_indices)):.1f}%")
    print(f"  Indices only in v4:    {len(only_v4)}")
    print(f"  Indices only in v4-UV: {len(only_v4uv)}")

    if not shared:
        print("\nVERDICT: zero shared indices. Filenames do NOT indicate paired "
              "acquisitions -- v4 and v4-UV appear to be independently numbered "
              "sequences. Treat them as separate, unpaired acquisitions unless "
              "other evidence emerges.")
        return

    print("\n" + "-" * 70)
    print("  SAMPLE MATCHED PAIRS (first 15) -- inspect these visually")
    print("-" * 70)
    for idx in sorted(shared, key=lambda x: int(x))[:15]:
        v4_entry = v4_map[idx][0]
        v4uv_entry = v4uv_map[idx][0]
        print(f"  index={idx}")
        print(f"    v4:    {v4_entry[1]}  (split={v4_entry[2]})")
        print(f"    v4-UV: {v4uv_entry[1]}  (split={v4uv_entry[2]})")

    # Secondary, weaker signal: modification-time proximity for shared indices.
    # Roboflow re-exports commonly reset mtimes on download/unzip, so a large
    # gap does NOT rule out pairing -- only a tight cluster is suggestive.
    print("\n" + "-" * 70)
    print("  SECONDARY SIGNAL -- file mtime gap for shared indices")
    print("  (weak signal only: export/unzip can reset mtimes)")
    print("-" * 70)
    gaps = []
    for idx in shared:
        v4_mtime = v4_map[idx][0][3]
        v4uv_mtime = v4uv_map[idx][0][3]
        gaps.append(abs(v4_mtime - v4uv_mtime))
    if gaps:
        gaps.sort()
        median_gap = gaps[len(gaps) // 2]
        print(f"  Median |mtime gap| across {len(gaps)} shared indices: "
              f"{median_gap:.0f} seconds ({median_gap/86400:.2f} days)")
        print("  A tight cluster (seconds-minutes) would support same-session "
              "capture. A wide spread is inconclusive either way given "
              "re-export resets, and should not be read as evidence AGAINST pairing.")

    print("\n" + "-" * 70)
    print("  VERDICT")
    print("-" * 70)
    overlap_pct = 100 * len(shared) / max(1, min(len(v4_indices), len(v4uv_indices)))
    if overlap_pct > 80:
        print(f"  {overlap_pct:.1f}% of the smaller set's indices are shared. "
              "This level of overlap is consistent with paired visible/UV "
              "capture of the same samples, but is NOT conclusive on its own -- "
              "confirm with visual inspection of the matched pairs printed above "
              "before stating this as fact in the paper.")
    else:
        print(f"  Only {overlap_pct:.1f}% index overlap. This is weak or "
              "inconsistent evidence for pairing -- do not assume co-registration "
              "without visual confirmation of the matched pairs above.")


if __name__ == "__main__":
    main()