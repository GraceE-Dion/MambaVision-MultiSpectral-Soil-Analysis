"""
13h_split_leakage_audit.py
=============================
Checks whether near-duplicate, closely-timed captures of the same
physical setup have landed in DIFFERENT splits (train vs valid vs
test) -- which would let a model partially "see" a near-identical
image at test/val time that it already encountered in training,
inflating every accuracy number built on these splits.

Motivated by a real, human-confirmed example: WIN_20231111_14_40_32
(Soil-Moisture-1/train) and WIN_20231111_14_41_03 (Soil-Moisture-1/
valid) are only 31 seconds apart, and the surrounding train frames
are captured 5-11 seconds apart -- consistent with this dataset being
a continuous capture sequence cut into splits at some point mid-
sequence, not independently sampled per split.

IMPORTANT LIMITATION, stated explicitly rather than glossed over:
only sources whose filenames embed a real capture timestamp
(WIN_YYYYMMDD_HH_MM_SS_Pro pattern) can be checked this way.
September and Stir-September use bare sequential index filenames
(e.g. "10_png.rf...") with NO embedded timestamp -- this script
CANNOT check those two sources for this specific leakage pattern.
That gap is reported explicitly in the output, not silently skipped.

Usage:
    python 13h_split_leakage_audit.py --threshold_seconds 60
"""

import os
import re
import json
import argparse
from datetime import datetime
from collections import defaultdict

RAW_DATASET_DIR = "/data/Grace/soil-moisture-dataset"
OUT_DIR = "./results/phase8c"

# Source folder name -> human-readable label
SOURCE_FOLDERS = {
    "Soil-Moisture-1": "5sagf",
    "Soil-Moisture-IR-1": "IR",
    "Soil-Moisture-v4-IR-1": "v4-IR",
    "Soil-Moisture-v4-UV-1": "v4-UV",
    "Soil-Moisture-v4-3": "v4",
    "Soil_Moisture_September-8": "September",
    "Soil_Moisture_Stir_September-4": "Stir-Sept",
}

SPLITS = ["train", "valid", "test"]

# Matches WIN_20231111_14_40_32_Pro... -> extracts YYYYMMDD, HH, MM, SS
TIMESTAMP_PATTERN = re.compile(r"WIN_(\d{8})_(\d{2})_(\d{2})_(\d{2})_Pro")


def extract_timestamp(filename):
    m = TIMESTAMP_PATTERN.search(filename)
    if not m:
        return None
    date_str, hh, mm, ss = m.groups()
    try:
        return datetime.strptime(f"{date_str}{hh}{mm}{ss}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold_seconds", type=int, default=60,
                         help="Flag any cross-split pair captured within this many seconds of each other")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  Split-Leakage Timestamp-Proximity Audit")
    print("=" * 70)
    print(f"  Threshold: flagging cross-split pairs within {args.threshold_seconds} seconds\n")

    all_flags = []
    no_timestamp_sources = []

    for folder, label in SOURCE_FOLDERS.items():
        source_path = os.path.join(RAW_DATASET_DIR, folder)
        if not os.path.isdir(source_path):
            print(f"  WARNING: expected folder not found: {source_path}")
            continue

        # Collect (timestamp, filename, split) for every image with an
        # extractable timestamp in this source
        entries = []
        any_files_seen = False
        for split in SPLITS:
            img_dir = os.path.join(source_path, split, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                any_files_seen = True
                ts = extract_timestamp(fname)
                if ts is not None:
                    entries.append((ts, fname, split))

        if any_files_seen and not entries:
            no_timestamp_sources.append(label)
            print(f"  {label}: NO extractable timestamps in any filename "
                  f"(different naming convention -- cannot check this source)")
            continue

        if not entries:
            print(f"  {label}: no image files found at all -- check path")
            continue

        entries.sort(key=lambda e: e[0])

        # Check every consecutive pair (after sorting by time) for
        # cross-split proximity within the threshold
        source_flags = []
        for i in range(len(entries) - 1):
            ts1, fname1, split1 = entries[i]
            ts2, fname2, split2 = entries[i + 1]
            if split1 == split2:
                continue  # same split, not a leakage concern
            gap = (ts2 - ts1).total_seconds()
            if gap <= args.threshold_seconds:
                source_flags.append({
                    "source": label,
                    "file_a": fname1, "split_a": split1, "time_a": ts1.isoformat(),
                    "file_b": fname2, "split_b": split2, "time_b": ts2.isoformat(),
                    "gap_seconds": gap,
                })

        all_flags.extend(source_flags)
        print(f"  {label}: {len(entries)} timestamped images, "
              f"{len(source_flags)} cross-split pairs within {args.threshold_seconds}s")

    print("\n" + "=" * 70)
    print(f"  TOTAL FLAGGED CROSS-SPLIT PAIRS: {len(all_flags)}")
    print("=" * 70)
    if all_flags:
        print("\n  Worst (smallest gap) cases:")
        for flag in sorted(all_flags, key=lambda f: f["gap_seconds"])[:10]:
            print(f"    [{flag['source']}] {flag['gap_seconds']:.0f}s apart: "
                  f"{flag['split_a']}/{flag['file_a']}  <->  {flag['split_b']}/{flag['file_b']}")

    if no_timestamp_sources:
        print(f"\n  COULD NOT CHECK (no embedded timestamp): {', '.join(no_timestamp_sources)}")
        print("  These sources need a different leakage-check approach (e.g. visual")
        print("  near-duplicate detection) if this pattern is confirmed elsewhere.")

    out_path = os.path.join(OUT_DIR, "split_leakage_audit.json")
    with open(out_path, "w") as f:
        json.dump({
            "threshold_seconds": args.threshold_seconds,
            "flagged_pairs": all_flags,
            "sources_without_timestamps": no_timestamp_sources,
        }, f, indent=2)
    print(f"\nFull results saved -> {out_path}")


if __name__ == "__main__":
    main()
