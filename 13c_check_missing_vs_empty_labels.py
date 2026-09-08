"""
13c_check_missing_vs_empty_labels.py
======================================
For the 16 test images identified as unmatched (13b), determine whether:
  (a) NO label .txt file exists anywhere under LABEL_DIR_BASE for that stem
  (b) A label .txt file EXISTS but is empty (build_label_lookup skips
      empty files via "if not lines: continue")
  (c) A label .txt file EXISTS with content but fewer than 5 fields
      (build_label_lookup also skips these — malformed line)

Reuses Script 13's constants/paths for consistency.

Run:
    python 13c_check_missing_vs_empty_labels.py
"""

import os
import sys
from importlib import import_module

sys.path.insert(0, '.')
script13 = import_module("13_confound_characterization")

DATA_DIR = script13.DATA_DIR
LABEL_DIR_BASE = script13.LABEL_DIR_BASE
NUM_CLASSES = script13.NUM_CLASSES
build_label_lookup = script13.build_label_lookup
resolve_stem = script13.resolve_stem
get_dataset_name = script13.get_dataset_name


def scan_all_label_files():
    """Walk every .txt label file under LABEL_DIR_BASE, regardless of
    whether it's usable, and classify each by status."""
    status_by_stem = {}  # stem -> "empty" | "malformed" | "ok"
    for ds_name in os.listdir(LABEL_DIR_BASE):
        ds_path = os.path.join(LABEL_DIR_BASE, ds_name)
        if not os.path.isdir(ds_path):
            continue
        for split in ["train", "valid", "test"]:
            lbl_dir = os.path.join(ds_path, split, "labels")
            if not os.path.exists(lbl_dir):
                continue
            for lbl_file in os.listdir(lbl_dir):
                if not lbl_file.endswith(".txt"):
                    continue
                lbl_path = os.path.join(lbl_dir, lbl_file)
                stem = os.path.splitext(lbl_file)[0]
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                if not lines:
                    status_by_stem[stem] = "empty"
                    continue
                parts = lines[0].strip().split()
                if len(parts) < 5:
                    status_by_stem[stem] = "malformed"
                    continue
                status_by_stem[stem] = "ok"
    return status_by_stem


def main():
    print("=" * 70)
    print("  Checking: missing label file vs empty/malformed label file")
    print("=" * 70)

    label_lookup = build_label_lookup()
    all_label_status = scan_all_label_files()

    test_dir = os.path.join(DATA_DIR, "test")
    unmatched = []

    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(test_dir, str(class_idx))
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            resolved = resolve_stem(stem, label_lookup)
            if resolved is not None:
                continue  # matched fine, not one of the 16

            # This is one of the 16 unmatched. Check both the raw stem
            # AND the fallback (post-underscore-split) stem against the
            # full status map to see WHY it failed.
            ds_name = get_dataset_name(fname)
            candidates = [stem]
            parts = stem.split('_', 1)
            if len(parts) > 1:
                candidates.append(parts[1])

            found_status = None
            found_as = None
            for cand in candidates:
                if cand in all_label_status:
                    found_status = all_label_status[cand]
                    found_as = cand
                    break

            if found_status is None:
                verdict = "NO LABEL FILE FOUND ANYWHERE (any naming variant)"
            else:
                verdict = f"label file EXISTS (status={found_status}, matched as '{found_as}')"

            unmatched.append({
                "filename": fname,
                "dataset": ds_name,
                "class": class_idx,
                "verdict": verdict,
            })

    print(f"\nTotal unmatched: {len(unmatched)}\n")
    print("-" * 70)
    for u in unmatched:
        print(f"  class={u['class']:2d}  dataset={u['dataset']:12s}  {u['filename']}")
        print(f"       -> {u['verdict']}")

    from collections import Counter
    verdict_types = Counter(u["verdict"].split("(")[0].strip() for u in unmatched)
    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    for v, count in verdict_types.most_common():
        print(f"  {v}: {count}")


if __name__ == "__main__":
    main()