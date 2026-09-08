"""
13b_diagnose_unmatched_bboxes.py
==================================
Identifies exactly which test images have no matching bounding box label,
and whether they cluster by dataset or class — reuses the EXACT same
build_label_lookup() / resolve_stem() logic from Script 13, so results
are guaranteed consistent with what the radius/ordinal experiments saw.

Run:
    python 13b_diagnose_unmatched_bboxes.py
"""

import os
import sys

# Reuse Script 13's exact functions rather than duplicating logic
sys.path.insert(0, '.')
from importlib import import_module

# Import Script 13 as a module to reuse its functions/constants directly.
# (Its module-level code just builds HF_TO_CORRECT and prints it — harmless.)
script13 = import_module("13_confound_characterization")

DATA_DIR = script13.DATA_DIR
NUM_CLASSES = script13.NUM_CLASSES
build_label_lookup = script13.build_label_lookup
resolve_stem = script13.resolve_stem
get_dataset_name = script13.get_dataset_name


def main():
    print("=" * 70)
    print("  Diagnosing unmatched bounding box labels — TEST split")
    print("=" * 70)

    label_lookup = build_label_lookup()
    test_dir = os.path.join(DATA_DIR, "test")

    unmatched = []
    total = 0

    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(test_dir, str(class_idx))
        if not os.path.isdir(class_dir):
            print(f"  WARNING: directory not found: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            total += 1
            stem = os.path.splitext(fname)[0]
            resolved = resolve_stem(stem, label_lookup)
            if resolved is None:
                ds_name = get_dataset_name(fname)
                unmatched.append({
                    "filename": fname,
                    "class": class_idx,
                    "dataset": ds_name,
                    "stem_tried": stem,
                })

    print(f"\nTotal test images scanned: {total}")
    print(f"Unmatched: {len(unmatched)}\n")

    print("-" * 70)
    print("  UNMATCHED FILES — detail")
    print("-" * 70)
    for u in unmatched:
        print(f"  class={u['class']:2d}  dataset={u['dataset']:12s}  "
              f"stem_tried='{u['stem_tried']}'  file={u['filename']}")

    # Clustering check
    from collections import Counter
    by_dataset = Counter(u["dataset"] for u in unmatched)
    by_class = Counter(u["class"] for u in unmatched)

    print("\n" + "-" * 70)
    print("  CLUSTERING — by dataset")
    print("-" * 70)
    for ds, count in by_dataset.most_common():
        print(f"  {ds:15s}: {count} unmatched")

    print("\n" + "-" * 70)
    print("  CLUSTERING — by class")
    print("-" * 70)
    for cls, count in sorted(by_class.items()):
        print(f"  Class {cls:2d}: {count} unmatched")


if __name__ == "__main__":
    main()