"""
13l_build_sanitized_classifier_val.py
========================================
Reorganizes the ALREADY-sanitized flat images (from 13i, in
Master_Detection_overlay_sanitized/val/images/) into the class-folder
structure the classifier's ImageFolder-based evaluation expects
(Master_Soil_Moisture/validation/<class>/*.jpg).

Does NOT re-run overlay removal -- confirmed via md5sum that
Master_Detection/val/images and Master_Soil_Moisture/validation
contain byte-identical files (same content, same filenames, just
different folder organization), so the already-repaired images can
be directly reused here rather than reprocessed.

Usage:
    python 13l_build_sanitized_classifier_val.py
"""

import os
import shutil

SANITIZED_FLAT_DIR = "/data/Grace/Master_Detection_overlay_sanitized/val/images"
ORIGINAL_CLASSIFIER_VAL_DIR = "/data/Grace/Master_Soil_Moisture/validation"
OUT_CLASSIFIER_VAL_DIR = "/data/Grace/Master_Soil_Moisture_overlay_sanitized/validation"


def main():
    print("=" * 70)
    print("  Building Sanitized Classifier Validation Set")
    print("=" * 70)

    if not os.path.isdir(SANITIZED_FLAT_DIR):
        print(f"ERROR: {SANITIZED_FLAT_DIR} not found -- run 13i first.")
        return

    os.makedirs(OUT_CLASSIFIER_VAL_DIR, exist_ok=True)

    class_folders = sorted(os.listdir(ORIGINAL_CLASSIFIER_VAL_DIR))
    print(f"Found {len(class_folders)} class folders: {class_folders}\n")

    total_copied = 0
    total_missing = 0
    missing_files = []

    for class_folder in class_folders:
        src_class_dir = os.path.join(ORIGINAL_CLASSIFIER_VAL_DIR, class_folder)
        if not os.path.isdir(src_class_dir):
            continue
        dst_class_dir = os.path.join(OUT_CLASSIFIER_VAL_DIR, class_folder)
        os.makedirs(dst_class_dir, exist_ok=True)

        class_count = 0
        for fname in os.listdir(src_class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            sanitized_src = os.path.join(SANITIZED_FLAT_DIR, fname)
            dst_path = os.path.join(dst_class_dir, fname)

            if not os.path.exists(sanitized_src):
                total_missing += 1
                missing_files.append((class_folder, fname))
                continue

            shutil.copy2(sanitized_src, dst_path)
            class_count += 1
            total_copied += 1

        print(f"  {class_folder:12s}: {class_count} images copied")

    print(f"\nTotal copied: {total_copied}")
    if total_missing > 0:
        print(f"WARNING: {total_missing} files could not be matched to a sanitized version:")
        for cls, fname in missing_files[:10]:
            print(f"    [{cls}] {fname}")
        print("These filenames exist in Master_Soil_Moisture/validation but NOT in")
        print("the sanitized flat directory -- investigate before trusting the")
        print("baseline-reproduction check (results would be on an incomplete set).")
    else:
        print(f"All files matched successfully -- {total_copied} sanitized images ready at:")
        print(f"  {OUT_CLASSIFIER_VAL_DIR}")
        print(f"\nNext: run classifier evaluation pointed at this directory instead of")
        print(f"the original Master_Soil_Moisture/validation, compare to 97.04%.")


if __name__ == "__main__":
    main()
