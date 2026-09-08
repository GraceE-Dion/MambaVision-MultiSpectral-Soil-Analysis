"""
13a_heldout_feasibility_check.py
=================================
Phase 8A.4 feasibility check — run BEFORE Script 13's held-out retraining loop.

Question: For each of the 7 acquisition datasets, if we hold it out entirely
from training, do the remaining 6 datasets still cover all 11 moisture
classes with enough examples per class to train a valid model?

This does NOT train or run inference. It only reads the TRAINING split
directory structure / labels and tabulates class-by-dataset counts, then
evaluates each of the 7 possible held-out configurations.

Output:
    results/heldout_feasibility_report.json
    results/heldout_feasibility_matrix.png   (class x dataset heatmap)

Run:
    python 13a_heldout_feasibility_check.py
"""

import os
import json
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths — adjust if your training split lives elsewhere ─────────────────────
TRAIN_DIR   = "/data/Grace/Master_Soil_Moisture/train"   # CONFIRM this path
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 11
CLASS_NAMES = [f"Level_{i}" for i in range(NUM_CLASSES)]

# Minimum examples per class required in the remaining 6 datasets for the
# held-out config to be considered valid. 1 is the bare floor (technically
# trainable); a higher threshold (e.g. 3-5) is more defensible for a paper
# since a single training example per class is not a meaningful sample.
MIN_EXAMPLES_PER_CLASS = 3

# ── Dataset name detection from filename ───────────────────────────────────────
# IMPORTANT: This must match the same detection logic already used in your
# inference scripts (get_dataset_name). Confirm/adjust these patterns against
# your actual filenames before trusting the output — copy the exact function
# from 06c_evaluation.py or 07_inference_pipeline.py if it differs from this.
DATASET_PATTERNS = {
    "v4":         re.compile(r"(?<!-)v4(?!-)", re.IGNORECASE),   # v4 but not v4-IR/v4-UV
    "v4-IR":      re.compile(r"v4-IR", re.IGNORECASE),
    "v4-UV":      re.compile(r"v4-UV", re.IGNORECASE),
    "IR":         re.compile(r"(?<!v4-)IR(?!-)", re.IGNORECASE), # IR but not v4-IR
    "5sagf":      re.compile(r"5sagf", re.IGNORECASE),
    "September":  re.compile(r"september(?!.*stir)", re.IGNORECASE),
    "Stir-Sept":  re.compile(r"stir.*september", re.IGNORECASE),
}


def get_dataset_name(filename):
    """Match filename against known dataset patterns. Returns 'Unknown' if
    no pattern matches — check the report for any 'Unknown' bucket, since
    that means the patterns above need adjusting for your actual naming."""
    for name, pattern in DATASET_PATTERNS.items():
        if pattern.search(filename):
            return name
    return "Unknown"


def main():
    print("=" * 70)
    print("  Phase 8A.4 Feasibility Check — Acquisition Held-Out Evaluation")
    print("=" * 70)
    print(f"Scanning training directory: {TRAIN_DIR}\n")

    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: {TRAIN_DIR} not found. Update TRAIN_DIR at the top of "
              f"this script to match your actual training split location.")
        return

    # class_dataset_counts[class_idx][dataset_name] = count
    class_dataset_counts = defaultdict(lambda: defaultdict(int))
    unknown_files = []

    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(TRAIN_DIR, CLASS_NAMES[class_idx])
        if not os.path.isdir(class_dir):
            # Try alternate naming (e.g. just "0", "1", ... or "Level 0")
            alt_names = [str(class_idx), f"Level {class_idx}", f"level_{class_idx}"]
            found = None
            for alt in alt_names:
                alt_dir = os.path.join(TRAIN_DIR, alt)
                if os.path.isdir(alt_dir):
                    found = alt_dir
                    break
            if found is None:
                print(f"WARNING: no directory found for class {class_idx} "
                      f"(tried {CLASS_NAMES[class_idx]} and alternates)")
                continue
            class_dir = found

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ds = get_dataset_name(fname)
            if ds == "Unknown":
                unknown_files.append(fname)
            class_dataset_counts[class_idx][ds] += 1

    all_datasets = sorted(DATASET_PATTERNS.keys())

    # ── Build matrix for reporting/plotting ────────────────────────────────────
    matrix = np.zeros((NUM_CLASSES, len(all_datasets)), dtype=int)
    for class_idx in range(NUM_CLASSES):
        for j, ds in enumerate(all_datasets):
            matrix[class_idx, j] = class_dataset_counts[class_idx].get(ds, 0)

    # ── Feasibility check: for each candidate held-out dataset, do the
    #    remaining 6 still cover all 11 classes with >= MIN_EXAMPLES_PER_CLASS? ─
    feasibility = {}
    for held_out_idx, held_out_ds in enumerate(all_datasets):
        remaining_cols = [j for j in range(len(all_datasets)) if j != held_out_idx]
        remaining_counts_per_class = matrix[:, remaining_cols].sum(axis=1)

        missing_classes = [
            CLASS_NAMES[c] for c in range(NUM_CLASSES)
            if remaining_counts_per_class[c] < MIN_EXAMPLES_PER_CLASS
        ]
        held_out_test_images = matrix[:, held_out_idx].sum()
        held_out_classes_present = int(np.count_nonzero(matrix[:, held_out_idx]))

        feasibility[held_out_ds] = {
            "feasible_full_11_class": len(missing_classes) == 0,
            "missing_or_underrepresented_classes": missing_classes,
            "min_examples_threshold": MIN_EXAMPLES_PER_CLASS,
            "held_out_set_total_images": int(held_out_test_images),
            "held_out_set_classes_present": held_out_classes_present,
            "remaining_training_class_counts": {
                CLASS_NAMES[c]: int(remaining_counts_per_class[c])
                for c in range(NUM_CLASSES)
            },
        }

    # ── Print summary ────────────────────────────────────────────────────────
    print("-" * 70)
    print("  FEASIBILITY SUMMARY")
    print("-" * 70)
    for ds, result in feasibility.items():
        status = "FEASIBLE" if result["feasible_full_11_class"] else "NOT FEASIBLE (11-class)"
        print(f"  Held-out = {ds:12s}  {status}")
        if not result["feasible_full_11_class"]:
            print(f"    Missing/thin classes if held out: "
                  f"{', '.join(result['missing_or_underrepresented_classes'])}")
        print(f"    {ds} test set: {result['held_out_set_total_images']} images, "
              f"{result['held_out_set_classes_present']}/11 classes present")
    print("-" * 70)

    if unknown_files:
        print(f"\nWARNING: {len(unknown_files)} files did not match any known "
              f"dataset pattern. Dataset-detection regex needs adjustment. "
              f"Sample unmatched filenames:")
        for f in unknown_files[:5]:
            print(f"    {f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    report = {
        "min_examples_per_class_threshold": MIN_EXAMPLES_PER_CLASS,
        "datasets_checked": all_datasets,
        "unknown_file_count": len(unknown_files),
        "unknown_file_samples": unknown_files[:20],
        "feasibility_by_held_out_dataset": feasibility,
        "full_class_dataset_matrix": {
            CLASS_NAMES[c]: {
                all_datasets[j]: int(matrix[c, j]) for j in range(len(all_datasets))
            } for c in range(NUM_CLASSES)
        },
    }
    out_path = os.path.join(RESULTS_DIR, "heldout_feasibility_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {out_path}")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(all_datasets)))
    ax.set_xticklabels(all_datasets, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title("Training Set: Class x Dataset Count Matrix\n"
                  "(feasibility check for Phase 8A.4 held-out evaluation)")
    for c in range(NUM_CLASSES):
        for j in range(len(all_datasets)):
            val = matrix[c, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, c, str(val), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label="Training image count")
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "heldout_feasibility_matrix.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Heatmap saved -> {fig_path}")

    print("\n" + "=" * 70)
    print("  NEXT STEP")
    print("=" * 70)
    print("  Review the FEASIBLE datasets above. Only those should be used")
    print("  as held-out targets in Script 13's 8A.4 retraining loop.")
    print("  For NOT FEASIBLE datasets, document the limitation per v7c's")
    print("  'where class coverage permits' qualifier — do not force an")
    print("  11-class held-out test on a dataset that can't support one.")


if __name__ == "__main__":
    main()