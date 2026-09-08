"""
13_confound_characterization.py
================================
Phase 8A — Confound Characterization (v7c locked plan)

Four sub-experiments, run independently via subcommand:

    python 13_confound_characterization.py radius
    python 13_confound_characterization.py source_probe
    python 13_confound_characterization.py ordinal
    python 13_confound_characterization.py heldout --dataset v4
    python 13_confound_characterization.py heldout --dataset v4-UV
    python 13_confound_characterization.py heldout --dataset September
    python 13_confound_characterization.py heldout --dataset v4-IR --partial
    ...

IMPORTANT — confirm before running:
  1. MODEL_PATH below matches your actual checkpoint filename.
  2. Bounding box loading now uses the EXACT logic copied from
     12_background_roi_experiment.py (build_label_lookup + resolve_stem
     fallback + normalized_bbox_to_pixels), confirmed against the real
     Script 12 source. `ordinal` still needs per-image prediction logs
     that may not exist yet — see run_ordinal_metrics() docstring.

Outputs go to results/phase8a/<experiment_name>/
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
from PIL import Image as PILImage
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, cohen_kappa_score
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── MambaVision ─────────────────────────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "/data/Grace/Master_Soil_Moisture"
LABEL_DIR_BASE = "/data/Grace/soil-moisture-dataset"
MODEL_PATH     = "./results/mambavision_fullimage_best_model.pth"   # CONFIRM
RESULTS_DIR    = "./results/phase8a"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 11
IMAGE_SIZE  = 224
BATCH_SIZE  = 16
device      = torch.device("cuda")

# Confirmed class remapping — computed DYNAMICALLY from directory listing,
# exactly matching Script 12's method (not hardcoded, so it stays correct
# even if folder contents/ordering ever change)
_train_folders = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
HF_TO_CORRECT = {idx: int(f) for idx, f in enumerate(_train_folders)}
print(f"Class remapping: {HF_TO_CORRECT}")
CLASS_NAMES = [f"Level_{i}" for i in range(NUM_CLASSES)]

# Dataset detection — confirmed against real filenames in 13a
DATASET_PATTERNS = {
    "v4"        : lambda f: f.startswith("Soil-Moisture-v4-") and "-IR-" not in f and "-UV-" not in f,
    "v4-IR"     : lambda f: f.startswith("Soil-Moisture-v4-IR-"),
    "v4-UV"     : lambda f: f.startswith("Soil-Moisture-v4-UV-"),
    "IR"        : lambda f: f.startswith("Soil-Moisture-IR-"),
    "5sagf"     : lambda f: f.startswith("Soil-Moisture-1_"),
    "September" : lambda f: f.startswith("Soil_Moisture_September-"),
    "Stir-Sept" : lambda f: f.startswith("Soil_Moisture_Stir_September-"),
}

# Datasets with full 11-class coverage in their held-out test set (confirmed
# via 13a_heldout_feasibility_check.py + manual remaining-count check)
FULL_COVERAGE_DATASETS = ["v4", "v4-UV", "September"]
PARTIAL_COVERAGE_DATASETS = ["v4-IR", "IR", "5sagf", "Stir-Sept"]


def get_dataset_name(filename):
    for name, matcher in DATASET_PATTERNS.items():
        if matcher(filename):
            return name
    return "Unknown"


# ═════════════════════════════════════════════════════════════════════════════
# Shared: model loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path=MODEL_PATH):
    model = models.mamba_vision_S(pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, NUM_CLASSES))
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_model_new_head(num_output_classes, freeze_backbone=True):
    """For 8A.2 — frozen-backbone linear probe with a fresh head."""
    model = models.mamba_vision_S(pretrained=False)
    in_features = model.head.in_features
    if freeze_backbone:
        # Load the full-image pretrained backbone weights first
        full_state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(full_state, strict=False)
        for param in model.parameters():
            param.requires_grad = False
    model.head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, num_output_classes))
    for param in model.head.parameters():
        param.requires_grad = True  # new head is always trainable
    model.to(device)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Shared: bounding box loading — PLACEHOLDER, paste exact logic from Script 12
# ═════════════════════════════════════════════════════════════════════════════

def build_label_lookup():
    """
    EXACT logic from 12_background_roi_experiment.py (confirmed via
    conversation search — this produced "Loaded 1026 bounding box labels").

    Returns dict: filename_stem -> (cx, cy, w, h) in NORMALIZED YOLO format
    (fractions of image width/height, 0-1 range) — NOT pixel coordinates.
    Callers must convert to pixel space using the actual image dimensions.
    """
    lookup = {}
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
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                if not lines:
                    continue
                parts = lines[0].strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                stem = os.path.splitext(lbl_file)[0]
                lookup[stem] = (cx, cy, w, h)
    print(f"Loaded {len(lookup)} bounding box labels")
    return lookup


def resolve_stem(stem, label_lookup):
    """Match Script 12's _get_box fallback exactly: if the stem isn't found
    directly, strip everything before the first underscore and retry.
    Returns the resolved stem if found, else None."""
    if stem in label_lookup:
        return stem
    parts = stem.split('_', 1)
    if len(parts) > 1 and parts[1] in label_lookup:
        return parts[1]
    return None


def normalized_bbox_to_pixels(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO (cx, cy, w, h) to pixel (x_min, y_min, x_max, y_max),
    clipped to image bounds — matches Script 12's _get_box clipping."""
    x_min = max(0, int((cx - w / 2) * img_w))
    y_min = max(0, int((cy - h / 2) * img_h))
    x_max = min(img_w, int((cx + w / 2) * img_w))
    y_max = min(img_h, int((cy + h / 2) * img_h))
    return x_min, y_min, x_max, y_max


def load_bounding_boxes():
    """
    Wrapper matching the interface used elsewhere in this script:
    returns dict {filename: (x_min, y_min, x_max, y_max)} in PIXEL space.

    IMPORTANT: build_label_lookup() keys on filename STEM (no extension),
    while callers in this script key on full filename (with extension).
    Also note it returns normalized coords, so pixel conversion requires
    the actual image dimensions — this wrapper does NOT do that conversion
    since it has no access to per-image dimensions at load time.

    Callers (crop_at_scale, masking logic) must instead call
    build_label_lookup() directly and convert per-image using
    normalized_bbox_to_pixels() with that image's actual width/height,
    matched by filename STEM not full filename. The functions below
    (run_context_radius_curve, BackgroundOnlyDataset) have been written
    to do this correctly — this stub is kept only so any other caller
    fails loudly instead of silently using wrong coordinates.
    """
    raise NotImplementedError(
        "Do not call load_bounding_boxes() directly. Use build_label_lookup() "
        "+ normalized_bbox_to_pixels() with per-image dimensions and "
        "filename-stem matching, as done in run_context_radius_curve() "
        "and BackgroundOnlyDataset.__getitem__()."
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8A.1 — Context Radius Curve
# ═════════════════════════════════════════════════════════════════════════════

def crop_at_scale(image, bbox, scale, img_w, img_h):
    """Extract a crop centered on bbox, scaled by `scale`x the bbox
    dimensions, clipped to image boundaries. Returns the crop and the
    fraction of the full image area it covers (for honest reporting,
    per peer review — a 5x box may cover very different fractions of
    the image across samples)."""
    x_min, y_min, x_max, y_max = bbox
    bw, bh = x_max - x_min, y_max - y_min
    cx, cy = x_min + bw / 2, y_min + bh / 2

    new_w, new_h = bw * scale, bh * scale
    nx_min = max(0, cx - new_w / 2)
    ny_min = max(0, cy - new_h / 2)
    nx_max = min(img_w, cx + new_w / 2)
    ny_max = min(img_h, cy + new_h / 2)

    crop = image[int(ny_min):int(ny_max), int(nx_min):int(nx_max)]
    # Match Script 12's minimum-patch-size guard — a crop smaller than 4x4
    # pixels is degenerate (e.g. from a corrupted/near-zero bbox) and should
    # not be silently resized into a meaningless 224x224 blur.
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None, 0.0
    area_fraction = ((nx_max - nx_min) * (ny_max - ny_min)) / (img_w * img_h)
    return crop, area_fraction


def run_context_radius_curve():
    print("=" * 70)
    print("  8A.1 — Context Radius Sensitivity")
    print("=" * 70)
    print("  Research question: How does classification accuracy change as")
    print("  spatial context around the laser ROI increases, compared with")
    print("  the background-only control? (per peer review framing)")
    print()
    print("  NOTE: results characterize THIS full-image-trained model's")
    print("  behavior on out-of-distribution crops/masks. They are NOT a")
    print("  definitive estimate of best-achievable ROI-only accuracy.")
    print()

    label_lookup = build_label_lookup()  # {stem: (cx, cy, w, h)} normalized
    model = load_model()

    test_dir = os.path.join(DATA_DIR, "test")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    scales = [1, 2, 3, 4, 5]
    results_by_scale = {s: {"correct": 0, "total": 0, "area_fractions": []} for s in scales}
    unmatched = 0

    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(test_dir, CLASS_NAMES[class_idx])
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            resolved_stem = resolve_stem(stem, label_lookup)
            if resolved_stem is None:
                unmatched += 1
                continue
            img_path = os.path.join(class_dir, fname)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]
            cx, cy, w, h = label_lookup[resolved_stem]
            bbox = normalized_bbox_to_pixels(cx, cy, w, h, img_w, img_h)

            for scale in scales:
                crop, area_frac = crop_at_scale(image, bbox, scale, img_w, img_h)
                if crop is None or crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE),
                                           interpolation=cv2.INTER_LINEAR)
                tensor = transform(PILImage.fromarray(crop_resized)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model(tensor).argmax(dim=1).item()
                pred_correct = HF_TO_CORRECT.get(pred, pred)

                results_by_scale[scale]["total"] += 1
                results_by_scale[scale]["area_fractions"].append(area_frac)
                if pred_correct == class_idx:
                    results_by_scale[scale]["correct"] += 1

    summary = {}
    for scale in scales:
        r = results_by_scale[scale]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0.0
        mean_area_frac = float(np.mean(r["area_fractions"])) if r["area_fractions"] else 0.0
        summary[f"{scale}x"] = {
            "accuracy": acc,
            "n": r["total"],
            "mean_area_fraction_of_full_image": mean_area_frac,
        }
        print(f"  {scale}x ROI: {acc*100:.2f}%  (n={r['total']}, "
              f"avg {mean_area_frac*100:.1f}% of full image area)")

    if unmatched > 0:
        print(f"\n  WARNING: {unmatched} test images had no matching bbox label "
              f"(filename stem not found in label_lookup). Check that image "
              f"filenames and label .txt filenames share the same stem.")

    print(f"\n  Sanity check — does 1x reproduce Script 12's CROP_RESIZE result?")
    print(f"  (Not 'roi' 24.53% — this crop-then-resize methodology matches")
    print(f"   Script 12's crop_resize condition, 23.58%, since both extract")
    print(f"   the patch and resize just that region, rather than blacking")
    print(f"   out background within the full frame like 'roi' does.)")
    print(f"  1x result here: {summary['1x']['accuracy']*100:.2f}%")
    print(f"  Script 12 crop_resize reference: 23.58%")
    print(f"  (If these differ substantially, investigate before trusting")
    print(f"   the rest of the curve — same bbox/crop logic should be used.)")

    out_path = os.path.join(RESULTS_DIR, "context_radius_curve.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [1, 2, 3, 4, 5]
    ys = [summary[f"{s}x"]["accuracy"] * 100 for s in xs]
    ax.plot(xs, ys, marker="o", linewidth=2, label="ROI at increasing scale")
    ax.axhline(y=85.85, color="coral", linestyle="--", label="Background-only (85.85%, reference)")
    ax.axhline(y=95.28, color="gray", linestyle=":", label="Full image (95.28%, reference)")
    ax.set_xlabel("ROI scale (x original bbox size)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Context Radius Sensitivity — 8A.1")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "context_radius_curve.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved -> {fig_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8A.2 — Source Dataset Classifier (frozen-backbone linear probe)
# ═════════════════════════════════════════════════════════════════════════════

class BackgroundOnlyDataset(Dataset):
    """Loads full images with the laser ROI blacked out, labeled by
    acquisition SOURCE DATASET (not moisture class). Label lookup is built
    ONCE at construction (not per __getitem__ call) and matched by
    filename stem to normalized (cx, cy, w, h) YOLO coordinates."""
    def __init__(self, root_dir, label_lookup=None, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(sorted(DATASET_PATTERNS.keys()))}
        self.label_lookup = label_lookup if label_lookup is not None else build_label_lookup()

        unmatched = 0
        for class_dir_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                ds_name = get_dataset_name(fname)
                if ds_name == "Unknown":
                    continue
                stem = os.path.splitext(fname)[0]
                resolved_stem = resolve_stem(stem, self.label_lookup)
                if resolved_stem is None:
                    unmatched += 1
                    continue
                self.samples.append((os.path.join(class_path, fname), resolved_stem, ds_name))
        if unmatched > 0:
            print(f"  BackgroundOnlyDataset({root_dir}): {unmatched} images "
                  f"skipped (no matching bbox label)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, stem, ds_name = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        cx, cy, w, h = self.label_lookup[stem]
        x_min, y_min, x_max, y_max = normalized_bbox_to_pixels(cx, cy, w, h, img_w, img_h)
        image[int(y_min):int(y_max), int(x_min):int(x_max)] = 0
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform:
            image = self.transform(PILImage.fromarray(image))
        label = self.class_to_idx[ds_name]
        return image, label


def run_source_probe(epochs=15, lr=1e-4):
    print("=" * 70)
    print("  8A.2 — Source-Identity Probe (frozen-backbone linear probe)")
    print("=" * 70)
    print("  Tests whether acquisition-source information is accessible in")
    print("  the existing representation. Trained ONLY on train split,")
    print("  evaluated ONCE on test split. Per peer review.")
    print()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    shared_label_lookup = build_label_lookup()  # built once, reused across all 3 splits
    train_ds = BackgroundOnlyDataset(os.path.join(DATA_DIR, "train"), shared_label_lookup, transform)
    val_ds   = BackgroundOnlyDataset(os.path.join(DATA_DIR, "val"), shared_label_lookup, transform)
    test_ds  = BackgroundOnlyDataset(os.path.join(DATA_DIR, "test"), shared_label_lookup, transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_sources = len(train_ds.class_to_idx)
    model = build_model_new_head(num_output_classes=num_sources, freeze_backbone=True)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"  Epoch {epoch+1}/{epochs}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    # Baselines
    uniform_baseline = 1.0 / num_sources
    label_counts = np.bincount(all_labels, minlength=num_sources)
    majority_baseline = label_counts.max() / len(all_labels)

    print(f"\n  Test accuracy       : {test_acc*100:.2f}%")
    print(f"  Balanced accuracy   : {balanced_acc*100:.2f}%")
    print(f"  Macro F1            : {macro_f1:.4f}")
    print(f"  Uniform baseline    : {uniform_baseline*100:.2f}%")
    print(f"  Majority baseline   : {majority_baseline*100:.2f}%")

    source_names = sorted(train_ds.class_to_idx.keys())
    results = {
        "test_accuracy": test_acc,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "uniform_baseline": uniform_baseline,
        "majority_baseline": majority_baseline,
        "confusion_matrix": cm.tolist(),
        "source_names": source_names,
        "methodology": "frozen_backbone_linear_probe",
    }
    out_path = os.path.join(RESULTS_DIR, "source_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(source_names)))
    ax.set_xticklabels(source_names, rotation=45, ha="right")
    ax.set_yticks(range(len(source_names)))
    ax.set_yticklabels(source_names)
    ax.set_xlabel("Predicted source")
    ax.set_ylabel("True source")
    ax.set_title("8A.2 — Source-Identity Probe Confusion Matrix")
    for i in range(len(source_names)):
        for j in range(len(source_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "source_probe_confusion.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved -> {fig_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 8A.3 — Ordinal Metrics (reuses existing saved predictions)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ordinal_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    within_1 = float(np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    return {"mae": mae, "within_1_accuracy": within_1, "quadratic_weighted_kappa": qwk}


def run_ordinal_metrics():
    """
    Self-contained — computes its own ROI-only and full-image predictions
    directly, rather than depending on Script 12 to have saved per-image
    logs (it currently only saves aggregate accuracy per condition, per
    the actual background_roi_experiment.json output). Reuses the exact
    masking logic verified from Script 12's MaskedDataset for the 'roi'
    condition (background blacked out, the condition the v7c audit table
    calls "ROI only" — 24.53% — NOT crop_resize/23.58%, a different
    condition).
    """
    print("=" * 70)
    print("  8A.3 — Ordinal Error Analysis")
    print("=" * 70)
    print("  Computing predictions directly for ROI-only and full-image")
    print("  conditions (self-contained — does not depend on Script 12")
    print("  having saved per-image logs, which it currently does not).")
    print()

    label_lookup = build_label_lookup()
    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dir = os.path.join(DATA_DIR, "test")
    conditions_results = {"roi": {"true": [], "pred": []}, "full": {"true": [], "pred": []}}
    unmatched = 0

    for class_idx in range(NUM_CLASSES):
        class_dir = os.path.join(test_dir, CLASS_NAMES[class_idx])
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(class_dir, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

            stem = os.path.splitext(fname)[0]
            resolved_stem = resolve_stem(stem, label_lookup)
            if resolved_stem is None:
                unmatched += 1
                continue
            cx, cy, w, h = label_lookup[resolved_stem]
            x_min, y_min, x_max, y_max = normalized_bbox_to_pixels(cx, cy, w, h, img_w, img_h)

            for condition in ["roi", "full"]:
                if condition == "roi":
                    masked = np.zeros_like(image)
                    masked[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
                    img_to_use = masked
                else:  # full
                    img_to_use = image

                tensor = transform(PILImage.fromarray(img_to_use)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_hf = model(tensor).argmax(dim=1).item()
                pred_correct = HF_TO_CORRECT.get(pred_hf, pred_hf)

                conditions_results[condition]["true"].append(class_idx)
                conditions_results[condition]["pred"].append(pred_correct)

    if unmatched > 0:
        print(f"  WARNING: {unmatched} test images skipped (no matching bbox label)\n")

    for label, key in [("ROI-only", "roi"), ("Full-image", "full")]:
        y_true = conditions_results[key]["true"]
        y_pred = conditions_results[key]["pred"]
        if not y_true:
            print(f"  WARNING: no predictions collected for {label}, skipping.")
            continue

        metrics = compute_ordinal_metrics(y_true, y_pred)
        exact_acc = accuracy_score(y_true, y_pred)

        print(f"  [{label}]  (n={len(y_true)})")
        print(f"    Exact accuracy          : {exact_acc*100:.2f}%")
        print(f"    MAE                      : {metrics['mae']:.3f}")
        print(f"    Within-±1 accuracy       : {metrics['within_1_accuracy']*100:.2f}%")
        print(f"    Quadratic weighted kappa : {metrics['quadratic_weighted_kappa']:.3f}")
        if key == "roi":
            print(f"    (Cross-check: exact accuracy should be close to Script 12's")
            print(f"     'roi' condition result, 24.53% — this uses the same masking.)")
        print()

        out_path = os.path.join(RESULTS_DIR, f"ordinal_metrics_{key}.json")
        with open(out_path, "w") as f:
            json.dump({
                "condition": label,
                "n": len(y_true),
                "exact_accuracy": exact_acc,
                **metrics,
                "true_labels": y_true,
                "pred_labels": y_pred,
            }, f, indent=2)
        print(f"    Saved -> {out_path}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 8A.4 — Acquisition-Held-Out Generalization
# ═════════════════════════════════════════════════════════════════════════════

class HeldOutDataset(Dataset):
    """Loads full images for training/testing a held-out-source model.
    If exclude_dataset is set, filters OUT that dataset (for training).
    If include_only_dataset is set, filters IN only that dataset (for testing)."""
    def __init__(self, root_dir, exclude_dataset=None, include_only_dataset=None, transform=None):
        self.samples = []
        self.transform = transform
        for class_idx in range(NUM_CLASSES):
            class_dir = os.path.join(root_dir, CLASS_NAMES[class_idx])
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                ds_name = get_dataset_name(fname)
                if exclude_dataset and ds_name == exclude_dataset:
                    continue
                if include_only_dataset and ds_name != include_only_dataset:
                    continue
                self.samples.append((os.path.join(class_dir, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        if self.transform:
            image = self.transform(PILImage.fromarray(image))
        return image, label


def run_heldout(held_out_dataset, partial=False, epochs=80, lr=2e-5):
    print("=" * 70)
    print(f"  8A.4 — Acquisition-Held-Out Generalization: {held_out_dataset}")
    print("=" * 70)
    if held_out_dataset in PARTIAL_COVERAGE_DATASETS and not partial:
        print(f"  WARNING: {held_out_dataset} has partial class coverage "
              f"(per 13a feasibility check). Pass --partial to acknowledge "
              f"this and evaluate only on classes present in its test set.")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = HeldOutDataset(os.path.join(DATA_DIR, "train"),
                               exclude_dataset=held_out_dataset, transform=transform)
    val_ds   = HeldOutDataset(os.path.join(DATA_DIR, "val"),
                               exclude_dataset=held_out_dataset, transform=transform)
    # Test on the held-out dataset — pooled across train+val+test splits of
    # that source, since it's entirely excluded from training
    test_ds_parts = []
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(DATA_DIR, split)
        if os.path.isdir(split_dir):
            test_ds_parts.append(HeldOutDataset(split_dir, include_only_dataset=held_out_dataset,
                                                  transform=transform))
    test_samples = []
    for part in test_ds_parts:
        test_samples.extend(part.samples)

    print(f"  Train set (excl. {held_out_dataset}): {len(train_ds)} images")
    print(f"  Held-out test set ({held_out_dataset} only): {len(test_samples)} images")

    classes_present = sorted(set(label for _, label in test_samples))
    print(f"  Classes present in held-out set: {len(classes_present)}/11")
    if partial:
        print(f"  Evaluating ONLY on present classes: "
              f"{[CLASS_NAMES[c] for c in classes_present]}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = models.mamba_vision_S(pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, NUM_CLASSES))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}  val_acc={val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate on held-out test set
    from torch.utils.data import TensorDataset
    all_preds, all_labels = [], []
    for img_path, label in test_samples:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        tensor = transform(PILImage.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor).argmax(dim=1).item()
        pred_correct = HF_TO_CORRECT.get(pred, pred)
        all_preds.append(pred_correct)
        all_labels.append(label)

    if partial:
        mask = [l in classes_present for l in all_labels]
        eval_preds = [p for p, m in zip(all_preds, mask) if m]
        eval_labels = [l for l, m in zip(all_labels, mask) if m]
    else:
        eval_preds, eval_labels = all_preds, all_labels

    held_out_acc = accuracy_score(eval_labels, eval_preds)
    print(f"\n  Held-out accuracy ({held_out_dataset}): {held_out_acc*100:.2f}%")
    print(f"  (n={len(eval_labels)}, best full-image val acc during training: {best_val_acc*100:.2f}%)")

    results = {
        "held_out_dataset": held_out_dataset,
        "partial_coverage": partial,
        "classes_evaluated": [CLASS_NAMES[c] for c in classes_present] if partial else "all",
        "n_test_images": len(eval_labels),
        "held_out_accuracy": held_out_acc,
        "best_val_accuracy_during_training": best_val_acc,
        "train_set_size": len(train_ds),
    }
    out_path = os.path.join(RESULTS_DIR, f"heldout_{held_out_dataset.replace(' ', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")

    model_out = os.path.join(RESULTS_DIR, f"heldout_{held_out_dataset.replace(' ', '_')}_model.pth")
    torch.save(model.state_dict(), model_out)
    print(f"Saved -> {model_out}")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8A confound characterization")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    subparsers.add_parser("radius", help="8A.1 — context radius curve")
    p_probe = subparsers.add_parser("source_probe", help="8A.2 — source dataset probe")
    p_probe.add_argument("--epochs", type=int, default=15)
    p_probe.add_argument("--lr", type=float, default=1e-4)

    subparsers.add_parser("ordinal", help="8A.3 — ordinal metrics")

    p_heldout = subparsers.add_parser("heldout", help="8A.4 — acquisition held-out")
    p_heldout.add_argument("--dataset", required=True,
                            choices=list(DATASET_PATTERNS.keys()))
    p_heldout.add_argument("--partial", action="store_true",
                            help="Acknowledge partial class coverage and evaluate only on present classes")
    p_heldout.add_argument("--epochs", type=int, default=80)
    p_heldout.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    if args.experiment == "radius":
        run_context_radius_curve()
    elif args.experiment == "source_probe":
        run_source_probe(epochs=args.epochs, lr=args.lr)
    elif args.experiment == "ordinal":
        run_ordinal_metrics()
    elif args.experiment == "heldout":
        run_heldout(args.dataset, partial=args.partial, epochs=args.epochs, lr=args.lr)
