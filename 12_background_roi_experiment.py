"""
12_background_roi_experiment.py
================================
Dataset integrity and generalization audit.
Tests four conditions to determine whether the network is learning
moisture-related features or exploiting contextual/acquisition variables:

  Condition 1 — Full image (baseline, already known: 97.04% val)
  Condition 2 — ROI only (laser spot region, cropped from bounding box)
  Condition 3 — Background only (full image with laser ROI masked/blacked out)
  Condition 4 — Outside region only (everything outside a 2x expanded ROI)

If Condition 3 achieves substantial accuracy, the model is exploiting
background/acquisition variables rather than laser moisture signatures.
If Condition 2 matches Condition 1, the laser ROI contains all the
discriminative information and full-image context adds nothing meaningful.

Run:
    python 12_background_roi_experiment.py

Output:
    results/background_roi_experiment.json
    results/background_roi_experiment.png
"""

import os
import sys
import json
import time
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
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# ── MambaVision ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = "/data/Grace/Master_Soil_Moisture"
LABEL_DIR_BASE = "/data/Grace/soil-moisture-dataset"
MODEL_PATH     = "./results/mambavision_fullimage_best_model.pth"
RESULTS_DIR    = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_CLASSES = 11
IMAGE_SIZE  = 224
BATCH_SIZE  = 16
device      = torch.device("cuda")

print("=" * 65)
print("  Background vs ROI Experiment — Dataset Integrity Audit")
print("=" * 65)
print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. CLASS REMAPPING
# ═════════════════════════════════════════════════════════════════════════════

train_folders = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
hf_to_correct = {idx: int(f) for idx, f in enumerate(train_folders)}
print(f"\nClass remapping: {hf_to_correct}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. LOAD BOUNDING BOX LABELS
# Maps filename stem → [cx, cy, w, h] (normalized YOLO format)
# ═════════════════════════════════════════════════════════════════════════════

def build_label_lookup():
    """
    Build a lookup dict: filename_stem -> (cx, cy, w, h) normalized.
    Searches all source dataset label folders.
    """
    lookup = {}
    for ds_name in os.listdir(LABEL_DIR_BASE):
        ds_path = os.path.join(LABEL_DIR_BASE, ds_name)
        if not os.path.isdir(ds_path):
            continue
        for split in ["train", "valid", "test"]:
            lbl_dir = os.path.join(ds_path, split, "labels")
            img_dir = os.path.join(ds_path, split, "images")
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
                # Take first box only
                parts = lines[0].strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                stem = os.path.splitext(lbl_file)[0]
                lookup[stem] = (cx, cy, w, h)
    print(f"Loaded {len(lookup)} bounding box labels")
    return lookup

label_lookup = build_label_lookup()

# ═════════════════════════════════════════════════════════════════════════════
# 3. CUSTOM DATASET — four conditions
# ═════════════════════════════════════════════════════════════════════════════

class MaskedDataset(Dataset):
    """
    Loads test images and applies one of four masking conditions:
      'full'       — original full image (baseline)
      'roi'        — laser ROI only, background blacked out
      'background' — background only, laser ROI blacked out
      'outside'    — everything outside 2x expanded ROI blacked out
    """
    def __init__(self, split, condition, label_lookup, transform):
        assert condition in ["full", "roi", "background", "outside"]
        self.condition    = condition
        self.label_lookup = label_lookup
        self.transform    = transform
        self.samples      = []

        base = datasets.ImageFolder(
            os.path.join(DATA_DIR, split), transform=None)
        for img_path, class_idx in base.samples:
            true_class = hf_to_correct[class_idx]
            self.samples.append((img_path, true_class))

    def __len__(self):
        return len(self.samples)

    def _get_box(self, img_path, W, H):
        """Look up bounding box for this image."""
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if stem in self.label_lookup:
            cx, cy, w, h = self.label_lookup[stem]
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            return x1, y1, x2, y2
        return None

    def __getitem__(self, idx):
        img_path, true_class = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        if self.condition != "full":
            box = self._get_box(img_path, W, H)

            if box is not None:
                x1, y1, x2, y2 = box
                masked = img.copy()

                if self.condition == "roi":
                    # Keep only the laser ROI, black out everything else
                    mask = np.zeros_like(img)
                    mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                    masked = mask

                elif self.condition == "background":
                    # Black out the laser ROI, keep background
                    masked[y1:y2, x1:x2] = 0

                elif self.condition == "outside":
                    # Expand ROI by 2x and black out everything outside
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    bw = (x2 - x1) * 2
                    bh = (y2 - y1) * 2
                    ex1 = max(0, cx - bw // 2)
                    ey1 = max(0, cy - bh // 2)
                    ex2 = min(W, cx + bw // 2)
                    ey2 = min(H, cy + bh // 2)
                    mask = np.zeros_like(img)
                    mask[ey1:ey2, ex1:ex2] = img[ey1:ey2, ex1:ex2]
                    masked = mask

                img = masked

        img_pil = PILImage.fromarray(img)
        tensor  = self.transform(img_pil)
        return tensor, true_class

# ═════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMS AND MODEL
# ═════════════════════════════════════════════════════════════════════════════

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

print("\nLoading MambaVision_S full image best model...")
model = models.mamba_vision_S(pretrained=False)
model.head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.head.in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()
print("Model loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_condition(condition, split="test"):
    dataset = MaskedDataset(split, condition, label_lookup, val_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4)

    correct = 0
    total   = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = correct / total * 100
    return acc, all_preds, all_labels

# ═════════════════════════════════════════════════════════════════════════════
# 6. RUN ALL FOUR CONDITIONS
# ═════════════════════════════════════════════════════════════════════════════

print("\nRunning four masking conditions on test set...")
print("-" * 65)

conditions = ["full", "roi", "background", "outside"]
results    = {}

for cond in conditions:
    print(f"\n  Condition: {cond.upper()}")
    acc, preds, labels = evaluate_condition(cond)
    results[cond] = {
        "accuracy_pct": round(acc, 2),
        "correct"     : sum(p == l for p, l in zip(preds, labels)),
        "total"       : len(labels),
    }
    print(f"  Accuracy: {acc:.2f}% ({results[cond]['correct']}/{results[cond]['total']})")

# ═════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY AND INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"  {'Condition':<20} {'Accuracy':>10} {'Interpretation'}")
print("-" * 65)
interp = {
    "full":       "Baseline — known result",
    "roi":        "Laser spot only — pure moisture signal",
    "background": "Background only — acquisition confound check",
    "outside":    "Expanded context — spatial context check",
}
for cond in conditions:
    acc = results[cond]["accuracy_pct"]
    print(f"  {cond:<20} {acc:>9.2f}%  {interp[cond]}")
print("=" * 65)

# Interpretation guidance
bg_acc   = results["background"]["accuracy_pct"]
roi_acc  = results["roi"]["accuracy_pct"]
full_acc = results["full"]["accuracy_pct"]

print("\n  INTERPRETATION:")
if bg_acc > 60:
    print(f"  WARNING: Background-only accuracy {bg_acc:.2f}% is substantial.")
    print(f"  This suggests the model may be exploiting acquisition")
    print(f"  conditions (background, lighting, tray, session) rather")
    print(f"  than laser moisture signatures alone.")
elif bg_acc > 30:
    print(f"  CAUTION: Background-only accuracy {bg_acc:.2f}% is moderate.")
    print(f"  Some acquisition-correlated information may be present.")
else:
    print(f"  GOOD: Background-only accuracy {bg_acc:.2f}% is low.")
    print(f"  The model is not primarily exploiting background context.")

if roi_acc > 90:
    print(f"  GOOD: ROI-only accuracy {roi_acc:.2f}% is high.")
    print(f"  The laser spot contains strong discriminative signal.")
elif roi_acc > 70:
    print(f"  MODERATE: ROI-only accuracy {roi_acc:.2f}%.")
    print(f"  Laser spot has useful signal but context also contributes.")
else:
    print(f"  NOTE: ROI-only accuracy {roi_acc:.2f}% is lower than expected.")
    print(f"  Full image context is critical for classification.")

gap = full_acc - roi_acc
if gap > 10:
    print(f"  The {gap:.2f}% gap between full ({full_acc:.2f}%) and ROI-only")
    print(f"  ({roi_acc:.2f}%) suggests context adds genuine signal — but")
    print(f"  cross-check with background accuracy to rule out confounding.")

# ═════════════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS AND FIGURE
# ═════════════════════════════════════════════════════════════════════════════

output = {
    "model"     : "MambaVision_S Full Image Best",
    "split"     : "test",
    "conditions": results,
    "interpretation": {
        "background_confound_risk": "high" if bg_acc > 60 else "moderate" if bg_acc > 30 else "low",
        "laser_signal_strength"   : "high" if roi_acc > 90 else "moderate" if roi_acc > 70 else "low",
        "full_vs_roi_gap_pct"     : round(full_acc - roi_acc, 2),
    }
}

with open(os.path.join(RESULTS_DIR, "background_roi_experiment.json"), "w") as f:
    json.dump(output, f, indent=2)

# Figure
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")

labels_plot = ["Full Image\n(baseline)", "ROI Only\n(laser spot)", "Background Only\n(confound check)", "Expanded Context\n(2x ROI)"]
accs        = [results[c]["accuracy_pct"] for c in conditions]
colors      = ["#2E7D5E", "#4C72B0", "#C0392B", "#E07B39"]

bars = ax.bar(labels_plot, accs, color=colors, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", fontsize=11, fontweight="bold")

ax.axhline(y=50, color="gray", linestyle="--", linewidth=1.5,
           alpha=0.5, label="Random chance baseline (~9% for 11 classes)")
ax.axhline(y=100/11, color="red", linestyle=":", linewidth=1.5,
           alpha=0.5, label="Random chance (9.09%)")

ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Background vs ROI Experiment — Dataset Integrity Audit\n"
             "MambaVision_S Full Image Model — What Is the Network Learning?",
             fontsize=12, fontweight="bold")
ax.set_ylim([0, 110])
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "background_roi_experiment.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nResults saved → results/background_roi_experiment.json")
print(f"Figure saved  → results/background_roi_experiment.png")
print("\nDone! Share results with peer reviewer before proceeding.")