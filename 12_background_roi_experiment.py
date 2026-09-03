"""
12_background_roi_experiment.py
================================
Dataset integrity and generalization audit.
Tests five conditions to determine whether the network is learning
moisture-related features or exploiting contextual/acquisition variables:

  Condition 1 — Full image (baseline)
  Condition 2 — ROI only (laser spot, background blacked out)
  Condition 3 — Background only (full image with laser ROI blacked out)
  Condition 4 — Outside region only (everything outside 2x expanded ROI)
  Condition 5 — Cropped and resized ROI (matches Phase 4A/4B methodology)

Condition 5 is the key addition — it extracts the laser patch and resizes
to 224x224, matching exactly how Paper 1 Phase 4A/4B achieved 87-90%.

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
# ═════════════════════════════════════════════════════════════════════════════

def build_label_lookup():
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

label_lookup = build_label_lookup()

# ═════════════════════════════════════════════════════════════════════════════
# 3. CUSTOM DATASET — five conditions
# ═════════════════════════════════════════════════════════════════════════════

class MaskedDataset(Dataset):
    """
    Loads test images and applies one of five conditions:
      'full'        — original full image (baseline)
      'roi'         — laser ROI only, background blacked out
      'background'  — background only, laser ROI blacked out
      'outside'     — everything outside 2x expanded ROI blacked out
      'crop_resize' — laser patch cropped and resized to 224x224
                      (matches Phase 4A/4B methodology from Paper 1)
    """
    def __init__(self, split, condition, label_lookup, transform):
        assert condition in ["full", "roi", "background",
                             "outside", "crop_resize"]
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
        if stem not in self.label_lookup:
            parts = stem.split('_', 1)
            if len(parts) > 1:
                stem = parts[1]
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

        # ── Condition 5: crop and resize — matches Phase 4A/4B ────────────
        if self.condition == "crop_resize":
            box = self._get_box(img_path, W, H)
            if box is not None:
                x1, y1, x2, y2 = box
                # Ensure minimum patch size
                if (x2 - x1) > 4 and (y2 - y1) > 4:
                    patch = img[y1:y2, x1:x2]
                    img   = patch  # Will be resized by transform
                # If box too small fall back to full image
            img_pil = PILImage.fromarray(img)
            tensor  = self.transform(img_pil)
            return tensor, true_class

        # ── Conditions 1-4: masking ────────────────────────────────────────
        if self.condition != "full":
            box = self._get_box(img_path, W, H)
            if box is not None:
                x1, y1, x2, y2 = box
                masked = img.copy()

                if self.condition == "roi":
                    mask = np.zeros_like(img)
                    mask[y1:y2, x1:x2] = img[y1:y2, x1:x2]
                    masked = mask

                elif self.condition == "background":
                    masked[y1:y2, x1:x2] = 0

                elif self.condition == "outside":
                    cx_c = (x1 + x2) // 2
                    cy_c = (y1 + y2) // 2
                    bw   = (x2 - x1) * 2
                    bh   = (y2 - y1) * 2
                    ex1  = max(0, cx_c - bw // 2)
                    ey1  = max(0, cy_c - bh // 2)
                    ex2  = min(W, cx_c + bw // 2)
                    ey2  = min(H, cy_c + bh // 2)
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

    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            pred_hf = outputs.argmax(dim=1)
            preds   = torch.tensor(
                [hf_to_correct[p.item()] for p in pred_hf],
                device=device)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = correct / total * 100
    return acc, all_preds, all_labels

# ═════════════════════════════════════════════════════════════════════════════
# 6. RUN ALL FIVE CONDITIONS
# ═════════════════════════════════════════════════════════════════════════════

print("\nRunning five conditions on test set...")
print("-" * 65)

conditions = ["full", "crop_resize", "roi", "background", "outside"]
results    = {}

for cond in conditions:
    print(f"\n  Condition: {cond.upper()}")
    acc, preds, labels = evaluate_condition(cond)
    results[cond] = {
        "accuracy_pct": round(acc, 2),
        "correct"     : sum(p == l for p, l in zip(preds, labels)),
        "total"        : len(labels),
    }
    print(f"  Accuracy: {acc:.2f}% ({results[cond]['correct']}/{results[cond]['total']})")

# ═════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  RESULTS SUMMARY")
print("=" * 65)
print(f"  {'Condition':<25} {'Accuracy':>10} {'Interpretation'}")
print("-" * 65)
interp = {
    "full"       : "Baseline — full image",
    "crop_resize": "Cropped ROI resized 224x224 — matches Phase 4A/4B",
    "roi"        : "ROI in-place, background blacked out",
    "background" : "Background only — acquisition confound check",
    "outside"    : "Expanded 2x context only",
}
for cond in conditions:
    acc = results[cond]["accuracy_pct"]
    print(f"  {cond:<25} {acc:>9.2f}%  {interp[cond]}")
print("=" * 65)

full_acc   = results["full"]["accuracy_pct"]
crop_acc   = results["crop_resize"]["accuracy_pct"]
roi_acc    = results["roi"]["accuracy_pct"]
bg_acc     = results["background"]["accuracy_pct"]

print(f"\n  KEY COMPARISONS:")
print(f"  Full image          : {full_acc:.2f}%")
print(f"  Cropped ROI (Ph4B)  : {crop_acc:.2f}%  "
      f"({'consistent' if crop_acc > 80 else 'lower than'} with Paper 1 Phase 4B ~90%)")
print(f"  ROI in-place        : {roi_acc:.2f}%")
print(f"  Background only     : {bg_acc:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS AND FIGURE
# ═════════════════════════════════════════════════════════════════════════════

output = {
    "model"     : "MambaVision_S Full Image Best",
    "split"     : "test",
    "conditions": results,
    "paper1_phase4b_reference": 90.64,
    "notes": {
        "crop_resize": "Extracted laser patch resized to 224x224 — matches Phase 4A/4B methodology",
        "roi"        : "ROI kept in original position, background blacked out",
        "background" : "ROI blacked out, background retained — confound check",
    }
}

with open(os.path.join(RESULTS_DIR, "background_roi_experiment.json"), "w") as f:
    json.dump(output, f, indent=2)

# Figure
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

labels_plot = [
    "Full Image\n(baseline)",
    "Cropped ROI\n(Phase 4A/4B method)",
    "ROI In-Place\n(background blacked)",
    "Background Only\n(confound check)",
    "Expanded Context\n(2x ROI)",
]
accs   = [results[c]["accuracy_pct"] for c in conditions]
colors = ["#2E7D5E", "#4C72B0", "#1A5C44", "#C0392B", "#E07B39"]

bars = ax.bar(labels_plot, accs, color=colors, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")

ax.axhline(y=90.64, color="#4C72B0", linestyle="--",
           linewidth=1.5, alpha=0.7,
           label="Paper 1 Phase 4B reference: 90.64%")
ax.axhline(y=100/11, color="red", linestyle=":",
           linewidth=1.5, alpha=0.5,
           label="Random chance (9.09%)")

ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("Background vs ROI Experiment — Dataset Integrity Audit\n"
             "Five Conditions including Phase 4A/4B-matched Crop-Resize",
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

# ═════════════════════════════════════════════════════════════════════════════
# 9. GENERATE VISUAL EXAMPLES FOR PPT / PAPER FIGURE
# ═════════════════════════════════════════════════════════════════════════════

print("\nGenerating visual condition examples for PPT...")

# Pick one representative test image that has a bounding box
sample_img_path = None
sample_box      = None

test_dataset_check = MaskedDataset("test", "full", label_lookup, val_transform)
for img_path, _ in test_dataset_check.samples:
    H_t = cv2.imread(img_path)
    if H_t is None:
        continue
    H_, W_ = H_t.shape[:2]
    box = test_dataset_check._get_box(img_path, W_, H_)
    if box is not None:
        sample_img_path = img_path
        sample_box      = box
        break

if sample_img_path is not None:
    img_orig = cv2.imread(sample_img_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    H_, W_   = img_orig.shape[:2]
    x1, y1, x2, y2 = sample_box

    # Condition images
    def make_conditions(img, x1, y1, x2, y2):
        # Full
        full = img.copy()

        # ROI only — background blacked
        roi = np.zeros_like(img)
        roi[y1:y2, x1:x2] = img[y1:y2, x1:x2]

        # Background only — ROI blacked
        bg = img.copy()
        bg[y1:y2, x1:x2] = 0

        # Crop resize — extract and resize
        patch = img[y1:y2, x1:x2]
        crop  = cv2.resize(patch, (224, 224),
                           interpolation=cv2.INTER_LINEAR)

        # Full with bounding box drawn
        boxed = img.copy()
        cv2.rectangle(boxed, (x1, y1), (x2, y2),
                      (255, 50, 50), 3)

        return full, roi, bg, crop, boxed

    full, roi, bg, crop, boxed = make_conditions(
        img_orig, x1, y1, x2, y2)

    # Resize all to same display size
    display_size = (320, 320)
    imgs_display = [
        cv2.resize(full,  display_size),
        cv2.resize(boxed, display_size),
        cv2.resize(roi,   display_size),
        cv2.resize(bg,    display_size),
        crop if crop.shape[:2] == (224, 224)
             else cv2.resize(crop, display_size),
    ]
    titles = [
        f"1. Full Image\n(baseline: 95.28%)",
        f"2. Bounding Box\n(laser spot located)",
        f"3. ROI Only\n(background blacked: 24.53%)",
        f"4. Background Only\n(ROI blacked: 85.85%)",
        f"5. Crop Resize\n(patch extracted: 23.58%)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        "Masking Conditions — Dataset Integrity Audit (Script 12)\n"
        "Same image shown under five conditions",
        fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("white")

    colors_title = ["#2E7D5E", "#534AB7",
                    "#1A5C44", "#C0392B", "#4C72B0"]

    for ax_i, (img_d, title, col) in enumerate(
            zip(imgs_display, titles, colors_title)):
        axes[ax_i].imshow(img_d)
        axes[ax_i].set_title(title, fontsize=10,
                              fontweight="bold", color=col,
                              pad=8)
        axes[ax_i].axis("off")
        for spine in axes[ax_i].spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2)
            spine.set_visible(True)

    plt.tight_layout()
    vis_path = os.path.join(RESULTS_DIR,
                            "masking_conditions_visual.png")
    plt.savefig(vis_path, dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Visual examples saved → {vis_path}")

    # Also save individual condition images for PPT
    os.makedirs(os.path.join(RESULTS_DIR, "masking_conditions"),
                exist_ok=True)
    pairs = [
        ("1_full_image.png",       full),
        ("2_bounding_box.png",     boxed),
        ("3_roi_only.png",         roi),
        ("4_background_only.png",  bg),
        ("5_crop_resize_224.png",  crop),
    ]
    for fname, img_save in pairs:
        save_path = os.path.join(RESULTS_DIR,
                                 "masking_conditions", fname)
        cv2.imwrite(save_path,
                    cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
    print(f"Individual condition images saved → "
          f"results/masking_conditions/")
else:
    print("No sample image with bounding box found — "
          "skipping visual generation")
print("\nDone!")