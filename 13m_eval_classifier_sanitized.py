"""
13m_eval_classifier_sanitized.py
====================================
Evaluation-only (no training) reuse of 05b_training_fullimage.py's
model-loading, class-remapping, and accuracy-computation logic --
loads the ALREADY-TRAINED checkpoint and evaluates it against the
sanitized (overlay-removed) validation set, compared to the locked
97.04% baseline.

IMPORTANT: must run in the 'mambavision' conda environment (needs
mamba_ssm), NOT 'mambayolo' -- confirmed this session that running
MambaVision code in the wrong environment fails with
ModuleNotFoundError: No module named 'mamba_ssm'.

    conda activate mambavision
    python 13m_eval_classifier_sanitized.py

Class remapping is rebuilt from the ORIGINAL Master_Soil_Moisture
train folder listing (same logic as 05b) -- the sanitized validation
directory doesn't have its own train folder, but since class-folder
names ("0".."10") are identical across train/val, ImageFolder assigns
the same alphabetical-sort index order to both, so reusing the
train-derived remapping is correct, matching 05b's own approach
exactly.
"""

import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

torch.backends.cudnn.enabled = False

ORIGINAL_DATA_DIR = "/data/Grace/Master_Soil_Moisture"  # for train folder listing only
SANITIZED_VAL_DIR = "/data/Grace/Master_Soil_Moisture_overlay_sanitized/validation"
CHECKPOINT_PATH = "./results/mambavision_fullimage_best_model.pth"
NUM_CLASSES = 11
IMAGE_SIZE = 224
BATCH_SIZE = 16

LOCKED_BASELINE_VAL_ACC = 97.04

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"ERROR: checkpoint not found at {CHECKPOINT_PATH}")
    print("Run this from the same directory 05b_training_fullimage.py was run from.")
    sys.exit(1)

if not os.path.isdir(SANITIZED_VAL_DIR):
    print(f"ERROR: sanitized val directory not found at {SANITIZED_VAL_DIR}")
    print("Run 13l_build_sanitized_classifier_val.py first.")
    sys.exit(1)

# Same class remapping logic as 05b (alphabetical-folder-sort vs real
# integer class label -- the same fix documented elsewhere in this
# project for the "10" sorting between "1" and "2" issue)
train_folders = sorted(os.listdir(os.path.join(ORIGINAL_DATA_DIR, "train")))
hf_to_correct = {idx: int(folder) for idx, folder in enumerate(train_folders)}
print(f"Class remapping (from train folder order): {hf_to_correct}")

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(SANITIZED_VAL_DIR, transform=val_transform)
val_dataset.targets = [hf_to_correct[t] for t in val_dataset.targets]
print(f"Sanitized val dataset: {len(val_dataset)} images")

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

print("\nLoading MambaVision_S...")
model = models.mamba_vision_S(pretrained=False)
in_features = model.head.in_features
model.head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features, NUM_CLASSES))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model + checkpoint loaded.")

correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

sanitized_val_acc = correct / total * 100

print("\n" + "=" * 60)
print("  SANITIZED-BASELINE REPRODUCTION CHECK")
print("=" * 60)
print(f"  Original locked val accuracy   : {LOCKED_BASELINE_VAL_ACC}%")
print(f"  Sanitized (overlay-removed)    : {sanitized_val_acc:.2f}%")
print(f"  Delta                          : {sanitized_val_acc - LOCKED_BASELINE_VAL_ACC:+.2f} points")
print("=" * 60)

if abs(sanitized_val_acc - LOCKED_BASELINE_VAL_ACC) < 2.0:
    print("\nSmall delta -- overlay removal did not meaningfully change baseline")
    print("performance. Safe to treat this as reproducing the locked result.")
else:
    print("\nLARGE delta -- investigate before treating overlay removal as safe.")
    print("Could mean: (a) the overlay itself was being used as a signal by the")
    print("model [a genuinely interesting finding in its own right], or (b) the")
    print("repair damaged real image content the model relies on. Check which by")
    print("visually inspecting several sanitized images the model got wrong.")

results = {
    "original_locked_val_acc_pct": LOCKED_BASELINE_VAL_ACC,
    "sanitized_val_acc_pct": round(sanitized_val_acc, 2),
    "delta_pct_points": round(sanitized_val_acc - LOCKED_BASELINE_VAL_ACC, 2),
    "n_images": total,
    "n_correct": correct,
}
os.makedirs("./results/phase8c", exist_ok=True)
with open("./results/phase8c/sanitized_classifier_baseline_check.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved -> ./results/phase8c/sanitized_classifier_baseline_check.json")
