"""
06d_perclass_analysis.py
========================
Runs per-class F1, precision, recall, confusion matrices, and bootstrap
confidence intervals on the test set for all four MambaVision variants.
ViT-Base is excluded as test set predictions are not available from Kaggle.

Run:
    python 06d_perclass_analysis.py

Output:
    results/perclass/perclass_report_mamba_crop.json
    results/perclass/perclass_report_mamba_full.json
    results/perclass/perclass_report_mamba_crop_fw.json
    results/perclass/perclass_report_mamba_full_fw.json
    results/perclass/confusion_matrix_mamba_crop.png
    results/perclass/confusion_matrix_mamba_full.png
    results/perclass/confusion_matrix_mamba_crop_fw.png
    results/perclass/confusion_matrix_mamba_full_fw.png
    results/perclass/perclass_f1_comparison.png
    results/perclass/bootstrap_ci_summary.json
"""

import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
import pywt
from numpy.fft import fft2, fftshift
from PIL import Image as PILImage

# ── Load MambaVision ──────────────────────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

torch.backends.cudnn.enabled = False

# ── Directories ───────────────────────────────────────────────────────────────
RESULTS_DIR  = "./results"
PERCLASS_DIR = "./results/perclass"
os.makedirs(PERCLASS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NUM_CLASSES  = 11
IMAGE_SIZE   = 224
BATCH_SIZE   = 16
N_BOOTSTRAP  = 1000
CLASS_NAMES  = [f"Level {i}" for i in range(NUM_CLASSES)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. FFT + WAVELET HELPERS (same as 05c / 05bc)
# ═════════════════════════════════════════════════════════════════════════════

def extract_fft_channel(img_np):
    gray = np.mean(img_np, axis=2)
    f    = fftshift(fft2(gray))
    mag  = np.log1p(np.abs(f))
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.astype(np.float32)

def extract_wavelet_channels(img_np):
    gray = np.mean(img_np, axis=2)
    _, (cH, cV, cD) = pywt.dwt2(gray, 'haar')
    H, W = gray.shape
    channels = []
    for coef in [cH, cV, cD]:
        coef = np.abs(coef)
        coef = (coef - coef.min()) / (coef.max() - coef.min() + 1e-8)
        coef_img = PILImage.fromarray((coef * 255).astype(np.uint8))
        coef_img = coef_img.resize((W, H), PILImage.BILINEAR)
        channels.append(np.array(coef_img).astype(np.float32) / 255.0)
    return channels

class FFTWaveletDataset(Dataset):
    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tensor, label = self.dataset[idx]
        img_np     = tensor.permute(1, 2, 0).numpy()
        fft_ch     = extract_fft_channel(img_np)
        wav_chs    = extract_wavelet_channels(img_np)
        fft_tensor = torch.from_numpy(fft_ch).unsqueeze(0)
        wav_tensor = torch.stack(
            [torch.from_numpy(c) for c in wav_chs], dim=0)
        return torch.cat([tensor, fft_tensor, wav_tensor], dim=0), label

# ═════════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

class MambaVisionFFTWavelet(nn.Module):
    def __init__(self, num_classes, in_channels=7):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.backbone   = models.mamba_vision_S(pretrained=False)
        in_features     = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        return self.backbone(x)

# ═════════════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CI FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_accuracy_ci(y_true, y_pred, n=1000, ci=95):
    """
    Bootstrap confidence interval on overall accuracy.
    Returns (mean, lower, upper).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    boot_accs = []
    for _ in range(n):
        idx  = np.random.choice(n_samples, n_samples, replace=True)
        acc  = np.mean(y_true[idx] == y_pred[idx])
        boot_accs.append(acc)
    lower = np.percentile(boot_accs, (100 - ci) / 2)
    upper = np.percentile(boot_accs, 100 - (100 - ci) / 2)
    mean  = np.mean(boot_accs)
    return round(mean * 100, 2), round(lower * 100, 2), round(upper * 100, 2)

# ═════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name, model, test_loader,
                   hf_to_correct, use_fw=False):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            # remap predictions
            preds   = np.array([hf_to_correct[p] for p in preds])
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Classification report ─────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # ── Bootstrap CI ──────────────────────────────────────────────────────
    mean_acc, ci_lower, ci_upper = bootstrap_accuracy_ci(
        all_labels, all_preds, n=N_BOOTSTRAP
    )

    # ── Print per-class table ─────────────────────────────────────────────
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*54}")
    for cls in CLASS_NAMES:
        r = report[cls]
        print(f"  {cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1-score']:>10.4f} {int(r['support']):>10}")
    print(f"  {'-'*54}")
    print(f"  {'Accuracy':<12} {'':>10} {'':>10} "
          f"{report['accuracy']:>10.4f}")
    print(f"  95% CI: [{ci_lower}%, {ci_upper}%]")

    # ── Confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(NUM_CLASSES)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion Matrix — {model_name}\n"
                 f"Test Accuracy: {report['accuracy']*100:.2f}%  "
                 f"95% CI: [{ci_lower}%, {ci_upper}%]",
                 fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("white")

    # Raw counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Raw Counts", fontsize=11)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticks(range(NUM_CLASSES))
    axes[0].set_yticks(range(NUM_CLASSES))
    axes[0].set_xticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=9)
    axes[0].set_yticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=9)
    plt.colorbar(im0, ax=axes[0])
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=8,
                         color="white" if cm[i, j] > cm.max() * 0.5 else "black")

    # Normalized
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    im1 = axes[1].imshow(cm_norm, interpolation="nearest",
                         cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Normalized", fontsize=11)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_xticks(range(NUM_CLASSES))
    axes[1].set_yticks(range(NUM_CLASSES))
    axes[1].set_xticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=9)
    axes[1].set_yticklabels([str(i) for i in range(NUM_CLASSES)], fontsize=9)
    plt.colorbar(im1, ax=axes[1])
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha="center",
                         va="center", fontsize=7,
                         color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.tight_layout()
    slug     = model_name.lower().replace(" ", "_").replace("+", "").replace("/", "")
    fig_path = os.path.join(PERCLASS_DIR, f"confusion_matrix_{slug}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Confusion matrix saved → {fig_path}")

    # ── Save JSON report ──────────────────────────────────────────────────
    output = {
        "model"           : model_name,
        "test_accuracy"   : round(report["accuracy"] * 100, 2),
        "bootstrap_ci_95" : {"mean": mean_acc, "lower": ci_lower,
                             "upper": ci_upper},
        "per_class"       : {
            cls: {
                "precision" : round(report[cls]["precision"], 4),
                "recall"    : round(report[cls]["recall"], 4),
                "f1"        : round(report[cls]["f1-score"], 4),
                "support"   : int(report[cls]["support"])
            } for cls in CLASS_NAMES
        },
        "macro_avg"       : {
            "precision" : round(report["macro avg"]["precision"], 4),
            "recall"    : round(report["macro avg"]["recall"], 4),
            "f1"        : round(report["macro avg"]["f1-score"], 4),
        },
        "weighted_avg"    : {
            "precision" : round(report["weighted avg"]["precision"], 4),
            "recall"    : round(report["weighted avg"]["recall"], 4),
            "f1"        : round(report["weighted avg"]["f1-score"], 4),
        },
        "confusion_matrix": cm.tolist(),
    }

    json_path = os.path.join(PERCLASS_DIR,
                             f"perclass_report_{slug}.png".replace(".png", ".json"))
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Per-class report saved → {json_path}")

    return output

# ═════════════════════════════════════════════════════════════════════════════
# 7. LOAD ALL FOUR TEST SETS AND MODELS
# ═════════════════════════════════════════════════════════════════════════════

CROP_DATA_DIR = "/data/Grace/Master_Laser_Crops"
FULL_DATA_DIR = "/data/Grace/Master_Soil_Moisture"

crop_train_folders = sorted(os.listdir(
    os.path.join(CROP_DATA_DIR, "train")))
full_train_folders = sorted(os.listdir(
    os.path.join(FULL_DATA_DIR, "train")))

crop_hf_to_correct = {idx: int(f) for idx, f in enumerate(crop_train_folders)}
full_hf_to_correct = {idx: int(f) for idx, f in enumerate(full_train_folders)}

# ── Test datasets ─────────────────────────────────────────────────────────
_crop_test = datasets.ImageFolder(
    os.path.join(CROP_DATA_DIR, "test"), transform=val_transform)
_crop_test.targets = [crop_hf_to_correct[t] for t in _crop_test.targets]
_crop_test.samples = [(s, crop_hf_to_correct[l]) for s, l in _crop_test.samples]

_full_test = datasets.ImageFolder(
    os.path.join(FULL_DATA_DIR, "test"), transform=val_transform)

_full_test.targets = [full_hf_to_correct[t] for t in _full_test.targets]
_full_test.samples = [(s, full_hf_to_correct[l]) for s, l in _full_test.samples]

_crop_test_fw = datasets.ImageFolder(
    os.path.join(CROP_DATA_DIR, "test"), transform=val_transform)
_crop_test_fw.targets = [crop_hf_to_correct[t] for t in _crop_test_fw.targets]
_crop_test_fw.samples = [(s, crop_hf_to_correct[l]) for s, l in _crop_test_fw.samples]

_full_test_fw = datasets.ImageFolder(
    os.path.join(FULL_DATA_DIR, "test"), transform=val_transform)
_full_test_fw.targets = [full_hf_to_correct[t] for t in _full_test_fw.targets]
_full_test_fw.samples = [(s, full_hf_to_correct[l]) for s, l in _full_test_fw.samples]

crop_test_loader    = DataLoader(_crop_test,    batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)
full_test_loader    = DataLoader(_full_test,    batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)
crop_fw_test_loader = DataLoader(FFTWaveletDataset(_crop_test_fw),
                                 batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)
full_fw_test_loader = DataLoader(FFTWaveletDataset(_full_test_fw),
                                 batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

print(f"Crop test set  : {len(_crop_test)} images")
print(f"Full test set  : {len(_full_test)} images")

# ── Load models ───────────────────────────────────────────────────────────
def load_rgb_model(path):
    m = models.mamba_vision_S(pretrained=False)
    m.head = nn.Sequential(nn.Dropout(p=0.3),
                           nn.Linear(m.head.in_features, NUM_CLASSES))
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)

def load_fw_model(path):
    m = MambaVisionFFTWavelet(num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device)

print("\nLoading models...")
model_crop    = load_rgb_model(
    os.path.join(RESULTS_DIR, "mambavision_best_model.pth"))
model_full    = load_rgb_model(
    os.path.join(RESULTS_DIR, "mambavision_fullimage_best_model.pth"))
model_crop_fw = load_fw_model(
    os.path.join(RESULTS_DIR, "mambavision_fft_wavelet_best_model.pth"))
model_full_fw = load_fw_model(
    os.path.join(RESULTS_DIR, "mambavision_fft_wavelet_fullimage_best_model.pth"))
print("All models loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 8. RUN EVALUATION FOR ALL FOUR MODELS
# ═════════════════════════════════════════════════════════════════════════════

results = {}

results["mamba_crop"] = evaluate_model(
    "MambaVision_S Laser Crops RGB",
    model_crop, crop_test_loader, crop_hf_to_correct)

results["mamba_full"] = evaluate_model(
    "MambaVision_S Full Image RGB",
    model_full, full_test_loader, full_hf_to_correct)

results["mamba_crop_fw"] = evaluate_model(
    "MambaVision_S Laser Crops FFT Wavelet",
    model_crop_fw, crop_fw_test_loader, crop_hf_to_correct, use_fw=True)

results["mamba_full_fw"] = evaluate_model(
    "MambaVision_S Full Image FFT Wavelet",
    model_full_fw, full_fw_test_loader, full_hf_to_correct, use_fw=True)

# ═════════════════════════════════════════════════════════════════════════════
# 9. FIGURE — PER-CLASS F1 COMPARISON (FOUR-WAY)
# ═════════════════════════════════════════════════════════════════════════════

print("\nGenerating per-class F1 comparison figure...")

x      = np.arange(NUM_CLASSES)
width  = 0.2
colors = ["#E07B39", "#2E7D5E", "#C0853A", "#1A5C44"]
labels = ["Mamba-Crop", "Mamba-Full", "Crop+FW", "Full+FW"]
keys   = ["mamba_crop", "mamba_full", "mamba_crop_fw", "mamba_full_fw"]

fig, ax = plt.subplots(figsize=(16, 6))
fig.patch.set_facecolor("white")

for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
    f1_vals = [results[key]["per_class"][f"Level {j}"]["f1"]
               for j in range(NUM_CLASSES)]
    ax.bar(x + i * width, f1_vals, width,
           label=label, color=color, alpha=0.85)

ax.set_xlabel("Moisture Level", fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("Per-Class F1 Score — Four MambaVision Variants\n"
             "Multi-Spectral Soil Moisture Classification (11 Classes)",
             fontsize=12, fontweight="bold")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([str(i) for i in range(NUM_CLASSES)])
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(PERCLASS_DIR, "perclass_f1_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Per-class F1 comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 10. BOOTSTRAP CI SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

ci_summary = {
    key: {
        "model"           : results[key]["model"],
        "test_accuracy"   : results[key]["test_accuracy"],
        "bootstrap_ci_95" : results[key]["bootstrap_ci_95"],
    }
    for key in results
}

ci_path = os.path.join(PERCLASS_DIR, "bootstrap_ci_summary.json")
with open(ci_path, "w") as f:
    json.dump(ci_summary, f, indent=2)
print(f"\nBootstrap CI summary saved → {ci_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 11. FINAL SUMMARY PRINT
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BOOTSTRAP CI SUMMARY — 95% CONFIDENCE INTERVALS")
print("=" * 70)
for key, val in ci_summary.items():
    ci = val["bootstrap_ci_95"]
    print(f"  {val['model']:<45} "
          f"{val['test_accuracy']}%  "
          f"CI: [{ci['lower']}%, {ci['upper']}%]")
print("=" * 70)
print(f"\nAll outputs saved to {PERCLASS_DIR}/")
print("Done! Ready for README rewrite.")