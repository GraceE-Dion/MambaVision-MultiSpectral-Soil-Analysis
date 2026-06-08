"""
05bc_training_fft_wavelet_fullimage.py
=======================================
Trains MambaVision_S on the soil moisture full image dataset with
FFT and Wavelet Transform features appended as additional input channels.

Input channels: 3 RGB + 1 FFT magnitude + 3 Wavelet (H, V, D) = 7 channels
A lightweight input_proj Conv2d(7->3, 1x1) projects back to 3 channels
before the MambaVision backbone, keeping pretrained weights intact.

Run:
    python 05bc_training_fft_wavelet_fullimage.py

Output:
    results/mambavision_fft_wavelet_fullimage_training_results.json
    results/mambavision_fft_wavelet_fullimage_training_curves.png
    results/mambavision_fft_wavelet_fullimage_best_model.pth
"""

import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
import pywt
from numpy.fft import fft2, fftshift
from PIL import Image as PILImage

# ── Load MambaVision from NVlabs fork ─────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

# ── Required for cluster ──────────────────────────────────────────────────────
torch.backends.cudnn.enabled = False

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR     = "/data/Grace/Master_Soil_Moisture"
DATA_DIR_AUG = "/data/Grace/Master_Soil_Moisture_Augmented"
NUM_CLASSES  = 11
IMAGE_SIZE   = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 80
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100

# ═════════════════════════════════════════════════════════════════════════════
# 2. DEVICE
# ═════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═════════════════════════════════════════════════════════════════════════════
# 3. FFT + WAVELET FEATURE EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def extract_fft_channel(img_np):
    """
    Compute 2D FFT magnitude spectrum from grayscale image.
    Returns a single normalized channel of shape (H, W).
    """
    gray = np.mean(img_np, axis=2)
    f    = fftshift(fft2(gray))
    mag  = np.log1p(np.abs(f))
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag.astype(np.float32)

def extract_wavelet_channels(img_np):
    """
    Compute single-level Haar wavelet decomposition from grayscale image.
    Returns 3 detail sub-bands (H, V, D) each resized to (H, W).
    """
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

# ═════════════════════════════════════════════════════════════════════════════
# 4. CUSTOM DATASET — wraps ImageFolder, appends FFT + Wavelet channels
# ═════════════════════════════════════════════════════════════════════════════

class FFTWaveletDataset(Dataset):
    """
    Wraps a torchvision ImageFolder dataset.
    For each sample, appends 1 FFT channel + 3 Wavelet channels
    to the 3 RGB channels → 7-channel tensor.
    """
    def __init__(self, image_folder_dataset):
        self.dataset = image_folder_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tensor, label = self.dataset[idx]

        img_np  = tensor.permute(1, 2, 0).numpy()

        fft_ch  = extract_fft_channel(img_np)
        wav_chs = extract_wavelet_channels(img_np)

        fft_tensor = torch.from_numpy(fft_ch).unsqueeze(0)
        wav_tensor = torch.stack(
            [torch.from_numpy(c) for c in wav_chs], dim=0)

        combined = torch.cat([tensor, fft_tensor, wav_tensor], dim=0)
        return combined, label

# ═════════════════════════════════════════════════════════════════════════════
# 5. DATA TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ═════════════════════════════════════════════════════════════════════════════
# 6. DATASETS AND DATALOADERS
# ═════════════════════════════════════════════════════════════════════════════
# 5B. PHYSICAL AUGMENTATION — expand training set 3x (same as 05b_training_fullimage.py)
# Only runs if augmented folder does not already exist
# ═════════════════════════════════════════════════════════════════════════════

import shutil
import random

def add_gaussian_noise(img, mean=0, std=25):
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(mean, std, img_array.shape)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return PILImage.fromarray(noisy)

def add_salt_pepper_noise(img, amount=0.05):
    img_array = np.array(img).astype(np.uint8)
    noisy = img_array.copy()
    num_salt = int(amount * img_array.size * 0.5)
    salt_coords = [np.random.randint(0, i, num_salt)
                   for i in img_array.shape[:2]]
    noisy[salt_coords[0], salt_coords[1]] = 255
    num_pepper = int(amount * img_array.size * 0.5)
    pepper_coords = [np.random.randint(0, i, num_pepper)
                     for i in img_array.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return PILImage.fromarray(noisy)

def flip_image(img):
    choice = random.randint(0, 2)
    if choice == 0:
        return img.transpose(PILImage.FLIP_LEFT_RIGHT)
    elif choice == 1:
        return img.transpose(PILImage.FLIP_TOP_BOTTOM)
    else:
        img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
        return img.transpose(PILImage.FLIP_TOP_BOTTOM)

if not os.path.exists(DATA_DIR_AUG):
    print("\nGenerating physically augmented dataset (3x expansion)...")
    shutil.copytree(DATA_DIR, DATA_DIR_AUG)

    train_path      = os.path.join(DATA_DIR_AUG, "train")
    augmented_count = 0

    for class_folder in os.listdir(train_path):
        class_path = os.path.join(train_path, class_folder)
        if not os.path.isdir(class_path):
            continue
        original_files = [f for f in os.listdir(class_path)
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in original_files:
            img_path  = os.path.join(class_path, img_file)
            img       = PILImage.open(img_path).convert("RGB")
            base_name = img_file.rsplit('.', 1)[0]

            aug1 = flip_image(img)
            aug1 = add_gaussian_noise(aug1, mean=0, std=25)
            aug1.save(os.path.join(class_path,
                      f"{base_name}_aug_gaussian.jpg"))

            aug2 = flip_image(img)
            aug2 = add_salt_pepper_noise(aug2, amount=0.05)
            aug2.save(os.path.join(class_path,
                      f"{base_name}_aug_saltpepper.jpg"))

            augmented_count += 2

    print(f"Augmented copies added: {augmented_count}")
else:
    print(f"\nAugmented dataset already exists at {DATA_DIR_AUG} — skipping generation")

print("\nLoading datasets...")

train_folders = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
hf_to_correct = {idx: int(folder) for idx, folder in enumerate(train_folders)}
print(f"Class remapping: {hf_to_correct}")

_train_base = datasets.ImageFolder(
    os.path.join(DATA_DIR_AUG, "train"),
    transform=train_transform
)
_train_base.targets = [hf_to_correct[t] for t in _train_base.targets]

_val_base = datasets.ImageFolder(
    os.path.join(DATA_DIR, "validation"),
    transform=val_transform
)
_val_base.targets = [hf_to_correct[t] for t in _val_base.targets]

_test_base = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=val_transform
)
_test_base.targets = [hf_to_correct[t] for t in _test_base.targets]

# Wrap with FFTWaveletDataset
train_dataset = FFTWaveletDataset(_train_base)
val_dataset   = FFTWaveletDataset(_val_base)
test_dataset  = FFTWaveletDataset(_test_base)

# Weighted sampler
class_counts = np.array([
    len([x for x in _train_base.targets if x == i])
    for i in range(NUM_CLASSES)
])
class_weights  = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[t] for t in _train_base.targets]
sampler        = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    sampler=sampler, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, pin_memory=True
)

print(f"Train   : {len(train_dataset)} images")
print(f"Val     : {len(val_dataset)} images")
print(f"Test    : {len(test_dataset)} images")
print(f"Input channels per image : 7 (3 RGB + 1 FFT + 3 Wavelet)")

# ═════════════════════════════════════════════════════════════════════════════
# 7. MODEL — MambaVision_S with input_proj for 7-channel input
# ═════════════════════════════════════════════════════════════════════════════

class MambaVisionFFTWavelet(nn.Module):
    """
    Wraps MambaVision_S with a lightweight 1x1 conv input projection
    that maps 7 input channels → 3 channels before the backbone.
    Pretrained backbone weights are preserved.
    """
    def __init__(self, num_classes, in_channels=7):
        super().__init__()
        self.input_proj = nn.Conv2d(
            in_channels, 3, kernel_size=1, bias=False
        )
        nn.init.xavier_uniform_(self.input_proj.weight)

        self.backbone = models.mamba_vision_S(pretrained=True)
        in_features   = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        return self.backbone(x)

print("\nLoading MambaVision_S with FFT+Wavelet input projection...")
model = MambaVisionFFTWavelet(num_classes=NUM_CLASSES).to(device)
print("Model ready!")
print(f"input_proj: Conv2d(7 → 3, 1x1)")

# ═════════════════════════════════════════════════════════════════════════════
# 8. LOSS, OPTIMIZER, SCHEDULER
# ═════════════════════════════════════════════════════════════════════════════

loss_weights = torch.tensor(class_weights / class_weights.sum(),
                            dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(
    weight=loss_weights,
    label_smoothing=0.1
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

total_steps = NUM_EPOCHS * len(train_loader)
scheduler   = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - WARMUP_STEPS
)

# ═════════════════════════════════════════════════════════════════════════════
# 9. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
print("=" * 60)

train_losses   = []
val_losses     = []
val_accuracies = []
epoch_times    = []
best_val_acc   = 0.0
best_epoch     = 0

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        step = (epoch - 1) * len(train_loader) + batch_idx
        if step < WARMUP_STEPS:
            warmup_lr = LR * (step + 1) / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if step >= WARMUP_STEPS:
            scheduler.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    correct  = 0
    total    = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            val_loss += loss.item()
            preds    = outputs.argmax(dim=1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR,
                                "mambavision_fft_wavelet_fullimage_best_model.pth"))

    peak_mem = torch.cuda.max_memory_allocated(0) / 1e9 \
               if torch.cuda.is_available() else 0.0

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc * 100:.2f}% | "
          f"Time: {epoch_time:.1f}s | "
          f"Peak GPU: {peak_mem:.2f}GB")

print("=" * 60)
print(f"Training complete! Best Val Acc: {best_val_acc * 100:.2f}% at epoch {best_epoch}")

# ═════════════════════════════════════════════════════════════════════════════
# 10. TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

print("\nEvaluating on test set...")
model.load_state_dict(
    torch.load(os.path.join(RESULTS_DIR,
                            "mambavision_fft_wavelet_fullimage_best_model.pth"))
)
model.eval()

correct = 0
total   = 0
inference_times = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        t0      = time.time()
        outputs = model(images)
        inference_times.append((time.time() - t0) / images.size(0) * 1000)
        preds   = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

test_acc         = correct / total
avg_inference_ms = np.mean(inference_times)
print(f"Test Accuracy    : {test_acc * 100:.2f}%")
print(f"Avg Inference    : {avg_inference_ms:.2f} ms/image")

# ═════════════════════════════════════════════════════════════════════════════
# 11. SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

results = {
    "model"                    : "MambaVision_S_FFTWavelet_FullImage",
    "input_channels"           : 7,
    "feature_channels"         : "3 RGB + 1 FFT + 3 Wavelet (Haar H/V/D)",
    "num_epochs"               : NUM_EPOCHS,
    "best_val_accuracy"        : round(best_val_acc, 6),
    "best_val_accuracy_pct"    : round(best_val_acc * 100, 2),
    "best_epoch"               : best_epoch,
    "test_accuracy_pct"        : round(test_acc * 100, 2),
    "avg_epoch_time_seconds"   : round(np.mean(epoch_times), 2),
    "inference_latency_ms"     : round(avg_inference_ms, 2),
    "peak_gpu_memory_gb"       : round(peak_mem, 2),
    "training_curves": {
        "epochs"         : list(range(1, NUM_EPOCHS + 1)),
        "train_losses"   : [round(x, 6) for x in train_losses],
        "val_losses"     : [round(x, 6) for x in val_losses],
        "val_accuracies" : [round(x, 6) for x in val_accuracies],
        "epoch_times"    : [round(x, 2) for x in epoch_times],
    }
}

output_path = os.path.join(RESULTS_DIR,
                           "mambavision_fft_wavelet_fullimage_training_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved → {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 12. PLOT TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

epochs = list(range(1, NUM_EPOCHS + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MambaVision_S + FFT + Wavelet — Full Image Training\n"
             "Multi-Spectral Soil Moisture Classification (11 Classes)",
             fontsize=13, fontweight="bold")

axes[0].plot(epochs, train_losses, label="Train Loss", marker="o",
             markersize=3, color="#4C72B0", linewidth=2)
axes[0].plot(epochs, val_losses, label="Val Loss", marker="s",
             markersize=3, color="#C0392B", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(epochs, [a * 100 for a in val_accuracies],
             label="Val Accuracy", marker="o", markersize=3,
             color="#2E7D5E", linewidth=2)
axes[1].axhline(y=94.58, color="#4C72B0", linestyle="--",
                linewidth=1.5, label="ViT baseline: 94.58%")
axes[1].axhline(y=97.04, color="#2E7D5E", linestyle="--",
                linewidth=1.5, alpha=0.6, label="Mamba-Full baseline: 97.04%")
axes[1].axhline(y=90.64, color="#E07B39", linestyle="--",
                linewidth=1.5, label="Mamba-Crops baseline: 90.64%")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation Accuracy vs Baselines")
axes[1].set_ylim([0, 100])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR,
                        "mambavision_fft_wavelet_fullimage_training_curves.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Training curves saved → {fig_path}")

print("\nDone! Ready for 06c_evaluation.py")