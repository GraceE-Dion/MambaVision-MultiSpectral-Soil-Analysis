"""
07bc_inference_pipeline.py
==========================
Runs inference on full image test images using the best
MambaVision_S + FFT + Wavelet full image model.
Samples 50 images across 7 source datasets.
Produces annotated images in the same format as 07b_inference_pipeline.py.

Fonts  : LiberationSans-Bold 36pt, LiberationSans-Regular 28pt
Panel  : 640x750, image pasted at (0,200), text starts at y=25
Result : bar at [0, 690, 640, 750], text at y=698

Run:
    python 07bc_inference_pipeline.py

Output:
    results/inference_images_fft_wavelet_fullimage/   — annotated images
    results/inference_summary_fft_wavelet_fullimage.json
"""

import json
import os
import sys
import time
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import pywt
from numpy.fft import fft2, fftshift

# ── Load MambaVision from NVlabs fork ─────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

# ── Required for cluster ──────────────────────────────────────────────────────
torch.backends.cudnn.enabled = False

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
OUTPUT_DIR  = "./results/inference_images_fft_wavelet_fullimage"
MODEL_PATH  = "./results/mambavision_fft_wavelet_fullimage_best_model.pth"
DATA_DIR    = "/data/Grace/Master_Soil_Moisture"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = 11
IMAGE_SIZE  = 224

SAMPLES_PER_DATASET = {
    "Soil-Moisture-v4"            : 8,
    "Soil-Moisture-v4-IR"         : 7,
    "Soil-Moisture-v4-UV"         : 7,
    "Soil-Moisture-IR"            : 7,
    "Soil-Moisture-5sagf"         : 7,
    "Soil-Moisture-September"     : 7,
    "Soil-Moisture-Stir-September": 7,
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. DEVICE
# ═════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. FFT + WAVELET FEATURE EXTRACTOR
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
        coef_img = Image.fromarray((coef * 255).astype(np.uint8))
        coef_img = coef_img.resize((W, H), Image.BILINEAR)
        channels.append(np.array(coef_img).astype(np.float32) / 255.0)
    return channels

def build_7channel_tensor(img_pil, transform):
    """Apply transform then append FFT + Wavelet channels."""
    tensor  = transform(img_pil)
    img_np  = tensor.permute(1, 2, 0).numpy()
    fft_ch  = extract_fft_channel(img_np)
    wav_chs = extract_wavelet_channels(img_np)
    fft_tensor = torch.from_numpy(fft_ch).unsqueeze(0)
    wav_tensor = torch.stack(
        [torch.from_numpy(c) for c in wav_chs], dim=0)
    return torch.cat([tensor, fft_tensor, wav_tensor], dim=0)

# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL — MambaVisionFFTWavelet wrapper (same as 05bc)
# ═════════════════════════════════════════════════════════════════════════════

class MambaVisionFFTWavelet(nn.Module):
    def __init__(self, num_classes, in_channels=7):
        super().__init__()
        self.input_proj = nn.Conv2d(
            in_channels, 3, kernel_size=1, bias=False
        )
        self.backbone = models.mamba_vision_S(pretrained=False)
        in_features   = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        return self.backbone(x)

print("Loading MambaVision_S + FFT/Wavelet full image model...")
model = MambaVisionFFTWavelet(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 5. TRANSFORMS
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
# 6. CLASS REMAPPING
# ═════════════════════════════════════════════════════════════════════════════

train_folders = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
hf_to_correct = {idx: int(folder)
                 for idx, folder in enumerate(train_folders)}
print(f"Class remapping: {hf_to_correct}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FONTS
# ═════════════════════════════════════════════════════════════════════════════

try:
    font_large  = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 36)
    font_normal = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 28)
    print("Fonts loaded: LiberationSans 36pt / 28pt")
except:
    font_large  = ImageFont.load_default()
    font_normal = ImageFont.load_default()
    print("Warning: LiberationSans not found — using default font")

# ═════════════════════════════════════════════════════════════════════════════
# 8. DATASET NAME EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

def get_dataset_name(filename):
    fname = os.path.basename(filename)
    f = fname.lower()
    if "stir" in f:
        return "Soil-Moisture-Stir-September"
    elif "september" in f:
        return "Soil-Moisture-September"
    elif "5sagf" in f:
        return "Soil-Moisture-5sagf"
    elif "v4-uv" in f:
        return "Soil-Moisture-v4-UV"
    elif "v4-ir" in f:
        return "Soil-Moisture-v4-IR"
    elif "v4-" in f or "v4_" in f:
        return "Soil-Moisture-v4"
    elif "soil-moisture-ir" in f or "moisture-ir" in f:
        return "Soil-Moisture-IR"
    else:
        return "Soil-Moisture-v4"

# ═════════════════════════════════════════════════════════════════════════════
# 9. ANNOTATE FUNCTION — exact same panel spec
# ═════════════════════════════════════════════════════════════════════════════

def annotate_image(img, dataset_name, img_id, pred_label,
                   true_label, inference_ms):
    img   = img.convert("RGB")
    img   = img.resize((640, 480))
    panel = Image.new("RGB", (640, 750), (0, 0, 0))
    panel.paste(img, (0, 200))
    draw  = ImageDraw.Draw(panel)

    line_height = font_large.size + 8
    y = 25

    draw.text((10, y), f"Dataset: {dataset_name}",
              fill=(255, 255, 255), font=font_large)
    y += line_height

    img_id_display = img_id[:20] + "..." if len(img_id) > 20 else img_id
    draw.text((10, y), f"Image ID: {img_id_display}",
              fill=(255, 255, 255), font=font_large)
    y += line_height

    pred_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.text((10, y), f"Predicted: {pred_label}",
              fill=pred_color, font=font_large)
    y += line_height

    draw.text((10, y), f"Ground Truth: {true_label}",
              fill=(255, 255, 0), font=font_large)
    y += line_height + 30

    draw.text((10, y), f"Inference: {inference_ms:.1f} ms",
              fill=(0, 200, 255), font=font_large)

    result_text  = "CORRECT" if pred_label == true_label else "INCORRECT"
    result_color = (0, 255, 0) if pred_label == true_label else (255, 0, 0)
    draw.rectangle([0, 690, 640, 750], fill=(0, 0, 0))
    draw.text((10, 698), result_text,
              fill=result_color, font=font_large)

    return panel

# ═════════════════════════════════════════════════════════════════════════════
# 10. COLLECT ALL TEST IMAGES GROUPED BY DATASET
# ═════════════════════════════════════════════════════════════════════════════

print("\nCollecting test images by dataset...")

dataset_images = {ds: [] for ds in SAMPLES_PER_DATASET}

for class_folder in sorted(os.listdir(os.path.join(DATA_DIR, "test"))):
    class_path = os.path.join(DATA_DIR, "test", class_folder)
    if not os.path.isdir(class_path):
        continue
    hf_idx     = train_folders.index(class_folder)
    true_class = hf_to_correct[hf_idx]

    for img_file in os.listdir(class_path):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(class_path, img_file)
        ds_name  = get_dataset_name(img_file)
        if ds_name in dataset_images:
            dataset_images[ds_name].append((img_path, true_class))

for ds, imgs in dataset_images.items():
    print(f"  {ds}: {len(imgs)} images")

# ═════════════════════════════════════════════════════════════════════════════
# 11. RUN INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

print("\nRunning inference...")

all_saved       = []
sample_counter  = 1
correct_count   = 0
total_count     = 0
inference_times = []
dataset_results = {}

for ds_name, count in SAMPLES_PER_DATASET.items():
    imgs     = dataset_images.get(ds_name, [])
    selected = random.sample(imgs, min(count, len(imgs)))

    ds_correct = 0
    ds_total   = 0

    for img_path, true_class in selected:
        img_pil = Image.open(img_path).convert("RGB")
        tensor  = build_7channel_tensor(img_pil, val_transform).unsqueeze(0).to(device)

        with torch.no_grad():
            t0      = time.time()
            output  = model(tensor)
            elapsed = (time.time() - t0) * 1000

        inference_times.append(elapsed)
        pred_hf    = output.argmax(dim=1).item()
        pred_class = hf_to_correct[pred_hf]
        true_label = f"Level {true_class}"
        pred_label = f"Level {pred_class}"
        is_correct = pred_class == true_class

        if is_correct:
            correct_count += 1
            ds_correct    += 1
        total_count += 1
        ds_total    += 1

        img_id    = os.path.basename(img_path).rsplit(".", 1)[0]
        annotated = annotate_image(img_pil, ds_name, img_id,
                                   pred_label, true_label, elapsed)

        save_name = (f"sample_{sample_counter:03d}"
                     f"_{ds_name.replace(' ', '_')}"
                     f"_{'correct' if is_correct else 'incorrect'}.jpg")
        save_path = os.path.join(OUTPUT_DIR, save_name)
        annotated.save(save_path)
        all_saved.append(save_path)

        print(f"Sample {sample_counter:03d} | {ds_name} | "
              f"Pred: {pred_label} | Truth: {true_label} | "
              f"{'✓' if is_correct else '✗'} | {elapsed:.1f}ms")
        sample_counter += 1

    ds_acc = ds_correct / ds_total * 100 if ds_total > 0 else 0
    dataset_results[ds_name] = {
        "correct" : ds_correct,
        "total"   : ds_total,
        "accuracy": round(ds_acc, 2)
    }

# ═════════════════════════════════════════════════════════════════════════════
# 12. SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

overall_acc      = correct_count / total_count * 100
avg_inference_ms = np.mean(inference_times)

print("\n" + "=" * 60)
print("  INFERENCE SUMMARY — PER DATASET")
print("=" * 60)
for ds, res in dataset_results.items():
    print(f"  {ds:<35} {res['correct']}/{res['total']} "
          f"({res['accuracy']}%)")
print("-" * 60)
print(f"  {'OVERALL':<35} {correct_count}/{total_count} "
      f"({overall_acc:.2f}%)")
print(f"  Avg inference latency : {avg_inference_ms:.2f} ms/image")
print("=" * 60)

summary = {
    "model"            : "MambaVision_S_FFTWavelet_FullImage",
    "total_samples"    : total_count,
    "correct"          : correct_count,
    "overall_accuracy" : round(overall_acc, 2),
    "avg_latency_ms"   : round(avg_inference_ms, 2),
    "per_dataset"      : dataset_results,
    "images_saved"     : len(all_saved),
    "output_dir"       : OUTPUT_DIR,
}

summary_path = os.path.join(RESULTS_DIR,
                            "inference_summary_fft_wavelet_fullimage.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved → {summary_path}")
print(f"Images saved  → {OUTPUT_DIR}")
print("\nDone! Pipeline complete.")