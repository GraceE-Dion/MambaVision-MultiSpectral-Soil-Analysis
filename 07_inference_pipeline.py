"""
07_inference_pipeline.py
========================
Runs inference on test images using the best MambaVision_S model.
Samples 50 images across 7 source datasets (same as Kaggle Step 27).
Produces annotated images in the same format as the Kaggle ViT
Step 27 notebook for direct visual comparison in the paper.

Fonts  : LiberationSans-Bold 36pt, LiberationSans-Regular 28pt
Panel  : 640x750, image pasted at (0,200), text starts at y=25
Result : bar at [0, 690, 640, 750], text at y=698

Run:
    python 07_inference_pipeline.py

Output:
    results/inference_images/   — annotated images
    results/inference_summary.json
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

# ── Load MambaVision from NVlabs fork ─────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

# ── Required for cluster ──────────────────────────────────────────────────────
torch.backends.cudnn.enabled = False

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
OUTPUT_DIR  = "./results/inference_images"
MODEL_PATH  = "./results/mambavision_best_model.pth"
DATA_DIR    = "/data/Grace/Master_Laser_Crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG — same sample counts as Kaggle Step 27
# ═════════════════════════════════════════════════════════════════════════════

NUM_CLASSES = 11
IMAGE_SIZE  = 224

SAMPLES_PER_DATASET = {
    "Soil-Moisture-v4"           : 8,
    "Soil-Moisture-v4-IR"        : 7,
    "Soil-Moisture-v4-UV"        : 7,
    "Soil-Moisture-IR"           : 7,
    "Soil-Moisture-5sagf"        : 7,
    "Soil-Moisture-September"    : 7,
    "Soil-Moisture-Stir-September": 7,
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. DEVICE
# ═════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════

print("Loading MambaVision_S best model...")
model = models.mamba_vision_S(pretrained=False)
in_features = model.head.in_features
model.head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMS
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
# 5. CLASS REMAPPING
# ═════════════════════════════════════════════════════════════════════════════

train_folders = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
hf_to_correct = {idx: int(folder)
                 for idx, folder in enumerate(train_folders)}
print(f"Class remapping: {hf_to_correct}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. FONTS — exact same as Kaggle Step 27
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
# 7. DATASET NAME EXTRACTOR
# Filenames look like: Soil-Moisture-v4-3_12_png.rf.xxx.jpg
# ═════════════════════════════════════════════════════════════════════════════

def get_dataset_name(filename):
    """Extract source dataset name from filename prefix."""
    fname = os.path.basename(filename).lower()
    if "stir" in fname:
        return "Soil-Moisture-Stir-September"
    elif "september" in fname:
        return "Soil-Moisture-September"
    elif "5sagf" in fname:
        return "Soil-Moisture-5sagf"
    elif "v4-uv" in fname or "v4-uv" in fname:
        return "Soil-Moisture-v4-UV"
    elif "v4-ir" in fname:
        return "Soil-Moisture-v4-IR"
    elif "v4-ir" in fname or "soil-moisture-ir" in fname:
        return "Soil-Moisture-IR"
    elif "v4" in fname:
        return "Soil-Moisture-v4"
    else:
        return "Unknown"

# ═════════════════════════════════════════════════════════════════════════════
# 8. ANNOTATE FUNCTION — exact same panel spec as Kaggle Step 27
# ═════════════════════════════════════════════════════════════════════════════

def annotate_image(img, dataset_name, img_id, pred_label,
                   true_label, inference_ms):
    img  = img.convert("RGB")
    img  = img.resize((640, 480))

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
# 9. COLLECT ALL TEST IMAGES GROUPED BY DATASET
# ═════════════════════════════════════════════════════════════════════════════

print("\nCollecting test images by dataset...")

# Build lookup: dataset_name -> list of (img_path, true_class)
dataset_images = {ds: [] for ds in SAMPLES_PER_DATASET}

for class_folder in sorted(os.listdir(os.path.join(DATA_DIR, "test"))):
    class_path = os.path.join(DATA_DIR, "test", class_folder)
    if not os.path.isdir(class_path):
        continue
    # hf index for this folder
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
# 10. RUN INFERENCE — sample per dataset
# ═════════════════════════════════════════════════════════════════════════════

print("\nRunning inference...")

all_saved        = []
sample_counter   = 1
correct_count    = 0
total_count      = 0
inference_times  = []
dataset_results  = {}

for ds_name, count in SAMPLES_PER_DATASET.items():
    imgs     = dataset_images.get(ds_name, [])
    selected = random.sample(imgs, min(count, len(imgs)))

    ds_correct = 0
    ds_total   = 0

    for img_path, true_class in selected:
        img_pil = Image.open(img_path).convert("RGB")
        tensor  = val_transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            t0      = time.time()
            output  = model(tensor)
            elapsed = (time.time() - t0) * 1000

        inference_times.append(elapsed)
        pred_class = output.argmax(dim=1).item()
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
# 11. SUMMARY
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
    "model"              : "MambaVision_S",
    "total_samples"      : total_count,
    "correct"            : correct_count,
    "overall_accuracy"   : round(overall_acc, 2),
    "avg_latency_ms"     : round(avg_inference_ms, 2),
    "per_dataset"        : dataset_results,
    "images_saved"       : len(all_saved),
    "output_dir"         : OUTPUT_DIR,
}

summary_path = os.path.join(RESULTS_DIR, "inference_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved → {summary_path}")
print(f"Images saved  → {OUTPUT_DIR}")
print("\nDone! Pipeline complete.")