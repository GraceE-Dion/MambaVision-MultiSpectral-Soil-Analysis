"""
07_inference_pipeline.py
========================
Runs inference on test images using the best MambaVision_S model.
Produces annotated images in the same format as the Kaggle ViT
Step 27 notebook for direct visual comparison in the paper.

Fonts  : LiberationSans-Bold 36pt, LiberationSans-Regular 28pt
Panel  : 640x750, image pasted at (0, 200), text starts at y=25
Result : bar at [0, 690, 640, 750], text at y=698

Run:
    python 07_inference_pipeline.py

Output:
    results/inference_images/   — annotated images per dataset
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
from torchvision import transforms, datasets

# ── Load MambaVision from NVlabs fork ─────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

# ── Required for cluster ──────────────────────────────────────────────────────
torch.backends.cudnn.enabled = False

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = "./results"
OUTPUT_DIR    = "./results/inference_images"
MODEL_PATH    = "./results/mambavision_best_model.pth"
DATA_DIR      = "/data/Grace/Master_Laser_Crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NUM_CLASSES  = 11
IMAGE_SIZE   = 224
SAMPLES_PER_DATASET = 7

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
# 5. CLASS REMAPPING (same hf_to_correct fix)
# ═════════════════════════════════════════════════════════════════════════════

train_folders = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
hf_to_correct = {idx: int(folder) for idx, folder in enumerate(train_folders)}
correct_to_hf = {v: k for k, v in hf_to_correct.items()}
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
# 7. ANNOTATE FUNCTION — exact same panel spec as Kaggle Step 27
# Panel: 640x750, image at (0,200), text y=25, result bar [0,690,640,750]
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
# 8. RUN INFERENCE ON TEST SET
# ═════════════════════════════════════════════════════════════════════════════

print("\nRunning inference on test set...")

test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=val_transform
)
test_dataset.targets = [hf_to_correct[t] for t in test_dataset.targets]

# Group test images by class for sampling
class_to_samples = {i: [] for i in range(NUM_CLASSES)}
for idx, (img_path, hf_label) in enumerate(test_dataset.samples):
    correct_label = hf_to_correct[hf_label]
    class_to_samples[correct_label].append((idx, img_path, correct_label))

all_saved      = []
sample_counter = 1
correct_count  = 0
total_count    = 0
inference_times = []

# Sample from each class
for class_id in range(NUM_CLASSES):
    samples = class_to_samples[class_id]
    selected = random.sample(samples, min(SAMPLES_PER_DATASET, len(samples)))

    for idx, img_path, true_class in selected:
        # Load and preprocess
        img_pil = Image.open(img_path).convert("RGB")
        tensor  = val_transform(img_pil).unsqueeze(0).to(device)

        # Inference
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
        total_count += 1

        # Get dataset name from path
        parts       = img_path.split(os.sep)
        dataset_name = parts[-1].split("_")[0] + "_" + parts[-1].split("_")[1] \
                       if "_" in parts[-1] else parts[-1]
        img_id      = os.path.basename(img_path).rsplit(".", 1)[0]

        # Annotate
        annotated  = annotate_image(img_pil, f"Level {true_class}",
                                    img_id, pred_label,
                                    true_label, elapsed)
        save_name  = f"sample_{sample_counter:03d}_class{true_class}_{is_correct}.jpg"
        save_path  = os.path.join(OUTPUT_DIR, save_name)
        annotated.save(save_path)
        all_saved.append(save_path)

        print(f"Sample {sample_counter:03d} | Class {true_class} | "
              f"Pred: {pred_label} | Truth: {true_label} | "
              f"{'✓' if is_correct else '✗'} | {elapsed:.1f}ms")
        sample_counter += 1

# ═════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

inference_acc    = correct_count / total_count * 100
avg_inference_ms = np.mean(inference_times)

print(f"\nInference complete!")
print(f"Correct       : {correct_count}/{total_count}")
print(f"Accuracy      : {inference_acc:.2f}%")
print(f"Avg latency   : {avg_inference_ms:.2f} ms/image")
print(f"Images saved  : {len(all_saved)} → {OUTPUT_DIR}")

summary = {
    "model"              : "MambaVision_S",
    "total_samples"      : total_count,
    "correct"            : correct_count,
    "inference_accuracy" : round(inference_acc, 2),
    "avg_latency_ms"     : round(avg_inference_ms, 2),
    "images_saved"       : len(all_saved),
    "output_dir"         : OUTPUT_DIR,
}

summary_path = os.path.join(RESULTS_DIR, "inference_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved → {summary_path}")
print("\nDone! Pipeline complete.") 
