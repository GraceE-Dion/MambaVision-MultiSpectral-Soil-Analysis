"""
03b_vit_cluster_benchmark.py
=============================
Trains ViT-Base Phase 2 on the MTSU Lambda cluster (RTX 3090) using the
exact same configuration as the Kaggle notebook, then runs inference speed
benchmark for hardware-normalized comparison with MambaVision_S.

Run:
    python 03b_vit_cluster_benchmark.py

Output:
    results/vit_cluster_training_results.json
    results/vit_cluster_benchmark.json
    results/vit_cluster_training_curves.png
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import evaluate
from PIL import Image as PILImage
from torchvision import transforms
from datasets import load_dataset
from datasets import Image as HFImage
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
MASTER_DIR  = "/data/Grace/Master_Soil_Moisture"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═════════════════════════════════════════════════════════════════════════════
# 1. CLASS REMAPPING — same as Kaggle notebook Step 4B
# ═════════════════════════════════════════════════════════════════════════════

folders       = sorted(os.listdir(os.path.join(MASTER_DIR, "train")))
hf_to_correct = {idx: int(folder) for idx, folder in enumerate(folders)}
print(f"\nClass remapping: {hf_to_correct}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. LOAD DATASET — same as Kaggle notebook Step 6
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading dataset...")
raw_ds = load_dataset("imagefolder", data_dir=MASTER_DIR, drop_labels=False)
raw_ds = raw_ds.cast_column("image", HFImage(decode=True))

def remap_label(example):
    example["label"] = hf_to_correct[example["label"]]
    return example

raw_ds = raw_ds.map(remap_label)
print(f"Train      : {len(raw_ds['train'])} images")
print(f"Validation : {len(raw_ds['validation'])} images")
print(f"Test       : {len(raw_ds['test'])} images")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PROCESSOR AND TRANSFORMS — same as Kaggle notebook Steps 7 and 8
# ═════════════════════════════════════════════════════════════════════════════

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
print("Processor loaded!")

train_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3,
        saturation=0.2, hue=0.1
    ),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
])

def transform_train(example_batch):
    augmented = [
        train_augmentation(img.convert("RGB"))
        for img in example_batch["image"]
    ]
    inputs = processor(images=augmented, return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs

def transform_val(example_batch):
    inputs = processor(
        images=[img.convert("RGB") for img in example_batch["image"]],
        return_tensors="pt"
    )
    inputs["labels"] = example_batch["label"]
    return inputs

prepared_train = raw_ds["train"].with_transform(transform_train)
prepared_val   = raw_ds["validation"].with_transform(transform_val)
prepared_test  = raw_ds["test"].with_transform(transform_val)
print("Transforms applied!")

# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL — same as Kaggle notebook Step 9
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading ViT-Base model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=11,
    id2label={i: f"Level {i}" for i in range(11)},
    label2id={f"Level {i}": i for i in range(11)},
    ignore_mismatched_sizes=True,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
print("Model loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAINING — same as Kaggle notebook Step 9
# ═════════════════════════════════════════════════════════════════════════════

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir           = "./results/vit_cluster_checkpoints",
    save_total_limit     = 1,
    save_strategy        = "no",
    load_best_model_at_end = False,
    eval_strategy        = "epoch",
    logging_strategy     = "epoch",
    num_train_epochs     = 25,
    learning_rate        = 2e-5,
    weight_decay         = 0.01,
    warmup_steps         = 100,
    lr_scheduler_type    = "cosine",
    metric_for_best_model = "accuracy",
    greater_is_better    = True,
    remove_unused_columns = False,
    label_smoothing_factor = 0.1,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    dataloader_num_workers      = 4,
    fp16                        = True,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = prepared_train,
    eval_dataset    = prepared_val,
    compute_metrics = compute_metrics,
)

print(f"\nStarting ViT Phase 2 training for 25 epochs...")
print("=" * 60)

import time as time_module
train_start = time_module.time()
train_result = trainer.train()
train_time   = time_module.time() - train_start

print("=" * 60)
print(f"Training complete! Total time: {train_time/60:.1f} minutes")

# Extract training history
log_history   = trainer.state.log_history
train_losses  = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]
val_losses    = [x["eval_loss"] for x in log_history if "eval_loss" in x]
val_accs      = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]
best_val_acc  = max(val_accs) if val_accs else 0.0
best_epoch    = val_accs.index(best_val_acc) + 1

print(f"Best Val Accuracy : {best_val_acc * 100:.2f}% at epoch {best_epoch}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

print("\nEvaluating on test set...")
test_results = trainer.evaluate(prepared_test)
test_acc     = test_results.get("eval_accuracy", 0.0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# 7. INFERENCE SPEED BENCHMARK — RTX 3090
# ═════════════════════════════════════════════════════════════════════════════

print("\nRunning inference speed benchmark on RTX 3090...")

model.eval()
model.to(device)

# Collect test images
test_images = []
for split_folder in os.listdir(os.path.join(MASTER_DIR, "test")):
    class_path = os.path.join(MASTER_DIR, "test", split_folder)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path)[:10]:
        if img_file.endswith((".jpg", ".jpeg", ".png")):
            test_images.append(os.path.join(class_path, img_file))
        if len(test_images) >= 55:
            break
    if len(test_images) >= 55:
        break

print(f"Benchmark images: {len(test_images)}")

# Warmup
print("Warming up...")
for img_path in test_images[:5]:
    img    = PILImage.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)
torch.cuda.synchronize()

# Benchmark
print("Benchmarking single-image inference (batch size 1)...")
latencies = []
for img_path in test_images:
    img    = PILImage.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    torch.cuda.synchronize()
    t0 = time_module.time()
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize()
    latencies.append((time_module.time() - t0) * 1000)

mean_lat   = round(float(np.mean(latencies)), 2)
std_lat    = round(float(np.std(latencies)), 2)
median_lat = round(float(np.median(latencies)), 2)
min_lat    = round(float(np.min(latencies)), 2)
max_lat    = round(float(np.max(latencies)), 2)
p95_lat    = round(float(np.percentile(latencies, 95)), 2)

peak_mem   = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)

print(f"\n=== ViT-Base Inference Latency Benchmark (RTX 3090) ===")
print(f"Device          : {device}")
print(f"GPU             : {torch.cuda.get_device_name(0)}")
print(f"Images tested   : {len(latencies)}")
print(f"Mean latency    : {mean_lat} ms/image")
print(f"Std deviation   : {std_lat} ms")
print(f"Median latency  : {median_lat} ms/image")
print(f"Min latency     : {min_lat} ms/image")
print(f"Max latency     : {max_lat} ms/image")
print(f"P95 latency     : {p95_lat} ms/image")
print(f"Peak GPU memory : {peak_mem} GB")

print(f"\n=== Hardware-Normalized Architecture Comparison ===")
print(f"{'Model':<40} {'Latency (ms/img)':>18} {'Hardware':>15} {'Parameters':>12}")
print("-" * 88)
print(f"{'ViT-Base-patch16-224':<40} {'14.42 (Kaggle T4)':>18} {'Kaggle T4':>15} {'85.8M':>12}")
print(f"{'ViT-Base-patch16-224':<40} {str(mean_lat)+' (RTX 3090)':>18} {'MTSU RTX 3090':>15} {'85.8M':>12}")
print(f"{'MambaVision_S (laser crops)':<40} {'0.90':>18} {'MTSU RTX 3090':>15} {'~50M':>12}")
print(f"{'MambaVision_S (full image)':<40} {'0.98':>18} {'MTSU RTX 3090':>15} {'~50M':>12}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

training_results = {
    "model"                 : "ViT-Base-patch16-224",
    "platform"              : "MTSU Lambda RTX 3090",
    "num_epochs"            : 25,
    "best_val_accuracy_pct" : round(best_val_acc * 100, 2),
    "best_epoch"            : best_epoch,
    "test_accuracy_pct"     : round(test_acc * 100, 2),
    "training_time_minutes" : round(train_time / 60, 1),
    "training_curves"       : {
        "epochs"         : list(range(1, len(val_accs) + 1)),
        "train_losses"   : [round(x, 6) for x in train_losses],
        "val_losses"     : [round(x, 6) for x in val_losses],
        "val_accuracies" : [round(x, 6) for x in val_accs],
    }
}

benchmark_results = {
    "model"            : "ViT-Base-patch16-224",
    "platform"         : "MTSU Lambda RTX 3090",
    "gpu"              : torch.cuda.get_device_name(0),
    "images_tested"    : len(latencies),
    "mean_latency_ms"  : mean_lat,
    "std_latency_ms"   : std_lat,
    "median_latency_ms": median_lat,
    "min_latency_ms"   : min_lat,
    "max_latency_ms"   : max_lat,
    "p95_latency_ms"   : p95_lat,
    "peak_gpu_memory_gb": peak_mem,
    "parameters_millions": 85.8,
    "kaggle_t4_latency_ms": 14.42,
    "comparison": {
        "vit_rtx3090_vs_mamba_crop"  : round(mean_lat / 0.90, 2),
        "vit_rtx3090_vs_mamba_full"  : round(mean_lat / 0.98, 2),
        "vit_t4_vs_vit_rtx3090"      : round(14.42 / mean_lat, 2),
    }
}

with open(os.path.join(RESULTS_DIR, "vit_cluster_training_results.json"), "w") as f:
    json.dump(training_results, f, indent=2)

with open(os.path.join(RESULTS_DIR, "vit_cluster_benchmark.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"\nTraining results saved → results/vit_cluster_training_results.json")
print(f"Benchmark results saved → results/vit_cluster_benchmark.json")

# ═════════════════════════════════════════════════════════════════════════════
# 9. TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════════

epochs = list(range(1, len(val_accs) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ViT-Base Phase 2 Training Curves — MTSU Lambda RTX 3090\n"
             "Multi-Spectral Soil Moisture Classification (2,151 training images)",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

axes[0].plot(range(1, len(train_losses)+1), train_losses,
             label="Train Loss", color="#4C72B0", linewidth=2,
             marker="o", markersize=3)
axes[0].plot(epochs, val_losses,
             label="Val Loss", color="#C0392B", linewidth=2,
             marker="s", markersize=3)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(epochs, [a * 100 for a in val_accs],
             label="ViT-Base Val Accuracy (RTX 3090)",
             color="#4C72B0", linewidth=2, marker="o", markersize=3)
axes[1].axhline(y=94.58, color="#4C72B0", linestyle="--",
                linewidth=1.5, alpha=0.6, label="ViT Phase 2 baseline: 94.58% (Kaggle T4)")
axes[1].axhline(y=97.04, color="#2E7D5E", linestyle="--",
                linewidth=1.5, alpha=0.6, label="MambaVision_S Full Image: 97.04%")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation Accuracy")
axes[1].set_ylim([0, 100])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "vit_cluster_training_curves.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Training curves saved → {fig_path}")

print("\nDone! ViT cluster benchmark complete.")
print(f"Hardware-normalized latency: ViT-Base {mean_lat} ms/img vs "
      f"MambaVision_S 0.90-0.98 ms/img (both RTX 3090)")