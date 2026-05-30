 """
03_baseline_vit_comparison.py
==============================
Packages the ViT baseline results from Kaggle (soil_moisture_level2.ipynb)
into a structured JSON file for use in 06_evaluation.py.

No training happens here. These are the exact values from Phase 2 of the
Kaggle ViT experiment (google/vit-base-patch16-224-in21k, 25 epochs).

Run:
    python 03_baseline_vit_comparison.py

Output:
    results/vit_baseline.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. TRAINING METRICS (exact values from Kaggle notebook Step 10)
# ═════════════════════════════════════════════════════════════════════════════

train_losses = [
    4.673071, 4.453415, 4.095893, 3.647770, 3.428395,
    3.116595, 2.688093, 2.507006, 2.278150, 2.079329,
    1.984986, 1.814550, 1.846718, 1.590630, 1.656840,
    1.560463, 1.605761, 1.555979, 1.414505, 1.446769,
    1.404795, 1.368590, 1.344135, 1.435816, 1.418155
]

val_losses = [
    4.751688, 4.460595, 4.137558, 3.761130, 3.428313,
    3.167595, 2.809613, 2.599695, 2.414394, 2.290302,
    2.129147, 1.914390, 1.913347, 1.790328, 1.751759,
    1.676690, 1.670827, 1.611479, 1.599333, 1.581134,
    1.574428, 1.565577, 1.563511, 1.564832, 1.564560
]

val_accuracies = [
    0.133005, 0.339901, 0.389163, 0.453202, 0.556650,
    0.748768, 0.837438, 0.857143, 0.876847, 0.862069,
    0.866995, 0.940887, 0.886700, 0.896552, 0.921182,
    0.945813, 0.926108, 0.945813, 0.945813, 0.945813,
    0.945813, 0.945813, 0.945813, 0.945813, 0.945813
]

epochs = list(range(1, 26))

# ═════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONFIG (from Kaggle notebook Step 9)
# ═════════════════════════════════════════════════════════════════════════════

vit_config = {
    "model_name": "google/vit-base-patch16-224-in21k",
    "num_labels": 11,
    "patch_size": 16,
    "image_size": 224,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "num_epochs": 25,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 100,
    "lr_scheduler": "cosine",
    "label_smoothing": 0.1,
    "platform": "Kaggle (T4 GPU)",
    "notes": "Phase 2 — with augmentation. No timing data recorded on Kaggle."
}

# ═════════════════════════════════════════════════════════════════════════════
# 3. SUMMARY METRICS
# ═════════════════════════════════════════════════════════════════════════════

best_epoch = val_accuracies.index(max(val_accuracies)) + 1
best_val_acc = max(val_accuracies)
final_val_acc = val_accuracies[-1]
plateau_start_epoch = 16

summary = {
    "best_val_accuracy": round(best_val_acc, 6),
    "best_val_accuracy_pct": round(best_val_acc * 100, 2),
    "final_val_accuracy": round(final_val_acc, 6),
    "plateau_start_epoch": plateau_start_epoch,
    "best_train_loss": round(min(train_losses), 6),
    "best_val_loss": round(min(val_losses), 6),
    "final_train_loss": round(train_losses[-1], 6),
    "final_val_loss": round(val_losses[-1], 6),
    "avg_epoch_time_seconds": None,
    "inference_latency_ms_per_image": None,
    "peak_gpu_memory_gb": None,
    "model_parameters_millions": 86.0,
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATA SPLIT (from 02_data_preparation.py output)
# ═════════════════════════════════════════════════════════════════════════════

data_split = {
    "train_images": 717,
    "val_images": 203,
    "test_images": 106,
    "num_classes": 11,
    "class_labels": [f"Level {i}" for i in range(11)],
    "laser_crops_total": 1026,
    "augmentation": [
        "RandomHorizontalFlip(p=0.5)",
        "RandomVerticalFlip(p=0.5)",
        "RandomRotation(degrees=15)",
        "ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)",
        "RandomResizedCrop(224, scale=(0.7,1.0))",
        "GaussianBlur(kernel_size=3)",
        "RandomAdjustSharpness(sharpness_factor=2, p=0.3)"
    ]
}

# ═════════════════════════════════════════════════════════════════════════════
# 5. ASSEMBLE AND SAVE BASELINE JSON
# ═════════════════════════════════════════════════════════════════════════════

baseline = {
    "model": "ViT",
    "config": vit_config,
    "data": data_split,
    "training_curves": {
        "epochs": epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    },
    "summary": summary,
}

output_path = os.path.join(RESULTS_DIR, "vit_baseline.json")
with open(output_path, "w") as f:
    json.dump(baseline, f, indent=2)

print(f"ViT baseline saved → {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. PLOT TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ViT Baseline — Kaggle Phase 2 (google/vit-base-patch16-224-in21k)",
             fontsize=13, fontweight="bold")

axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3,
             color="#4C72B0", linewidth=2)
axes[0].plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3,
             color="#C0392B", linewidth=2)
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
axes[1].axhline(y=94.58, color="gray", linestyle="--", linewidth=1.5,
                label="Plateau: 94.58%")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation Accuracy")
axes[1].set_ylim([10, 100])
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "vit_baseline_curves.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Training curves saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. PRINT SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  ViT BASELINE SUMMARY")
print("=" * 55)
print(f"  Model          : {vit_config['model_name']}")
print(f"  Parameters     : ~{summary['model_parameters_millions']}M")
print(f"  Epochs trained : {vit_config['num_epochs']}")
print(f"  Best Val Acc   : {summary['best_val_accuracy_pct']}%  (epoch {plateau_start_epoch}+)")
print(f"  Final Val Loss : {summary['final_val_loss']}")
print(f"  Train images   : {data_split['train_images']}")
print(f"  Val images     : {data_split['val_images']}")
print(f"  Test images    : {data_split['test_images']}")
print(f"  Epoch timing   : Not recorded on Kaggle")
print(f"  Inference ms   : Not recorded on Kaggle")
print("=" * 55)
print("\nMambaVision will be benchmarked against this baseline in 06_evaluation.py")
print("Speed metrics will be captured during MambaVision training in 05_training.py")
