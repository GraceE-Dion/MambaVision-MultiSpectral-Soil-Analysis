"""
06b_evaluation.py
=================
Loads ViT baseline, MambaVision_S (laser crops), and MambaVision_S (full image)
training results and produces the final three-way comparison tables and figures
for the research paper.

Run:
    python 06b_evaluation.py

Output:
    results/comparison_table_fullimage.json
    results/accuracy_comparison_fullimage.png
    results/efficiency_comparison_fullimage.png
    results/mambavision_fullimage_final_curves.png
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD RESULTS
# ═════════════════════════════════════════════════════════════════════════════

print("Loading results...")

with open(os.path.join(RESULTS_DIR, "vit_baseline.json"), "r") as f:
    vit = json.load(f)

with open(os.path.join(RESULTS_DIR, "mambavision_training_results.json"), "r") as f:
    mamba_crop = json.load(f)

with open(os.path.join(RESULTS_DIR, "mambavision_fullimage_training_results.json"), "r") as f:
    mamba_full = json.load(f)

print("ViT baseline loaded!")
print("MambaVision_S (laser crops) results loaded!")
print("MambaVision_S (full image) results loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 2. COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

comparison = {
    "ViT-Base": {
        "model"                    : "google/vit-base-patch16-224-in21k",
        "parameters_millions"      : 86.0,
        "input_type"               : "Full image",
        "best_val_accuracy_pct"    : vit["summary"]["best_val_accuracy_pct"],
        "test_accuracy_pct"        : "N/A (not recorded on Kaggle)",
        "avg_epoch_time_seconds"   : "N/A (not recorded on Kaggle)",
        "inference_latency_ms"     : "N/A (not recorded on Kaggle)",
        "peak_gpu_memory_gb"       : "N/A (not recorded on Kaggle)",
        "num_epochs"               : vit["config"]["num_epochs"],
        "training_images"          : 2151,
        "platform"                 : "Kaggle T4 GPU",
    },
    "MambaVision_S_LaserCrops": {
        "model"                    : "mamba_vision_S (NVlabs fork)",
        "parameters_millions"      : 50.0,
        "input_type"               : "Laser crops",
        "best_val_accuracy_pct"    : mamba_crop["best_val_accuracy_pct"],
        "test_accuracy_pct"        : mamba_crop["test_accuracy_pct"],
        "avg_epoch_time_seconds"   : mamba_crop["avg_epoch_time_seconds"],
        "inference_latency_ms"     : mamba_crop["inference_latency_ms"],
        "peak_gpu_memory_gb"       : mamba_crop["peak_gpu_memory_gb"],
        "num_epochs"               : mamba_crop["num_epochs"],
        "training_images"          : 2151,
        "platform"                 : "MTSU Lambda RTX 3090",
    },
    "MambaVision_S_FullImage": {
        "model"                    : "mamba_vision_S (NVlabs fork)",
        "parameters_millions"      : 50.0,
        "input_type"               : "Full image",
        "best_val_accuracy_pct"    : mamba_full["best_val_accuracy_pct"],
        "best_epoch"               : mamba_full["best_epoch"],
        "test_accuracy_pct"        : mamba_full["test_accuracy_pct"],
        "avg_epoch_time_seconds"   : mamba_full["avg_epoch_time_seconds"],
        "inference_latency_ms"     : mamba_full["inference_latency_ms"],
        "peak_gpu_memory_gb"       : mamba_full["peak_gpu_memory_gb"],
        "num_epochs"               : mamba_full["num_epochs"],
        "training_images"          : 2151,
        "platform"                 : "MTSU Lambda RTX 3090",
    }
}

output_path = os.path.join(RESULTS_DIR, "comparison_table_fullimage.json")
with open(output_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nComparison table saved → {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PRINT COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("  MAMBAVISION_S (FULL IMAGE) vs MAMBAVISION_S (LASER CROPS) vs ViT-BASE")
print("=" * 75)
print(f"  {'Metric':<30} {'ViT-Base':>14} {'Mamba-Crop':>14} {'Mamba-Full':>14}")
print("-" * 75)
print(f"  {'Input Type':<30} {'Full image':>14} {'Laser crops':>14} {'Full image':>14}")
print(f"  {'Parameters (M)':<30} {'86.0M':>14} {'~50M':>14} {'~50M':>14}")
print(f"  {'Best Val Accuracy':<30} {'94.58%':>14} {str(mamba_crop['best_val_accuracy_pct'])+'%':>14} {str(mamba_full['best_val_accuracy_pct'])+'%':>14}")
print(f"  {'Test Accuracy':<30} {'N/A':>14} {str(mamba_crop['test_accuracy_pct'])+'%':>14} {str(mamba_full['test_accuracy_pct'])+'%':>14}")
print(f"  {'Convergence Epoch':<30} {'N/A':>14} {'80':>14} {str(mamba_full['best_epoch']):>14}")
print(f"  {'Epochs Trained':<30} {str(vit['config']['num_epochs']):>14} {str(mamba_crop['num_epochs']):>14} {str(mamba_full['num_epochs']):>14}")
print(f"  {'Training Images':<30} {'2,151':>14} {'2,151':>14} {'2,151':>14}")
print(f"  {'Avg Epoch Time':<30} {'N/A (Kaggle)':>14} {str(mamba_crop['avg_epoch_time_seconds'])+'s':>14} {str(mamba_full['avg_epoch_time_seconds'])+'s':>14}")
print(f"  {'Inference Latency':<30} {'N/A (Kaggle)':>14} {str(mamba_crop['inference_latency_ms'])+'ms':>14} {str(mamba_full['inference_latency_ms'])+'ms':>14}")
print(f"  {'Peak GPU Memory':<30} {'N/A (Kaggle)':>14} {str(mamba_crop['peak_gpu_memory_gb'])+'GB':>14} {str(mamba_full['peak_gpu_memory_gb'])+'GB':>14}")
print(f"  {'Platform':<30} {'Kaggle T4':>14} {'RTX 3090':>14} {'RTX 3090':>14}")
print("=" * 75)

# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — ACCURACY CURVES COMPARISON (THREE-WAY)
# ═════════════════════════════════════════════════════════════════════════════

print("\nGenerating accuracy comparison figure...")

vit_epochs        = vit["training_curves"]["epochs"]
vit_acc           = [a * 100 for a in vit["training_curves"]["val_accuracies"]]

mamba_crop_epochs = mamba_crop["training_curves"]["epochs"]
mamba_crop_acc    = [a * 100 for a in mamba_crop["training_curves"]["val_accuracies"]]

mamba_full_epochs = mamba_full["training_curves"]["epochs"]
mamba_full_acc    = [a * 100 for a in mamba_full["training_curves"]["val_accuracies"]]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

ax.plot(vit_epochs, vit_acc,
        label="ViT-Base (Kaggle T4, full image)",
        color="#4C72B0", linewidth=2.5, marker="o", markersize=3)
ax.plot(mamba_crop_epochs, mamba_crop_acc,
        label=f"MambaVision_S — Laser Crops (RTX 3090, {mamba_crop['num_epochs']} epochs)",
        color="#E07B39", linewidth=2.5, marker="^", markersize=3)
ax.plot(mamba_full_epochs, mamba_full_acc,
        label=f"MambaVision_S — Full Image (RTX 3090, {mamba_full['num_epochs']} epochs)",
        color="#2E7D5E", linewidth=2.5, marker="s", markersize=3)

ax.axhline(y=94.58, color="#4C72B0", linestyle="--",
           linewidth=1.5, alpha=0.5, label="ViT plateau: 94.58%")
ax.axhline(y=mamba_crop["best_val_accuracy_pct"], color="#E07B39",
           linestyle="--", linewidth=1.5, alpha=0.5,
           label=f"Mamba-Crops best: {mamba_crop['best_val_accuracy_pct']}%")
ax.axhline(y=mamba_full["best_val_accuracy_pct"], color="#2E7D5E",
           linestyle="--", linewidth=1.5, alpha=0.5,
           label=f"Mamba-Full best: {mamba_full['best_val_accuracy_pct']}%")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Validation Accuracy (%)", fontsize=13)
ax.set_title("Validation Accuracy: MambaVision_S (Full Image) vs MambaVision_S (Laser Crops) vs ViT-Base\n"
             "Multi-Spectral Soil Moisture Classification (11 Classes)",
             fontsize=13, fontweight="bold")
ax.set_ylim([0, 100])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "accuracy_comparison_fullimage.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Accuracy comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — EFFICIENCY COMPARISON BAR CHART (THREE-WAY)
# ═════════════════════════════════════════════════════════════════════════════

print("Generating efficiency comparison figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("MambaVision_S (Full Image) vs MambaVision_S (Laser Crops) vs ViT-Base — Efficiency Metrics\n"
             "(Speed metrics only available for MambaVision_S variants on RTX 3090)",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

labels  = ["ViT-Base", "Mamba-Crops", "Mamba-Full"]
colors  = ["#4C72B0", "#E07B39", "#2E7D5E"]

# Panel 1 — Parameters
axes[0].bar(labels, [86.0, 50.0, 50.0], color=colors, width=0.5)
axes[0].set_title("Model Parameters (M)", fontsize=11)
axes[0].set_ylabel("Millions", fontsize=10)
for i, v in enumerate([86.0, 50.0, 50.0]):
    axes[0].text(i, v + 0.5, f"{v}M", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylim([0, 110])
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(axis="y", alpha=0.3)

# Panel 2 — Val Accuracy
val_accs = [94.58, mamba_crop["best_val_accuracy_pct"], mamba_full["best_val_accuracy_pct"]]
axes[1].bar(labels, val_accs, color=colors, width=0.5)
axes[1].set_title("Best Val Accuracy (%)", fontsize=11)
axes[1].set_ylabel("Accuracy (%)", fontsize=10)
for i, v in enumerate(val_accs):
    axes[1].text(i, v + 0.2, f"{v}%", ha="center", fontsize=11, fontweight="bold")
axes[1].set_ylim([80, 100])
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(axis="y", alpha=0.3)

# Panel 3 — GPU Memory (ViT not recorded)
gpu_mem = [0, mamba_crop["peak_gpu_memory_gb"], mamba_full["peak_gpu_memory_gb"]]
bar_colors = ["#CCCCCC", "#E07B39", "#2E7D5E"]
axes[2].bar(["ViT-Base\n(not recorded)", "Mamba-Crops", "Mamba-Full"],
            gpu_mem, color=bar_colors, width=0.5)
axes[2].set_title("Peak GPU Memory (GB)", fontsize=11)
axes[2].set_ylabel("GB", fontsize=10)
axes[2].text(0, 0.05, "N/A", ha="center", fontsize=11,
             fontweight="bold", color="#999999")
for i, (v, lbl) in enumerate(zip(gpu_mem[1:], ["Mamba-Crops", "Mamba-Full"]), start=1):
    axes[2].text(i, v + 0.05, f"{v}GB", ha="center", fontsize=11, fontweight="bold")
axes[2].set_ylim([0, 5])
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "efficiency_comparison_fullimage.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Efficiency comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 3 — MAMBAVISION FULL IMAGE TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════════

print("Generating MambaVision full image training curves figure...")

epochs     = mamba_full["training_curves"]["epochs"]
train_loss = mamba_full["training_curves"]["train_losses"]
val_loss   = mamba_full["training_curves"]["val_losses"]
val_acc    = [a * 100 for a in mamba_full["training_curves"]["val_accuracies"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"MambaVision_S Full Image Training Curves — {mamba_full['num_epochs']} Epochs\n"
             "Multi-Spectral Soil Moisture Classification (2,151 training images)",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

axes[0].plot(epochs, train_loss, label="Train Loss",
             color="#4C72B0", linewidth=2, marker="o", markersize=2)
axes[0].plot(epochs, val_loss, label="Val Loss",
             color="#C0392B", linewidth=2, marker="s", markersize=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(epochs, val_acc, label="MambaVision_S Full Image Val Accuracy",
             color="#2E7D5E", linewidth=2, marker="o", markersize=2)
axes[1].axhline(y=94.58, color="#4C72B0", linestyle="--",
                linewidth=1.5, label="ViT-Base baseline: 94.58%")
axes[1].axhline(y=mamba_crop["best_val_accuracy_pct"], color="#E07B39",
                linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Mamba-Crops best: {mamba_crop['best_val_accuracy_pct']}%")
axes[1].axhline(y=mamba_full["best_val_accuracy_pct"], color="#2E7D5E",
                linestyle="--", linewidth=1.5, alpha=0.5,
                label=f"Mamba-Full best: {mamba_full['best_val_accuracy_pct']}%")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation Accuracy vs All Baselines")
axes[1].set_ylim([0, 100])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "mambavision_fullimage_final_curves.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"MambaVision full image training curves saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

vit_gap   = round(mamba_full["best_val_accuracy_pct"] - 94.58, 2)
crop_gap  = round(mamba_full["best_val_accuracy_pct"] - mamba_crop["best_val_accuracy_pct"], 2)

print("\n" + "=" * 65)
print("  PAPER-READY SUMMARY")
print("=" * 65)
print(f"  MambaVision_S Full Image best val accuracy : {mamba_full['best_val_accuracy_pct']}%")
print(f"  MambaVision_S Laser Crops best val accuracy: {mamba_crop['best_val_accuracy_pct']}%")
print(f"  ViT-Base best val accuracy                 : 94.58%")
print(f"  Full image gain over ViT-Base              : +{vit_gap}%")
print(f"  Full image gain over laser crops           : +{crop_gap}%")
print(f"  MambaVision_S Full Image test accuracy     : {mamba_full['test_accuracy_pct']}%")
print(f"  MambaVision_S Full Image convergence epoch : {mamba_full['best_epoch']}")
print(f"  MambaVision_S Full Image inference latency : {mamba_full['inference_latency_ms']} ms/image")
print(f"  MambaVision_S Full Image peak GPU memory   : {mamba_full['peak_gpu_memory_gb']} GB")
print(f"  MambaVision_S Full Image avg epoch time    : {mamba_full['avg_epoch_time_seconds']} s")
print(f"  Parameter reduction vs ViT                 : 86M → ~50M (42% fewer)")
print("=" * 65)
print("\nAll figures saved to ./results/")
print("Ready for 07b_inference_pipeline.py")