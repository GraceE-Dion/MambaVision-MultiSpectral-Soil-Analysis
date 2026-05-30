"""
06_evaluation.py
================
Loads ViT baseline and MambaVision_S training results and produces
the final comparison tables and figures for the research paper.

Run:
    python 06_evaluation.py

Output:
    results/comparison_table.json
    results/comparison_table.png
    results/accuracy_comparison.png
    results/efficiency_comparison.png
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
    mamba = json.load(f)

print("ViT baseline loaded!")
print("MambaVision results loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 2. COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

comparison = {
    "ViT-Base": {
        "model"                    : "google/vit-base-patch16-224-in21k",
        "parameters_millions"      : 86.0,
        "best_val_accuracy_pct"    : vit["summary"]["best_val_accuracy_pct"],
        "test_accuracy_pct"        : "N/A (not recorded on Kaggle)",
        "avg_epoch_time_seconds"   : "N/A (not recorded on Kaggle)",
        "inference_latency_ms"     : "N/A (not recorded on Kaggle)",
        "peak_gpu_memory_gb"       : "N/A (not recorded on Kaggle)",
        "num_epochs"               : vit["config"]["num_epochs"],
        "training_images"          : 2151,
        "platform"                 : "Kaggle T4 GPU",
    },
    "MambaVision_S": {
        "model"                    : "mamba_vision_S (NVlabs fork)",
        "parameters_millions"      : 50.0,
        "best_val_accuracy_pct"    : mamba["best_val_accuracy_pct"],
        "test_accuracy_pct"        : mamba["test_accuracy_pct"],
        "avg_epoch_time_seconds"   : mamba["avg_epoch_time_seconds"],
        "inference_latency_ms"     : mamba["inference_latency_ms"],
        "peak_gpu_memory_gb"       : mamba["peak_gpu_memory_gb"],
        "num_epochs"               : mamba["num_epochs"],
        "training_images"          : 2151,
        "platform"                 : "MTSU Lambda RTX 3090",
    }
}

output_path = os.path.join(RESULTS_DIR, "comparison_table.json")
with open(output_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nComparison table saved → {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PRINT COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  MAMBAVISION_S vs ViT-BASE — FINAL COMPARISON")
print("=" * 65)
print(f"  {'Metric':<30} {'ViT-Base':>15} {'MambaVision_S':>15}")
print("-" * 65)
print(f"  {'Parameters (M)':<30} {'86.0M':>15} {'~50M':>15}")
print(f"  {'Best Val Accuracy':<30} {'94.58%':>15} {mamba['best_val_accuracy_pct']:>14}%")
print(f"  {'Test Accuracy':<30} {'N/A':>15} {mamba['test_accuracy_pct']:>14}%")
print(f"  {'Epochs Trained':<30} {'25':>15} {mamba['num_epochs']:>15}")
print(f"  {'Training Images':<30} {'2,151':>15} {'2,151':>15}")
print(f"  {'Avg Epoch Time':<30} {'N/A (Kaggle)':>15} {str(mamba['avg_epoch_time_seconds'])+'s':>15}")
print(f"  {'Inference Latency':<30} {'N/A (Kaggle)':>15} {str(mamba['inference_latency_ms'])+'ms':>15}")
print(f"  {'Peak GPU Memory':<30} {'N/A (Kaggle)':>15} {str(mamba['peak_gpu_memory_gb'])+'GB':>15}")
print(f"  {'Platform':<30} {'Kaggle T4':>15} {'RTX 3090':>15}")
print("=" * 65)

# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — ACCURACY CURVES COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

print("\nGenerating accuracy comparison figure...")

vit_epochs   = vit["training_curves"]["epochs"]
vit_acc      = [a * 100 for a in vit["training_curves"]["val_accuracies"]]
mamba_epochs = mamba["training_curves"]["epochs"]
mamba_acc    = [a * 100 for a in mamba["training_curves"]["val_accuracies"]]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

ax.plot(vit_epochs, vit_acc,
        label="ViT-Base (Kaggle T4, 25 epochs, 2151 images)",
        color="#4C72B0", linewidth=2.5, marker="o", markersize=3)
ax.plot(mamba_epochs, mamba_acc,
        label=f"MambaVision_S (RTX 3090, {mamba['num_epochs']} epochs, 2151 images)",
        color="#2E7D5E", linewidth=2.5, marker="s", markersize=3)

ax.axhline(y=94.58, color="#4C72B0", linestyle="--",
           linewidth=1.5, alpha=0.5, label="ViT plateau: 94.58%")
ax.axhline(y=mamba["best_val_accuracy_pct"], color="#2E7D5E",
           linestyle="--", linewidth=1.5, alpha=0.5,
           label=f"MambaVision_S best: {mamba['best_val_accuracy_pct']}%")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Validation Accuracy (%)", fontsize=13)
ax.set_title("Validation Accuracy: MambaVision_S vs ViT-Base\n"
             "Multi-Spectral Soil Moisture Classification (11 Classes)",
             fontsize=13, fontweight="bold")
ax.set_ylim([0, 100])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Accuracy comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — EFFICIENCY COMPARISON BAR CHART
# ═════════════════════════════════════════════════════════════════════════════

print("Generating efficiency comparison figure...")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("MambaVision_S vs ViT-Base — Efficiency Metrics\n"
             "(Speed metrics only available for MambaVision_S on RTX 3090)",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

# Panel 1 — Parameters
axes[0].bar(["ViT-Base", "MambaVision_S"], [86.0, 50.0],
            color=["#4C72B0", "#2E7D5E"], width=0.5)
axes[0].set_title("Model Parameters (M)", fontsize=11)
axes[0].set_ylabel("Millions", fontsize=10)
for i, v in enumerate([86.0, 50.0]):
    axes[0].text(i, v + 0.5, f"{v}M", ha="center", fontsize=11,
                 fontweight="bold")
axes[0].set_ylim([0, 110])
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(axis="y", alpha=0.3)

# Panel 2 — Val Accuracy
axes[1].bar(["ViT-Base", "MambaVision_S"],
            [94.58, mamba["best_val_accuracy_pct"]],
            color=["#4C72B0", "#2E7D5E"], width=0.5)
axes[1].set_title("Best Val Accuracy (%)", fontsize=11)
axes[1].set_ylabel("Accuracy (%)", fontsize=10)
for i, v in enumerate([94.58, mamba["best_val_accuracy_pct"]]):
    axes[1].text(i, v + 0.3, f"{v}%", ha="center", fontsize=11,
                 fontweight="bold")
axes[1].set_ylim([80, 100])
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(axis="y", alpha=0.3)

# Panel 3 — GPU Memory (MambaVision only — ViT not recorded)
axes[2].bar(["ViT-Base\n(not recorded)", "MambaVision_S"],
            [0, mamba["peak_gpu_memory_gb"]],
            color=["#CCCCCC", "#2E7D5E"], width=0.5)
axes[2].set_title("Peak GPU Memory (GB)", fontsize=11)
axes[2].set_ylabel("GB", fontsize=10)
axes[2].text(0, 0.05, "N/A", ha="center", fontsize=11,
             fontweight="bold", color="#999999")
axes[2].text(1, mamba["peak_gpu_memory_gb"] + 0.05,
             f"{mamba['peak_gpu_memory_gb']}GB",
             ha="center", fontsize=11, fontweight="bold")
axes[2].set_ylim([0, 5])
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "efficiency_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Efficiency comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 3 — MAMBAVISION TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════════

print("Generating MambaVision training curves figure...")

epochs     = mamba["training_curves"]["epochs"]
train_loss = mamba["training_curves"]["train_losses"]
val_loss   = mamba["training_curves"]["val_losses"]
val_acc    = [a * 100 for a in mamba["training_curves"]["val_accuracies"]]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"MambaVision_S Training Curves — {mamba['num_epochs']} Epochs\n"
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

axes[1].plot(epochs, val_acc, label="MambaVision_S Val Accuracy",
             color="#2E7D5E", linewidth=2, marker="o", markersize=2)
axes[1].axhline(y=94.58, color="#4C72B0", linestyle="--",
                linewidth=1.5, label="ViT-Base baseline: 94.58%")
axes[1].axhline(y=mamba["best_val_accuracy_pct"], color="#2E7D5E",
                linestyle="--", linewidth=1.5, alpha=0.5,
                label=f"MambaVision_S best: {mamba['best_val_accuracy_pct']}%")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Validation Accuracy vs ViT Baseline")
axes[1].set_ylim([0, 100])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "mambavision_final_curves.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"MambaVision training curves saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PAPER-READY SUMMARY")
print("=" * 65)
print(f"  MambaVision_S best val accuracy : {mamba['best_val_accuracy_pct']}%")
print(f"  ViT-Base best val accuracy      : 94.58%")
print(f"  Accuracy gap                    : {round(94.58 - mamba['best_val_accuracy_pct'], 2)}%")
print(f"  MambaVision_S test accuracy     : {mamba['test_accuracy_pct']}%")
print(f"  MambaVision_S inference latency : {mamba['inference_latency_ms']} ms/image")
print(f"  MambaVision_S peak GPU memory   : {mamba['peak_gpu_memory_gb']} GB")
print(f"  MambaVision_S avg epoch time    : {mamba['avg_epoch_time_seconds']} s")
print(f"  Parameter reduction             : 86M → ~50M (42% fewer)")
print("=" * 65)
print("\nAll figures saved to ./results/")
print("Ready for 07_inference_pipeline.py") 
