"""
06c_evaluation.py
=================
Loads all five model results and produces the final five-way comparison
tables and figures for the research paper:
  1. ViT-Base (full image, RGB)
  2. MambaVision_S — Laser Crops (RGB)
  3. MambaVision_S — Full Image (RGB)
  4. MambaVision_S — Laser Crops + FFT + Wavelet
  5. MambaVision_S — Full Image + FFT + Wavelet

Run:
    python 06c_evaluation.py

Output:
    results/comparison_table_final.json
    results/accuracy_comparison_final.png
    results/efficiency_comparison_final.png
    results/fft_wavelet_curves_comparison.png
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

with open(os.path.join(RESULTS_DIR, "mambavision_fft_wavelet_training_results.json"), "r") as f:
    mamba_crop_fw = json.load(f)

with open(os.path.join(RESULTS_DIR, "mambavision_fft_wavelet_fullimage_training_results.json"), "r") as f:
    mamba_full_fw = json.load(f)

print("ViT baseline loaded!")
print("MambaVision_S (laser crops) loaded!")
print("MambaVision_S (full image) loaded!")
print("MambaVision_S (laser crops + FFT/Wavelet) loaded!")
print("MambaVision_S (full image + FFT/Wavelet) loaded!")

# ═════════════════════════════════════════════════════════════════════════════
# 2. COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

comparison = {
    "ViT-Base": {
        "model"                  : "google/vit-base-patch16-224-in21k",
        "parameters_millions"    : 86.0,
        "input_type"             : "Full image",
        "features"               : "RGB",
        "best_val_accuracy_pct"  : vit["summary"]["best_val_accuracy_pct"],
        "test_accuracy_pct"      : "N/A",
        "avg_epoch_time_seconds" : "N/A",
        "inference_latency_ms"   : "N/A",
        "peak_gpu_memory_gb"     : "N/A",
        "convergence_epoch"      : "N/A",
        "num_epochs"             : vit["config"]["num_epochs"],
        "training_images"        : 2151,
        "platform"               : "Kaggle T4 GPU",
    },
    "MambaVision_S_LaserCrops": {
        "model"                  : "mamba_vision_S (NVlabs fork)",
        "parameters_millions"    : 50.0,
        "input_type"             : "Laser crops",
        "features"               : "RGB",
        "best_val_accuracy_pct"  : mamba_crop["best_val_accuracy_pct"],
        "test_accuracy_pct"      : mamba_crop["test_accuracy_pct"],
        "avg_epoch_time_seconds" : mamba_crop["avg_epoch_time_seconds"],
        "inference_latency_ms"   : mamba_crop["inference_latency_ms"],
        "peak_gpu_memory_gb"     : mamba_crop["peak_gpu_memory_gb"],
        "convergence_epoch"      : mamba_crop["best_epoch"],
        "num_epochs"             : mamba_crop["num_epochs"],
        "training_images"        : 2151,
        "platform"               : "MTSU Lambda RTX 3090",
    },
    "MambaVision_S_FullImage": {
        "model"                  : "mamba_vision_S (NVlabs fork)",
        "parameters_millions"    : 50.0,
        "input_type"             : "Full image",
        "features"               : "RGB",
        "best_val_accuracy_pct"  : mamba_full["best_val_accuracy_pct"],
        "test_accuracy_pct"      : mamba_full["test_accuracy_pct"],
        "avg_epoch_time_seconds" : mamba_full["avg_epoch_time_seconds"],
        "inference_latency_ms"   : mamba_full["inference_latency_ms"],
        "peak_gpu_memory_gb"     : mamba_full["peak_gpu_memory_gb"],
        "convergence_epoch"      : mamba_full["best_epoch"],
        "num_epochs"             : mamba_full["num_epochs"],
        "training_images"        : 2151,
        "platform"               : "MTSU Lambda RTX 3090",
    },
    "MambaVision_S_LaserCrops_FFTWavelet": {
        "model"                  : "mamba_vision_S + input_proj (NVlabs fork)",
        "parameters_millions"    : 50.0,
        "input_type"             : "Laser crops",
        "features"               : "RGB + FFT + Wavelet",
        "best_val_accuracy_pct"  : mamba_crop_fw["best_val_accuracy_pct"],
        "test_accuracy_pct"      : mamba_crop_fw["test_accuracy_pct"],
        "avg_epoch_time_seconds" : mamba_crop_fw["avg_epoch_time_seconds"],
        "inference_latency_ms"   : mamba_crop_fw["inference_latency_ms"],
        "peak_gpu_memory_gb"     : mamba_crop_fw["peak_gpu_memory_gb"],
        "convergence_epoch"      : mamba_crop_fw["best_epoch"],
        "num_epochs"             : mamba_crop_fw["num_epochs"],
        "training_images"        : 2151,
        "platform"               : "MTSU Lambda RTX 3090",
    },
    "MambaVision_S_FullImage_FFTWavelet": {
        "model"                  : "mamba_vision_S + input_proj (NVlabs fork)",
        "parameters_millions"    : 50.0,
        "input_type"             : "Full image",
        "features"               : "RGB + FFT + Wavelet",
        "best_val_accuracy_pct"  : mamba_full_fw["best_val_accuracy_pct"],
        "test_accuracy_pct"      : mamba_full_fw["test_accuracy_pct"],
        "avg_epoch_time_seconds" : mamba_full_fw["avg_epoch_time_seconds"],
        "inference_latency_ms"   : mamba_full_fw["inference_latency_ms"],
        "peak_gpu_memory_gb"     : mamba_full_fw["peak_gpu_memory_gb"],
        "convergence_epoch"      : mamba_full_fw["best_epoch"],
        "num_epochs"             : mamba_full_fw["num_epochs"],
        "training_images"        : 2151,
        "platform"               : "MTSU Lambda RTX 3090",
    },
}

output_path = os.path.join(RESULTS_DIR, "comparison_table_final.json")
with open(output_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nComparison table saved → {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. PRINT COMPARISON TABLE
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 95)
print("  FIVE-WAY FINAL COMPARISON")
print("=" * 95)
print(f"  {'Metric':<28} {'ViT-Base':>12} {'Mamba-Crop':>12} {'Mamba-Full':>12} {'Crop+FW':>12} {'Full+FW':>12}")
print("-" * 95)
print(f"  {'Input Type':<28} {'Full image':>12} {'Laser crops':>12} {'Full image':>12} {'Laser crops':>12} {'Full image':>12}")
print(f"  {'Features':<28} {'RGB':>12} {'RGB':>12} {'RGB':>12} {'RGB+FW':>12} {'RGB+FW':>12}")
print(f"  {'Parameters (M)':<28} {'86.0M':>12} {'~50M':>12} {'~50M':>12} {'~50M':>12} {'~50M':>12}")
print(f"  {'Best Val Accuracy':<28} {'94.58%':>12} {str(mamba_crop['best_val_accuracy_pct'])+'%':>12} {str(mamba_full['best_val_accuracy_pct'])+'%':>12} {str(mamba_crop_fw['best_val_accuracy_pct'])+'%':>12} {str(mamba_full_fw['best_val_accuracy_pct'])+'%':>12}")
print(f"  {'Test Accuracy':<28} {'N/A':>12} {str(mamba_crop['test_accuracy_pct'])+'%':>12} {str(mamba_full['test_accuracy_pct'])+'%':>12} {str(mamba_crop_fw['test_accuracy_pct'])+'%':>12} {str(mamba_full_fw['test_accuracy_pct'])+'%':>12}")
print(f"  {'Convergence Epoch':<28} {'N/A':>12} {str(mamba_crop['best_epoch']):>12} {str(mamba_full['best_epoch']):>12} {str(mamba_crop_fw['best_epoch']):>12} {str(mamba_full_fw['best_epoch']):>12}")
print(f"  {'Epochs Trained':<28} {str(vit['config']['num_epochs']):>12} {str(mamba_crop['num_epochs']):>12} {str(mamba_full['num_epochs']):>12} {str(mamba_crop_fw['num_epochs']):>12} {str(mamba_full_fw['num_epochs']):>12}")
print(f"  {'Inference Latency':<28} {'N/A':>12} {str(mamba_crop['inference_latency_ms'])+'ms':>12} {str(mamba_full['inference_latency_ms'])+'ms':>12} {str(mamba_crop_fw['inference_latency_ms'])+'ms':>12} {str(mamba_full_fw['inference_latency_ms'])+'ms':>12}")
print(f"  {'Peak GPU Memory':<28} {'N/A':>12} {str(mamba_crop['peak_gpu_memory_gb'])+'GB':>12} {str(mamba_full['peak_gpu_memory_gb'])+'GB':>12} {str(mamba_crop_fw['peak_gpu_memory_gb'])+'GB':>12} {str(mamba_full_fw['peak_gpu_memory_gb'])+'GB':>12}")
print(f"  {'Platform':<28} {'Kaggle T4':>12} {'RTX 3090':>12} {'RTX 3090':>12} {'RTX 3090':>12} {'RTX 3090':>12}")
print("=" * 95)

# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 1 — ACCURACY CURVES COMPARISON (FIVE-WAY)
# ═════════════════════════════════════════════════════════════════════════════

print("\nGenerating accuracy comparison figure...")

vit_epochs       = vit["training_curves"]["epochs"]
vit_acc          = [a * 100 for a in vit["training_curves"]["val_accuracies"]]

crop_epochs      = mamba_crop["training_curves"]["epochs"]
crop_acc         = [a * 100 for a in mamba_crop["training_curves"]["val_accuracies"]]

full_epochs      = mamba_full["training_curves"]["epochs"]
full_acc         = [a * 100 for a in mamba_full["training_curves"]["val_accuracies"]]

crop_fw_epochs   = mamba_crop_fw["training_curves"]["epochs"]
crop_fw_acc      = [a * 100 for a in mamba_crop_fw["training_curves"]["val_accuracies"]]

full_fw_epochs   = mamba_full_fw["training_curves"]["epochs"]
full_fw_acc      = [a * 100 for a in mamba_full_fw["training_curves"]["val_accuracies"]]

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("white")

ax.plot(vit_epochs, vit_acc,
        label="ViT-Base (full image, RGB)",
        color="#4C72B0", linewidth=2, marker="o", markersize=2)
ax.plot(crop_epochs, crop_acc,
        label="MambaVision_S — Laser Crops (RGB)",
        color="#E07B39", linewidth=2, marker="^", markersize=2)
ax.plot(full_epochs, full_acc,
        label="MambaVision_S — Full Image (RGB)",
        color="#2E7D5E", linewidth=2, marker="s", markersize=2)
ax.plot(crop_fw_epochs, crop_fw_acc,
        label="MambaVision_S — Laser Crops (RGB+FFT+Wav)",
        color="#E07B39", linewidth=2, linestyle="--", marker="^", markersize=2)
ax.plot(full_fw_epochs, full_fw_acc,
        label="MambaVision_S — Full Image (RGB+FFT+Wav)",
        color="#2E7D5E", linewidth=2, linestyle="--", marker="s", markersize=2)

ax.axhline(y=94.58, color="#4C72B0", linestyle=":",
           linewidth=1.2, alpha=0.6, label="ViT plateau: 94.58%")
ax.axhline(y=mamba_full["best_val_accuracy_pct"], color="#2E7D5E",
           linestyle=":", linewidth=1.2, alpha=0.4,
           label=f"Mamba-Full best: {mamba_full['best_val_accuracy_pct']}%")

ax.set_xlabel("Epoch", fontsize=13)
ax.set_ylabel("Validation Accuracy (%)", fontsize=13)
ax.set_title("Validation Accuracy: Five-Way Comparison\n"
             "Multi-Spectral Soil Moisture Classification (11 Classes)",
             fontsize=13, fontweight="bold")
ax.set_ylim([0, 100])
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "accuracy_comparison_final.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Accuracy comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 2 — EFFICIENCY COMPARISON BAR CHART (FIVE-WAY)
# ═════════════════════════════════════════════════════════════════════════════

print("Generating efficiency comparison figure...")

labels     = ["ViT-Base", "Mamba\nCrop", "Mamba\nFull", "Crop\n+FW", "Full\n+FW"]
colors     = ["#4C72B0", "#E07B39", "#2E7D5E", "#C0853A", "#1A5C44"]

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Five-Way Efficiency Comparison — Multi-Spectral Soil Moisture Classification\n"
             "(Speed metrics not available for ViT-Base on Kaggle)",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

# Panel 1 — Val Accuracy
val_accs = [
    94.58,
    mamba_crop["best_val_accuracy_pct"],
    mamba_full["best_val_accuracy_pct"],
    mamba_crop_fw["best_val_accuracy_pct"],
    mamba_full_fw["best_val_accuracy_pct"],
]
axes[0].bar(labels, val_accs, color=colors, width=0.5)
axes[0].set_title("Best Val Accuracy (%)", fontsize=11)
axes[0].set_ylabel("Accuracy (%)", fontsize=10)
for i, v in enumerate(val_accs):
    axes[0].text(i, v + 0.2, f"{v}%", ha="center", fontsize=9, fontweight="bold")
axes[0].set_ylim([80, 100])
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)
axes[0].grid(axis="y", alpha=0.3)

# Panel 2 — Test Accuracy (ViT N/A)
test_accs  = [0,
              mamba_crop["test_accuracy_pct"],
              mamba_full["test_accuracy_pct"],
              mamba_crop_fw["test_accuracy_pct"],
              mamba_full_fw["test_accuracy_pct"]]
bar_colors = ["#CCCCCC", "#E07B39", "#2E7D5E", "#C0853A", "#1A5C44"]
axes[1].bar(labels, test_accs, color=bar_colors, width=0.5)
axes[1].set_title("Test Accuracy (%)", fontsize=11)
axes[1].set_ylabel("Accuracy (%)", fontsize=10)
axes[1].text(0, 1.0, "N/A", ha="center", fontsize=9,
             fontweight="bold", color="#999999")
for i, v in enumerate(test_accs[1:], start=1):
    axes[1].text(i, v + 0.2, f"{v}%", ha="center", fontsize=9, fontweight="bold")
axes[1].set_ylim([70, 100])
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
axes[1].grid(axis="y", alpha=0.3)

# Panel 3 — Convergence Epoch (ViT N/A)
conv_epochs = [0,
               mamba_crop["best_epoch"],
               mamba_full["best_epoch"],
               mamba_crop_fw["best_epoch"],
               mamba_full_fw["best_epoch"]]
axes[2].bar(labels, conv_epochs, color=bar_colors, width=0.5)
axes[2].set_title("Convergence Epoch", fontsize=11)
axes[2].set_ylabel("Epoch", fontsize=10)
axes[2].text(0, 1.0, "N/A", ha="center", fontsize=9,
             fontweight="bold", color="#999999")
for i, v in enumerate(conv_epochs[1:], start=1):
    axes[2].text(i, v + 0.5, str(v), ha="center", fontsize=9, fontweight="bold")
axes[2].set_ylim([0, 90])
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "efficiency_comparison_final.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Efficiency comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 3 — FFT/WAVELET EFFECT: CROP vs FULL SIDE BY SIDE
# ═════════════════════════════════════════════════════════════════════════════

print("Generating FFT/Wavelet effect comparison figure...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FFT + Wavelet Feature Effect: Laser Crops vs Full Image\n"
             "Solid = RGB only | Dashed = RGB + FFT + Wavelet",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

# Left — Laser Crops
axes[0].plot(crop_epochs, crop_acc,
             label="Laser Crops — RGB", color="#E07B39",
             linewidth=2, marker="o", markersize=2)
axes[0].plot(crop_fw_epochs, crop_fw_acc,
             label="Laser Crops — RGB+FFT+Wav", color="#E07B39",
             linewidth=2, linestyle="--", marker="^", markersize=2)
axes[0].axhline(y=94.58, color="#4C72B0", linestyle=":",
                linewidth=1.2, alpha=0.6, label="ViT baseline: 94.58%")
axes[0].set_title("Laser Crops", fontsize=12)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Val Accuracy (%)")
axes[0].set_ylim([0, 100])
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right — Full Image
axes[1].plot(full_epochs, full_acc,
             label="Full Image — RGB", color="#2E7D5E",
             linewidth=2, marker="o", markersize=2)
axes[1].plot(full_fw_epochs, full_fw_acc,
             label="Full Image — RGB+FFT+Wav", color="#2E7D5E",
             linewidth=2, linestyle="--", marker="s", markersize=2)
axes[1].axhline(y=94.58, color="#4C72B0", linestyle=":",
                linewidth=1.2, alpha=0.6, label="ViT baseline: 94.58%")
axes[1].set_title("Full Image", fontsize=12)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val Accuracy (%)")
axes[1].set_ylim([0, 100])
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "fft_wavelet_curves_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"FFT/Wavelet effect comparison saved → {fig_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PAPER-READY SUMMARY — FIVE-WAY COMPARISON")
print("=" * 65)
print(f"  ViT-Base val accuracy                      : 94.58%")
print(f"  Mamba-Crops (RGB) val / test               : {mamba_crop['best_val_accuracy_pct']}% / {mamba_crop['test_accuracy_pct']}%")
print(f"  Mamba-Full (RGB) val / test                : {mamba_full['best_val_accuracy_pct']}% / {mamba_full['test_accuracy_pct']}%")
print(f"  Mamba-Crops + FFT/Wav val / test           : {mamba_crop_fw['best_val_accuracy_pct']}% / {mamba_crop_fw['test_accuracy_pct']}%")
print(f"  Mamba-Full + FFT/Wav val / test            : {mamba_full_fw['best_val_accuracy_pct']}% / {mamba_full_fw['test_accuracy_pct']}%")
print("-" * 65)
print(f"  Full image gain over ViT                   : +{round(mamba_full['best_val_accuracy_pct'] - 94.58, 2)}%")
print(f"  FFT/Wav test gain on full image            : +{round(mamba_full_fw['test_accuracy_pct'] - mamba_full['test_accuracy_pct'], 2)}%")
print(f"  FFT/Wav test impact on laser crops         : {round(mamba_crop_fw['test_accuracy_pct'] - mamba_crop['test_accuracy_pct'], 2)}%")
print(f"  Best overall test accuracy                 : {mamba_full_fw['test_accuracy_pct']}% (Mamba-Full + FFT/Wav)")
print("=" * 65)
print("\nAll figures saved to ./results/")
print("Ready for 07c_inference_pipeline.py")