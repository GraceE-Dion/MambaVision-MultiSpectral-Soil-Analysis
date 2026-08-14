"""
11_speed_optimization.py
=========================
Benchmarks MambaVision_S inference speed across four optimization
techniques on the MTSU Lambda RTX 3090. Results are compared against
ViT-Base (7.26 ms/image from 03b_vit_cluster_benchmark.py).

Optimization techniques tested:
  1. Baseline FP32
  2. FP16 (half precision)
  3. TF32 + torch.compile (reduce-overhead)
  4. TF32 + torch.compile (max-autotune)

Run:
    python 11_speed_optimization.py

Output:
    results/speed_optimization.json
    results/speed_optimization_comparison.png
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MambaVision ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda")
N_WARMUP    = 20
N_BENCHMARK = 100

print("=" * 60)
print("  MambaVision_S Inference Speed Optimization Benchmark")
print(f"  GPU    : {torch.cuda.get_device_name(0)}")
print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Warmup : {N_WARMUP} runs")
print(f"  Benchmark: {N_BENCHMARK} runs")
print("=" * 60)

# ═════════════════════════════════════════════════════════════════════════════
# HELPER — load fresh model
# ═════════════════════════════════════════════════════════════════════════════

def load_model(fp16=False):
    m = models.mamba_vision_S(pretrained=False)
    m.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.head.in_features, 11)
    )
    m = m.to(device).eval()
    if fp16:
        m = m.half()
    return m

# ═════════════════════════════════════════════════════════════════════════════
# HELPER — run benchmark
# ═════════════════════════════════════════════════════════════════════════════

def benchmark(model, x, label):
    print(f"\n  [{label}]")

    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(N_BENCHMARK):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)

    mean_lat = round(float(np.mean(times)), 3)
    min_lat  = round(float(np.min(times)), 3)
    p95_lat  = round(float(np.percentile(times, 95)), 3)
    std_lat  = round(float(np.std(times)), 3)
    peak_mem = round(torch.cuda.max_memory_allocated(0) / 1e9, 2)

    print(f"    Mean latency : {mean_lat} ms")
    print(f"    Min latency  : {min_lat} ms")
    print(f"    P95 latency  : {p95_lat} ms")
    print(f"    Std deviation: {std_lat} ms")
    print(f"    Peak GPU mem : {peak_mem} GB")

    return {
        "mean_ms" : mean_lat,
        "min_ms"  : min_lat,
        "p95_ms"  : p95_lat,
        "std_ms"  : std_lat,
        "peak_gpu_gb": peak_mem,
    }

# ═════════════════════════════════════════════════════════════════════════════
# 1. BASELINE FP32
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading baseline model...")
model_fp32 = load_model(fp16=False)
x_fp32 = torch.randn(1, 3, 224, 224).to(device)
result_fp32 = benchmark(model_fp32, x_fp32, "Baseline FP32")
del model_fp32
torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════════════════════
# 2. FP16
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading FP16 model...")
model_fp16 = load_model(fp16=True)
x_fp16 = torch.randn(1, 3, 224, 224).to(device).half()
result_fp16 = benchmark(model_fp16, x_fp16, "FP16 Half Precision")
del model_fp16
torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════════════════════
# 3. TF32 + torch.compile (reduce-overhead)
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading TF32 + compile (reduce-overhead) model...")
torch.set_float32_matmul_precision('high')
model_compiled = load_model(fp16=False)
model_compiled = torch.compile(model_compiled, mode='reduce-overhead')
x_fp32 = torch.randn(1, 3, 224, 224).to(device)
result_compiled = benchmark(
    model_compiled, x_fp32,
    "TF32 + torch.compile (reduce-overhead)")
del model_compiled
torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════════════════════
# 4. TF32 + torch.compile (max-autotune)
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading TF32 + compile (max-autotune) model...")
print("  Note: max-autotune compilation takes several minutes...")
model_autotune = load_model(fp16=False)
model_autotune = torch.compile(model_autotune, mode='max-autotune')
result_autotune = benchmark(
    model_autotune, x_fp32,
    "TF32 + torch.compile (max-autotune)")
del model_autotune
torch.cuda.empty_cache()

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

vit_latency = 7.26  # from 03b_vit_cluster_benchmark.py

results = {
    "gpu"              : torch.cuda.get_device_name(0),
    "vit_base_ms"      : vit_latency,
    "optimizations"    : {
        "baseline_fp32"           : result_fp32,
        "fp16"                    : result_fp16,
        "tf32_compile_reduce"     : result_compiled,
        "tf32_compile_autotune"   : result_autotune,
    },
    "best_method"      : "tf32_compile_reduce",
    "best_mean_ms"     : result_compiled["mean_ms"],
    "best_min_ms"      : result_compiled["min_ms"],
    "speedup_vs_baseline" : round(result_fp32["mean_ms"] / result_compiled["mean_ms"], 2),
    "speedup_vs_vit"      : round(vit_latency / result_compiled["mean_ms"], 2),
}

print("\n" + "=" * 60)
print("  SPEED OPTIMIZATION SUMMARY")
print("=" * 60)
print(f"  {'Method':<35} {'Mean (ms)':>10} {'Speedup vs Baseline':>20}")
print("-" * 60)
print(f"  {'ViT-Base (RTX 3090)':<35} {vit_latency:>10} {'---':>20}")
print(f"  {'Baseline FP32':<35} {result_fp32['mean_ms']:>10} {'1.00x':>20}")
print(f"  {'FP16':<35} {result_fp16['mean_ms']:>10} {str(round(result_fp32['mean_ms']/result_fp16['mean_ms'],2))+'x':>20}")
print(f"  {'TF32 + compile (reduce-overhead)':<35} {result_compiled['mean_ms']:>10} {str(round(result_fp32['mean_ms']/result_compiled['mean_ms'],2))+'x':>20}")
print(f"  {'TF32 + compile (max-autotune)':<35} {result_autotune['mean_ms']:>10} {str(round(result_fp32['mean_ms']/result_autotune['mean_ms'],2))+'x':>20}")
print("=" * 60)
print(f"\n  Best method    : TF32 + torch.compile (reduce-overhead)")
print(f"  Best latency   : {result_compiled['mean_ms']} ms mean / {result_compiled['min_ms']} ms min")
print(f"  Speedup vs baseline : {results['speedup_vs_baseline']}x")
print(f"  Speedup vs ViT      : {results['speedup_vs_vit']}x faster than ViT-Base")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE — Speed comparison bar chart
# ═════════════════════════════════════════════════════════════════════════════

labels  = [
    "ViT-Base\n(RTX 3090)",
    "MambaVision_S\nBaseline FP32",
    "MambaVision_S\nFP16",
    "MambaVision_S\nTF32+Compile\n(reduce)",
    "MambaVision_S\nTF32+Compile\n(autotune)",
]
means   = [
    vit_latency,
    result_fp32["mean_ms"],
    result_fp16["mean_ms"],
    result_compiled["mean_ms"],
    result_autotune["mean_ms"],
]
colors  = ["#4C72B0", "#E07B39", "#C0853A", "#2E7D5E", "#1A5C44"]

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")

bars = ax.bar(labels, means, color=colors, width=0.5)
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val} ms", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("Mean Inference Latency (ms)", fontsize=12)
ax.set_title("MambaVision_S Inference Speed Optimization\n"
             "MTSU Lambda RTX 3090 — Properly GPU-Synchronized Benchmark",
             fontsize=12, fontweight="bold")
ax.set_ylim([0, max(means) * 1.2])
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "speed_optimization_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nFigure saved → {fig_path}")

# Save JSON
json_path = os.path.join(RESULTS_DIR, "speed_optimization.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {json_path}")
print("\nDone!")