"""
04_mambavision_backbone.py
==========================
Sets up the MambaVision backbone with an 11-class classification head
for the soil moisture dataset. Loads from the NVlabs fork at
/data/Grace/MambaVision. Verifies the model loads correctly and
runs a forward pass on the cluster hardware.

Run:
    python 04_mambavision_backbone.py

Output:
    results/mambavision_model_summary.json
"""

import json
import os
import sys
import time
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ── Load MambaVision from NVlabs fork ─────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

# ── Required for cluster (RTX 3090 + mamba-ssm) ───────────────────────────────
torch.backends.cudnn.enabled = False

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NUM_CLASSES   = 11
IMAGE_SIZE    = 224
BATCH_SIZE    = 4
MODEL_VARIANT = "mamba_vision_T"

# ═════════════════════════════════════════════════════════════════════════════
# 2. DEVICE
# ═════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {device}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═════════════════════════════════════════════════════════════════════════════
# 3. LOAD MAMBAVISION BACKBONE
# ═════════════════════════════════════════════════════════════════════════════

print(f"\nLoading {MODEL_VARIANT} from NVlabs fork...")

model = models.mamba_vision_T(pretrained=True)

# Replace classification head with 11-class head
in_features = model.head.in_features
model.head = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)
print("Model loaded and head replaced successfully!")

# ═════════════════════════════════════════════════════════════════════════════
# 4. PARAMETER COUNT
# ═════════════════════════════════════════════════════════════════════════════

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters     : {total_params / 1e6:.2f}M")
print(f"Trainable parameters : {trainable_params / 1e6:.2f}M")

# ═════════════════════════════════════════════════════════════════════════════
# 5. FORWARD PASS TEST
# ═════════════════════════════════════════════════════════════════════════════

print(f"\nRunning forward pass test (batch_size={BATCH_SIZE}, image_size={IMAGE_SIZE})...")

dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

model.eval()
with torch.no_grad():
    start = time.time()
    output = model(dummy_input)
    elapsed = time.time() - start

print(f"Output shape         : {output.shape}")
print(f"Forward pass time    : {elapsed * 1000:.1f} ms for {BATCH_SIZE} images")
print(f"Per image            : {elapsed * 1000 / BATCH_SIZE:.1f} ms")

assert output.shape == (BATCH_SIZE, NUM_CLASSES), \
    f"Expected ({BATCH_SIZE}, {NUM_CLASSES}), got {output.shape}"
print("Forward pass assertion passed!")

# ═════════════════════════════════════════════════════════════════════════════
# 6. GPU MEMORY AFTER FORWARD PASS
# ═════════════════════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    mem_allocated = torch.cuda.memory_allocated(0) / 1e9
    mem_reserved  = torch.cuda.memory_reserved(0) / 1e9
    print(f"\nGPU memory allocated : {mem_allocated:.2f} GB")
    print(f"GPU memory reserved  : {mem_reserved:.2f} GB")
else:
    mem_allocated = None
    mem_reserved  = None

# ═════════════════════════════════════════════════════════════════════════════
# 7. SAVE MODEL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

summary = {
    "model_variant"             : MODEL_VARIANT,
    "num_classes"               : NUM_CLASSES,
    "image_size"                : IMAGE_SIZE,
    "total_params_millions"     : round(total_params / 1e6, 2),
    "trainable_params_millions" : round(trainable_params / 1e6, 2),
    "forward_pass_ms_batch4"    : round(elapsed * 1000, 1),
    "forward_pass_ms_per_img"   : round(elapsed * 1000 / BATCH_SIZE, 1),
    "gpu_memory_allocated_gb"   : round(mem_allocated, 2) if mem_allocated else None,
    "gpu_memory_reserved_gb"    : round(mem_reserved, 2) if mem_reserved else None,
    "device"                    : str(device),
    "cudnn_enabled"             : torch.backends.cudnn.enabled,
}

output_path = os.path.join(RESULTS_DIR, "mambavision_model_summary.json")
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nModel summary saved → {output_path}")

# ═════════════════════════════════════════════════════════════════════════════
# 8. PRINT COMPARISON PREVIEW
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("  MAMBAVISION vs ViT — PARAMETER COMPARISON")
print("=" * 55)
print(f"  ViT-Base       : ~86.0M parameters")
print(f"  {MODEL_VARIANT:<14} : ~{total_params / 1e6:.2f}M parameters")
diff = 86.0 - total_params / 1e6
print(f"  Difference     : {abs(diff):.2f}M fewer params" if diff > 0
      else f"  Difference     : {abs(diff):.2f}M more params")
print("=" * 55)
print("\nBackbone verified. Ready for 05_training.py")