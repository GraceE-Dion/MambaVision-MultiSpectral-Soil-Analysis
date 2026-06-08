# MambaVision-MultiSpectral-Soil-Analysis
MambaVision Hybrid Mamba-Transformer backbone applied to multi-spectral laser soil moisture classification - extending the ViT baseline with state-space sequence modeling
# MambaVision vs ViT: Multi-Spectral Soil Moisture Classification

**MambaVision_S | MTSU Lambda Cluster | RTX 3090 | 11-Class Moisture Classification**

This repository presents a systematic architectural comparison of MambaVision_S against the ViT-Base baseline established in the companion project, evaluated on multi-spectral laser soil moisture imagery across five controlled experimental conditions. The work extends the ViT pipeline with three original research contributions: a full image MambaVision training regime, a laser crop vs full image input representation ablation, and a Fourier Transform + Wavelet Transform frequency feature integration experiment.

---

## Repository Structure

```
├── 02_data_preparation.py
├── 03_baseline_vit_comparison.py
├── 04_mambavision_backbone.py
├── 05_training.py                                # MambaVision_S — laser crops, RGB
├── 05b_training_fullimage.py                     # MambaVision_S — full image, RGB
├── 05c_training_fft_wavelet.py                   # MambaVision_S — laser crops, RGB+FFT+Wav
├── 05bc_training_fft_wavelet_fullimage.py        # MambaVision_S — full image, RGB+FFT+Wav
├── 06_evaluation.py
├── 06b_evaluation.py                             # Three-way comparison
├── 06c_evaluation.py                             # Five-way final comparison
├── 07_inference_pipeline.py
├── 07b_inference_pipeline.py
├── 07c_inference_pipeline.py
├── 07bc_inference_pipeline.py
└── results/
```

---

## Background and Motivation

The companion project established ViT-Base as the baseline classifier for multi-spectral laser soil moisture classification across 11 discrete moisture levels (0–10), using seven Roboflow datasets spanning standard visible, infrared, and ultraviolet spectral modalities. The best ViT result was 94.58% validation accuracy on full images (Phase 2) and 90.64% on laser crops (Phase 4B), with YOLOv8 achieving 95.3% mAP50 as the production detection model (Phase 6).

This project investigates whether MambaVision_S — a hybrid Mamba-Transformer architecture combining selective state space modeling with transformer attention — can match or exceed ViT performance on this task with fewer parameters and faster inference. A secondary question is whether frequency-domain features (FFT magnitude, Haar wavelet decomposition) provide additive discriminative signal beyond RGB alone, and whether that signal is input-scale dependent.

---

## Datasets

Seven multi-spectral laser datasets sourced from Roboflow, unified into a single 11-class moisture classification corpus:

| Dataset | Spectrum | Focus |
|---|---|---|
| soil-moisture-v4 | Standard visible | Baseline laser reflection patterns |
| soil-moisture-v4-IR | Infrared | Thermal moisture signatures |
| soil-moisture-v4-UV | Ultraviolet | High-contrast mineral/moisture separation |
| soil-moisture-IR | Infrared | Secondary heat-based validation |
| soil-moisture-5sagf | General field | Diverse environmental conditions |
| soil_moisture_september | Temporal (Sept) | Seasonal moisture variation |
| soil_moisture_stir_september | Temporal (Sept) | Stirred soil reflectance |

**Training sets:** Master_Laser_Crops and Master_Soil_Moisture each contain 2,151 images (717 originals + 2x physical augmentation via Gaussian noise and salt-and-pepper noise copies saved to disk, identical to ViT Step 16 augmentation for controlled comparison).

---

## Model Architecture

### MambaVision_S (Baseline)

MambaVision_S (~50M parameters) is loaded from the NVlabs fork (`GraceE-Dion/MambaVision`, branch `grace/research-main`) via `sys.path.insert`. The pretrained classification head is replaced with:

```python
model.head = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 11)
)
```

`torch.backends.cudnn.enabled = False` is required on the MTSU Lambda cluster. timm-based loading does not work — the NVlabs fork must be used via `sys.path.insert`.

### MambaVisionFFTWavelet (FFT + Wavelet Variant)

A lightweight `input_proj` Conv2d(7→3, 1×1) layer is prepended to the backbone, projecting 7-channel inputs (3 RGB + 1 FFT + 3 Wavelet) back to 3 channels before the pretrained backbone. Pretrained backbone weights are fully preserved.

```python
class MambaVisionFFTWavelet(nn.Module):
    def __init__(self, num_classes, in_channels=7):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.input_proj.weight)
        self.backbone = models.mamba_vision_S(pretrained=True)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        return self.backbone(x)
```

**FFT channel:** 2D FFT magnitude spectrum (log-scaled, normalized) computed from grayscale image. Captures global frequency patterns and periodic texture structure.

**Wavelet channels:** Single-level Haar DWT producing three detail sub-bands — horizontal (cH), vertical (cV), and diagonal (cD) — each resized to input spatial dimensions. Captures multi-scale local frequency detail at different orientations.

---

## Training Configuration

All MambaVision experiments use identical hyperparameters for controlled comparison:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Scheduler | CosineAnnealingLR |
| Epochs | 80 |
| Batch size | 16 |
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Class weighting | Inverse frequency WeightedRandomSampler |
| Image size | 224×224 |
| Platform | MTSU Lambda RTX 3090 |

---

## Results

### Five-Way Comparison

| Model | Input | Features | Val Acc | Test Acc | Convergence | GPU Mem |
|---|---|---|---|---|---|---|
| ViT-Base | Full image | RGB | 94.58% | N/A | Epoch 25 | N/A |
| MambaVision_S | Laser crops | RGB | 90.64% | 85.85% | Epoch 36 | 2.01 GB |
| **MambaVision_S** | **Full image** | **RGB** | **97.04%** | **95.28%** | **Epoch 15** | **2.01 GB** |
| MambaVision_S | Laser crops | RGB+FFT+Wav | 90.64% | 82.08% | Epoch 54 | 2.03 GB |
| MambaVision_S | Full image | RGB+FFT+Wav | 97.04% | 96.23% | Epoch 17 | 2.03 GB |

### MambaVision_S vs ViT-Base (Full Image)

| Metric | ViT-Base | MambaVision_S | Delta |
|---|---|---|---|
| Val Accuracy | 94.58% | 97.04% | +2.46% |
| Test Accuracy | N/A | 95.28% | — |
| Parameters | 86M | ~50M | -42% |
| Convergence Epoch | 25 | 15 | -40% |
| Inference Latency | N/A | 0.98 ms/image | — |
| Peak GPU Memory | N/A | 2.01 GB | — |

### Per-Dataset Inference Results

| Dataset | Mamba-Crop | Mamba-Full | Crop+FW | Full+FW |
|---|---|---|---|---|
| Soil-Moisture-v4 | 8/8 (100%) | 8/8 (100%) | 5/8 (62.5%) | 8/8 (100%) |
| Soil-Moisture-v4-IR | 7/7 (100%) | 7/7 (100%) | 7/7 (100%) | 7/7 (100%) |
| Soil-Moisture-v4-UV | 7/7 (100%) | 7/7 (100%) | 7/7 (100%) | 7/7 (100%) |
| Soil-Moisture-IR | 7/7 (100%) | 7/7 (100%) | 3/7 (42.86%) | 7/7 (100%) |
| Soil-Moisture-5sagf | 0/0 | 0/0 | 0/0 | 0/0 |
| Soil-Moisture-September | 5/7 (71.43%) | 5/7 (71.43%) | 4/7 (57.14%) | 5/7 (71.43%) |
| Soil-Moisture-Stir-September | 4/5 (80%) | 4/5 (80%) | 0/5 (0%) | 4/5 (80%) |
| **OVERALL** | **38/41 (92.68%)** | **38/41 (92.68%)** | **26/41 (63.41%)** | **38/41 (92.68%)** |

---

## Key Findings

### Finding 1: Input Representation Dominates Architecture Choice

MambaVision_S achieves 97.04% val accuracy on full images versus 90.64% on laser crops — a 6.40% gap from input type alone, using the same model, same hardware, and same training configuration. Full spatial context provides substantially richer discriminative signal than isolated laser region crops for this 11-class moisture classification task. Investing in input representation quality yields larger accuracy gains than architectural changes or feature engineering, at least within the parameter range studied here.

### Finding 2: MambaVision_S Outperforms ViT-Base on Full Images

On full multispectral images, MambaVision_S exceeds the ViT Phase 2 baseline by +2.46% validation accuracy while using 42% fewer parameters, converging 40% faster (epoch 15 vs epoch 25), and maintaining zero overfitting across 65 subsequent epochs. The model locked at 97.04% from epoch 15 through epoch 80 with no accuracy degradation — an unusually strong generalization stability result. Peak GPU memory of 2.01 GB and 0.98 ms/image inference latency confirm real-time deployment viability in precision agriculture edge contexts.

### Finding 3: Frequency Features Are Input-Scale Dependent

FFT magnitude and Haar wavelet detail channels provide a small but genuine test accuracy improvement on full images (+0.95%: 95.28% → 96.23%) but actively degrade generalization on laser crops (-3.77%: 85.85% → 82.08%). FFT captures global periodic texture patterns across the full image where moisture gradients manifest as structured frequency content. On small laser crop regions, the same FFT operation captures primarily high-frequency noise from crop boundary artifacts rather than meaningful moisture-correlated texture. Frequency feature augmentation should only be applied when input spatial context is sufficient to produce meaningful frequency content.

### Finding 4: September Dataset Challenge Is Architecture-Independent

Soil-Moisture-September and Soil-Moisture-Stir-September remain consistent weak spots across all five model variants. This confirms the limitation is environmental capture inconsistency and soil disturbance physics, consistent with the root cause identified in the companion ViT Phase 6 visual investigation. Addressing these datasets requires standardizing the capture environment — augmentation-based mitigation is insufficient.

### Finding 5: Generalization Stability Under Extended Training

The full image model locked at 97.04% val accuracy from epoch 15 through epoch 80 with no degradation — 65 consecutive epochs of zero accuracy movement. Val loss continued a mild downward trend (0.7500 at epoch 15 to 0.7130 at epoch 80) while train loss steadily decreased, confirming the model refined internal representations without generalization degradation. This stability is atypical and worth reporting as a standalone finding.

---

## Architecture Evolution and Training History

### MambaVision_T → MambaVision_S

Initial training used MambaVision_T (Tiny, 31.16M parameters). After 40 epochs it plateaued at 79.80% val accuracy — insufficient capacity for 11-class fine-grained laser moisture classification. Upgrade to MambaVision_S (~50M parameters) resolved this.

### Laser Crops Epoch Progression

| Round | Model | Epochs | Augmentation | Best Val Acc | Decision |
|---|---|---|---|---|---|
| 1 | MambaVision_T | 25 | No | 70.94% | Insufficient — switched to S |
| 2 | MambaVision_T | 40 | No | 79.80% | Still insufficient — upgraded |
| 3 | MambaVision_S | 40 | No | 82.27% | Improving, plateau present |
| 4 | MambaVision_S | 60 | No | 81.28% | Overfitting present |
| 5 | MambaVision_S | 60 | Yes (2,151) | 90.15% | Augmentation resolved overfitting |
| 6 | MambaVision_S | 80 | Yes (2,151) | **90.64%** | Stable plateau — final laser crops model |

---

## Comparison with Companion ViT Project

| Phase | Model | Input | Val Acc | Notes |
|---|---|---|---|---|
| ViT Phase 2 | ViT-Base (86M) | Full image | 94.58% | Augmented baseline |
| ViT Phase 4B | ViT-Base (86M) | Laser crops | 90.64% | Best ViT crop result |
| ViT Phase 6 | YOLOv8s | Full image | 95.3% mAP50 | Detection pipeline |
| **This work** | **MambaVision_S (50M)** | **Full image** | **97.04%** | **Best overall result** |
| This work | MambaVision_S (50M) | Laser crops | 90.64% | Matches ViT Phase 4B |
| This work | MambaVision_S (50M) | Full image + FW | 97.04% / 96.23% test | Best test accuracy |

MambaVision_S on full images exceeds both the ViT full image baseline (+2.46%) and matches the ViT laser crops result with 42% fewer parameters. The YOLOv8 Phase 6 result (95.3% mAP50) operates as an object detector rather than a classifier and is not directly comparable, but the MambaVision full image test accuracy (95.28%–96.23%) is competitive on the classification task alone.

---

## Known Dataset Limitations

**Soil-Moisture-5sagf:** No test images found in the laser crops folder across any inference run. This dataset contributes 0/0 across all five models.

**Soil-Moisture-Stir-September:** Consistent 80% inference accuracy on full image RGB and RGB+FFT/Wav models; 0% on laser crop FFT/Wav variant. Root cause established from companion project visual investigation: (1) soil stirring physically disrupts the laser reflection pattern, and (2) uncontrolled field capture produces laser spots that are frequently dim, small, or invisible. This is an environmental capture problem, not a model failure.

**Class index remapping:** HuggingFace ImageFolder assigns class indices alphabetically. For 11 numerical classes (0–10), alphabetical order places Level_10 at index 1, not index 10. All training and inference scripts apply `hf_to_correct` remapping through both ground truth labels and predicted outputs. A bug in the original 07 and 07b inference scripts where `argmax` output was not remapped was identified through anomaly detection (9.76% inference accuracy inconsistent with 95.28% test accuracy), root-cause investigated, and corrected across all inference pipelines.

---

## AI Governance and Responsible Development

**Honest negative findings:** The FFT/Wavelet experiment on laser crops produced a -3.77% test accuracy degradation. This result is fully documented and preserved rather than discarded. Negative findings have equal evidentiary value to positive ones and are essential to reproducible research.

**Audit trail of design decisions:** The full epoch progression table is preserved, documenting why each architectural and hyperparameter decision was made. This prevents selective reporting of only the best results and supports downstream reproducibility.

**Dataset integrity:** The class index remapping bug was identified through anomaly detection in inference results — 9.76% inference accuracy flagged as inconsistent with 95.28% test accuracy, triggering root cause investigation. Governance-aware validation surfaces pipeline errors invisible to training metrics alone.

**Deployment risk profiling:** Per-dataset inference results are reported separately rather than as aggregate accuracy only, enabling risk-informed deployment decisions. Five of seven datasets are reliable; September and Stir-September represent known deployment risk requiring capture environment remediation before production use.

**Parameter efficiency transparency:** The 42% parameter reduction (86M → 50M) is reported alongside accuracy, not in isolation. Parameter efficiency is only meaningful when accuracy is not sacrificed — the full image results show both can be achieved simultaneously.

---

## Technical Specification

| Parameter | Mamba-Crop | Mamba-Full | Crop+FW | Full+FW | ViT-Base |
|---|---|---|---|---|---|
| Architecture | MambaVision_S | MambaVision_S | MambaVision_S+proj | MambaVision_S+proj | ViT-Base-patch16 |
| Parameters | ~50M | ~50M | ~50M | ~50M | 86M |
| Input channels | 3 | 3 | 7 | 7 | 3 |
| Input type | Laser crops | Full image | Laser crops | Full image | Full image |
| Hardware | RTX 3090 | RTX 3090 | RTX 3090 | RTX 3090 | Kaggle T4 |
| Optimizer | AdamW (2e-5) | AdamW (2e-5) | AdamW (2e-5) | AdamW (2e-5) | AdamW (2e-5) |
| Epochs | 80 | 80 | 80 | 80 | 25 |
| Convergence | 36 | 15 | 54 | 17 | 25 |
| Best Val Acc | 90.64% | 97.04% | 90.64% | 97.04% | 94.58% |
| Test Acc | 85.85% | 95.28% | 82.08% | 96.23% | N/A |
| Peak GPU Mem | 2.01 GB | 2.01 GB | 2.03 GB | 2.03 GB | N/A |
| Inference | 0.90 ms | 0.98 ms | 0.98 ms | 0.94 ms | N/A |

---

## Reproducibility

**Environment:**
- MTSU Lambda Cluster, 2x RTX 3090 (24GB VRAM)
- Conda environment: `mambavision`
- PyTorch 2.4.1+cu121, mamba-ssm 2.2.2, triton 3.0.0
- PyWavelets 1.8.0
- `torch.backends.cudnn.enabled = False` required

**Session start:**
```bash
cd /data/Grace/MambaVision-MultiSpectral-Soil-Analysis
conda activate mambavision
```

**MambaVision loading:**
```python
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models
model = models.mamba_vision_S(pretrained=True)
```

timm-based loading does not work. NVlabs fork must be used via `sys.path.insert`.

---

## Conclusion

This project demonstrates that MambaVision_S outperforms ViT-Base on multi-spectral laser soil moisture classification when operating on full images, achieving 97.04% validation accuracy and 95.28% test accuracy with 42% fewer parameters and 40% faster convergence. Three original findings are established:

1. Input representation quality (full image vs laser crop) has a larger impact on classification accuracy than architectural choice within the parameter range studied, producing a 6.40% accuracy gap from input type alone.

2. MambaVision_S exceeds ViT-Base by +2.46% on the same full image input with substantially better efficiency, convergence stability, and zero overfitting across 80 training epochs.

3. Frequency-domain features (FFT + Wavelet) provide marginal but genuine test accuracy improvement on full images (+0.95%) and are counterproductive on laser crops (-3.77%), establishing that frequency feature utility is conditional on input spatial scale.

The Soil-Moisture-September and Stir-September datasets remain challenging across all model variants, consistent with the environmental capture limitations identified in the companion ViT project. Addressing these limitations requires standardizing the capture environment rather than architectural or augmentation-based mitigation.

Future work includes: (a) integrating MambaVision_S as the YOLOv8 detection backbone to compare against Phase 6 on the full detection task, (b) per-class F1 analysis and confusion matrix generation for the full image models, and (c) attention visualization to identify which spatial regions MambaVision_S prioritizes relative to ViT on moisture-ambiguous samples.

---

**GitHub:** https://github.com/GraceE-Dion/MambaVision-MultiSpectral-Soil-Analysis  
**Contact:** efahnegbedion@gmail.com

