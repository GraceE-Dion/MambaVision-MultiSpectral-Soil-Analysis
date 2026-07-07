"""
08_detection_training.py
========================
MambaVision_S backbone + FPN neck + anchor-based detection head.
Single forward pass predicting bounding box + moisture class jointly.
Trains on unified detection dataset built from all 7 Roboflow sources.
Directly comparable to YOLOv8 Phase 6 (95.3% mAP50, 89.1% inference accuracy).

Run:
    python 08_detection_training.py

Output:
    results/detection/mambavision_detection_best.pth
    results/detection/mambavision_detection_training_results.json
    results/detection/mambavision_detection_training_curves.png
"""

import os
import sys
import json
import time
import math
import shutil
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MambaVision ───────────────────────────────────────────────────────────────
sys.path.insert(0, '/data/Grace/MambaVision')
from mambavision import models

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_DIR   = "/data/Grace/soil-moisture-dataset"
UNIFIED_DIR  = "/data/Grace/Master_Detection"
RESULTS_DIR  = "./results/detection"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

NUM_CLASSES  = 11
IMAGE_SIZE   = 640          # YOLOv8 standard input size
BATCH_SIZE   = 8
NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
PATIENCE     = 10

# Anchor boxes (3 scales x 3 ratios) — designed for laser spot sizes
# Laser spots typically occupy 6-30% of image area
ANCHORS = [
    # P3 — small spots (stride 8)
    [(10, 13), (16, 30), (33, 23)],
    # P4 — medium spots (stride 16)
    [(30, 61), (62, 45), (59, 119)],
    # P5 — large spots (stride 32)
    [(116, 90), (156, 198), (373, 326)],
]
STRIDES = [8, 16, 32]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. PER-DATASET CLASS REMAPPING
# ═════════════════════════════════════════════════════════════════════════════

# Maps each dataset's local class index → unified moisture level (0-10)
DATASET_CLASS_MAPS = {
    "Soil-Moisture-v4-3": {
        0:0, 1:1, 2:10, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9
    },
    "Soil-Moisture-v4-IR-1": {
        0:10, 1:2, 2:4, 3:5, 4:6, 5:8, 6:9
    },
    "Soil-Moisture-v4-UV-1": {
        0:0, 1:1, 2:10, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9
    },
    "Soil_Moisture_September-8": {
        0:0, 1:1, 2:10, 3:3, 4:4, 5:5, 6:7, 7:8, 8:9
    },
    "Soil_Moisture_Stir_September-4": {
        0:1, 1:10, 2:3, 3:7, 4:8
    },
    "Soil-Moisture-1": {
        0:1, 1:2, 2:3, 3:5, 4:8   # 1.0→1, 2.0→2, 3.0→3, 5.0→5, 8.2→8
    },
    "Soil-Moisture-IR-1": {
        0:1, 1:2, 2:3, 3:5, 4:8
    },
}

# ═════════════════════════════════════════════════════════════════════════════
# 3. BUILD UNIFIED DETECTION DATASET
# ═════════════════════════════════════════════════════════════════════════════

def build_unified_dataset():
    """
    Copies all images and remapped label files from the 7 source datasets
    into a unified YOLO-format directory at UNIFIED_DIR.
    """
    if os.path.exists(UNIFIED_DIR):
        print(f"Unified dataset already exists at {UNIFIED_DIR} — skipping build")
        return

    print("\nBuilding unified detection dataset...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(UNIFIED_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(UNIFIED_DIR, split, "labels"), exist_ok=True)

    total = {"train": 0, "val": 0, "test": 0}

    for ds_name, class_map in DATASET_CLASS_MAPS.items():
        ds_path = os.path.join(SOURCE_DIR, ds_name)
        if not os.path.exists(ds_path):
            print(f"  WARNING: {ds_name} not found — skipping")
            continue

        # Roboflow uses 'valid' not 'val'
        split_map = {"train": "train", "val": "valid", "test": "test"}

        for unified_split, src_split in split_map.items():
            img_dir = os.path.join(ds_path, src_split, "images")
            lbl_dir = os.path.join(ds_path, src_split, "labels")

            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                continue

            for img_file in os.listdir(img_dir):
                if not img_file.endswith((".jpg", ".jpeg", ".png")):
                    continue

                base      = os.path.splitext(img_file)[0]
                lbl_file  = base + ".txt"
                lbl_path  = os.path.join(lbl_dir, lbl_file)

                if not os.path.exists(lbl_path):
                    continue

                # Unique filename to avoid collisions across datasets
                unique_name = f"{ds_name}_{base}"

                # Copy image
                src_img  = os.path.join(img_dir, img_file)
                dst_img  = os.path.join(UNIFIED_DIR, unified_split,
                                        "images", unique_name + ".jpg")
                shutil.copy2(src_img, dst_img)

                # Remap and copy label
                dst_lbl = os.path.join(UNIFIED_DIR, unified_split,
                                       "labels", unique_name + ".txt")
                with open(lbl_path, "r") as f:
                    lines = f.readlines()

                remapped = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    orig_cls = int(parts[0])
                    new_cls  = class_map.get(orig_cls, orig_cls)
                    remapped.append(
                        f"{new_cls} {' '.join(parts[1:])}\n"
                    )

                with open(dst_lbl, "w") as f:
                    f.writelines(remapped)

                total[unified_split] += 1

    print(f"Unified dataset built:")
    print(f"  Train : {total['train']} images")
    print(f"  Val   : {total['val']} images")
    print(f"  Test  : {total['test']} images")

build_unified_dataset()

# ═════════════════════════════════════════════════════════════════════════════
# 4. DATASET CLASS
# ═════════════════════════════════════════════════════════════════════════════

class DetectionDataset(Dataset):
    """
    Loads images and YOLO-format labels from the unified dataset.
    Returns resized image tensor and list of [class, cx, cy, w, h] boxes.
    """
    def __init__(self, split, img_size=IMAGE_SIZE, augment=False):
        self.img_size = img_size
        self.augment  = augment
        self.img_dir  = os.path.join(UNIFIED_DIR, split, "images")
        self.lbl_dir  = os.path.join(UNIFIED_DIR, split, "labels")

        self.samples = []
        for img_file in sorted(os.listdir(self.img_dir)):
            if not img_file.endswith((".jpg", ".jpeg", ".png")):
                continue
            base     = os.path.splitext(img_file)[0]
            lbl_path = os.path.join(self.lbl_dir, base + ".txt")
            if os.path.exists(lbl_path):
                self.samples.append((
                    os.path.join(self.img_dir, img_file),
                    lbl_path
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                img = cv2.flip(img, 1)   # horizontal flip

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / \
              np.array([0.229, 0.224, 0.225])
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # Load labels
        boxes = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append([int(cls), cx, cy, w, h])

        return img_tensor, boxes

def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets)

# ═════════════════════════════════════════════════════════════════════════════
# 5. MODEL — MambaVision_S + FPN + Detection Head
# ═════════════════════════════════════════════════════════════════════════════

class FPNNeck(nn.Module):
    """
    Feature Pyramid Network neck.
    Takes P3 (192ch), P4 (384ch), P5 (768ch) from MambaVision_S levels.
    Outputs three feature maps with 256 channels each.
    """
    def __init__(self, in_channels=(192, 384, 768), out_channels=256):
        super().__init__()
        # Lateral connections
        self.lat3 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lat5 = nn.Conv2d(in_channels[2], out_channels, 1)

        # Output convolutions
        self.out3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.out5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, p3, p4, p5):
        # Top-down pathway
        l5 = self.lat5(p5)
        l4 = self.lat4(p4) + F.interpolate(
            l5, size=p4.shape[-2:], mode="nearest")
        l3 = self.lat3(p3) + F.interpolate(
            l4, size=p3.shape[-2:], mode="nearest")
        return self.out3(l3), self.out4(l4), self.out5(l5)


class DetectionHead(nn.Module):
    """
    Anchor-based detection head for one scale.
    Predicts [objectness, cx, cy, w, h, class_0..class_10] per anchor.
    """
    def __init__(self, in_channels=256, num_anchors=3,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(in_channels, out_ch, 1)

    def forward(self, x):
        return self.pred(self.conv(x))


class MambaVisionDetector(nn.Module):
    """
    MambaVision_S backbone + FPN neck + 3-scale anchor detection head.
    Single forward pass producing detections at P3/P4/P5.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        # Backbone — strip head and norm, keep levels
        backbone = models.mamba_vision_S(pretrained=True)
        self.patch_embed = backbone.patch_embed
        self.levels      = backbone.levels
        del backbone

        # FPN neck
        self.fpn = FPNNeck(
            in_channels=(192, 384, 768), out_channels=256
        )

        # Detection heads — one per scale
        self.head3 = DetectionHead(256, num_anchors=3,
                                   num_classes=num_classes)
        self.head4 = DetectionHead(256, num_anchors=3,
                                   num_classes=num_classes)
        self.head5 = DetectionHead(256, num_anchors=3,
                                   num_classes=num_classes)

    def forward(self, x):
        # Backbone feature extraction
        x = self.patch_embed(x)
        p3 = self.levels[0](x)    # (B, 192, H/8, W/8)
        p4 = self.levels[1](p3)   # (B, 384, H/16, W/16)
        p5 = self.levels[2](p4)   # (B, 768, H/32, W/32)
        p5 = self.levels[3](p5)   # final level

        # FPN
        f3, f4, f5 = self.fpn(p3, p4, p5)

        # Detection heads
        out3 = self.head3(f3)  # (B, 3*(5+11), H/8, W/8)
        out4 = self.head4(f4)  # (B, 3*(5+11), H/16, W/16)
        out5 = self.head5(f5)  # (B, 3*(5+11), H/32, W/32)

        return out3, out4, out5

# ═════════════════════════════════════════════════════════════════════════════
# 6. LOSS FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [cx, cy, w, h] format."""
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area    = box1[2] * box1[3]
    b2_area    = box2[2] * box2[3]
    union_area = b1_area + b2_area - inter_area + 1e-6

    return inter_area / union_area


class YOLOLoss(nn.Module):
    """
    YOLO-style detection loss combining:
    - BCE objectness loss
    - MSE box regression loss
    - BCE classification loss
    """
    def __init__(self, anchors, stride, num_classes=NUM_CLASSES,
                 lambda_obj=1.0, lambda_noobj=0.5,
                 lambda_box=5.0, lambda_cls=1.0):
        super().__init__()
        self.anchors     = anchors
        self.stride      = stride
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.lambda_obj   = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_box   = lambda_box
        self.lambda_cls   = lambda_cls

        self.bce     = nn.BCEWithLogitsLoss()
        self.mse     = nn.MSELoss()

    def forward(self, pred, targets, img_size):
        B, _, H, W = pred.shape
        A  = self.num_anchors
        NC = self.num_classes

        # Reshape: (B, A, H, W, 5+NC)
        pred = pred.view(B, A, 5 + NC, H, W).permute(0, 1, 3, 4, 2)

        # Build target tensors
        obj_mask    = torch.zeros(B, A, H, W, device=pred.device)
        noobj_mask  = torch.ones(B, A, H, W, device=pred.device)
        tx          = torch.zeros(B, A, H, W, device=pred.device)
        ty          = torch.zeros(B, A, H, W, device=pred.device)
        tw          = torch.zeros(B, A, H, W, device=pred.device)
        th          = torch.zeros(B, A, H, W, device=pred.device)
        tcls        = torch.zeros(B, A, H, W, NC, device=pred.device)

        for b_idx, boxes in enumerate(targets):
            for box in boxes:
                cls, cx, cy, w, h = box
                gx = cx * W
                gy = cy * H
                gi = int(gx)
                gj = int(gy)

                if gi >= W:
                    gi = W - 1
                if gj >= H:
                    gj = H - 1

                # Find best anchor
                box_wh = [w * img_size, h * img_size]
                best_iou = 0
                best_a   = 0
                for a_idx, (aw, ah) in enumerate(self.anchors):
                    inter = min(box_wh[0], aw) * min(box_wh[1], ah)
                    union = box_wh[0]*box_wh[1] + aw*ah - inter + 1e-6
                    iou   = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_a   = a_idx

                obj_mask  [b_idx, best_a, gj, gi] = 1
                noobj_mask[b_idx, best_a, gj, gi] = 0
                tx[b_idx, best_a, gj, gi] = gx - gi
                ty[b_idx, best_a, gj, gi] = gy - gj
                tw[b_idx, best_a, gj, gi] = math.log(
                    w * img_size / self.anchors[best_a][0] + 1e-6)
                th[b_idx, best_a, gj, gi] = math.log(
                    h * img_size / self.anchors[best_a][1] + 1e-6)
                tcls[b_idx, best_a, gj, gi, int(cls)] = 1.0

        # Predictions
        pred_obj  = pred[..., 4]
        pred_box  = pred[..., :4]
        pred_cls  = pred[..., 5:]

        # Losses
        loss_obj   = self.lambda_obj * self.bce(
            pred_obj[obj_mask == 1],
            obj_mask[obj_mask == 1])
        loss_noobj = self.lambda_noobj * self.bce(
            pred_obj[noobj_mask == 1],
            noobj_mask[noobj_mask == 1] * 0)

        if obj_mask.sum() > 0:
            loss_box = self.lambda_box * (
                self.mse(pred_box[..., 0][obj_mask == 1],
                         tx[obj_mask == 1]) +
                self.mse(pred_box[..., 1][obj_mask == 1],
                         ty[obj_mask == 1]) +
                self.mse(pred_box[..., 2][obj_mask == 1],
                         tw[obj_mask == 1]) +
                self.mse(pred_box[..., 3][obj_mask == 1],
                         th[obj_mask == 1])
            )
            loss_cls = self.lambda_cls * self.bce(
                pred_cls[obj_mask == 1],
                tcls[obj_mask == 1])
        else:
            loss_box = torch.tensor(0.0, device=pred.device)
            loss_cls = torch.tensor(0.0, device=pred.device)

        return loss_obj + loss_noobj + loss_box + loss_cls

# ═════════════════════════════════════════════════════════════════════════════
# 7. mAP EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def decode_predictions(pred, anchors, stride, img_size,
                       conf_thresh=0.25):
    """Decode raw predictions to [conf, cx, cy, w, h, cls] boxes."""
    B, _, H, W = pred.shape
    A  = len(anchors)
    NC = NUM_CLASSES

    pred = pred.view(B, A, 5 + NC, H, W).permute(0, 1, 3, 4, 2)
    pred = pred.sigmoid()

    boxes_all = []
    for b in range(B):
        boxes = []
        for a_idx in range(A):
            aw, ah = anchors[a_idx]
            for gj in range(H):
                for gi in range(W):
                    conf = pred[b, a_idx, gj, gi, 4].item()
                    if conf < conf_thresh:
                        continue
                    cx = (gi + pred[b, a_idx, gj, gi, 0].item()) / W
                    cy = (gj + pred[b, a_idx, gj, gi, 1].item()) / H
                    w  = (aw * torch.exp(
                        pred[b, a_idx, gj, gi, 2]).item()) / img_size
                    h  = (ah * torch.exp(
                        pred[b, a_idx, gj, gi, 3]).item()) / img_size
                    cls_scores = pred[b, a_idx, gj, gi, 5:].cpu().numpy()
                    cls_id     = int(np.argmax(cls_scores))
                    cls_conf   = conf * cls_scores[cls_id]
                    boxes.append([cls_conf, cx, cy, w, h, cls_id])
        boxes_all.append(boxes)
    return boxes_all


def compute_map50(model, loader, anchors_list, strides,
                  img_size=IMAGE_SIZE, conf_thresh=0.25,
                  iou_thresh=0.5):
    """Compute mAP50 across all classes."""
    model.eval()
    all_preds  = {c: [] for c in range(NUM_CLASSES)}
    all_labels = {c: [] for c in range(NUM_CLASSES)}

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outs = model(imgs)

            for scale_idx, (out, anchors, stride) in enumerate(
                    zip(outs, anchors_list, strides)):
                preds = decode_predictions(
                    out, anchors, stride, img_size, conf_thresh)

                for b_idx, (pred_boxes, gt_boxes) in enumerate(
                        zip(preds, targets)):
                    for cls in range(NUM_CLASSES):
                        gt  = [box for box in gt_boxes
                               if int(box[0]) == cls]
                        det = [box for box in pred_boxes
                               if int(box[5]) == cls]

                        det_sorted = sorted(
                            det, key=lambda x: x[0], reverse=True)

                        matched = [False] * len(gt)
                        for d in det_sorted:
                            best_iou  = 0
                            best_gt   = -1
                            for g_idx, g in enumerate(gt):
                                if matched[g_idx]:
                                    continue
                                iou = compute_iou(d[1:5], g[1:5])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt  = g_idx
                            if best_iou >= iou_thresh and best_gt >= 0:
                                all_preds[cls].append(
                                    (d[0], 1))
                                matched[best_gt] = True
                            else:
                                all_preds[cls].append(
                                    (d[0], 0))

                        all_labels[cls].append(len(gt))

    # Compute AP per class
    aps = []
    for cls in range(NUM_CLASSES):
        preds_cls = sorted(
            all_preds[cls], key=lambda x: x[0], reverse=True)
        n_gt = sum(all_labels[cls])
        if n_gt == 0:
            continue

        tp = 0
        fp = 0
        precisions = []
        recalls    = []

        for conf, is_tp in preds_cls:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / (tp + fp))
            recalls.append(tp / n_gt)

        # Area under PR curve
        ap = 0.0
        for i in range(1, len(precisions)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        aps.append(ap)

    return np.mean(aps) * 100 if aps else 0.0

# ═════════════════════════════════════════════════════════════════════════════
# 8. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading detection datasets...")
train_dataset = DetectionDataset("train", augment=True)
val_dataset   = DetectionDataset("val",   augment=False)
test_dataset  = DetectionDataset("test",  augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4,
                          collate_fn=collate_fn, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4,
                          collate_fn=collate_fn, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4,
                          collate_fn=collate_fn, pin_memory=True)

print(f"Train : {len(train_dataset)} images")
print(f"Val   : {len(val_dataset)} images")
print(f"Test  : {len(test_dataset)} images")

# ═════════════════════════════════════════════════════════════════════════════
# 9. MODEL, OPTIMIZER, SCHEDULER, LOSS
# ═════════════════════════════════════════════════════════════════════════════

print("\nLoading MambaVision_S detector...")
model = MambaVisionDetector(num_classes=NUM_CLASSES).to(device)
print("Model loaded!")

# Separate LR for backbone vs head
backbone_params = list(model.patch_embed.parameters()) + \
                  list(model.levels.parameters())
head_params     = list(model.fpn.parameters()) + \
                  list(model.head3.parameters()) + \
                  list(model.head4.parameters()) + \
                  list(model.head5.parameters())

optimizer = optim.AdamW([
    {"params": backbone_params, "lr": LR * 0.1},
    {"params": head_params,     "lr": LR},
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)

# Loss functions per scale
loss_fn3 = YOLOLoss(ANCHORS[0], STRIDES[0]).to(device)
loss_fn4 = YOLOLoss(ANCHORS[1], STRIDES[1]).to(device)
loss_fn5 = YOLOLoss(ANCHORS[2], STRIDES[2]).to(device)

# ═════════════════════════════════════════════════════════════════════════════
# 10. TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

print(f"\nStarting MambaVision detection training for {NUM_EPOCHS} epochs...")
print("=" * 65)

train_losses  = []
val_maps      = []
epoch_times   = []
best_map      = 0.0
best_epoch    = 0
patience_cnt  = 0

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # Warmup LR
    if epoch <= WARMUP_EPOCHS:
        warmup_factor = epoch / WARMUP_EPOCHS
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr"] * warmup_factor

    # ── Train ────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()

        out3, out4, out5 = model(imgs)

        l3 = loss_fn3(out3, targets, IMAGE_SIZE)
        l4 = loss_fn4(out4, targets, IMAGE_SIZE)
        l5 = loss_fn5(out5, targets, IMAGE_SIZE)
        loss = l3 + l4 + l5

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        running_loss += loss.item()

    if epoch > WARMUP_EPOCHS:
        scheduler.step()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # ── Validate ─────────────────────────────────────────────────────────
    val_map = compute_map50(model, val_loader, ANCHORS, STRIDES)
    val_maps.append(val_map)

    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    peak_mem = torch.cuda.max_memory_allocated(0) / 1e9

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Val mAP50: {val_map:.2f}% | "
          f"Time: {epoch_time:.1f}s | "
          f"Peak GPU: {peak_mem:.2f}GB")

    if val_map > best_map:
        best_map   = val_map
        best_epoch = epoch
        patience_cnt = 0
        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR,
                                "mambavision_detection_best.pth"))
        print(f"  *** New best mAP50: {best_map:.2f}% — saved ***")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

print("=" * 65)
print(f"Training complete! Best mAP50: {best_map:.2f}% at epoch {best_epoch}")

# ═════════════════════════════════════════════════════════════════════════════
# 11. TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

print("\nEvaluating on test set...")
model.load_state_dict(
    torch.load(os.path.join(RESULTS_DIR,
                            "mambavision_detection_best.pth"),
               map_location=device))

test_map = compute_map50(model, test_loader, ANCHORS, STRIDES)
print(f"Test mAP50: {test_map:.2f}%")
print(f"\nYOLOv8 Phase 6 comparison:")
print(f"  YOLOv8s Phase 6     : 95.3% mAP50")
print(f"  MambaVision_S Det   : {best_map:.2f}% val mAP50 / {test_map:.2f}% test mAP50")

# ═════════════════════════════════════════════════════════════════════════════
# 12. SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════

results = {
    "model"                   : "MambaVisionDetector",
    "backbone"                : "MambaVision_S",
    "neck"                    : "FPN (192→256, 384→256, 768→256)",
    "head"                    : "Anchor-based 3-scale detection head",
    "num_classes"             : NUM_CLASSES,
    "image_size"              : IMAGE_SIZE,
    "num_epochs_trained"      : epoch,
    "best_val_map50"          : round(best_map, 2),
    "best_epoch"              : best_epoch,
    "test_map50"              : round(test_map, 2),
    "avg_epoch_time_seconds"  : round(np.mean(epoch_times), 2),
    "yolov8_phase6_map50"     : 95.3,
    "training_curves"         : {
        "epochs"       : list(range(1, len(train_losses) + 1)),
        "train_losses" : [round(x, 6) for x in train_losses],
        "val_maps"     : [round(x, 4) for x in val_maps],
    }
}

with open(os.path.join(RESULTS_DIR,
                       "mambavision_detection_training_results.json"),
          "w") as f:
    json.dump(results, f, indent=2)

# ═════════════════════════════════════════════════════════════════════════════
# 13. TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════════════

epochs_ran = list(range(1, len(train_losses) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MambaVision_S Detection Training\n"
             "MambaVision_S + FPN + Anchor Head vs YOLOv8s Phase 6",
             fontsize=12, fontweight="bold")
fig.patch.set_facecolor("white")

axes[0].plot(epochs_ran, train_losses, color="#4C72B0",
             linewidth=2, marker="o", markersize=3, label="Train Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

axes[1].plot(epochs_ran, val_maps, color="#2E7D5E",
             linewidth=2, marker="o", markersize=3,
             label="MambaVision_S mAP50")
axes[1].axhline(y=95.3, color="#E07B39", linestyle="--",
                linewidth=1.5, label="YOLOv8 Phase 6: 95.3% mAP50")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mAP50 (%)")
axes[1].set_title("Val mAP50 vs YOLOv8 Phase 6 Baseline")
axes[1].set_ylim([0, 100])
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,
                         "mambavision_detection_training_curves.png"),
            dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nResults saved → {RESULTS_DIR}/")
print("Done! Ready for 09_detection_evaluation.py")