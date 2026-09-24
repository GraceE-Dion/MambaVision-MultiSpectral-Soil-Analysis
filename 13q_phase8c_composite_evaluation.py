"""
13q_phase8c_composite_evaluation.py
====================================
Phase 8C composite evaluation. Runs the four frozen models (MambaVision_S
full-image classifier; YOLOv8s-pretrained, YOLOv8s-scratch, and
Mamba-YOLO-T-scratch detectors, all unchanged since Phase 8B) against the
1760 frozen Phase 8C composites (conditions A/B/C) plus a matched clean
baseline, and writes raw, non-aggregated, per-observation prediction rows.

This script implements the locked design in
`phase8c_evaluation_design_review.md` exactly. It does NOT compute
confidence intervals, does NOT average across donors, does NOT exclude any
composite based on repair-quality metadata, and does NOT subtract Condition
A as a noise floor -- all of that is explicitly out of scope here and
belongs to 13q's raw output plus Phase 8E.

`phase8b_evaluate.py` is untouched. This is a new script.

------------------------------------------------------------------------
IMPORTANT -- integration points, resolved against the actual
phase8b_evaluate.py where possible, flagged where not:

  * CONF_THRESH (0.001) / IOU_NMS_THRESH (0.7) / DETECTOR_IMGSZ (640) /
    HALF (False) are now sourced directly from phase8b_evaluate.py:
    evaluate_accuracy() calls model.val(data=..., split=..., half=...)
    with NO explicit conf/iou override, so these are Ultralytics' own
    .val()-mode defaults, which is what 8B actually measured against.
    13q's own mAP computation (run_condition_map) calls model.val() the
    same bare way 8B did, for the same reason. The matched-prediction
    step (detector_predict, used for localization retention / donor
    attraction / end-to-end accuracy, NOT for AP) has no 8B predict-time
    precedent to copy -- 8B never called .predict() -- so it explicitly
    passes the same values as a documented, sourced choice rather than a
    second silent default.
  * Mamba-YOLO-T's `.predict()` / `.val()` call signature is assumed here
    to be Ultralytics-API-compatible (mirroring the two YOLOv8s variants and
    phase8b_evaluate.py's own model = YOLO(args.weights) pattern, which it
    used for all three detectors including Mamba-YOLO-T). If the actual
    Mamba-YOLO-T checkpoint requires a different invocation, only
    `load_detector()` needs to change -- the rest of the pipeline
    (matching, metrics, checkpointing) is model-agnostic.
  * MambaVision_S loading follows the project's locked convention: import
    via `sys.path.insert(0, MAMBAVISION_REPO_PATH)` then
    `from mambavision import models`, NOT `timm.create_model(...)`, and
    `torch.backends.cudnn.enabled = False` (cuDNN is incompatible with
    mamba-ssm 2.2.2 on the MTSU Lambda cluster). The `hf_to_correct` class
    index remapping is applied to both ground truth and argmax predictions,
    per the project's documented known-limitations history.
  * LABEL_DIR_BASE, CLASSIFIER_DATA_DIR, and the classifier checkpoint path
    are now sourced directly from 12_background_roi_experiment.py /
    13_confound_characterization.py / 05_training.py, not placeholders --
    see the CONFIG section below for exactly which script each came from.
------------------------------------------------------------------------

Usage:
    conda activate mambavision
    python 13q_phase8c_composite_evaluation.py --model classifier

    conda activate mambayolo
    python 13q_phase8c_composite_evaluation.py --model yolo_pretrained
    python 13q_phase8c_composite_evaluation.py --model yolo_scratch
    python 13q_phase8c_composite_evaluation.py --model mamba_yolo

    # Re-running with the same --model resumes from the existing
    # results/phase8c/eval_raw/{model}.jsonl checkpoint rather than
    # restarting.

Output:
    results/phase8c/eval_raw/{model}.jsonl            -- raw per-row output
    results/phase8c/eval_raw/{model}_detections.jsonl -- full post-NMS
                                                          detection sets
                                                          (detectors only)
    results/phase8c/eval_raw/{model}_map/              -- per-condition
                                                          Ultralytics val()
                                                          run artifacts
                                                          (detectors only)
    results/phase8c/eval_summary/{model}_summary.json  -- descriptive
                                                          metric rollup
                                                          (Section: Metrics)
"""

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

PHASE8C_DIR = "results/phase8c/full"                  # frozen 13e output
METADATA_PATH = os.path.join(PHASE8C_DIR, "composite_metadata.json")

# No separate clean-target-index file is maintained anywhere in this
# pipeline -- 12_background_roi_experiment.py and 13_confound_
# characterization.py both derive bbox lookups the same way at runtime
# (build_label_lookup(), keyed by filename stem across each source
# dataset's labels/ directory). The clean-target index below is built the
# same way, sourcing the target list itself from composite_metadata.json
# (which already carries target_id/target_source/target_class/
# target_modality for every target) rather than a hand-maintained file.
#
# SOURCED (not guessed) from 12_background_roi_experiment.py and
# 13_confound_characterization.py: LABEL_DIR_BASE is the raw per-source
# Roboflow export root -- each project folder under it has
# <split>/images/ and <split>/labels/ as SIBLINGS (confirmed in
# 02_data_preparation.py's img_src/lbl_src construction), split in
# {"train", "valid", "test"}. composite_metadata.json's target_id stems
# (e.g. "Soil-Moisture-v4-IR-1_48_png.rf.<hash>") match this raw
# directory's filenames directly -- the Master_Soil_Moisture classifier-
# training copy renames/prefixes files on merge (02_data_preparation.py),
# so it is NOT where target images are resolved from; only LABEL_DIR_BASE
# is used for both the image and the label lookup below.
LABEL_DIR_BASE = "/data/Grace/soil-moisture-dataset"

# Classifier's own training data root (DIFFERENT from LABEL_DIR_BASE --
# this is the merged, class-organized copy used only to derive
# hf_to_correct, matching 05_training.py / 12_background_roi_experiment.py
# / 13_confound_characterization.py exactly).
CLASSIFIER_DATA_DIR = "/data/Grace/Master_Soil_Moisture"

RESULTS_DIR = "results/phase8c/eval_raw"
SUMMARY_DIR = "results/phase8c/eval_summary"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

NUM_CLASSES = 11
IMAGE_SIZE = 224

# --- Detector confidence/NMS settings -- SOURCED FROM phase8b_evaluate.py --
# phase8b_evaluate.py's evaluate_accuracy() calls model.val(data=..., split=
# ..., half=...) with NO explicit conf/iou override -- it relies entirely on
# Ultralytics' own internal .val() defaults (conf~0.001, iou=0.7 NMS). 8B
# never called .predict() at all (only .val() for accuracy + a separate
# latency benchmark on raw torch forward passes), so there is no 8B
# predict-time conf/iou to copy -- these ARE the Ultralytics framework
# defaults 8B implicitly used, not a guess. half also mirrors 8B's own
# default (False / FP32).
CONF_THRESH = 0.001
IOU_NMS_THRESH = 0.7
MAX_DET = 300
HALF = False
DETECTOR_IMGSZ = 640   # matches phase8b_evaluate.py's --imgsz default (YOLO input size; classifier IMAGE_SIZE=224 is unrelated)

# --- Localization thresholds ------------------------------------------------
IOU_PRIMARY = 0.50    # primary localization-retention / conditional-accuracy gate
IOU_SECONDARY = 0.75   # secondary localization-retention read only

MAMBAVISION_REPO_PATH = "/data/Grace/MambaVision"

MODEL_CONFIG = {
    "classifier": {
        "kind": "classifier",
        # Sourced from 12_background_roi_experiment.py / 13_confound_
        # characterization.py's own MODEL_PATH (13_'s own header still
        # says "CONFIRM" against the actual checkpoint filename on disk --
        # carried over here for the same reason).
        "checkpoint": "./results/mambavision_fullimage_best_model.pth",
    },
    "yolo_pretrained": {
        "kind": "detector",
        "checkpoint": "results/phase8b/yolov8s_pretrained_best.pt",
    },
    "yolo_scratch": {
        "kind": "detector",
        "checkpoint": "results/phase8b/yolov8s_scratch_best.pt",
    },
    "mamba_yolo": {
        "kind": "detector",
        "checkpoint": "results/phase8b/mamba_yolo_t_scratch_best.pt",
    },
}

IDENTITY_FIELDS = [
    "target_id", "target_class", "target_source", "target_modality",
    "donor_id", "donor_class", "donor_source", "donor_modality",
    "condition", "donor_rep", "same_modality",
    "repair_tier", "repair_method", "repair_candidate_count",
    "repair_candidate_support", "weak_candidate_support",
    "repair_selected_score", "repair_shrink_fraction_used",
]


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING / ELIGIBLE-TARGET BOOKKEEPING
# ═════════════════════════════════════════════════════════════════════════════

def load_composite_records():
    with open(METADATA_PATH) as f:
        records = json.load(f)
    return records


def build_label_lookup():
    """Mirrors 12_background_roi_experiment.py / 13_confound_characterization
    .py's build_label_lookup() exactly: scans every source dataset's
    labels/ directory once, keyed by filename stem, to a normalized
    (cx, cy, w, h) YOLO box. Reused here rather than re-implemented
    differently, since this project's convention is one label-lookup
    routine shared across scripts, not a per-script reimplementation."""
    lookup = {}
    for ds_name in os.listdir(LABEL_DIR_BASE):
        ds_path = os.path.join(LABEL_DIR_BASE, ds_name)
        if not os.path.isdir(ds_path):
            continue
        for split in ["train", "valid", "test"]:
            lbl_dir = os.path.join(ds_path, split, "labels")
            if not os.path.exists(lbl_dir):
                continue
            for lbl_file in os.listdir(lbl_dir):
                if not lbl_file.endswith(".txt"):
                    continue
                with open(os.path.join(lbl_dir, lbl_file)) as f:
                    lines = f.readlines()
                if not lines:
                    continue
                parts = lines[0].strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                stem = os.path.splitext(lbl_file)[0]
                lookup[stem] = (cx, cy, w, h)
    print(f"Loaded {len(lookup)} bounding box labels")
    return lookup


def build_image_lookup():
    """Mirrors build_label_lookup() exactly, but for the sibling images/
    directories under the same LABEL_DIR_BASE project folders, keyed by
    filename stem -> full path. Since composite_metadata.json's target_id
    stems already match this raw directory's filenames directly (see
    CONFIG comment above), no resolve_stem fallback tiering is needed
    here -- that fallback exists elsewhere only because Master_Soil_
    Moisture's merged/renamed copies don't share a stem with the raw
    export, which is not the lookup being done here."""
    lookup = {}
    for ds_name in os.listdir(LABEL_DIR_BASE):
        ds_path = os.path.join(LABEL_DIR_BASE, ds_name)
        if not os.path.isdir(ds_path):
            continue
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(ds_path, split, "images")
            if not os.path.isdir(img_dir):
                continue
            for img_file in os.listdir(img_dir):
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem = os.path.splitext(img_file)[0]
                lookup[stem] = os.path.join(img_dir, img_file)
    print(f"Loaded {len(lookup)} source images from {LABEL_DIR_BASE}")
    return lookup


def normalized_bbox_to_pixels(cx, cy, w, h, img_w, img_h):
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    x_max = (cx + w / 2) * img_w
    y_max = (cy + h / 2) * img_h
    return [x_min, y_min, x_max, y_max]


def build_clean_target_index(records):
    """Derives target_id -> {path, class, source, modality, bbox} directly
    from composite_metadata.json's own target fields (already mechanically
    verified by the Finding-2 integrity check) plus the shared
    build_label_lookup() convention, instead of depending on a separate
    hand-maintained index file that doesn't exist anywhere else in this
    pipeline."""
    from PIL import Image as PILImage

    targets = {}
    for r in records:
        tid = r["target_id"]
        if tid in targets:
            continue
        targets[tid] = {
            "class": r["target_class"], "source": r["target_source"],
            "modality": r.get("target_modality"),
        }

    label_lookup = build_label_lookup()
    image_lookup = build_image_lookup()
    index = {}
    missing = []
    for tid, meta in targets.items():
        path = image_lookup.get(tid)
        if path is None:
            missing.append(tid)
            continue
        meta["path"] = path
        bbox = None
        if tid in label_lookup:
            with PILImage.open(path) as im:
                img_w, img_h = im.size
            cx, cy, w, h = label_lookup[tid]
            bbox = normalized_bbox_to_pixels(cx, cy, w, h, img_w, img_h)
        meta["bbox"] = bbox
        index[tid] = meta

    if missing:
        print(f"WARNING: {len(missing)} targets from composite_metadata.json "
              f"had no direct-stem match in {LABEL_DIR_BASE}'s images/ dirs "
              f"(first few: {missing[:5]}). Since composite_metadata.json's "
              f"target_id stems are expected to come directly from this same "
              f"raw directory (per the composite generator), a nonzero count "
              f"here is worth investigating rather than assuming -- e.g. "
              f"confirm LABEL_DIR_BASE is the exact root the 13e generator "
              f"itself read from, on this cluster.")
    print(f"Clean target index: {len(index)}/{len(targets)} targets resolved "
          f"({sum(1 for v in index.values() if v['bbox'] is not None)} with a bbox).")
    return index


def eligible_targets_by_condition(records):
    """condition -> sorted list of unique target_ids appearing under that
    condition in the frozen composite set. Used only to derive the correct
    matched subset for each contrast, never to filter which composites get
    evaluated (all 1760 are evaluated)."""
    by_cond = {"A": set(), "B": set(), "C": set()}
    for r in records:
        by_cond[r["condition"]].add(r["target_id"])
    return {k: sorted(v) for k, v in by_cond.items()}


def identity_hash(record_or_row):
    """Hash of the pass-through identity fields, used as part of the resume
    safeguard so a stale checkpoint row is never trusted just because its
    composite_id matches -- the underlying dataset must match too."""
    payload = json.dumps(
        {k: record_or_row.get(k) for k in IDENTITY_FIELDS},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINTED JSONL WRITER / RESUME
# ═════════════════════════════════════════════════════════════════════════════

class CheckpointedWriter:
    """Appends JSON rows to a .jsonl file and tracks which (model_id,
    composite_id) + identity_hash combinations are already present, so a
    restarted run skips completed work but never trusts a row whose
    identity fields don't match the current dataset."""

    def __init__(self, path, model_id):
        self.path = path
        self.model_id = model_id
        self.done = {}  # composite_id -> identity_hash
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("model_id") == model_id:
                        self.done[row["composite_id"]] = row.get("_identity_hash")
        self._fh = open(path, "a")

    def already_done(self, composite_id, expected_hash):
        stored_hash = self.done.get(composite_id)
        if stored_hash is None:
            return False
        if stored_hash != expected_hash:
            # Stale/mismatched row from a different dataset version --
            # treat as not done, re-run, and the new row (appended) will
            # simply co-exist; downstream loading always keeps the LAST
            # matching row per (model_id, composite_id) for this reason.
            return False
        return True

    def write(self, row):
        row = dict(row)
        row["model_id"] = self.model_id
        self._fh.write(json.dumps(row, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.done[row["composite_id"]] = row.get("_identity_hash")

    def close(self):
        self._fh.close()


# ═════════════════════════════════════════════════════════════════════════════
# 4. CLASSIFIER (MambaVision_S) INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def load_classifier():
    import torch
    import torch.nn as nn

    torch.backends.cudnn.enabled = False  # required: cuDNN incompatible with mamba-ssm 2.2.2

    sys.path.insert(0, MAMBAVISION_REPO_PATH)
    from mambavision import models  # NVlabs fork only; timm.create_model raises on this cluster

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = models.mamba_vision_S(pretrained=False)
    in_features = backbone.head.in_features
    backbone.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    ckpt_path = MODEL_CONFIG["classifier"]["checkpoint"]
    state = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(state)
    backbone.to(device).eval()
    return backbone, device


def classifier_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def classifier_predict(model, device, transform, image_path, hf_to_correct):
    import torch
    from PIL import Image as PILImage

    img = PILImage.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    raw_pred_idx = int(np.argmax(probs))
    pred_class = hf_to_correct.get(raw_pred_idx, raw_pred_idx)  # remap per project's known-limitations fix
    confidence = float(probs[raw_pred_idx])

    # Remap the whole probability vector into corrected class order for
    # downstream comparability.
    remapped_probs = [0.0] * NUM_CLASSES
    for raw_idx, p in enumerate(probs):
        remapped_probs[hf_to_correct.get(raw_idx, raw_idx)] = float(p)

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "prob_vector": remapped_probs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. DETECTOR (YOLOv8s x2, Mamba-YOLO-T) INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def load_detector(model_name):
    """Returns an Ultralytics-API-compatible model object with .predict()
    and .val(). See module docstring: Mamba-YOLO-T is assumed compatible
    with this interface, matching how it was trained/evaluated in 8B."""
    from ultralytics import YOLO
    ckpt_path = MODEL_CONFIG[model_name]["checkpoint"]
    return YOLO(ckpt_path)


def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detector_predict(model, image_path, gt_bbox, gt_class):
    """Runs inference at the locked confidence/NMS settings, persists the
    FULL post-NMS detection set (for AP reconstruction / audit), and
    separately derives the single deterministic matched prediction (highest
    IoU with GT, never highest confidence) used for every metric other than
    AP itself."""
    results = model.predict(
        source=image_path, conf=CONF_THRESH, iou=IOU_NMS_THRESH,
        max_det=MAX_DET, imgsz=DETECTOR_IMGSZ, half=HALF, verbose=False,
    )[0]

    all_detections = []
    boxes = results.boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].tolist()
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        all_detections.append({"bbox": xyxy, "pred_class": cls, "confidence": conf})

    num_detections = len(all_detections)

    matched = None
    best_iou = 0.0
    if gt_bbox is not None:
        for det in all_detections:
            iou = iou_xyxy(det["bbox"], gt_bbox)
            if iou > best_iou:
                best_iou = iou
                matched = det

    if matched is None:
        matched_out = {
            "matched_pred_class": None, "matched_confidence": None,
            "matched_bbox": None, "matched_iou": 0.0,
            "localized_iou50": False, "localized_iou75": False,
        }
    else:
        matched_out = {
            "matched_pred_class": matched["pred_class"],
            "matched_confidence": matched["confidence"],
            "matched_bbox": matched["bbox"],
            "matched_iou": best_iou,
            "localized_iou50": best_iou >= IOU_PRIMARY,
            "localized_iou75": best_iou >= IOU_SECONDARY,
        }

    matched_out["num_detections"] = num_detections
    return matched_out, all_detections


def build_condition_yolo_dataset(model_name, condition_label, image_label_pairs, dest_root):
    """Materializes a minimal Ultralytics-format image/label set for one
    (model, condition) pair so mAP50/mAP50-95 can be computed via the same
    val() machinery used in 8B -- NOT reconstructed from the single
    matched-prediction rows. image_label_pairs: list of (image_path,
    yolo_label_lines)."""
    img_dir = os.path.join(dest_root, "images")
    lbl_dir = os.path.join(dest_root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for idx, (image_path, label_lines) in enumerate(image_label_pairs):
        ext = os.path.splitext(image_path)[1]
        stem = f"{idx:05d}"
        link_path = os.path.join(img_dir, stem + ext)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(image_path), link_path)
        with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(label_lines))

    yaml_path = os.path.join(dest_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(dest_root)}\n")
        f.write("train: images\nval: images\n")
        f.write(f"nc: {NUM_CLASSES}\n")
        f.write("names: " + json.dumps([str(i) for i in range(NUM_CLASSES)]) + "\n")
    return yaml_path


def run_condition_map(model, model_name, condition_label, image_label_pairs):
    """Computes mAP50/mAP50-95 for one condition via Ultralytics' own
    validator -- the AP-implementation requirement from peer review.
    Deliberately calls model.val() the SAME way phase8b_evaluate.py's
    evaluate_accuracy() does: no explicit conf/iou override, so Ultralytics'
    internal .val() defaults apply exactly as they did in 8B (those
    defaults are what CONF_THRESH/IOU_NMS_THRESH document above, but they
    are not re-passed here -- passing them explicitly could in principle
    diverge from a future Ultralytics default change 8B itself would have
    picked up silently; omitting them, like 8B did, is the more exact
    match). half also matches 8B's own default. Returns the raw ultralytics
    metrics dict plus precision/recall for parity with 8B's saved fields."""
    dest_root = os.path.join(RESULTS_DIR, f"{model_name}_map", condition_label)
    yaml_path = build_condition_yolo_dataset(model_name, condition_label, image_label_pairs, dest_root)
    metrics = model.val(data=yaml_path, split="val", half=HALF, imgsz=DETECTOR_IMGSZ)
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "n_images": len(image_label_pairs),
    }


def to_yolo_label_line(gt_class, gt_bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = gt_bbox_xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{gt_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN PER-MODEL EVALUATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def get_image_size(path):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.size  # (w, h)


def evaluate_model(model_name):
    cfg = MODEL_CONFIG[model_name]
    kind = cfg["kind"]

    records = load_composite_records()
    clean_index = build_clean_target_index(records)
    eligible = eligible_targets_by_condition(records)

    raw_path = os.path.join(RESULTS_DIR, f"{model_name}.jsonl")
    det_path = os.path.join(RESULTS_DIR, f"{model_name}_detections.jsonl")
    writer = CheckpointedWriter(raw_path, model_name)
    det_writer = CheckpointedWriter(det_path, model_name) if kind == "detector" else None

    print(f"=== Evaluating {model_name} ({kind}) ===")
    print(f"Resuming: {len(writer.done)} composite rows already present for this model.")

    if kind == "classifier":
        hf_to_correct = load_hf_to_correct_map()
        model, device = load_classifier()
        transform = classifier_transform()
    else:
        model = load_detector(model_name)

    # --- Clean baseline: once per unique target, not per condition --------
    print(f"--- Clean baseline: {len(clean_index)} unique targets ---")
    clean_pairs_for_map = []  # (image_path, gt_class, gt_bbox) for detector mAP
    for target_id, meta in clean_index.items():
        composite_id = f"clean__{target_id}"
        row_identity = {
            "target_id": target_id, "target_class": meta["class"],
            "target_source": meta["source"], "target_modality": meta.get("modality"),
            "donor_id": None, "donor_class": None, "donor_source": None,
            "donor_modality": None, "condition": "clean", "donor_rep": None,
            "same_modality": None, "repair_tier": None, "repair_method": None,
            "repair_candidate_count": None, "repair_candidate_support": None,
            "weak_candidate_support": None, "repair_selected_score": None,
            "repair_shrink_fraction_used": None,
        }
        expected_hash = identity_hash(row_identity)
        if writer.already_done(composite_id, expected_hash):
            if kind == "detector":
                clean_pairs_for_map.append((meta["path"], meta["class"], meta.get("bbox")))
            continue

        if kind == "classifier":
            pred = classifier_predict(model, device, transform, meta["path"], hf_to_correct)
            row = {
                "composite_id": composite_id, "pred_class": pred["pred_class"],
                "confidence": pred["confidence"], "prob_vector": pred["prob_vector"],
                "gt_class": meta["class"], "correct": pred["pred_class"] == meta["class"],
                **row_identity,
            }
        else:
            gt_bbox = meta.get("bbox")
            matched, all_dets = detector_predict(model, meta["path"], gt_bbox, meta["class"])
            row = {
                "composite_id": composite_id, "gt_class": meta["class"],
                "gt_bbox": gt_bbox, **matched, **row_identity,
                "conditional_correct": (
                    matched["localized_iou50"] and matched["matched_pred_class"] == meta["class"]
                ),
                "endtoend_correct": (
                    matched["localized_iou50"] and matched["matched_pred_class"] == meta["class"]
                ),
            }
            det_writer.write({
                "composite_id": composite_id, "detections": all_dets,
                "_identity_hash": expected_hash,
            })
            clean_pairs_for_map.append((meta["path"], meta["class"], gt_bbox))

        row["_identity_hash"] = expected_hash
        writer.write(row)

    # --- Composites: all 1760, conditions A/B/C ----------------------------
    print(f"--- Composites: {len(records)} rows (A/B/C) ---")
    map_pairs_by_condition = {"A": [], "B": [], "C": []}
    for i, r in enumerate(records):
        composite_id = r["composite_id"]
        row_identity = {k: r.get(k) for k in IDENTITY_FIELDS}
        expected_hash = identity_hash(row_identity)
        if writer.already_done(composite_id, expected_hash):
            if kind == "detector":
                map_pairs_by_condition[r["condition"]].append(
                    (r["composite_path"], r["target_class"], r.get("target_bbox")))
            continue

        image_path = r["composite_path"]

        if kind == "classifier":
            pred = classifier_predict(model, device, transform, image_path, hf_to_correct)
            pred_class = pred["pred_class"]
            gt_class = r["target_class"]
            misclassified = pred_class != gt_class
            row = {
                "composite_id": composite_id, "pred_class": pred_class,
                "confidence": pred["confidence"], "prob_vector": pred["prob_vector"],
                "gt_class": gt_class, "correct": not misclassified,
            }
            if r["condition"] in ("B", "C"):
                row["donor_attracted_conditional"] = (
                    misclassified and pred_class == r["donor_class"]
                )
                row["donor_attracted_unconditional"] = (pred_class == r["donor_class"])
        else:
            gt_bbox = r.get("target_bbox")  # composite's target ROI, in composite-image coordinates
            gt_class = r["target_class"]
            matched, all_dets = detector_predict(model, image_path, gt_bbox, gt_class)
            localized = matched["localized_iou50"]
            correct_class = matched["matched_pred_class"] == gt_class
            misclassified_localized = localized and not correct_class
            row = {
                "composite_id": composite_id, "gt_class": gt_class, "gt_bbox": gt_bbox,
                **matched,
                "conditional_correct": bool(localized and correct_class),
                "endtoend_correct": bool(localized and correct_class),
            }
            if r["condition"] in ("B", "C"):
                donor_attracted_cond = bool(
                    misclassified_localized and matched["matched_pred_class"] == r["donor_class"]
                )
                donor_attracted_uncond = bool(
                    localized and matched["matched_pred_class"] == r["donor_class"]
                )
                row["donor_attracted_conditional"] = donor_attracted_cond
                row["donor_attracted_unconditional"] = donor_attracted_uncond
            det_writer.write({
                "composite_id": composite_id, "detections": all_dets,
                "_identity_hash": expected_hash,
            })
            map_pairs_by_condition[r["condition"]].append((image_path, gt_class, gt_bbox))

        row.update(row_identity)
        row["_identity_hash"] = expected_hash
        writer.write(row)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(records)} composites done")

    writer.close()
    if det_writer:
        det_writer.close()

    # --- mAP50 / mAP50-95, per condition, via Ultralytics' own validator ---
    map_results = {}
    if kind == "detector":
        print("--- Computing mAP50 / mAP50-95 per condition via Ultralytics val() ---")
        map_results["clean"] = compute_map_for_pairs(model, model_name, "clean", clean_pairs_for_map)
        for cond in ("A", "B", "C"):
            map_results[cond] = compute_map_for_pairs(model, model_name, cond, map_pairs_by_condition[cond])

    write_summary(model_name, kind, eligible, map_results)
    print(f"=== {model_name} evaluation complete ===")


def compute_map_for_pairs(model, model_name, condition_label, pairs):
    if not pairs:
        return None
    image_label_pairs = []
    for image_path, gt_class, gt_bbox in pairs:
        if gt_bbox is None:
            continue
        w, h = get_image_size(image_path)
        image_label_pairs.append((image_path, [to_yolo_label_line(gt_class, gt_bbox, w, h)]))
    if not image_label_pairs:
        return None
    return run_condition_map(model, model_name, condition_label, image_label_pairs)


# ═════════════════════════════════════════════════════════════════════════════
# 7. HF-TO-CORRECT CLASS REMAPPING (project-locked convention)
# ═════════════════════════════════════════════════════════════════════════════

# Sourced directly (not guessed): identical in 05_training.py,
# 05b_training_fullimage.py, 12_background_roi_experiment.py, and
# 13_confound_characterization.py.
CLASSIFIER_TRAIN_DIR = os.path.join(CLASSIFIER_DATA_DIR, "train")


def load_hf_to_correct_map():
    """Computed at runtime, NOT loaded from a static file -- this is how
    every training/eval script in this repo does it (05_training.py,
    12_background_roi_experiment.py): HuggingFace/ImageFolder assigns
    class indices alphabetically, so Level_10 lands at index 1 instead of
    10. The fix is always: sort the train folder names, map enumeration
    index -> int(folder name). Re-deriving it here the same way, against
    the SAME train directory the classifier checkpoint was actually
    trained on, avoids the documented class-index-remap bug (9.76% vs.
    95.28% inconsistency) recurring in a new script via a stale or
    hand-copied mapping."""
    if not os.path.isdir(CLASSIFIER_TRAIN_DIR):
        print(f"WARNING: CLASSIFIER_TRAIN_DIR ({CLASSIFIER_TRAIN_DIR}) not found -- "
              f"falling back to identity mapping. This WILL reproduce the "
              f"documented class-index-remap bug if wrong. Verify this path "
              f"against whichever DATA_DIR the classifier checkpoint in "
              f"MODEL_CONFIG['classifier']['checkpoint'] was actually trained on.")
        return {i: i for i in range(NUM_CLASSES)}
    train_folders = sorted(os.listdir(CLASSIFIER_TRAIN_DIR))
    hf_to_correct = {idx: int(folder) for idx, folder in enumerate(train_folders)}
    print(f"Class remapping (derived from {CLASSIFIER_TRAIN_DIR}): {hf_to_correct}")
    return hf_to_correct


# ═════════════════════════════════════════════════════════════════════════════
# 8. DESCRIPTIVE SUMMARY (Section: Metrics -- rollup only, no CIs)
# ═════════════════════════════════════════════════════════════════════════════

def load_rows(model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}.jsonl")
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            rows[row["composite_id"]] = row  # last write wins, per resume semantics
    return list(rows.values())


def safe_rate(numer, denom):
    return {"rate": (numer / denom) if denom else None, "n": numer, "of": denom}


def write_summary(model_name, kind, eligible, map_results):
    rows = load_rows(model_name)
    clean_rows = [r for r in rows if r["condition"] == "clean"]
    cond_rows = {c: [r for r in rows if r["condition"] == c] for c in ("A", "B", "C")}

    common199 = set(eligible["A"])  # A's eligible-target subset defines the common 199

    def restrict_to_common(rows_):
        return [r for r in rows_ if r["target_id"] in common199]

    summary = {"model_id": model_name, "kind": kind, "n_clean": len(clean_rows)}

    if kind == "classifier":
        def acc(rs):
            return safe_rate(sum(r["correct"] for r in rs), len(rs))

        summary["clean_accuracy"] = acc(clean_rows)
        for c in ("A", "B", "C"):
            summary[f"{c}_accuracy"] = acc(cond_rows[c])
            if c in ("B", "C"):
                da_rows = [r for r in cond_rows[c] if not r["correct"]]
                summary[f"{c}_donor_attraction_conditional"] = safe_rate(
                    sum(r["donor_attracted_conditional"] for r in da_rows), len(da_rows))
                summary[f"{c}_donor_attraction_unconditional"] = safe_rate(
                    sum(r["donor_attracted_unconditional"] for r in cond_rows[c]), len(cond_rows[c]))
        b_common = restrict_to_common(cond_rows["B"])
        a_common = restrict_to_common(cond_rows["A"])
        summary["B_minus_A_common199"] = {
            "B_common199_accuracy": acc(b_common), "A_common199_accuracy": acc(a_common),
        }

    else:  # detector
        def cond_acc(rs):
            return safe_rate(sum(r["conditional_correct"] for r in rs if r["localized_iou50"]),
                              sum(1 for r in rs if r["localized_iou50"]))

        def endtoend(rs):
            return safe_rate(sum(r["endtoend_correct"] for r in rs), len(rs))

        def loc_retention(rs, thresh_key):
            return safe_rate(sum(1 for r in rs if r[thresh_key]), len(rs))

        summary["clean_localization_iou50"] = loc_retention(clean_rows, "localized_iou50")
        summary["clean_localization_iou75"] = loc_retention(clean_rows, "localized_iou75")
        summary["clean_conditional_accuracy"] = cond_acc(clean_rows)
        summary["clean_endtoend_accuracy"] = endtoend(clean_rows)

        for c in ("A", "B", "C"):
            summary[f"{c}_localization_iou50"] = loc_retention(cond_rows[c], "localized_iou50")
            summary[f"{c}_localization_iou75"] = loc_retention(cond_rows[c], "localized_iou75")
            summary[f"{c}_conditional_accuracy"] = cond_acc(cond_rows[c])
            summary[f"{c}_endtoend_accuracy"] = endtoend(cond_rows[c])
            if c in ("B", "C"):
                loc_misclassified = [r for r in cond_rows[c] if r["localized_iou50"] and not r["conditional_correct"]]
                summary[f"{c}_donor_attraction_conditional"] = safe_rate(
                    sum(r["donor_attracted_conditional"] for r in loc_misclassified), len(loc_misclassified))
                summary[f"{c}_donor_attraction_unconditional"] = safe_rate(
                    sum(r["donor_attracted_unconditional"] for r in cond_rows[c]), len(cond_rows[c]))

        b_common = restrict_to_common(cond_rows["B"])
        a_common = restrict_to_common(cond_rows["A"])
        summary["B_minus_A_common199"] = {
            "B_common199_endtoend_accuracy": endtoend(b_common),
            "A_common199_endtoend_accuracy": endtoend(a_common),
            "B_common199_conditional_accuracy": cond_acc(b_common),
            "A_common199_conditional_accuracy": cond_acc(a_common),
        }
        summary["mAP"] = map_results

    with open(os.path.join(SUMMARY_DIR, f"{model_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary written: {SUMMARY_DIR}/{model_name}_summary.json")


# ═════════════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    args = parser.parse_args()

    t0 = time.time()
    evaluate_model(args.model)
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
