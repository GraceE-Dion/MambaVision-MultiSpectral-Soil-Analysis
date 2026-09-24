"""
13q_preflight_check.py
=======================
Runs every currently-unverified assumption in 13q_phase8c_composite_
evaluation.py against the real cluster filesystem/environment in ONE pass,
so all the open questions get answered together instead of one at a time.

This script imports 13q_phase8c_composite_evaluation.py directly (not a
reimplementation) and exercises its actual constants and functions --
LABEL_DIR_BASE, CLASSIFIER_TRAIN_DIR, MODEL_CONFIG checkpoint paths,
build_label_lookup(), build_image_lookup(), build_clean_target_index(),
load_hf_to_correct_map(), and (env permitting) actually loading each model
checkpoint. If a check here passes, 13q's own use of the same code is
verified, not just "probably fine."

Two things genuinely need BOTH conda environments to fully verify:
  - Classifier checks (torch + mambavision import, checkpoint load) need
    the `mambavision` environment.
  - Detector checks (ultralytics import, YOLO(...) load for all three
    detector checkpoints) need the `mambayolo` environment.
Run this script once in each environment; every check gracefully SKIPs
(not FAILs) when its required package isn't importable in the current env,
so partial runs are still useful and nothing has to be reconciled by hand.

Usage:
    conda activate mambavision
    python 13q_preflight_check.py

    conda activate mambayolo
    python 13q_preflight_check.py

Exit code: 0 if no HARD FAILUREs (WARN and SKIP are both fine to proceed
past, though every WARN should be read); 1 if any HARD FAILURE was found.
Nothing in this script writes to or modifies any result/checkpoint file.
"""

import importlib.util
import json
import os
import sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
Q_SCRIPT_PATH = os.path.join(THIS_DIR, "13q_phase8c_composite_evaluation.py")

results = []  # list of (status, check_name, detail) -- status in PASS/WARN/FAIL/SKIP


def record(status, name, detail=""):
    results.append((status, name, detail))
    tag = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"{tag} {name}" + (f" -- {detail}" if detail else ""))


def try_check(name, fn):
    """Runs fn(), catches any exception as a FAIL with the traceback
    condensed to one line, so one broken check never kills the rest of
    the preflight run."""
    try:
        fn()
    except Exception as e:
        record("FAIL", name, f"{type(e).__name__}: {e}")
        if os.environ.get("PREFLIGHT_VERBOSE"):
            traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# 0. Import 13q itself (not a reimplementation)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  13q Pre-Flight Check")
print("=" * 70)

if not os.path.exists(Q_SCRIPT_PATH):
    print(f"FATAL: {Q_SCRIPT_PATH} not found. Place this script in the same "
          f"directory as 13q_phase8c_composite_evaluation.py.")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("q13", Q_SCRIPT_PATH)
q13 = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(q13)
    record("PASS", "Import 13q_phase8c_composite_evaluation.py",
           "module-level constants/functions loaded")
except Exception as e:
    print(f"FATAL: 13q itself failed to import -- fix this before anything "
          f"else. {type(e).__name__}: {e}")
    if os.environ.get("PREFLIGHT_VERBOSE"):
        traceback.print_exc()
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Path existence checks (env-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

def check_label_dir_base():
    if not os.path.isdir(q13.LABEL_DIR_BASE):
        record("FAIL", "LABEL_DIR_BASE exists", q13.LABEL_DIR_BASE)
        return
    subdirs = [d for d in os.listdir(q13.LABEL_DIR_BASE)
               if os.path.isdir(os.path.join(q13.LABEL_DIR_BASE, d))]
    record("PASS", "LABEL_DIR_BASE exists", f"{q13.LABEL_DIR_BASE} ({len(subdirs)} project folders)")


def check_classifier_dirs():
    if not os.path.isdir(q13.CLASSIFIER_DATA_DIR):
        record("FAIL", "CLASSIFIER_DATA_DIR exists", q13.CLASSIFIER_DATA_DIR)
        return
    record("PASS", "CLASSIFIER_DATA_DIR exists", q13.CLASSIFIER_DATA_DIR)
    if not os.path.isdir(q13.CLASSIFIER_TRAIN_DIR):
        record("FAIL", "CLASSIFIER_TRAIN_DIR exists", q13.CLASSIFIER_TRAIN_DIR)
        return
    folders = sorted(os.listdir(q13.CLASSIFIER_TRAIN_DIR))
    record("PASS", "CLASSIFIER_TRAIN_DIR exists", f"{q13.CLASSIFIER_TRAIN_DIR} ({len(folders)} class folders: {folders})")
    if len(folders) != q13.NUM_CLASSES:
        record("WARN", "CLASSIFIER_TRAIN_DIR class-folder count",
               f"found {len(folders)}, expected NUM_CLASSES={q13.NUM_CLASSES}")


def check_checkpoints():
    for model_name, cfg in q13.MODEL_CONFIG.items():
        path = cfg["checkpoint"]
        exists = os.path.exists(path)
        note = "" if model_name != "classifier" else "sourced from 12_/13_ MODEL_PATH (still marked CONFIRM there)"
        detector_note = "" if model_name == "classifier" else "PLACEHOLDER PATH -- not sourced from any script; verify against actual 8B training output location"
        if exists:
            size_mb = os.path.getsize(path) / 1e6
            record("PASS", f"Checkpoint exists: {model_name}", f"{path} ({size_mb:.1f} MB)")
        else:
            record("FAIL", f"Checkpoint exists: {model_name}", f"{path} NOT FOUND. {note}{detector_note}")


def check_composite_metadata():
    if not os.path.exists(q13.METADATA_PATH):
        record("FAIL", "composite_metadata.json exists", q13.METADATA_PATH)
        return
    records_ = q13.load_composite_records()
    record("PASS", "composite_metadata.json loads", f"{len(records_)} composite records")
    if len(records_) != 1760:
        record("WARN", "composite_metadata.json record count", f"found {len(records_)}, expected 1760 per the locked spec")
    conditions = {}
    for r in records_:
        conditions[r["condition"]] = conditions.get(r["condition"], 0) + 1
    record("PASS" if set(conditions) == {"A", "B", "C"} else "FAIL",
           "composite_metadata.json condition coverage", str(conditions))
    missing_fields = []
    for r in records_[:50]:  # sample check, not exhaustive -- 13p already did the exhaustive one
        for f in q13.IDENTITY_FIELDS:
            if f not in r:
                missing_fields.append(f)
    if missing_fields:
        record("FAIL", "composite_metadata.json required fields (sample of 50)", f"missing: {sorted(set(missing_fields))}")
    else:
        record("PASS", "composite_metadata.json required fields (sample of 50)", "all IDENTITY_FIELDS present")
    return records_


def check_target_resolution(records_):
    if records_ is None:
        record("SKIP", "Target image/bbox resolution against LABEL_DIR_BASE", "composite_metadata.json didn't load")
        return
    if not os.path.isdir(q13.LABEL_DIR_BASE):
        record("SKIP", "Target image/bbox resolution against LABEL_DIR_BASE", "LABEL_DIR_BASE not found")
        return
    print("  (scanning LABEL_DIR_BASE -- this can take a minute on the full dataset)")
    label_lookup = q13.build_label_lookup()
    image_lookup = q13.build_image_lookup()
    unique_targets = {r["target_id"]: r for r in records_}
    resolved_img = sum(1 for tid in unique_targets if tid in image_lookup)
    resolved_bbox = sum(1 for tid in unique_targets if tid in label_lookup)
    n = len(unique_targets)
    status_img = "PASS" if resolved_img == n else ("WARN" if resolved_img > 0.95 * n else "FAIL")
    record(status_img, "Target images resolved via direct stem match", f"{resolved_img}/{n} targets")
    status_bbox = "PASS" if resolved_bbox == n else ("WARN" if resolved_bbox > 0.95 * n else "FAIL")
    record(status_bbox, "Target bboxes resolved via direct stem match", f"{resolved_bbox}/{n} targets")
    if resolved_img < n:
        missing = [tid for tid in unique_targets if tid not in image_lookup][:5]
        record("WARN", "Sample of unresolved target_ids", str(missing))


def check_results_dirs_writable():
    for d in (q13.RESULTS_DIR, q13.SUMMARY_DIR):
        try:
            os.makedirs(d, exist_ok=True)
            testfile = os.path.join(d, ".preflight_write_test")
            with open(testfile, "w") as f:
                f.write("ok")
            os.remove(testfile)
            record("PASS", f"Output directory writable: {d}")
        except Exception as e:
            record("FAIL", f"Output directory writable: {d}", str(e))


try_check("LABEL_DIR_BASE", check_label_dir_base)
try_check("Classifier data dirs", check_classifier_dirs)
try_check("Model checkpoints", check_checkpoints)
_records = None
try:
    _records = check_composite_metadata()
except Exception as e:
    record("FAIL", "composite_metadata.json checks", f"{type(e).__name__}: {e}")
try_check("Target resolution", lambda: check_target_resolution(_records))
try_check("Output directories writable", check_results_dirs_writable)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Classifier environment checks (needs `mambavision` conda env)
# ═════════════════════════════════════════════════════════════════════════════

def check_classifier_env():
    try:
        import torch
    except ImportError:
        record("SKIP", "Classifier env (torch/mambavision)", "torch not importable in this environment -- run in `mambavision` env")
        return

    record("PASS", "torch importable", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    record("PASS" if cuda_ok else "WARN", "CUDA available",
           torch.cuda.get_device_name(0) if cuda_ok else "no CUDA device visible")

    if not os.path.isdir(q13.MAMBAVISION_REPO_PATH):
        record("FAIL", "MAMBAVISION_REPO_PATH exists", q13.MAMBAVISION_REPO_PATH)
        return
    record("PASS", "MAMBAVISION_REPO_PATH exists", q13.MAMBAVISION_REPO_PATH)

    try:
        sys.path.insert(0, q13.MAMBAVISION_REPO_PATH)
        from mambavision import models  # noqa: F401
        record("PASS", "`from mambavision import models` succeeds (NVlabs fork)")
    except Exception as e:
        record("FAIL", "`from mambavision import models` succeeds", f"{type(e).__name__}: {e}")
        return

    # hf_to_correct
    try:
        hf_map = q13.load_hf_to_correct_map()
        record("PASS", "hf_to_correct derived", str(hf_map))
    except Exception as e:
        record("FAIL", "hf_to_correct derived", f"{type(e).__name__}: {e}")

    # Actually load the classifier checkpoint into the real architecture
    ckpt_path = q13.MODEL_CONFIG["classifier"]["checkpoint"]
    if not os.path.exists(ckpt_path):
        record("SKIP", "Classifier checkpoint loads into architecture", "checkpoint file not found (see above)")
        return
    try:
        model, device = q13.load_classifier()
        record("PASS", "Classifier checkpoint loads into MambaVision_S architecture",
               f"strict load succeeded, device={device}")
    except Exception as e:
        record("FAIL", "Classifier checkpoint loads into MambaVision_S architecture", f"{type(e).__name__}: {e}")


try_check("Classifier environment", check_classifier_env)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Detector environment checks (needs `mambayolo` conda env)
# ═════════════════════════════════════════════════════════════════════════════

def check_detector_env():
    try:
        from ultralytics import YOLO
        import ultralytics
    except ImportError:
        record("SKIP", "Detector env (ultralytics)", "ultralytics not importable in this environment -- run in `mambayolo` env")
        return

    record("PASS", "ultralytics importable", ultralytics.__version__)

    for model_name in ("yolo_pretrained", "yolo_scratch", "mamba_yolo"):
        ckpt_path = q13.MODEL_CONFIG[model_name]["checkpoint"]
        if not os.path.exists(ckpt_path):
            record("SKIP", f"YOLO(...) loads: {model_name}", "checkpoint file not found (see above) -- path is an unverified placeholder")
            continue
        try:
            model = YOLO(ckpt_path)
            task = getattr(model, "task", "unknown")
            names = getattr(model, "names", {})
            nc = len(names) if names else "unknown"
            record("PASS", f"YOLO(...) loads: {model_name}",
                   f"task={task}, nc={nc}, names_sample={dict(list(names.items())[:3]) if names else {}}")
            if isinstance(nc, int) and nc != q13.NUM_CLASSES:
                record("WARN", f"Class count mismatch: {model_name}",
                       f"checkpoint reports nc={nc}, but NUM_CLASSES={q13.NUM_CLASSES} -- confirm this checkpoint is the right one")
        except Exception as e:
            record("FAIL", f"YOLO(...) loads: {model_name}",
                   f"{type(e).__name__}: {e} -- if mamba_yolo fails here specifically, "
                   f"it confirms the Ultralytics-API-compatibility assumption in 13q's "
                   f"load_detector() is wrong and needs a different loader for this checkpoint")


try_check("Detector environment", check_detector_env)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Summary
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
for status, name, detail in results:
    counts[status] += 1
for status in ("PASS", "WARN", "FAIL", "SKIP"):
    print(f"  {status}: {counts[status]}")

if counts["FAIL"] > 0:
    print("\n  Hard failures found -- do NOT run 13q for real until these are")
    print("  resolved. Re-run this script after fixing to confirm.")
    print("  Failures:")
    for status, name, detail in results:
        if status == "FAIL":
            print(f"    - {name}: {detail}")
    sys.exit(1)
else:
    print("\n  No hard failures in this environment's checks. If this was run")
    print("  in only one conda env, run it in the other env too before")
    print("  trusting the full picture (classifier checks need `mambavision`,")
    print("  detector checks need `mambayolo`).")
    sys.exit(0)
