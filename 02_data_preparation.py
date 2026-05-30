# 02_data_preparation.py
# Merge 7 Roboflow datasets into Master_Soil_Moisture and extract laser crops
# Adapted from ViT pipeline for MambaVision comparative study
# Cluster paths: /data/Grace/soil-moisture-dataset

import os
import shutil
import yaml
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = '/data/Grace/soil-moisture-dataset'
MASTER_DIR = '/data/Grace/Master_Soil_Moisture'
LASER_DIR  = '/data/Grace/Master_Laser_Crops'

# ── Class name mapping ────────────────────────────────────────────────────────
# Handles numeric, named, and Level_X formats across all 7 datasets
mapping = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', '10': '10',
    'soil-moisture-1.0': '1',
    'soil-moisture-2.0': '2',
    'soil-moisture-3.0': '3',
    'soil-moisture-5.0': '5',
    'soil-moisture-8.2': '8',
    'Level_0': '0', 'Level_1': '1', 'Level_2': '2', 'Level_3': '3',
    'Level_4': '4', 'Level_5': '5', 'Level_6': '6', 'Level_7': '7',
    'Level_8': '8', 'Level_9': '9', 'Level_10': '10',
}

# ── Step 1: Merge into Master_Soil_Moisture ───────────────────────────────────
def merge_datasets():
    if os.path.exists(MASTER_DIR):
        shutil.rmtree(MASTER_DIR)

    for proj_folder in os.listdir(BASE_DIR):
        if not os.path.isdir(os.path.join(BASE_DIR, proj_folder)):
            continue
        yaml_path = os.path.join(BASE_DIR, proj_folder, 'data.yaml')
        if not os.path.exists(yaml_path):
            continue

        with open(yaml_path, 'r') as f:
            class_names = yaml.safe_load(f)['names']

        for split in ['train', 'valid', 'test']:
            img_src = os.path.join(BASE_DIR, proj_folder, split, 'images')
            lbl_src = os.path.join(BASE_DIR, proj_folder, split, 'labels')
            target_split = 'validation' if split == 'valid' else split

            if not os.path.exists(img_src):
                continue

            for img_file in os.listdir(img_src):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
                lbl_p = os.path.join(lbl_src, lbl_file)

                if not os.path.exists(lbl_p):
                    continue

                with open(lbl_p, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    continue

                raw_name = str(class_names[int(lines[0].split()[0])])
                clean_name = mapping.get(raw_name, None)

                if clean_name is None:
                    print(f"Unmapped class: {raw_name} in {proj_folder}")
                    continue

                dest = os.path.join(MASTER_DIR, target_split, clean_name)
                os.makedirs(dest, exist_ok=True)
                unique_img = f"{proj_folder}_{img_file}"
                shutil.copy(
                    os.path.join(img_src, img_file),
                    os.path.join(dest, unique_img)
                )

    print("Merge complete!")

# ── Step 2: Verify merge ──────────────────────────────────────────────────────
def verify_merge():
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(MASTER_DIR, split)
        if os.path.exists(split_path):
            classes = os.listdir(split_path)
            total = sum(
                len(os.listdir(os.path.join(split_path, c)))
                for c in classes
            )
            print(f"\n{split}: {len(classes)} classes, {total} images")
            for c in sorted(classes):
                count = len(os.listdir(os.path.join(split_path, c)))
                print(f"  Class {c}: {count} images")

# ── Step 3: Extract laser crops ───────────────────────────────────────────────
def crop_laser(img, x_center, y_center, width, height, padding=0.05):
    W, H = img.size
    x1 = max(0, int((x_center - width / 2 - padding) * W))
    y1 = max(0, int((y_center - height / 2 - padding) * H))
    x2 = min(W, int((x_center + width / 2 + padding) * W))
    y2 = min(H, int((y_center + height / 2 + padding) * H))
    if width >= 0.95 and height >= 0.95:
        return img
    return img.crop((x1, y1, x2, y2))

def extract_laser_crops():
    if os.path.exists(LASER_DIR):
        shutil.rmtree(LASER_DIR)

    skipped = 0
    copied  = 0

    for proj_folder in os.listdir(BASE_DIR):
        proj_path = os.path.join(BASE_DIR, proj_folder)
        yaml_path = os.path.join(proj_path, 'data.yaml')
        if not os.path.exists(yaml_path):
            continue

        with open(yaml_path, 'r') as f:
            class_names = yaml.safe_load(f)['names']

        for split in ['train', 'valid', 'test']:
            img_dir = os.path.join(proj_path, split, 'images')
            lbl_dir = os.path.join(proj_path, split, 'labels')
            target_split = 'validation' if split == 'valid' else split

            if not os.path.exists(img_dir):
                continue

            for img_file in os.listdir(img_dir):
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue

                lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
                lbl_path = os.path.join(lbl_dir, lbl_file)

                if not os.path.exists(lbl_path):
                    skipped += 1
                    continue

                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    skipped += 1
                    continue

                parts    = lines[0].strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width    = float(parts[3])
                height   = float(parts[4])

                raw_name   = str(class_names[class_id])
                clean_name = mapping.get(raw_name, None)
                if clean_name is None:
                    skipped += 1
                    continue

                dest = os.path.join(LASER_DIR, target_split, clean_name)
                os.makedirs(dest, exist_ok=True)

                img = Image.open(
                    os.path.join(img_dir, img_file)
                ).convert('RGB')
                cropped = crop_laser(img, x_center, y_center, width, height)

                unique_name = f"{proj_folder}_{img_file}"
                cropped.save(os.path.join(dest, unique_name))
                copied += 1

    print(f"\nLaser crop extraction complete!")
    print(f"Saved: {copied} | Skipped: {skipped}")

# ── Step 4: Verify laser crops ────────────────────────────────────────────────
def verify_laser_crops():
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(LASER_DIR, split)
        if os.path.exists(split_path):
            classes = os.listdir(split_path)
            total = sum(
                len(os.listdir(os.path.join(split_path, c)))
                for c in classes
            )
            print(f"\n{split}: {len(classes)} classes, {total} images")
            for c in sorted(classes):
                count = len(os.listdir(os.path.join(split_path, c)))
                print(f"  Class {c}: {count} images")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Step 1: Merging datasets ===")
    merge_datasets()

    print("\n=== Step 2: Verifying merge ===")
    verify_merge()

    print("\n=== Step 3: Extracting laser crops ===")
    extract_laser_crops()

    print("\n=== Step 4: Verifying laser crops ===")
    verify_laser_crops() 
