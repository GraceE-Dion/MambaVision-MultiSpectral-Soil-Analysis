"""
inspect_predictions.py
========================
Diagnostic: loads a trained Mamba-YOLO-T checkpoint and runs inference
on a handful of validation images, printing the RAW prediction output
(boxes, confidences, classes) before any metric computation.
"""

from ultralytics import YOLO
import os

WEIGHTS = "./output_dir/soil_moisture/mambayolo_t_no_amp_test/weights/best.pt"
VAL_IMAGES_DIR = "/data/Grace/Master_Detection/val/images"

model = YOLO(WEIGHTS)

image_files = sorted(os.listdir(VAL_IMAGES_DIR))[:5]

for img_name in image_files:
    img_path = os.path.join(VAL_IMAGES_DIR, img_name)
    print("=" * 70)
    print(f"Image: {img_name}")

    for conf_thresh in [0.25, 0.05, 0.001]:
        results = model.predict(img_path, conf=conf_thresh, verbose=False)
        r = results[0]
        n_boxes = len(r.boxes) if r.boxes is not None else 0
        print(f"  conf>={conf_thresh}: {n_boxes} boxes detected")
        if n_boxes > 0:
            for box in r.boxes[:3]:
                xyxy = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                print(f"    class={cls}, conf={conf:.4f}, xyxy={[round(v,1) for v in xyxy]}")

print("\n" + "=" * 70)
print("If 0 boxes appear even at conf>=0.001, the model is producing")
print("essentially no confident detections at all.")
print("If boxes DO appear at low conf but not at 0.25, the model IS")
print("detecting things, just with low confidence.")