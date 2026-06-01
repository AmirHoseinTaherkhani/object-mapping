"""
Auto-label labeling_data/images/ with Grounding DINO.
Outputs:
  labeling_data/labels/        — YOLO-format .txt label files
  labeling_data/previews/      — annotated JPEG previews for human review
  labeling_data/contact_sheet.jpg — single image grid of all previews

Classes: 0=person  1=car

After running, review the previews (or contact_sheet.jpg), fix obvious
errors in Roboflow or any label editor, then run finetune_on_camera.py.
"""

import cv2
import math
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

PROJECT    = Path(__file__).parent
IMG_DIR    = PROJECT / "labeling_data" / "images"
LABEL_DIR  = PROJECT / "labeling_data" / "labels"
PREV_DIR   = PROJECT / "labeling_data" / "previews"
SHEET_PATH = PROJECT / "labeling_data" / "contact_sheet.jpg"

LABEL_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID   = "IDEA-Research/grounding-dino-tiny"
# Two separate prompts — DINO maps detected label text back to class index
TEXT_PROMPTS = "person . car ."

# Confidence thresholds
CONF_PERSON = 0.30
CONF_CAR    = 0.28
BOX_THRESH  = 0.25   # passed to processor

CLASS_COLORS = {0: (0, 200, 0), 1: (30, 120, 220)}   # green=person, blue=car
CLASS_NAMES  = {0: "person", 1: "car"}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def label_to_class(label: str):
    l = label.lower()
    if "person" in l or "pedestrian" in l or "people" in l:
        return 0
    if "car" in l or "vehicle" in l or "van" in l or "truck" in l or "bus" in l:
        return 1
    return None


def xyxy_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), img_w))
    y1 = max(0.0, min(float(y1), img_h))
    x2 = max(0.0, min(float(x2), img_w))
    y2 = max(0.0, min(float(y2), img_h))
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def draw_preview(img_bgr, detections):
    vis = img_bgr.copy()
    for cls_id, conf, (x1, y1, x2, y2) in detections:
        color = CLASS_COLORS[cls_id]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.putText(vis, label, (int(x1), max(int(y1) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return vis


def make_contact_sheet(preview_paths, out_path, cols=10, thumb_w=320):
    thumbs = []
    for p in sorted(preview_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h = int(img.shape[0] * thumb_w / img.shape[1])
        thumbs.append(cv2.resize(img, (thumb_w, h)))

    if not thumbs:
        return

    thumb_h = thumbs[0].shape[0]
    rows    = math.ceil(len(thumbs) / cols)
    sheet   = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = t

    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"Contact sheet → {out_path}")


def main():
    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading {MODEL_ID}...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model     = model.to(device)
    model.eval()

    images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    print(f"Images to label: {len(images)}\n")

    total_dets  = 0
    preview_paths = []
    stats = {0: 0, 1: 0}

    for idx, img_path in enumerate(images):
        img_bgr = cv2.imread(str(img_path))
        img_h, img_w = img_bgr.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        inputs = processor(
            images=img_pil,
            text=TEXT_PROMPTS,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESH,
            text_threshold=BOX_THRESH,
            target_sizes=[(img_h, img_w)],
        )[0]

        detections = []
        yolo_lines  = []

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            conf    = float(score)
            cls_id  = label_to_class(label)
            if cls_id is None:
                continue
            thresh = CONF_PERSON if cls_id == 0 else CONF_CAR
            if conf < thresh:
                continue

            box_np = box.cpu().numpy()
            cx, cy, bw, bh = xyxy_to_yolo(box_np, img_w, img_h)
            if bw < 0.005 or bh < 0.005:   # skip tiny/degenerate boxes
                continue

            detections.append((cls_id, conf, box_np))
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats[cls_id] += 1

        # save label file
        label_path = LABEL_DIR / (img_path.stem + ".txt")
        label_path.write_text("\n".join(yolo_lines))

        # save preview
        prev_path = PREV_DIR / img_path.name
        cv2.imwrite(str(prev_path), draw_preview(img_bgr, detections),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        preview_paths.append(prev_path)
        total_dets += len(detections)

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1:3d}/{len(images)}]  {img_path.name}  "
                  f"dets={len(detections)}  (P:{stats[0]} C:{stats[1]} total)")

    print(f"\n{'='*55}")
    print(f"Done. {len(images)} images labeled.")
    print(f"  Person annotations : {stats[0]}")
    print(f"  Car annotations    : {stats[1]}")
    print(f"  Total boxes        : {total_dets}")
    print(f"  Avg boxes/image    : {total_dets/max(len(images),1):.1f}")
    print(f"  Labels dir         : {LABEL_DIR}")
    print(f"  Previews dir       : {PREV_DIR}")
    print(f"{'='*55}")

    print("\nBuilding contact sheet...")
    make_contact_sheet(preview_paths, SHEET_PATH, cols=10, thumb_w=320)

    print("\nNext steps:")
    print("  1. Open labeling_data/contact_sheet.jpg — scan for obvious errors")
    print("  2. Fix wrong/missing boxes in Roboflow or Label Studio")
    print("  3. Export as YOLOv8 PyTorch → unzip to labeling_data/dataset/")
    print("  4. Run: python finetune_on_camera.py")


if __name__ == "__main__":
    main()
