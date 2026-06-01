"""
SAHI (Slicing Aided Hyper Inference) wrapper for intersection camera.

Slices each frame into overlapping 320x320 patches, runs YOLO on each,
then merges detections via NMS. Catches small/distant cars that full-frame
inference misses.

Usage:
    python sahi_inference.py                          # uses best_v3_camera.pt if exists
    python sahi_inference.py --model models/weights/best.pt
    python sahi_inference.py --no-sahi               # baseline (full-frame only)
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

PROJECT = Path(__file__).parent
VIDEO   = PROJECT / "Demo/ANMR0006.mp4"

# Per-class confidence thresholds
# SAHI zooms into patches so distant cars appear big — no need for a low threshold.
# Low threshold is only needed for full-frame baseline where distant cars look tiny.
CONF_PERSON      = 0.45   # same for both modes
CONF_CAR_SAHI    = 0.42   # SAHI: patches already zoom in, be strict to avoid FP
CONF_CAR_BASELINE = 0.18  # full-frame: distant cars are tiny, be permissive

# SAHI slicing parameters
SLICE_H      = 512
SLICE_W      = 512
OVERLAP      = 0.25   # 25% overlap between slices
POSTPROCESS  = "NMM"  # Non-Maximum Merging across slices

START_FRAME  = 1000
DURATION_S   = 30

CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 200, 0), 1: (30, 120, 220)}


def pick_weights() -> Path:
    camera = PROJECT / "models/weights/best_v3_camera.pt"
    if camera.exists():
        return camera
    v3 = PROJECT / "models/weights/best_v3.pt"
    if v3.exists():
        return v3
    return PROJECT / "models/weights/best.pt"


def filter_by_class_conf(result, conf_person=CONF_PERSON, conf_car=CONF_CAR_SAHI):
    """Return only detections that pass per-class thresholds."""
    kept = []
    for obj in result.object_prediction_list:
        cls_id = obj.category.id
        score  = obj.score.value
        thresh = conf_person if cls_id == 0 else conf_car
        if score >= thresh:
            kept.append(obj)
    return kept


def draw_detections(frame, detections):
    vis = frame.copy()
    for obj in detections:
        cls_id = obj.category.id
        score  = obj.score.value
        b      = obj.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return vis


def run_sahi(model_path: Path, use_sahi: bool, out_path: Path):
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    end_frame = START_FRAME + int(fps * DURATION_S)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h + 50))

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=min(CONF_PERSON, CONF_CAR),  # pre-filter; per-class done after
        device="mps",
    )

    frame_idx   = 0
    total_dets  = 0
    person_dets = 0
    car_dets    = 0
    t0          = time.time()

    print(f"Model : {model_path.name}")
    print(f"SAHI  : {'ON' if use_sahi else 'OFF (baseline)'}")
    print(f"Frames: {START_FRAME} → {end_frame}  ({DURATION_S}s)\n")

    while frame_idx < (end_frame - START_FRAME):
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if use_sahi:
            result = get_sliced_prediction(
                img_rgb,
                detection_model,
                slice_height=SLICE_H,
                slice_width=SLICE_W,
                overlap_height_ratio=OVERLAP,
                overlap_width_ratio=OVERLAP,
                postprocess_type=POSTPROCESS,
                verbose=0,
            )
        else:
            result = get_sliced_prediction(
                img_rgb,
                detection_model,
                slice_height=h,
                slice_width=w,
                overlap_height_ratio=0.0,
                overlap_width_ratio=0.0,
                verbose=0,
            )

        car_thresh = CONF_CAR_SAHI if use_sahi else CONF_CAR_BASELINE
        detections = filter_by_class_conf(result, conf_car=car_thresh)
        n          = len(detections)
        total_dets += n
        person_dets += sum(1 for d in detections if d.category.id == 0)
        car_dets    += sum(1 for d in detections if d.category.id == 1)

        vis = draw_detections(frame, detections)

        bar = np.zeros((50, w, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        elapsed = frame_idx / fps
        mode_str = "SAHI" if use_sahi else "BASE"
        txt = (f"[{mode_str}] t={elapsed:.1f}s  dets={n}  "
               f"person={sum(1 for d in detections if d.category.id==0)}  "
               f"car={sum(1 for d in detections if d.category.id==1)}")
        cv2.putText(bar, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 180), 2, cv2.LINE_AA)

        writer.write(np.vstack([vis, bar]))
        frame_idx += 1

        if frame_idx % 300 == 0:
            elapsed_real = time.time() - t0
            print(f"  frame {frame_idx}/{end_frame - START_FRAME}  "
                  f"avg_dets={total_dets/frame_idx:.1f}  "
                  f"fps_real={frame_idx/elapsed_real:.1f}")

    cap.release()
    writer.release()

    print(f"\n{'='*55}")
    print(f"SAHI results — {model_path.name} ({'SAHI ON' if use_sahi else 'baseline'})")
    print(f"{'='*55}")
    print(f"Frames processed : {frame_idx}")
    print(f"Avg dets/frame   : {total_dets/max(frame_idx,1):.2f}")
    print(f"  Person         : {person_dets/max(frame_idx,1):.2f}/frame")
    print(f"  Car            : {car_dets/max(frame_idx,1):.2f}/frame")
    print(f"Output           : {out_path}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--no-sahi", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else pick_weights()
    use_sahi   = not args.no_sahi

    out_dir = PROJECT / "outputs/model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix   = "sahi" if use_sahi else "base"
    out_path = Path(args.out) if args.out else (
        out_dir / f"{model_path.stem}_{suffix}.mp4"
    )

    run_sahi(model_path, use_sahi, out_path)


if __name__ == "__main__":
    main()
