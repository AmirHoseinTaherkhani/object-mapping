"""
Side-by-side tracking comparison video: best.pt (old) vs best_v3.pt (new).
Uses Demo/ANMR0006.mp4 frames 1000–2000 where both models have active detections.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).parent
VIDEO    = PROJECT / "Demo/ANMR0006.mp4"
OLD_W    = PROJECT / "models/weights/best.pt"
NEW_W    = PROJECT / "models/weights/best_v3.pt"
OUT_PATH = PROJECT / "outputs/model_comparison/tracking_comparison_h264.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

START_FRAME = 1000
END_FRAME   = 2000
CONF        = 0.25
LABEL_H     = 50   # pixels for the label bar at top of each panel


def label_bar(width: int, text: str, bg: tuple, fg=(255, 255, 255)) -> np.ndarray:
    bar = np.zeros((LABEL_H, width, 3), dtype=np.uint8)
    bar[:] = bg
    cv2.putText(bar, text, (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, fg, 2, cv2.LINE_AA)
    return bar


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w = w * 2
    out_h = h + LABEL_H
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # h264 — smaller files
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (out_w, out_h))

    model_old = YOLO(str(OLD_W))
    model_new = YOLO(str(NEW_W))

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    total = END_FRAME - START_FRAME
    fi = 0

    print(f"Rendering {total} frames ({total/fps:.1f}s) → {OUT_PATH.name}")

    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break

        # run trackers
        r_old = model_old.track(frame, persist=True, tracker="bytetrack.yaml",
                                conf=CONF, iou=0.45, verbose=False)
        r_new = model_new.track(frame, persist=True, tracker="bytetrack.yaml",
                                conf=CONF, iou=0.45, verbose=False)

        ann_old = r_old[0].plot()
        ann_new = r_new[0].plot()

        n_old = len(r_old[0].boxes) if r_old[0].boxes is not None else 0
        n_new = len(r_new[0].boxes) if r_new[0].boxes is not None else 0

        bar_old = label_bar(w, f"OLD  best.pt       detections: {n_old}", (180, 40, 40))
        bar_new = label_bar(w, f"NEW  best_v3.pt    detections: {n_new}", (40, 140, 60))

        left  = np.vstack([bar_old, ann_old])
        right = np.vstack([bar_new, ann_new])
        combined = np.hstack([left, right])

        writer.write(combined)
        fi += 1

        if fi % 100 == 0:
            print(f"  {fi}/{total} frames")

    cap.release()
    writer.release()
    print(f"Done → {OUT_PATH}")


if __name__ == "__main__":
    main()
