"""
3-way side-by-side tracking comparison:
  Left:   best.pt      (old, Oct-2024)
  Center: best_v3.pt   (new, May-2026 — trained on Roboflow data)
  Right:  visdrone_s.pt (VisDrone pretrained — overhead-camera specialist)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT  = Path(__file__).parent
VIDEO    = PROJECT / "Demo/ANMR0006.mp4"
OUT_PATH = PROJECT / "outputs/model_comparison/visdrone_comparison.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

START_FRAME = 1000
END_FRAME   = 2000
PANEL_W     = 640    # each panel scaled to this width
LABEL_H     = 46

# VisDrone → our semantic classes
VISDRONE_PERSON = {0, 1}          # pedestrian, people
VISDRONE_CAR    = {3, 4, 5, 8}    # car, van, truck, bus

LABELS = [
    ("OLD   best.pt",          (160, 40,  40)),
    ("NEW   best_v3.pt",       ( 30, 90, 160)),
    ("VISDRONE  visdrone_s.pt", ( 30, 130, 60)),
]


def label_bar(w: int, text: str, bg: tuple) -> np.ndarray:
    bar = np.zeros((LABEL_H, w, 3), dtype=np.uint8)
    bar[:] = bg
    cv2.putText(bar, text, (10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def det_count(results) -> tuple[int, int]:
    """Return (n_person, n_car) from YOLO results."""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0
    cls_list = [int(c) for c in boxes.cls]
    return cls_list.count(0), cls_list.count(1)


def visdrone_det_count(results, model) -> tuple[int, int]:
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0, 0
    cls_list = [int(c) for c in boxes.cls]
    n_person = sum(1 for c in cls_list if c in VISDRONE_PERSON)
    n_car    = sum(1 for c in cls_list if c in VISDRONE_CAR)
    return n_person, n_car


def remap_visdrone(results):
    """
    Re-colour VisDrone annotations:
      person classes → green boxes labelled 'person'
      car classes    → blue boxes labelled 'car'
      others         → grey
    """
    frame = results[0].orig_img.copy()
    boxes = results[0].boxes
    if boxes is None:
        return frame
    for i in range(len(boxes)):
        cls   = int(boxes.cls[i])
        conf  = float(boxes.conf[i])
        tid   = int(boxes.id[i]) if boxes.id is not None else -1
        xyxy  = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        if cls in VISDRONE_PERSON:
            color, label = (0, 220, 0),   f"person {conf:.2f}"
        elif cls in VISDRONE_CAR:
            color, label = (220, 100, 0), f"car {conf:.2f}"
        else:
            color, label = (120, 120, 120), results[0].names[cls]

        if tid >= 0:
            label += f" #{tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale   = PANEL_W / orig_w
    panel_h = int(orig_h * scale)
    out_w   = PANEL_W * 3
    out_h   = panel_h + LABEL_H

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (out_w, out_h))

    m_old  = YOLO(str(PROJECT / "models/weights/best.pt"))
    m_new  = YOLO(str(PROJECT / "models/weights/best_v3.pt"))
    m_vis  = YOLO(str(PROJECT / "models/weights/visdrone_s.pt"))

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    total = END_FRAME - START_FRAME
    fi    = 0

    print(f"Rendering {total} frames ({total/fps:.1f}s) → {OUT_PATH.name}")

    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break

        r_old = m_old.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=0.25, iou=0.45, verbose=False)
        r_new = m_new.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=0.25, iou=0.45, verbose=False)
        r_vis = m_vis.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=0.20, iou=0.45, verbose=False)

        np_old, nc_old = det_count(r_old)
        np_new, nc_new = det_count(r_new)
        np_vis, nc_vis = visdrone_det_count(r_vis, m_vis)

        ann_old = cv2.resize(r_old[0].plot(),  (PANEL_W, panel_h))
        ann_new = cv2.resize(r_new[0].plot(),  (PANEL_W, panel_h))
        ann_vis = cv2.resize(remap_visdrone(r_vis), (PANEL_W, panel_h))

        bar_old = label_bar(PANEL_W, f"{LABELS[0][0]}  P:{np_old} C:{nc_old}", LABELS[0][1])
        bar_new = label_bar(PANEL_W, f"{LABELS[1][0]}  P:{np_new} C:{nc_new}", LABELS[1][1])
        bar_vis = label_bar(PANEL_W, f"{LABELS[2][0]}  P:{np_vis} C:{nc_vis}", LABELS[2][1])

        combined = np.hstack([
            np.vstack([bar_old, ann_old]),
            np.vstack([bar_new, ann_new]),
            np.vstack([bar_vis, ann_vis]),
        ])

        writer.write(combined)
        fi += 1
        if fi % 100 == 0:
            print(f"  {fi}/{total}")

    cap.release()
    writer.release()
    print(f"Done → {OUT_PATH}")
    import os
    print(f"Size: {os.path.getsize(OUT_PATH) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
