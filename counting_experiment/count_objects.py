"""
Vehicle and person counting pipeline using YOLOv8 + ByteTrack (BoxMOT).

Features:
  - ROI polygon masking (loads roi.json; falls back to full frame)
  - Per-class confidence thresholds
  - Parked car filtering via centroid displacement buffer
  - Ghost-zone deduplication: when a track dies its last position is remembered
    for GHOST_TIMEOUT frames; new tracks appearing within GHOST_RADIUS pixels of
    a ghost are treated as the same object and not double-counted.

Usage:
    python count_objects.py [--no-display] [--conf-car 0.50] [--conf-person 0.45]
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ── paths ──────────────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
VIDEO   = HERE.parent / "Demo" / "ANMR0006.mp4"
WEIGHTS = HERE.parent / "models" / "weights" / "best_v3_merged.pt"
ROI_FILE = HERE / "roi.json"
OUT_VID  = HERE / "output_counted.mp4"

# ── constants ──────────────────────────────────────────────────────────────────
CLASS_PERSON = 0
CLASS_CAR    = 1
COLORS = {CLASS_CAR: (0, 220, 0), CLASS_PERSON: (255, 100, 0)}  # BGR

# Stationary-car filter: 45-frame buffer at 60fps ≈ 0.75s
MOTION_BUFFER  = 45
MOTION_MIN_PX  = 8       # displacement below this → parked

# Minimum consecutive frames a track must accumulate before it's counted.
MIN_TRACK_AGE  = 30      # 0.5s at 60fps

# Ghost-zone deduplication: after a track dies, hold its last centroid for
# GHOST_TIMEOUT frames. A new track of the same class within GHOST_RADIUS px
# of a ghost is treated as the same object and not incremented.
GHOST_RADIUS   = 80      # pixels
GHOST_TIMEOUT  = 300     # frames (5s at 60fps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode: skip imshow, still save output video")
    p.add_argument("--conf-car",    type=float, default=0.50)
    p.add_argument("--conf-person", type=float, default=0.45)
    return p.parse_args()


def load_roi(path: Path, frame_w: int, frame_h: int) -> Optional[np.ndarray]:
    if not path.exists():
        print("No roi.json found — using full frame.")
        return None
    pts = json.loads(path.read_text())
    print(f"Loaded ROI: {len(pts)} points from {path.name}")
    return np.array(pts, dtype=np.int32)


def inside_roi(cx: float, cy: float, roi: Optional[np.ndarray]) -> bool:
    if roi is None:
        return True
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0


def centroid(x1, y1, x2, y2) -> tuple[float, float]:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_stationary(history: deque) -> bool:
    if len(history) < MOTION_BUFFER:
        return False
    pts = np.array(list(history))
    disp = np.max(np.linalg.norm(pts - pts[0], axis=1))
    return disp < MOTION_MIN_PX


def draw_roi(frame: np.ndarray, roi: Optional[np.ndarray]):
    if roi is None:
        return
    cv2.polylines(frame, [roi], True, (0, 255, 255), 2)


def draw_counts(frame: np.ndarray, cars: int, people: int):
    cv2.rectangle(frame, (5, 5), (220, 65), (20, 20, 20), -1)
    cv2.putText(frame, f"Cars  : {cars}",   (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    cv2.putText(frame, f"People: {people}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)


def main():
    args = parse_args()

    # ── video ──────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {VIDEO}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}  {fps:.1f}fps  {total} frames")

    roi = load_roi(ROI_FILE, width, height)

    # ── writer ─────────────────────────────────────────────────────────────────
    import platform
    fourcc = cv2.VideoWriter_fourcc(*("avc1" if platform.system() == "Darwin" else "mp4v"))
    writer = cv2.VideoWriter(str(OUT_VID), fourcc, fps, (width, height))

    # ── model ──────────────────────────────────────────────────────────────────
    model = YOLO(str(WEIGHTS))
    print(f"Model : {WEIGHTS.name}")

    # ── tracker ────────────────────────────────────────────────────────────────
    try:
        from boxmot.trackers import ByteTrack
    except ImportError:
        sys.exit("boxmot not installed. Run: pip install boxmot")

    # ByteTrack: two-stage IoU+Kalman matching, no Re-ID.
    # Stage 1 matches high-confidence boxes; stage 2 sweeps low-confidence boxes
    # to recover tracks before they die — this keeps the arm-extended person
    # matched even when the bounding box changes shape.
    # track_buffer=600: hold lost tracks for 10s at 60fps.
    tracker = ByteTrack(frame_rate=int(fps), track_buffer=600)
    print("Tracker ready (ByteTrack, no Re-ID).")

    # ── state ──────────────────────────────────────────────────────────────────
    centroids:    dict[int, deque] = {}   # tid → deque of (cx, cy)
    track_age:    dict[int, int]   = {}   # tid → frames seen (unbounded)
    cls_map:      dict[int, int]   = {}   # tid → class
    counted_ids:  set[int]         = set()
    # ghost_zones: tid → {cx, cy, cls, died_frame}
    ghost_zones:  dict[int, dict]  = {}
    active_tids:  set[int]         = set()
    car_count    = 0
    person_count = 0
    conf_floor   = min(args.conf_car, args.conf_person)

    frame_idx = 0

    if not args.no_display:
        cv2.namedWindow("Counting", cv2.WINDOW_NORMAL)

    print(f"\nRunning… press Q to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── inference ──────────────────────────────────────────────────────────
        results  = model(frame, conf=conf_floor, verbose=False)[0]
        boxes_np = results.boxes

        # Build detection array for BoxMOT: [x1, y1, x2, y2, conf, class_id]
        dets = []
        if boxes_np is not None and len(boxes_np):
            for i in range(len(boxes_np)):
                cls  = int(boxes_np.cls[i])
                conf = float(boxes_np.conf[i])
                thresh = args.conf_person if cls == CLASS_PERSON else args.conf_car
                if conf < thresh:
                    continue
                x1, y1, x2, y2 = boxes_np.xyxy[i].tolist()
                cx, cy = centroid(x1, y1, x2, y2)
                if not inside_roi(cx, cy, roi):
                    continue
                dets.append([x1, y1, x2, y2, conf, cls])

        # ── suppress persons inside car boxes ─────────────────────────────────
        # A real pedestrian's centroid should never sit inside a car's bounding
        # box from an overhead camera. People visible through car windows are
        # filtered here before the tracker ever sees them.
        car_boxes = [(d[0], d[1], d[2], d[3]) for d in dets if int(d[5]) == CLASS_CAR]
        if car_boxes:
            filtered = []
            for d in dets:
                if int(d[5]) == CLASS_PERSON:
                    pcx, pcy = centroid(d[0], d[1], d[2], d[3])
                    if any(cx1 <= pcx <= cx2 and cy1 <= pcy <= cy2
                           for cx1, cy1, cx2, cy2 in car_boxes):
                        continue  # person centroid is inside a car box → drop
                filtered.append(d)
            dets = filtered

        dets_arr = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

        # ── update tracker ────────────────────────────────────────────────────
        tracks = tracker.update(dets_arr, frame)

        annotated = frame.copy()
        draw_roi(annotated, roi)

        current_tids: set[int] = set()

        if tracks is not None and len(tracks):
            for i in range(len(tracks)):
                x1, y1, x2, y2 = [int(v) for v in tracks.xyxy[i]]
                tid  = int(tracks.id[i])
                conf = float(tracks.conf[i])
                cls  = int(tracks.cls[i])
                cx, cy = centroid(x1, y1, x2, y2)
                current_tids.add(tid)

                # update centroid history and track age
                if tid not in centroids:
                    centroids[tid] = deque(maxlen=MOTION_BUFFER)
                    track_age[tid] = 0
                centroids[tid].append((cx, cy))
                track_age[tid] += 1
                cls_map[tid] = cls

                # parked car filter
                if cls == CLASS_CAR and is_stationary(centroids[tid]):
                    color = (100, 100, 100)  # grey = parked
                    label = f"parked #{tid}"
                else:
                    color = COLORS.get(cls, (200, 200, 200))
                    label = f"{'car' if cls == CLASS_CAR else 'person'} #{tid}"

                    # count only after MIN_TRACK_AGE frames and not a ghost
                    if tid not in counted_ids and track_age[tid] >= MIN_TRACK_AGE:
                        # ghost-zone check: is this a recently-dead track reborn nearby?
                        in_ghost = any(
                            g["cls"] == cls
                            and np.hypot(cx - g["cx"], cy - g["cy"]) < GHOST_RADIUS
                            and (frame_idx - g["died_frame"]) < GHOST_TIMEOUT
                            for g in ghost_zones.values()
                        )
                        if not in_ghost:
                            counted_ids.add(tid)
                            if cls == CLASS_CAR:
                                car_count += 1
                            else:
                                person_count += 1
                        else:
                            # absorb into counted_ids without incrementing so we
                            # don't re-check the ghost on every subsequent frame
                            counted_ids.add(tid)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # ── ghost-zone bookkeeping ─────────────────────────────────────────────
        # tracks that were active last frame but gone this frame → store as ghost
        for dead_tid in active_tids - current_tids:
            if dead_tid in centroids and len(centroids[dead_tid]):
                last_cx, last_cy = centroids[dead_tid][-1]
                ghost_zones[dead_tid] = {
                    "cx": last_cx, "cy": last_cy,
                    "cls": cls_map.get(dead_tid, -1),
                    "died_frame": frame_idx,
                }
        # expire old ghosts
        ghost_zones = {
            tid: g for tid, g in ghost_zones.items()
            if (frame_idx - g["died_frame"]) < GHOST_TIMEOUT
        }
        active_tids = current_tids

        draw_counts(annotated, car_count, person_count)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

        frame_idx += 1
        if frame_idx % 600 == 0:
            pct = 100 * frame_idx / max(total, 1)
            print(f"  {frame_idx}/{total} ({pct:.0f}%)  cars={car_count}  people={person_count}")

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("FINAL COUNTS")
    print("=" * 50)
    print(f"Cars counted   : {car_count}")
    print(f"People counted : {person_count}")
    print(f"Frames processed: {frame_idx}")
    print(f"Output video    : {OUT_VID}")
    print("=" * 50)


if __name__ == "__main__":
    main()
