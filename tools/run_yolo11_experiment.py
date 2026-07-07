"""
YOLO11 + BoT-SORT counting pipeline for the 8-experiment comparison.

Replaces run_coreml.py for experiment evaluation — uses PyTorch YOLO11
(any size: s/m/l/x) with BoT-SORT tracking and the same ghost-zone
deduplication system as the production pipeline.

Modes:
  --pretrained : weights are COCO-pretrained; COCO classes are remapped
                 to our 2-class schema on-the-fly.  No fine-tuning required.
  (no flag)    : weights are fine-tuned on our 2-class dataset; direct mapping.

COCO → our schema (pretrained mode):
  COCO  0 person       → 0 person
  COCO  2 car          → 1 car
  COCO  3 motorcycle   → 1 car
  COCO  5 bus          → 1 car
  COCO  7 truck        → 1 car
  All other COCO classes → filtered out

Usage:
  # pretrained (COCO weights)
  python tools/run_yolo11_experiment.py \\
    --weights models/weights/yolo11s.pt --pretrained \\
    --source Demo/ANMR0006.mp4 --roi none --output experiments/...

  # fine-tuned
  python tools/run_yolo11_experiment.py \\
    --weights experiments/yolo11_experiments/finetuned/yolo11s/weights/best.pt \\
    --source Demo/ANMR0006.mp4

Returns (via stdout on last lines):
  Cars   : <N>
  People : <N>
  Frames : <N>
  Output : <path>
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT    = Path(__file__).parent.parent
ROI_FILE = ROOT / "counting_experiment" / "roi.json"

# ── counting constants ─────────────────────────────────────────────────────────
CLASS_PERSON = 0
CLASS_CAR    = 1
COLORS       = {CLASS_CAR: (0, 220, 0), CLASS_PERSON: (255, 100, 0)}

MIN_TRACK_AGE = {CLASS_CAR: 5,  CLASS_PERSON: 20}
GHOST_RADIUS  = {CLASS_CAR: 80, CLASS_PERSON: 80}
GHOST_TIMEOUT = {CLASS_CAR: 90, CLASS_PERSON: 120}

MOTION_BUFFER = 45
MOTION_MIN_PX = 8

# ── COCO class remapping for pretrained mode ───────────────────────────────────
# Keep only these COCO classes, map to our 2-class schema.
COCO_TO_OURS = {
    0: CLASS_PERSON,   # person
    2: CLASS_CAR,      # car
    3: CLASS_CAR,      # motorcycle
    5: CLASS_CAR,      # bus
    7: CLASS_CAR,      # truck
}


def load_roi(path):
    if not path.exists():
        return None
    pts = json.loads(path.read_text())
    return np.array(pts, dtype=np.int32)


def inside_roi(cx, cy, roi):
    if roi is None:
        return True
    return cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0


def centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def car_ghost_params(bw: float):
    radius  = max(GHOST_RADIUS[CLASS_CAR],  bw * 0.6)
    timeout = max(45, int(GHOST_TIMEOUT[CLASS_CAR] * 100.0 / max(bw, 100)))
    return radius, timeout


def is_stationary(hist: deque) -> bool:
    if len(hist) < MOTION_BUFFER:
        return False
    pts = np.array(list(hist))
    return float(np.max(np.linalg.norm(pts - pts[0], axis=1))) < MOTION_MIN_PX


def build_tracker(fps: float):
    from boxmot.trackers.botsort.botsort import BotSort
    return BotSort(
        with_reid=False,
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=600,
        frame_rate=int(fps),
        cmc_method="sof",      # supported: ecc, orb, sift, sof
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights",     required=True,
                   help="Path to .pt weights (COCO pretrained or fine-tuned)")
    p.add_argument("--source",      required=True, help="Video file path")
    p.add_argument("--pretrained",  action="store_true",
                   help="Weights are COCO-pretrained; remap classes on-the-fly")
    p.add_argument("--roi",         default=None,
                   help="'none' for full frame, path to JSON, or default roi.json")
    p.add_argument("--output",      default=None,
                   help="Output video path (auto-generated if omitted)")
    p.add_argument("--conf-car",    type=float, default=0.25,
                   help="Confidence threshold for car class (default 0.25)")
    p.add_argument("--conf-person", type=float, default=0.35,
                   help="Confidence threshold for person class (default 0.35)")
    p.add_argument("--skip-n",      type=int, default=1,
                   help="Run YOLO every N frames (default 1)")
    p.add_argument("--max-frames",  type=int, default=0,
                   help="Stop after N frames (0 = full video)")
    p.add_argument("--no-display",  action="store_true")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    # ── ROI ──────────────────────────────────────────────────────────────────
    if args.roi and args.roi.lower() == "none":
        roi = None
    elif args.roi:
        roi = load_roi(Path(args.roi))
    else:
        roi = load_roi(ROI_FILE)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"Loading {weights.name} …")
    model = YOLO(str(weights))
    mode_tag = "COCO-pretrained" if args.pretrained else "fine-tuned"
    print(f"Mode: {mode_tag}")

    # ── capture ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.source}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Source: {args.source}  ({width}×{height} @ {fps:.0f}fps)")

    # ── output path ───────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        src_stem = Path(args.source).stem
        tag      = "pretrained" if args.pretrained else "finetuned"
        out_path = ROOT / "experiments" / "yolo11_experiments" / "videos" / \
                   f"{tag}_{weights.stem}" / f"{src_stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    # ── tracker ───────────────────────────────────────────────────────────────
    tracker = build_tracker(fps)
    print(f"Tracker: BoT-SORT (with_reid=False)  skip-n={args.skip_n}\n")

    # ── counting state ────────────────────────────────────────────────────────
    centroids:   dict = {}
    track_age:   dict = {}
    cls_map:     dict = {}
    box_width:   dict = {}
    counted_ids: set  = set()
    ghost_zones: dict = {}
    active_tids: set  = set()
    car_count    = 0
    person_count = 0
    fps_times: deque = deque(maxlen=30)

    frame_idx   = 0
    last_dets   = np.empty((0, 6), dtype=np.float32)

    print("Processing…")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_now = time.perf_counter()
        fps_times.append(t_now)
        live_fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 1e-6)

        # ── inference ─────────────────────────────────────────────────────────
        if frame_idx % args.skip_n == 0:
            results  = model(frame, conf=0.10, verbose=False)[0]
            boxes_np = results.boxes
            dets = []
            if boxes_np is not None and len(boxes_np):
                for i in range(len(boxes_np)):
                    raw_cls  = int(boxes_np.cls[i])
                    conf     = float(boxes_np.conf[i])

                    if args.pretrained:
                        our_cls = COCO_TO_OURS.get(raw_cls)
                        if our_cls is None:
                            continue
                    else:
                        if raw_cls not in (CLASS_PERSON, CLASS_CAR):
                            continue
                        our_cls = raw_cls

                    thresh = args.conf_person if our_cls == CLASS_PERSON else args.conf_car
                    if conf < thresh:
                        continue

                    x1, y1, x2, y2 = boxes_np.xyxy[i].tolist()
                    cx, cy = centroid(x1, y1, x2, y2)
                    if not inside_roi(cx, cy, roi):
                        continue
                    dets.append([x1, y1, x2, y2, conf, our_cls])

            # suppress persons inside car boxes
            car_boxes = [(d[0], d[1], d[2], d[3]) for d in dets if d[5] == CLASS_CAR]
            if car_boxes:
                filtered = []
                for d in dets:
                    if d[5] == CLASS_PERSON:
                        pcx, pcy = centroid(d[0], d[1], d[2], d[3])
                        if any(x1 <= pcx <= x2 and y1 <= pcy <= y2
                               for x1, y1, x2, y2 in car_boxes):
                            continue
                    filtered.append(d)
                dets = filtered

            last_dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

        # ── tracker update ────────────────────────────────────────────────────
        tracks = tracker.update(last_dets, frame)

        annotated = frame.copy()
        if roi is not None:
            cv2.polylines(annotated, [roi], True, (0, 255, 255), 2)
        current_tids: set = set()

        if tracks is not None and len(tracks):
            for i in range(len(tracks)):
                x1, y1, x2, y2 = [int(v) for v in tracks.xyxy[i]]
                tid  = int(tracks.id[i])
                cls  = int(tracks.cls[i])
                cx, cy = centroid(x1, y1, x2, y2)
                current_tids.add(tid)

                if tid not in centroids:
                    centroids[tid]  = deque(maxlen=MOTION_BUFFER)
                    track_age[tid]  = 0
                centroids[tid].append((cx, cy))
                track_age[tid] += 1
                cls_map[tid]   = cls
                box_width[tid] = x2 - x1

                if cls == CLASS_CAR and is_stationary(centroids[tid]):
                    color = (100, 100, 100)
                    label = f"parked #{tid}"
                else:
                    color = COLORS.get(cls, (200, 200, 200))
                    label = f"{'car' if cls == CLASS_CAR else 'person'} #{tid}"

                    if tid not in counted_ids and track_age[tid] >= MIN_TRACK_AGE[cls]:
                        if cls == CLASS_PERSON:
                            pts = list(centroids[tid])
                            check_cx, check_cy = pts[0] if pts else (cx, cy)
                        else:
                            check_cx, check_cy = cx, cy

                        in_ghost = any(
                            g["cls"] == cls
                            and np.hypot(check_cx - g["cx"], check_cy - g["cy"]) < g["radius"]
                            and (frame_idx - g["died_frame"]) < g["timeout"]
                            for g in ghost_zones.values()
                        )

                        if not in_ghost and cls == CLASS_PERSON:
                            age = track_age[tid]
                            for other_tid in current_tids:
                                if other_tid == tid or other_tid not in counted_ids:
                                    continue
                                if cls_map.get(other_tid) != CLASS_PERSON:
                                    continue
                                c = list(centroids.get(other_tid, deque()))
                                if not c:
                                    continue
                                ago = min(age - 1, len(c) - 1)
                                hx, hy = c[-ago - 1]
                                if np.hypot(check_cx - hx, check_cy - hy) < GHOST_RADIUS[CLASS_PERSON]:
                                    in_ghost = True
                                    if args.verbose:
                                        print(f"f{frame_idx:5d} LIVE-SUPP  tid={tid} person")
                                    break

                        counted_ids.add(tid)
                        if not in_ghost:
                            if cls == CLASS_CAR:
                                car_count += 1
                            else:
                                person_count += 1
                            if args.verbose:
                                print(f"f{frame_idx:5d} COUNTED tid={tid} "
                                      f"{'car' if cls==CLASS_CAR else 'person'} "
                                      f"cars={car_count} ppl={person_count}")
                        elif args.verbose:
                            print(f"f{frame_idx:5d} GHOST-SUPP tid={tid}")

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # ── ghost bookkeeping ─────────────────────────────────────────────────
        for dead in active_tids - current_tids:
            if dead not in centroids or not len(centroids[dead]):
                continue
            lx, ly   = centroids[dead][-1]
            dead_cls = cls_map.get(dead, -1)
            dead_w   = box_width.get(dead, 0)
            if dead not in counted_ids and dead_cls == CLASS_PERSON:
                continue  # person died-young → no ghost
            if dead_cls == CLASS_CAR:
                g_r, g_t = car_ghost_params(dead_w)
            else:
                g_r, g_t = GHOST_RADIUS[CLASS_PERSON], GHOST_TIMEOUT[CLASS_PERSON]
            ghost_zones[dead] = {
                "cx": lx, "cy": ly, "cls": dead_cls,
                "died_frame": frame_idx, "radius": g_r, "timeout": g_t,
            }
        ghost_zones = {t: g for t, g in ghost_zones.items()
                       if (frame_idx - g["died_frame"]) < g["timeout"]}
        active_tids = current_tids

        # ── overlay ───────────────────────────────────────────────────────────
        cv2.rectangle(annotated, (5, 5), (300, 85), (20, 20, 20), -1)
        cv2.putText(annotated, f"Cars  : {car_count}",   (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
        cv2.putText(annotated, f"People: {person_count}", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(annotated, f"FPS: {live_fps:.1f}  {weights.stem}",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("YOLO11+BoT-SORT", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 600 == 0:
            print(f"  frame {frame_idx}  fps={live_fps:.1f}  cars={car_count}  people={person_count}")
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()

    print(f"\n{'='*50}")
    print(f"FINAL COUNTS  (YOLO11 {weights.stem} + BoT-SORT)")
    print(f"{'='*50}")
    print(f"Cars   : {car_count}")
    print(f"People : {person_count}")
    print(f"Frames : {frame_idx}")
    print(f"Output : {out_path}")

    try:
        writer.release()
    except Exception as e:
        print(f"Warning: writer.release() failed: {e}")

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
