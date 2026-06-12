"""
Real-time vehicle & person counting — CoreML backend (Apple Silicon).

Optimisations applied:
  1. CoreML model runs on the Apple Neural Engine (3-4× faster than MPS/PyTorch)
  2. Async video capture in a background thread (eliminates frame-read stall)
  3. Frame skipping: YOLO runs every N frames; ByteTrack Kalman bridges gaps

Run export_coreml.py once first to produce the .mlpackage file.

Usage:
    python run_coreml.py                              # default video, skip-n=2
    python run_coreml.py --source 0                   # live webcam
    python run_coreml.py --source rtsp://...          # IP camera
    python run_coreml.py --skip-n 3 --no-display      # headless, skip 2/3 frames
    python run_coreml.py --weights path/to/model.mlpackage
"""

import argparse
import json
import platform
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ── paths ──────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
ROOT     = HERE.parent
DEFAULT_WEIGHTS = ROOT / "models/weights/best_v3_merged.mlpackage"
DEFAULT_VIDEO   = ROOT / "Demo/ANMR0006.mp4"
ROI_FILE        = ROOT / "counting_experiment/roi.json"
OUT_VID         = HERE / "output_coreml.mp4"

# ── counting constants ─────────────────────────────────────────────────────────
CLASS_PERSON  = 0
CLASS_CAR     = 1
COLORS        = {CLASS_CAR: (0, 220, 0), CLASS_PERSON: (255, 100, 0)}
MOTION_BUFFER = 45
MOTION_MIN_PX = 8
MIN_TRACK_AGE = 5    # CoreML fragmentation: lower threshold so fragments still reach count
GHOST_RADIUS  = 80
GHOST_TIMEOUT = 90   # 1.5s at 60fps — long enough to suppress fragments, short enough for busy roads

# CoreML converts the model's confidence scores to a lower range than PyTorch.
# Cars consistently output 0.15–0.37; persons stay near 0.93. We use a low
# pre-NMS floor here and let the per-class thresholds (conf_car / conf_person)
# do the real filtering.
COREML_CONF_FLOOR = 0.10


# ── async capture ──────────────────────────────────────────────────────────────
class AsyncCapture:
    """Background thread reads frames so inference never waits on I/O."""

    def __init__(self, source):
        src = int(source) if str(source).isdigit() else str(source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open source: {source}")
        self._q       = deque(maxlen=4)
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self._running = False
                break
            with self._lock:
                self._q.append(frame)

    def read(self):
        with self._lock:
            if self._q:
                return True, self._q.popleft()
        return self._running, None  # None while buffer momentarily empty

    @property
    def fps(self):    return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
    @property
    def width(self):  return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    @property
    def height(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    @property
    def alive(self):  return self._running

    def release(self):
        self._running = False
        self.cap.release()


# ── helpers ────────────────────────────────────────────────────────────────────
def load_roi(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        print("No roi.json found — using full frame.")
        return None
    pts = json.loads(path.read_text())
    print(f"ROI loaded: {len(pts)} points from {path}")
    return np.array(pts, dtype=np.int32)


def inside_roi(cx: float, cy: float, roi: Optional[np.ndarray]) -> bool:
    if roi is None:
        return True
    return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0


def centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_stationary(hist: deque) -> bool:
    if len(hist) < MOTION_BUFFER:
        return False
    pts = np.array(list(hist))
    return float(np.max(np.linalg.norm(pts - pts[0], axis=1))) < MOTION_MIN_PX


def draw_roi(frame, roi):
    if roi is not None:
        cv2.polylines(frame, [roi], True, (0, 255, 255), 2)


def draw_overlay(frame, cars, people, fps, skip_n):
    cv2.rectangle(frame, (5, 5), (300, 85), (20, 20, 20), -1)
    cv2.putText(frame, f"Cars  : {cars}",   (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0),   2)
    cv2.putText(frame, f"People: {people}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}  skip:{skip_n}",
                (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    default=str(DEFAULT_WEIGHTS),
                   help="Path to .mlpackage (default: models/weights/best_v3_merged.mlpackage)")
    p.add_argument("--source",     default=str(DEFAULT_VIDEO),
                   help="Video path, '0' for webcam, or rtsp:// URL")
    p.add_argument("--skip-n",     type=int, default=2,
                   help="Run YOLO every N frames; tracker Kalman-predicts the rest (default 2)")
    p.add_argument("--conf-car",   type=float, default=0.15,
                   help="CoreML outputs cars at 0.15-0.37 conf; 0.15 keeps detections stable across frames")
    p.add_argument("--conf-person",type=float, default=0.45)
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode — skip imshow, still write output video")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    if platform.system() != "Darwin":
        print("WARNING: CoreML is Apple-only. On Linux/Windows use run_tensorrt.py instead.")

    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(
            f"Weights not found: {weights}\n"
            "Run  python export_coreml.py  first to generate the .mlpackage file."
        )

    roi = load_roi(ROI_FILE)

    print(f"Loading model: {weights.name} …")
    model = YOLO(str(weights))
    print("Model ready.")

    try:
        from boxmot.trackers import ByteTrack
    except ImportError:
        sys.exit("boxmot not installed. Run: pip install boxmot")

    cap = AsyncCapture(args.source)
    print(f"Source: {args.source}  ({cap.width}×{cap.height} @ {cap.fps:.0f}fps)")

    # track_thresh=0.15: CoreML outputs car confidences in the 0.15-0.37 range
    # (vs 0.45+ for PyTorch). ByteTrack only creates new tracks from detections
    # above track_thresh, so we lower it to match CoreML's output distribution.
    tracker = ByteTrack(frame_rate=int(cap.fps), track_buffer=600, track_thresh=0.15)
    print(f"Tracker ready. YOLO every {args.skip_n} frame(s).\n")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(OUT_VID), fourcc, cap.fps, (cap.width, cap.height))

    if not args.no_display:
        cv2.namedWindow("CoreML — Real-time Counting", cv2.WINDOW_NORMAL)

    # ── counting state ─────────────────────────────────────────────────────────
    centroids:   dict = {}
    track_age:   dict = {}
    cls_map:     dict = {}
    counted_ids: set  = set()
    ghost_zones: dict = {}
    active_tids: set  = set()
    car_count    = 0
    person_count = 0

    # ── fps counter ────────────────────────────────────────────────────────────
    fps_times: deque = deque(maxlen=30)

    frame_idx = 0
    last_dets = np.empty((0, 6), dtype=np.float32)

    print("Running — press Q to stop.\n")

    while cap.alive:
        ret, frame = cap.read()
        if not ret or frame is None:   # frame is None when buffer momentarily empty
            time.sleep(0.001)
            continue

        t_now = time.perf_counter()
        fps_times.append(t_now)
        live_fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-6)

        # ── inference (every skip_n frames) ───────────────────────────────────
        if frame_idx % args.skip_n == 0:
            results  = model(frame, conf=COREML_CONF_FLOOR, verbose=False)[0]
            boxes_np = results.boxes
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

            # suppress persons whose centroid falls inside a car box
            car_boxes = [(d[0], d[1], d[2], d[3]) for d in dets if int(d[5]) == CLASS_CAR]
            if car_boxes:
                filtered = []
                for d in dets:
                    if int(d[5]) == CLASS_PERSON:
                        pcx, pcy = centroid(d[0], d[1], d[2], d[3])
                        if any(cx1 <= pcx <= cx2 and cy1 <= pcy <= cy2
                               for cx1, cy1, cx2, cy2 in car_boxes):
                            continue
                    filtered.append(d)
                dets = filtered

            last_dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

        # ── tracker update (every frame) ──────────────────────────────────────
        # Pass last_dets on skipped frames rather than empty so ByteTrack keeps
        # tentative tracks alive between detection hits.
        tracks = tracker.update(last_dets, frame)

        annotated = frame.copy()
        draw_roi(annotated, roi)
        current_tids: set = set()

        if tracks is not None and len(tracks):
            for i in range(len(tracks)):
                x1, y1, x2, y2 = [int(v) for v in tracks.xyxy[i]]
                tid  = int(tracks.id[i])
                cls  = int(tracks.cls[i])
                cx, cy = centroid(x1, y1, x2, y2)
                current_tids.add(tid)

                if tid not in centroids:
                    centroids[tid] = deque(maxlen=MOTION_BUFFER)
                    track_age[tid] = 0
                centroids[tid].append((cx, cy))
                track_age[tid] += 1
                cls_map[tid] = cls

                if cls == CLASS_CAR and is_stationary(centroids[tid]):
                    color = (100, 100, 100)
                    label = f"parked #{tid}"
                else:
                    color = COLORS.get(cls, (200, 200, 200))
                    label = f"{'car' if cls == CLASS_CAR else 'person'} #{tid}"

                    if tid not in counted_ids and track_age[tid] >= MIN_TRACK_AGE:
                        in_ghost = any(
                            g["cls"] == cls
                            and np.hypot(cx - g["cx"], cy - g["cy"]) < GHOST_RADIUS
                            and (frame_idx - g["died_frame"]) < GHOST_TIMEOUT
                            for g in ghost_zones.values()
                        )
                        counted_ids.add(tid)
                        if not in_ghost:
                            if cls == CLASS_CAR:
                                car_count += 1
                            else:
                                person_count += 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # ghost bookkeeping
        for dead in active_tids - current_tids:
            if dead in centroids and len(centroids[dead]):
                lx, ly = centroids[dead][-1]
                ghost_zones[dead] = {"cx": lx, "cy": ly,
                                     "cls": cls_map.get(dead, -1),
                                     "died_frame": frame_idx}
        ghost_zones = {t: g for t, g in ghost_zones.items()
                       if (frame_idx - g["died_frame"]) < GHOST_TIMEOUT}
        active_tids = current_tids

        draw_overlay(annotated, car_count, person_count, live_fps, args.skip_n)
        writer.write(annotated)

        if not args.no_display:
            cv2.imshow("CoreML — Real-time Counting", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

        frame_idx += 1
        if frame_idx % 600 == 0:
            print(f"  frame {frame_idx}  fps={live_fps:.1f}  cars={car_count}  people={person_count}")

    cap.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n{'='*45}")
    print("FINAL COUNTS  (CoreML + ByteTrack)")
    print(f"{'='*45}")
    print(f"Cars   : {car_count}")
    print(f"People : {person_count}")
    print(f"Frames : {frame_idx}")
    print(f"Output : {OUT_VID}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
