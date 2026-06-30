"""
Real-time vehicle & person counting — CoreML backend (Apple Silicon).

Optimisations applied:
  1. CoreML model runs on the Apple Neural Engine (3-4× faster than MPS/PyTorch)
  2. Async video capture in a background thread (eliminates frame-read stall)
  3. Frame skipping: YOLO runs every N frames; ByteTrack Kalman bridges gaps

Run export_coreml.py once first to produce the .mlpackage file.

Usage:
    python run_coreml.py                              # default video, skip-n=1
    python run_coreml.py --source 0                   # live webcam
    python run_coreml.py --source rtsp://...          # IP camera
    python run_coreml.py --skip-n 2 --no-display      # headless, legacy skip-2 mode
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
OUTPUTS_DIR     = HERE / "outputs"

# ── counting constants ─────────────────────────────────────────────────────────
CLASS_PERSON  = 0
CLASS_CAR     = 1
COLORS        = {CLASS_CAR: (0, 220, 0), CLASS_PERSON: (255, 100, 0)}
MOTION_BUFFER = 45
MOTION_MIN_PX = 8
# Class-specific thresholds.
# Cars: CoreML confidence fluctuates → tracks fragment → count early, long ghost window.
# Persons: arm-opening fragments rarely live 20 frames with last_dets keeping tracks
#   stable; GHOST_TIMEOUT=120 absorbs any fragment that does create a new track.
MIN_TRACK_AGE = {CLASS_CAR: 5,  CLASS_PERSON: 20}
GHOST_RADIUS  = {CLASS_CAR: 80, CLASS_PERSON: 80}
GHOST_TIMEOUT = {CLASS_CAR: 90, CLASS_PERSON: 120}

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


def car_ghost_params(bw: float):
    """Return (radius, timeout) scaled by box width for a car ghost zone.

    Close/large cars (big bw) need a wider radius because their centroid moves
    more pixels between fragmentation events, but a shorter timeout because
    fragments appear within a few frames — not 90. This prevents the wide radius
    from blocking new cars entering the same corridor later.

      radius  = max(80,  bw * 0.6)     e.g. 300px car → 180px radius
      timeout = max(45,  90 * 100/bw)  e.g. 300px car → 45 frames (0.75s)
    """
    radius  = max(GHOST_RADIUS[CLASS_CAR],  bw * 0.6)
    timeout = max(45, int(GHOST_TIMEOUT[CLASS_CAR] * 100.0 / max(bw, 100)))
    return radius, timeout


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
    p.add_argument("--skip-n",     type=int, default=1,
                   help="Run YOLO every N frames; tracker Kalman-predicts the rest (default 1)")
    p.add_argument("--conf-car",   type=float, default=0.15,
                   help="CoreML outputs cars at 0.15-0.37 conf; 0.15 keeps detections stable across frames")
    p.add_argument("--conf-person",type=float, default=0.45)
    p.add_argument("--no-display", action="store_true",
                   help="Headless mode — skip imshow, still write output video")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-track counting decisions to terminal for diagnosis")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many frames (0 = run to end; useful for benchmarking)")
    p.add_argument("--roi", default=None,
                   help="Path to ROI JSON file, or 'none' to use full frame (default: counting_experiment/roi.json)")
    p.add_argument("--output", default=None,
                   help="Output video path. Default: realtime_inference/outputs/<video_stem>/skip_n_<N>.mp4")
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

    if args.roi and args.roi.lower() == "none":
        roi = None
        print("ROI: full frame (--roi none)")
    elif args.roi:
        roi = load_roi(Path(args.roi))
    else:
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

    if args.output:
        out_path = Path(args.output)
    else:
        src_stem = Path(args.source).stem if args.source != "0" else "webcam"
        out_path = OUTPUTS_DIR / src_stem / f"skip_n_{args.skip_n}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, cap.fps, (cap.width, cap.height))
    print(f"Output: {out_path}")

    if not args.no_display:
        cv2.namedWindow("CoreML — Real-time Counting", cv2.WINDOW_NORMAL)

    # ── counting state ─────────────────────────────────────────────────────────
    centroids:   dict = {}
    track_age:   dict = {}
    cls_map:     dict = {}
    box_width:   dict = {}   # tid → last known box width (pixels), for adaptive ghost radius
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
        # Always pass last_dets (not empty on skip frames). Passing empty sends
        # all tracks to ByteTrack's "lost" state every other frame — lost tracks
        # are not output, so track_age stalls and nothing reaches MIN_TRACK_AGE.
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
                box_width[tid] = x2 - x1

                if cls == CLASS_CAR and is_stationary(centroids[tid]):
                    color = (100, 100, 100)
                    label = f"parked #{tid}"
                else:
                    color = COLORS.get(cls, (200, 200, 200))
                    label = f"{'car' if cls == CLASS_CAR else 'person'} #{tid}"

                    if tid not in counted_ids and track_age[tid] >= MIN_TRACK_AGE[cls]:
                        # Persons: use first centroid (where the track appeared).
                        # With MIN_TRACK_AGE=20 a person walks ~80px before the
                        # ghost check fires; current centroid drifts to the edge of
                        # GHOST_RADIUS=80 and escapes. Cars keep current centroid
                        # (MIN_TRACK_AGE=5 → minimal drift).
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
                        # For persons: also suppress if a currently-live counted
                        # person was in this same spot when THIS track first appeared.
                        # Catches arm-opening fragments (T1 still alive → no ghost
                        # zone yet), by comparing T2's first centroid to T1's
                        # historical centroid at the frame T2 first appeared.
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
                                        print(f"f{frame_idx:5d} LIVE-SUPP  tid={tid:4d} person — same spot as live tid={other_tid}")
                                    break
                        counted_ids.add(tid)
                        if not in_ghost:
                            if cls == CLASS_CAR:
                                car_count += 1
                            else:
                                person_count += 1
                            if args.verbose:
                                print(f"f{frame_idx:5d} COUNTED    tid={tid:4d} {'car' if cls==CLASS_CAR else 'person'} age={track_age[tid]}  total cars={car_count} ppl={person_count}")
                        elif args.verbose:
                            print(f"f{frame_idx:5d} GHOST-SUPP tid={tid:4d} {'car' if cls==CLASS_CAR else 'person'} age={track_age[tid]}")
                    elif tid not in counted_ids and args.verbose and track_age[tid] % 10 == 0:
                        print(f"f{frame_idx:5d} WAITING    tid={tid:4d} {'car' if cls==CLASS_CAR else 'person'} age={track_age[tid]}/{MIN_TRACK_AGE[cls]}")

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        # ghost bookkeeping
        for dead in active_tids - current_tids:
            if dead not in centroids or not len(centroids[dead]):
                continue
            lx, ly   = centroids[dead][-1]
            dead_cls = cls_map.get(dead, -1)
            dead_w   = box_width.get(dead, 0)
            if dead not in counted_ids and dead_cls == CLASS_PERSON:
                # PERSON DIED-YOUNG: no ghost zone.
                # Short person fragments at a crosswalk entry must not block
                # the next legitimate person starting at the same position.
                # Arm-opening fragments are caught instead by the live-track
                # proximity check at counting time (T1 still alive → no dead ghost
                # needed, the live check compares to T1's historical position).
                if args.verbose:
                    print(f"f{frame_idx:5d} DIED-YOUNG tid={dead:4d} person age={track_age.get(dead, 0)}/{MIN_TRACK_AGE.get(CLASS_PERSON,'?')} — no ghost")
                continue
            if dead_cls == CLASS_CAR:
                g_r, g_t = car_ghost_params(dead_w)
            else:
                g_r, g_t = GHOST_RADIUS[CLASS_PERSON], GHOST_TIMEOUT[CLASS_PERSON]
            ghost_zones[dead] = {
                "cx": lx, "cy": ly,
                "cls": dead_cls,
                "died_frame": frame_idx,
                "radius":  g_r,
                "timeout": g_t,
            }
            if args.verbose:
                tag = "GHOST-ZONE" if dead in counted_ids else "DIED-YOUNG"
                print(f"f{frame_idx:5d} {tag}    tid={dead:4d} {'car' if dead_cls==CLASS_CAR else 'person'} at ({lx:.0f},{ly:.0f}) r={g_r:.0f} t={g_t}")
        ghost_zones = {t: g for t, g in ghost_zones.items()
                       if (frame_idx - g["died_frame"]) < g["timeout"]}
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
        if args.max_frames and frame_idx >= args.max_frames:
            break

    print(f"\n{'='*45}")
    print("FINAL COUNTS  (CoreML + ByteTrack)")
    print(f"{'='*45}")
    print(f"Cars   : {car_count}")
    print(f"People : {person_count}")
    print(f"Frames : {frame_idx}")
    print(f"Output : {out_path}")
    print(f"{'='*45}")

    cap.release()
    try:
        writer.release()
    except Exception as e:
        print(f"Warning: VideoWriter.release() failed: {e}")
    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
