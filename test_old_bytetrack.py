"""
Test: old best.pt + ByteTrack on 30s of Demo video.
Measures track ID stability and saves an annotated clip.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

PROJECT  = Path(__file__).parent
VIDEO    = PROJECT / "Demo/ANMR0006.mp4"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("weights", nargs="?", default=None)
parser.add_argument("--full", action="store_true", help="Run on entire video")
parser.add_argument("--conf-car",    type=float, default=0.40)
parser.add_argument("--conf-person", type=float, default=0.50)
args = parser.parse_args()

if args.weights:
    WEIGHTS = Path(args.weights)
    suffix  = "full" if args.full else "30s"
    cc_tag  = f"car{int(args.conf_car*100)}_p{int(args.conf_person*100)}"
    OUT_VID = PROJECT / f"outputs/model_comparison/{WEIGHTS.stem}_{cc_tag}_{suffix}.mp4"
else:
    WEIGHTS = PROJECT / "models/weights/best.pt"
    OUT_VID = PROJECT / "outputs/model_comparison/old_bytetrack_30s.mp4"
OUT_VID.parent.mkdir(parents=True, exist_ok=True)

CONF_PERSON = args.conf_person
CONF_CAR    = args.conf_car
CONF        = min(CONF_PERSON, CONF_CAR)   # pre-filter floor passed to YOLO
START       = 0
DURATION_S  = None if args.full else 30


def main():
    cap   = cv2.VideoCapture(str(VIDEO))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = total_frames if DURATION_S is None else START + int(fps * DURATION_S)
    duration_label = f"{end_frame/fps:.0f}s" if DURATION_S is None else f"{DURATION_S}s"

    import platform
    fourcc = cv2.VideoWriter_fourcc(*("avc1" if platform.system() == "Darwin" else "mp4v"))
    writer = cv2.VideoWriter(str(OUT_VID), fourcc, fps, (w, h + 50))

    model = YOLO(str(WEIGHTS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)

    # tracking stats
    track_history   = defaultdict(list)   # tid → [cls, ...]
    seen_ids        = set()
    id_switches     = 0       # new ID appearing after frame 0 (= a track that restarted)
    first_frame     = True
    frame_idx       = 0
    det_counts      = []

    print(f"Running {WEIGHTS.name} + ByteTrack for {duration_label} ({end_frame} frames)...")

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=CONF, iou=0.45, verbose=False)

        boxes  = results[0].boxes
        n_dets = 0
        current_ids = set()

        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                tid  = int(boxes.id[i])
                cls  = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                thresh = CONF_PERSON if cls == 0 else CONF_CAR
                if conf < thresh:
                    continue
                current_ids.add(tid)
                track_history[tid].append(cls)
                n_dets += 1

                if tid not in seen_ids:
                    if not first_frame:
                        id_switches += 1
                    seen_ids.add(tid)

        first_frame = False
        det_counts.append(n_dets)

        # annotate frame
        ann = results[0].plot()

        # stats bar
        bar = np.zeros((50, w, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        elapsed = frame_idx / fps
        txt = (f"t={elapsed:.1f}s | dets={n_dets} | "
               f"active_tracks={len(current_ids)} | "
               f"total_unique={len(seen_ids)} | id_switches={id_switches}")
        cv2.putText(bar, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 220, 180), 2, cv2.LINE_AA)

        writer.write(np.vstack([ann, bar]))
        frame_idx += 1

        if frame_idx % 300 == 0:
            print(f"  {frame_idx}/{end_frame}  unique_ids={len(seen_ids)}  switches={id_switches}")

    cap.release()
    writer.release()

    # summary
    print("\n" + "=" * 55)
    print(f"RESULTS — {WEIGHTS.name} + ByteTrack ({duration_label})")
    print(f"conf_car={CONF_CAR}  conf_person={CONF_PERSON}")
    print("=" * 55)
    print(f"Total frames processed : {frame_idx}")
    print(f"Mean detections/frame  : {np.mean(det_counts):.1f}")
    print(f"Total unique track IDs : {len(seen_ids)}")
    print(f"Track ID switches      : {id_switches}")

    # per-class track count
    person_tracks = sum(1 for tid, clss in track_history.items()
                        if max(set(clss), key=clss.count) == 0)
    car_tracks    = sum(1 for tid, clss in track_history.items()
                        if max(set(clss), key=clss.count) == 1)
    print(f"Person tracks          : {person_tracks}")
    print(f"Car tracks             : {car_tracks}")

    # quality signal: tracks with > 30 frames = persistent
    persistent = sum(1 for clss in track_history.values() if len(clss) >= 30)
    print(f"Persistent tracks(≥30f): {persistent}")
    print("=" * 55)
    print(f"\nVideo → {OUT_VID}")

    if id_switches < 15:
        print("\n✓ Tracking looks stable — consider shipping old best.pt + ByteTrack.")
    else:
        print("\n✗ Too many ID switches — fine-tuning on camera data is justified.")


if __name__ == "__main__":
    main()
