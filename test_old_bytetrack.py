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

import sys
# allow passing model path as arg: python test_old_bytetrack.py [weights]
_arg_weights = sys.argv[1] if len(sys.argv) > 1 else None
if _arg_weights:
    WEIGHTS = Path(_arg_weights)
    OUT_VID = PROJECT / f"outputs/model_comparison/{WEIGHTS.stem}_bytetrack_30s.mp4"
else:
    WEIGHTS = PROJECT / "models/weights/best.pt"
    OUT_VID = PROJECT / "outputs/model_comparison/old_bytetrack_30s.mp4"
OUT_VID.parent.mkdir(parents=True, exist_ok=True)

# Per-class confidence thresholds — tuned from user label review
CONF_PERSON = 0.45   # raised: reduce person FP
CONF_CAR    = 0.18   # lowered: catch distant/small cars
CONF        = min(CONF_PERSON, CONF_CAR)   # pre-filter floor passed to YOLO
START       = 0
DURATION_S  = 30


def main():
    cap   = cv2.VideoCapture(str(VIDEO))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end_frame = START + int(fps * DURATION_S)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
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

    print(f"Running old best.pt + ByteTrack for {DURATION_S}s ({end_frame} frames)...")

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
    print("RESULTS — old best.pt + ByteTrack (30s)")
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
