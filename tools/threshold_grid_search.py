"""
Grid search over (conf_car, conf_person) threshold combinations.
Runs ByteTrack on the 30-second clip for each combination and
scores by track stability. Saves results to outputs/threshold_search.csv.

Usage:
    python threshold_grid_search.py                  # local (MPS/CPU)
    python threshold_grid_search.py --video-seconds 60  # longer clip
"""

import argparse
import csv
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

PROJECT  = Path(__file__).parent
VIDEO    = PROJECT / "Demo/ANMR0006.mp4"
WEIGHTS  = PROJECT / "models/weights/best_v3_merged.pt"
OUT_CSV  = PROJECT / "outputs/threshold_search.csv"

CONF_CAR_VALUES    = [0.18, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
CONF_PERSON_VALUES = [0.40, 0.45, 0.50, 0.55, 0.60]


def run_one(model, conf_car, conf_person, end_frame, fps):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    conf_floor    = min(conf_car, conf_person)
    track_history = defaultdict(list)
    seen_ids      = set()
    id_switches   = 0
    first_frame   = True
    frame_idx     = 0
    det_counts    = []

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=conf_floor, iou=0.45, verbose=False)
        boxes = results[0].boxes
        n_dets = 0

        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                tid  = int(boxes.id[i])
                cls  = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                thresh = conf_person if cls == 0 else conf_car
                if conf < thresh:
                    continue
                track_history[tid].append(cls)
                n_dets += 1
                if tid not in seen_ids:
                    if not first_frame:
                        id_switches += 1
                    seen_ids.add(tid)

        first_frame = False
        det_counts.append(n_dets)
        frame_idx += 1

    cap.release()

    total_ids  = len(seen_ids)
    persistent = sum(1 for clss in track_history.values() if len(clss) >= 30)
    pers_ratio = persistent / max(total_ids, 1)
    mean_dets  = float(np.mean(det_counts)) if det_counts else 0.0

    person_tracks = sum(1 for tid, clss in track_history.items()
                        if max(set(clss), key=clss.count) == 0)
    car_tracks    = sum(1 for tid, clss in track_history.items()
                        if max(set(clss), key=clss.count) == 1)

    # composite score: reward stability, penalise spurious IDs and switches
    # pers_ratio already in [0,1]; normalise id_switches against total_ids
    switch_penalty = id_switches / max(total_ids, 1)
    score = pers_ratio - 0.5 * switch_penalty

    return dict(
        conf_car=conf_car,
        conf_person=conf_person,
        total_unique_ids=total_ids,
        id_switches=id_switches,
        mean_dets_per_frame=round(mean_dets, 2),
        persistent_tracks=persistent,
        persistent_ratio=round(pers_ratio, 3),
        person_tracks=person_tracks,
        car_tracks=car_tracks,
        score=round(score, 4),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-seconds", type=int, default=30,
                        help="Seconds of video to use per trial (default 30)")
    args = parser.parse_args()

    cap  = cv2.VideoCapture(str(VIDEO))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    end_frame = int(fps * args.video_seconds)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    combos = list(product(CONF_CAR_VALUES, CONF_PERSON_VALUES))
    print(f"Grid search: {len(combos)} combinations on {args.video_seconds}s clip")
    print(f"Weights: {WEIGHTS.name}  |  Video: {VIDEO.name}\n")

    model   = YOLO(str(WEIGHTS))
    results = []

    for i, (cc, cp) in enumerate(combos, 1):
        t0 = time.time()
        row = run_one(model, cc, cp, end_frame, fps)
        elapsed = time.time() - t0
        results.append(row)
        print(f"[{i:2d}/{len(combos)}] car={cc:.2f} person={cp:.2f} | "
              f"ids={row['total_unique_ids']:3d} switches={row['id_switches']:3d} "
              f"dets/f={row['mean_dets_per_frame']:.1f} "
              f"pers={row['persistent_ratio']:.2f} score={row['score']:.3f} "
              f"({elapsed:.0f}s)", flush=True)

    # sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    fields = list(results[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved → {OUT_CSV}")
    print("\nTop 5 combinations:")
    print(f"{'car':>6} {'person':>7} {'ids':>5} {'sw':>4} {'d/f':>5} "
          f"{'pers%':>6} {'score':>7}")
    print("-" * 50)
    for r in results[:5]:
        print(f"{r['conf_car']:6.2f} {r['conf_person']:7.2f} "
              f"{r['total_unique_ids']:5d} {r['id_switches']:4d} "
              f"{r['mean_dets_per_frame']:5.1f} "
              f"{r['persistent_ratio']:6.2f} {r['score']:7.3f}")


if __name__ == "__main__":
    main()
