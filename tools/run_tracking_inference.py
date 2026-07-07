"""
run_tracking_inference.py

YOLO11 + BoT-SORT tracking visualization.  Tracking only — no counting,
no ghost zones.  Use this to visually compare how each model tracks across
all demo videos before deciding on counting parameters.

Output structure (batch mode, no reid):
  experiments/yolo11_experiments/tracking_inference/
    pretrained_yolo11s/   ← COCO-pretrained yolo11s, person/car remapped
    pretrained_yolo11m/
    pretrained_yolo11l/
    pretrained_yolo11x/
    finetuned_yolo11s/    ← fine-tuned on our 2-class dataset
    finetuned_yolo11m/
    finetuned_yolo11l/
    finetuned_yolo11x/

Output structure (--reid, ANMR0006 only):
  experiments/yolo11_experiments/tracking_inference/
    reid/
      pretrained_yolo11s/ANMR0006.mp4
      pretrained_yolo11m/ANMR0006.mp4
      ...
      finetuned_yolo11x/ANMR0006.mp4

Each subfolder contains one .mp4 per video, annotated with bounding boxes
and track IDs.  Existing files are never overwritten.

Single-video mode:
  python tools/run_tracking_inference.py \\
    --weights models/weights/yolo11s.pt \\
    --source Demo/ANMR0006.mp4 \\
    --pretrained \\
    --output path/to/output.mp4

Batch mode (all 8 models × all Demo videos, skips existing):
  python tools/run_tracking_inference.py --batch
  python tools/run_tracking_inference.py --batch --device mps
  python tools/run_tracking_inference.py --batch --resize 1280

ReID comparison (all 8 models on ANMR0006.mp4, appearance embedding enabled):
  python tools/run_tracking_inference.py --batch --reid --device mps --resize 1280
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT         = Path(__file__).parent.parent
TRACKING_OUT = ROOT / "experiments" / "yolo11_experiments" / "tracking_inference"
DEMO_DIR     = ROOT / "Demo"

CLASS_PERSON = 0
CLASS_CAR    = 1
COLORS       = {CLASS_CAR: (0, 220, 0), CLASS_PERSON: (255, 100, 0)}
LABELS       = {CLASS_CAR: "car", CLASS_PERSON: "person"}

# COCO 0-indexed class → our 2-class schema (for pretrained mode)
COCO_TO_OURS = {
    0: CLASS_PERSON,   # person
    2: CLASS_CAR,      # car
    3: CLASS_CAR,      # motorcycle
    5: CLASS_CAR,      # bus
    7: CLASS_CAR,      # truck
}

# ── batch configuration ────────────────────────────────────────────────────────

def _w(rel):
    return ROOT / rel

BATCH_MODELS = [
    {
        "tag":       "pretrained_yolo11s",
        "weights":   _w("models/weights/yolo11s.pt"),
        "fallback":  "yolo11s.pt",
        "pretrained": True,
    },
    {
        "tag":       "pretrained_yolo11m",
        "weights":   _w("models/weights/yolo11m.pt"),
        "fallback":  "yolo11m.pt",
        "pretrained": True,
    },
    {
        "tag":       "pretrained_yolo11l",
        "weights":   _w("models/weights/yolo11l.pt"),
        "fallback":  "yolo11l.pt",
        "pretrained": True,
    },
    {
        "tag":       "pretrained_yolo11x",
        "weights":   _w("models/weights/yolo11x.pt"),
        "fallback":  "yolo11x.pt",
        "pretrained": True,
    },
    {
        "tag":       "finetuned_yolo11s",
        "weights":   _w("experiments/yolo11_experiments/finetuned/yolo11s/train/weights/best.pt"),
        "pretrained": False,
    },
    {
        "tag":       "finetuned_yolo11m",
        "weights":   _w("experiments/yolo11_experiments/finetuned/yolo11m/train/weights/best.pt"),
        "pretrained": False,
    },
    {
        "tag":       "finetuned_yolo11l",
        "weights":   _w("experiments/yolo11_experiments/finetuned/yolo11l/train/weights/best.pt"),
        "pretrained": False,
    },
    {
        "tag":       "finetuned_yolo11x",
        "weights":   _w("experiments/yolo11_experiments/finetuned/yolo11x/train/weights/best.pt"),
        "pretrained": False,
    },
]

def discover_videos():
    vids = []
    for p in sorted(DEMO_DIR.glob("*.mp4")) + sorted(DEMO_DIR.glob("*.avi")):
        vids.append(p)
    for p in sorted((DEMO_DIR / "videos").glob("*.mp4")) + \
             sorted((DEMO_DIR / "videos").glob("*.avi")):
        vids.append(p)
    return vids


# ── tracker ───────────────────────────────────────────────────────────────────

REID_WEIGHTS = "osnet_x0_25_msmt17.pt"   # pre-installed with boxmot

def build_tracker(fps: float, with_reid: bool = False, reid_device: str = "cpu"):
    from boxmot.trackers.botsort.botsort import BotSort
    reid_model = None
    if with_reid:
        from boxmot.reid.core.reid import ReID
        reid_obj   = ReID(weights=REID_WEIGHTS, device=reid_device)
        reid_model = reid_obj.model   # BotSort expects the backend, not the wrapper
    return BotSort(
        reid_model=reid_model,
        with_reid=with_reid,
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=600,
        frame_rate=int(fps),
        cmc_method="sof",
    )


# ── single-video inference ────────────────────────────────────────────────────

def run_one(weights_path, source_path, out_path, is_pretrained,
            device="cpu", resize_w=0, conf_person=0.35, conf_car=0.25,
            with_reid=False, reid_device="cpu"):
    """
    Run YOLO11 + BoT-SORT on one video.  Saves annotated video to out_path.
    Returns True on success, False on error.
    """
    if out_path.exists():
        print(f"  [SKIP] already exists: {out_path.name}")
        return True

    # ── load model ────────────────────────────────────────────────────────────
    wstr = str(weights_path)
    print(f"  Loading {wstr} …", end=" ", flush=True)
    try:
        model = YOLO(wstr)
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    print("ok")

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {source_path}")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if resize_w and w_orig > resize_w:
        scale  = resize_w / w_orig
        inf_w  = resize_w
        inf_h  = int(h_orig * scale)
    else:
        scale = 1.0
        inf_w  = w_orig
        inf_h  = h_orig

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_orig, h_orig))

    tracker   = build_tracker(fps, with_reid=with_reid, reid_device=reid_device)
    fps_times: deque = deque(maxlen=60)
    last_dets = np.empty((0, 6), dtype=np.float32)
    frame_idx = 0
    t_start   = time.perf_counter()

    reid_tag = f"  reid=osnet_x0_25 ({reid_device})" if with_reid else "  reid=off"
    print(f"  Source: {source_path.name}  {w_orig}×{h_orig} @ {fps:.0f}fps"
          f"  frames≈{total_frames}  inf-size={inf_w}×{inf_h}"
          f"  device={device}{reid_tag}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_now = time.perf_counter()
        fps_times.append(t_now)
        live_fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 1e-6)

        # ── inference ─────────────────────────────────────────────────────────
        inf_frame = cv2.resize(frame, (inf_w, inf_h)) if scale != 1.0 else frame
        results   = model(inf_frame, conf=0.10, device=device, verbose=False)[0]
        boxes_np  = results.boxes
        dets = []
        if boxes_np is not None and len(boxes_np):
            for i in range(len(boxes_np)):
                raw_cls = int(boxes_np.cls[i])
                conf    = float(boxes_np.conf[i])

                if is_pretrained:
                    our_cls = COCO_TO_OURS.get(raw_cls)
                    if our_cls is None:
                        continue
                else:
                    if raw_cls not in (CLASS_PERSON, CLASS_CAR):
                        continue
                    our_cls = raw_cls

                thresh = conf_person if our_cls == CLASS_PERSON else conf_car
                if conf < thresh:
                    continue

                x1, y1, x2, y2 = boxes_np.xyxy[i].tolist()
                # scale coords back to original resolution if we resized
                if scale != 1.0:
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                dets.append([x1, y1, x2, y2, conf, our_cls])

        last_dets = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

        # ── tracker update ────────────────────────────────────────────────────
        # Pass full-res frame so BoT-SORT's CMC (camera motion compensation)
        # operates on the original resolution.
        tracks = tracker.update(last_dets, frame)

        annotated = frame.copy()
        person_count = 0
        car_count    = 0

        if tracks is not None and len(tracks):
            for i in range(len(tracks)):
                x1, y1, x2, y2 = [int(v) for v in tracks.xyxy[i]]
                tid  = int(tracks.id[i])
                cls  = int(tracks.cls[i])
                color = COLORS.get(cls, (200, 200, 200))
                name  = LABELS.get(cls, "obj")
                label = f"{name} #{tid}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                if cls == CLASS_CAR:
                    car_count += 1
                else:
                    person_count += 1

        # ── overlay (model tag + live stats) ──────────────────────────────────
        tag_label = out_path.parent.name   # e.g. "finetuned_yolo11m"
        reid_line = "ReID  : osnet_x0_25 ON" if with_reid else "ReID  : off"
        overlay_lines = [
            f"Model : {tag_label}",
            f"Cars  : {car_count}  People: {person_count}",
            f"FPS   : {live_fps:.1f}   frame {frame_idx}",
            reid_line,
        ]
        box_h = 30 * len(overlay_lines) + 10
        cv2.rectangle(annotated, (5, 5), (420, box_h), (20, 20, 20), -1)
        for li, text in enumerate(overlay_lines):
            cv2.putText(annotated, text, (12, 28 + li * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 600 == 0:
            elapsed = time.perf_counter() - t_start
            pct     = 100 * frame_idx / max(total_frames, 1)
            eta     = (elapsed / frame_idx) * (total_frames - frame_idx)
            print(f"    frame {frame_idx}/{total_frames} ({pct:.0f}%)  "
                  f"fps={live_fps:.1f}  ETA {eta:.0f}s")

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - t_start
    print(f"  Done: {frame_idx} frames in {elapsed:.0f}s → {out_path}")
    return True


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch",       action="store_true",
                   help="Run all 8 models × all Demo/ videos (skip existing)")
    p.add_argument("--weights",     help="Path to .pt weights (single-video mode)")
    p.add_argument("--source",      help="Video path (single-video mode)")
    p.add_argument("--pretrained",  action="store_true",
                   help="Remap COCO classes to our 2-class schema")
    p.add_argument("--output",      help="Output video path (single-video mode)")
    p.add_argument("--tag",         default=None,
                   help="Subfolder name under tracking_inference/ (single-video mode)")
    p.add_argument("--device",      default="cpu",
                   help="Inference device: cpu, mps, cuda:0 (default: cpu)")
    p.add_argument("--resize",      type=int, default=0,
                   help="Resize width for inference (0 = original size). "
                        "E.g. 1280 speeds up inference on 4K videos.")
    p.add_argument("--conf-car",    type=float, default=0.25)
    p.add_argument("--conf-person", type=float, default=0.35)
    p.add_argument("--reid",        action="store_true",
                   help="Enable appearance re-ID (OSNet x0.25 MSMT17). "
                        "In batch mode, runs only on ANMR0006.mp4. "
                        "Outputs go to tracking_inference/reid/<model>/")
    p.add_argument("--reid-device", default="cpu",
                   help="Device for the re-ID model (default: cpu). "
                        "Use 'mps' or 'cuda:0' for GPU.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.batch:
        # ── batch mode ────────────────────────────────────────────────────────
        if args.reid:
            # ReID mode: only ANMR0006.mp4, output under tracking_inference/reid/
            videos   = [DEMO_DIR / "ANMR0006.mp4"]
            out_root = TRACKING_OUT / "reid"
            print(f"ReID mode: running all 8 models on ANMR0006.mp4 only")
            print(f"Re-ID model: OSNet x0.25 MSMT17  reid_device={args.reid_device}")
        else:
            videos   = discover_videos()
            out_root = TRACKING_OUT
            print(f"Found {len(videos)} videos under {DEMO_DIR}")
            print(f"Running {len(BATCH_MODELS)} models × {len(videos)} videos "
                  f"= {len(BATCH_MODELS) * len(videos)} combinations")

        print(f"Device: {args.device}  resize: {args.resize or 'off'}")
        print()

        total_skipped = 0
        total_done    = 0
        total_errors  = 0

        for m in BATCH_MODELS:
            tag     = m["tag"]
            w_path  = m["weights"]
            is_pre  = m["pretrained"]
            fallback = m.get("fallback", None)

            # resolve weights: prefer local file, fall back to ultralytics auto-dl
            if not w_path.exists():
                if fallback:
                    print(f"\n{'='*60}")
                    print(f"Model: {tag}  (weights not in models/weights/ — "
                          f"ultralytics will auto-download {fallback})")
                    print(f"{'='*60}")
                    w_resolved = fallback
                else:
                    print(f"\n[SKIP model] weights not found: {w_path}")
                    continue
            else:
                print(f"\n{'='*60}")
                print(f"Model: {tag}  weights: {w_path.name}")
                print(f"{'='*60}")
                w_resolved = w_path

            out_dir = out_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for vid in videos:
                out_file = out_dir / (vid.stem + ".mp4")
                if out_file.exists():
                    print(f"  [SKIP] {tag}/{out_file.name}")
                    total_skipped += 1
                    continue

                print(f"\n  → {tag}/{vid.stem}")
                ok = run_one(
                    weights_path  = Path(w_resolved) if isinstance(w_resolved, str) and Path(w_resolved).exists() else w_resolved,
                    source_path   = vid,
                    out_path      = out_file,
                    is_pretrained = is_pre,
                    device        = args.device,
                    resize_w      = args.resize,
                    conf_person   = args.conf_person,
                    conf_car      = args.conf_car,
                    with_reid     = args.reid,
                    reid_device   = args.reid_device,
                )
                if ok:
                    total_done += 1
                else:
                    total_errors += 1

        print(f"\n{'='*60}")
        print(f"Batch complete.  done={total_done}  skipped={total_skipped}  errors={total_errors}")
        print(f"Output root: {out_root}")

    else:
        # ── single-video mode ─────────────────────────────────────────────────
        if not args.weights or not args.source:
            sys.exit("Single-video mode requires --weights and --source. "
                     "Use --batch to process all models × all videos.")
        w_path = Path(args.weights)
        s_path = Path(args.source)

        if args.output:
            out_path = Path(args.output)
        elif args.tag:
            out_path = TRACKING_OUT / args.tag / (s_path.stem + ".mp4")
        else:
            out_path = TRACKING_OUT / w_path.stem / (s_path.stem + ".mp4")

        ok = run_one(
            weights_path  = w_path,
            source_path   = s_path,
            out_path      = out_path,
            is_pretrained = args.pretrained,
            device        = args.device,
            resize_w      = args.resize,
            conf_person   = args.conf_person,
            conf_car      = args.conf_car,
            with_reid     = args.reid,
            reid_device   = args.reid_device,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
