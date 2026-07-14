"""
ReID model comparison — tracking inference with 6 different ReID embeddings.

All experiments use the same detector (finetuned YOLO11x) and tracker (BoT-SORT).
Only the ReID embedding model is swapped so results are directly comparable.

Models compared
---------------
  osnet_x0_25_msmt17    OSNet x0.25 — tiny, baseline (pre-installed)
  osnet_ain_x1_0_msmt17 OSNet AIN x1.0 — full-width with attentive IN (Beta=1.0)
  lmbn_n_duke           LightMBN — Light Multiple-Branch Network
  hacnn_msmt17          HACNN — Harmonious Attention CNN
  mlfn_msmt17           MLFN — Multi-Level Factorisation Net
  clip_market1501       CLIP-ReID — ViT backbone, vision-language pre-training

Videos
------
  Demo/ANMR0006_1min.mp4      first 60s of the main surveillance recording
  Demo/videos/*.mp4            10 public Pexels benchmark clips

Output
------
  experiments/yolo11_experiments/tracking_inference/reid_comparison/
  └── <model_name>/
      └── <video_stem>.mp4

Usage
-----
  conda activate objectmapping
  python tools/run_reid_comparison.py

  # Skip already-finished videos (safe to interrupt and resume)
  python tools/run_reid_comparison.py

  # Single model only
  python tools/run_reid_comparison.py --model osnet_ain_x1_0_msmt17
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

ROOT     = Path(__file__).parent.parent
DEMO     = ROOT / "Demo"
OUT_BASE = ROOT / "experiments" / "yolo11_experiments" / "tracking_inference" / "reid_comparison"

DETECTOR_WEIGHTS = (
    ROOT / "experiments" / "yolo11_experiments" / "finetuned" /
    "yolo11x" / "train" / "weights" / "best.pt"
)

# ReID model name → weight filename (all resolved via boxmot's WEIGHTS dir + auto-download)
REID_MODELS = {
    "osnet_x0_25_msmt17":    "osnet_x0_25_msmt17.pt",    # pre-installed, ~2 MB
    "osnet_ain_x1_0_msmt17": "osnet_ain_x1_0_msmt17.pt", # ~39 MB
    "lmbn_n_duke":           "lmbn_n_duke.pt",            # ~50 MB
    "hacnn_msmt17":          "hacnn_msmt17.pt",           # ~4 MB
    "mlfn_msmt17":           "mlfn_msmt17.pt",            # ~51 MB
    "clip_market1501":       "clip_market1501.pt",        # ~340 MB — slowest (ViT)
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_videos() -> list[Path]:
    videos = []
    anmr_1min = DEMO / "ANMR0006_1min.mp4"
    if anmr_1min.exists():
        videos.append(anmr_1min)
    else:
        print(f"WARNING: {anmr_1min} not found — run: "
              "ffmpeg -i Demo/ANMR0006.mp4 -t 60 -c copy Demo/ANMR0006_1min.mp4")

    vid_dir = DEMO / "videos"
    if vid_dir.is_dir():
        videos.extend(sorted(p for p in vid_dir.iterdir()
                             if p.suffix.lower() in VIDEO_EXTS))
    return videos


def build_tracker(fps: float, reid_model_backend):
    from boxmot.trackers.botsort.botsort import BotSort
    return BotSort(
        reid_model=reid_model_backend,
        with_reid=True,
        track_high_thresh=0.25,
        track_low_thresh=0.10,
        new_track_thresh=0.25,
        track_buffer=600,
        frame_rate=int(fps),
        cmc_method="sof",
    )


def load_reid(weight_filename: str, device: str = "cpu"):
    """Load a ReID model by weight filename; boxmot auto-downloads if missing."""
    from boxmot.reid.core.reid import ReID
    from boxmot.utils import WEIGHTS
    path = WEIGHTS / weight_filename
    print(f"  Loading ReID weights: {path}")
    reid_obj = ReID(weights=path, device=device)
    return reid_obj.model   # BotSort needs the backend, not the wrapper


COLORS = {}

def track_color(track_id: int) -> tuple:
    if track_id not in COLORS:
        rng = np.random.default_rng(track_id * 17 + 3)
        COLORS[track_id] = tuple(int(c) for c in rng.integers(80, 220, 3))
    return COLORS[track_id]


def process_video(detector, reid_backend, video_path: Path, out_path: Path,
                  device: str = "mps", resize: int = 1280) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    tracker = build_tracker(fps, reid_backend)

    pbar = tqdm(total=total, desc=f"    {video_path.name}", unit="fr",
                leave=False, ncols=90)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = detector.predict(frame, verbose=False, conf=0.25, iou=0.45,
                                   imgsz=resize, device=device)
        dets = results[0].boxes

        if dets is not None and len(dets):
            det_array = np.column_stack([
                dets.xyxy.cpu().numpy(),
                dets.conf.cpu().numpy(),
                dets.cls.cpu().numpy(),
            ])
        else:
            det_array = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update(det_array, frame)

        # Draw tracks
        for t in tracks:
            x1, y1, x2, y2 = map(int, t[:4])
            tid  = int(t[4])
            cls  = int(t[6]) if len(t) > 6 else 0
            label = ("person" if cls == 0 else "car")
            color = track_color(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} #{tid}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        writer.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()


def preflight_check():
    """Verify detector weights exist before starting."""
    if not DETECTOR_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Detector weights not found: {DETECTOR_WEIGHTS}\n"
            "Sync finetuned weights from HPC first."
        )
    vids = get_videos()
    if not vids:
        raise FileNotFoundError("No videos found. Check Demo/ directory.")
    print(f"Detector : {DETECTOR_WEIGHTS.name}")
    print(f"Videos   : {len(vids)}")
    print(f"Output   : {OUT_BASE}")
    return vids


def main():
    parser = argparse.ArgumentParser(description="ReID model comparison — tracking inference")
    parser.add_argument("--model",  default=None,
                        help="Run only this ReID model key (default: all)")
    parser.add_argument("--device", default="mps",
                        help="Detector device (mps / cpu / cuda:0)")
    parser.add_argument("--reid-device", default="cpu",
                        help="ReID device (cpu recommended)")
    parser.add_argument("--resize", type=int, default=1280,
                        help="Detector input size")
    args = parser.parse_args()

    videos = preflight_check()

    models_to_run = (
        {args.model: REID_MODELS[args.model]}
        if args.model else REID_MODELS
    )

    if args.model and args.model not in REID_MODELS:
        raise ValueError(f"Unknown model '{args.model}'. Choose from: {list(REID_MODELS)}")

    print(f"\nLoading detector …")
    detector = YOLO(str(DETECTOR_WEIGHTS))
    print(f"Detector loaded: {DETECTOR_WEIGHTS.name}\n")

    model_bar = tqdm(models_to_run.items(), desc="ReID models", unit="model",
                     ncols=90, position=0)

    for model_key, weight_file in model_bar:
        model_bar.set_description(f"ReID: {model_key}")
        out_dir = OUT_BASE / model_key

        # ── load ReID embedding model ──────────────────────────────────────────
        try:
            reid_backend = load_reid(weight_file, device=args.reid_device)
        except Exception as e:
            tqdm.write(f"  [SKIP] Failed to load {weight_file}: {e}")
            continue

        # ── process each video ─────────────────────────────────────────────────
        video_bar = tqdm(videos, desc="  Videos", unit="vid",
                         ncols=90, position=1, leave=False)

        for video_path in video_bar:
            out_path = out_dir / (video_path.stem + ".mp4")

            if out_path.exists():
                tqdm.write(f"  [SKIP] {out_path.relative_to(ROOT)} already exists")
                continue

            video_bar.set_description(f"  {video_path.name[:35]}")
            t0 = time.time()

            try:
                process_video(detector, reid_backend, video_path, out_path,
                              device=args.device, resize=args.resize)
                elapsed = time.time() - t0
                size_mb = out_path.stat().st_size / 1e6
                tqdm.write(f"  ✓ {out_path.relative_to(ROOT)}  "
                           f"({size_mb:.0f} MB, {elapsed:.0f}s)")
            except Exception as e:
                tqdm.write(f"  [ERROR] {video_path.name}: {e}")

        video_bar.close()

    model_bar.close()
    print("\nDone. All videos written to:")
    print(f"  {OUT_BASE}")


if __name__ == "__main__":
    main()
