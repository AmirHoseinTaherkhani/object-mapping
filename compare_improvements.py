"""
3-way comparison: old best.pt vs fine-tuned camera model vs fine-tuned + SAHI.
Each variant also applies per-class confidence thresholds (person↑, car↓).

Outputs:
  outputs/model_comparison/improvement_metrics.png  — bar charts + table
  outputs/model_comparison/improvement_comparison.mp4 — 3-panel side-by-side video
  (console) — full numeric summary

Run after fine-tuning completes:
    python compare_improvements.py
"""

import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

PROJECT = Path(__file__).parent
VIDEO   = PROJECT / "Demo/ANMR0006.mp4"

# Models
OLD_MODEL    = PROJECT / "models/weights/best.pt"
CAM_MODEL    = PROJECT / "models/weights/best_v3_camera.pt"

# Per-class thresholds (tuned to observed failure modes)
# SAHI zooms into patches so distant cars appear big — stricter threshold prevents FP.
# Low threshold only needed for full-frame where distant cars are tiny.
CONF_PERSON       = 0.45   # same for all modes
CONF_CAR_SAHI     = 0.42   # SAHI mode: patches zoom in, be strict
CONF_CAR_BASELINE = 0.18   # full-frame: distant cars are tiny, be permissive

# SAHI slicing
SLICE_H      = 512
SLICE_W      = 512
OVERLAP      = 0.25

# Video segment to test
START_FRAME  = 1000
DURATION_S   = 10    # quick sanity check; change to 30 for full run

CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (0, 200, 0), 1: (30, 120, 220)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def filter_cls_conf(predictions, conf_person=CONF_PERSON, conf_car=CONF_CAR_BASELINE):
    kept = []
    for obj in predictions:
        thresh = conf_person if obj.category.id == 0 else conf_car
        if obj.score.value >= thresh:
            kept.append(obj)
    return kept


def sahi_detect(det_model, frame_bgr, use_sahi: bool, conf_car=CONF_CAR_BASELINE):
    h, w = frame_bgr.shape[:2]
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if use_sahi:
        result = get_sliced_prediction(
            img_rgb, det_model,
            slice_height=SLICE_H, slice_width=SLICE_W,
            overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
            postprocess_type="NMM", verbose=0,
        )
    else:
        result = get_sliced_prediction(
            img_rgb, det_model,
            slice_height=h, slice_width=w,
            overlap_height_ratio=0.0, overlap_width_ratio=0.0,
            verbose=0,
        )
    return filter_cls_conf(result.object_prediction_list, conf_car=conf_car)


def yolo_track_detect(model, frame_bgr):
    """Old-model path using ultralytics ByteTrack (no SAHI, original thresholds)."""
    results = model.track(frame_bgr, persist=True, tracker="bytetrack.yaml",
                          conf=0.25, iou=0.45, verbose=False)
    boxes = results[0].boxes
    dets  = []
    if boxes is not None and boxes.id is not None:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf   = float(boxes.conf[i])
            tid    = int(boxes.id[i])
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            dets.append({"cls": cls_id, "conf": conf, "tid": tid,
                         "box": (x1, y1, x2, y2)})
    return dets


def draw_yolo_dets(frame, dets):
    vis = frame.copy()
    for d in dets:
        cls_id = d["cls"]
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"#{d['tid']} {CLASS_NAMES.get(cls_id,'')} {d['conf']:.2f}"
        cv2.putText(vis, label, (x1, max(y1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return vis


def draw_sahi_dets(frame, dets):
    vis = frame.copy()
    for obj in dets:
        cls_id = obj.category.id
        score  = obj.score.value
        b      = obj.bbox
        x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(cls_id,'')} {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return vis


def label_bar(w, text, color=(0, 220, 180)):
    bar = np.zeros((50, w, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)
    cv2.putText(bar, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)
    return bar


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert CAM_MODEL.exists(), (
        f"Fine-tuned model not found: {CAM_MODEL}\n"
        "Run finetune_on_camera.py first."
    )

    device = pick_device()
    print(f"Device: {device}", flush=True)

    cap     = cv2.VideoCapture(str(VIDEO))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    fw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(fps * DURATION_S)
    end_frame = START_FRAME + n_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    out_dir = PROJECT / "outputs/model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_w = fw // 2          # scale panels down so 3-panel fits on screen
    panel_h = int(fh * panel_w / fw)
    total_w = panel_w * 3

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    vid_out = cv2.VideoWriter(
        str(out_dir / "improvement_comparison.mp4"),
        fourcc, fps, (total_w, panel_h + 50)
    )

    # Load models
    print("Loading old model...", flush=True)
    old_yolo = YOLO(str(OLD_MODEL))

    conf_floor = min(CONF_PERSON, CONF_CAR_BASELINE)  # use lowest to not pre-filter valid detections
    print("Loading camera fine-tuned model (SAHI)...", flush=True)
    cam_sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(CAM_MODEL),
        confidence_threshold=conf_floor,
        device=device,
    )

    print("Loading camera fine-tuned model (baseline, no SAHI)...", flush=True)
    cam_base_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(CAM_MODEL),
        confidence_threshold=conf_floor,
        device=device,
    )

    # Stats collectors
    stats = {k: defaultdict(list) for k in ("old", "cam_base", "cam_sahi")}
    track_ids = {"old": set()}

    id_switches_old = 0
    seen_old = set()
    first_old = True

    print(f"\nProcessing {n_frames} frames ({DURATION_S}s)...\n", flush=True)
    t0 = time.time()

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = fi / fps

        # --- Panel A: old best.pt + ByteTrack (no per-class thresh) ---
        dets_old = yolo_track_detect(old_yolo, frame)
        for d in dets_old:
            if d["tid"] not in seen_old:
                if not first_old:
                    id_switches_old += 1
                seen_old.add(d["tid"])
        first_old = False

        stats["old"]["n_person"].append(sum(1 for d in dets_old if d["cls"] == 0))
        stats["old"]["n_car"].append(sum(1 for d in dets_old if d["cls"] == 1))
        stats["old"]["n_total"].append(len(dets_old))

        vis_old = draw_yolo_dets(frame, dets_old)
        vis_old = cv2.resize(vis_old, (panel_w, panel_h))

        # --- Panel B: cam fine-tuned, no SAHI, per-class thresh ---
        dets_base = sahi_detect(cam_base_model, frame, use_sahi=False, conf_car=CONF_CAR_BASELINE)
        stats["cam_base"]["n_person"].append(sum(1 for d in dets_base if d.category.id == 0))
        stats["cam_base"]["n_car"].append(sum(1 for d in dets_base if d.category.id == 1))
        stats["cam_base"]["n_total"].append(len(dets_base))

        vis_base = draw_sahi_dets(frame, dets_base)
        vis_base = cv2.resize(vis_base, (panel_w, panel_h))

        # --- Panel C: cam fine-tuned + SAHI + per-class thresh ---
        dets_sahi = sahi_detect(cam_sahi_model, frame, use_sahi=True, conf_car=CONF_CAR_SAHI)
        stats["cam_sahi"]["n_person"].append(sum(1 for d in dets_sahi if d.category.id == 0))
        stats["cam_sahi"]["n_car"].append(sum(1 for d in dets_sahi if d.category.id == 1))
        stats["cam_sahi"]["n_total"].append(len(dets_sahi))

        vis_sahi = draw_sahi_dets(frame, dets_sahi)
        vis_sahi = cv2.resize(vis_sahi, (panel_w, panel_h))

        # Compose 3-panel
        row = np.hstack([vis_old, vis_base, vis_sahi])
        bar_txt = (f"t={elapsed:.1f}s  |  "
                   f"OLD: {len(dets_old)}det  |  "
                   f"CAM: {len(dets_base)}det  |  "
                   f"CAM+SAHI: {len(dets_sahi)}det")
        bar = label_bar(total_w, bar_txt)
        vid_out.write(np.vstack([row, bar]))

        if (fi + 1) % 300 == 0:
            elapsed_real = time.time() - t0
            print(f"  {fi+1}/{n_frames}  fps_real={fi/elapsed_real:.1f}  "
                  f"old={np.mean(stats['old']['n_total']):.1f}  "
                  f"cam={np.mean(stats['cam_base']['n_total']):.1f}  "
                  f"sahi={np.mean(stats['cam_sahi']['n_total']):.1f}")

    cap.release()
    vid_out.release()

    # --- Summary table ---
    print(f"\n{'='*65}", flush=True)
    print(f"{'Metric':<28} {'OLD':>10} {'CAM':>10} {'CAM+SAHI':>10}", flush=True)
    print(f"{'-'*65}", flush=True)

    def avg(key, sub): return np.mean(stats[key][sub]) if stats[key][sub] else 0

    rows = [
        ("Avg total dets/frame",  avg("old","n_total"),  avg("cam_base","n_total"),  avg("cam_sahi","n_total")),
        ("Avg person dets/frame", avg("old","n_person"), avg("cam_base","n_person"), avg("cam_sahi","n_person")),
        ("Avg car dets/frame",    avg("old","n_car"),    avg("cam_base","n_car"),    avg("cam_sahi","n_car")),
        ("Track ID switches",     id_switches_old,       "N/A",                      "N/A"),
    ]
    for label, v_old, v_cam, v_sahi in rows:
        def fmt(v): return f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"  {label:<26} {fmt(v_old):>10} {fmt(v_cam):>10} {fmt(v_sahi):>10}", flush=True)

    print(f"{'='*65}", flush=True)

    # --- Metrics plot ---
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle("Detection improvement: OLD vs Camera Fine-tuned vs +SAHI",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    labels   = ["OLD\nbest.pt", "CAM\nfine-tuned", "CAM\n+SAHI"]
    colors   = ["#e07b54", "#4e9af1", "#5cb85c"]

    def bar_ax(ax, values, title, ylabel="avg dets/frame"):
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02 * max(values + [0.01]),
                    f"{val:.2f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(values + [0.01]) * 1.25)
        ax.tick_params(labelsize=8)

    vals_total  = [avg("old","n_total"),  avg("cam_base","n_total"),  avg("cam_sahi","n_total")]
    vals_person = [avg("old","n_person"), avg("cam_base","n_person"), avg("cam_sahi","n_person")]
    vals_car    = [avg("old","n_car"),    avg("cam_base","n_car"),    avg("cam_sahi","n_car")]

    bar_ax(fig.add_subplot(gs[0, 0]), vals_total,  "Total detections/frame")
    bar_ax(fig.add_subplot(gs[0, 1]), vals_person, "Person detections/frame")
    bar_ax(fig.add_subplot(gs[0, 2]), vals_car,    "Car detections/frame")

    # Per-frame line plots
    ax_p = fig.add_subplot(gs[1, :2])
    frames_x = list(range(len(stats["old"]["n_car"])))
    ax_p.plot(frames_x, stats["old"]["n_car"],      color=colors[0], alpha=0.7, lw=1, label="OLD")
    ax_p.plot(frames_x, stats["cam_base"]["n_car"], color=colors[1], alpha=0.7, lw=1, label="CAM")
    ax_p.plot(frames_x, stats["cam_sahi"]["n_car"], color=colors[2], alpha=0.7, lw=1, label="CAM+SAHI")
    ax_p.set_title("Car detections per frame (key failure mode)", fontsize=10)
    ax_p.set_xlabel("Frame", fontsize=8)
    ax_p.set_ylabel("# cars", fontsize=8)
    ax_p.legend(fontsize=8)
    ax_p.tick_params(labelsize=8)

    # Summary text box
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis("off")
    summary_lines = [
        f"Thresholds:",
        f"  person conf    ≥ {CONF_PERSON}",
        f"  car (base)     ≥ {CONF_CAR_BASELINE}",
        f"  car (SAHI)     ≥ {CONF_CAR_SAHI}",
        f"",
        f"SAHI slice: {SLICE_H}×{SLICE_W}",
        f"SAHI overlap: {int(OVERLAP*100)}%",
        f"",
        f"OLD id_switches: {id_switches_old}",
        f"(30s window)",
    ]
    ax_t.text(0.05, 0.95, "\n".join(summary_lines),
              transform=ax_t.transAxes,
              fontsize=9, va="top", fontfamily="monospace",
              bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plot_path = out_dir / "improvement_metrics.png"
    plt.savefig(str(plot_path), dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\nPlot  → {plot_path}", flush=True)
    print(f"Video → {out_dir / 'improvement_comparison.mp4'}", flush=True)


if __name__ == "__main__":
    main()
