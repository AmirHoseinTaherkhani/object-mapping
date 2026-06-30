"""
Model performance comparison: best.pt (old) vs best_v3.pt (new).
Runs both models on the test video with ByteTrack and generates charts.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

PROJECT = Path(__file__).parent
VIDEO = PROJECT / "Demo/ANMR0006.mp4"
OLD_WEIGHTS = PROJECT / "models/weights/best.pt"
NEW_WEIGHTS = PROJECT / "models/weights/best_v3.pt"
OUT_DIR = PROJECT / "outputs/model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.25
START_FRAME = 1000   # skip empty opening section
MAX_FRAMES = 1000    # ~16s at 60fps

CLASS_NAMES = {0: "person", 1: "car"}
COLORS = {"old": "#e74c3c", "new": "#2ecc71"}


def run_tracking(weights_path: Path, label: str) -> dict:
    print(f"\n[{label}] Loading {weights_path.name}...")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    frame_detections = []   # list of {conf, cls, track_id} per detection per frame
    seen_tracks: set = set()
    track_id_switches = 0  # new track IDs appearing after frame 0
    prev_track_ids: set = set()
    frame_idx = 0

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            iou=0.45,
            verbose=False,
        )

        frame_data = []
        current_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                tid = int(boxes.id[i])
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                frame_data.append({"conf": conf, "cls": cls, "track_id": tid})
                current_ids.add(tid)
                if tid not in seen_tracks:
                    if frame_idx > 0:
                        track_id_switches += 1
                    seen_tracks.add(tid)

        prev_track_ids = current_ids
        frame_detections.append(frame_data)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  [{label}] {frame_idx}/{MAX_FRAMES} frames processed")

    cap.release()
    model.predictor = None  # reset tracker state

    # aggregate
    det_per_frame = [len(fd) for fd in frame_detections]
    all_confs = [d["conf"] for fd in frame_detections for d in fd]
    all_cls = [d["cls"] for fd in frame_detections for d in fd]

    person_confs = [d["conf"] for fd in frame_detections for d in fd if d["cls"] == 0]
    car_confs = [d["conf"] for fd in frame_detections for d in fd if d["cls"] == 1]

    unique_tracks_person = len({d["track_id"] for fd in frame_detections for d in fd if d["cls"] == 0})
    unique_tracks_car = len({d["track_id"] for fd in frame_detections for d in fd if d["cls"] == 1})

    return {
        "label": label,
        "det_per_frame": det_per_frame,
        "all_confs": all_confs,
        "person_confs": person_confs,
        "car_confs": car_confs,
        "cls_counts": {0: all_cls.count(0), 1: all_cls.count(1)},
        "total_detections": len(all_confs),
        "mean_conf": np.mean(all_confs) if all_confs else 0,
        "unique_tracks_person": unique_tracks_person,
        "unique_tracks_car": unique_tracks_car,
        "total_unique_tracks": len(seen_tracks),
        "track_id_switches": track_id_switches,
        "fps": fps,
        "total_frames": total,
    }


def save_comparison_frames(old_weights, new_weights, out_path):
    """Save a side-by-side frame grid at a few sample points."""
    models = [YOLO(str(old_weights)), YOLO(str(new_weights))]
    labels = ["Old (best.pt)", "New (best_v3.pt)"]

    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = [1000, 1500, 2000, 3000]

    rows = []
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        row_imgs = []
        for model, lbl in zip(models, labels):
            res = model.predict(frame, conf=0.25, verbose=False)
            annotated = res[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            # add label bar
            h, w = annotated_rgb.shape[:2]
            bar = np.zeros((40, w, 3), dtype=np.uint8)
            bar[:] = (50, 50, 50)
            cv2.putText(bar, lbl, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            row_imgs.append(np.vstack([bar, annotated_rgb]))
        rows.append(np.hstack(row_imgs))
    cap.release()

    if rows:
        grid = np.vstack(rows)
        fig, ax = plt.subplots(figsize=(18, len(rows) * 5))
        ax.imshow(grid)
        ax.axis("off")
        ax.set_title("Detection comparison: Old vs New model", fontsize=14, pad=10)
        fig.tight_layout()
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved frame comparison → {out_path}")


def plot_metrics(old: dict, new: dict, out_path: Path):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Model Performance Comparison — best.pt (old, Oct-2024) vs best_v3.pt (new, May-2026)\n"
        f"Video: Demo/ANMR0006.mp4 | Frames {START_FRAME}–{START_FRAME+MAX_FRAMES} | conf≥{CONF_THRESHOLD}",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    frames = list(range(len(old["det_per_frame"])))

    # ── 1. Detections per frame over time ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(frames, old["det_per_frame"], color=COLORS["old"], alpha=0.7, linewidth=1, label="Old")
    ax1.plot(frames, new["det_per_frame"], color=COLORS["new"], alpha=0.7, linewidth=1, label="New")
    # rolling avg
    window = 10
    if len(frames) >= window:
        old_roll = np.convolve(old["det_per_frame"], np.ones(window)/window, mode="valid")
        new_roll = np.convolve(new["det_per_frame"], np.ones(window)/window, mode="valid")
        ax1.plot(range(window-1, len(frames)), old_roll, color=COLORS["old"], linewidth=2.5)
        ax1.plot(range(window-1, len(frames)), new_roll, color=COLORS["new"], linewidth=2.5)
    ax1.set_title("Detections per Frame")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("# Detections")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── 2. Summary bar chart ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ["Total\nDetections", "Mean\nConfidence", "Unique\nTracks"]
    old_vals = [old["total_detections"], old["mean_conf"] * 100, old["total_unique_tracks"]]
    new_vals = [new["total_detections"], new["mean_conf"] * 100, new["total_unique_tracks"]]
    x = np.arange(len(metrics))
    w = 0.35
    bars_old = ax2.bar(x - w/2, old_vals, w, label="Old", color=COLORS["old"], alpha=0.85)
    bars_new = ax2.bar(x + w/2, new_vals, w, label="New", color=COLORS["new"], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_title("Summary Metrics")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for bar in list(bars_old) + list(bars_new):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)

    # ── 3. Confidence distribution (box) ───────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    data = [old["all_confs"], new["all_confs"]]
    bp = ax3.boxplot(data, labels=["Old", "New"], patch_artist=True,
                     medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(COLORS["old"])
    bp["boxes"][1].set_facecolor(COLORS["new"])
    ax3.set_title("Confidence Distribution\n(All classes)")
    ax3.set_ylabel("Confidence")
    ax3.grid(axis="y", alpha=0.3)

    # ── 4. Per-class confidence ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    person_data = [old["person_confs"] or [0], new["person_confs"] or [0]]
    car_data = [old["car_confs"] or [0], new["car_confs"] or [0]]
    positions = [1, 2, 3.5, 4.5]
    bp2 = ax4.boxplot(person_data + car_data, positions=positions,
                      widths=0.6, patch_artist=True,
                      medianprops=dict(color="white", linewidth=2))
    colors_list = [COLORS["old"], COLORS["new"], COLORS["old"], COLORS["new"]]
    for patch, c in zip(bp2["boxes"], colors_list):
        patch.set_facecolor(c)
    ax4.set_xticks([1.5, 4])
    ax4.set_xticklabels(["Person", "Car"])
    ax4.set_title("Confidence by Class")
    ax4.set_ylabel("Confidence")
    ax4.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax4.legend(handles=[Patch(color=COLORS["old"], label="Old"),
                         Patch(color=COLORS["new"], label="New")], fontsize=8)

    # ── 5. Class distribution ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    classes = ["Person", "Car"]
    old_cls = [old["cls_counts"].get(0, 0), old["cls_counts"].get(1, 0)]
    new_cls = [new["cls_counts"].get(0, 0), new["cls_counts"].get(1, 0)]
    x = np.arange(len(classes))
    ax5.bar(x - w/2, old_cls, w, label="Old", color=COLORS["old"], alpha=0.85)
    ax5.bar(x + w/2, new_cls, w, label="New", color=COLORS["new"], alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(classes)
    ax5.set_title("Total Detections by Class")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=8)
    ax5.grid(axis="y", alpha=0.3)

    # ── 6. Track stability ──────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 0])
    stability_metrics = ["Track ID\nSwitches", "Unique\nPerson Tracks", "Unique\nCar Tracks"]
    old_s = [old["track_id_switches"], old["unique_tracks_person"], old["unique_tracks_car"]]
    new_s = [new["track_id_switches"], new["unique_tracks_person"], new["unique_tracks_car"]]
    x = np.arange(len(stability_metrics))
    ax6.bar(x - w/2, old_s, w, label="Old", color=COLORS["old"], alpha=0.85)
    ax6.bar(x + w/2, new_s, w, label="New", color=COLORS["new"], alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels(stability_metrics, fontsize=8)
    ax6.set_title("Tracking Stability\n(fewer switches = better)")
    ax6.legend(fontsize=8)
    ax6.grid(axis="y", alpha=0.3)

    # ── 7. Improvement summary text ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis("off")

    def pct_change(old_v, new_v, lower_is_better=False):
        if old_v == 0:
            return "N/A"
        change = (new_v - old_v) / old_v * 100
        if lower_is_better:
            change = -change
        sign = "+" if change >= 0 else ""
        return f"{sign}{change:.1f}%"

    summary_lines = [
        ("Metric", "Old", "New", "Change"),
        ("─" * 18, "─" * 10, "─" * 10, "─" * 12),
        ("Total detections", str(old["total_detections"]), str(new["total_detections"]),
         pct_change(old["total_detections"], new["total_detections"])),
        ("Mean confidence", f"{old['mean_conf']:.3f}", f"{new['mean_conf']:.3f}",
         pct_change(old["mean_conf"], new["mean_conf"])),
        ("Person detections", str(old["cls_counts"].get(0, 0)), str(new["cls_counts"].get(0, 0)),
         pct_change(old["cls_counts"].get(0, 0), new["cls_counts"].get(0, 0))),
        ("Car detections", str(old["cls_counts"].get(1, 0)), str(new["cls_counts"].get(1, 0)),
         pct_change(old["cls_counts"].get(1, 0), new["cls_counts"].get(1, 0))),
        ("Track ID switches", str(old["track_id_switches"]), str(new["track_id_switches"]),
         pct_change(old["track_id_switches"], new["track_id_switches"], lower_is_better=True)),
        ("Unique person tracks", str(old["unique_tracks_person"]), str(new["unique_tracks_person"]),
         pct_change(old["unique_tracks_person"], new["unique_tracks_person"], lower_is_better=True)),
        ("Unique car tracks", str(old["unique_tracks_car"]), str(new["unique_tracks_car"]),
         pct_change(old["unique_tracks_car"], new["unique_tracks_car"], lower_is_better=True)),
    ]

    col_x = [0.02, 0.38, 0.58, 0.78]
    row_y = 0.95
    for i, row in enumerate(summary_lines):
        for j, (val, cx) in enumerate(zip(row, col_x)):
            weight = "bold" if i <= 1 else "normal"
            color = "black"
            if i > 1 and j == 3:
                color = "#2ecc71" if (val.startswith("+") or (val != "N/A" and float(val.rstrip("%")) > 0)) else "#e74c3c"
            ax7.text(cx, row_y - i * 0.11, val, transform=ax7.transAxes,
                    fontsize=9, verticalalignment="top", fontweight=weight, color=color,
                    fontfamily="monospace")

    ax7.set_title("Summary Table", fontweight="bold")

    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved metrics chart → {out_path}")


def main():
    print("=" * 60)
    print("Model Comparison: best.pt vs best_v3.pt")
    print("=" * 60)

    old_stats = run_tracking(OLD_WEIGHTS, "Old (best.pt)")
    new_stats = run_tracking(NEW_WEIGHTS, "New (best_v3.pt)")

    print("\n── Running on old model for frame samples...")
    save_comparison_frames(OLD_WEIGHTS, NEW_WEIGHTS, OUT_DIR / "frame_comparison.png")

    print("\n── Generating metrics chart...")
    plot_metrics(old_stats, new_stats, OUT_DIR / "metrics_comparison.png")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Old':>8} {'New':>8}")
    print("-" * 45)
    print(f"{'Total detections':<25} {old_stats['total_detections']:>8} {new_stats['total_detections']:>8}")
    print(f"{'Mean confidence':<25} {old_stats['mean_conf']:>8.3f} {new_stats['mean_conf']:>8.3f}")
    print(f"{'Person detections':<25} {old_stats['cls_counts'].get(0,0):>8} {new_stats['cls_counts'].get(0,0):>8}")
    print(f"{'Car detections':<25} {old_stats['cls_counts'].get(1,0):>8} {new_stats['cls_counts'].get(1,0):>8}")
    print(f"{'Track ID switches':<25} {old_stats['track_id_switches']:>8} {new_stats['track_id_switches']:>8}")
    print(f"{'Unique person tracks':<25} {old_stats['unique_tracks_person']:>8} {new_stats['unique_tracks_person']:>8}")
    print(f"{'Unique car tracks':<25} {old_stats['unique_tracks_car']:>8} {new_stats['unique_tracks_car']:>8}")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
