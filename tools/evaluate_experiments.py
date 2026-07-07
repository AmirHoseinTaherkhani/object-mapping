"""
Run all 8 YOLO11 + BoT-SORT experiments and produce a comparison table.

Experiments:
  1-4  Pretrained YOLO11 s/m/l/x (COCO weights, no fine-tuning)
       → test: benchmark video counting only (class schema differs from COCO)
  5-8  Fine-tuned YOLO11 s/m/l/x (trained on our 2-class merged dataset)
       → test: mAP on held-out test set + benchmark video counting

Usage:
    # Run all 8 experiments (pretrained + finetuned)
    python tools/evaluate_experiments.py

    # Run only pretrained experiments (no fine-tuned weights needed)
    python tools/evaluate_experiments.py --mode pretrained

    # Run only fine-tuned experiments (weights must exist in experiments/ dir)
    python tools/evaluate_experiments.py --mode finetuned

    # Run a single experiment
    python tools/evaluate_experiments.py --model yolo11s --pretrained

Output files:
    experiments/yolo11_experiments/results/
      pretrained_yolo11s.json  ...
      finetuned_yolo11s.json   ...
    experiments/yolo11_experiments/summary.csv
    experiments/yolo11_experiments/summary.md
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).parent.parent
EXP_BASE = ROOT / "experiments" / "yolo11_experiments"
RESULTS  = EXP_BASE / "results"
WEIGHTS  = ROOT / "models" / "weights"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_benchmark_videos() -> list[tuple[str, str, str]]:
    """Return (path, label, roi) for all available benchmark videos."""
    videos = []
    custom = ROOT / "Demo" / "ANMR0006.mp4"
    if custom.exists():
        videos.append((str(custom), "ANMR0006", ""))          # use real ROI

    vids_dir = ROOT / "Demo" / "videos"   # lowercase — matches actual dir name
    if vids_dir.is_dir():
        for p in sorted(vids_dir.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append((str(p), p.stem, "none"))

    stmarc = ROOT / "Demo" / "benchmark_stmarc.avi"
    if stmarc.exists():
        videos.append((str(stmarc), "stmarc", "none"))

    return videos


def run_counting(weights: str, source: str, roi: str, pretrained: bool,
                 max_frames: int = 0, out_dir: Path = None) -> dict:
    """Run run_yolo11_experiment.py and parse the output."""
    tag   = Path(weights).stem
    src   = Path(source).stem
    mode  = "pretrained" if pretrained else "finetuned"

    out_path = out_dir / f"{src}.mp4" if out_dir else \
               EXP_BASE / "videos" / f"{mode}_{tag}" / f"{src}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_yolo11_experiment.py"),
        "--weights", weights,
        "--source",  source,
        "--no-display",
        "--output",  str(out_path),
    ]
    if pretrained:
        cmd.append("--pretrained")
    if roi:
        cmd += ["--roi", roi]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]

    t0     = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    stdout = result.stdout + result.stderr
    cars   = int(m.group(1)) if (m := re.search(r"Cars\s*:\s*(\d+)",   stdout)) else -1
    people = int(m.group(1)) if (m := re.search(r"People\s*:\s*(\d+)", stdout)) else -1
    frames = int(m.group(1)) if (m := re.search(r"Frames\s*:\s*(\d+)", stdout)) else -1

    return {
        "video":   src,
        "cars":    cars,
        "people":  people,
        "frames":  frames,
        "wall_s":  round(elapsed, 1),
        "fps":     round(frames / max(elapsed, 0.1), 1),
        "ok":      result.returncode == 0,
        "stderr":  result.stderr[-400:] if result.returncode != 0 else "",
    }


def run_val(weights: str, data_yaml: str) -> dict:
    """Run ultralytics val on the test split and return metrics."""
    code = f"""
import json
from ultralytics import YOLO
model  = YOLO('{weights}')
m      = model.val(data='{data_yaml}', split='test', device='cpu',
                   verbose=False, plots=False)
result = dict(
    map50    = round(float(m.box.map50), 4),
    map50_95 = round(float(m.box.map),   4),
    precision= round(float(m.box.mp),    4),
    recall   = round(float(m.box.mr),    4),
)
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT)
    )
    if result.returncode != 0:
        return {"error": result.stderr[-200:]}
    try:
        last_line = [l for l in result.stdout.splitlines() if l.strip().startswith("{")]
        return json.loads(last_line[-1]) if last_line else {"error": "no json output"}
    except Exception as e:
        return {"error": str(e)}


def run_experiment(model_name: str, pretrained: bool,
                   max_frames: int = 1800, data_yaml: str = None) -> dict:
    """Run one of the 8 experiments."""
    mode = "pretrained" if pretrained else "finetuned"
    print(f"\n{'='*60}")
    print(f"  Experiment: {model_name} — {mode}")
    print(f"{'='*60}")

    # Locate weights
    if pretrained:
        w = WEIGHTS / f"{model_name}.pt"
        if not w.exists():
            # trigger auto-download
            from ultralytics import YOLO
            YOLO(f"{model_name}.pt")
            w = WEIGHTS / f"{model_name}.pt"
            if not w.exists():
                # downloaded to cwd by ultralytics
                from pathlib import Path as P
                cwd_pt = P(f"{model_name}.pt")
                if cwd_pt.exists():
                    import shutil
                    shutil.move(str(cwd_pt), str(w))
    else:
        w = EXP_BASE / "finetuned" / model_name / "train" / "weights" / "best.pt"
        if not w.exists():
            print(f"  [SKIP] Fine-tuned weights not found: {w}")
            print("         Submit hpc/finetune_{model_name}.slurm first.")
            return {"model": model_name, "mode": mode, "status": "weights_missing"}

    print(f"  Weights: {w}")

    out = {
        "model":      model_name,
        "mode":       mode,
        "weights":    str(w),
        "status":     "ok",
        "val_metrics": None,
        "videos":     [],
    }

    # ── val mAP (fine-tuned only — class schema matches) ──────────────────
    if not pretrained and data_yaml:
        print("  Running test-set evaluation (mAP)…")
        out["val_metrics"] = run_val(str(w), data_yaml)
        print(f"  mAP50={out['val_metrics'].get('map50','N/A')}  "
              f"mAP50-95={out['val_metrics'].get('map50_95','N/A')}")
    else:
        print("  Skipping mAP (pretrained model has different class schema from our test set)")

    # ── benchmark video counting ───────────────────────────────────────────
    out_dir = EXP_BASE / "videos" / f"{mode}_{model_name}"
    videos  = get_benchmark_videos()
    if not videos:
        print("  No benchmark videos found in Demo/ — skipping counting tests.")
    for source, label, roi in videos:
        print(f"  Processing {label}…", end=" ", flush=True)
        vr = run_counting(str(w), source, roi, pretrained, max_frames=max_frames,
                          out_dir=out_dir)
        vr["label"] = label
        out["videos"].append(vr)
        if vr["ok"]:
            print(f"cars={vr['cars']} people={vr['people']} fps={vr['fps']}")
        else:
            print(f"FAILED — {vr['stderr'][:80]}")

    # Save per-experiment JSON
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"{mode}_{model_name}.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {json_path}")
    return out


def write_summary(all_results: list[dict]):
    """Write summary.csv and summary.md from all experiment results."""
    csv_lines = [
        "mode,model,map50,map50-95,precision,recall,"
        + ",".join(f"{v['label']}_cars,{v['label']}_people"
                   for r in all_results if r.get("videos")
                   for v in r["videos"][:1])
    ]
    # Actually build a proper summary
    rows = []
    for r in all_results:
        vm = r.get("val_metrics") or {}
        row = {
            "mode":      r.get("mode", "?"),
            "model":     r.get("model", "?"),
            "map50":     vm.get("map50",    "N/A"),
            "map50_95":  vm.get("map50_95", "N/A"),
            "precision": vm.get("precision","N/A"),
            "recall":    vm.get("recall",   "N/A"),
        }
        for vr in r.get("videos", []):
            row[f"{vr['label']}_cars"]   = vr.get("cars",   -1)
            row[f"{vr['label']}_people"] = vr.get("people", -1)
            row[f"{vr['label']}_fps"]    = vr.get("fps",    -1)
        rows.append(row)

    # CSV
    if rows:
        keys = list(rows[0].keys())
        csv_lines = [",".join(keys)]
        for row in rows:
            csv_lines.append(",".join(str(row.get(k, "")) for k in keys))
        csv_path = EXP_BASE / "summary.csv"
        csv_path.write_text("\n".join(csv_lines) + "\n")
        print(f"\nSaved: {csv_path}")

    # Markdown
    md_lines = ["# YOLO11 + BoT-SORT Experiment Results\n"]
    md_lines.append("## Model accuracy on test set (fine-tuned only)\n")
    md_lines.append("| Model | mAP50 | mAP50-95 | Precision | Recall |")
    md_lines.append("|-------|-------|----------|-----------|--------|")
    for r in all_results:
        if r.get("mode") != "finetuned":
            continue
        vm = r.get("val_metrics") or {}
        md_lines.append(
            f"| {r['model']} | {vm.get('map50','—')} | {vm.get('map50_95','—')} "
            f"| {vm.get('precision','—')} | {vm.get('recall','—')} |"
        )

    md_lines.append("\n## Benchmark video counts\n")
    all_vid_labels = []
    for r in all_results:
        for vr in r.get("videos", []):
            if vr["label"] not in all_vid_labels:
                all_vid_labels.append(vr["label"])

    if all_vid_labels:
        header = "| Mode | Model | " + " | ".join(
            f"{v} cars | {v} people" for v in all_vid_labels
        ) + " |"
        sep = "|------|-------|" + "|".join("-------|-------" for _ in all_vid_labels) + "|"
        md_lines.append(header)
        md_lines.append(sep)
        for r in all_results:
            vid_map = {vr["label"]: vr for vr in r.get("videos", [])}
            cells = []
            for lbl in all_vid_labels:
                vr = vid_map.get(lbl, {})
                cells += [str(vr.get("cars", "—")), str(vr.get("people", "—"))]
            md_lines.append(f"| {r.get('mode','?')} | {r.get('model','?')} | " +
                            " | ".join(cells) + " |")

    md_path = EXP_BASE / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"Saved: {md_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode",  choices=["all", "pretrained", "finetuned"], default="all")
    p.add_argument("--model", choices=["yolo11s","yolo11m","yolo11l","yolo11x","all"],
                   default="all")
    p.add_argument("--max-frames", type=int, default=1800,
                   help="Frames per benchmark video (default 1800 ≈ 30s at 60fps)")
    args = p.parse_args()

    data_yaml = str(ROOT / "labeling_data" / "trainingData" / "data.yaml")
    model_sizes = ["yolo11s", "yolo11m", "yolo11l", "yolo11x"] \
                  if args.model == "all" else [args.model]

    all_results = []

    # ── pretrained experiments (1-4) ──────────────────────────────────────
    if args.mode in ("all", "pretrained"):
        print("\n" + "█"*60)
        print(" PRETRAINED EXPERIMENTS (1–4) — no fine-tuning")
        print("█"*60)
        for m in model_sizes:
            r = run_experiment(m, pretrained=True, max_frames=args.max_frames)
            all_results.append(r)

    # ── fine-tuned experiments (5-8) ──────────────────────────────────────
    if args.mode in ("all", "finetuned"):
        print("\n" + "█"*60)
        print(" FINE-TUNED EXPERIMENTS (5–8) — weights from HPC")
        print("█"*60)
        for m in model_sizes:
            r = run_experiment(m, pretrained=False, max_frames=args.max_frames,
                               data_yaml=data_yaml)
            all_results.append(r)

    write_summary(all_results)

    print("\n" + "="*60)
    print("  All experiments complete.")
    print(f"  Results: {RESULTS}")
    print(f"  Summary: {EXP_BASE / 'summary.md'}")
    print("="*60)


if __name__ == "__main__":
    main()
