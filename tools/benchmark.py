"""
Benchmark run_coreml.py across skip-n values and video sources.

Usage:
    python tools/benchmark.py
    python tools/benchmark.py --skip-n 1 2 --max-frames 900
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_pipeline(source: str, skip_n: int, max_frames: int, weights: str,
                 roi: str = "") -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "realtime_inference/run_coreml.py"),
        "--source", source,
        "--skip-n", str(skip_n),
        "--no-display",
        "--weights", weights,
    ]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    if roi:
        cmd += ["--roi", roi]

    src_stem = Path(source).stem
    out_path = ROOT / "realtime_inference" / "outputs" / src_stem / f"skip_n_{skip_n}.mp4"
    cmd += ["--output", str(out_path)]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    stdout = result.stdout + result.stderr
    cars    = int(m.group(1)) if (m := re.search(r"Cars\s*:\s*(\d+)",   stdout)) else -1
    people  = int(m.group(1)) if (m := re.search(r"People\s*:\s*(\d+)", stdout)) else -1
    frames  = int(m.group(1)) if (m := re.search(r"Frames\s*:\s*(\d+)", stdout)) else -1

    fps_matches = re.findall(r"fps=([0-9.]+)", stdout)
    if fps_matches:
        avg_fps = sum(float(f) for f in fps_matches) / len(fps_matches)
    elif frames > 0 and elapsed > 0:
        avg_fps = frames / elapsed   # fallback: wall-clock throughput
    else:
        avg_fps = 0.0

    return {
        "skip_n":  skip_n,
        "cars":    cars,
        "people":  people,
        "frames":  frames,
        "avg_fps": avg_fps,
        "wall_s":  elapsed,
        "ok":      result.returncode == 0,
        "stderr":  result.stderr[-800:] if result.returncode != 0 else "",
    }


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'━'*60}")
    print(f"  {title}")
    print(f"{'━'*60}")
    print(f"  {'skip-n':>6}  {'cars':>5}  {'people':>6}  {'frames':>7}  {'avg fps':>8}  {'wall (s)':>9}")
    print(f"  {'──────':>6}  {'─────':>5}  {'──────':>6}  {'───────':>7}  {'───────':>8}  {'────────':>9}")
    for r in rows:
        status = "" if r["ok"] else "  ← FAILED"
        print(f"  {r['skip_n']:>6}  {r['cars']:>5}  {r['people']:>6}  "
              f"{r['frames']:>7}  {r['avg_fps']:>8.1f}  {r['wall_s']:>9.1f}{status}")
        if r["stderr"]:
            print(f"         {r['stderr'][:80]}")
    print(f"{'━'*60}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-n",     type=int, nargs="+", default=[1, 2],
                   help="skip-n values to compare (default: 1 2)")
    p.add_argument("--max-frames", type=int, default=1800,
                   help="Frames to process per run (default: 1800 ≈ 30s at 60fps)")
    p.add_argument("--weights",    default=str(ROOT / "models/weights/best_v3_merged.mlpackage"),
                   help="CoreML weights path")
    args = p.parse_args()

    # Custom surveillance video always runs first with its proper ROI polygon.
    # Every file in Demo/Videos/ is then picked up automatically — no edits needed
    # when new videos are added to that folder.
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos_dir = ROOT / "Demo" / "Videos"

    videos = []
    custom = ROOT / "Demo" / "ANMR0006.mp4"
    if custom.exists():
        videos.append((str(custom), "Custom surveillance (ANMR0006)", ""))

    if videos_dir.is_dir():
        for p in sorted(videos_dir.iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                videos.append((str(p), p.name, "none"))

    stmarc = ROOT / "Demo" / "benchmark_stmarc.avi"
    if stmarc.exists():
        videos.append((str(stmarc), "UrbanTracker St-Marc", "none"))

    print(f"\nBenchmark — skip-n: {args.skip_n}  max-frames: {args.max_frames}")
    print(f"Weights: {Path(args.weights).name}")

    for source, label, roi in videos:
        if not Path(source).exists():
            print(f"\n[SKIP] {label} — file not found: {source}")
            continue
        rows = []
        for sn in args.skip_n:
            print(f"\n  Running {label}  skip-n={sn} …", flush=True)
            rows.append(run_pipeline(source, sn, args.max_frames, args.weights, roi))
        print_table(label, rows)

    print()


if __name__ == "__main__":
    main()
