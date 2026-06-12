"""
Export best_v3_merged.pt → TensorRT engine (.engine) for NVIDIA GPUs.

IMPORTANT: TensorRT engines are tied to the GPU architecture they were
built on. Run this script on the same machine (or same GPU model) that
will run the demo. If the demo GPU is unknown, use export_tensorrt.slurm
to build on the Premise A100 and then test on the target machine.

Usage:
    python export_tensorrt.py              # FP16 (recommended, fastest)
    python export_tensorrt.py --fp32       # FP32 (use if FP16 causes accuracy loss)
    python export_tensorrt.py --device 1   # specific GPU index
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

WEIGHTS_IN = Path(__file__).parent.parent / "models/weights/best_v3_merged.pt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32",   action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    args = parser.parse_args()

    if not WEIGHTS_IN.exists():
        raise FileNotFoundError(f"Source weights not found: {WEIGHTS_IN}")

    precision = "FP32" if args.fp32 else "FP16"
    print(f"Exporting {WEIGHTS_IN.name} → TensorRT {precision} on device {args.device} …")
    print("This takes 2–5 minutes on first run (TRT calibration).\n")

    model = YOLO(str(WEIGHTS_IN))
    out = model.export(
        format="engine",
        device=args.device,
        half=not args.fp32,   # FP16 gives ~2× speedup over FP32 on modern GPUs
        imgsz=640,
    )
    print(f"\nExported → {out}")
    print("Copy the .engine file to models/weights/ on the demo machine.")
    print("Then run:  python run_tensorrt.py")

if __name__ == "__main__":
    main()
