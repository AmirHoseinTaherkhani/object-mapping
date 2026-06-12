"""
Export best_v3_merged.pt → CoreML (.mlpackage) for Apple Silicon.

Run this once on a Mac. The exported package uses the Apple Neural Engine
automatically when loaded by ultralytics — no code changes needed in
run_coreml.py.

Usage:
    python export_coreml.py
"""

from pathlib import Path
from ultralytics import YOLO

WEIGHTS_IN  = Path(__file__).parent.parent / "models/weights/best_v3_merged.pt"
WEIGHTS_OUT = Path(__file__).parent.parent / "models/weights"  # ultralytics writes here

def main():
    if not WEIGHTS_IN.exists():
        raise FileNotFoundError(f"Source weights not found: {WEIGHTS_IN}")

    print(f"Exporting {WEIGHTS_IN.name} → CoreML …")
    model = YOLO(str(WEIGHTS_IN))

    # nms=True: bakes NMS into the model graph so the CoreML runtime handles it,
    # avoiding a Python-side NMS step after inference.
    out = model.export(format="coreml", imgsz=640, nms=True)
    print(f"\nExported → {out}")
    print("Copy the .mlpackage folder next to run_coreml.py or into models/weights/")
    print("Then run:  python run_coreml.py")

if __name__ == "__main__":
    main()
