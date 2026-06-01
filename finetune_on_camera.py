"""
Fine-tune best_v3.pt on labeled intersection-camera frames.
Expects Roboflow YOLOv8 export unzipped into labeling_data/dataset/:
    labeling_data/dataset/
        data.yaml
        train/images/  train/labels/
        valid/images/  valid/labels/

Run: python finetune_on_camera.py
"""

import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT     = Path(__file__).parent
DATASET_DIR = PROJECT / "labeling_data" / "dataset"
DATA_YAML   = DATASET_DIR / "data.yaml"
BASE_MODEL  = PROJECT / "models/weights/best_v3.pt"
OUT_WEIGHTS = PROJECT / "models/weights/best_v3_camera.pt"
RUN_NAME    = "yolov8s_v3_camera"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return str(torch.cuda.device_count() - 1)
    return "cpu"


def main():
    assert DATA_YAML.exists(), (
        f"Dataset not found at {DATA_YAML}\n"
        "Export from Roboflow as 'YOLOv8 PyTorch' and unzip into labeling_data/dataset/"
    )

    device = pick_device()
    print(f"Device: {device}")
    print(f"Base model: {BASE_MODEL.name}")
    print(f"Dataset: {DATA_YAML}")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    model = YOLO(str(BASE_MODEL))

    params = dict(
        data=str(DATA_YAML),
        project=str(PROJECT / "runs/finetune"),
        name=RUN_NAME,
        exist_ok=True,
        epochs=40,
        batch=4,           # imgsz=1280 needs smaller batch on 16GB MPS
        imgsz=1280,        # preserves detail of small/distant cars
        device=device,
        workers=2,
        amp=False,         # MPS AMP unreliable
        patience=15,
        cos_lr=True,
        lr0=0.0005,        # low LR — fine-tuning, not training from scratch
        lrf=0.01,
        warmup_epochs=3,
        freeze=10,         # freeze first 10 backbone layers, train head only
        mosaic=1.0,        # always-on mosaic: forces model to see small objects
        degrees=5.0,
        fliplr=0.5,
        scale=0.6,         # stronger scale jitter → simulates distant/small cars
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
    )

    print(f"\nStarting fine-tune: {params['epochs']} epochs, batch={params['batch']}, freeze={params['freeze']}")
    results = model.train(**params)

    best_src = PROJECT / "runs/finetune" / RUN_NAME / "weights/best.pt"
    if best_src.exists():
        shutil.copy(str(best_src), str(OUT_WEIGHTS))
        print(f"\nBest weights → {OUT_WEIGHTS}")
    else:
        print("Warning: best.pt not found in run output.")

    print("Fine-tune complete.")


if __name__ == "__main__":
    main()
