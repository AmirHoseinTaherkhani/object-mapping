"""
Fine-tune on labeled camera frames + merged Roboflow datasets.

Usage:
    python finetune_on_camera.py                  # merged dataset (default)
    python finetune_on_camera.py --source original # original 189-frame dataset only
    python finetune_on_camera.py --base best_v3_camera.pt  # warmstart from camera model
    python finetune_on_camera.py --hpc            # HPC / CUDA mode (4 workers, larger batch)
"""

import argparse
import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(__file__).parent


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def test_mps_amp():
    """Return True if MPS autocast works on this PyTorch build."""
    if not torch.backends.mps.is_available():
        return False
    try:
        x = torch.randn(4, 4, device="mps")
        with torch.autocast("mps"):
            _ = x @ x
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="merged",
                        choices=["merged", "original"],
                        help="Dataset pool to use")
    parser.add_argument("--base", default="last_epoch3.pt",
                        help="Base weights filename (in models/weights/)")
    parser.add_argument("--hpc", action="store_true",
                        help="HPC/CUDA mode: larger batch, more workers")
    args = parser.parse_args()

    if args.source == "merged":
        dataset_dir = PROJECT / "labeling_data" / "merged_dataset"
        run_name    = "yolov8s_merged_v2"
        out_weights = PROJECT / "models/weights/best_v3_merged.pt"
    else:
        dataset_dir = PROJECT / "labeling_data" / "dataset"
        run_name    = "yolov8s_v3_camera"
        out_weights = PROJECT / "models/weights/best_v3_camera.pt"

    data_yaml  = dataset_dir / "data.yaml"
    base_model = PROJECT / "models/weights" / args.base

    assert data_yaml.exists(), (
        f"Dataset not found at {data_yaml}\n"
        f"Run: python merge_datasets.py && python prepare_dataset.py --source {args.source}"
    )
    assert base_model.exists(), f"Base model not found: {base_model}"

    # Fix path in data.yaml to match current machine (handles Mac → HPC transfer)
    yaml_text = data_yaml.read_text()
    import re
    yaml_text = re.sub(r"^path:.*$", f"path: {dataset_dir}", yaml_text, flags=re.MULTILINE)
    data_yaml.write_text(yaml_text)

    device = pick_device()
    amp    = test_mps_amp() if device == "mps" else (device == "cuda")

    print(f"Device      : {device}")
    print(f"AMP         : {amp}  (PyTorch {torch.__version__})")
    print(f"Base model  : {base_model.name}")
    print(f"Dataset     : {data_yaml}")
    print(f"Output      : {out_weights.name}")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    model = YOLO(str(base_model))

    is_large = args.source == "merged"

    # HPC (CUDA): larger batch, proper multi-worker loading, no MPS workarounds
    batch   = 32 if args.hpc else (16 if is_large else 4)
    workers = 4  if args.hpc else 0

    params = dict(
        data=str(data_yaml),
        project=str(PROJECT / "runs/finetune"),
        name=run_name,
        exist_ok=True,
        epochs=50,
        batch=batch,
        imgsz=640 if is_large else 1280,
        device=device,
        workers=workers,
        amp=amp,                        # enabled on PyTorch 2.8+ MPS
        patience=15,
        cos_lr=True,
        lr0=0.001  if is_large else 0.0005,  # slightly higher LR with more data
        lrf=0.01,
        warmup_epochs=3,
        freeze=10,          # keep backbone frozen; train detection head
        conf=0.25,          # filter raw preds before NMS during val — prevents NMS timeout
        mosaic=1.0,
        degrees=5.0,
        fliplr=0.5,
        scale=0.5,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print(f"HPC mode    : {args.hpc}")
    print(f"\nStarting: {params['epochs']} epochs, "
          f"batch={params['batch']}, imgsz={params['imgsz']}, amp={amp}")
    model.train(**params)

    best_src = PROJECT / "runs/finetune" / run_name / "weights/best.pt"
    if best_src.exists():
        shutil.copy(str(best_src), str(out_weights))
        print(f"\nBest weights → {out_weights}")
    else:
        print("Warning: best.pt not found in run output.")

    print("Fine-tune complete.")


if __name__ == "__main__":
    main()
