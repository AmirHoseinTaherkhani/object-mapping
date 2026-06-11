"""
Cloud training script for Vast.ai / RunPod.
Assumes project is uploaded to /workspace/ on the instance.
CUDA with AMP enabled. Saves last.pt/best.pt after every epoch.
"""

import os
import shutil
import torch
import mlflow
from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path("/workspace")
    data_config = project_root / "configs" / "training" / "data_v2_cloud.yaml"
    base_model = project_root / "yolov8s.pt"
    run_name = "yolov8s_v3"
    weights_dir = project_root / "runs" / "train" / run_name / "weights"

    assert torch.cuda.is_available(), "No CUDA GPU found. Check instance setup."
    device = "0"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {vram_gb:.1f} GB")

    if vram_gb >= 40:
        batch = 64
    elif vram_gb >= 20:
        batch = 32
    else:
        batch = 16

    mlflow.set_tracking_uri(f"file://{project_root / 'mlruns'}")
    mlflow.set_experiment("YOLOv8s_V3_Training")

    params = dict(
        epochs=150,
        batch=batch,
        imgsz=640,
        device=device,
        workers=8,
        amp=True,
        patience=40,
        cos_lr=True,
        label_smoothing=0.1,
        mixup=0.0,
        copy_paste=0.0,
        mosaic=1.0,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        close_mosaic=20,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        iou=0.7,
        conf=0.001,
        save=True,
        save_period=5,   # checkpoint every 5 epochs (epoch5.pt, epoch10.pt …)
        plots=True,
        verbose=True,
    )

    print(f"Batch: {batch} | AMP: True | Workers: 8 | save_period: 5")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("base_model", "yolov8s.pt")
        mlflow.log_param("data_config", "data_v2_cloud.yaml")
        mlflow.log_param("class_mapping", "0=person, 1=car")

        model = YOLO(str(base_model))

        print("Starting training...")
        results = model.train(
            data=str(data_config),
            project=str(project_root / "runs" / "train"),
            name=run_name,
            exist_ok=True,
            **params,
        )

        best_src = weights_dir / "best.pt"
        best_dst = project_root / "models" / "weights" / "best_v3.pt"
        best_dst.parent.mkdir(parents=True, exist_ok=True)
        if best_src.exists():
            shutil.copy(str(best_src), str(best_dst))
            mlflow.log_artifact(str(best_dst), "model")
            print(f"Best weights saved to: {best_dst}")

        if hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                try:
                    mlflow.log_metric(k.replace("/", "_"), float(v))
                except Exception:
                    pass

        print("Training complete.")


if __name__ == "__main__":
    main()
