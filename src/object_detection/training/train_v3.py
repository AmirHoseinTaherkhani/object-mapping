"""
Training script v3: YOLOv8s on all available data with corrected class mapping.

Key changes vs previous runs:
- Base model: yolov8s.pt (small, not nano) — better accuracy
- Fixed class order: person=0, car=1 (matches actual label files)
- All available datasets included (~38K person images, ~22K car images)
- Cosine LR decay, mixup + copy_paste augmentation
- Larger patience to avoid premature stopping
"""

import os
import shutil
import torch
import mlflow
from pathlib import Path
from ultralytics import YOLO


def main():
    # MPS fallback allows ops without MPS kernels to run on CPU transparently.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    project_root = Path(__file__).parent.parent.parent.parent

    device = "mps"
    print(f"Device: {device}")

    data_config = project_root / "configs" / "training" / "data_v2.yaml"
    base_model = project_root / "yolov8s.pt"
    run_name = "yolov8s_v3"

    mlflow.set_tracking_uri(f"file://{project_root / 'mlruns'}")
    mlflow.set_experiment("YOLOv8s_V3_Training")

    params = dict(
        epochs=150,
        batch=16,
        imgsz=640,
        device=device,
        workers=4,           # PyTorch 2.8 handles MPS multiprocessing correctly
        amp=False,           # MPS AMP is unreliable
        patience=40,         # give training room to converge
        cos_lr=True,         # cosine LR schedule — smoother convergence
        label_smoothing=0.1, # prevents overconfidence on noisy labels
        mixup=0.0,
        copy_paste=0.0,
        mosaic=1.0,
        degrees=5.0,         # slight rotation augmentation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        close_mosaic=20,     # disable mosaic last 20 epochs for stable convergence
        optimizer="AdamW",
        lr0=0.001,           # lower LR for fine-tuning from pretrained s weights
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        iou=0.7,
        conf=0.001,          # low threshold during val to capture all mAP
        save=True,
        plots=True,
        verbose=True,
    )

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("base_model", "yolov8s.pt")
        mlflow.log_param("data_config", "data_v2.yaml")
        mlflow.log_param("class_mapping", "0=person, 1=car")

        model = YOLO(str(base_model))

        print("Starting training...")
        results = model.train(
            data=str(data_config),
            project=str(project_root / "runs" / "train"),
            name=run_name,
            exist_ok=False,
            **params,
        )

        # Copy best weights to models/weights for inference
        best_src = project_root / "runs" / "train" / run_name / "weights" / "best.pt"
        best_dst = project_root / "models" / "weights" / "best_v3.pt"
        if best_src.exists():
            shutil.copy(str(best_src), str(best_dst))
            mlflow.log_artifact(str(best_dst), "model")
            print(f"Best weights saved to: {best_dst}")

        # Log final metrics
        if hasattr(results, "results_dict"):
            for k, v in results.results_dict.items():
                try:
                    mlflow.log_metric(k.replace("/", "_"), float(v))
                except Exception:
                    pass

        print("Training complete.")


if __name__ == "__main__":
    main()
