import mlflow
import torch
import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Set environment variables to prevent multiprocessing issues
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Force single-threaded to avoid bus errors
    torch.set_num_threads(1)
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "YOLOv8_Stable_Training"
    
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        model_path = project_root / "models" / "weights" / "best.pt"
        data_config_path = project_root / "configs" / "training" / "data.yaml"
        
        model = YOLO(str(model_path))
        
        # Log parameters
        mlflow.log_param("epochs", 300)
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", 8)  # Smaller batch to prevent memory issues
        
        print("Starting stable training...")
        
        # Use more conservative settings to prevent crashes
        results = model.train(
            data=str(data_config_path),
            epochs=100,
            device=device,
            batch=8,        # Smaller batch size
            workers=0,      # Disable multiprocessing
            project=str(project_root / "runs" / "train"),
            name="stable_training_v02",
            save=True,
            verbose=True,
            patience=15,    # Early stopping
            amp=False       # Disable automatic mixed precision
        )
        
        print("Training completed successfully!")
        
        # Save the model
        training_output = project_root / "runs" / "train" / "stable_training_v02" / "weights" / "best.pt"
        output_path = project_root / "models" / "weights" / "bestV02.pt"
        
        if training_output.exists():
            import shutil
            shutil.copy(str(training_output), str(output_path))
            mlflow.log_artifact(str(output_path), "model")
            print(f"Model saved as: {output_path}")

if __name__ == "__main__":
    main()
