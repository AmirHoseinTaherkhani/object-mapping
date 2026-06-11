import mlflow
import mlflow.pytorch
import os
import sys
import shutil
import torch
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import YOLO
from ultralytics import YOLO

def check_device():
    if torch.backends.mps.is_available():
        device = "mps"
        print("MPS acceleration available - using GPU!")
    else:
        device = "cpu"
        print("Using CPU")
    return device

def main():
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_name = "YOLOv8_Clean_Training_V02"
    
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    
    mlflow.set_experiment(experiment_name)
    
    device = check_device()
    
    with mlflow.start_run():
        # Define paths
        model_path = project_root / "models" / "weights" / "best.pt"
        data_config_path = project_root / "configs" / "training" / "data.yaml"
        output_model_path = project_root / "models" / "weights" / "bestV02.pt"
        
        # Load model
        model = YOLO(str(model_path))
        print(f"Loaded model: {model_path}")
        
        # Log parameters
        mlflow.log_param("model_path", str(model_path))
        mlflow.log_param("epochs", 25)  # Reduced epochs for testing
        mlflow.log_param("device", device)
        
        # Train with minimal, guaranteed-valid parameters only
        print("Starting training...")
        results = model.train(
            data=str(data_config_path),
            epochs=25,
            device=device,
            project=str(project_root / "runs" / "train"),
            name="clean_training_v02",
            save=True,
            verbose=True
        )
        
        print("Training completed")
        
        # Save model
        training_output = project_root / "runs" / "train" / "clean_training_v02" / "weights" / "best.pt"
        
        if training_output.exists():
            shutil.copy(str(training_output), str(output_model_path))
            print(f"Model saved as: {output_model_path}")
            mlflow.log_artifact(str(output_model_path), "model")
        
        # Simple validation
        val_results = model.val(data=str(data_config_path))
        
        if hasattr(val_results, 'box'):
            mlflow.log_metric("val_mAP50", val_results.box.map50)
            mlflow.log_metric("val_mAP50-95", val_results.box.map)
            print(f"Validation mAP@0.5: {val_results.box.map50:.4f}")
        
        print("Training and validation completed successfully!")

if __name__ == "__main__":
    main()
