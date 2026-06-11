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

# Check for MPS availability
def check_device():
    if torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Metal Performance Shaders) is available - using GPU acceleration!")
    elif torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available - using GPU acceleration!")
    else:
        device = "cpu"
        print("Using CPU - no GPU acceleration available")
    
    print(f"Selected device: {device}")
    return device

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set or create experiment
experiment_name = "YOLOv8_Resume_Training_V02_MPS"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)
print(f"Using MLflow experiment: {experiment_name}")

# Import YOLO after setting up paths
from ultralytics import YOLO

def main():
    # Check device availability
    device = check_device()
    
    # Start MLflow run
    with mlflow.start_run():
        # Define paths relative to project root
        model_path = project_root / "models" / "weights" / "best.pt"
        data_config_path = project_root / "configs" / "training" / "data.yaml"
        output_model_path = project_root / "models" / "weights" / "bestV02.pt"
        
        # Load the pre-trained model to resume training
        model = YOLO(str(model_path))
        print(f"Loaded existing model: {model_path}")
        
        # Log model parameters including device
        mlflow.log_param("model_type", "YOLOv8s")
        mlflow.log_param("resume_from", str(model_path))
        mlflow.log_param("epochs", 50)
        mlflow.log_param("data_config", str(data_config_path))
        mlflow.log_param("output_model", str(output_model_path))
        mlflow.log_param("device", device)
        mlflow.log_param("acceleration", "MPS" if device == "mps" else device.upper())
        
        print("MLflow run started and parameters logged")
        
        # Resume training with only valid YOLO parameters
        print(f"Starting resumed training on {device.upper()}...")
        results = model.train(
            data=str(data_config_path),
            epochs=50,
            resume=True,
            device=device,
            project=str(project_root / "runs" / "train"),
            name="resume_training_v02_mps",
            save=True,
            verbose=True,
            batch=16,
            workers=4,
            imgsz=640,
            patience=10,
            lr0=0.01,
            warmup_epochs=3
        )
        
        # Log training metrics
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
        
        print("Training completed successfully")
        
        # Save the new trained model as bestV02.pt
        training_output_path = project_root / "runs" / "train" / "resume_training_v02_mps" / "weights" / "best.pt"
        
        # Create output directory if it doesn't exist
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the best model from training results to our desired location
        if training_output_path.exists():
            shutil.copy(str(training_output_path), str(output_model_path))
            print(f"Model saved as: {output_model_path}")
        else:
            print(f"Warning: Training output not found at {training_output_path}")
        
        # Log the trained model as MLflow artifact
        if output_model_path.exists():
            mlflow.log_artifact(str(output_model_path), "model")
            print("Model artifacts logged to MLflow")
        
        # Validate the newly trained model
        print("Running validation on the trained model...")
        validation_results = model.val(data=str(data_config_path), device=device)
        
        # Log validation metrics
        if hasattr(validation_results, 'results_dict'):
            for key, value in validation_results.results_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{key}", value)
        
        # Log key performance metrics
        if hasattr(validation_results, 'box'):
            mlflow.log_metric("val_mAP50", validation_results.box.map50)
            mlflow.log_metric("val_mAP50-95", validation_results.box.map)
            print(f"Validation mAP@0.5: {validation_results.box.map50:.4f}")
            print(f"Validation mAP@0.5-0.95: {validation_results.box.map:.4f}")
        
        # Log training completion
        mlflow.log_param("training_status", "completed")
        mlflow.log_param("final_model_path", str(output_model_path))
        
        print("Validation completed and metrics logged")
        print("MLflow run completed successfully")

if __name__ == "__main__":
    main()
