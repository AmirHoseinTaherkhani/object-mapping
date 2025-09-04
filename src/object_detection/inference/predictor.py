"""Professional object detection predictor with configuration support."""

from pathlib import Path
from typing import List, Union, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from ..utils.config import DetectionConfig, load_config


class ObjectDetector:
    """Professional object detection class with configuration support."""
    
    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        """Initialize detector with config file or direct model path."""
        if config_path:
            # Load from config file
            self.config = load_config(config_path)
        else:
            # Use default config with optional model override
            self.config = DetectionConfig()
            if model_path:
                self.config.weights = model_path
        
        self.model = YOLO(self.config.weights)
        
    def predict(self, source: Union[str, np.ndarray], save_results: Optional[bool] = None):
        """Detect objects using configuration settings."""
        # Use config setting or override
        should_save = save_results if save_results is not None else self.config.save_predictions
        
        results = self.model.predict(
            source=source,
            conf=self.config.confidence_threshold,
            device=self.config.device,
            save=should_save,
            project=self.config.output_dir
        )
        
        # Process results (same as before)
        all_detections = []
        for i, result in enumerate(results):
            image_detections = {
                'image_index': i,
                'image_path': getattr(result, 'path', f'image_{i}'),
                'detections': []
            }
            
            if result.boxes is not None:
                for box in result.boxes:
                    detection = {
                        'class_name': result.names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xywh[0].tolist()
                    }
                    image_detections['detections'].append(detection)
            
            all_detections.append(image_detections)
                    
        return all_detections
