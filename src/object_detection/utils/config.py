"""Configuration management."""

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    weights: str = "models/weights/yolov8n.pt"
    confidence_threshold: float = 0.5
    device: str = "auto"
    save_predictions: bool = True
    output_dir: str = "outputs/predictions"


def load_config(config_path: str) -> DetectionConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    model_config = config_data.get('model', {})
    output_config = config_data.get('output', {})
    
    return DetectionConfig(
        weights=model_config.get('weights', 'models/weights/yolov8n.pt'),
        confidence_threshold=model_config.get('confidence_threshold', 0.5),
        device=model_config.get('device', 'auto'),
        save_predictions=output_config.get('save_predictions', True),
        output_dir=output_config.get('output_dir', 'outputs/predictions')
    )
