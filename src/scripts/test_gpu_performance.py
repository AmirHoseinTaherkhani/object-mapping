#!/usr/bin/env python3
"""Test GPU performance vs CPU"""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from object_detection.inference.predictor import ObjectDetector

def test_device_performance(model_path: str, test_frames: int = 10):
    """Test inference speed on different devices"""
    
    # Create dummy frame
    import numpy as np
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    for device in devices:
        print(f"\nTesting {device.upper()}:")
        
        # Initialize detector
        detector = ObjectDetector(model_path=model_path)
        detector.config.device = device
        
        # Warmup
        for _ in range(3):
            detector.predict(dummy_frame)
        
        # Time inference
        start_time = time.time()
        for i in range(test_frames):
            results = detector.predict(dummy_frame)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / test_frames
        fps = 1.0 / avg_time
        
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Estimated FPS: {fps:.1f}")

if __name__ == "__main__":
    model_path = "models/weights/best.onnx"
    test_device_performance(model_path)
