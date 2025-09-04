#!/usr/bin/env python3
"""Professional prediction script with configuration support."""

import argparse
from pathlib import Path
import sys
sys.path.append('src')

from object_detection.inference.predictor import ObjectDetector


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Run object detection")
    parser.add_argument("--input", required=True, help="Input image/video path")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--model", help="Model path (overrides config)")
    
    args = parser.parse_args()
    
    print(f"Processing: {args.input}")
    
    # Initialize detector with config or model path
    if args.config:
        print(f"Using config: {args.config}")
        detector = ObjectDetector(config_path=args.config)
    elif args.model:
        print(f"Using model: {args.model}")
        detector = ObjectDetector(model_path=args.model)
    else:
        print("Using default configuration")
        detector = ObjectDetector()
    
    print(f"Confidence threshold: {detector.config.confidence_threshold}")
    
    results = detector.predict(args.input)
    
    # Display results per image
    for image_result in results:
        image_path = image_result['image_path']
        detections = image_result['detections']
        
        print(f"\nImage: {Path(image_path).name}")
        print(f"Detected {len(detections)} objects:")
        
        for detection in detections:
            print(f"- {detection['class_name']}: {detection['confidence']:.2f}")


if __name__ == "__main__":
    main()
