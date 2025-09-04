#!/usr/bin/env python3
"""
Complete Object Mapping Pipeline Test
Tests the full video-to-map transformation system
"""

import sys
import os
from pathlib import Path
import logging
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from object_detection.mapping.object_mapper import ObjectMapper

def main():
    """Test complete object mapping pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description='Test complete object mapping pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, help='Path to YOLO model weights', 
                       default='models/weights/best.onnx')
    parser.add_argument('--ground-truth', type=str, help='Path to ground truth JSON file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default='outputs/mapping', help='Output directory')
    parser.add_argument('--no-display', action='store_true', help='Disable real-time display')
    parser.add_argument('--no-save-video', action='store_true', help='Disable video saving')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("   Make sure you have a YOLO model in models/weights/")
        return
    
    # Find ground truth file if not specified
    if not args.ground_truth:
        gt_dir = Path("outputs/ground_truth")
        gt_files = list(gt_dir.glob("*.json")) if gt_dir.exists() else []
        
        if not gt_files:
            print("❌ No ground truth files found!")
            print("   Run the point selection UI first: python src/scripts/run_ui.py")
            return
        
        args.ground_truth = str(gt_files[0])
        print(f"📍 Using ground truth file: {gt_files[0].name}")
    
    try:
        print("🚀 Initializing Object Mapper...")
        
        # Class names mapping (adjust based on your model)
        class_names = {0: 'person', 1: 'car'}
        
        # Initialize object mapper
        mapper = ObjectMapper(
            model_path=args.model,
            homography_file=args.ground_truth,
            class_names=class_names,
            confidence_threshold=args.confidence
        )
        
        print(f"🎥 Processing video: {Path(args.video).name}")
        print(f"🎯 Confidence threshold: {args.confidence}")
        print("   Press 'q' to quit during processing")
        
        # Process video
        csv_path = mapper.process_video_stream(
            video_source=args.video,
            output_dir=args.output,
            save_video=not args.no_save_video,
            show_display=not args.no_display,
            tracker="bytetrack.yaml"
        )
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Results saved to: {csv_path}")
        
        # Show summary statistics
        if mapper.tracking_data:
            total_detections = len(mapper.tracking_data)
            unique_objects = len(set(record["track_id"] for record in mapper.tracking_data if record["track_id"] is not None))
            frames_processed = max(record["frame"] for record in mapper.tracking_data)
            
            print(f"\n📈 Summary Statistics:")
            print(f"   Frames processed: {frames_processed}")
            print(f"   Total detections: {total_detections}")
            print(f"   Unique objects tracked: {unique_objects}")
            print(f"   Confidence threshold: {args.confidence}")
            
            # Show class distribution
            class_counts = {}
            for record in mapper.tracking_data:
                class_name = record["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print(f"   Detection by class:")
            for class_name, count in class_counts.items():
                print(f"     {class_name}: {count}")
            
            # Get trajectories
            trajectories = mapper.get_object_trajectories()
            print(f"   Objects with complete trajectories: {len(trajectories)}")
            
            # Show sample world coordinates
            print(f"\n🌍 Sample World Coordinates:")
            for i, record in enumerate(mapper.tracking_data[:5]):  # Show first 5
                print(f"   Frame {record['frame']}: {record['class_name']} ID-{record['track_id']} → ({record['world_x']:.2f}, {record['world_y']:.2f})m")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Check results CSV: {csv_path}")
        print(f"   2. View saved video in: {args.output}")
        print(f"   3. Analyze object trajectories for mapping insights")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        logging.error(f"Pipeline error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
