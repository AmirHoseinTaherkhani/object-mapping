#!/usr/bin/env python3
"""
Real-time Video-to-Map Tracking System
Side-by-side visualization of video tracking and 2D mapping
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from object_detection.visualization.realtime_display import RealtimeDisplay

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('outputs/realtime_mapping.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Real-time Video-to-Map Tracking System")
    
    # Required arguments
    parser.add_argument("--video", "-v", required=True, 
                       help="Path to input video file or camera index (0, 1, etc.)")
    parser.add_argument("--gt", required=True,
                       help="Path to ground truth JSON file for homography calculation")
    
    # Model arguments
    parser.add_argument("--model", "-m", default="models/weights/best.onnx",
                       help="Path to ONNX model file (default: models/weights/best.onnx)")
    parser.add_argument("--confidence", "-c", type=float, default=0.7,
                       help="Confidence threshold for detections (default: 0.7)")
    
    # Visualization arguments
    parser.add_argument("--config", default="configs/visualization/realtime.yaml",
                       help="Path to visualization config file")
    parser.add_argument("--no-fps", action="store_true",
                       help="Hide FPS counter")
    
    # Performance arguments
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device for inference (cpu, cuda, mps - default: cpu)")
    parser.add_argument("--verbose", "-vv", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Validate inputs
        if not Path(args.gt).exists():
            logger.error(f"Ground truth file not found: {args.gt}")
            return 1
        
        if not Path(args.model).exists():
            logger.error(f"Model file not found: {args.model}")
            return 1
        
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        # Convert video argument (handle camera index)
        video_source = args.video
        try:
            video_source = int(args.video)  # Try as camera index
            logger.info(f"Using camera index: {video_source}")
        except ValueError:
            if not Path(args.video).exists():
                logger.error(f"Video file not found: {args.video}")
                return 1
            logger.info(f"Using video file: {video_source}")
        
        # Initialize realtime display
        logger.info("Initializing realtime display...")
        realtime_display = RealtimeDisplay(config_path=args.config)
        
        # Print system info
        logger.info("="*60)
        logger.info("REAL-TIME VIDEO-TO-MAP TRACKING SYSTEM")
        logger.info("="*60)
        logger.info(f"Video source: {video_source}")
        logger.info(f"Model: {args.model}")
        logger.info(f"Homography: {args.gt}")
        logger.info(f"Confidence threshold: {args.confidence}")
        logger.info("="*60)
        logger.info("Controls:")
        logger.info("  'q' or ESC: Exit")
        logger.info("  'c': Clear trails")
        logger.info("="*60)
        
        # Start realtime processing
        success = realtime_display.process_video_realtime(
            model_path=args.model,
            homography_file=args.gt,
            confidence_threshold=args.confidence,
            video_source=video_source,
            device=args.device,
            show_fps=not args.no_fps
        )
        
        if success:
            logger.info("Processing completed successfully")
            return 0
        else:
            logger.error("Processing failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        try:
            realtime_display.close()
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())
