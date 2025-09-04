#!/usr/bin/env python3
"""
Test script for homography calculation
Tests the mapping functionality with saved ground truth data
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from object_detection.mapping.homography import HomographyCalculator

def main():
    """Test homography calculation with saved ground truth"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Look for ground truth files
    gt_dir = Path("outputs/ground_truth")
    gt_files = list(gt_dir.glob("*.json")) if gt_dir.exists() else []
    
    if not gt_files:
        print("❌ No ground truth files found in outputs/ground_truth/")
        print("   Run the point selection UI first to create ground truth data")
        return
    
    # Use the first ground truth file found
    gt_file = gt_files[0]
    print(f"🎯 Testing homography with: {gt_file.name}")
    
    # Initialize calculator
    calculator = HomographyCalculator(str(gt_file))
    
    # Calculate homography
    if calculator.calculate_homography():
        print("✅ Homography matrix calculated successfully!")
        
        # Get transformation summary
        summary = calculator.get_transformation_summary()
        print(f"\n📊 Transformation Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Validate transformation
        validation = calculator.validate_homography(max_error_threshold=1.0)
        print(f"\n🔍 Validation Results:")
        print(f"   Valid: {validation['valid']}")
        print(f"   Mean Error: {validation['mean_error']:.3f}m")
        print(f"   Max Error: {validation['max_error']:.3f}m")
        
        # Test single point transformation
        print(f"\n🧪 Testing point transformation:")
        test_pixel = (400, 300)  # Example pixel coordinate
        world_coord = calculator.transform_point(test_pixel[0], test_pixel[1])
        print(f"   Pixel {test_pixel} → World ({world_coord[0]:.2f}, {world_coord[1]:.2f})")
        
        # Create visualization
        print(f"\n📈 Creating accuracy visualization...")
        viz_path = f"outputs/homography_validation_{gt_file.stem}.png"
        calculator.visualize_transformation_accuracy(viz_path)
        
    else:
        print("❌ Failed to calculate homography matrix")
        print("   Check that you have at least 4 valid point correspondences")

if __name__ == "__main__":
    main()
