"""
Homography Calculation Module
Computes perspective transformation matrix from ground truth points
"""

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

class HomographyCalculator:
    """
    Calculates and validates homography transformation from image to real-world coordinates
    """
    
    def __init__(self, ground_truth_file: Optional[str] = None):
        """
        Initialize homography calculator
        
        Args:
            ground_truth_file: Path to JSON file with point correspondences
        """
        self.homography_matrix = None
        self.image_points = []
        self.world_points = []
        self.reprojection_errors = []
        
        if ground_truth_file:
            self.load_ground_truth(ground_truth_file)
    
    def load_ground_truth(self, json_file: str) -> bool:
        """
        Load ground truth points from JSON file
        
        Args:
            json_file: Path to JSON file created by point selector
            
        Returns:
            bool: Success status
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract image points (pixel coordinates)
            self.image_points = np.array(data['image_points'], dtype=np.float32)
            
            # Extract world points (real-world coordinates)
            world_coords = data['world_points']
            self.world_points = np.array([
                [coord['x'], coord['y']] for coord in world_coords
            ], dtype=np.float32)
            
            logging.info(f"Loaded {len(self.image_points)} point correspondences")
            return True
            
        except Exception as e:
            logging.error(f"Error loading ground truth: {e}")
            return False
    
    def calculate_homography(self) -> bool:
        """
        Calculate homography matrix using OpenCV
        
        Returns:
            bool: Success status
        """
        if len(self.image_points) < 4:
            logging.error("Need at least 4 point correspondences for homography")
            return False
        
        try:
            # Calculate homography using RANSAC for robustness
            self.homography_matrix, mask = cv2.findHomography(
                self.image_points, 
                self.world_points,
                cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if self.homography_matrix is not None:
                self._calculate_reprojection_errors()
                logging.info("Homography matrix calculated successfully")
                return True
            else:
                logging.error("Failed to calculate homography matrix")
                return False
                
        except Exception as e:
            logging.error(f"Error calculating homography: {e}")
            return False
    
    def _calculate_reprojection_errors(self):
        """Calculate reprojection errors for validation"""
        if self.homography_matrix is None:
            return
        
        # Transform image points using calculated homography
        transformed_points = cv2.perspectiveTransform(
            self.image_points.reshape(-1, 1, 2),
            self.homography_matrix
        ).reshape(-1, 2)
        
        # Calculate Euclidean distances (errors)
        self.reprojection_errors = [
            np.linalg.norm(transformed - actual)
            for transformed, actual in zip(transformed_points, self.world_points)
        ]
    
    def transform_point(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Transform a single pixel coordinate to real-world coordinates
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple[float, float]: (x, y) in real-world coordinates
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated. Call calculate_homography() first.")
        
        # Create point array for transformation
        point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        
        # Transform using homography
        transformed = cv2.perspectiveTransform(point, self.homography_matrix)
        
        return float(transformed[0][0][0]), float(transformed[0][0][1])
    
    def transform_points_batch(self, pixel_coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Transform multiple pixel coordinates to real-world coordinates
        
        Args:
            pixel_coordinates: List of (x, y) pixel coordinates
            
        Returns:
            List[Tuple[float, float]]: List of (x, y) real-world coordinates
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calculated. Call calculate_homography() first.")
        
        if not pixel_coordinates:
            return []
        
        # Convert to numpy array
        points = np.array(pixel_coordinates, dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform using homography
        transformed = cv2.perspectiveTransform(points, self.homography_matrix)
        
        # Convert back to list of tuples
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]
    
    def validate_homography(self, max_error_threshold: float = 1.0) -> Dict:
        """
        Validate homography quality using reprojection errors
        
        Args:
            max_error_threshold: Maximum acceptable error in meters
            
        Returns:
            Dict: Validation results with statistics
        """
        if not self.reprojection_errors:
            return {"valid": False, "error": "No reprojection errors calculated"}
        
        mean_error = np.mean(self.reprojection_errors)
        max_error = np.max(self.reprojection_errors)
        std_error = np.std(self.reprojection_errors)
        
        validation_results = {
            "valid": max_error <= max_error_threshold,
            "mean_error": mean_error,
            "max_error": max_error,
            "std_error": std_error,
            "error_threshold": max_error_threshold,
            "num_points": len(self.reprojection_errors),
            "individual_errors": self.reprojection_errors
        }
        
        return validation_results
    
    def visualize_transformation_accuracy(self, save_path: Optional[str] = None):
        """
        Create visualization similar to your existing code
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.homography_matrix is None or not self.reprojection_errors:
            logging.error("Cannot visualize: homography not calculated or no errors available")
            return
        
        # Transform image points using calculated homography
        transformed_points = cv2.perspectiveTransform(
            self.image_points.reshape(-1, 1, 2),
            self.homography_matrix
        ).reshape(-1, 2)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot actual ground truth points
        actual_x, actual_y = self.world_points[:, 0], self.world_points[:, 1]
        plt.scatter(actual_x, actual_y, c='green', label='Ground Truth Points', s=100, marker='o')
        
        # Plot transformed points
        trans_x, trans_y = transformed_points[:, 0], transformed_points[:, 1]
        plt.scatter(trans_x, trans_y, c='red', label='Transformed Points', s=100, marker='x')
        
        # Draw error lines
        for i in range(len(self.world_points)):
            plt.plot([actual_x[i], trans_x[i]], [actual_y[i], trans_y[i]], 'k--', alpha=0.5)
        
        # Annotate errors
        for i, error in enumerate(self.reprojection_errors):
            plt.annotate(f'Err: {error:.2f}m', 
                        (actual_x[i], actual_y[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='purple')
        
        # Formatting
        plt.legend()
        plt.title(f"Homography Transformation Accuracy\nMean Error: {np.mean(self.reprojection_errors):.2f}m")
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.grid(alpha=0.3)
        plt.axis("equal")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_transformation_summary(self) -> Dict:
        """Get summary of the transformation setup"""
        if self.homography_matrix is None:
            return {"status": "No homography calculated"}
        
        validation = self.validate_homography()
        
        return {
            "status": "Ready for transformation",
            "num_calibration_points": len(self.image_points),
            "mean_reprojection_error": validation["mean_error"],
            "max_reprojection_error": validation["max_error"],
            "transformation_valid": validation["valid"],
            "homography_matrix_shape": self.homography_matrix.shape
        }