"""
Object Mapper Module
Combines object detection, tracking, and coordinate transformation for real-world mapping
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
from ultralytics import YOLO

from .homography import HomographyCalculator


class ObjectMapper:
    """
    Complete pipeline for mapping detected objects from video to real-world coordinates
    """
    
    def __init__(self, 
                 model_path: str,
                 homography_file: str,
                 class_names: Optional[Dict[int, str]] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize object mapper with model and homography
        
        Args:
            model_path: Path to YOLO model weights
            homography_file: Path to ground truth JSON for homography calculation
            class_names: Dictionary mapping class IDs to names
            confidence_threshold: Minimum confidence score for detections (default: 0.5)
        """
        self.model = YOLO(model_path, task="detect")
        self.homography_calc = HomographyCalculator(homography_file)
        
        # For your custom retrained model: person=0, car=1
        self.class_names = class_names or {0: 'person', 1: 'car'}
        self.confidence_threshold = confidence_threshold
        
        # Calculate homography matrix
        if not self.homography_calc.calculate_homography():
            raise ValueError("Failed to calculate homography matrix")
        
        # Tracking data storage
        self.tracking_data = []
        
        logging.info(f"ObjectMapper initialized with confidence threshold: {confidence_threshold}")
        logging.info(f"Custom model classes: {self.class_names}")
    
    def get_bottom_center_coordinates(self, boxes, ids) -> List[Dict]:
        """
        Extract bottom center coordinates with track IDs and confidence filtering
        """
        bottom_coordinates = []
        if ids is None:
            ids = [None] * len(boxes)
        
        for box, track_id in zip(boxes, ids):
            # First check confidence
            confidence = float(box.conf.item())
            if confidence < self.confidence_threshold:
                continue  # Skip low-confidence detections
            
            # Get class ID - should be 0 (person) or 1 (car) for your custom model
            class_id = int(box.cls.item())
            if class_id not in self.class_names:
                logging.warning(f"Unknown class ID detected: {class_id}")
                continue  # Skip unknown classes
            
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            
            # Calculate bottom center (where object touches ground)
            bottom_center_x = int((x_min + x_max) / 2)
            bottom_center_y = int(y_max)
            
            # Also keep left/right for visualization
            bottom_left = (int(x_min), int(y_max))
            bottom_right = (int(x_max), int(y_max))
            
            bottom_coordinates.append({
                "track_id": int(track_id) if track_id is not None else None,
                "bottom_center": (bottom_center_x, bottom_center_y),
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": confidence,
                "bbox": (int(x_min), int(y_min), int(x_max), int(y_max))
            })
        
        return bottom_coordinates
    
    def transform_to_world_coordinates(self, pixel_coordinates: List[Dict]) -> List[Dict]:
        """
        Transform pixel coordinates to real-world coordinates
        """
        world_coordinates = []
        
        for coord_data in pixel_coordinates:
            # Transform bottom center point
            pixel_x, pixel_y = coord_data["bottom_center"]
            world_x, world_y = self.homography_calc.transform_point(pixel_x, pixel_y)
            
            # Add world coordinates to the data
            enhanced_data = coord_data.copy()
            enhanced_data.update({
                "world_x": world_x,
                "world_y": world_y,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y
            })
            
            world_coordinates.append(enhanced_data)
        
        return world_coordinates
    
    def process_video_stream(self, 
                           video_source: Union[str, int],
                           output_dir: str = "outputs/mapping",
                           save_video: bool = True,
                           show_display: bool = True,
                           tracker: str = "bytetrack.yaml") -> str:
        """
        Process video stream with object detection, tracking, and coordinate mapping
        """
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking data storage
        self.tracking_data = []
        
        # Process video with tracking and confidence threshold
        results = self.model.track(
            source=video_source,
            stream=True,
            show=False,
            tracker=tracker,
            save=save_video,
            save_dir=str(output_path),
            conf=self.confidence_threshold  # Apply confidence threshold at YOLO level
        )
        
        # Set up display if requested
        if show_display:
            cv2.namedWindow("Object Mapping", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        filtered_detections = 0
        total_detections = 0
        
        try:
            for result in results:
                frame_count += 1
                boxes = result.boxes
                ids = boxes.id if boxes is not None else None
                
                # Skip if no detections
                if boxes is None or len(boxes) == 0:
                    continue
                
                # Count total detections before filtering
                total_detections += len(boxes)
                
                # Get bottom coordinates (with additional filtering)
                pixel_coords = self.get_bottom_center_coordinates(boxes, ids)
                filtered_detections += len(pixel_coords)
                
                # Transform to world coordinates
                world_coords = self.transform_to_world_coordinates(pixel_coords)
                
                # Store tracking data
                for coord_data in world_coords:
                    self.tracking_data.append({
                        "frame": frame_count,
                        "track_id": coord_data["track_id"],
                        "class_name": coord_data["class_name"],
                        "confidence": coord_data["confidence"],
                        "pixel_x": coord_data["pixel_x"],
                        "pixel_y": coord_data["pixel_y"],
                        "world_x": coord_data["world_x"],
                        "world_y": coord_data["world_y"]
                    })
                
                # Visualization
                if show_display:
                    frame = self._create_annotated_frame(result, world_coords)
                    frame = self._resize_frame(frame, max_width=1280)
                    
                    cv2.imshow("Object Mapping", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    logging.info(f"Processed {frame_count} frames, kept {filtered_detections}/{total_detections} detections")
        
        except KeyboardInterrupt:
            logging.info("Processing interrupted by user")
        
        finally:
            if show_display:
                cv2.destroyAllWindows()
        
        # Save results to CSV
        csv_path = output_path / f"object_mapping_results.csv"
        self._save_tracking_data(csv_path)
        
        logging.info(f"Processing complete. Results saved to {csv_path}")
        logging.info(f"Final filtering: kept {len(self.tracking_data)}/{total_detections} total detections")
        
        return str(csv_path)
    
    def _create_annotated_frame(self, result, world_coords: List[Dict]) -> np.ndarray:
        """Create annotated frame with bounding boxes and world coordinates"""
        frame = result.plot()  # Draw bounding boxes and IDs
        
        for coord_data in world_coords:
            if coord_data["track_id"] is not None:
                # Draw bottom center point
                center = coord_data["bottom_center"]
                cv2.circle(frame, center, 5, (0, 255, 255), -1)
                
                # Draw bottom line for reference
                cv2.line(frame, coord_data["bottom_left"], coord_data["bottom_right"], (0, 255, 0), 2)
                
                # Annotation text with world coordinates and confidence
                world_text = f"World: ({coord_data['world_x']:.1f}, {coord_data['world_y']:.1f})"
                pixel_text = f"Pixel: ({coord_data['pixel_x']}, {coord_data['pixel_y']})"
                id_text = f"ID: {coord_data['track_id']} ({coord_data['class_name']}) {coord_data['confidence']:.2f}"
                
                # Position text above the object
                text_y = coord_data["bbox"][1] - 45
                cv2.putText(frame, id_text, (center[0] - 50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, world_text, (center[0] - 50, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.putText(frame, pixel_text, (center[0] - 50, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return frame
    
    def _resize_frame(self, frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
        """Resize frame while preserving aspect ratio"""
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame
    
    def _save_tracking_data(self, csv_path: Path):
        """Save tracking data to CSV file"""
        if self.tracking_data:
            df = pd.DataFrame(self.tracking_data)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved {len(self.tracking_data)} tracking records to {csv_path}")
            
            # Log confidence statistics for verification
            confidences = [record["confidence"] for record in self.tracking_data]
            min_conf = min(confidences)
            max_conf = max(confidences)
            avg_conf = sum(confidences) / len(confidences)
            
            logging.info(f"Confidence stats - Min: {min_conf:.3f}, Max: {max_conf:.3f}, Avg: {avg_conf:.3f}")
            
            # Verify all are above threshold
            below_threshold = [c for c in confidences if c < self.confidence_threshold]
            if below_threshold:
                logging.warning(f"Found {len(below_threshold)} detections below threshold!")
            else:
                logging.info(f"✅ All detections above {self.confidence_threshold} threshold")
        else:
            logging.warning("No tracking data to save")
    
    def get_object_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Extract complete trajectories for each tracked object in world coordinates
        """
        trajectories = {}
        
        for record in self.tracking_data:
            track_id = record["track_id"]
            if track_id is not None:
                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append((record["world_x"], record["world_y"]))
        
        return trajectories
