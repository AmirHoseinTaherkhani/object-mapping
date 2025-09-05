"""
2D Map Canvas for Real-World Coordinate Visualization
Dynamic bounds based on ground truth annotations
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional

class MapCanvas:
    """2D map visualization with dynamic world coordinate bounds"""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 ground_truth_file: Optional[str] = None,
                 buffer_meters: float = 10.0,
                 background_color: Tuple[int, int, int] = (50, 50, 50)):
        self.width = width
        self.height = height
        self.background_color = background_color
        self.buffer_meters = buffer_meters
        
        # Calculate world bounds from ground truth
        if ground_truth_file:
            self.world_bounds = self._calculate_bounds_from_gt(ground_truth_file)
        else:
            # Default fallback bounds
            self.world_bounds = (-10.0, -10.0, 10.0, 10.0)
        
        self.objects = {}
        self.trails = {}
        
        print(f"Map bounds: X({self.world_bounds[0]:.1f}, {self.world_bounds[2]:.1f}) Y({self.world_bounds[1]:.1f}, {self.world_bounds[3]:.1f})")
    
    def _calculate_bounds_from_gt(self, gt_file: str) -> Tuple[float, float, float, float]:
        """Calculate world bounds from ground truth file with buffer"""
        try:
            with open(gt_file, 'r') as f:
                data = json.load(f)
            
            world_points = data['world_points']
            x_coords = [p['x'] for p in world_points]
            y_coords = [p['y'] for p in world_points]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add buffer
            return (
                x_min - self.buffer_meters,  # x_min
                y_min - self.buffer_meters,  # y_min  
                x_max + self.buffer_meters,  # x_max
                y_max + self.buffer_meters   # y_max
            )
        except Exception as e:
            print(f"Warning: Could not calculate bounds from {gt_file}: {e}")
            return (-10.0, -10.0, 10.0, 10.0)
    
    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # Normalize to [0, 1]
        norm_x = (world_x - x_min) / (x_max - x_min)
        norm_y = (world_y - y_min) / (y_max - y_min)
        
        # Convert to pixel coordinates (flip Y axis)
        pixel_x = int(norm_x * self.width)
        pixel_y = int((1 - norm_y) * self.height)  # Flip Y
        
        return pixel_x, pixel_y
    
    def update_object(self, track_id: int, world_x: float, world_y: float, 
                     class_name: str, confidence: float):
        """Update object position and trail"""
        pixel_x, pixel_y = self.world_to_pixel(world_x, world_y)
        
        self.objects[track_id] = {
            'pixel_pos': (pixel_x, pixel_y),
            'world_pos': (world_x, world_y),
            'class_name': class_name,
            'confidence': confidence
        }
        
        # Update trail
        if track_id not in self.trails:
            self.trails[track_id] = []
        
        self.trails[track_id].append((pixel_x, pixel_y))
        
        # Limit trail length
        max_trail_length = 50
        if len(self.trails[track_id]) > max_trail_length:
            self.trails[track_id] = self.trails[track_id][-max_trail_length:]
    
    def render(self, show_trails: bool = True, show_grid: bool = True) -> np.ndarray:
        """Render the map with objects and trails"""
        # Create blank canvas
        canvas = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)
        
        # Draw grid
        if show_grid:
            self._draw_grid(canvas)
        
        # Draw trails
        if show_trails:
            self._draw_trails(canvas)
        
        # Draw objects
        self._draw_objects(canvas)
        
        # Draw coordinate labels
        self._draw_coordinate_labels(canvas)
        
        return canvas
    
    def _draw_grid(self, canvas: np.ndarray):
        """Draw coordinate grid"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # Grid spacing (every 5 meters)
        grid_spacing = 5.0
        
        # Vertical lines
        x = x_min
        while x <= x_max:
            if x % grid_spacing == 0:
                pixel_x, _ = self.world_to_pixel(x, y_min)
                cv2.line(canvas, (pixel_x, 0), (pixel_x, self.height), (80, 80, 80), 1)
            x += grid_spacing
        
        # Horizontal lines  
        y = y_min
        while y <= y_max:
            if y % grid_spacing == 0:
                _, pixel_y = self.world_to_pixel(x_min, y)
                cv2.line(canvas, (0, pixel_y), (self.width, pixel_y), (80, 80, 80), 1)
            y += grid_spacing
    
    def _draw_trails(self, canvas: np.ndarray):
        """Draw object movement trails with fade effect"""
        for track_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            
            for i in range(1, len(trail)):
                # Fade effect: newer points are brighter
                alpha = i / len(trail)
                color = (int(100 * alpha), int(150 * alpha), int(255 * alpha))
                
                cv2.line(canvas, trail[i-1], trail[i], color, 2)
    
    def _draw_objects(self, canvas: np.ndarray):
        """Draw current object positions"""
        for track_id, obj in self.objects.items():
            pixel_x, pixel_y = obj['pixel_pos']
            class_name = obj['class_name']
            
            # Color based on class
            color = (0, 255, 0) if class_name == 'person' else (255, 100, 0)
            
            # Draw circle
            cv2.circle(canvas, (pixel_x, pixel_y), 6, color, -1)
            cv2.circle(canvas, (pixel_x, pixel_y), 8, (255, 255, 255), 2)
            
            # Draw track ID
            cv2.putText(canvas, str(track_id), (pixel_x + 10, pixel_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_coordinate_labels(self, canvas: np.ndarray):
        """Draw coordinate system labels"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # Corner labels
        cv2.putText(canvas, f"({x_min:.0f},{y_max:.0f})", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(canvas, f"({x_max:.0f},{y_min:.0f})", (self.width-100, self.height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def clear_trails(self):
        """Clear all movement trails"""
        self.trails.clear()
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a track ID"""
        # Generate consistent colors based on track ID
        colors = [
            (0, 255, 0),    # Green
            (255, 100, 0),  # Orange  
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
            (255, 0, 0),    # Red
            (0, 0, 255),    # Blue
            (128, 255, 0),  # Lime
        ]
        return colors[track_id % len(colors)]
