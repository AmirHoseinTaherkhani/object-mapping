"""
Enhanced 2D Map Canvas with Modern Visual Design
Improved graphics, better colors, shadows, and professional appearance
"""

import cv2
import numpy as np
import json
import math
from typing import Dict, List, Tuple, Optional

class EnhancedMapCanvas:
    """Modern 2D map visualization with enhanced graphics"""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 ground_truth_file: Optional[str] = None,
                 buffer_meters: float = 10.0):
        self.width = width
        self.height = height
        self.buffer_meters = buffer_meters
        
        # Modern color scheme
        self.colors = {
            'background': (45, 45, 55),      # Dark blue-gray
            'grid_major': (80, 80, 90),      # Lighter grid lines
            'grid_minor': (65, 65, 75),      # Subtle grid lines
            'origin': (120, 120, 140),       # Origin axes
            'person': (0, 255, 150),         # Bright green
            'car': (255, 140, 0),            # Orange
            'trail': (100, 200, 255),        # Light blue
            'text': (200, 200, 220),         # Light gray text
            'shadow': (20, 20, 25)           # Dark shadow
        }
        
        if ground_truth_file:
            self.world_bounds = self._calculate_bounds_from_gt(ground_truth_file)
        else:
            self.world_bounds = (-10.0, -10.0, 10.0, 10.0)
        
        self.objects = {}
        self.trails = {}
        
    def _calculate_bounds_from_gt(self, gt_file: str) -> Tuple[float, float, float, float]:
        """Calculate world bounds from ground truth file"""
        try:
            with open(gt_file, 'r') as f:
                data = json.load(f)
            
            world_points = data['world_points']
            x_coords = [p['x'] for p in world_points]
            y_coords = [p['y'] for p in world_points]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            return (
                x_min - self.buffer_meters,
                y_min - self.buffer_meters,
                x_max + self.buffer_meters,
                y_max + self.buffer_meters
            )
        except Exception as e:
            print(f"Warning: Could not calculate bounds from {gt_file}: {e}")
            return (-10.0, -10.0, 10.0, 10.0)
    
    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        norm_x = (world_x - x_min) / (x_max - x_min)
        norm_y = (world_y - y_min) / (y_max - y_min)
        
        pixel_x = int(norm_x * self.width)
        pixel_y = int((1 - norm_y) * self.height)
        
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
        
        if track_id not in self.trails:
            self.trails[track_id] = []
        
        self.trails[track_id].append((pixel_x, pixel_y))
        
        if len(self.trails[track_id]) > 100:
            self.trails[track_id] = self.trails[track_id][-100:]
    
    def render(self, show_trails: bool = True, show_grid: bool = True) -> np.ndarray:
        """Render enhanced map with modern styling"""
        # Create dark background
        canvas = np.full((self.height, self.width, 3), self.colors['background'], dtype=np.uint8)
        
        if show_grid:
            self._draw_enhanced_grid(canvas)
        
        if show_trails:
            self._draw_enhanced_trails(canvas)
        
        self._draw_enhanced_objects(canvas)
        self._draw_enhanced_labels(canvas)
        
        return canvas
    
    def _draw_enhanced_grid(self, canvas: np.ndarray):
        """Draw modern grid with major/minor lines"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # Minor grid (1m spacing)
        for x in np.arange(math.ceil(x_min), math.floor(x_max) + 1, 1.0):
            pixel_x, _ = self.world_to_pixel(x, y_min)
            cv2.line(canvas, (pixel_x, 0), (pixel_x, self.height), self.colors['grid_minor'], 1)
        
        for y in np.arange(math.ceil(y_min), math.floor(y_max) + 1, 1.0):
            _, pixel_y = self.world_to_pixel(x_min, y)
            cv2.line(canvas, (0, pixel_y), (self.width, pixel_y), self.colors['grid_minor'], 1)
        
        # Major grid (5m spacing)
        for x in np.arange(math.ceil(x_min/5)*5, math.floor(x_max/5)*5 + 1, 5.0):
            pixel_x, _ = self.world_to_pixel(x, y_min)
            cv2.line(canvas, (pixel_x, 0), (pixel_x, self.height), self.colors['grid_major'], 2)
            # Label
            cv2.putText(canvas, f"{int(x)}", (pixel_x - 8, self.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        for y in np.arange(math.ceil(y_min/5)*5, math.floor(y_max/5)*5 + 1, 5.0):
            _, pixel_y = self.world_to_pixel(x_min, y)
            cv2.line(canvas, (0, pixel_y), (self.width, pixel_y), self.colors['grid_major'], 2)
            # Label
            cv2.putText(canvas, f"{int(y)}", (10, pixel_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
        
        # Origin axes
        if x_min <= 0 <= x_max:
            pixel_x, _ = self.world_to_pixel(0, y_min)
            cv2.line(canvas, (pixel_x, 0), (pixel_x, self.height), self.colors['origin'], 3)
        
        if y_min <= 0 <= y_max:
            _, pixel_y = self.world_to_pixel(x_min, 0)
            cv2.line(canvas, (0, pixel_y), (self.width, pixel_y), self.colors['origin'], 3)
    
    def _draw_enhanced_trails(self, canvas: np.ndarray):
        """Draw trails with gradient fade and glow effect"""
        for track_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            
            color = self.colors['person'] if track_id % 2 == 0 else self.colors['car']
            
            for i in range(1, len(trail)):
                alpha = (i / len(trail)) * 0.8 + 0.2
                thickness = int(3 * alpha)
                
                # Glow effect
                glow_color = tuple(int(c * alpha * 0.3) for c in color)
                cv2.line(canvas, trail[i-1], trail[i], glow_color, thickness + 4)
                
                # Main line
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(canvas, trail[i-1], trail[i], line_color, thickness)
    
    def _draw_enhanced_objects(self, canvas: np.ndarray):
        """Draw objects with shadows and modern styling"""
        for track_id, obj in self.objects.items():
            pixel_x, pixel_y = obj['pixel_pos']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Color based on class
            color = self.colors['person'] if class_name == 'person' else self.colors['car']
            
            # Shadow
            cv2.circle(canvas, (pixel_x + 2, pixel_y + 2), 10, self.colors['shadow'], -1)
            
            # Main circle with gradient effect
            cv2.circle(canvas, (pixel_x, pixel_y), 10, color, -1)
            cv2.circle(canvas, (pixel_x, pixel_y), 8, tuple(int(c*1.3) for c in color), -1)
            
            # Border
            cv2.circle(canvas, (pixel_x, pixel_y), 10, (255, 255, 255), 2)
            
            # ID label with background
            text = f"ID:{track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Text background
            cv2.rectangle(canvas, 
                         (pixel_x + 15, pixel_y - 15),
                         (pixel_x + 15 + text_size[0] + 6, pixel_y - 15 + text_size[1] + 6),
                         (0, 0, 0), -1)
            
            # Text
            cv2.putText(canvas, text, (pixel_x + 18, pixel_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _draw_enhanced_labels(self, canvas: np.ndarray):
        """Draw coordinate labels and title"""
        # Title
        cv2.putText(canvas, "Real-World Coordinate Map", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Scale indicator
        x_range = self.world_bounds[2] - self.world_bounds[0]
        cv2.putText(canvas, f"Scale: {x_range:.1f}m x {x_range:.1f}m", 
                   (self.width - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
    
    def clear_trails(self):
        """Clear all trails"""
        self.trails.clear()
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get track color for video overlay"""
        colors = [
            (0, 255, 150),   # Green
            (255, 140, 0),   # Orange
            (255, 100, 255), # Pink
            (100, 255, 255), # Cyan
            (255, 255, 100), # Yellow
            (255, 100, 100), # Light red
        ]
        return colors[track_id % len(colors)]
