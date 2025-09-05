import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import colorsys

class MapCanvas:
    """Real-time 2D map visualization for object tracking"""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 world_bounds: Tuple[float, float, float, float] = (-10, -10, 10, 10),
                 background_color: Tuple[int, int, int] = (50, 50, 50)):
        """
        Initialize map canvas
        
        Args:
            width, height: Canvas dimensions in pixels
            world_bounds: (min_x, min_y, max_x, max_y) in world coordinates
            background_color: RGB background color
        """
        self.width = width
        self.height = height
        self.world_bounds = world_bounds  # (min_x, min_y, max_x, max_y)
        self.background_color = background_color
        
        # Create canvas
        self.canvas = np.full((height, width, 3), background_color, dtype=np.uint8)
        self.trails = {}  # track_id -> list of (x, y) positions
        self.track_colors = {}  # track_id -> color
        
        # Calculate scale factors
        self.scale_x = width / (world_bounds[2] - world_bounds[0])
        self.scale_y = height / (world_bounds[3] - world_bounds[1])
        
    def world_to_pixel(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        pixel_x = int((world_x - self.world_bounds[0]) * self.scale_x)
        pixel_y = int(self.height - (world_y - self.world_bounds[1]) * self.scale_y)  # Flip Y
        return pixel_x, pixel_y
    
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for track ID"""
        if track_id not in self.track_colors:
            hue = (track_id * 0.618033988749895) % 1.0  # Golden ratio for good color distribution
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            self.track_colors[track_id] = tuple(int(c * 255) for c in rgb)
        return self.track_colors[track_id]
    
    def update_object(self, track_id: int, world_x: float, world_y: float, 
                     class_name: str = "", confidence: float = 0.0):
        """Update object position on map"""
        pixel_x, pixel_y = self.world_to_pixel(world_x, world_y)
        
        # Check if point is within canvas bounds
        if 0 <= pixel_x < self.width and 0 <= pixel_y < self.height:
            # Add to trail
            if track_id not in self.trails:
                self.trails[track_id] = []
            self.trails[track_id].append((pixel_x, pixel_y))
            
            # Limit trail length
            if len(self.trails[track_id]) > 50:
                self.trails[track_id].pop(0)
    
    def draw_grid(self, grid_spacing: float = 1.0):
        """Draw coordinate grid on canvas"""
        grid_color = (80, 80, 80)
        
        # Vertical lines
        for x in np.arange(self.world_bounds[0], self.world_bounds[2], grid_spacing):
            px, py1 = self.world_to_pixel(x, self.world_bounds[1])
            px, py2 = self.world_to_pixel(x, self.world_bounds[3])
            if 0 <= px < self.width:
                cv2.line(self.canvas, (px, py1), (px, py2), grid_color, 1)
        
        # Horizontal lines
        for y in np.arange(self.world_bounds[1], self.world_bounds[3], grid_spacing):
            px1, py = self.world_to_pixel(self.world_bounds[0], y)
            px2, py = self.world_to_pixel(self.world_bounds[2], y)
            if 0 <= py < self.height:
                cv2.line(self.canvas, (px1, py), (px2, py), grid_color, 1)
    
    def render(self, show_trails: bool = True, show_grid: bool = True) -> np.ndarray:
        """Render the complete map"""
        # Reset canvas
        self.canvas.fill(0)
        self.canvas[:] = self.background_color
        
        # Draw grid
        if show_grid:
            self.draw_grid()
        
        # Draw trails and current positions
        for track_id, trail in self.trails.items():
            if not trail:
                continue
                
            color = self.get_track_color(track_id)
            
            # Draw trail
            if show_trails and len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha = i / len(trail)  # Fade effect
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(self.canvas, trail[i-1], trail[i], trail_color, 2)
            
            # Draw current position
            if trail:
                current_pos = trail[-1]
                cv2.circle(self.canvas, current_pos, 6, color, -1)
                cv2.circle(self.canvas, current_pos, 8, (255, 255, 255), 2)
                
                # Draw track ID
                cv2.putText(self.canvas, str(track_id), 
                          (current_pos[0] + 10, current_pos[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return self.canvas.copy()
    
    def clear_trails(self):
        """Clear all trails"""
        self.trails.clear()
    
    def remove_track(self, track_id: int):
        """Remove specific track"""
        if track_id in self.trails:
            del self.trails[track_id]
        if track_id in self.track_colors:
            del self.track_colors[track_id]
