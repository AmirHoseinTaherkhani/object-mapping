    def _draw_grid(self, canvas: np.ndarray):
        """Draw coordinate grid with axis labels and tick marks"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # Grid spacing (every 5 meters)
        grid_spacing = 5.0
        
        # Draw vertical grid lines with labels
        x = x_min
        while x <= x_max:
            if x % grid_spacing == 0:
                pixel_x, _ = self.world_to_pixel(x, y_min)
                # Grid line
                cv2.line(canvas, (pixel_x, 0), (pixel_x, self.height), (200, 200, 200), 1)
                # X-axis label at bottom
                cv2.putText(canvas, f"{int(x)}m", (pixel_x - 15, self.height - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            x += grid_spacing
        
        # Draw horizontal grid lines with labels
        y = y_min
        while y <= y_max:
            if y % grid_spacing == 0:
                _, pixel_y = self.world_to_pixel(x_min, y)
                # Grid line
                cv2.line(canvas, (0, pixel_y), (self.width, pixel_y), (200, 200, 200), 1)
                # Y-axis label at left
                cv2.putText(canvas, f"{int(y)}m", (5, pixel_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            y += grid_spacing
        
        # Draw origin axes (x=0, y=0) in darker color if they exist in bounds
        if x_min <= 0 <= x_max:
            pixel_x, _ = self.world_to_pixel(0, y_min)
            cv2.line(canvas, (pixel_x, 0), (pixel_x, self.height), (100, 100, 100), 2)
        
        if y_min <= 0 <= y_max:
            _, pixel_y = self.world_to_pixel(x_min, 0)
            cv2.line(canvas, (0, pixel_y), (self.width, pixel_y), (100, 100, 100), 2)
