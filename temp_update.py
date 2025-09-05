# This shows the changes needed in RealtimeDisplay.__init__

def __init__(self, config_path: str = "configs/visualization/realtime.yaml"):
    """Initialize realtime display system"""
    self.config = self._load_config(config_path)
    
    # Store ground truth file path for later use
    self.ground_truth_file = None
    
    # Initialize map canvas (will be recreated with proper bounds when GT file is provided)
    canvas_config = self.config['visualization']['canvas']
    
    self.map_canvas = MapCanvas(
        width=canvas_config['width'],
        height=canvas_config['height'],
        ground_truth_file=None,  # Will be updated later
        buffer_meters=10.0,
        background_color=tuple(canvas_config['background_color'])
    )
    
    # Rest of initialization stays the same...
