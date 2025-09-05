"""
Real-time visualization components for object tracking and mapping.

This module provides:
- MapCanvas: 2D map rendering with trails and coordinate transformation
- RealtimeDisplay: Side-by-side video and map visualization coordination
"""

from .map_canvas import MapCanvas
from .realtime_display import RealtimeDisplay

__all__ = [
   'MapCanvas',
   'RealtimeDisplay'
]

__version__ = '1.0.0'
