"""
User Interface Module for Object Detection System

Provides interactive interfaces for:
- Ground truth point selection
- Video annotation
- Perspective transformation setup
"""

from .point_selector import PointSelector

__all__ = ['PointSelector']
