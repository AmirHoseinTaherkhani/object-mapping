"""
Mapping Module for Object Detection System

Handles perspective transformation and coordinate mapping:
- Homography calculation from ground truth points
- Pixel to real-world coordinate transformation
- Transformation validation and error analysis
- Complete video-to-map object tracking pipeline
"""

from .homography import HomographyCalculator
from .object_mapper import ObjectMapper

__all__ = ['HomographyCalculator', 'ObjectMapper']
