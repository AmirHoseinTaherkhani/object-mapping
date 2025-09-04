"""Pydantic schemas for API requests and responses."""

from typing import List
from pydantic import BaseModel


class DetectionResponse(BaseModel):
    """Response model for object detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x_center, y_center, width, height]
    bbox_xyxy: List[float]  # [x1, y1, x2, y2]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


class BatchDetectionRequest(BaseModel):
    """Request model for batch detection."""
    image_urls: List[str]
    confidence_threshold: float = 0.5
