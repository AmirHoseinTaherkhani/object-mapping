import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Track:
    id: int
    bbox: List[float]
    confidence: float
    class_name: str
    age: int = 0
    hits: int = 0

class SimpleTracker:
    """Simple IoU-based object tracker"""
    
    def __init__(self, max_disappeared: int = 10, iou_threshold: float = 0.3):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        if not detections:
            self._age_tracks()
            return []
        
        matched_tracks = {}
        unmatched_detections = []
        
        for detection in detections:
            det_bbox = detection['bbox']
            best_match_id = None
            best_iou = self.iou_threshold
            
            for track_id, track in self.tracks.items():
                if track.class_name == detection['class_name']:
                    iou = self.calculate_iou(det_bbox, track.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                self.tracks[best_match_id].bbox = det_bbox
                self.tracks[best_match_id].confidence = detection['confidence']
                self.tracks[best_match_id].age = 0
                self.tracks[best_match_id].hits += 1
                matched_tracks[best_match_id] = detection
            else:
                unmatched_detections.append(detection)
        
        for detection in unmatched_detections:
            track = Track(
                id=self.next_id,
                bbox=detection['bbox'],
                confidence=detection['confidence'],
                class_name=detection['class_name'],
                hits=1
            )
            self.tracks[self.next_id] = track
            matched_tracks[self.next_id] = detection
            self.next_id += 1
        
        self._age_tracks()
        
        tracked_detections = []
        for track_id, detection in matched_tracks.items():
            detection['track_id'] = track_id
            tracked_detections.append(detection)
        
        return tracked_detections
    
    def _age_tracks(self):
        """Age tracks and remove disappeared ones"""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            track.age += 1
            if track.age > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
