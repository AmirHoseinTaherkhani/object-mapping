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
    """Simple IoU-based object tracker with bbox format handling"""
    
    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
    
    def xywh_to_xyxy(self, bbox: List[float]) -> List[float]:
        """Convert YOLO format (center_x, center_y, width, height) to (x1, y1, x2, y2)"""
        x, y, w, h = bbox
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return [x1, y1, x2, y2]
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union - handles both formats"""
        # Convert both boxes to xyxy format
        if len(box1) == 4:
            # Assume YOLO format if values look like center coordinates
            if box1[0] < box1[2] and box1[1] < box1[3]:
                # Already in xyxy format
                xyxy1 = box1
            else:
                # Convert from xywh
                xyxy1 = self.xywh_to_xyxy(box1)
        else:
            xyxy1 = box1
            
        if len(box2) == 4:
            if box2[0] < box2[2] and box2[1] < box2[3]:
                xyxy2 = box2
            else:
                xyxy2 = self.xywh_to_xyxy(box2)
        else:
            xyxy2 = box2
        
        # Calculate intersection
        x1 = max(xyxy1[0], xyxy2[0])
        y1 = max(xyxy1[1], xyxy2[1])
        x2 = min(xyxy1[2], xyxy2[2])
        y2 = min(xyxy1[3], xyxy2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (xyxy1[2] - xyxy1[0]) * (xyxy1[3] - xyxy1[1])
        area2 = (xyxy2[2] - xyxy2[0]) * (xyxy2[3] - xyxy2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        if not detections:
            self._age_tracks()
            return []
        
        # Create cost matrix for Hungarian algorithm (simplified greedy matching)
        matched_pairs = []
        used_detection_idx = set()
        used_track_ids = set()
        
        # Match existing tracks
        for det_idx, detection in enumerate(detections):
            if det_idx in used_detection_idx:
                continue
                
            det_bbox = detection['bbox']
            best_match_id = None
            best_iou = self.iou_threshold
            
            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                if track.class_name == detection['class_name']:
                    iou = self.calculate_iou(det_bbox, track.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                matched_pairs.append((best_match_id, det_idx))
                used_track_ids.add(best_match_id)
                used_detection_idx.add(det_idx)
        
        # Update matched tracks
        for track_id, det_idx in matched_pairs:
            detection = detections[det_idx]
            self.tracks[track_id].bbox = detection['bbox']
            self.tracks[track_id].confidence = detection['confidence']
            self.tracks[track_id].age = 0
            self.tracks[track_id].hits += 1
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detection_idx:
                track = Track(
                    id=self.next_id,
                    bbox=detection['bbox'],
                    confidence=detection['confidence'],
                    class_name=detection['class_name'],
                    hits=1
                )
                self.tracks[self.next_id] = track
                matched_pairs.append((self.next_id, det_idx))
                self.next_id += 1
        
        # Age tracks
        self._age_tracks()
        
        # Prepare output
        tracked_detections = []
        for track_id, det_idx in matched_pairs:
            detection = detections[det_idx].copy()
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
            
    def get_active_tracks(self) -> Dict[int, Track]:
        """Get currently active tracks"""
        return {tid: track for tid, track in self.tracks.items() if track.age <= self.max_disappeared}
