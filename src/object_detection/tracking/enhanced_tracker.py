import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2

@dataclass
class Track:
    id: int
    bbox: List[float]
    confidence: float
    class_name: str
    age: int = 0
    hits: int = 0
    velocity: Tuple[float, float] = (0.0, 0.0)
    predicted_center: Optional[Tuple[float, float]] = None

class EnhancedTracker:
    """Enhanced tracker with motion prediction for moving objects"""
    
    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3, 
                 velocity_weight: float = 0.3, min_hits_for_tracking: int = 3):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.velocity_weight = velocity_weight
        self.min_hits_for_tracking = min_hits_for_tracking
    
    def xywh_to_xyxy(self, bbox: List[float]) -> List[float]:
        """Convert YOLO format to xyxy"""
        x, y, w, h = bbox
        return [x - w/2, y - h/2, x + w/2, y + h/2]
    
    def get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bbox"""
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:  # Already xyxy
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        else:  # xywh format
            return (bbox[0], bbox[1])
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU with format handling"""
        xyxy1 = box1 if box1[0] < box1[2] else self.xywh_to_xyxy(box1)
        xyxy2 = box2 if box2[0] < box2[2] else self.xywh_to_xyxy(box2)
        
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
    
    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between centers"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def predict_position(self, track: Track) -> Tuple[float, float]:
        """Predict next position based on velocity"""
        current_center = self.get_center(track.bbox)
        predicted_x = current_center[0] + track.velocity[0]
        predicted_y = current_center[1] + track.velocity[1]
        return (predicted_x, predicted_y)
    
    def update_velocity(self, track: Track, new_center: Tuple[float, float]):
        """Update track velocity with smoothing"""
        if track.hits > 1:
            old_center = self.get_center(track.bbox)
            new_velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
            
            # Smooth velocity with previous velocity
            alpha = 0.7  # Smoothing factor
            track.velocity = (
                alpha * new_velocity[0] + (1 - alpha) * track.velocity[0],
                alpha * new_velocity[1] + (1 - alpha) * track.velocity[1]
            )
        track.predicted_center = self.predict_position(track)
    
    def calculate_match_score(self, detection_bbox: List[float], track: Track) -> float:
        """Calculate comprehensive matching score"""
        # IoU score
        iou_score = self.calculate_iou(detection_bbox, track.bbox)
        
        # Distance to predicted position
        det_center = self.get_center(detection_bbox)
        if track.predicted_center and track.hits > 2:
            pred_distance = self.calculate_distance(det_center, track.predicted_center)
            # Normalize distance score (smaller distance = higher score)
            max_distance = 100  # pixels
            distance_score = max(0, 1 - (pred_distance / max_distance))
        else:
            distance_score = 0.5  # Neutral if no prediction
        
        # Combine scores
        combined_score = (1 - self.velocity_weight) * iou_score + self.velocity_weight * distance_score
        return combined_score
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Enhanced update with motion prediction"""
        if not detections:
            self._age_tracks()
            return []
        
        # Create cost matrix for matching
        cost_matrix = []
        track_ids = list(self.tracks.keys())
        
        for detection in detections:
            det_costs = []
            for track_id in track_ids:
                track = self.tracks[track_id]
                if track.class_name == detection['class_name']:
                    score = self.calculate_match_score(detection['bbox'], track)
                    cost = 1 - score  # Convert to cost (lower is better)
                    det_costs.append(cost if score > self.iou_threshold else 1.0)
                else:
                    det_costs.append(1.0)  # High cost for class mismatch
            cost_matrix.append(det_costs)
        
        # Hungarian algorithm (simplified greedy matching)
        matches = []
        used_detections = set()
        used_tracks = set()
        
        # Sort by best scores first
        detection_track_pairs = []
        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_ids):
                if track_id not in used_tracks and det_idx not in used_detections:
                    cost = cost_matrix[det_idx][track_idx]
                    if cost < (1 - self.iou_threshold):
                        detection_track_pairs.append((cost, det_idx, track_id))
        
        # Sort by cost (best matches first)
        detection_track_pairs.sort(key=lambda x: x[0])
        
        # Assign matches
        for cost, det_idx, track_id in detection_track_pairs:
            if det_idx not in used_detections and track_id not in used_tracks:
                matches.append((det_idx, track_id))
                used_detections.add(det_idx)
                used_tracks.add(track_id)
        
        # Update matched tracks
        for det_idx, track_id in matches:
            detection = detections[det_idx]
            track = self.tracks[track_id]
            
            new_center = self.get_center(detection['bbox'])
            self.update_velocity(track, new_center)
            
            track.bbox = detection['bbox']
            track.confidence = detection['confidence']
            track.age = 0
            track.hits += 1
        
        # Create new tracks for unmatched detections
        new_tracks = []
        for det_idx, detection in enumerate(detections):
            if det_idx not in used_detections:
                track = Track(
                    id=self.next_id,
                    bbox=detection['bbox'],
                    confidence=detection['confidence'],
                    class_name=detection['class_name'],
                    hits=1
                )
                self.tracks[self.next_id] = track
                new_tracks.append((det_idx, self.next_id))
                self.next_id += 1
        
        # Age tracks
        self._age_tracks()
        
        # Prepare output (only return tracks with enough hits)
        tracked_detections = []
        for det_idx, track_id in matches + new_tracks:
            track = self.tracks.get(track_id)
            if track and track.hits >= self.min_hits_for_tracking:
                detection = detections[det_idx].copy()
                detection['track_id'] = track_id
                tracked_detections.append(detection)
        
        return tracked_detections
    
    def _age_tracks(self):
        """Age tracks and remove old ones"""
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            track.age += 1
            if track.age > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
