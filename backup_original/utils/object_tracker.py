"""
Object Tracking System untuk License Plate Detection
Mengatasi masalah bounding box yang tidak konsisten dengan tracking ID dan korelasi spasial
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import math

@dataclass
class TrackedObject:
    """Data class untuk objek yang sedang di-track"""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    last_seen: float
    age: int = 0  # Berapa frame objek ini sudah di-track
    hit_streak: int = 0  # Berapa frame berturut-turut objek ini terdeteksi
    time_since_update: int = 0  # Frame sejak terakhir di-update
    velocity: Tuple[float, float] = (0.0, 0.0)  # Velocity (dx, dy) per frame
    predicted_bbox: Tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))
    is_vehicle: bool = False
    associated_plates: List[int] = field(default_factory=list)  # ID plat yang terkait
    
    def update_position(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update posisi objek dengan bounding box baru"""
        old_center = self.get_center()
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.hit_streak += 1
        self.time_since_update = 0
        self.age += 1
        
        # Calculate velocity
        new_center = self.get_center()
        self.velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
    
    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """Prediksi posisi berikutnya berdasarkan velocity"""
        x, y, w, h = self.bbox
        dx, dy = self.velocity
        
        # Apply velocity with damping factor
        damping = 0.8
        predicted_x = int(x + dx * damping)
        predicted_y = int(y + dy * damping)
        
        self.predicted_bbox = (predicted_x, predicted_y, w, h)
        return self.predicted_bbox
    
    def get_center(self) -> Tuple[float, float]:
        """Dapatkan center point dari bounding box"""
        x, y, w, h = self.bbox
        return (x + w/2, y + h/2)
    
    def get_area(self) -> float:
        """Dapatkan area dari bounding box"""
        _, _, w, h = self.bbox
        return w * h
    
    def is_active(self) -> bool:
        """Check apakah objek masih aktif (baru saja terdeteksi)"""
        return self.hit_streak >= 1 and self.time_since_update < 3

@dataclass 
class Detection:
    """Data class untuk hasil deteksi baru"""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    is_vehicle: bool = False

class ObjectTracker:
    """
    Multi-object tracking system dengan Hungarian algorithm untuk assignment
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100, 
                 min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize object tracker
        
        Args:
            max_disappeared: Max frame objek hilang sebelum dihapus
            max_distance: Max distance untuk matching (pixel)
            min_hits: Min deteksi berturut sebelum konfirmasi tracking
            iou_threshold: IoU threshold untuk matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_id = 1
        
        # Statistics
        self.total_tracks = 0
        self.active_tracks = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ObjectTracker initialized - max_disappeared: {max_disappeared}, max_distance: {max_distance}")
    
    def calculate_distance(self, bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate distance antara dua bounding box (center to center)
        
        Args:
            bbox1, bbox2: Bounding boxes (x, y, w, h)
            
        Returns:
            float: Distance dalam pixel
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) antara dua bounding box
        
        Args:
            bbox1, bbox2: Bounding boxes (x, y, w, h)
            
        Returns:
            float: IoU value [0, 1]
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Convert ke format (x1, y1, x2, y2)
        box1 = (x1, y1, x1 + w1, y1 + h1)
        box2 = (x2, y2, x2 + w2, y2 + h2)
        
        # Calculate intersection
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_cost_matrix(self, detections: List[Detection]) -> np.ndarray:
        """
        Create cost matrix untuk Hungarian algorithm
        
        Args:
            detections: List deteksi baru
            
        Returns:
            np.ndarray: Cost matrix (tracks x detections)
        """
        if not self.tracked_objects or not detections:
            return np.array([])
        
        tracks = list(self.tracked_objects.values())
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, detection in enumerate(detections):
                # Gunakan predicted position untuk cost calculation
                predicted_bbox = track.predict_next_position()
                
                # Calculate distance cost
                distance = self.calculate_distance(predicted_bbox, detection.bbox)
                
                # Calculate IoU cost (1 - IoU untuk minimization)
                iou = self.calculate_iou(predicted_bbox, detection.bbox)
                iou_cost = 1.0 - iou
                
                # Combine costs dengan weight
                distance_weight = 0.6
                iou_weight = 0.4
                
                # Normalize distance (max 200 pixel)
                normalized_distance = min(distance / 200.0, 1.0)
                
                total_cost = distance_weight * normalized_distance + iou_weight * iou_cost
                
                # Penalty untuk class mismatch
                if track.class_name != detection.class_name:
                    total_cost += 0.5
                
                # Penalty jika distance/IoU terlalu buruk
                if distance > self.max_distance or iou < self.iou_threshold:
                    total_cost = 999999  # Very high cost
                
                cost_matrix[i, j] = total_cost
        
        return cost_matrix
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracking dengan deteksi baru
        
        Args:
            detections: List deteksi baru dari detector
            
        Returns:
            List[TrackedObject]: List objek yang sedang di-track
        """
        # Update time_since_update untuk semua tracks
        for track in self.tracked_objects.values():
            track.time_since_update += 1
        
        # Jika tidak ada deteksi, hanya prediksi posisi existing tracks
        if not detections:
            self._cleanup_tracks()
            return self._get_active_tracks()
        
        # Create cost matrix dan solve assignment problem
        cost_matrix = self.create_cost_matrix(detections)
        
        if cost_matrix.size > 0:
            # Solve Hungarian assignment problem
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            
            # Apply valid assignments
            used_detections = set()
            for track_idx, det_idx in zip(track_indices, detection_indices):
                if cost_matrix[track_idx, det_idx] < 999999:  # Valid assignment
                    track_id = list(self.tracked_objects.keys())[track_idx]
                    detection = detections[det_idx]
                    
                    # Update existing track
                    self.tracked_objects[track_id].update_position(
                        detection.bbox, detection.confidence
                    )
                    used_detections.add(det_idx)
                    
                    self.logger.debug(f"Updated track {track_id} with detection {det_idx}")
        else:
            used_detections = set()
        
        # Create new tracks untuk unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                new_track = TrackedObject(
                    id=self.next_id,
                    class_name=detection.class_name,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    last_seen=time.time(),
                    is_vehicle=detection.is_vehicle
                )
                
                self.tracked_objects[self.next_id] = new_track
                self.total_tracks += 1
                self.next_id += 1
                
                self.logger.debug(f"Created new track {new_track.id} for class {detection.class_name}")
        
        # Cleanup old tracks
        self._cleanup_tracks()
        
        # Update statistics
        self.active_tracks = len(self._get_active_tracks())
        
        return self._get_active_tracks()
    
    def _cleanup_tracks(self):
        """Remove tracks yang sudah terlalu lama hilang"""
        to_remove = []
        
        for track_id, track in self.tracked_objects.items():
            if track.time_since_update > self.max_disappeared:
                to_remove.append(track_id)
                self.logger.debug(f"Removing track {track_id} (disappeared for {track.time_since_update} frames)")
        
        for track_id in to_remove:
            del self.tracked_objects[track_id]
    
    def _get_active_tracks(self) -> List[TrackedObject]:
        """Get tracks yang aktif (memenuhi min_hits requirement)"""
        active_tracks = []
        
        for track in self.tracked_objects.values():
            if track.hit_streak >= self.min_hits or track.time_since_update < 2:
                active_tracks.append(track)
        
        return active_tracks
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Get track berdasarkan ID"""
        return self.tracked_objects.get(track_id)
    
    def get_vehicle_tracks(self) -> List[TrackedObject]:
        """Get hanya tracks kendaraan"""
        return [track for track in self._get_active_tracks() if track.is_vehicle]
    
    def associate_plate_with_vehicle(self, plate_bbox: Tuple[int, int, int, int], 
                                   plate_id: int) -> Optional[int]:
        """
        Associate plat nomor dengan kendaraan terdekat
        
        Args:
            plate_bbox: Bounding box plat nomor
            plate_id: ID plat nomor
            
        Returns:
            Optional[int]: Vehicle track ID jika berhasil di-associate
        """
        vehicle_tracks = self.get_vehicle_tracks()
        
        if not vehicle_tracks:
            return None
        
        best_vehicle = None
        best_score = float('inf')
        
        px, py, pw, ph = plate_bbox
        plate_center = (px + pw/2, py + ph/2)
        
        for vehicle in vehicle_tracks:
            vx, vy, vw, vh = vehicle.bbox
            
            # Check apakah plat berada di dalam atau dekat dengan kendaraan
            # Plat harus berada di bagian bawah kendaraan (biasanya)
            if (vx <= px + pw and px <= vx + vw and  # Overlap horizontal
                vy <= py + ph and py <= vy + vh):    # Overlap vertical
                
                # Calculate distance dari center plat ke center kendaraan
                vehicle_center = vehicle.get_center()
                distance = math.sqrt((plate_center[0] - vehicle_center[0])**2 + 
                                   (plate_center[1] - vehicle_center[1])**2)
                
                # Prefer kendaraan yang lebih dekat
                if distance < best_score:
                    best_score = distance
                    best_vehicle = vehicle
        
        if best_vehicle and plate_id not in best_vehicle.associated_plates:
            best_vehicle.associated_plates.append(plate_id)
            return best_vehicle.id
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        active_tracks = self._get_active_tracks()
        vehicle_tracks = [t for t in active_tracks if t.is_vehicle]
        
        return {
            'total_tracks_created': self.total_tracks,
            'active_tracks': len(active_tracks),
            'vehicle_tracks': len(vehicle_tracks),
            'average_track_age': np.mean([t.age for t in active_tracks]) if active_tracks else 0,
            'average_confidence': np.mean([t.confidence for t in active_tracks]) if active_tracks else 0
        }
    
    def reset(self):
        """Reset semua tracking state"""
        self.tracked_objects.clear()
        self.next_id = 1
        self.total_tracks = 0
        self.active_tracks = 0
        self.logger.info("Tracking state reset")


def test_object_tracker():
    """Test function untuk object tracker"""
    print("Testing ObjectTracker...")
    
    tracker = ObjectTracker(max_disappeared=10, max_distance=50)
    
    # Simulate detections over multiple frames
    test_detections = [
        # Frame 1
        [Detection((100, 100, 50, 30), 0.9, "car", True),
         Detection((200, 150, 45, 28), 0.8, "car", True)],
        
        # Frame 2 - objects moved slightly
        [Detection((105, 102, 50, 30), 0.9, "car", True),
         Detection((205, 148, 45, 28), 0.85, "car", True)],
        
        # Frame 3 - one object disappeared, new one appeared
        [Detection((110, 104, 50, 30), 0.95, "car", True),
         Detection((300, 200, 40, 25), 0.7, "motorcycle", True)],
    ]
    
    for frame_num, detections in enumerate(test_detections):
        print(f"\nFrame {frame_num + 1}:")
        tracked_objects = tracker.update(detections)
        
        for obj in tracked_objects:
            print(f"  Track {obj.id}: {obj.class_name} at {obj.bbox} "
                  f"(age: {obj.age}, hits: {obj.hit_streak})")
        
        stats = tracker.get_statistics()
        print(f"  Stats: {stats}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_object_tracker()