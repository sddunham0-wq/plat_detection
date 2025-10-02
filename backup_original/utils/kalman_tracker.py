"""
Kalman Filter implementation untuk smooth object tracking
Mengurangi noise dan jitter pada bounding box tracking
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class KalmanState:
    """State untuk Kalman filter tracking"""
    x: float  # Center X
    y: float  # Center Y
    w: float  # Width
    h: float  # Height
    dx: float  # Velocity X
    dy: float  # Velocity Y
    dw: float  # Width change rate
    dh: float  # Height change rate

class KalmanBoxTracker:
    """
    Kalman filter untuk tracking single bounding box dengan smooth movement
    
    State vector: [x, y, w, h, dx, dy, dw, dh]
    - (x, y): center coordinate
    - (w, h): width dan height
    - (dx, dy): velocity
    - (dw, dh): size change rate
    """
    
    def __init__(self, bbox: Tuple[int, int, int, int], track_id: int):
        """
        Initialize Kalman filter untuk bounding box
        
        Args:
            bbox: Initial bounding box (x, y, width, height)
            track_id: Unique identifier untuk tracker ini
        """
        self.track_id = track_id
        self.age = 0
        self.hit_streak = 1  # Number of consecutive detections
        self.time_since_update = 0
        
        # Convert bbox to center format
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Initialize Kalman filter
        self.kf = cv2.KalmanFilter(8, 4)  # 8 state vars, 4 measurement vars
        
        # State transition model (constant velocity model)
        # [x, y, w, h, dx, dy, dw, dh]
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1]   # dh = dh
        ], dtype=np.float32)
        
        # Measurement model (kita observe x, y, w, h)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # observe x
            [0, 1, 0, 0, 0, 0, 0, 0],  # observe y
            [0, 0, 1, 0, 0, 0, 0, 0],  # observe w
            [0, 0, 0, 1, 0, 0, 0, 0]   # observe h
        ], dtype=np.float32)
        
        # Process noise covariance (Q) - how much we trust our motion model
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        # Increase noise for velocity and size change
        self.kf.processNoiseCov[4:6, 4:6] *= 0.01  # velocity noise
        self.kf.processNoiseCov[6:8, 6:8] *= 0.001  # size change noise
        
        # Measurement noise covariance (R) - measurement uncertainty
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 10.0
        
        # Error covariance (P) - initial uncertainty
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 100.0
        self.kf.errorCovPost[4:8, 4:8] *= 1000.0  # Higher uncertainty for velocities
        
        # Initialize state
        self.kf.statePre = np.array([center_x, center_y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()
        
        self.history = []
        
    def predict(self) -> Tuple[int, int, int, int]:
        """
        Predict next position menggunakan Kalman filter
        
        Returns:
            Tuple[int, int, int, int]: Predicted bounding box (x, y, w, h)
        """
        # Predict next state
        predicted_state = self.kf.predict()
        
        # Convert back to bounding box format
        center_x, center_y, w, h = predicted_state[:4]
        
        # Ensure positive width and height
        w = max(w, 10)
        h = max(h, 10)
        
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        w = int(w)
        h = int(h)
        
        # Store prediction
        self.history.append([x, y, w, h])
        if len(self.history) > 10:  # Keep only last 10 predictions
            self.history.pop(0)
        
        return (x, y, w, h)
    
    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Update Kalman filter dengan measurement baru
        
        Args:
            bbox: Measured bounding box (x, y, width, height)
            
        Returns:
            Tuple[int, int, int, int]: Updated bounding box (x, y, w, h)
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
        
        # Convert bbox to center format for measurement
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Update with measurement
        measurement = np.array([center_x, center_y, w, h], dtype=np.float32)
        self.kf.correct(measurement)
        
        # Get corrected state
        corrected_state = self.kf.statePost
        
        # Convert back to bounding box format
        corr_center_x, corr_center_y, corr_w, corr_h = corrected_state[:4]
        
        # Ensure positive dimensions
        corr_w = max(corr_w, 10)
        corr_h = max(corr_h, 10)
        
        corr_x = int(corr_center_x - corr_w / 2)
        corr_y = int(corr_center_y - corr_h / 2)
        corr_w = int(corr_w)
        corr_h = int(corr_h)
        
        return (corr_x, corr_y, corr_w, corr_h)
    
    def get_current_bbox(self) -> Tuple[int, int, int, int]:
        """
        Get current bounding box dari state
        
        Returns:
            Tuple[int, int, int, int]: Current bounding box
        """
        state = self.kf.statePost
        center_x, center_y, w, h = state[:4]
        
        # Ensure positive dimensions
        w = max(w, 10)
        h = max(h, 10)
        
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        w = int(w)
        h = int(h)
        
        return (x, y, w, h)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity (dx, dy)"""
        state = self.kf.statePost
        return (float(state[4]), float(state[5]))
    
    def mark_missed(self):
        """Mark bahwa detection missed di frame ini"""
        self.time_since_update += 1
        self.hit_streak = 0
        self.age += 1


class AdaptiveKalmanTracker(KalmanBoxTracker):
    """
    Enhanced Kalman tracker dengan adaptive noise berdasarkan tracking quality
    """
    
    def __init__(self, bbox: Tuple[int, int, int, int], track_id: int):
        super().__init__(bbox, track_id)
        
        # Adaptive parameters
        self.measurement_errors = []
        self.max_error_history = 10
        self.base_process_noise = 0.1
        self.base_measurement_noise = 10.0
        
    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Update dengan adaptive noise adjustment"""
        
        # Predict sebelum update untuk calculate error
        predicted_bbox = self.predict()
        
        # Calculate prediction error
        pred_center = (predicted_bbox[0] + predicted_bbox[2]/2, predicted_bbox[1] + predicted_bbox[3]/2)
        meas_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
        error = np.sqrt((pred_center[0] - meas_center[0])**2 + (pred_center[1] - meas_center[1])**2)
        
        # Store error untuk adaptive adjustment
        self.measurement_errors.append(error)
        if len(self.measurement_errors) > self.max_error_history:
            self.measurement_errors.pop(0)
        
        # Adjust noise berdasarkan recent errors
        if len(self.measurement_errors) >= 3:
            avg_error = np.mean(self.measurement_errors)
            
            # Jika error tinggi, increase measurement noise (trust measurements less)
            if avg_error > 30:
                noise_factor = min(avg_error / 20, 5.0)  # Cap at 5x
                self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * (self.base_measurement_noise * noise_factor)
            else:
                # Low error, trust measurements more
                self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * self.base_measurement_noise
        
        # Call parent update
        return super().update(bbox)


class MultiObjectKalmanTracker:
    """
    Manager untuk multiple Kalman trackers
    """
    
    def __init__(self, use_adaptive: bool = True, max_age: int = 30, min_hits: int = 3):
        """
        Initialize multi-object tracker
        
        Args:
            use_adaptive: Use adaptive Kalman tracker
            max_age: Maximum frames to keep tracker without updates
            min_hits: Minimum hits before considering tracker valid
        """
        self.trackers: dict = {}
        self.next_id = 1
        self.use_adaptive = use_adaptive
        self.max_age = max_age
        self.min_hits = min_hits
        
        # Statistics
        self.total_trackers_created = 0
        
    def update(self, detections: list) -> dict:
        """
        Update semua trackers dengan detections baru
        
        Args:
            detections: List of bounding boxes [(x, y, w, h), ...]
            
        Returns:
            dict: {track_id: (x, y, w, h)} untuk active trackers
        """
        # Predict all existing trackers
        predicted_bboxes = {}
        for track_id, tracker in self.trackers.items():
            predicted_bboxes[track_id] = tracker.predict()
        
        # Simple assignment berdasarkan distance (bisa diganti dengan Hungarian)
        assignments = self._assign_detections_to_trackers(detections, predicted_bboxes)
        
        # Update assigned trackers
        updated_trackers = set()
        for detection_idx, track_id in assignments.items():
            if track_id in self.trackers:
                self.trackers[track_id].update(detections[detection_idx])
                updated_trackers.add(track_id)
        
        # Mark missed trackers
        for track_id in self.trackers:
            if track_id not in updated_trackers:
                self.trackers[track_id].mark_missed()
        
        # Create new trackers untuk unassigned detections
        assigned_detections = set(assignments.keys())
        for i, detection in enumerate(detections):
            if i not in assigned_detections:
                self._create_new_tracker(detection)
        
        # Remove old trackers
        self._remove_old_trackers()
        
        # Return active trackers
        active_trackers = {}
        for track_id, tracker in self.trackers.items():
            if (tracker.hit_streak >= self.min_hits or 
                tracker.time_since_update <= 1):
                active_trackers[track_id] = tracker.get_current_bbox()
        
        return active_trackers
    
    def _assign_detections_to_trackers(self, detections: list, predicted_bboxes: dict) -> dict:
        """Simple assignment berdasarkan minimum distance"""
        assignments = {}
        used_trackers = set()
        
        for det_idx, detection in enumerate(detections):
            best_tracker = None
            best_distance = float('inf')
            
            det_center = (detection[0] + detection[2]/2, detection[1] + detection[3]/2)
            
            for track_id, pred_bbox in predicted_bboxes.items():
                if track_id in used_trackers:
                    continue
                
                pred_center = (pred_bbox[0] + pred_bbox[2]/2, pred_bbox[1] + pred_bbox[3]/2)
                distance = np.sqrt((det_center[0] - pred_center[0])**2 + (det_center[1] - pred_center[1])**2)
                
                if distance < best_distance and distance < 100:  # Max 100 pixel distance
                    best_distance = distance
                    best_tracker = track_id
            
            if best_tracker:
                assignments[det_idx] = best_tracker
                used_trackers.add(best_tracker)
        
        return assignments
    
    def _create_new_tracker(self, bbox):
        """Create new Kalman tracker"""
        if self.use_adaptive:
            tracker = AdaptiveKalmanTracker(bbox, self.next_id)
        else:
            tracker = KalmanBoxTracker(bbox, self.next_id)
        
        self.trackers[self.next_id] = tracker
        self.total_trackers_created += 1
        self.next_id += 1
    
    def _remove_old_trackers(self):
        """Remove trackers yang sudah terlalu lama tidak terupdate"""
        to_remove = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.trackers[track_id]
    
    def get_statistics(self) -> dict:
        """Get tracking statistics"""
        active_trackers = [t for t in self.trackers.values() 
                          if t.hit_streak >= self.min_hits or t.time_since_update <= 1]
        
        return {
            'total_trackers': len(self.trackers),
            'active_trackers': len(active_trackers),
            'total_created': self.total_trackers_created,
            'avg_age': np.mean([t.age for t in active_trackers]) if active_trackers else 0
        }


def test_kalman_tracker():
    """Test function untuk Kalman tracker"""
    print("Testing Kalman Tracker...")
    
    # Create tracker
    tracker = MultiObjectKalmanTracker(use_adaptive=True)
    
    # Simulate moving objects
    test_sequences = [
        # Frame 1: Two objects
        [(100, 100, 50, 30), (200, 150, 45, 28)],
        
        # Frame 2: Objects moved
        [(105, 102, 50, 30), (205, 148, 45, 28)],
        
        # Frame 3: One object missing (occlusion)
        [(110, 104, 50, 30)],
        
        # Frame 4: Both objects back
        [(115, 106, 50, 30), (215, 144, 45, 28)],
        
        # Frame 5: Noisy measurements
        [(118, 109, 52, 32), (218, 141, 43, 26)]
    ]
    
    for frame_num, detections in enumerate(test_sequences):
        print(f"\nFrame {frame_num + 1}:")
        print(f"Input detections: {detections}")
        
        tracked_objects = tracker.update(detections)
        
        print(f"Tracked objects:")
        for track_id, bbox in tracked_objects.items():
            print(f"  Track {track_id}: {bbox}")
        
        stats = tracker.get_statistics()
        print(f"Stats: {stats}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_kalman_tracker()