"""
Tracking Manager untuk Integrasi Object Tracking dengan License Plate Detection
Mengelola korelasi antara kendaraan dan plat nomor
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .object_tracker import ObjectTracker, TrackedObject, Detection
from .plate_detector import PlateDetection
from .yolo_detector import ObjectDetection

@dataclass
class TrackedPlate:
    """Data class untuk plat nomor yang di-track"""
    id: int
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    vehicle_id: Optional[int] = None  # ID kendaraan yang terkait
    first_detected: float = 0.0
    last_seen: float = 0.0
    detection_count: int = 0
    confirmed: bool = False  # Apakah plat sudah dikonfirmasi dengan multiple detections
    
    def __post_init__(self):
        if self.first_detected == 0.0:
            self.first_detected = time.time()
        if self.last_seen == 0.0:
            self.last_seen = time.time()

class TrackingManager:
    """
    Manager utama untuk tracking objek dan korelasi dengan plat nomor
    """
    
    def __init__(self, 
                 tracking_config: Optional[Dict] = None,
                 plate_confirmation_threshold: int = 3,
                 max_plate_age: float = 10.0):
        """
        Initialize tracking manager
        
        Args:
            tracking_config: Konfigurasi untuk ObjectTracker
            plate_confirmation_threshold: Min deteksi untuk konfirmasi plat
            max_plate_age: Max umur plat sebelum dihapus (detik)
        """
        # Default tracking config
        default_config = {
            'max_disappeared': 30,
            'max_distance': 100,
            'min_hits': 3,
            'iou_threshold': 0.3
        }
        
        if tracking_config:
            default_config.update(tracking_config)
        
        # Initialize object tracker
        self.object_tracker = ObjectTracker(**default_config)
        
        # Plate tracking
        self.tracked_plates: Dict[int, TrackedPlate] = {}
        self.next_plate_id = 1
        self.plate_confirmation_threshold = plate_confirmation_threshold
        self.max_plate_age = max_plate_age
        
        # Statistics
        self.stats = {
            'total_vehicles_tracked': 0,
            'total_plates_tracked': 0,
            'successful_associations': 0,
            'confirmed_plates': 0
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("TrackingManager initialized")
    
    def process_frame(self, 
                     object_detections: List[ObjectDetection],
                     plate_detections: List[PlateDetection]) -> Tuple[List[TrackedObject], List[TrackedPlate]]:
        """
        Process frame dengan object dan plate detections
        
        Args:
            object_detections: Deteksi objek dari YOLO
            plate_detections: Deteksi plat dari plate detector
            
        Returns:
            Tuple[List[TrackedObject], List[TrackedPlate]]: Tracked objects dan plates
        """
        # Convert object detections ke format tracker
        tracker_detections = []
        for obj in object_detections:
            detection = Detection(
                bbox=obj.bbox,
                confidence=obj.confidence,
                class_name=obj.class_name,
                is_vehicle=obj.is_vehicle
            )
            tracker_detections.append(detection)
        
        # Update object tracking
        tracked_objects = self.object_tracker.update(tracker_detections)
        
        # Update plate tracking
        tracked_plates = self._update_plate_tracking(plate_detections, tracked_objects)
        
        # Update statistics
        vehicle_tracks = [t for t in tracked_objects if t.is_vehicle]
        self.stats['total_vehicles_tracked'] = len(vehicle_tracks)
        self.stats['total_plates_tracked'] = len(tracked_plates)
        self.stats['confirmed_plates'] = len([p for p in tracked_plates if p.confirmed])
        
        return tracked_objects, tracked_plates
    
    def _update_plate_tracking(self, 
                              plate_detections: List[PlateDetection],
                              tracked_objects: List[TrackedObject]) -> List[TrackedPlate]:
        """
        Update plate tracking dan associate dengan kendaraan
        
        Args:
            plate_detections: Deteksi plat baru
            tracked_objects: Objek yang sedang di-track
            
        Returns:
            List[TrackedPlate]: Tracked plates
        """
        current_time = time.time()
        
        # Match plate detections dengan existing tracked plates
        matched_plates = set()
        
        for detection in plate_detections:
            best_match_id = None
            best_match_score = float('inf')
            
            # Cari existing plate yang paling cocok
            for plate_id, tracked_plate in self.tracked_plates.items():
                # Match berdasarkan text dan proximity
                if tracked_plate.text == detection.text:
                    distance = self._calculate_bbox_distance(detection.bbox, tracked_plate.bbox)
                    if distance < 50 and distance < best_match_score:  # Max 50 pixel movement
                        best_match_score = distance
                        best_match_id = plate_id
            
            if best_match_id:
                # Update existing plate
                tracked_plate = self.tracked_plates[best_match_id]
                tracked_plate.bbox = detection.bbox
                tracked_plate.confidence = max(tracked_plate.confidence, detection.confidence)
                tracked_plate.last_seen = current_time
                tracked_plate.detection_count += 1
                
                # Confirm plate jika sudah cukup deteksi
                if tracked_plate.detection_count >= self.plate_confirmation_threshold:
                    tracked_plate.confirmed = True
                
                matched_plates.add(best_match_id)
                
                self.logger.debug(f"Updated tracked plate {best_match_id}: {detection.text}")
            else:
                # Create new tracked plate
                new_plate = TrackedPlate(
                    id=self.next_plate_id,
                    text=detection.text,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    detection_count=1
                )
                
                self.tracked_plates[self.next_plate_id] = new_plate
                matched_plates.add(self.next_plate_id)
                self.next_plate_id += 1
                
                self.logger.debug(f"Created new tracked plate {new_plate.id}: {detection.text}")
        
        # Associate plates dengan vehicles
        for plate_id in matched_plates:
            plate = self.tracked_plates[plate_id]
            if not plate.vehicle_id:  # Belum di-associate
                vehicle_id = self.object_tracker.associate_plate_with_vehicle(
                    plate.bbox, plate_id
                )
                if vehicle_id:
                    plate.vehicle_id = vehicle_id
                    self.stats['successful_associations'] += 1
                    self.logger.info(f"Associated plate {plate_id} ({plate.text}) with vehicle {vehicle_id}")
        
        # Clean up old plates
        self._cleanup_old_plates(current_time)
        
        return list(self.tracked_plates.values())
    
    def _calculate_bbox_distance(self, bbox1: Tuple[int, int, int, int], 
                                bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance antara center dua bounding box"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def _cleanup_old_plates(self, current_time: float):
        """Remove plates yang sudah terlalu lama tidak terdeteksi"""
        to_remove = []
        
        for plate_id, plate in self.tracked_plates.items():
            age = current_time - plate.last_seen
            if age > self.max_plate_age:
                to_remove.append(plate_id)
                self.logger.debug(f"Removing old plate {plate_id}: {plate.text} (age: {age:.1f}s)")
        
        for plate_id in to_remove:
            del self.tracked_plates[plate_id]
    
    def get_confirmed_plates(self) -> List[TrackedPlate]:
        """Get hanya plates yang sudah dikonfirmasi"""
        return [plate for plate in self.tracked_plates.values() if plate.confirmed]
    
    def get_vehicle_with_plate(self, vehicle_id: int) -> Optional[Tuple[TrackedObject, List[TrackedPlate]]]:
        """
        Get kendaraan beserta plat yang terkait
        
        Args:
            vehicle_id: ID kendaraan
            
        Returns:
            Optional[Tuple[TrackedObject, List[TrackedPlate]]]: Kendaraan dan plates-nya
        """
        vehicle = self.object_tracker.get_track_by_id(vehicle_id)
        if not vehicle:
            return None
        
        associated_plates = [
            plate for plate in self.tracked_plates.values() 
            if plate.vehicle_id == vehicle_id
        ]
        
        return vehicle, associated_plates
    
    def draw_tracking_results(self, 
                             frame: np.ndarray,
                             tracked_objects: List[TrackedObject],
                             tracked_plates: List[TrackedPlate],
                             show_ids: bool = True,
                             show_trails: bool = False) -> np.ndarray:
        """
        Draw hasil tracking di frame
        
        Args:
            frame: Input frame
            tracked_objects: Tracked objects
            tracked_plates: Tracked plates
            show_ids: Tampilkan tracking IDs
            show_trails: Tampilkan trails (not implemented yet)
            
        Returns:
            np.ndarray: Annotated frame
        """
        result_frame = frame.copy()
        
        # Draw tracked objects (vehicles)
        for obj in tracked_objects:
            if obj.is_vehicle and obj.is_active():
                x, y, w, h = obj.bbox
                
                # Color berdasarkan apakah ada plat yang terkait
                associated_plates = [p for p in tracked_plates if p.vehicle_id == obj.id]
                if associated_plates:
                    color = (0, 255, 0)  # Green untuk kendaraan dengan plat
                else:
                    color = (255, 100, 0)  # Orange untuk kendaraan tanpa plat
                
                # Draw bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw tracking info
                if show_ids:
                    label = f"Vehicle {obj.id}"
                    if associated_plates:
                        confirmed_plates = [p for p in associated_plates if p.confirmed]
                        if confirmed_plates:
                            label += f" [{confirmed_plates[0].text}]"
                    
                    # Background untuk text
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(result_frame, 
                                (x, y - label_size[1] - 10),
                                (x + label_size[0], y),
                                color, -1)
                    
                    cv2.putText(result_frame, label, (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw tracked plates
        for plate in tracked_plates:
            x, y, w, h = plate.bbox
            
            # Color berdasarkan status konfirmasi
            if plate.confirmed:
                color = (0, 255, 255)  # Cyan untuk confirmed
            else:
                color = (0, 165, 255)  # Orange untuk unconfirmed
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw plate info
            if show_ids:
                confidence_text = f"{plate.confidence:.1f}%"
                count_text = f"({plate.detection_count})"
                status_text = "✓" if plate.confirmed else "?"
                
                label = f"P{plate.id}: {plate.text} {confidence_text} {count_text} {status_text}"
                
                # Background untuk text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(result_frame,
                            (x, y - label_size[1] - 8),
                            (x + label_size[0], y),
                            color, -1)
                
                cv2.putText(result_frame, label, (x, y - 3),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw tracking statistics
        stats_text = [
            f"Vehicles: {self.stats['total_vehicles_tracked']}",
            f"Plates: {self.stats['total_plates_tracked']} ({self.stats['confirmed_plates']} confirmed)",
            f"Associations: {self.stats['successful_associations']}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(result_frame, text, (10, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_frame, text, (10, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return result_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        object_stats = self.object_tracker.get_statistics()
        
        confirmed_plates = [p for p in self.tracked_plates.values() if p.confirmed]
        avg_plate_confidence = np.mean([p.confidence for p in confirmed_plates]) if confirmed_plates else 0
        
        return {
            **object_stats,
            **self.stats,
            'active_plates': len(self.tracked_plates),
            'avg_plate_confidence': round(avg_plate_confidence, 2),
            'avg_detections_per_plate': np.mean([p.detection_count for p in confirmed_plates]) if confirmed_plates else 0
        }
    
    def reset(self):
        """Reset semua tracking state"""
        self.object_tracker.reset()
        self.tracked_plates.clear()
        self.next_plate_id = 1
        self.stats = {
            'total_vehicles_tracked': 0,
            'total_plates_tracked': 0,
            'successful_associations': 0,
            'confirmed_plates': 0
        }
        self.logger.info("TrackingManager state reset")


def test_tracking_manager():
    """Test function untuk tracking manager"""
    print("Testing TrackingManager...")
    
    from utils.yolo_detector import ObjectDetection
    from utils.plate_detector import PlateDetection
    
    # Create tracking manager
    manager = TrackingManager()
    
    # Simulate detections
    test_frames = [
        # Frame 1: Vehicle dengan plat
        {
            'objects': [ObjectDetection("car", 0.9, (100, 100, 80, 60), 2, True)],
            'plates': [PlateDetection("B1234ABC", 85.0, (120, 140, 40, 15), None, time.time())]
        },
        # Frame 2: Same vehicle moved, same plate
        {
            'objects': [ObjectDetection("car", 0.92, (105, 102, 80, 60), 2, True)],
            'plates': [PlateDetection("B1234ABC", 87.0, (125, 142, 40, 15), None, time.time())]
        },
        # Frame 3: Confirm plate with third detection
        {
            'objects': [ObjectDetection("car", 0.88, (110, 104, 80, 60), 2, True)],
            'plates': [PlateDetection("B1234ABC", 90.0, (130, 144, 40, 15), None, time.time())]
        }
    ]
    
    for frame_num, frame_data in enumerate(test_frames):
        print(f"\nFrame {frame_num + 1}:")
        
        tracked_objects, tracked_plates = manager.process_frame(
            frame_data['objects'], 
            frame_data['plates']
        )
        
        print(f"Tracked Objects: {len(tracked_objects)}")
        for obj in tracked_objects:
            print(f"  Vehicle {obj.id}: {obj.class_name} at {obj.bbox}")
        
        print(f"Tracked Plates: {len(tracked_plates)}")
        for plate in tracked_plates:
            status = "confirmed" if plate.confirmed else "pending"
            vehicle_info = f" -> Vehicle {plate.vehicle_id}" if plate.vehicle_id else ""
            print(f"  Plate {plate.id}: {plate.text} ({status}){vehicle_info}")
        
        stats = manager.get_statistics()
        print(f"Stats: {stats}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_tracking_manager()