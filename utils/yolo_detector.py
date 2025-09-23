"""
YOLOv8 Object Detector untuk Live Stream
Integrasi YOLOv8 dengan sistem deteksi plat nomor
"""

import cv2
import numpy as np
import logging
import time
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Check YOLOv8 availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

@dataclass
class ObjectDetection:
    """Data class untuk hasil deteksi objek"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    class_id: int
    is_vehicle: bool = False

class YOLOObjectDetector:
    """
    YOLOv8 Object Detector untuk live streaming
    """
    
    def __init__(self, model_path='yolov8n.pt', confidence=0.4, iou_threshold=0.5, max_detections=15):
        """
        Initialize YOLOv8 detector dengan optimized parameters untuk CCTV

        Args:
            model_path: Path to YOLO model
            confidence: Confidence threshold (default: 0.4 - optimized untuk CCTV)
            iou_threshold: IoU threshold for NMS (default: 0.5 - better duplicate removal)
            max_detections: Maximum detections per frame (default: 15 - optimized untuk speed)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.model = None
        self.enabled = False
        self.logger = logging.getLogger(__name__)
        
        # Vehicle classes in COCO dataset dengan optimized confidence thresholds
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

        # Class-specific confidence thresholds untuk Indonesian vehicles
        self.class_confidence_thresholds = {
            2: 0.4,   # car - standard threshold
            3: 0.35,  # motorcycle - lower threshold (lebih banyak motor di Indonesia)
            5: 0.5,   # bus - higher threshold (less common, need higher confidence)
            7: 0.45   # truck - standard threshold
        }
        
        # COCO class names
        self.coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
            44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
            49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
            54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
        
        # Statistics
        self.total_detections = 0
        self.total_vehicles = 0
        self.detection_times = []

        # Crowded Scene Handling
        self.crowded_scene_mode = False
        self.density_threshold = 0.15  # Motorcycle density threshold for crowded detection
        self.crowded_confidence = 0.25  # Lower confidence for crowded scenes
        self.crowded_iou_threshold = 0.3  # Lower IoU for crowded scenes
        self.last_density_check = 0
        self.density_check_interval = 1.0  # Check density every second

        # Sequential Detection System
        self.sequential_mode = False
        self.grid_zones = (3, 3)  # 3x3 grid default
        self.current_zone = 0
        self.zone_cycle_time = 2.0  # seconds per zone
        self.last_zone_switch = time.time()
        self.frame_width = 1280
        self.frame_height = 720
        
        # Initialize model if possible
        self.initialize()
    
    def initialize(self):
        """Initialize YOLOv8 model dengan error handling"""
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLOv8 (ultralytics) not available. Object detection disabled.")
            self.logger.info("To enable: pip install ultralytics")
            return False
        
        try:
            self.logger.info(f"Loading YOLOv8 model: {self.model_path}")
            
            # Load model - auto-download jika belum ada
            self.model = YOLO(self.model_path)
            self.enabled = True
            
            self.logger.info("✅ YOLOv8 model loaded successfully")
            self.logger.info(f"📊 Model classes: {len(self.coco_classes)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {str(e)}")
            self.logger.warning("Object detection will be disabled")
            self.enabled = False
            return False
    
    def is_enabled(self) -> bool:
        """Check if YOLOv8 detection is enabled"""
        return self.enabled and self.model is not None
    
    def detect_objects(self, frame: np.ndarray, vehicles_only: bool = False) -> List[ObjectDetection]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame
            vehicles_only: Only return vehicle detections
            
        Returns:
            List of ObjectDetection objects
        """
        if not self.is_enabled():
            return []
        
        start_time = time.time()
        
        try:
            # Check for crowded scene periodically
            current_time = time.time()
            if current_time - self.last_density_check > self.density_check_interval:
                self._detect_crowded_scene(frame)
                self.last_density_check = current_time

            # Adaptive parameters based on scene density
            conf_threshold = self.crowded_confidence if self.crowded_scene_mode else self.confidence
            iou_threshold = self.crowded_iou_threshold if self.crowded_scene_mode else self.iou_threshold
            max_detections = min(25, self.max_detections * 2) if self.crowded_scene_mode else self.max_detections

            # Run YOLOv8 detection with adaptive parameters
            results = self.model(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Get class name
                        class_name = self.coco_classes.get(class_id, f'class_{class_id}')

                        # Check if vehicle
                        is_vehicle = class_id in self.vehicle_classes

                        # Apply class-specific confidence filtering
                        if is_vehicle:
                            min_confidence = self.class_confidence_thresholds.get(class_id, self.confidence)
                            if confidence < min_confidence:
                                continue

                        # Filter by vehicles if requested
                        if vehicles_only and not is_vehicle:
                            continue

                        # Enhanced bounding box processing untuk vehicles
                        bbox = self._enhance_vehicle_bbox(x1, y1, x2, y2, class_id, frame) if is_vehicle else (x1, y1, x2 - x1, y2 - y1)

                        # Create detection object
                        detection = ObjectDetection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            class_id=class_id,
                            is_vehicle=is_vehicle
                        )

                        detections.append(detection)
            
            # Update statistics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_detections += len(detections)
            self.total_vehicles += sum(1 for d in detections if d.is_vehicle)
            
            # Keep only last 100 detection times for moving average
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]
            
            # Apply additional post-processing filtering
            filtered_detections = self._post_process_detections(detections)

            # Enhanced post-processing untuk crowded scenes
            if self.crowded_scene_mode and vehicles_only:
                filtered_detections = self._handle_crowded_motorcycles(filtered_detections, frame)

            return filtered_detections
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {str(e)}")
            return []
    
    def detect_objects_sequential(self, frame: np.ndarray, vehicles_only: bool = False) -> List[ObjectDetection]:
        """
        Sequential zone-based object detection
        
        Args:
            frame: Input frame
            vehicles_only: Only return vehicle detections
            
        Returns:
            List of ObjectDetection objects from current active zone
        """
        if not self.is_enabled() or not self.sequential_mode:
            return self.detect_objects(frame, vehicles_only)
        
        # Update frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Check if it's time to switch zones
        current_time = time.time()
        if current_time - self.last_zone_switch >= self.zone_cycle_time:
            self._switch_to_next_zone()
            self.last_zone_switch = current_time
        
        # Get current zone region
        zone_frame = self._get_zone_frame(frame, self.current_zone)
        zone_offset = self._get_zone_offset(self.current_zone)
        
        if zone_frame is None:
            return []
        
        # Run detection on zone
        zone_detections = self.detect_objects(zone_frame, vehicles_only)
        
        # Adjust bounding boxes to full frame coordinates
        adjusted_detections = []
        for detection in zone_detections:
            x, y, w, h = detection.bbox
            # Adjust coordinates back to full frame
            adjusted_bbox = (
                x + zone_offset[0],
                y + zone_offset[1], 
                w, h
            )
            
            adjusted_detection = ObjectDetection(
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=adjusted_bbox,
                class_id=detection.class_id,
                is_vehicle=detection.is_vehicle
            )
            adjusted_detections.append(adjusted_detection)
        
        return adjusted_detections
    
    def _switch_to_next_zone(self):
        """Switch to next zone in cycle"""
        total_zones = self.grid_zones[0] * self.grid_zones[1]
        self.current_zone = (self.current_zone + 1) % total_zones
        self.logger.debug(f"Switched to zone {self.current_zone}")
    
    def _get_zone_frame(self, frame: np.ndarray, zone_index: int) -> Optional[np.ndarray]:
        """
        Extract frame region for specific zone
        
        Args:
            frame: Full frame
            zone_index: Zone index (0-based)
            
        Returns:
            Zone frame or None if invalid
        """
        rows, cols = self.grid_zones
        total_zones = rows * cols
        
        if zone_index >= total_zones:
            return None
        
        # Calculate zone position in grid
        zone_row = zone_index // cols
        zone_col = zone_index % cols
        
        # Calculate zone boundaries
        zone_width = self.frame_width // cols
        zone_height = self.frame_height // rows
        
        x1 = zone_col * zone_width
        y1 = zone_row * zone_height
        x2 = min(x1 + zone_width, self.frame_width)
        y2 = min(y1 + zone_height, self.frame_height)
        
        # Extract zone
        zone_frame = frame[y1:y2, x1:x2]
        
        return zone_frame if zone_frame.size > 0 else None
    
    def _get_zone_offset(self, zone_index: int) -> Tuple[int, int]:
        """
        Get offset coordinates for zone
        
        Args:
            zone_index: Zone index
            
        Returns:
            (x_offset, y_offset) tuple
        """
        rows, cols = self.grid_zones
        
        zone_row = zone_index // cols
        zone_col = zone_index % cols
        
        zone_width = self.frame_width // cols
        zone_height = self.frame_height // rows
        
        x_offset = zone_col * zone_width
        y_offset = zone_row * zone_height
        
        return (x_offset, y_offset)
    
    def enable_sequential_mode(self, grid_zones: Tuple[int, int] = (3, 3), cycle_time: float = 2.0):
        """
        Enable sequential detection mode
        
        Args:
            grid_zones: Grid dimensions (rows, cols)
            cycle_time: Time per zone in seconds
        """
        self.sequential_mode = True
        self.grid_zones = grid_zones
        self.zone_cycle_time = cycle_time
        self.current_zone = 0
        self.last_zone_switch = time.time()
        
        total_zones = grid_zones[0] * grid_zones[1]
        self.logger.info(f"Sequential mode enabled: {grid_zones[0]}x{grid_zones[1]} grid, {total_zones} zones, {cycle_time}s per zone")
    
    def disable_sequential_mode(self):
        """Disable sequential detection mode"""
        self.sequential_mode = False
        self.logger.info("Sequential mode disabled")
    
    def get_current_zone_info(self) -> Dict:
        """Get information about current active zone"""
        if not self.sequential_mode:
            return {'sequential_mode': False}
        
        rows, cols = self.grid_zones
        zone_row = self.current_zone // cols
        zone_col = self.current_zone % cols
        total_zones = rows * cols
        
        time_since_switch = time.time() - self.last_zone_switch
        time_remaining = max(0, self.zone_cycle_time - time_since_switch)
        
        return {
            'sequential_mode': True,
            'grid_zones': self.grid_zones,
            'current_zone': self.current_zone,
            'zone_row': zone_row,
            'zone_col': zone_col,
            'total_zones': total_zones,
            'cycle_time': self.zone_cycle_time,
            'time_remaining': time_remaining,
            'progress': time_since_switch / self.zone_cycle_time
        }
    
    def _post_process_detections(self, detections: List[ObjectDetection]) -> List[ObjectDetection]:
        """
        Additional post-processing to remove duplicate detections
        
        Args:
            detections: List of raw detections
            
        Returns:
            List of filtered detections
        """
        if not detections:
            return detections
        
        # Filter by minimum area (remove very small detections)
        min_area = 1000  # Minimum bounding box area in pixels
        filtered = []
        
        for detection in detections:
            x, y, w, h = detection.bbox
            area = w * h
            
            if area >= min_area:
                filtered.append(detection)
        
        # Additional distance-based filtering for very close detections
        final_detections = []
        
        for i, detection in enumerate(filtered):
            is_duplicate = False
            x1, y1, w1, h1 = detection.bbox
            center1 = (x1 + w1/2, y1 + h1/2)
            
            for j, existing in enumerate(final_detections):
                x2, y2, w2, h2 = existing.bbox
                center2 = (x2 + w2/2, y2 + h2/2)
                
                # Calculate distance between centers
                distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                
                # If centers are very close and same class, consider as duplicate
                if (distance < 50 and 
                    detection.class_id == existing.class_id and
                    detection.confidence <= existing.confidence):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        # Log filtering results
        if len(detections) != len(final_detections):
            self.logger.debug(
                f"Filtered detections: {len(detections)} → {len(final_detections)} "
                f"(removed {len(detections) - len(final_detections)} duplicates)"
            )
        
        return final_detections

    def _enhance_vehicle_bbox(self, x1: int, y1: int, x2: int, y2: int, class_id: int, frame: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Enhance vehicle bounding box untuk better fit dengan vehicle shape
        """
        try:
            # Get original dimensions
            orig_w = x2 - x1
            orig_h = y2 - y1
            orig_area = orig_w * orig_h

            # Ensure reasonable size
            if orig_area < 500:  # Too small
                return (x1, y1, orig_w, orig_h)

            # Extract vehicle region
            vehicle_roi = frame[y1:y2, x1:x2]

            if vehicle_roi.size == 0:
                return (x1, y1, orig_w, orig_h)

            # Vehicle type specific enhancement
            if class_id == 3:  # Motorcycle
                enhanced_bbox = self._enhance_motorcycle_bbox(vehicle_roi, x1, y1, orig_w, orig_h)
            elif class_id in [2, 5, 7]:  # Car, Bus, Truck
                enhanced_bbox = self._enhance_car_bbox(vehicle_roi, x1, y1, orig_w, orig_h)
            else:
                enhanced_bbox = (x1, y1, orig_w, orig_h)

            # Validate enhanced bbox
            new_x, new_y, new_w, new_h = enhanced_bbox
            new_area = new_w * new_h

            # Don't allow bbox to grow too much or shrink too much
            area_ratio = new_area / orig_area if orig_area > 0 else 1
            if 0.7 <= area_ratio <= 1.3:  # 30% change maximum
                return enhanced_bbox
            else:
                return (x1, y1, orig_w, orig_h)

        except Exception:
            return (x1, y1, x2 - x1, y2 - y1)

    def _enhance_motorcycle_bbox(self, roi: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Enhance bounding box specifically untuk motorcycles
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Edge detection untuk find motorcycle contours
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return (x, y, w, h)

            # Find largest contour (likely the motorcycle)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rect dari contour
            cont_x, cont_y, cont_w, cont_h = cv2.boundingRect(largest_contour)

            # Add small margin untuk motorcycles
            margin = 5
            tight_x = max(0, x + cont_x - margin)
            tight_y = max(0, y + cont_y - margin)
            tight_w = min(w, cont_w + 2 * margin)
            tight_h = min(h, cont_h + 2 * margin)

            return (tight_x, tight_y, tight_w, tight_h)

        except Exception:
            return (x, y, w, h)

    def _enhance_car_bbox(self, roi: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """
        Enhance bounding box specifically untuk cars/trucks/buses
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Use adaptive threshold untuk better vehicle outline
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Morphological operations untuk clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return (x, y, w, h)

            # Combine all significant contours
            significant_contours = [c for c in contours if cv2.contourArea(c) > 100]

            if not significant_contours:
                return (x, y, w, h)

            # Get overall bounding rect
            all_points = np.vstack([c.reshape(-1, 2) for c in significant_contours])
            overall_x, overall_y, overall_w, overall_h = cv2.boundingRect(all_points)

            # Add reasonable margin untuk cars
            margin = 8
            tight_x = max(0, x + overall_x - margin)
            tight_y = max(0, y + overall_y - margin)
            tight_w = min(w, overall_w + 2 * margin)
            tight_h = min(h, overall_h + 2 * margin)

            return (tight_x, tight_y, tight_w, tight_h)

        except Exception:
            return (x, y, w, h)
    
    def draw_zone_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw zone grid overlay on frame for sequential mode
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with zone overlay
        """
        if not self.sequential_mode:
            return frame
        
        overlay_frame = frame.copy()
        self.frame_height, self.frame_width = frame.shape[:2]
        
        rows, cols = self.grid_zones
        zone_width = self.frame_width // cols
        zone_height = self.frame_height // rows
        
        # Draw grid lines
        for i in range(1, cols):
            x = i * zone_width
            cv2.line(overlay_frame, (x, 0), (x, self.frame_height), (255, 255, 255), 1)
        
        for i in range(1, rows):
            y = i * zone_height
            cv2.line(overlay_frame, (0, y), (self.frame_width, y), (255, 255, 255), 1)
        
        # Highlight current active zone
        zone_row = self.current_zone // cols
        zone_col = self.current_zone % cols
        
        x1 = zone_col * zone_width
        y1 = zone_row * zone_height
        x2 = min(x1 + zone_width, self.frame_width)
        y2 = min(y1 + zone_height, self.frame_height)
        
        # Draw active zone highlight
        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        # Add zone info text
        zone_info = self.get_current_zone_info()
        if zone_info['sequential_mode']:
            text = f"Zone {self.current_zone + 1}/{zone_info['total_zones']} ({zone_info['time_remaining']:.1f}s)"
            cv2.putText(overlay_frame, text, (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return overlay_frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[ObjectDetection], 
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw object detections on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Show confidence scores
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Choose color based on object type
            if detection.is_vehicle:
                color = (0, 255, 0)  # Green for vehicles
            elif detection.class_name == 'person':
                color = (255, 0, 0)  # Blue for person
            else:
                color = (0, 165, 255)  # Orange for others
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_confidence:
                label = f'{detection.class_name}: {detection.confidence:.2f}'
            else:
                label = detection.class_name
                
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background rectangle for label
            cv2.rectangle(annotated_frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Text
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_vehicle_regions(self, detections: List[ObjectDetection]) -> List[Tuple[int, int, int, int]]:
        """
        Get vehicle bounding boxes untuk plate detection
        
        Args:
            detections: List of object detections
            
        Returns:
            List of vehicle bounding boxes (x, y, w, h)
        """
        vehicle_regions = []
        
        for detection in detections:
            if detection.is_vehicle:
                vehicle_regions.append(detection.bbox)
        
        return vehicle_regions
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        
        return {
            'enabled': self.enabled,
            'total_detections': self.total_detections,
            'total_vehicles': self.total_vehicles,
            'avg_detection_time': avg_detection_time,
            'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence,
            'iou_threshold': self.iou_threshold,
            'max_detections': self.max_detections
        }
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.total_detections = 0
        self.total_vehicles = 0
        self.detection_times = []
    
    def set_confidence(self, confidence: float):
        """Set confidence threshold"""
        self.confidence = max(0.0, min(1.0, confidence))
        self.logger.info(f"Confidence threshold set to: {self.confidence}")
    
    def set_iou_threshold(self, iou_threshold: float):
        """Set IoU threshold for NMS"""
        self.iou_threshold = max(0.0, min(1.0, iou_threshold))
        self.logger.info(f"IoU threshold set to: {self.iou_threshold}")
    
    def set_max_detections(self, max_detections: int):
        """Set maximum detections per frame"""
        self.max_detections = max(1, min(1000, max_detections))
        self.logger.info(f"Max detections set to: {self.max_detections}")
    
    def enable(self):
        """Enable object detection"""
        if YOLO_AVAILABLE and self.model is not None:
            self.enabled = True
            self.logger.info("Object detection enabled")
        else:
            self.logger.warning("Cannot enable object detection - model not available")
    
    def disable(self):
        """Disable object detection"""
        self.enabled = False
        self.logger.info("Object detection disabled")

    def _detect_crowded_scene(self, frame: np.ndarray):
        """
        Detect if scene contains crowded motorcycles (parkiran numpuk) with multi-scale analysis
        """
        try:
            self.frame_height, self.frame_width = frame.shape[:2]

            # Multi-scale detection untuk better crowding assessment
            scales = [1.0, 0.75, 0.5] if self.frame_width > 1280 else [1.0, 0.75]
            motorcycle_counts = []
            density_values = []

            for scale in scales:
                # Resize frame untuk multi-scale detection
                if scale != 1.0:
                    new_width = int(frame.shape[1] * scale)
                    new_height = int(frame.shape[0] * scale)
                    scaled_frame = cv2.resize(frame, (new_width, new_height))
                else:
                    scaled_frame = frame

                # Adaptive confidence untuk different scales - smaller scales need higher confidence
                conf_threshold = self.confidence * (1.2 + 0.3 * (1 - scale))
                quick_results = self.model(
                    scaled_frame,
                    conf=conf_threshold,
                    iou=0.6,  # Higher IoU untuk avoid false crowding
                    max_det=int(30 / scale),  # More detections for smaller scales
                    verbose=False
                )

                motorcycle_count = 0
                total_motorcycle_area = 0
                frame_area = scaled_frame.shape[0] * scaled_frame.shape[1]

                for result in quick_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            class_id = int(box.cls[0])
                            if class_id == 3:  # Motorcycle class
                                motorcycle_count += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                motorcycle_area = (x2 - x1) * (y2 - y1)
                                total_motorcycle_area += motorcycle_area

                # Calculate density untuk scale ini
                density = total_motorcycle_area / frame_area if frame_area > 0 else 0
                motorcycle_counts.append(motorcycle_count)
                density_values.append(density)

            # Aggregate results dari multi-scale detection
            max_motorcycle_count = max(motorcycle_counts) if motorcycle_counts else 0
            avg_density = sum(density_values) / len(density_values) if density_values else 0
            max_density = max(density_values) if density_values else 0

            # Enhanced crowding detection dengan multi-scale metrics
            was_crowded = self.crowded_scene_mode

            # Scale variability indicates crowding - different scales detect different amounts
            scale_variability = max(motorcycle_counts) - min(motorcycle_counts) if len(motorcycle_counts) > 1 else 0

            self.crowded_scene_mode = (
                avg_density > self.density_threshold or  # High average density
                max_density > self.density_threshold * 1.5 or  # Very high peak density
                max_motorcycle_count >= 5 or  # Many motorcycles at any scale
                scale_variability >= 3  # High scale variability indicates overlapping
            )

            # Log mode changes dengan detailed metrics
            if was_crowded != self.crowded_scene_mode:
                mode = "CROWDED" if self.crowded_scene_mode else "NORMAL"
                self.logger.info(f"🏍️ Scene mode: {mode} (avg_density: {avg_density:.3f}, max_count: {max_motorcycle_count}, variability: {scale_variability})")

        except Exception as e:
            self.logger.debug(f"Error in crowded scene detection: {e}")

    def _handle_crowded_motorcycles(self, detections: List[ObjectDetection], frame: np.ndarray) -> List[ObjectDetection]:
        """
        Enhanced handling untuk overlapping motorcycles dalam crowded scenes
        """
        if not detections:
            return detections

        try:
            # Separate motorcycles from other vehicles
            motorcycles = [d for d in detections if d.class_name == 'motorcycle']
            others = [d for d in detections if d.class_name != 'motorcycle']

            if len(motorcycles) < 2:
                return detections

            # Group overlapping motorcycles
            motorcycle_groups = self._group_overlapping_motorcycles(motorcycles)

            # Process each group
            processed_motorcycles = []
            for group in motorcycle_groups:
                if len(group) == 1:
                    # Single motorcycle, no processing needed
                    processed_motorcycles.extend(group)
                else:
                    # Multiple overlapping motorcycles
                    enhanced_group = self._split_overlapping_motorcycles(group, frame)
                    processed_motorcycles.extend(enhanced_group)

            return processed_motorcycles + others

        except Exception as e:
            self.logger.debug(f"Error in crowded motorcycle handling: {e}")
            return detections

    def _group_overlapping_motorcycles(self, motorcycles: List[ObjectDetection]) -> List[List[ObjectDetection]]:
        """
        Enhanced grouping of overlapping motorcycles dengan spatial awareness
        """
        groups = []
        processed = set()

        # Sort motorcycles by confidence untuk better grouping
        sorted_motorcycles = sorted(enumerate(motorcycles), key=lambda x: x[1].confidence, reverse=True)

        for orig_idx, motorcycle in sorted_motorcycles:
            if orig_idx in processed:
                continue

            group = [(orig_idx, motorcycle)]
            processed.add(orig_idx)

            # Find all overlapping motorcycles using hierarchical clustering
            for other_orig_idx, other_motorcycle in sorted_motorcycles:
                if other_orig_idx in processed:
                    continue

                # Check overlap with any motorcycle in current group
                has_overlap = False
                for group_idx, group_motorcycle in group:
                    overlap = self._calculate_bbox_overlap(group_motorcycle.bbox, other_motorcycle.bbox)
                    spatial_distance = self._calculate_spatial_distance(group_motorcycle.bbox, other_motorcycle.bbox)

                    # Multi-criteria overlap detection
                    if (overlap > 0.15 or  # IoU overlap
                        (overlap > 0.05 and spatial_distance < 50) or  # Close proximity dengan slight overlap
                        self._are_motorcycles_clustered(group_motorcycle.bbox, other_motorcycle.bbox)):  # Spatial clustering
                        has_overlap = True
                        break

                if has_overlap:
                    group.append((other_orig_idx, other_motorcycle))
                    processed.add(other_orig_idx)

            # Convert back to just ObjectDetection objects
            detection_group = [motorcycle for _, motorcycle in group]
            groups.append(detection_group)

        return groups

    def _split_overlapping_motorcycles(self, group: List[ObjectDetection], frame: np.ndarray) -> List[ObjectDetection]:
        """
        Try to split overlapping motorcycles into individual detections
        """
        if len(group) <= 1:
            return group

        # Find the combined bounding box
        min_x = min(m.bbox[0] for m in group)
        min_y = min(m.bbox[1] for m in group)
        max_x = max(m.bbox[0] + m.bbox[2] for m in group)
        max_y = max(m.bbox[1] + m.bbox[3] for m in group)

        # Extract combined region
        combined_roi = frame[min_y:max_y, min_x:max_x]

        if combined_roi.size == 0:
            return group

        try:
            # Try to find individual motorcycle contours
            split_motorcycles = self._find_individual_motorcycles(combined_roi, min_x, min_y, group)

            if len(split_motorcycles) >= len(group):
                # Successfully split, use split results
                return split_motorcycles[:len(group)]  # Limit to original count
            else:
                # Splitting failed, return original with adjusted confidence
                adjusted_group = []
                for motorcycle in group:
                    adjusted_motorcycle = ObjectDetection(
                        class_name=motorcycle.class_name,
                        confidence=motorcycle.confidence * 0.8,  # Penalty for overlapping
                        bbox=motorcycle.bbox,
                        class_id=motorcycle.class_id,
                        is_vehicle=motorcycle.is_vehicle
                    )
                    adjusted_group.append(adjusted_motorcycle)
                return adjusted_group

        except Exception:
            return group

    def _find_individual_motorcycles(self, roi: np.ndarray, offset_x: int, offset_y: int,
                                   original_group: List[ObjectDetection]) -> List[ObjectDetection]:
        """
        Find individual motorcycles dalam combined ROI
        """
        individual_motorcycles = []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # Morphological operations untuk separate objects
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter dan convert contours to motorcycles
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500:  # Too small
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Check size constraints untuk motorcycle
                if w < 30 or h < 30 or w > 300 or h > 200:
                    continue

                # Check aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if not (0.5 <= aspect_ratio <= 3.0):
                    continue

                # Convert back to frame coordinates
                frame_x = x + offset_x
                frame_y = y + offset_y

                # Estimate confidence based on area dan contour quality
                base_confidence = min([m.confidence for m in original_group])
                area_factor = min(1.0, area / 2000)  # Normalize by expected motorcycle area
                estimated_confidence = base_confidence * area_factor * 0.7  # Penalty for splitting

                individual_motorcycle = ObjectDetection(
                    class_name='motorcycle',
                    confidence=estimated_confidence,
                    bbox=(frame_x, frame_y, w, h),
                    class_id=3,
                    is_vehicle=True
                )

                individual_motorcycles.append(individual_motorcycle)

            return individual_motorcycles

        except Exception:
            return []

    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap ratio between two bounding boxes
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _calculate_spatial_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate spatial distance between centers of two bounding boxes
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2

        return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    def _are_motorcycles_clustered(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
        """
        Check if two motorcycles are part of the same cluster using spatial analysis
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate average size
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2

        # Distance between closest edges
        horizontal_distance = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
        vertical_distance = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

        # Consider clustered if within 1.5x average motorcycle size
        clustering_threshold_x = avg_width * 1.5
        clustering_threshold_y = avg_height * 1.2

        return (horizontal_distance < clustering_threshold_x and
                vertical_distance < clustering_threshold_y)

def create_yolo_detector(model='yolov8n.pt', confidence=0.7, iou_threshold=0.45, max_detections=50) -> Optional[YOLOObjectDetector]:
    """
    Factory function untuk create YOLOv8 detector
    
    Args:
        model: Model name or path
        confidence: Confidence threshold (default: 0.7)
        iou_threshold: IoU threshold for NMS (default: 0.45)
        max_detections: Maximum detections per frame (default: 50)
        
    Returns:
        YOLOObjectDetector instance atau None jika gagal
    """
    try:
        detector = YOLOObjectDetector(model, confidence, iou_threshold, max_detections)
        if detector.is_enabled():
            return detector
        else:
            return None
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create YOLO detector: {str(e)}")
        return None

# Auto-install check
def check_and_install_yolo():
    """Check dan install YOLOv8 jika diperlukan"""
    if not YOLO_AVAILABLE:
        print("⚠️  YOLOv8 (ultralytics) not installed!")
        print("📦 Installing ultralytics...")
        
        try:
            import subprocess
            import sys
            
            # Install ultralytics
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'ultralytics'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ ultralytics installed successfully!")
                print("🔄 Please restart the application to use YOLOv8")
                return True
            else:
                print(f"❌ Failed to install ultralytics: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error installing ultralytics: {str(e)}")
            return False

    return True
