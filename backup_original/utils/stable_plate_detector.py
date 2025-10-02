#!/usr/bin/env python3
"""
Stable Plate Detector
Versi yang lebih stabil dengan temporal filtering dan multi-stage validation
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque, defaultdict

# Import components
from utils.yolo_detector import YOLOObjectDetector
from utils.robust_plate_detector import RobustPlateDetector

@dataclass
class StablePlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    angle: float = 0.0
    processed_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    vehicle_type: str = "unknown"
    detection_method: str = "stable"
    stability_score: float = 0.0
    detection_count: int = 1

class StablePlateDetector:
    """
    Detector plat yang stabil dengan temporal filtering dan validasi multi-stage
    """
    
    def __init__(self, streaming_mode=True):
        """Initialize stable detector"""
        self.streaming_mode = streaming_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLO for vehicle detection
        try:
            self.yolo_detector = YOLOObjectDetector(
                confidence=0.3,  # Lower untuk capture more vehicles
                max_detections=15
            )
            self.yolo_enabled = True
            self.logger.info("✅ YOLO vehicle detector initialized")
        except Exception as e:
            self.yolo_detector = None
            self.yolo_enabled = False
            self.logger.warning(f"YOLO not available: {e}")
        
        # Initialize plate detector dengan settings yang lebih stabil
        self.plate_detector = RobustPlateDetector(streaming_mode=True)
        
        # Temporal stability tracking
        self.detection_history = deque(maxlen=10)  # Last 10 frames
        self.stable_detections = {}  # Track stable detections
        self.detection_id = 0
        
        # Stability parameters
        self.min_detection_count = 3  # Minimum detections untuk stable
        self.position_threshold = 50  # Max pixel distance untuk same detection
        self.confidence_threshold = 0.6  # Minimum confidence
        self.text_consistency_threshold = 0.7  # Text similarity requirement
        
        # Statistics
        self.total_detections = 0
        self.successful_ocr = 0
        self.failed_ocr = 0
        self.stable_count = 0
        self.vehicle_regions_found = 0
        
        self.logger.info("🛡️ Stable Plate Detector initialized")
    
    def detect_plates(self, image: np.ndarray) -> List[StablePlateDetection]:
        """
        Stable detection dengan temporal filtering
        """
        current_detections = []
        start_time = time.time()
        
        try:
            if self.yolo_enabled and self.yolo_detector:
                # Step 1: YOLO detect vehicles
                vehicle_regions = self._detect_vehicle_regions(image)
                self.vehicle_regions_found = len(vehicle_regions)
                
                if vehicle_regions:
                    # Step 2: Enhanced plate detection in regions
                    current_detections = self._enhanced_plate_detection_in_regions(image, vehicle_regions)
                else:
                    # Fallback: Full image detection
                    current_detections = self._fallback_full_detection(image)
            else:
                current_detections = self._fallback_full_detection(image)
            
            # Step 3: Apply temporal filtering
            stable_detections = self._apply_temporal_filtering(current_detections, time.time())
            
            detection_time = time.time() - start_time
            self.logger.info(f"🛡️ Stable detection: {len(stable_detections)} stable plates in {detection_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in stable detection: {e}")
            stable_detections = []
        
        return stable_detections
    
    def _detect_vehicle_regions(self, image: np.ndarray) -> List[Dict]:
        """Enhanced vehicle detection dengan better filtering"""
        vehicle_regions = []
        
        try:
            object_detections = self.yolo_detector.detect_objects(image)
            vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
            
            # Filter dan enhance vehicle regions
            for detection in object_detections:
                if detection.class_name in vehicle_classes:
                    x, y, w, h = detection.bbox
                    
                    # Enhanced expansion berdasarkan vehicle type
                    if detection.class_name == 'motorcycle':
                        expansion = 40  # Lebih besar untuk motor
                    elif detection.class_name in ['bus', 'truck']:
                        expansion = 25  # Lebih kecil untuk vehicle besar
                    else:
                        expansion = 35  # Default untuk car
                    
                    expanded_x = max(0, x - expansion)
                    expanded_y = max(0, y - expansion)
                    expanded_w = min(image.shape[1] - expanded_x, w + 2*expansion)
                    expanded_h = min(image.shape[0] - expanded_y, h + 2*expansion)
                    
                    # Validasi ukuran region
                    if expanded_w > 100 and expanded_h > 50:  # Minimum size
                        vehicle_regions.append({
                            'bbox': (expanded_x, expanded_y, expanded_w, expanded_h),
                            'vehicle_type': detection.class_name,
                            'confidence': detection.confidence,
                            'original_bbox': detection.bbox
                        })
            
            self.logger.debug(f"🚗 Found {len(vehicle_regions)} vehicle regions")
            
        except Exception as e:
            self.logger.warning(f"Vehicle detection failed: {e}")
        
        return vehicle_regions
    
    def _enhanced_plate_detection_in_regions(self, image: np.ndarray, vehicle_regions: List[Dict]) -> List[StablePlateDetection]:
        """Enhanced detection dengan multiple techniques"""
        all_detections = []
        
        for i, region in enumerate(vehicle_regions):
            try:
                x, y, w, h = region['bbox']
                roi = image[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Multi-technique detection
                region_detections = []
                
                # Method 1: Standard detection
                standard_detections = self.plate_detector.detect_plates(roi)
                region_detections.extend(standard_detections)
                
                # Method 2: Enhanced preprocessing untuk difficult cases
                if len(standard_detections) == 0:
                    enhanced_roi = self._enhance_roi_for_detection(roi)
                    enhanced_detections = self.plate_detector.detect_plates(enhanced_roi)
                    region_detections.extend(enhanced_detections)
                
                # Convert to stable detections
                for detection in region_detections:
                    det_x, det_y, det_w, det_h = detection.bbox
                    
                    # Validate detection quality
                    if self._validate_detection_quality(detection, roi):
                        stable_detection = StablePlateDetection(
                            text=detection.text,
                            confidence=min(100.0, detection.confidence * 1.1),  # Moderate bonus
                            bbox=(x + det_x, y + det_y, det_w, det_h),
                            angle=detection.angle,
                            vehicle_type=region['vehicle_type'],
                            detection_method=f"stable_{region['vehicle_type']}",
                            timestamp=time.time()
                        )
                        
                        all_detections.append(stable_detection)
                        
                        # Update stats
                        self.total_detections += 1
                        if detection.text and len(detection.text) >= 4:  # Stricter requirement
                            self.successful_ocr += 1
                        else:
                            self.failed_ocr += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process vehicle region {i+1}: {e}")
        
        return all_detections
    
    def _enhance_roi_for_detection(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing untuk detection yang sulit"""
        try:
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Apply CLAHE untuk better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Bilateral filter untuk noise reduction
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            self.logger.warning(f"ROI enhancement failed: {e}")
            return roi
    
    def _validate_detection_quality(self, detection, roi: np.ndarray) -> bool:
        """Validate detection quality"""
        try:
            # Text validation
            if not detection.text or len(detection.text) < 2:
                return False
            
            # Confidence validation
            if detection.confidence < self.confidence_threshold * 100:
                return False
            
            # Bbox validation
            x, y, w, h = detection.bbox
            roi_h, roi_w = roi.shape[:2]
            
            # Check if bbox is reasonable
            if w < 30 or h < 15:  # Too small
                return False
            
            if w > roi_w * 0.8 or h > roi_h * 0.6:  # Too large
                return False
            
            # Aspect ratio validation
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:  # Indonesian plates typically 2-5
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_temporal_filtering(self, current_detections: List[StablePlateDetection], timestamp: float) -> List[StablePlateDetection]:
        """Apply temporal filtering untuk stabilitas"""
        
        # Add current detections to history
        self.detection_history.append({
            'timestamp': timestamp,
            'detections': current_detections.copy()
        })
        
        # Update stable detections tracking
        for detection in current_detections:
            self._update_stable_tracking(detection, timestamp)
        
        # Get stable detections
        stable_detections = self._get_stable_detections(timestamp)
        
        return stable_detections
    
    def _update_stable_tracking(self, detection: StablePlateDetection, timestamp: float):
        """Update tracking untuk detection"""
        
        # Find matching stable detection
        best_match_id = None
        best_distance = float('inf')
        
        for stable_id, stable_data in self.stable_detections.items():
            distance = self._calculate_detection_distance(detection, stable_data['latest'])
            
            if distance < self.position_threshold and distance < best_distance:
                best_distance = distance
                best_match_id = stable_id
        
        if best_match_id is not None:
            # Update existing stable detection
            stable_data = self.stable_detections[best_match_id]
            stable_data['detections'].append(detection)
            stable_data['latest'] = detection
            stable_data['last_seen'] = timestamp
            stable_data['count'] += 1
            
            # Update stability score
            stable_data['stability_score'] = min(1.0, stable_data['count'] / 10.0)
            
        else:
            # Create new stable tracking
            self.detection_id += 1
            self.stable_detections[self.detection_id] = {
                'detections': [detection],
                'latest': detection,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'count': 1,
                'stability_score': 0.1
            }
    
    def _calculate_detection_distance(self, det1: StablePlateDetection, det2: StablePlateDetection) -> float:
        """Calculate distance between detections"""
        x1, y1, w1, h1 = det1.bbox
        x2, y2, w2, h2 = det2.bbox
        
        # Center points
        center1_x, center1_y = x1 + w1/2, y1 + h1/2
        center2_x, center2_y = x2 + w2/2, y2 + h2/2
        
        # Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Add text similarity penalty
        if det1.text and det2.text:
            text_similarity = self._calculate_text_similarity(det1.text, det2.text)
            if text_similarity < self.text_consistency_threshold:
                distance += 50  # Penalty for different text
        
        return distance
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Simple similarity based on common characters
        set1 = set(text1.upper())
        set2 = set(text2.upper())
        
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if len(union) > 0 else 0.0
    
    def _get_stable_detections(self, current_timestamp: float) -> List[StablePlateDetection]:
        """Get only stable detections"""
        stable_detections = []
        
        # Clean old detections
        expired_ids = []
        for stable_id, stable_data in self.stable_detections.items():
            if current_timestamp - stable_data['last_seen'] > 5.0:  # 5 second timeout
                expired_ids.append(stable_id)
        
        for expired_id in expired_ids:
            del self.stable_detections[expired_id]
        
        # Get stable detections
        for stable_data in self.stable_detections.values():
            if stable_data['count'] >= self.min_detection_count:
                latest_detection = stable_data['latest']
                latest_detection.stability_score = stable_data['stability_score']
                latest_detection.detection_count = stable_data['count']
                
                stable_detections.append(latest_detection)
                self.stable_count += 1
        
        # Sort by stability score
        stable_detections.sort(key=lambda x: x.stability_score, reverse=True)
        
        return stable_detections
    
    def _fallback_full_detection(self, image: np.ndarray) -> List[StablePlateDetection]:
        """Fallback detection"""
        self.logger.info("🔄 Using fallback full image detection")
        
        detections = self.plate_detector.detect_plates(image)
        
        stable_detections = []
        for detection in detections:
            if self._validate_detection_quality(detection, image):
                stable_detection = StablePlateDetection(
                    text=detection.text,
                    confidence=detection.confidence * 0.9,  # Slight penalty
                    bbox=detection.bbox,
                    angle=detection.angle,
                    detection_method="stable_fallback",
                    timestamp=time.time()
                )
                stable_detections.append(stable_detection)
                
                self.total_detections += 1
                if detection.text and len(detection.text) >= 4:
                    self.successful_ocr += 1
                else:
                    self.failed_ocr += 1
        
        return stable_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[StablePlateDetection], 
                       show_roi: bool = True) -> np.ndarray:
        """Draw stable detections dengan stability indicators"""
        result = frame.copy()
        
        sorted_detections = sorted(detections, key=lambda x: x.stability_score, reverse=True)
        
        for i, detection in enumerate(sorted_detections):
            x, y, w, h = detection.bbox
            
            # Color berdasarkan stability score
            if detection.stability_score > 0.8:
                plate_color = (0, 255, 0)  # GREEN untuk very stable
            elif detection.stability_score > 0.5:
                plate_color = (0, 255, 255)  # YELLOW untuk stable
            else:
                plate_color = (0, 128, 255)  # ORANGE untuk unstable
            
            thickness = max(2, int(4 * detection.stability_score))
            
            # Stable border dengan thickness berdasarkan stability
            cv2.rectangle(result, (x-2, y-2), (x + w + 2, y + h + 2), (0, 0, 0), thickness+1)
            cv2.rectangle(result, (x, y), (x + w, y + h), plate_color, thickness)
            
            # Stability indicators
            if detection.stability_score > 0.5:
                # Corner markers untuk stable detections
                corner_size = 15
                cv2.line(result, (x, y), (x + corner_size, y), plate_color, 3)
                cv2.line(result, (x, y), (x, y + corner_size), plate_color, 3)
                cv2.line(result, (x + w, y), (x + w - corner_size, y), plate_color, 3)
                cv2.line(result, (x + w, y), (x + w, y + corner_size), plate_color, 3)
            
            # Label dengan stability info
            if detection.text:
                stability_pct = int(detection.stability_score * 100)
                label = f"🛡️ STABLE: {detection.text} ({detection.confidence:.0f}%) S:{stability_pct}% [{detection.vehicle_type}]"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Background
                bg_x1, bg_y1 = x - 5, y - text_h - 15
                bg_x2, bg_y2 = x + text_w + 5, y - 5
                
                cv2.rectangle(result, (bg_x1-1, bg_y1-1), (bg_x2+1, bg_y2+1), (0, 0, 0), -1)
                cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), plate_color, -1)
                
                # Text
                cv2.putText(result, label, (x, y - 8), font, font_scale, (255, 255, 255), font_thickness)
        
        return result
    
    def get_statistics(self) -> Dict[str, any]:
        """Get stable detector statistics"""
        success_rate = (self.successful_ocr / self.total_detections * 100) if self.total_detections > 0 else 0
        stability_rate = (self.stable_count / max(1, self.total_detections) * 100)
        
        stats = {
            "total_detections": self.total_detections,
            "successful_ocr": self.successful_ocr,
            "failed_ocr": self.failed_ocr,
            "success_rate": round(success_rate, 1),
            "stable_detections": self.stable_count,
            "stability_rate": round(stability_rate, 1),
            "detection_method": "STABLE (YOLO+OpenCV+Temporal)",
            "vehicle_regions_found": self.vehicle_regions_found,
            "yolo_enabled": self.yolo_enabled,
            "active_stable_tracks": len(self.stable_detections)
        }
        
        if self.yolo_enabled and hasattr(self.yolo_detector, 'get_statistics'):
            yolo_stats = self.yolo_detector.get_statistics()
            stats.update({f"yolo_{k}": v for k, v in yolo_stats.items()})
        
        return stats

if __name__ == "__main__":
    # Test stable detector
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            detector = StablePlateDetector()
            
            # Test multiple frames untuk stability
            print("🛡️ Testing STABLE detector dengan multiple iterations...")
            
            for frame_num in range(5):
                print(f"\n--- Frame {frame_num + 1} ---")
                detections = detector.detect_plates(image)
                
                print(f"🎯 STABLE detected {len(detections)} plates:")
                for i, det in enumerate(detections):
                    print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%) S:{det.stability_score:.2f} [{det.vehicle_type}]")
                
                # Simulate frame delay
                time.sleep(0.1)
            
            # Final result
            final_detections = detector.detect_plates(image)
            result = detector.draw_detections(image, final_detections)
            cv2.imwrite("stable_plate_result.jpg", result)
            print(f"\n💾 Result saved: stable_plate_result.jpg")
            
            # Statistics
            stats = detector.get_statistics()
            print(f"📊 Stability: {stats['stability_rate']:.1f}% | Success: {stats['success_rate']:.1f}%")
            
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python stable_plate_detector.py <image_path>")