#!/usr/bin/env python3
"""
Hybrid Plate Detector
Kombinasi YOLO (untuk detect vehicles/objects) + OpenCV (untuk detect plates dalam region)
Approach ini menggunakan kekuatan kedua method
"""

import cv2
import numpy as np
import logging
import time
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Import components
from utils.yolo_detector import YOLOObjectDetector
from utils.robust_plate_detector import RobustPlateDetector
from utils.ocr_ensemble import OCREnsemble

@dataclass
class PlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    angle: float = 0.0
    processed_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    vehicle_type: str = "unknown"
    detection_method: str = "hybrid"

class HybridPlateDetector:
    """
    Hybrid approach: YOLO untuk detect vehicles + OpenCV untuk detect plates
    Ini menggabungkan akurasi YOLO object detection dengan plate-specific detection
    """
    
    def __init__(self, streaming_mode=True):
        """
        Initialize hybrid detector
        """
        self.streaming_mode = streaming_mode
        self.logger = logging.getLogger(__name__)
        
        # Initialize YOLO for vehicle detection dengan optimasi stability
        try:
            self.yolo_detector = YOLOObjectDetector(
                confidence=0.35,  # Optimized confidence untuk stability
                max_detections=15  # Reduced untuk better performance
            )
            self.yolo_enabled = True
            self.logger.info("✅ YOLO vehicle detector initialized (optimized)")
        except Exception as e:
            self.yolo_detector = None
            self.yolo_enabled = False
            self.logger.warning(f"YOLO not available: {e}")
        
        # Initialize plate detector dengan enhanced settings
        self.plate_detector = RobustPlateDetector(streaming_mode=True)
        
        # Initialize enhanced OCR
        try:
            self.ocr_ensemble = OCREnsemble()
            self.enhanced_ocr_enabled = True
            self.logger.info("✅ Enhanced OCR ensemble initialized")
        except Exception as e:
            self.ocr_ensemble = None
            self.enhanced_ocr_enabled = False
            self.logger.warning(f"Enhanced OCR not available: {e}")
        
        # Stability enhancements
        self.detection_history = []  # Track recent detections
        self.stability_threshold = 0.65  # Minimum confidence for stable detection
        
        # Statistics
        self.total_detections = 0
        self.successful_ocr = 0
        self.failed_ocr = 0
        self.vehicle_regions_found = 0
        
        self.logger.info("🔧 Hybrid Plate Detector initialized (YOLO + OpenCV)")
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Hybrid detection: YOLO vehicles -> OpenCV plates
        """
        detections = []
        start_time = time.time()
        
        try:
            if self.yolo_enabled and self.yolo_detector:
                # Step 1: YOLO detect vehicles/objects
                vehicle_regions = self._detect_vehicle_regions(image)
                self.vehicle_regions_found = len(vehicle_regions)
                
                if vehicle_regions:
                    # Step 2: Search for plates within vehicle regions
                    detections = self._detect_plates_in_regions(image, vehicle_regions)

                    # Step 3: Enhanced plate-vehicle association untuk crowded scenes
                    if len(detections) > 1 and len(vehicle_regions) > 0:
                        detections = self._enhance_plate_vehicle_association(detections, vehicle_regions)
                else:
                    # Fallback: Full image plate detection
                    detections = self._fallback_full_detection(image)
            else:
                # No YOLO: Use full image detection
                detections = self._fallback_full_detection(image)

            # Post-process detections
            detections = self._post_process_detections(detections)
            
            detection_time = time.time() - start_time
            self.logger.info(f"🎯 Hybrid detection: {len(detections)} plates in {detection_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in hybrid detection: {e}")
        
        return detections
    
    def _detect_vehicle_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Enhanced YOLO vehicle detection dengan intelligent region expansion
        """
        vehicle_regions = []

        try:
            # Get YOLO detections dengan vehicles only
            object_detections = self.yolo_detector.detect_objects(image, vehicles_only=True)

            for detection in object_detections:
                x, y, w, h = detection.bbox

                # Intelligent expansion based on vehicle type
                if detection.class_name == 'motorcycle':
                    # Motorcycles: smaller expansion, focus on front/rear
                    expansion_factor = 0.2
                    front_expansion = int(w * 0.3)  # More expansion in front
                    side_expansion = int(w * 0.1)   # Less on sides

                    expanded_x = max(0, x - side_expansion)
                    expanded_y = max(0, y - int(h * 0.1))
                    expanded_w = min(image.shape[1] - expanded_x, w + front_expansion + side_expansion)
                    expanded_h = min(image.shape[0] - expanded_y, h + int(h * 0.2))

                elif detection.class_name in ['car', 'bus', 'truck']:
                    # Cars/buses/trucks: larger expansion, more uniform
                    front_expansion = int(w * 0.25)
                    side_expansion = int(w * 0.15)

                    expanded_x = max(0, x - side_expansion)
                    expanded_y = max(0, y - int(h * 0.1))
                    expanded_w = min(image.shape[1] - expanded_x, w + front_expansion + side_expansion)
                    expanded_h = min(image.shape[0] - expanded_y, h + int(h * 0.15))

                else:
                    # Default expansion
                    expansion = 25
                    expanded_x = max(0, x - expansion)
                    expanded_y = max(0, y - expansion)
                    expanded_w = min(image.shape[1] - expanded_x, w + 2*expansion)
                    expanded_h = min(image.shape[0] - expanded_y, h + 2*expansion)

                # Add region to list dengan enhanced metadata
                vehicle_regions.append({
                    'bbox': (expanded_x, expanded_y, expanded_w, expanded_h),
                    'vehicle_type': detection.class_name,
                    'confidence': detection.confidence,
                    'original_bbox': detection.bbox,
                    'class_id': detection.class_id,
                    'expansion_applied': True
                })
            
            self.logger.info(f"🚗 Found {len(vehicle_regions)} vehicle regions")
            
        except Exception as e:
            self.logger.warning(f"Vehicle detection failed: {e}")
        
        return vehicle_regions
    
    def _detect_plates_in_regions(self, image: np.ndarray, vehicle_regions: List[Dict]) -> List[PlateDetection]:
        """
        Detect plates within vehicle regions
        """
        all_detections = []
        
        for i, region in enumerate(vehicle_regions):
            try:
                x, y, w, h = region['bbox']
                
                # Extract region
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # Apply plate detection to region
                region_detections = self.plate_detector.detect_plates(roi)
                
                # Enhanced OCR post-processing untuk better text recognition
                if self.enhanced_ocr_enabled and region_detections:
                    region_detections = self._enhance_detections_with_ocr(region_detections, roi)
                
                # Adjust coordinates back to full image
                for detection in region_detections:
                    det_x, det_y, det_w, det_h = detection.bbox
                    
                    # Enhanced stability filtering
                    base_confidence = detection.confidence
                    
                    # Apply spatial validation dan stability bonus
                    spatial_score = self._validate_plate_in_vehicle(detection.bbox, region, roi.shape)
                    if spatial_score < 0.3:  # Poor spatial fit
                        continue

                    stability_bonus = self._calculate_stability_bonus(detection, region['vehicle_type'])
                    spatial_bonus = spatial_score * 10  # Convert to bonus points
                    final_confidence = min(100.0, base_confidence + stability_bonus + spatial_bonus)
                    
                    # Create hybrid detection dengan enhanced validation
                    if self._validate_detection_stability(detection, final_confidence):
                        hybrid_detection = PlateDetection(
                            text=detection.text,
                            confidence=final_confidence,
                            bbox=(x + det_x, y + det_y, det_w, det_h),
                            angle=detection.angle,
                            vehicle_type=region['vehicle_type'],
                            detection_method=f"hybrid_{region['vehicle_type']}"
                        )
                        
                        all_detections.append(hybrid_detection)
                    
                    # Update stats
                    self.total_detections += 1
                    if detection.text and len(detection.text) >= 3:
                        self.successful_ocr += 1
                    else:
                        self.failed_ocr += 1
                
                self.logger.debug(f"Region {i+1} ({region['vehicle_type']}): {len(region_detections)} plates")
                
            except Exception as e:
                self.logger.warning(f"Failed to process vehicle region {i+1}: {e}")
        
        return all_detections
    
    def _fallback_full_detection(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Fallback: Full image plate detection
        """
        self.logger.info("🔄 Using fallback full image detection")
        
        detections = self.plate_detector.detect_plates(image)
        
        # Convert to hybrid format
        hybrid_detections = []
        for detection in detections:
            hybrid_detection = PlateDetection(
                text=detection.text,
                confidence=detection.confidence * 0.8,  # Slight penalty for not being vehicle-guided
                bbox=detection.bbox,
                angle=detection.angle,
                detection_method="hybrid_fallback"
            )
            hybrid_detections.append(hybrid_detection)
            
            # Update stats
            self.total_detections += 1
            if detection.text and len(detection.text) >= 3:
                self.successful_ocr += 1
            else:
                self.failed_ocr += 1
        
        return hybrid_detections
    
    def _post_process_detections(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """
        Post-process hybrid detections
        """
        if not detections:
            return detections
        
        # Remove duplicates dengan overlap detection
        filtered_detections = self._remove_duplicate_detections(detections)
        
        # Sort by confidence
        filtered_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit for streaming
        if self.streaming_mode and len(filtered_detections) > 3:
            filtered_detections = filtered_detections[:3]
        
        return filtered_detections
    
    def _remove_duplicate_detections(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """
        Remove duplicate detections berdasarkan overlap
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        filtered = []
        overlap_threshold = 0.5
        
        for detection in sorted_detections:
            bbox1 = detection.bbox
            is_duplicate = False
            
            for existing in filtered:
                bbox2 = existing.bbox
                overlap = self._calculate_overlap(bbox1, bbox2)
                
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap between two bounding boxes
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def _calculate_stability_bonus(self, detection, vehicle_type: str) -> float:
        """Calculate stability bonus berdasarkan detection quality"""
        bonus = 1.0
        
        try:
            # Text quality bonus
            if detection.text and len(detection.text) >= 4:
                bonus += 0.15  # Good text length
                
                # Check for Indonesian plate patterns
                if any(c.isalpha() for c in detection.text) and any(c.isdigit() for c in detection.text):
                    bonus += 0.1  # Mixed alphanumeric
            
            # Confidence bonus
            if detection.confidence >= 70:
                bonus += 0.1
            elif detection.confidence >= 80:
                bonus += 0.15
            
            # Vehicle type bonus
            if vehicle_type in ['car', 'motorcycle']:
                bonus += 0.05  # More common vehicles
            
            # Bbox quality bonus  
            x, y, w, h = detection.bbox
            aspect_ratio = w / h if h > 0 else 0
            
            if 2.0 <= aspect_ratio <= 5.0:  # Good plate ratio
                bonus += 0.1
            
            return min(1.3, bonus)  # Cap at 30% bonus
            
        except Exception:
            return 1.0
    
    def _validate_detection_stability(self, detection, confidence: float) -> bool:
        """Validate detection untuk stability"""
        try:
            # Minimum confidence check
            if confidence < self.stability_threshold * 100:
                return False
            
            # Text validation
            if not detection.text or len(detection.text) < 2:
                return False
            
            # Remove obvious false positives
            text = detection.text.upper()
            false_positive_patterns = ['WWW', 'HTTP', 'COM', '...', '___', '???']
            
            if any(pattern in text for pattern in false_positive_patterns):
                return False
            
            # Bbox validation
            x, y, w, h = detection.bbox
            
            # Size validation
            if w < 25 or h < 12:  # Too small
                return False
            
            if w > 300 or h > 100:  # Too large
                return False
            
            # Aspect ratio validation
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 1.2 or aspect_ratio > 8.0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Validation error: {e}")
            return False
    
    def _enhance_detections_with_ocr(self, detections: List[PlateDetection], roi: np.ndarray) -> List[PlateDetection]:
        """Enhance detections dengan advanced OCR"""
        enhanced_detections = []
        
        try:
            for detection in detections:
                # Extract plate region dari ROI
                det_x, det_y, det_w, det_h = detection.bbox
                
                # Pastikan coordinate dalam bounds
                det_x = max(0, min(det_x, roi.shape[1] - 1))
                det_y = max(0, min(det_y, roi.shape[0] - 1))
                det_w = max(1, min(det_w, roi.shape[1] - det_x))
                det_h = max(1, min(det_h, roi.shape[0] - det_y))
                
                plate_roi = roi[det_y:det_y+det_h, det_x:det_x+det_w]
                
                if plate_roi.size > 0:
                    # Apply enhanced OCR
                    enhanced_text, enhanced_confidence, ocr_details = self.ocr_ensemble.ensemble_ocr(
                        plate_roi, 
                        methods=['tesseract_eng', 'tesseract_indonesia']
                    )
                    
                    # Use enhanced result jika lebih baik
                    if enhanced_text and len(enhanced_text) >= len(detection.text):
                        detection.text = enhanced_text
                        # Combine confidence dengan weighted average
                        detection.confidence = (detection.confidence * 0.6 + enhanced_confidence * 0.4)
                
                enhanced_detections.append(detection)
                
        except Exception as e:
            self.logger.warning(f"OCR enhancement error: {e}")
            return detections  # Return original jika error
        
        return enhanced_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[PlateDetection], 
                       show_roi: bool = True) -> np.ndarray:
        """
        Draw hybrid detections dengan styling yang distinctive
        """
        result = frame.copy()
        
        # Sort detections by confidence untuk prioritas visual
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        for i, detection in enumerate(sorted_detections):
            x, y, w, h = detection.bbox
            
            # HYBRID BOUNDING BOX - Kombinasi YOLO + OpenCV colors
            if i == 0:  # Best detection - purple (hybrid indicator)
                plate_color = (255, 0, 128)  # PURPLE untuk hybrid terbaik
                thickness = 4
            else:  # Other detections - teal
                plate_color = (128, 255, 0)  # TEAL untuk hybrid lainnya
                thickness = 3
            
            # Triple border untuk hybrid signature
            # Border luar (hitam)
            cv2.rectangle(result, (x-3, y-3), (x + w + 3, y + h + 3), (0, 0, 0), thickness+2)
            # Border tengah (putih)
            cv2.rectangle(result, (x-1, y-1), (x + w + 1, y + h + 1), (255, 255, 255), 1)
            # Border dalam (hybrid color)
            cv2.rectangle(result, (x, y), (x + w, y + h), plate_color, thickness)
            
            # Hybrid corner markers dengan double lines
            corner_size = 15
            corner_thickness = 2
            offset = 3
            
            # Double corner markers
            for dx in [0, offset]:
                for dy in [0, offset]:
                    # Top-left
                    cv2.line(result, (x+dx, y+dy), (x + corner_size+dx, y+dy), plate_color, corner_thickness)
                    cv2.line(result, (x+dx, y+dy), (x+dx, y + corner_size+dy), plate_color, corner_thickness)
                    # Top-right
                    cv2.line(result, (x + w-dx, y+dy), (x + w - corner_size-dx, y+dy), plate_color, corner_thickness)
                    cv2.line(result, (x + w-dx, y+dy), (x + w-dx, y + corner_size+dy), plate_color, corner_thickness)
                    # Bottom corners
                    cv2.line(result, (x+dx, y + h-dy), (x + corner_size+dx, y + h-dy), plate_color, corner_thickness)
                    cv2.line(result, (x+dx, y + h-dy), (x+dx, y + h - corner_size-dy), plate_color, corner_thickness)
                    cv2.line(result, (x + w-dx, y + h-dy), (x + w - corner_size-dx, y + h-dy), plate_color, corner_thickness)
                    cv2.line(result, (x + w-dx, y + h-dy), (x + w-dx, y + h - corner_size-dy), plate_color, corner_thickness)
            
            # HYBRID LABEL dengan vehicle type info
            if detection.text:
                if i == 0:
                    label = f"🎯 HYBRID-PLATE: {detection.text} ({detection.confidence:.0f}%) [{detection.vehicle_type}]"
                    font_scale = 0.7
                else:
                    label = f"HYBRID-PLATE: {detection.text} ({detection.confidence:.0f}%) [{detection.vehicle_type}]"
                    font_scale = 0.6
                
                font = cv2.FONT_HERSHEY_DUPLEX
                font_thickness = 2
                
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Background dengan hybrid styling (gradient effect)
                bg_padding = 6
                bg_x1 = x - bg_padding
                bg_y1 = y - text_h - 18 - bg_padding  
                bg_x2 = x + text_w + bg_padding
                bg_y2 = y - 5 + bg_padding
                
                # Triple background (black -> white -> color)
                cv2.rectangle(result, (bg_x1-2, bg_y1-2), (bg_x2+2, bg_y2+2), (0, 0, 0), -1)
                cv2.rectangle(result, (bg_x1-1, bg_y1-1), (bg_x2+1, bg_y2+1), (255, 255, 255), -1)
                cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), plate_color, -1)
                
                # Text dengan shadow effect
                # Shadow
                cv2.putText(result, label, (x+1, y - 7), font, font_scale, (0, 0, 0), font_thickness+1)
                # Main text
                cv2.putText(result, label, (x, y - 8), font, font_scale, (255, 255, 255), font_thickness)
        
        return result
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get hybrid detection statistics
        """
        success_rate = (self.successful_ocr / self.total_detections * 100) if self.total_detections > 0 else 0
        
        stats = {
            "total_detections": self.total_detections,
            "successful_ocr": self.successful_ocr,
            "failed_ocr": self.failed_ocr,
            "success_rate": round(success_rate, 1),
            "detection_method": "HYBRID (YOLO+OpenCV)",
            "vehicle_regions_found": self.vehicle_regions_found,
            "yolo_enabled": self.yolo_enabled
        }
        
        # Add YOLO stats if available
        if self.yolo_enabled and hasattr(self.yolo_detector, 'get_statistics'):
            yolo_stats = self.yolo_detector.get_statistics()
            stats.update({f"yolo_{k}": v for k, v in yolo_stats.items()})
        
        return stats

    def _validate_plate_in_vehicle(self, plate_bbox: Tuple[int, int, int, int],
                                 vehicle_region: Dict, roi_shape: Tuple[int, int, int]) -> float:
        """
        Validate spatial relationship antara plate dan vehicle
        Returns score 0.0-1.0 dimana 1.0 = perfect spatial fit
        """
        try:
            plate_x, plate_y, plate_w, plate_h = plate_bbox
            vehicle_type = vehicle_region['vehicle_type']
            roi_h, roi_w = roi_shape[:2]

            # Calculate plate position relative to vehicle region
            plate_center_x = plate_x + plate_w / 2
            plate_center_y = plate_y + plate_h / 2

            # Normalize to ROI dimensions
            norm_x = plate_center_x / roi_w if roi_w > 0 else 0.5
            norm_y = plate_center_y / roi_h if roi_h > 0 else 0.5

            # Vehicle-specific spatial expectations
            if vehicle_type == 'motorcycle':
                # Motorcycles: plate biasanya di front atau rear, center horizontally
                expected_regions = [
                    (0.2, 0.7, 0.3, 0.9),  # Front area
                    (0.7, 0.9, 0.3, 0.9),  # Rear area
                ]
            elif vehicle_type in ['car', 'bus', 'truck']:
                # Cars: plate bisa front atau rear, lebih wide area
                expected_regions = [
                    (0.1, 0.9, 0.2, 0.8),  # Front bumper area
                    (0.1, 0.9, 0.6, 0.95), # Rear area
                ]
            else:
                # Default: anywhere in bottom half
                expected_regions = [
                    (0.0, 1.0, 0.4, 1.0),  # Bottom half
                ]

            # Calculate score based on proximity to expected regions
            max_score = 0.0
            for region in expected_regions:
                x_min, x_max, y_min, y_max = region

                # Check if plate center is within expected region
                if x_min <= norm_x <= x_max and y_min <= norm_y <= y_max:
                    # Perfect fit
                    max_score = 1.0
                    break
                else:
                    # Calculate distance to nearest expected region
                    x_dist = min(abs(norm_x - x_min), abs(norm_x - x_max)) if not (x_min <= norm_x <= x_max) else 0
                    y_dist = min(abs(norm_y - y_min), abs(norm_y - y_max)) if not (y_min <= norm_y <= y_max) else 0

                    # Distance-based score (closer = higher score)
                    distance = (x_dist**2 + y_dist**2)**0.5
                    region_score = max(0, 1.0 - distance * 2)  # Scale distance
                    max_score = max(max_score, region_score)

            # Size validation bonus
            plate_area = plate_w * plate_h
            roi_area = roi_w * roi_h

            if roi_area > 0:
                size_ratio = plate_area / roi_area

                # Reasonable plate size relative to vehicle (1-15% of vehicle area)
                if vehicle_type == 'motorcycle':
                    if 0.02 <= size_ratio <= 0.20:  # 2-20% for motorcycles
                        size_bonus = 0.2
                    elif 0.005 <= size_ratio <= 0.30:  # Acceptable range
                        size_bonus = 0.1
                    else:
                        size_bonus = 0.0
                else:  # Cars, buses, trucks
                    if 0.01 <= size_ratio <= 0.10:  # 1-10% for cars
                        size_bonus = 0.2
                    elif 0.005 <= size_ratio <= 0.15:  # Acceptable range
                        size_bonus = 0.1
                    else:
                        size_bonus = 0.0

                max_score = min(1.0, max_score + size_bonus)

            # Aspect ratio validation bonus
            aspect_ratio = plate_w / plate_h if plate_h > 0 else 0
            if 2.0 <= aspect_ratio <= 5.0:  # Typical plate aspect ratio
                max_score = min(1.0, max_score + 0.1)

            return max_score

        except Exception:
            return 0.5  # Neutral score jika validation gagal

    def _enhance_plate_vehicle_association(self, detections: List[PlateDetection],
                                         vehicle_regions: List[Dict]) -> List[PlateDetection]:
        """
        Enhanced plate-vehicle association untuk crowded scenes dengan multiple candidates
        """
        if not detections or not vehicle_regions:
            return detections

        try:
            # Group plates by spatial proximity untuk each vehicle
            enhanced_detections = []
            processed_plates = set()

            for vehicle_idx, vehicle_region in enumerate(vehicle_regions):
                vehicle_bbox = vehicle_region['bbox']
                vehicle_type = vehicle_region['vehicle_type']

                # Find all plates yang potentially belong to this vehicle
                candidate_plates = []

                for plate_idx, detection in enumerate(detections):
                    if plate_idx in processed_plates:
                        continue

                    # Calculate association score untuk this vehicle
                    association_score = self._calculate_plate_vehicle_association(
                        detection, vehicle_region
                    )

                    if association_score > 0.3:  # Minimum threshold
                        candidate_plates.append((plate_idx, detection, association_score))

                # Process multiple candidates untuk this vehicle
                if candidate_plates:
                    # Sort by association score
                    candidate_plates.sort(key=lambda x: x[2], reverse=True)

                    # Handle multiple plate scenario
                    if len(candidate_plates) > 1 and vehicle_type == 'motorcycle':
                        # Motorcycles can have front and rear plates
                        processed_candidates = self._resolve_multiple_motorcycle_plates(
                            candidate_plates, vehicle_region
                        )
                    else:
                        # For cars, usually take the best candidate only
                        processed_candidates = [candidate_plates[0]] if candidate_plates else []

                    # Add processed candidates dengan enhanced confidence
                    for plate_idx, detection, score in processed_candidates:
                        # Enhance confidence based on association quality
                        enhanced_detection = PlateDetection(
                            bbox=detection.bbox,
                            text=detection.text,
                            confidence=min(1.0, detection.confidence * (1.0 + score * 0.3)),
                            method=f"{detection.method}+association"
                        )
                        enhanced_detections.append(enhanced_detection)
                        processed_plates.add(plate_idx)

            # Add remaining unprocessed plates dengan reduced confidence (orphaned plates)
            for plate_idx, detection in enumerate(detections):
                if plate_idx not in processed_plates:
                    # Reduce confidence untuk orphaned plates
                    orphaned_detection = PlateDetection(
                        bbox=detection.bbox,
                        text=detection.text,
                        confidence=detection.confidence * 0.7,  # Confidence penalty
                        method=f"{detection.method}+orphaned"
                    )
                    enhanced_detections.append(orphaned_detection)

            return enhanced_detections

        except Exception as e:
            self.logger.warning(f"Error in enhanced plate-vehicle association: {e}")
            return detections

    def _calculate_plate_vehicle_association(self, plate_detection: PlateDetection,
                                           vehicle_region: Dict) -> float:
        """
        Calculate comprehensive association score antara plate dan vehicle
        """
        try:
            vehicle_bbox = vehicle_region['bbox']
            vehicle_type = vehicle_region['vehicle_type']

            # Extract vehicle region boundaries
            veh_x, veh_y, veh_w, veh_h = vehicle_bbox

            # Extract plate boundaries (need to convert to vehicle coordinate system)
            plate_x, plate_y, plate_w, plate_h = plate_detection.bbox

            # Check if plate is within expanded vehicle region
            expansion_factor = 1.2 if vehicle_type == 'motorcycle' else 1.1
            expanded_veh_w = int(veh_w * expansion_factor)
            expanded_veh_h = int(veh_h * expansion_factor)
            expanded_veh_x = max(0, veh_x - (expanded_veh_w - veh_w) // 2)
            expanded_veh_y = max(0, veh_y - (expanded_veh_h - veh_h) // 2)

            # 1. Containment score (is plate within vehicle region?)
            containment_score = self._calculate_containment_score(
                (plate_x, plate_y, plate_w, plate_h),
                (expanded_veh_x, expanded_veh_y, expanded_veh_w, expanded_veh_h)
            )

            # 2. Spatial position score (is plate in expected position?)
            spatial_score = self._calculate_spatial_position_score(
                (plate_x, plate_y, plate_w, plate_h),
                (veh_x, veh_y, veh_w, veh_h),
                vehicle_type
            )

            # 3. Size compatibility score
            size_score = self._calculate_size_compatibility_score(
                (plate_w, plate_h), (veh_w, veh_h), vehicle_type
            )

            # 4. Text quality bonus
            text_quality_score = self._calculate_text_quality_score(plate_detection.text)

            # Weighted combination
            weights = {
                'containment': 0.3,
                'spatial': 0.4,
                'size': 0.2,
                'text_quality': 0.1
            }

            total_score = (
                containment_score * weights['containment'] +
                spatial_score * weights['spatial'] +
                size_score * weights['size'] +
                text_quality_score * weights['text_quality']
            )

            return min(1.0, total_score)

        except Exception:
            return 0.3  # Default moderate score

    def _resolve_multiple_motorcycle_plates(self, candidate_plates: List, vehicle_region: Dict) -> List:
        """
        Resolve multiple plate candidates untuk motorcycles (front/rear plates)
        """
        if len(candidate_plates) <= 1:
            return candidate_plates

        try:
            vehicle_bbox = vehicle_region['bbox']
            veh_x, veh_y, veh_w, veh_h = vehicle_bbox

            # Categorize plates by position (front/rear)
            front_plates = []
            rear_plates = []

            for plate_idx, detection, score in candidate_plates:
                plate_x, plate_y, plate_w, plate_h = detection.bbox
                plate_center_x = plate_x + plate_w / 2
                vehicle_center_x = veh_x + veh_w / 2

                # Determine if plate is in front or rear half of vehicle
                if plate_center_x < vehicle_center_x:
                    front_plates.append((plate_idx, detection, score))
                else:
                    rear_plates.append((plate_idx, detection, score))

            # Select best from each position
            selected_plates = []

            if front_plates:
                best_front = max(front_plates, key=lambda x: x[2])
                selected_plates.append(best_front)

            if rear_plates:
                best_rear = max(rear_plates, key=lambda x: x[2])
                # Only add rear plate if it's significantly different from front
                if not front_plates or self._are_plates_different(best_rear[1], selected_plates[0][1]):
                    selected_plates.append(best_rear)

            # Limit to maximum 2 plates per motorcycle
            return selected_plates[:2]

        except Exception:
            # Fallback to best candidate only
            return [candidate_plates[0]] if candidate_plates else []

    def _calculate_containment_score(self, plate_bbox: Tuple, vehicle_bbox: Tuple) -> float:
        """Calculate how well plate is contained within vehicle region"""
        px, py, pw, ph = plate_bbox
        vx, vy, vw, vh = vehicle_bbox

        # Calculate overlap ratio
        overlap_x = max(0, min(px + pw, vx + vw) - max(px, vx))
        overlap_y = max(0, min(py + ph, vy + vh) - max(py, vy))
        overlap_area = overlap_x * overlap_y

        plate_area = pw * ph
        return overlap_area / plate_area if plate_area > 0 else 0.0

    def _calculate_spatial_position_score(self, plate_bbox: Tuple, vehicle_bbox: Tuple, vehicle_type: str) -> float:
        """Calculate score based on expected plate position for vehicle type"""
        px, py, pw, ph = plate_bbox
        vx, vy, vw, vh = vehicle_bbox

        # Normalize plate position relative to vehicle
        plate_center_x = (px + pw / 2 - vx) / vw if vw > 0 else 0.5
        plate_center_y = (py + ph / 2 - vy) / vh if vh > 0 else 0.5

        if vehicle_type == 'motorcycle':
            # Motorcycles: plates at front/rear, vertically centered
            vertical_score = 1.0 - abs(plate_center_y - 0.6)  # Slightly below center
            horizontal_score = max(
                1.0 - abs(plate_center_x - 0.2),  # Front position
                1.0 - abs(plate_center_x - 0.8)   # Rear position
            )
        else:
            # Cars: plates at front/rear bumpers, lower position
            vertical_score = max(
                1.0 - abs(plate_center_y - 0.15),  # Front bumper
                1.0 - abs(plate_center_y - 0.85)   # Rear bumper
            )
            horizontal_score = 1.0 - abs(plate_center_x - 0.5)  # Center horizontally

        return (vertical_score + horizontal_score) / 2

    def _calculate_size_compatibility_score(self, plate_size: Tuple, vehicle_size: Tuple, vehicle_type: str) -> float:
        """Calculate score based on plate size compatibility dengan vehicle"""
        pw, ph = plate_size
        vw, vh = vehicle_size

        plate_area = pw * ph
        vehicle_area = vw * vh

        if vehicle_area == 0:
            return 0.5

        size_ratio = plate_area / vehicle_area

        if vehicle_type == 'motorcycle':
            # Motorcycles: 2-20% of vehicle area
            optimal_range = (0.02, 0.20)
            acceptable_range = (0.005, 0.30)
        else:
            # Cars: 1-10% of vehicle area
            optimal_range = (0.01, 0.10)
            acceptable_range = (0.005, 0.15)

        if optimal_range[0] <= size_ratio <= optimal_range[1]:
            return 1.0
        elif acceptable_range[0] <= size_ratio <= acceptable_range[1]:
            return 0.7
        else:
            return 0.3

    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate score based on plate text quality"""
        if not text or len(text) < 3:
            return 0.0

        # Indonesian plate patterns
        score = 0.0

        # Length bonus (typical Indonesian plates: 6-9 characters)
        if 6 <= len(text) <= 9:
            score += 0.4
        elif 4 <= len(text) <= 11:
            score += 0.2

        # Character pattern bonus
        if text.replace(' ', '').isalnum():  # Alphanumeric only
            score += 0.3

        # Indonesian pattern detection (e.g., "B 1234 XYZ")
        import re
        indonesia_pattern = r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
        if re.match(indonesia_pattern, text.replace(' ', ' ').strip()):
            score += 0.3

        return min(1.0, score)

    def _are_plates_different(self, plate1: PlateDetection, plate2: PlateDetection) -> bool:
        """Check if two plates are significantly different (not duplicates)"""
        # Different positions
        p1x, p1y, p1w, p1h = plate1.bbox
        p2x, p2y, p2w, p2h = plate2.bbox

        # Calculate center distance
        center_dist = math.sqrt((p1x + p1w/2 - p2x - p2w/2)**2 + (p1y + p1h/2 - p2y - p2h/2)**2)

        # Different if centers are >50 pixels apart
        position_different = center_dist > 50

        # Different text content
        text_different = plate1.text != plate2.text if plate1.text and plate2.text else True

        return position_different or text_different

if __name__ == "__main__":
    # Test hybrid detector
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            detector = HybridPlateDetector()
            detections = detector.detect_plates(image)
            
            print(f"🎯 HYBRID detected {len(detections)} license plates:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%) [{det.vehicle_type}] via {det.detection_method}")
            
            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("hybrid_plate_result.jpg", result)
            print("💾 Result saved: hybrid_plate_result.jpg")
            
            # Show statistics
            stats = detector.get_statistics()
            print(f"📊 Statistics: {stats}")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python hybrid_plate_detector.py <image_path>")