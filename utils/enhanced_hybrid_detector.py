"""
Enhanced Hybrid Detector - Dual YOLO System
Menggabungkan Vehicle YOLO + License Plate YOLO + OCR Fallback
untuk akurasi maksimal deteksi plat Indonesia
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass

# Import existing detectors
from utils.yolo_detector import YOLOObjectDetector, ObjectDetection
from utils.license_plate_yolo_detector import LicensePlateYOLODetector, LicensePlateDetection
from utils.stable_plate_counter import StablePlateCounter

@dataclass
class EnhancedDetectionResult:
    """Enhanced detection result dengan multiple detection methods"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    detection_method: str  # 'plate_yolo', 'ocr', 'hybrid'
    vehicle_type: str
    is_valid_format: bool
    vehicle_bbox: Optional[Tuple[int, int, int, int]] = None

class EnhancedHybridDetector:
    """
    Enhanced Hybrid Detector dengan Dual YOLO System
    Pipeline: Vehicle YOLO → License Plate YOLO → OCR Fallback
    """

    def __init__(self, config: Dict):
        """
        Initialize Enhanced Hybrid Detector

        Args:
            config: Configuration dictionary dengan settings untuk semua components
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Initialize Vehicle YOLO (existing)
        self.vehicle_detector = None
        if config.get('yolo', {}).get('enabled', False):
            try:
                from utils.yolo_detector import YOLOObjectDetector
                yolo_config = config['yolo']
                self.vehicle_detector = YOLOObjectDetector(
                    model_path=yolo_config.get('model_path', 'yolov8n.pt'),
                    confidence=yolo_config.get('confidence', 0.4),
                    iou_threshold=yolo_config.get('iou_threshold', 0.5),
                    max_detections=yolo_config.get('max_detections', 15)
                )
                self.logger.info("✅ Vehicle YOLO detector initialized")
            except Exception as e:
                self.logger.warning(f"Vehicle YOLO initialization failed: {e}")
                self.vehicle_detector = None

        # Initialize License Plate YOLO (new)
        self.plate_detector = None
        if config.get('license_plate_yolo', {}).get('enabled', False):
            try:
                from utils.license_plate_yolo_detector import LicensePlateYOLODetector
                plate_config = config['license_plate_yolo']
                self.plate_detector = LicensePlateYOLODetector(
                    model_path=plate_config.get('model_path', 'license_plate_yolo.pt'),
                    confidence=plate_config.get('confidence', 0.3),
                    iou_threshold=plate_config.get('iou_threshold', 0.4)
                )
                self.logger.info("✅ License Plate YOLO detector initialized")
            except Exception as e:
                self.logger.warning(f"License Plate YOLO initialization failed: {e}")
                self.plate_detector = None

        # OCR Fallback (existing Tesseract)
        self.ocr_enabled = config.get('ocr', {}).get('enabled', True)
        if self.ocr_enabled:
            try:
                import pytesseract
                self.tesseract_config = config.get('tesseract', {})
                self.logger.info("✅ Tesseract OCR fallback available")
            except ImportError:
                self.logger.warning("Tesseract not available - install pytesseract")
                self.ocr_enabled = False

        # Detection statistics
        self.stats = {
            'plate_yolo_detections': 0,
            'ocr_detections': 0,
            'hybrid_detections': 0,
            'total_detections': 0,
            'detection_times': []
        }

        # Indonesian plate validation
        self.indonesian_validator = self._init_indonesian_validator()

    def _init_indonesian_validator(self):
        """Initialize Indonesian plate format validator"""
        import re
        return {
            'patterns': [
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',  # B 1234 ABC
                r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',        # B1234ABC
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{2,3}$',  # AA 1234 BB
            ]
        }

    def detect_license_plates(self, frame: np.ndarray) -> List[EnhancedDetectionResult]:
        """
        Enhanced license plate detection dengan dual YOLO system

        Args:
            frame: Input frame

        Returns:
            List of EnhancedDetectionResult objects
        """
        start_time = time.time()
        all_detections = []

        try:
            # Step 1: Vehicle Detection (jika enabled)
            vehicle_regions = []
            if self.vehicle_detector and self.vehicle_detector.is_enabled():
                vehicle_objects = self.vehicle_detector.detect_objects(frame, vehicles_only=True)
                vehicle_regions = [obj.bbox for obj in vehicle_objects if obj.is_vehicle]
                self.logger.debug(f"Detected {len(vehicle_regions)} vehicle regions")

            # Step 2: License Plate YOLO Detection
            plate_yolo_detections = []
            if self.plate_detector and self.plate_detector.is_enabled():
                # Detect plates dengan priority pada vehicle regions
                if vehicle_regions:
                    plate_yolo_detections = self.plate_detector.detect_license_plates(frame, vehicle_regions)
                else:
                    plate_yolo_detections = self.plate_detector.detect_license_plates(frame)

                # Convert to EnhancedDetectionResult
                for detection in plate_yolo_detections:
                    enhanced_detection = EnhancedDetectionResult(
                        text=detection.text,
                        confidence=detection.confidence,
                        bbox=detection.bbox,
                        detection_method='plate_yolo',
                        vehicle_type=detection.vehicle_type,
                        is_valid_format=detection.is_valid_format,
                        vehicle_bbox=self._find_parent_vehicle(detection.bbox, vehicle_regions)
                    )
                    all_detections.append(enhanced_detection)

                self.stats['plate_yolo_detections'] += len(plate_yolo_detections)
                self.logger.debug(f"Plate YOLO detected {len(plate_yolo_detections)} plates")

            # Step 3: OCR Fallback (untuk regions yang belum terdeteksi)
            if self.ocr_enabled:
                ocr_detections = self._ocr_fallback_detection(frame, vehicle_regions, plate_yolo_detections)
                all_detections.extend(ocr_detections)

            # Step 4: Hybrid Validation & Enhancement
            validated_detections = self._validate_and_enhance_detections(all_detections, frame)

            # Update statistics
            detection_time = time.time() - start_time
            self.stats['detection_times'].append(detection_time)
            self.stats['total_detections'] += len(validated_detections)

            # Keep last 100 detection times
            if len(self.stats['detection_times']) > 100:
                self.stats['detection_times'] = self.stats['detection_times'][-100:]

            return validated_detections

        except Exception as e:
            self.logger.error(f"Error in enhanced hybrid detection: {str(e)}")
            return []

    def _find_parent_vehicle(self, plate_bbox: Tuple[int, int, int, int],
                           vehicle_regions: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Find parent vehicle region untuk plate bbox"""
        px, py, pw, ph = plate_bbox
        plate_center = (px + pw//2, py + ph//2)

        for vehicle_bbox in vehicle_regions:
            vx, vy, vw, vh = vehicle_bbox
            # Check if plate center is inside vehicle region
            if (vx <= plate_center[0] <= vx + vw and
                vy <= plate_center[1] <= vy + vh):
                return vehicle_bbox

        return None

    def _ocr_fallback_detection(self, frame: np.ndarray,
                              vehicle_regions: List[Tuple[int, int, int, int]],
                              existing_detections: List[LicensePlateDetection]) -> List[EnhancedDetectionResult]:
        """OCR fallback untuk areas yang belum terdeteksi plate YOLO"""
        ocr_detections = []

        if not self.ocr_enabled:
            return ocr_detections

        try:
            import pytesseract

            # Get areas yang sudah di-cover oleh plate YOLO
            covered_areas = [d.bbox for d in existing_detections]

            # Process vehicle regions yang belum ada plate detection
            for vehicle_bbox in vehicle_regions:
                vx, vy, vw, vh = vehicle_bbox

                # Check if this vehicle region sudah ada plate detection
                has_plate_detection = any(
                    self._is_overlapping(vehicle_bbox, plate_bbox)
                    for plate_bbox in covered_areas
                )

                if has_plate_detection:
                    continue  # Skip, sudah ada plate YOLO detection

                # Extract vehicle ROI untuk OCR
                vehicle_roi = frame[vy:vy+vh, vx:vx+vw]

                if vehicle_roi.size == 0:
                    continue

                # Enhanced preprocessing untuk OCR
                processed_roi = self._preprocess_for_ocr(vehicle_roi)

                # OCR detection
                try:
                    ocr_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ocr_text = pytesseract.image_to_string(processed_roi, config=ocr_config).strip()

                    if len(ocr_text) >= 5:  # Minimum plate length
                        # Validate Indonesian format
                        is_valid = self._validate_indonesian_format(ocr_text)

                        # Create detection result
                        ocr_detection = EnhancedDetectionResult(
                            text=ocr_text,
                            confidence=0.6,  # Base OCR confidence
                            bbox=vehicle_bbox,  # Use vehicle bbox for now
                            detection_method='ocr',
                            vehicle_type='unknown',
                            is_valid_format=is_valid,
                            vehicle_bbox=vehicle_bbox
                        )

                        ocr_detections.append(ocr_detection)
                        self.stats['ocr_detections'] += 1

                except Exception as e:
                    self.logger.debug(f"OCR processing error: {e}")

        except ImportError:
            self.logger.warning("Tesseract not available for OCR fallback")

        return ocr_detections

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing untuk OCR"""
        try:
            # Convert to grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Resize untuk better OCR (if too small)
            height, width = cleaned.shape
            if height < 50 or width < 150:
                scale_factor = max(50/height, 150/width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            return cleaned

        except Exception as e:
            self.logger.debug(f"Preprocessing error: {e}")
            return roi

    def _validate_indonesian_format(self, text: str) -> bool:
        """Validate Indonesian license plate format"""
        if not text or len(text) < 5:
            return False

        import re
        clean_text = re.sub(r'[^A-Z0-9\s]', '', text.upper().strip())

        for pattern in self.indonesian_validator['patterns']:
            if re.match(pattern, clean_text):
                return True

        return False

    def _is_overlapping(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> bool:
        """Check if two bounding boxes overlap"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def _validate_and_enhance_detections(self, detections: List[EnhancedDetectionResult],
                                       frame: np.ndarray) -> List[EnhancedDetectionResult]:
        """Validate dan enhance detections dengan hybrid approach"""
        validated_detections = []

        for detection in detections:
            # Enhance confidence based on method dan validation
            enhanced_confidence = self._calculate_enhanced_confidence(detection, frame)

            # Create enhanced detection
            enhanced_detection = EnhancedDetectionResult(
                text=detection.text,
                confidence=enhanced_confidence,
                bbox=detection.bbox,
                detection_method=detection.detection_method,
                vehicle_type=detection.vehicle_type,
                is_valid_format=detection.is_valid_format,
                vehicle_bbox=detection.vehicle_bbox
            )

            # Only keep detections dengan reasonable confidence
            if enhanced_confidence >= 0.3:
                validated_detections.append(enhanced_detection)

        # Remove duplicates
        final_detections = self._remove_duplicate_detections(validated_detections)

        # Mark hybrid detections
        for detection in final_detections:
            if detection.detection_method in ['plate_yolo', 'ocr']:
                detection.detection_method = 'hybrid'
                self.stats['hybrid_detections'] += 1

        return final_detections

    def _calculate_enhanced_confidence(self, detection: EnhancedDetectionResult,
                                     frame: np.ndarray) -> float:
        """Calculate enhanced confidence based on multiple factors"""
        base_confidence = detection.confidence

        # Bonus for valid Indonesian format
        format_bonus = 0.2 if detection.is_valid_format else 0.0

        # Bonus based on detection method
        method_bonus = {
            'plate_yolo': 0.1,    # YOLO has good accuracy
            'ocr': -0.1,          # OCR less reliable
            'hybrid': 0.2         # Best of both worlds
        }.get(detection.detection_method, 0.0)

        # Size validation bonus
        x, y, w, h = detection.bbox
        area = w * h
        size_bonus = 0.1 if 800 <= area <= 15000 else -0.1

        # Calculate final confidence
        enhanced_confidence = base_confidence + format_bonus + method_bonus + size_bonus

        return max(0.0, min(1.0, enhanced_confidence))

    def _remove_duplicate_detections(self, detections: List[EnhancedDetectionResult]) -> List[EnhancedDetectionResult]:
        """Remove duplicate detections using IoU dan text similarity"""
        if len(detections) <= 1:
            return detections

        unique_detections = []

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        for detection in sorted_detections:
            is_duplicate = False

            for existing in unique_detections:
                # Calculate IoU overlap
                iou = self._calculate_iou(detection.bbox, existing.bbox)

                # Calculate text similarity
                text_sim = self._calculate_text_similarity(detection.text, existing.text)

                # Consider duplicate if high overlap atau same text
                if iou > 0.5 or text_sim > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_detections.append(detection)

        return unique_detections

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
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

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        if not text1 or not text2:
            return 0.0

        import re
        text1_clean = re.sub(r'[^A-Z0-9]', '', text1.upper())
        text2_clean = re.sub(r'[^A-Z0-9]', '', text2.upper())

        if text1_clean == text2_clean:
            return 1.0

        # Simple character-level similarity
        max_len = max(len(text1_clean), len(text2_clean))
        if max_len == 0:
            return 1.0

        common_chars = sum(1 for c1, c2 in zip(text1_clean, text2_clean) if c1 == c2)
        return common_chars / max_len

    def draw_detections(self, frame: np.ndarray,
                       detections: List[EnhancedDetectionResult]) -> np.ndarray:
        """Draw enhanced detections pada frame"""
        annotated_frame = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Color based on detection method
            method_colors = {
                'plate_yolo': (0, 255, 0),    # Green
                'ocr': (255, 165, 0),         # Orange
                'hybrid': (0, 255, 255)       # Cyan
            }
            color = method_colors.get(detection.detection_method, (255, 255, 255))

            # Draw bounding box dengan thickness based on confidence
            thickness = max(1, int(detection.confidence * 4))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw vehicle region jika ada
            if detection.vehicle_bbox:
                vx, vy, vw, vh = detection.vehicle_bbox
                cv2.rectangle(annotated_frame, (vx, vy), (vx + vw, vy + vh), (128, 128, 128), 1)

            # Label dengan method info
            label = f'{detection.text} ({detection.detection_method}:{detection.confidence:.2f})'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Background rectangle
            cv2.rectangle(annotated_frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)

            # Text
            cv2.putText(annotated_frame, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 2)

        return annotated_frame

    def get_statistics(self) -> Dict:
        """Get comprehensive detection statistics"""
        avg_detection_time = np.mean(self.stats['detection_times']) if self.stats['detection_times'] else 0
        total = self.stats['total_detections']

        return {
            'enhanced_hybrid_detector': {
                'total_detections': total,
                'plate_yolo_detections': self.stats['plate_yolo_detections'],
                'ocr_detections': self.stats['ocr_detections'],
                'hybrid_detections': self.stats['hybrid_detections'],
                'avg_detection_time': avg_detection_time,
                'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
                'plate_yolo_rate': (self.stats['plate_yolo_detections'] / total * 100) if total > 0 else 0,
                'ocr_fallback_rate': (self.stats['ocr_detections'] / total * 100) if total > 0 else 0
            },
            'vehicle_detector': self.vehicle_detector.get_statistics() if self.vehicle_detector else {},
            'plate_detector': self.plate_detector.get_statistics() if self.plate_detector else {}
        }

    def reset_statistics(self):
        """Reset all detection statistics"""
        self.stats = {
            'plate_yolo_detections': 0,
            'ocr_detections': 0,
            'hybrid_detections': 0,
            'total_detections': 0,
            'detection_times': []
        }

        if self.vehicle_detector:
            self.vehicle_detector.reset_statistics()
        if self.plate_detector:
            self.plate_detector.reset_statistics()

    def is_enabled(self) -> bool:
        """Check if enhanced hybrid detection is enabled"""
        return (
            (self.vehicle_detector and self.vehicle_detector.is_enabled()) or
            (self.plate_detector and self.plate_detector.is_enabled()) or
            self.ocr_enabled
        )

def create_enhanced_hybrid_detector(config: Dict) -> EnhancedHybridDetector:
    """Factory function untuk create enhanced hybrid detector"""
    return EnhancedHybridDetector(config)