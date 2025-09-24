"""
License Plate YOLO Detector untuk Indonesian License Plates
Integrasi YOLO khusus deteksi plat nomor dengan sistem existing
"""

import cv2
import numpy as np
import logging
import time
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Check YOLOv8 availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

@dataclass
class LicensePlateDetection:
    """Data class untuk hasil deteksi license plate"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    vehicle_type: str = "unknown"
    plate_type: str = "indonesian"
    is_valid_format: bool = False

class LicensePlateYOLODetector:
    """
    YOLO Detector khusus untuk Indonesian License Plates
    """

    def __init__(self, model_path='license_plate_yolo.pt', confidence=0.3, iou_threshold=0.4):
        """
        Initialize License Plate YOLO detector

        Args:
            model_path: Path to trained license plate YOLO model
            confidence: Confidence threshold (lower untuk plate detection)
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model = None
        self.enabled = False
        self.logger = logging.getLogger(__name__)

        # Indonesian license plate patterns
        self.indonesian_patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',  # B 1234 ABC
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',        # B1234ABC
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{2,3}$',  # AA 1234 BB
        ]

        # Vehicle type mapping dari YOLO class results
        self.vehicle_type_mapping = {
            0: 'car_plate',
            1: 'motorcycle_plate',
            2: 'truck_plate',
            3: 'bus_plate',
            4: 'generic_plate'
        }

        # Statistics
        self.total_plate_detections = 0
        self.valid_plates_detected = 0
        self.detection_times = []

        # Indonesian specific optimizations
        self.min_plate_area = 800  # Minimum plate area in pixels
        self.max_plate_area = 15000  # Maximum plate area
        self.aspect_ratio_range = (2.0, 6.0)  # Indonesian plate aspect ratio

        # Initialize model
        self.initialize()

    def initialize(self):
        """Initialize License Plate YOLO model"""
        if not YOLO_AVAILABLE:
            self.logger.warning("YOLOv8 (ultralytics) not available. License plate YOLO disabled.")
            return False

        try:
            self.logger.info(f"Loading License Plate YOLO model: {self.model_path}")

            # Load custom trained model
            self.model = YOLO(self.model_path)
            self.enabled = True

            self.logger.info("✅ License Plate YOLO model loaded successfully")
            return True

        except Exception as e:
            self.logger.warning(f"License Plate YOLO model not found: {str(e)}")
            self.logger.info("System will fallback to OCR-only detection")
            self.enabled = False
            return False

    def is_enabled(self) -> bool:
        """Check if License Plate YOLO is enabled"""
        return self.enabled and self.model is not None

    def detect_license_plates(self, frame: np.ndarray,
                            vehicle_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> List[LicensePlateDetection]:
        """
        Detect license plates in frame

        Args:
            frame: Input frame
            vehicle_regions: Optional vehicle regions to focus on

        Returns:
            List of LicensePlateDetection objects
        """
        if not self.is_enabled():
            return []

        start_time = time.time()
        detections = []

        try:
            # If vehicle regions provided, process each region
            if vehicle_regions:
                detections = self._detect_in_vehicle_regions(frame, vehicle_regions)
            else:
                # Full frame detection
                detections = self._detect_full_frame(frame)

            # Update statistics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_plate_detections += len(detections)
            self.valid_plates_detected += sum(1 for d in detections if d.is_valid_format)

            # Keep last 100 detection times
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]

            return detections

        except Exception as e:
            self.logger.error(f"Error in license plate detection: {str(e)}")
            return []

    def _detect_full_frame(self, frame: np.ndarray) -> List[LicensePlateDetection]:
        """Detect license plates in full frame"""
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False
        )

        return self._process_yolo_results(results, frame)

    def _detect_in_vehicle_regions(self, frame: np.ndarray,
                                 vehicle_regions: List[Tuple[int, int, int, int]]) -> List[LicensePlateDetection]:
        """Detect license plates within vehicle regions"""
        all_detections = []

        for vehicle_bbox in vehicle_regions:
            x, y, w, h = vehicle_bbox

            # Extract vehicle region
            vehicle_roi = frame[y:y+h, x:x+w]

            if vehicle_roi.size == 0:
                continue

            # Run YOLO on vehicle region
            results = self.model(
                vehicle_roi,
                conf=self.confidence * 0.8,  # Slightly lower confidence in vehicle regions
                iou=self.iou_threshold,
                verbose=False
            )

            # Process results and adjust coordinates to full frame
            region_detections = self._process_yolo_results(results, vehicle_roi, offset=(x, y))
            all_detections.extend(region_detections)

        # Remove duplicate detections
        return self._remove_duplicate_plates(all_detections)

    def _process_yolo_results(self, results, image: np.ndarray,
                            offset: Tuple[int, int] = (0, 0)) -> List[LicensePlateDetection]:
        """Process YOLO detection results"""
        detections = []
        offset_x, offset_y = offset

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Extract detection info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0

                # Adjust coordinates with offset
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y

                # Convert to width/height format
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # Validate plate detection
                if not self._validate_plate_detection(bbox, image.shape):
                    continue

                # Extract plate region for text recognition
                plate_roi = image[y1-offset_y:y2-offset_y, x1-offset_x:x2-offset_x]

                # Get plate text (from trained model or OCR fallback)
                plate_text = self._extract_plate_text(plate_roi)

                # Determine vehicle type
                vehicle_type = self.vehicle_type_mapping.get(class_id, 'unknown')

                # Validate Indonesian format
                is_valid = self._validate_indonesian_format(plate_text)

                detection = LicensePlateDetection(
                    text=plate_text,
                    confidence=confidence,
                    bbox=bbox,
                    vehicle_type=vehicle_type,
                    plate_type='indonesian',
                    is_valid_format=is_valid
                )

                detections.append(detection)

        return detections

    def _validate_plate_detection(self, bbox: Tuple[int, int, int, int],
                                frame_shape: Tuple[int, int, int]) -> bool:
        """Validate plate detection based on size and aspect ratio"""
        x, y, w, h = bbox

        # Check area
        area = w * h
        if area < self.min_plate_area or area > self.max_plate_area:
            return False

        # Check aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            return False

        # Check if within frame bounds
        frame_h, frame_w = frame_shape[:2]
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False

        return True

    def _extract_plate_text(self, plate_roi: np.ndarray) -> str:
        """Extract text from plate ROI"""
        if plate_roi.size == 0:
            return ""

        try:
            # If model has text detection capability, use it
            # Otherwise fallback to basic OCR

            # Enhanced preprocessing untuk better OCR
            gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) if len(plate_roi.shape) == 3 else plate_roi

            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Noise reduction
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Basic character recognition (placeholder)
            # In real implementation, this would use the trained YOLO model's text output
            # or integrate with Tesseract OCR

            return self._simulate_plate_text_extraction(cleaned)

        except Exception as e:
            self.logger.debug(f"Error extracting plate text: {e}")
            return ""

    def _simulate_plate_text_extraction(self, processed_roi: np.ndarray) -> str:
        """Simulate plate text extraction (placeholder for actual implementation)"""
        # This is a placeholder - in real implementation:
        # 1. Use trained YOLO model with text recognition
        # 2. Or integrate with Tesseract OCR
        # 3. Or use specialized OCR for license plates

        # For now, return a simulated Indonesian plate
        import random
        prefixes = ['B', 'D', 'F', 'AA', 'AB']
        numbers = [f"{random.randint(1000, 9999)}"]
        suffixes = ['ABC', 'XYZ', 'DEF', 'GHI']

        return f"{random.choice(prefixes)} {random.choice(numbers)} {random.choice(suffixes)}"

    def _validate_indonesian_format(self, text: str) -> bool:
        """Validate if text matches Indonesian license plate format"""
        if not text or len(text) < 5:
            return False

        # Clean text
        clean_text = re.sub(r'[^A-Z0-9\s]', '', text.upper().strip())

        # Check against Indonesian patterns
        for pattern in self.indonesian_patterns:
            if re.match(pattern, clean_text):
                return True

        return False

    def _remove_duplicate_plates(self, detections: List[LicensePlateDetection]) -> List[LicensePlateDetection]:
        """Remove duplicate plate detections"""
        if len(detections) <= 1:
            return detections

        unique_detections = []

        for detection in detections:
            is_duplicate = False

            for existing in unique_detections:
                # Calculate overlap
                overlap = self._calculate_bbox_overlap(detection.bbox, existing.bbox)

                # Check text similarity
                text_similarity = self._calculate_text_similarity(detection.text, existing.text)

                # Consider duplicate if high overlap or same text
                if overlap > 0.5 or text_similarity > 0.8:
                    is_duplicate = True
                    # Keep higher confidence detection
                    if detection.confidence > existing.confidence:
                        unique_detections.remove(existing)
                        unique_detections.append(detection)
                    break

            if not is_duplicate:
                unique_detections.append(detection)

        return unique_detections

    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int],
                              bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU overlap between two bounding boxes"""
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
        """Calculate text similarity between two plate texts"""
        if not text1 or not text2:
            return 0.0

        # Simple character similarity
        text1_clean = re.sub(r'[^A-Z0-9]', '', text1.upper())
        text2_clean = re.sub(r'[^A-Z0-9]', '', text2.upper())

        if text1_clean == text2_clean:
            return 1.0

        # Levenshtein distance approximation
        max_len = max(len(text1_clean), len(text2_clean))
        if max_len == 0:
            return 1.0

        common_chars = sum(1 for c1, c2 in zip(text1_clean, text2_clean) if c1 == c2)
        return common_chars / max_len

    def draw_detections(self, frame: np.ndarray, detections: List[LicensePlateDetection]) -> np.ndarray:
        """Draw license plate detections on frame"""
        annotated_frame = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Choose color based on validity
            color = (0, 255, 0) if detection.is_valid_format else (0, 165, 255)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f'{detection.text} ({detection.confidence:.2f})'
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
        """Get detection statistics"""
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        validity_rate = (self.valid_plates_detected / self.total_plate_detections * 100) if self.total_plate_detections > 0 else 0

        return {
            'enabled': self.enabled,
            'total_plate_detections': self.total_plate_detections,
            'valid_plates_detected': self.valid_plates_detected,
            'validity_rate_percent': validity_rate,
            'avg_detection_time': avg_detection_time,
            'detection_fps': 1.0 / avg_detection_time if avg_detection_time > 0 else 0,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence,
            'supported_formats': ['indonesian']
        }

    def set_confidence(self, confidence: float):
        """Set confidence threshold"""
        self.confidence = max(0.0, min(1.0, confidence))
        self.logger.info(f"License plate confidence threshold set to: {self.confidence}")

    def reset_statistics(self):
        """Reset detection statistics"""
        self.total_plate_detections = 0
        self.valid_plates_detected = 0
        self.detection_times = []

def create_license_plate_detector(model_path='license_plate_yolo.pt',
                                confidence=0.3,
                                iou_threshold=0.4) -> Optional[LicensePlateYOLODetector]:
    """
    Factory function untuk create License Plate YOLO detector

    Args:
        model_path: Path to trained model
        confidence: Confidence threshold
        iou_threshold: IoU threshold

    Returns:
        LicensePlateYOLODetector instance atau None jika gagal
    """
    try:
        detector = LicensePlateYOLODetector(model_path, confidence, iou_threshold)
        if detector.is_enabled():
            return detector
        else:
            return None
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create License Plate YOLO detector: {str(e)}")
        return None