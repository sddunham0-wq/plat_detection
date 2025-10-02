#!/usr/bin/env python3
"""
Stable Unified Plate Detector
Unified detector yang menggabungkan YOLO vehicle detection + plate extraction
dengan multi-frame validation untuk stabilitas maksimal
"""

import cv2
import numpy as np
import logging
import time
import threading
from typing import List, Tuple, Optional, Dict, Deque
from dataclasses import dataclass
from collections import deque, defaultdict
import re

# Import optimized components
try:
    from config import (
        OptimizedYOLOConfig, StablePlateDetectionConfig,
        TesseractConfig, IndonesianPlateConfig, DetectionConfig
    )
except ImportError:
    print("⚠️  Config classes not found, using default values")

@dataclass
class StablePlateDetection:
    """Enhanced plate detection dengan stability metrics"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    vehicle_bbox: Optional[Tuple[int, int, int, int]] = None
    vehicle_type: str = "unknown"
    stability_score: float = 0.0
    frame_count: int = 1
    temporal_confidence: float = 0.0
    spatial_consistency: float = 1.0
    timestamp: float = 0.0
    detection_id: str = ""

class MultiFrameValidator:
    """
    Multi-frame validation system untuk memastikan konsistensi deteksi
    """

    def __init__(self, validation_frames=3, max_spatial_drift=100):
        self.validation_frames = validation_frames
        self.max_spatial_drift = max_spatial_drift
        self.detection_buffer = deque(maxlen=validation_frames)
        self.confirmed_detections = {}
        self.detection_counter = 0

    def add_detection(self, detection: StablePlateDetection) -> Optional[StablePlateDetection]:
        """
        Add detection untuk validation dan return confirmed detection jika valid
        """
        self.detection_buffer.append(detection)

        if len(self.detection_buffer) < self.validation_frames:
            return None

        # Check consistency across frames
        consistent_detection = self._validate_consistency()
        if consistent_detection:
            consistent_detection.stability_score = self._calculate_stability_score()
            consistent_detection.detection_id = f"stable_{self.detection_counter}"
            self.detection_counter += 1
            return consistent_detection

        return None

    def _validate_consistency(self) -> Optional[StablePlateDetection]:
        """Validate consistency across buffered detections"""
        if not self.detection_buffer:
            return None

        # Check text consistency (majority vote)
        text_votes = defaultdict(int)
        for det in self.detection_buffer:
            text_votes[det.text] += 1

        # Get most common text
        best_text = max(text_votes.items(), key=lambda x: x[1])[0]

        # Check if majority agrees (at least 2/3)
        if text_votes[best_text] < max(2, len(self.detection_buffer) * 0.6):
            return None

        # Check spatial consistency
        positions = [(det.bbox[0], det.bbox[1]) for det in self.detection_buffer if det.text == best_text]
        if not self._check_spatial_consistency(positions):
            return None

        # Create consolidated detection
        valid_detections = [det for det in self.detection_buffer if det.text == best_text]
        return self._consolidate_detections(valid_detections, best_text)

    def _check_spatial_consistency(self, positions: List[Tuple[int, int]]) -> bool:
        """Check if positions are spatially consistent"""
        if len(positions) < 2:
            return True

        for i in range(1, len(positions)):
            drift = np.sqrt((positions[i][0] - positions[i-1][0])**2 +
                           (positions[i][1] - positions[i-1][1])**2)
            if drift > self.max_spatial_drift:
                return False
        return True

    def _consolidate_detections(self, detections: List[StablePlateDetection], text: str) -> StablePlateDetection:
        """Consolidate multiple detections into single stable detection"""
        # Average confidence
        avg_confidence = sum(det.confidence for det in detections) / len(detections)

        # Average bbox
        avg_x = int(sum(det.bbox[0] for det in detections) / len(detections))
        avg_y = int(sum(det.bbox[1] for det in detections) / len(detections))
        avg_w = int(sum(det.bbox[2] for det in detections) / len(detections))
        avg_h = int(sum(det.bbox[3] for det in detections) / len(detections))

        return StablePlateDetection(
            text=text,
            confidence=avg_confidence,
            bbox=(avg_x, avg_y, avg_w, avg_h),
            vehicle_bbox=detections[-1].vehicle_bbox,
            vehicle_type=detections[-1].vehicle_type,
            frame_count=len(detections),
            timestamp=time.time()
        )

    def _calculate_stability_score(self) -> float:
        """Calculate stability score based pada consistency metrics"""
        if len(self.detection_buffer) < 2:
            return 0.5

        # Text consistency score
        texts = [det.text for det in self.detection_buffer]
        unique_texts = len(set(texts))
        text_consistency = 1.0 - (unique_texts - 1) / len(texts)

        # Spatial consistency score
        positions = [(det.bbox[0], det.bbox[1]) for det in self.detection_buffer]
        spatial_variance = np.var([pos[0] for pos in positions]) + np.var([pos[1] for pos in positions])
        spatial_consistency = max(0.0, 1.0 - spatial_variance / 10000)  # Normalize

        return (text_consistency * 0.7 + spatial_consistency * 0.3)

class StableUnifiedDetector:
    """
    Unified detector yang stable dan responsif untuk plate detection
    """

    def __init__(self, enable_yolo=True, streaming_mode=True):
        """Initialize stable unified detector"""
        self.logger = logging.getLogger(__name__)
        self.streaming_mode = streaming_mode
        self.enable_yolo = enable_yolo

        # Initialize YOLO detector untuk vehicle detection
        self.yolo_detector = None
        if enable_yolo:
            try:
                from utils.yolo_detector import YOLOObjectDetector
                self.yolo_detector = YOLOObjectDetector(
                    model_path=getattr(OptimizedYOLOConfig, 'MODEL_PATH', 'yolov8s.pt'),
                    confidence=getattr(OptimizedYOLOConfig, 'CONFIDENCE_THRESHOLD', 0.65),
                    iou_threshold=getattr(OptimizedYOLOConfig, 'IOU_THRESHOLD', 0.45),
                    max_detections=getattr(OptimizedYOLOConfig, 'MAX_DETECTIONS', 8)
                )
                self.logger.info("✅ Optimized YOLO detector initialized")
            except Exception as e:
                self.logger.warning(f"YOLO not available: {e}")
                self.yolo_detector = None

        # Initialize multi-frame validator
        validation_frames = getattr(StablePlateDetectionConfig, 'VALIDATION_FRAMES', 3)
        max_drift = getattr(StablePlateDetectionConfig, 'MAX_SPATIAL_DRIFT', 100)
        self.validator = MultiFrameValidator(validation_frames, max_drift)

        # Performance monitoring
        self.frame_count = 0
        self.detection_count = 0
        self.stable_detection_count = 0
        self.last_detection_time = 0

        # OCR optimization
        try:
            import pytesseract
            self.tesseract_available = True
            # Set path jika diperlukan
            tesseract_path = getattr(TesseractConfig, 'TESSERACT_PATH', None)
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
        except ImportError:
            self.tesseract_available = False
            self.logger.warning("Tesseract not available")

        self.logger.info("🎯 Stable Unified Detector initialized")

    def detect_plates(self, image: np.ndarray) -> List[StablePlateDetection]:
        """
        Main detection method dengan stability optimization
        """
        if image is None or image.size == 0:
            return []

        self.frame_count += 1
        start_time = time.time()

        # Skip blurry frames if enabled
        if getattr(StablePlateDetectionConfig, 'SKIP_BLURRY_FRAMES', True):
            if self._is_frame_blurry(image):
                return []

        detections = []

        try:
            # Step 1: Vehicle detection dengan YOLO (optional)
            vehicle_regions = []
            if self.yolo_detector and self.enable_yolo:
                vehicle_detections = self.yolo_detector.detect_objects(image)
                vehicle_regions = self._extract_vehicle_regions(image, vehicle_detections)
            else:
                # Fallback: use full image
                vehicle_regions = [(image, (0, 0, image.shape[1], image.shape[0]), "unknown")]

            # Step 2: Plate detection dalam vehicle regions
            for vehicle_image, vehicle_bbox, vehicle_type in vehicle_regions:
                plate_candidates = self._detect_plates_in_region(vehicle_image, vehicle_bbox, vehicle_type)
                detections.extend(plate_candidates)

            # Step 3: Multi-frame validation
            stable_detections = []
            for detection in detections:
                validated = self.validator.add_detection(detection)
                if validated:
                    stable_detections.append(validated)
                    self.stable_detection_count += 1

            # Performance monitoring
            processing_time = time.time() - start_time
            if stable_detections:
                self.last_detection_time = processing_time
                self.logger.debug(f"Stable detection found in {processing_time:.3f}s")

            return stable_detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def _is_frame_blurry(self, image: np.ndarray) -> bool:
        """Check if frame is too blurry untuk processing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_threshold = getattr(StablePlateDetectionConfig, 'BLUR_THRESHOLD', 100)
        return laplacian_var < blur_threshold

    def _extract_vehicle_regions(self, image: np.ndarray, vehicle_detections) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], str]]:
        """Extract ROI dari vehicle detections"""
        regions = []

        for detection in vehicle_detections:
            if detection.is_vehicle:
                x, y, w, h = detection.bbox
                # Expand region sedikit untuk include plate area
                expand_factor = 0.1
                expanded_x = max(0, int(x - w * expand_factor))
                expanded_y = max(0, int(y - h * expand_factor))
                expanded_w = min(image.shape[1] - expanded_x, int(w * (1 + 2 * expand_factor)))
                expanded_h = min(image.shape[0] - expanded_y, int(h * (1 + 2 * expand_factor)))

                vehicle_roi = image[expanded_y:expanded_y + expanded_h,
                                  expanded_x:expanded_x + expanded_w]

                if vehicle_roi.size > 0:
                    regions.append((vehicle_roi, (expanded_x, expanded_y, expanded_w, expanded_h), detection.class_name))

        return regions if regions else [(image, (0, 0, image.shape[1], image.shape[0]), "unknown")]

    def _detect_plates_in_region(self, region_image: np.ndarray, region_bbox: Tuple[int, int, int, int],
                                vehicle_type: str) -> List[StablePlateDetection]:
        """Detect plates dalam vehicle region"""
        if region_image is None or region_image.size == 0:
            return []

        detections = []

        # Simple contour-based plate detection
        gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)

        # Adaptive preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection
        edges = cv2.Canny(enhanced, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter contour berdasarkan plate characteristics
            x, y, w, h = cv2.boundingRect(contour)

            # Aspect ratio check
            aspect_ratio = w / h if h > 0 else 0
            if not (1.5 <= aspect_ratio <= 6.0):
                continue

            # Size check
            area = w * h
            if area < 400 or area > 20000:
                continue

            # Extract potential plate region
            plate_roi = region_image[y:y + h, x:x + w]
            if plate_roi.size == 0:
                continue

            # OCR extraction
            plate_text, confidence = self._extract_plate_text(plate_roi)

            if plate_text and confidence >= getattr(TesseractConfig, 'MIN_CONFIDENCE', 60):
                # Validate Indonesian plate pattern
                if self._validate_indonesian_plate(plate_text):
                    # Convert koordinat ke image coordinates
                    global_x = region_bbox[0] + x
                    global_y = region_bbox[1] + y

                    detection = StablePlateDetection(
                        text=plate_text,
                        confidence=confidence,
                        bbox=(global_x, global_y, w, h),
                        vehicle_bbox=region_bbox,
                        vehicle_type=vehicle_type,
                        timestamp=time.time()
                    )
                    detections.append(detection)

        return detections

    def _extract_plate_text(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """Extract text dari plate image dengan OCR optimization"""
        if not self.tesseract_available:
            return "", 0.0

        try:
            import pytesseract

            # Preprocessing untuk OCR
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

            # Resize untuk better OCR
            height = gray.shape[0]
            if height < 40:
                scale = 40 / height
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Noise reduction
            denoised = cv2.medianBlur(gray, 3)

            # Multi-PSM approach untuk stability
            psm_modes = getattr(TesseractConfig, 'PSM_PRIORITY', [7, 8, 13])
            best_text = ""
            best_confidence = 0.0

            for psm in psm_modes:
                config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

                try:
                    data = pytesseract.image_to_data(denoised, config=config, output_type=pytesseract.Output.DICT)

                    # Extract text dan confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    texts = [text.strip() for text in data['text'] if text.strip()]

                    if confidences and texts:
                        avg_confidence = sum(confidences) / len(confidences)
                        combined_text = ''.join(texts)

                        if avg_confidence > best_confidence and len(combined_text) >= 5:
                            best_text = combined_text
                            best_confidence = avg_confidence

                except Exception:
                    continue

            return best_text, best_confidence

        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", 0.0

    def _validate_indonesian_plate(self, text: str) -> bool:
        """Validate Indonesian plate pattern"""
        if not text or len(text) < 5:
            return False

        # Clean text
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Check patterns
        patterns = getattr(IndonesianPlateConfig, 'PLATE_PATTERNS', [
            r'^[ABDEFGHJKLNPRSTU]\d{1,4}[A-Z]{2,3}$',
            r'^\d{1,4}[A-Z]{2,4}$'
        ])

        for pattern in patterns:
            if re.match(pattern, cleaned):
                return True

        return False

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_frames': self.frame_count,
            'total_detections': self.detection_count,
            'stable_detections': self.stable_detection_count,
            'stability_rate': self.stable_detection_count / max(1, self.detection_count),
            'last_detection_time': self.last_detection_time
        }