#!/usr/bin/env python3
"""
Immediate Plate Detector
Detector yang langsung mendeteksi plat tanpa delay dengan stabilitas tinggi
Fokus pada deteksi cepat dan akurat untuk real-time streaming
"""

import cv2
import numpy as np
import pytesseract
import time
import logging
import re
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class ImmediatePlateDetection:
    """Simple plate detection result"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    timestamp: float
    processing_time: float

class ImmediatePlateDetector:
    """
    Detector yang fokus pada deteksi immediate dengan stabilitas tinggi
    - Tidak ada multi-frame validation yang memperlambat
    - Direct OCR dengan preprocessing optimal
    - Threshold tinggi untuk accuracy
    """

    def __init__(self, debug=False):
        """Initialize immediate detector"""
        self.logger = logging.getLogger(__name__)
        self.debug = debug

        # Setup Tesseract
        try:
            # Set Tesseract path jika diperlukan
            tesseract_path = '/opt/homebrew/bin/tesseract'
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.tesseract_available = True
            self.logger.info("✅ Tesseract OCR initialized")
        except Exception as e:
            self.tesseract_available = False
            self.logger.error(f"❌ Tesseract not available: {e}")

        # Detection parameters untuk immediate response
        self.min_confidence = 50  # Balanced confidence untuk immediate detection
        self.min_plate_length = 4  # Slightly lower untuk catch more plates
        self.max_plate_length = 15

        # Geometric constraints untuk Indonesian plates (more permissive)
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 7.0
        self.min_area = 300  # Lower untuk distant plates
        self.max_area = 30000

        # Indonesian plate patterns
        self.indonesian_patterns = [
            r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',  # Standard format
            r'^\d{1,4}\s*[A-Z]{2,4}$',  # Number first
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'  # General format
        ]

        # Character corrections untuk Indonesian OCR
        self.char_corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2',
            '8': 'B', '6': 'G', '0': 'O'  # Reverse corrections
        }

        # Performance tracking
        self.detection_count = 0
        self.success_count = 0
        self.total_processing_time = 0.0

        self.logger.info("🎯 Immediate Plate Detector initialized")

    def detect_plates(self, image: np.ndarray) -> List[ImmediatePlateDetection]:
        """
        Main detection method - immediate response
        """
        if image is None or image.size == 0:
            return []

        start_time = time.time()
        detections = []

        try:
            # Step 1: Find potential plate regions (optimized)
            plate_candidates = self._find_plate_regions(image)

            # Step 2: OCR extraction pada setiap candidate
            for candidate_roi, bbox in plate_candidates:
                plate_text, confidence = self._extract_plate_text_immediate(candidate_roi)

                if self._is_valid_detection(plate_text, confidence):
                    detection = ImmediatePlateDetection(
                        text=plate_text,
                        confidence=confidence,
                        bbox=bbox,
                        timestamp=time.time(),
                        processing_time=time.time() - start_time
                    )
                    detections.append(detection)
                    self.success_count += 1

            self.detection_count += 1
            self.total_processing_time += time.time() - start_time

            return detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def _find_plate_regions(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Find potential plate regions dengan optimasi speed"""
        candidates = []

        # Convert ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive preprocessing untuk berbagai kondisi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection yang optimized
        edges = cv2.Canny(enhanced, 30, 150)

        # Morphological operations untuk connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Quick geometric filters
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue

            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Extract ROI
            plate_roi = image[y:y+h, x:x+w]
            if plate_roi.size == 0:
                continue

            # Basic quality check
            if self._is_roi_quality_good(plate_roi):
                candidates.append((plate_roi, (x, y, w, h)))

        # Sort by area (larger first) dan limit untuk performance
        candidates.sort(key=lambda x: x[1][2] * x[1][3], reverse=True)
        return candidates[:5]  # Top 5 candidates only

    def _is_roi_quality_good(self, roi: np.ndarray) -> bool:
        """Quick quality check untuk ROI"""
        if roi.size == 0:
            return False

        # Check contrast
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        contrast = gray.std()

        # Check edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        return contrast > 20 and edge_density > 0.05

    def _extract_plate_text_immediate(self, plate_roi: np.ndarray) -> Tuple[str, float]:
        """Extract text dengan immediate processing"""
        if not self.tesseract_available:
            return "", 0.0

        try:
            # Preprocessing untuk OCR optimal
            processed_roi = self._preprocess_for_ocr(plate_roi)

            # Multiple OCR attempts dengan different PSM
            best_text = ""
            best_confidence = 0.0

            # PSM modes yang paling efektif untuk Indonesian plates
            psm_modes = [7, 8, 13]  # Line, word, raw line

            for psm in psm_modes:
                try:
                    config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

                    # Get detailed data untuk confidence calculation
                    data = pytesseract.image_to_data(
                        processed_roi,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )

                    # Extract text dan confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    texts = [text.strip() for text in data['text'] if text.strip()]

                    if confidences and texts:
                        avg_confidence = sum(confidences) / len(confidences)
                        combined_text = ''.join(texts).upper()

                        # Apply character corrections
                        corrected_text = self._apply_corrections(combined_text)

                        if avg_confidence > best_confidence and len(corrected_text) >= self.min_plate_length:
                            best_text = corrected_text
                            best_confidence = avg_confidence

                except Exception:
                    continue

            return best_text, best_confidence

        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", 0.0

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Preprocessing optimal untuk OCR"""
        # Convert ke grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Resize jika terlalu kecil untuk OCR
        height, width = gray.shape
        if height < 40:
            scale = 40 / height
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def _apply_corrections(self, text: str) -> str:
        """Apply character corrections untuk Indonesian plates"""
        corrected = text
        for wrong, correct in self.char_corrections.items():
            corrected = corrected.replace(wrong, correct)
        return corrected

    def _is_valid_detection(self, text: str, confidence: float) -> bool:
        """Validate detection untuk immediate response"""
        if not text or confidence < self.min_confidence:
            return False

        if len(text) < self.min_plate_length or len(text) > self.max_plate_length:
            return False

        # Clean text untuk pattern matching
        cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Check Indonesian plate patterns
        for pattern in self.indonesian_patterns:
            if re.match(pattern, cleaned_text):
                return True

        return False

    def get_performance_stats(self) -> dict:
        """Get detector performance statistics"""
        if self.detection_count == 0:
            return {
                'total_detections': 0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0
            }

        return {
            'total_detections': self.detection_count,
            'successful_detections': self.success_count,
            'success_rate': self.success_count / self.detection_count,
            'avg_processing_time': self.total_processing_time / self.detection_count
        }

    def reset_stats(self):
        """Reset performance statistics"""
        self.detection_count = 0
        self.success_count = 0
        self.total_processing_time = 0.0
        self.logger.info("📊 Performance statistics reset")