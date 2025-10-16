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

# Import components - STREAMLINED untuk pure plate detection
from utils.robust_plate_detector import RobustPlateDetector  # Main plate detector
from utils.ocr_ensemble import OCREnsemble  # OCR processing

@dataclass
class PlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    angle: float = 0.0
    processed_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    vehicle_type: str = "unknown"  # Deprecated - no longer used
    detection_method: str = "pure_plate"

class HybridPlateDetector:
    """
    Pure Plate Detector - Direct plate detection without vehicle detection
    Fokus langsung ke deteksi plat nomor tanpa overhead vehicle detection
    """
    
    def __init__(self, streaming_mode=True):
        """
        Initialize pure plate detector
        """
        self.streaming_mode = streaming_mode
        self.logger = logging.getLogger(__name__)
        
        # VEHICLE DETECTION DISABLED FOR STABILITY - PURE PLATE DETECTION MODE
        self.yolo_detector = None
        self.yolo_enabled = False  # PERMANENTLY DISABLED untuk maximum stability
        self.logger.info("🎯 Vehicle detection DISABLED - Using pure plate detection for stability")
        
        # Initialize plate detector dengan enhanced settings
        self.plate_detector = RobustPlateDetector(streaming_mode=True)
        
        # Initialize enhanced OCR - ULTRA-ENHANCED MODE with exposure bracketing
        try:
            self.ocr_ensemble = OCREnsemble()
            self.enhanced_ocr_enabled = True  # ENABLED for challenging CCTV conditions
            self.use_exposure_bracketing = True  # Enable exposure bracketing for distant/reflective plates
            self.ocr_cache = {}  # Simple cache for repeated OCR results
            self.cache_hits = 0
            self.cache_misses = 0
            self.logger.info("✅ ULTRA-Enhanced OCR ensemble with exposure bracketing initialized")
        except Exception as e:
            self.ocr_ensemble = None
            self.enhanced_ocr_enabled = False
            self.use_exposure_bracketing = False
            self.ocr_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            self.logger.warning(f"Enhanced OCR not available: {e}")
        
        # Stability enhancements - ULTRA-SENSITIVE for distant/challenging plates
        self.detection_history = []  # Track recent detections
        self.stability_threshold = 0.18  # Ultra-sensitive threshold (18%) untuk maximum detection (was 0.22)
        
        # Statistics
        self.total_detections = 0
        self.successful_ocr = 0
        self.failed_ocr = 0
        
        self.logger.info("🔧 Pure Plate Detector initialized (OpenCV only - STABLE MODE)")
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Pure plate detection: Focus only on license plates without vehicle detection
        """
        detections = []
        start_time = time.time()

        try:
            # Direct plate detection on full image - no YOLO vehicle detection
            self.logger.info("🎯 Using direct plate detection (no vehicle detection)")
            detections = self._fallback_full_detection(image)

            # Post-process detections
            detections = self._post_process_detections(detections)

            detection_time = time.time() - start_time
            self.logger.info(f"🎯 Pure plate detection: {len(detections)} plates in {detection_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Error in pure plate detection: {e}")

        return detections
    
    
    def _fallback_full_detection(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Pure plate detection: Full image plate detection without vehicle guidance - CACHED
        """
        self.logger.debug("🔄 Using pure plate detection on full image (cached)")

        # Simple image hash for caching
        try:
            if isinstance(image, np.ndarray):
                image_hash = hash(image.tobytes())
            else:
                # Fallback: disable caching for non-numpy inputs
                image_hash = None
        except Exception:
            image_hash = None

        # Clear cache for stability - fresh detection to avoid false positive accumulation
        self.ocr_cache.clear()  # Clear cache untuk prevent false positive accumulation
        self.cache_misses += 1
        # Use smart ROI detection - motorcycle is common untuk CCTV
        detections = self.plate_detector.detect_plates(image, vehicle_type='motorcycle')

        # Convert to plate detection format
        pure_detections = []
        for detection in detections:
            pure_detection = PlateDetection(
                text=detection.text,
                confidence=detection.confidence,  # No penalty since this is now primary method
                bbox=detection.bbox,
                angle=detection.angle,
                detection_method="pure_plate_detection"
            )
            pure_detections.append(pure_detection)

            # Update stats
            self.total_detections += 1
            if detection.text and len(detection.text) >= 2:  # Adjusted for new minimum
                self.successful_ocr += 1
            else:
                self.failed_ocr += 1

        # Cache results (limit cache size to prevent memory issues)
        if image_hash is not None and len(self.ocr_cache) < 50:  # Max 50 cached results
            self.ocr_cache[image_hash] = pure_detections

        return pure_detections
    
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
    
    def _calculate_stability_bonus(self, detection) -> float:
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
            
            # Text validation - STRICT minimum length
            if not detection.text or len(detection.text) < 2:  # Minimum 2 chars
                return False

            # Remove obvious false positives and validate Indonesian plate format
            text = detection.text.upper().strip()

            # Character diversity check - REJECT repeated characters
            clean_text_for_diversity = text.replace(' ', '')
            unique_chars = len(set(clean_text_for_diversity))
            text_length = len(clean_text_for_diversity)

            # Reject if all characters are the same (e.g., "11", "AA", "222", "BBB")
            if text_length >= 2 and unique_chars < 2:
                return False  # REJECT: "11", "AA", "222", "BBB", "1111"

            # Extensive false positive patterns - STRICT filtering
            false_positive_patterns = [
                # Original patterns
                'WWW', 'HTTP', 'COM', '...', '___', '???',
                # Repeated numbers (3+ digits)
                '111', '222', '333', '444', '555', '666', '777', '888', '999', '000',
                # Repeated letters (3+ chars)
                'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG', 'HHH', 'III',
                'JJJ', 'KKK', 'LLL', 'MMM', 'NNN', 'OOO', 'PPP', 'QQQ', 'RRR',
                'SSS', 'TTT', 'UUU', 'VVV', 'WWW', 'XXX', 'YYY', 'ZZZ',
                # Common words
                'TEST', 'STOP', 'EXIT', 'MENU', 'INFO', 'HELP', 'OPEN', 'CLOSE',
                'PUSH', 'PULL', 'WAIT', 'NEXT', 'PREV', 'YES', 'NO', 'OK',
                'BACK', 'HOME', 'SAVE', 'LOAD', 'NEW', 'OLD', 'COPY', 'PASTE',
                'CLEAR', 'RESET', 'START', 'END', 'UP', 'DOWN', 'LEFT', 'RIGHT',
                # Noise patterns
                '----', '====', '||||', '////', '\\\\\\\\', '....', '::::',
                # Single confusing characters (already caught by diversity check, but double-check)
                'I', 'O', 'L', 'S', 'Z', 'Q'
            ]
            if any(pattern in text for pattern in false_positive_patterns):
                return False

            # Indonesian plate format validation - use scoring instead of hard rejection
            plate_format_score = self._score_indonesian_plate_format(text)

            # Allow detection if it has reasonable format score OR high confidence - ULTRA-SENSITIVE
            if plate_format_score < 0.25 and confidence < 28:  # Ultra-sensitive thresholds (was 0.30/35)
                return False
            
            # Bbox validation
            x, y, w, h = detection.bbox

            # Size validation - ULTRA-SENSITIVE for very distant plates
            if w < 25 or h < 8:  # Ultra-sensitive size (was 30x10) for maximum detection range
                return False

            if w > 450 or h > 180:  # Slightly relaxed from 400x150 to 450x180 untuk edge cases
                return False

            # Aspect ratio validation - FOCUSED for plate shapes
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 1.5 or aspect_ratio > 5.0:  # Focused range untuk reduce noise
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Validation error: {e}")
            return False
    
    def _enhance_detections_with_ocr(self, detections: List[PlateDetection], roi: np.ndarray) -> List[PlateDetection]:
        """Enhance detections dengan advanced OCR - CACHED VERSION"""
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
                    # Check OCR cache first
                    roi_hash = hash(plate_roi.tobytes())

                    if roi_hash in self.ocr_cache:
                        cached_result = self.ocr_cache[roi_hash]
                        detection.text = cached_result['text']
                        detection.confidence = cached_result['confidence']
                        self.cache_hits += 1
                    else:
                        # Apply ULTRA-ENHANCED OCR with exposure bracketing
                        enhanced_text, enhanced_confidence, ocr_details = self.ocr_ensemble.ensemble_ocr(
                            plate_roi,
                            methods=['cctv_block', 'single_line', 'single_word'],  # Multiple methods for accuracy
                            use_exposure_bracketing=self.use_exposure_bracketing  # Enable exposure variants
                        )

                        # Use enhanced result jika lebih baik
                        if enhanced_text and len(enhanced_text) >= len(detection.text):
                            detection.text = enhanced_text
                            # Combine confidence dengan weighted average
                            detection.confidence = (detection.confidence * 0.6 + enhanced_confidence * 0.4)

                        # Cache result
                        if len(self.ocr_cache) < 100:  # Limit cache size
                            self.ocr_cache[roi_hash] = {
                                'text': detection.text,
                                'confidence': detection.confidence
                            }
                        self.cache_misses += 1

                enhanced_detections.append(detection)

        except Exception as e:
            self.logger.warning(f"OCR enhancement error: {e}")
            return detections  # Return original jika error

        return enhanced_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[PlateDetection],
                       show_roi: bool = True) -> np.ndarray:
        """
        Draw plate detections dengan styling yang distinctive
        """
        result = frame.copy()

        # Sort detections by confidence untuk prioritas visual
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        for i, detection in enumerate(sorted_detections):
            x, y, w, h = detection.bbox

            # PLATE BOUNDING BOX - Clean and focused
            if i == 0:  # Best detection - green
                plate_color = (0, 255, 0)  # GREEN untuk best detection
                thickness = 4
            else:  # Other detections - cyan
                plate_color = (255, 255, 0)  # CYAN untuk other detections
                thickness = 3

            # Double border untuk clarity
            # Border luar (hitam)
            cv2.rectangle(result, (x-2, y-2), (x + w + 2, y + h + 2), (0, 0, 0), thickness+1)
            # Border dalam (plate color)
            cv2.rectangle(result, (x, y), (x + w, y + h), plate_color, thickness)

            # Corner markers
            corner_size = 12
            corner_thickness = 2

            # Simple corner markers
            # Top-left
            cv2.line(result, (x, y), (x + corner_size, y), plate_color, corner_thickness)
            cv2.line(result, (x, y), (x, y + corner_size), plate_color, corner_thickness)
            # Top-right
            cv2.line(result, (x + w, y), (x + w - corner_size, y), plate_color, corner_thickness)
            cv2.line(result, (x + w, y), (x + w, y + corner_size), plate_color, corner_thickness)
            # Bottom corners
            cv2.line(result, (x, y + h), (x + corner_size, y + h), plate_color, corner_thickness)
            cv2.line(result, (x, y + h), (x, y + h - corner_size), plate_color, corner_thickness)
            cv2.line(result, (x + w, y + h), (x + w - corner_size, y + h), plate_color, corner_thickness)
            cv2.line(result, (x + w, y + h), (x + w, y + h - corner_size), plate_color, corner_thickness)

            # PLATE LABEL - clean and focused
            if detection.text:
                if i == 0:
                    label = f"🎯 PLATE: {detection.text} ({detection.confidence:.0f}%)"
                    font_scale = 0.7
                else:
                    label = f"PLATE: {detection.text} ({detection.confidence:.0f}%)"
                    font_scale = 0.6

                font = cv2.FONT_HERSHEY_DUPLEX
                font_thickness = 2

                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                # Background dengan clean styling
                bg_padding = 6
                bg_x1 = x - bg_padding
                bg_y1 = y - text_h - 18 - bg_padding
                bg_x2 = x + text_w + bg_padding
                bg_y2 = y - 5 + bg_padding

                # Double background (black -> color)
                cv2.rectangle(result, (bg_x1-2, bg_y1-2), (bg_x2+2, bg_y2+2), (0, 0, 0), -1)
                cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), plate_color, -1)

                # Text dengan shadow effect
                # Shadow
                cv2.putText(result, label, (x+1, y - 7), font, font_scale, (0, 0, 0), font_thickness+1)
                # Main text
                cv2.putText(result, label, (x, y - 8), font, font_scale, (255, 255, 255), font_thickness)

        return result
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get pure plate detection statistics
        """
        success_rate = (self.successful_ocr / self.total_detections * 100) if self.total_detections > 0 else 0

        stats = {
            "total_detections": self.total_detections,
            "successful_ocr": self.successful_ocr,
            "failed_ocr": self.failed_ocr,
            "success_rate": round(success_rate, 1),
            "detection_method": "PURE PLATE (OpenCV only)",
            "yolo_enabled": False
        }

        return stats


    def _score_indonesian_plate_format(self, text: str) -> float:
        """
        Score Indonesian license plate format (0.0 to 1.0)
        Returns score instead of boolean for hybrid validation
        """
        import re

        if not text or len(text) < 2:
            return 0.0

        score = 0.0
        clean_text = text.replace(' ', '')

        # Length scoring - STRICT to eliminate short noise
        if 3 <= len(clean_text) <= 9:  # Minimum 3 chars for full score
            score += 0.4  # Full score for 3-9 characters
        elif len(clean_text) == 2:  # 2 chars get reduced score
            score += 0.2  # Reduced from 0.4 to 0.2 for 2-char fragments
        elif len(clean_text) == 1:  # Single char heavily penalized
            score += 0.1  # Reduced from 0.3 to 0.1 for single characters

        # Must contain at least one letter and one number
        has_letter = any(c.isalpha() for c in clean_text)
        has_number = any(c.isdigit() for c in clean_text)

        # STRICT letter/number scoring - Prefer mixed
        if has_letter and has_number:
            score += 0.5  # Complete fragments (mixed) - BEST
        elif has_number:
            score += 0.35  # Number-only: Reduced from 0.40
        elif has_letter:
            score += 0.25  # Letter-only: Reduced from 0.30

        # Penalize obvious false positives - RELAXED for fragment detection
        if len(set(clean_text)) <= 1:  # Only penalize single repeated character
            score -= 0.3  # Reduced penalty from 0.4 to 0.3

        # Penalty only for letter-only strings (allow numbers)
        if len(clean_text) > 3 and clean_text.isalpha():
            score -= 0.2  # Penalty hanya untuk huruf saja, NOT angka

        # Perfect pattern match bonus - WITH NUMBER-ONLY SUPPORT
        patterns = [
            (r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$', 0.4),  # Perfect format: B1234XYZ
            (r'^[A-Z]{1,2}\d{1,4}$', 0.3),            # Partial: B1234
            (r'^\d{1,4}[A-Z]{1,3}$', 0.3),            # Partial: 1234XYZ
            (r'^\d{2,4}$', 0.3),                      # Number-only: 2847, 123, 45
            (r'^[A-Z]\d{1,4}[A-Z]$', 0.25),           # Minimum: B1X
        ]

        # Check against patterns and add highest score
        max_pattern_score = 0.0
        for pattern, pattern_score in patterns:
            if re.match(pattern, clean_text):
                max_pattern_score = max(max_pattern_score, pattern_score)

        score += max_pattern_score

        # Heavy penalty for short non-pattern text (2-3 chars without pattern match)
        if len(clean_text) <= 3 and max_pattern_score == 0:
            score -= 0.3  # Penalize short text that doesn't match any pattern

        # Regional code bonus
        region_codes = [
            'A', 'AA', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N',
            'P', 'R', 'S', 'T', 'W', 'Z', 'AD', 'AE', 'AG', 'BA', 'BB',
            'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'BP', 'BR',
            'BT', 'CC', 'CD', 'CE', 'CG', 'DA', 'DB', 'DD', 'DE', 'DG',
            'DH', 'DK', 'DL', 'DM', 'DN', 'DP', 'DR', 'DS', 'DT'
        ]

        for code in region_codes:
            if clean_text.startswith(code) and len(clean_text) > len(code):
                remaining = clean_text[len(code):]
                if remaining and remaining[0].isdigit():
                    score += 0.2
                    break

        return min(1.0, max(0.0, score))

if __name__ == "__main__":
    # Test pure plate detector
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)

        if image is not None:
            detector = HybridPlateDetector()
            detections = detector.detect_plates(image)

            print(f"🎯 PURE PLATE detected {len(detections)} license plates:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%) via {det.detection_method}")

            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("contoh/hybrid_plate_result.jpg", result)
            print("💾 Result saved: contoh/hybrid_plate_result.jpg")
            
            # Show statistics
            stats = detector.get_statistics()
            print(f"📊 Statistics: {stats}")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python hybrid_plate_detector.py <image_path>")