#!/usr/bin/env python3
"""
Anti-False-Positive Detector
Enhanced plate detector dengan advanced filtering untuk mengurangi false positives
"""

import cv2
import numpy as np
import pytesseract
import logging
import time
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class ValidatedPlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    validation_score: float
    false_positive_score: float
    validation_details: Dict
    detection_method: str = "anti_false_positive"

class AntiFalsePositiveDetector:
    """
    Enhanced detector dengan comprehensive false positive filtering
    """

    def __init__(self):
        """Initialize anti-false-positive detector"""
        self.logger = logging.getLogger(__name__)

        # False positive keywords yang sering muncul di CCTV interface
        self.false_positive_keywords = [
            'HTTP', 'WWW', 'COM', 'NET', 'ORG', 'LIVE', 'CCTV', 'DETECTION',
            'CAMERA', 'VIDEO', 'STREAM', 'RECORD', 'PLAY', 'STOP', 'PAUSE',
            'MENU', 'SETTING', 'CONFIG', 'ADMIN', 'LOGIN', 'PASSWORD',
            'DATE', 'TIME', 'FPS', 'RESOLUTION', 'QUALITY', 'FORMAT',
            'CHANNEL', 'ZONE', 'MOTION', 'ALARM', 'EVENT', 'LOG'
        ]

        # Valid Indonesian regional codes untuk validation
        self.valid_regional_codes = [
            'A', 'AA', 'AB', 'AD', 'AE', 'AG', 'B', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH',
            'BK', 'BL', 'BM', 'BN', 'BP', 'BR', 'BT', 'CC', 'CD', 'CE', 'CG', 'D', 'DA',
            'DB', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DP', 'DR', 'DS', 'DT',
            'E', 'EA', 'EB', 'ED', 'F', 'G', 'H', 'K', 'KB', 'KH', 'KT', 'L', 'M', 'N',
            'P', 'PA', 'PB', 'R', 'S', 'T', 'W', 'Z'
        ]

        # Target search areas - fokus pada area bawah dimana plat nomor biasanya berada
        self.valid_detection_zones = [
            # Bottom 2/3 of image - where vehicle plates typically appear
            {"y_min_ratio": 0.35, "y_max_ratio": 1.0, "name": "vehicle_zone", "priority": 1.0},
            # Middle section - for elevated vehicles
            {"y_min_ratio": 0.25, "y_max_ratio": 0.65, "name": "elevated_zone", "priority": 0.7},
        ]

        self.logger.info("🛡️ Anti-False-Positive Detector initialized")

    def detect_plates(self, image: np.ndarray) -> List[ValidatedPlateDetection]:
        """Main detection dengan comprehensive false positive filtering"""
        detections = []

        if image is None or image.size == 0:
            return detections

        height, width = image.shape[:2]
        self.logger.info(f"🔍 Anti-FP detection on {width}x{height} image")

        # Step 1: Initial candidate detection dengan wide search
        candidates = self._find_plate_candidates(image)
        self.logger.info(f"📋 Found {len(candidates)} initial candidates")

        # Step 2: Apply comprehensive validation filters
        validated_detections = []
        for candidate in candidates:
            validation_result = self._validate_candidate(candidate, image.shape)

            if validation_result['is_valid']:
                detection = ValidatedPlateDetection(
                    text=candidate['text'],
                    confidence=candidate['confidence'],
                    bbox=candidate['bbox'],
                    validation_score=validation_result['validation_score'],
                    false_positive_score=validation_result['false_positive_score'],
                    validation_details=validation_result,
                    detection_method="anti_fp_validated"
                )
                validated_detections.append(detection)

        # Step 3: Rank dan filter final results
        final_detections = self._rank_and_filter_detections(validated_detections)

        self.logger.info(f"✅ Anti-FP detection complete: {len(final_detections)} valid plates")
        return final_detections

    def _find_plate_candidates(self, image: np.ndarray) -> List[Dict]:
        """Find potential plate candidates with relaxed initial filtering"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Multiple threshold approaches
        thresholds = [
            cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
            cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ]

        for i, thresh in enumerate(thresholds):
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Very relaxed initial filtering
                if 300 < area < 15000:  # Size range
                    aspect_ratio = w / h if h > 0 else 0
                    if 1.5 <= aspect_ratio <= 8.0:  # Aspect ratio range
                        # Extract ROI dan apply OCR
                        roi = image[y:y+h, x:x+w]
                        ocr_result = self._apply_ocr_to_roi(roi)

                        if ocr_result and ocr_result['text']:
                            candidate = {
                                'bbox': (x, y, w, h),
                                'text': ocr_result['text'],
                                'confidence': ocr_result['confidence'],
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'threshold_method': f"thresh_{i}",
                                'roi': roi
                            }
                            candidates.append(candidate)

        return candidates

    def _apply_ocr_to_roi(self, roi: np.ndarray) -> Optional[Dict]:
        """Apply OCR to region of interest"""
        try:
            # Upscale untuk better OCR
            scale_factor = max(3, 100 // max(roi.shape[:2]))
            upscaled = cv2.resize(roi, (roi.shape[1]*scale_factor, roi.shape[0]*scale_factor),
                                interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale if needed
            if len(upscaled.shape) == 3:
                upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            else:
                upscaled_gray = upscaled

            # Apply OCR dengan multiple configurations
            configs = [
                '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
            ]

            best_result = None
            best_confidence = 0

            for config in configs:
                # Get detailed OCR data
                data = pytesseract.image_to_data(upscaled_gray, config=config,
                                               output_type=pytesseract.Output.DICT)

                words = []
                confidences = []

                for i in range(len(data['text'])):
                    conf = int(data['conf'][i])
                    word = data['text'][i].strip()

                    if conf > 10 and word:  # Low threshold untuk initial detection
                        words.append(word)
                        confidences.append(conf)

                if words:
                    combined_text = ' '.join(words)
                    avg_confidence = np.mean(confidences)

                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_result = {
                            'text': combined_text,
                            'confidence': avg_confidence,
                            'word_count': len(words)
                        }

            return best_result

        except Exception as e:
            self.logger.debug(f"OCR error: {e}")
            return None

    def _validate_candidate(self, candidate: Dict, image_shape: Tuple) -> Dict:
        """Comprehensive validation untuk filter false positives"""
        validation_result = {
            'is_valid': False,
            'validation_score': 0.0,
            'false_positive_score': 0.0,
            'failed_checks': [],
            'passed_checks': []
        }

        x, y, w, h = candidate['bbox']
        text = candidate['text']
        roi = candidate['roi']
        image_height, image_width = image_shape[:2]

        # CHECK 1: Position Validation (Y-coordinate)
        y_ratio = y / image_height
        if y_ratio < 0.33:  # Too high in image
            validation_result['false_positive_score'] += 0.4
            validation_result['failed_checks'].append('position_too_high')
        else:
            validation_result['validation_score'] += 0.2
            validation_result['passed_checks'].append('position_valid')

        # CHECK 2: Text Length Validation
        clean_text = text.replace(' ', '')
        if len(clean_text) > 12:  # Too long for plate
            validation_result['false_positive_score'] += 0.3
            validation_result['failed_checks'].append('text_too_long')
        elif 4 <= len(clean_text) <= 10:  # Good length for Indonesian plate
            validation_result['validation_score'] += 0.3
            validation_result['passed_checks'].append('text_length_good')

        # CHECK 3: False Positive Keywords
        text_upper = text.upper()
        if any(keyword in text_upper for keyword in self.false_positive_keywords):
            validation_result['false_positive_score'] += 0.5
            validation_result['failed_checks'].append('contains_fp_keywords')
        else:
            validation_result['validation_score'] += 0.1
            validation_result['passed_checks'].append('no_fp_keywords')

        # CHECK 4: Indonesian Plate Pattern Validation
        pattern_score = self._validate_indonesian_pattern(text)
        if pattern_score > 0.5:
            validation_result['validation_score'] += 0.4
            validation_result['passed_checks'].append('indonesian_pattern_match')
        elif pattern_score < 0.2:
            validation_result['false_positive_score'] += 0.2
            validation_result['failed_checks'].append('poor_pattern_match')

        # CHECK 5: Regional Code Validation
        first_chars = clean_text[:2].upper()
        first_char = clean_text[:1].upper()
        if first_chars in self.valid_regional_codes or first_char in self.valid_regional_codes:
            validation_result['validation_score'] += 0.3
            validation_result['passed_checks'].append('valid_regional_code')

        # CHECK 6: Brightness Analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        mean_brightness = np.mean(gray_roi)
        if mean_brightness < 50:  # Too dark
            validation_result['false_positive_score'] += 0.2
            validation_result['failed_checks'].append('too_dark')
        elif mean_brightness > 80:  # Good brightness for plate
            validation_result['validation_score'] += 0.1
            validation_result['passed_checks'].append('good_brightness')

        # CHECK 7: Edge Density Analysis
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h)
        if edge_density > 0.2:  # Too complex/noisy
            validation_result['false_positive_score'] += 0.25
            validation_result['failed_checks'].append('too_complex')
        else:
            validation_result['validation_score'] += 0.1
            validation_result['passed_checks'].append('good_edge_density')

        # CHECK 8: Size Validation
        if area := w * h:
            if area < 500:  # Too small
                validation_result['false_positive_score'] += 0.2
                validation_result['failed_checks'].append('too_small')
            elif 1000 <= area <= 8000:  # Good size for CCTV plate
                validation_result['validation_score'] += 0.2
                validation_result['passed_checks'].append('good_size')

        # CHECK 9: Aspect Ratio Validation
        aspect_ratio = candidate['aspect_ratio']
        if 2.0 <= aspect_ratio <= 5.5:  # Typical Indonesian plate
            validation_result['validation_score'] += 0.2
            validation_result['passed_checks'].append('good_aspect_ratio')
        elif aspect_ratio < 1.5 or aspect_ratio > 8.0:  # Too extreme
            validation_result['false_positive_score'] += 0.15
            validation_result['failed_checks'].append('bad_aspect_ratio')

        # CHECK 10: Character Composition
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)
        if has_letters and has_numbers:
            validation_result['validation_score'] += 0.3
            validation_result['passed_checks'].append('mixed_alphanumeric')
        elif clean_text.isalpha() or clean_text.isdigit():  # Only letters or numbers
            validation_result['false_positive_score'] += 0.2
            validation_result['failed_checks'].append('single_character_type')

        # Final decision
        final_score = validation_result['validation_score'] - validation_result['false_positive_score']
        validation_result['final_score'] = final_score
        validation_result['is_valid'] = final_score > 0.3  # Threshold untuk valid detection

        return validation_result

    def _validate_indonesian_pattern(self, text: str) -> float:
        """Validate against Indonesian plate patterns"""
        if not text:
            return 0.0

        # Indonesian plate patterns dengan scoring
        patterns = [
            (r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$', 1.0),  # Perfect pattern
            (r'^[A-Z]\s*\d{3,4}\s*[A-Z]{2,3}$', 0.9),       # Common pattern
            (r'^[A-Z]{2}\s*\d{3,4}\s*[A-Z]$', 0.8),         # Regional pattern
            (r'^\d{1,4}\s*[A-Z]{1,3}$', 0.4),               # Partial pattern
            (r'^[A-Z]{1,2}\s*\d{1,4}$', 0.3),               # Partial pattern
        ]

        clean_text = text.strip().upper()

        for pattern, score in patterns:
            if re.match(pattern, clean_text):
                return score

        return 0.0

    def _rank_and_filter_detections(self, detections: List[ValidatedPlateDetection]) -> List[ValidatedPlateDetection]:
        """Rank and filter final detections"""
        if not detections:
            return detections

        # Remove similar detections
        unique_detections = self._remove_duplicate_detections(detections)

        # Sort by combined score (validation score + confidence)
        unique_detections.sort(key=lambda d: (
            d.validation_score * 0.6 +
            (d.confidence / 100.0) * 0.4 -
            d.false_positive_score * 0.3
        ), reverse=True)

        # Return top valid detections
        return unique_detections[:3]

    def _remove_duplicate_detections(self, detections: List[ValidatedPlateDetection]) -> List[ValidatedPlateDetection]:
        """Remove overlapping detections"""
        if len(detections) <= 1:
            return detections

        unique = []
        for detection in detections:
            is_duplicate = False

            for existing in unique:
                # Check spatial overlap
                overlap = self._calculate_bbox_overlap(detection.bbox, existing.bbox)
                if overlap > 0.3:  # 30% overlap threshold
                    is_duplicate = True
                    # Keep the one dengan higher combined score
                    if detection.validation_score > existing.validation_score:
                        unique.remove(existing)
                        unique.append(detection)
                    break

            if not is_duplicate:
                unique.append(detection)

        return unique

    def _calculate_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate IoU between bounding boxes"""
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

    def draw_detections(self, frame: np.ndarray, detections: List[ValidatedPlateDetection]) -> np.ndarray:
        """Draw validated detections dengan detailed info"""
        result = frame.copy()

        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox

            # Color based on validation quality
            if detection.validation_score >= 0.8:
                color = (0, 255, 0)  # Green - high quality
                thickness = 3
            elif detection.validation_score >= 0.5:
                color = (0, 165, 255)  # Orange - medium quality
                thickness = 2
            else:
                color = (0, 100, 255)  # Red - low quality
                thickness = 2

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

            # Main label
            label = f"VALIDATED: {detection.text} ({detection.confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Background untuk text
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(result, (x, y - text_h - 15), (x + text_w, y), color, -1)
            cv2.putText(result, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Validation details
            validation_info = f"Val:{detection.validation_score:.2f} FP:{detection.false_positive_score:.2f}"
            cv2.putText(result, validation_info, (x, y + h + 20), font, 0.5, color, 1)

            # Passed/failed checks indicator
            passed_count = len(detection.validation_details.get('passed_checks', []))
            failed_count = len(detection.validation_details.get('failed_checks', []))
            check_info = f"✓{passed_count} ✗{failed_count}"
            cv2.putText(result, check_info, (x, y + h + 35), font, 0.4, color, 1)

        return result

    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            "detector_type": "ANTI_FALSE_POSITIVE",
            "validation_checks": 10,
            "false_positive_keywords": len(self.false_positive_keywords),
            "regional_codes": len(self.valid_regional_codes),
            "detection_zones": len(self.valid_detection_zones)
        }

if __name__ == "__main__":
    # Test anti-false-positive detector
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)

        if image is not None:
            detector = AntiFalsePositiveDetector()
            detections = detector.detect_plates(image)

            print(f"🛡️ ANTI-FP detected {len(detections)} validated plates:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%)")
                print(f"       Validation score: {det.validation_score:.2f}")
                print(f"       False positive score: {det.false_positive_score:.2f}")
                print(f"       Passed checks: {len(det.validation_details.get('passed_checks', []))}")
                print(f"       Failed checks: {len(det.validation_details.get('failed_checks', []))}")
                if det.validation_details.get('failed_checks'):
                    print(f"       Failed: {det.validation_details['failed_checks']}")
                print()

            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("anti_false_positive_result.jpg", result)
            print("💾 Result saved: anti_false_positive_result.jpg")

            # Show statistics
            stats = detector.get_statistics()
            print(f"📊 Statistics: {stats}")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python anti_false_positive_detector.py <image_path>")