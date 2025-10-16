#!/usr/bin/env python3
"""
Balanced CCTV Detector
Detector yang seimbang antara mendeteksi plat asli dan memfilter false positives
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
class BalancedPlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_likely_plate: bool
    quality_score: float
    detection_details: Dict
    detection_method: str = "balanced_cctv"

class BalancedCCTVDetector:
    """
    Balanced detector dengan smart filtering untuk CCTV conditions
    """

    def __init__(self):
        """Initialize balanced detector"""
        self.logger = logging.getLogger(__name__)

        # UI/System text patterns yang sering muncul di CCTV interface
        self.ui_patterns = [
            r'AAS.*\(\d+\)',  # AAS (1043) pattern dari screenshot
            r'.*CCTV.*',
            r'.*DETECTION.*',
            r'.*LIVE.*',
            r'.*FPS.*',
            r'.*RESOLUTION.*',
            r'.*\d{4}-\d{2}-\d{2}.*',  # Date patterns
            r'.*\d{2}:\d{2}:\d{2}.*',  # Time patterns
        ]

        # Target areas untuk plat nomor - fokus pada area kendaraan
        self.detection_zones = [
            # Zone untuk mobil sedang (area bawah-tengah)
            {"x_min": 0.1, "x_max": 0.9, "y_min": 0.4, "y_max": 0.85, "name": "main_vehicle_zone", "priority": 1.0},
            # Zone untuk kendaraan tinggi atau sudut berbeda
            {"x_min": 0.2, "x_max": 0.8, "y_min": 0.3, "y_max": 0.7, "name": "elevated_vehicle_zone", "priority": 0.8},
            # Zone untuk kendaraan jauh
            {"x_min": 0.3, "x_max": 0.7, "y_min": 0.25, "y_max": 0.6, "name": "distant_vehicle_zone", "priority": 0.6},
        ]

        # Indonesian plate characteristics
        self.plate_patterns = [
            (r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$', 1.0, 'Perfect Indonesian plate'),
            (r'^[A-Z]\s*\d{3,4}\s*[A-Z]{2,3}$', 0.9, 'Standard car plate'),
            (r'^[A-Z]{2}\s*\d{3,4}\s*[A-Z]$', 0.8, 'Regional plate format'),
        ]

        self.logger.info("⚖️ Balanced CCTV Detector initialized")

    def detect_plates(self, image: np.ndarray) -> List[BalancedPlateDetection]:
        """Main detection dengan balanced filtering"""
        detections = []

        if image is None or image.size == 0:
            return detections

        height, width = image.shape[:2]
        self.logger.info(f"🔍 Balanced detection on {width}x{height} image")

        # Step 1: Find all potential candidates
        candidates = self._find_all_candidates(image)
        self.logger.info(f"📋 Found {len(candidates)} candidates")

        # Step 2: Apply smart filtering
        for candidate in candidates:
            detection_result = self._smart_validate_candidate(candidate, (height, width))

            if detection_result['should_include']:
                detection = BalancedPlateDetection(
                    text=candidate['text'],
                    confidence=candidate['confidence'],
                    bbox=candidate['bbox'],
                    is_likely_plate=detection_result['is_likely_plate'],
                    quality_score=detection_result['quality_score'],
                    detection_details=detection_result,
                    detection_method="balanced_validated"
                )
                detections.append(detection)

        # Step 3: Final ranking dan selection
        final_detections = self._select_best_detections(detections)

        self.logger.info(f"✅ Balanced detection complete: {len(final_detections)} detections")
        return final_detections

    def _find_all_candidates(self, image: np.ndarray) -> List[Dict]:
        """Find all potential candidates dengan expanded search"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhanced preprocessing untuk various lighting conditions
        preprocessed_images = [
            ("original", gray),
            ("clahe", cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)),
            ("contrast", cv2.convertScaleAbs(gray, alpha=1.5, beta=20)),
        ]

        for prep_name, prep_img in preprocessed_images:
            # Multiple thresholding approaches
            thresholds = [
                ("adaptive_normal", cv2.adaptiveThreshold(prep_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                ("adaptive_inv", cv2.adaptiveThreshold(prep_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
            ]

            for thresh_name, thresh in thresholds:
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h

                    # Relaxed initial filtering
                    if 200 < area < 20000:  # Wide area range
                        aspect_ratio = w / h if h > 0 else 0
                        if 1.2 <= aspect_ratio <= 10.0:  # Very permissive aspect ratio

                            # Apply OCR
                            roi = image[y:y+h, x:x+w]
                            ocr_results = self._comprehensive_ocr(roi)

                            for ocr_result in ocr_results:
                                if ocr_result and ocr_result.get('text'):
                                    candidate = {
                                        'bbox': (x, y, w, h),
                                        'text': ocr_result['text'],
                                        'confidence': ocr_result['confidence'],
                                        'area': area,
                                        'aspect_ratio': aspect_ratio,
                                        'preprocessing': prep_name,
                                        'threshold': thresh_name,
                                        'ocr_method': ocr_result['method'],
                                        'roi': roi
                                    }
                                    candidates.append(candidate)

        return candidates

    def _comprehensive_ocr(self, roi: np.ndarray) -> List[Dict]:
        """Apply comprehensive OCR dengan multiple methods"""
        results = []

        try:
            # Upscaling untuk better OCR
            scale_factor = max(3, 120 // max(roi.shape[:2]))
            upscaled = cv2.resize(roi, (roi.shape[1]*scale_factor, roi.shape[0]*scale_factor),
                                interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale
            if len(upscaled.shape) == 3:
                gray_upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
            else:
                gray_upscaled = upscaled

            # Multiple OCR configurations
            ocr_configs = [
                ('psm6_eng', '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng'),
                ('psm7_eng', '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng'),
                ('psm8_eng', '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng'),
                ('psm6_ind', '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l ind+eng'),
            ]

            for method_name, config in ocr_configs:
                try:
                    # Get detailed OCR data
                    data = pytesseract.image_to_data(gray_upscaled, config=config,
                                                   output_type=pytesseract.Output.DICT)

                    words = []
                    confidences = []

                    for i in range(len(data['text'])):
                        conf = int(data['conf'][i])
                        word = data['text'][i].strip()

                        # Very low threshold untuk initial capture
                        if conf > 5 and word and len(word) >= 1:
                            words.append(word)
                            confidences.append(conf)

                    if words:
                        combined_text = ' '.join(words)
                        avg_confidence = np.mean(confidences)

                        result = {
                            'text': combined_text,
                            'confidence': avg_confidence,
                            'method': method_name,
                            'word_count': len(words),
                            'individual_confidences': confidences
                        }
                        results.append(result)

                except Exception:
                    continue

        except Exception as e:
            self.logger.debug(f"OCR error: {e}")

        return results

    def _smart_validate_candidate(self, candidate: Dict, image_shape: Tuple) -> Dict:
        """Smart validation yang balance antara detection dan filtering"""
        result = {
            'should_include': False,
            'is_likely_plate': False,
            'quality_score': 0.0,
            'reasons': []
        }

        x, y, w, h = candidate['bbox']
        text = candidate['text'].strip()
        confidence = candidate['confidence']
        height, width = image_shape

        # SMART FILTER 1: UI Pattern Detection (Strong Filter)
        for pattern in self.ui_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                result['reasons'].append(f'UI_PATTERN_MATCH: {text}')
                return result  # Strong rejection

        # SMART FILTER 2: Position-based filtering (Moderate Filter)
        y_ratio = y / height
        x_ratio = x / width

        # Check if in valid vehicle zones
        in_valid_zone = False
        zone_priority = 0.0

        for zone in self.detection_zones:
            if (zone['x_min'] <= x_ratio <= zone['x_max'] and
                zone['y_min'] <= y_ratio <= zone['y_max']):
                in_valid_zone = True
                zone_priority = max(zone_priority, zone['priority'])
                result['reasons'].append(f'IN_ZONE: {zone["name"]}')
                break

        if not in_valid_zone:
            # If not in valid zone, require higher confidence
            if confidence < 40:
                result['reasons'].append('OUT_OF_ZONE_LOW_CONF')
                return result
            else:
                result['reasons'].append('OUT_OF_ZONE_HIGH_CONF')
                zone_priority = 0.3  # Low priority tapi still consider

        # SMART FILTER 3: Text Quality Analysis
        clean_text = text.replace(' ', '').upper()

        # Text length check
        if len(clean_text) < 2:
            result['reasons'].append('TEXT_TOO_SHORT')
            return result
        elif len(clean_text) > 15:  # Very long text likely not a plate
            if confidence < 60:  # Unless very confident
                result['reasons'].append('TEXT_TOO_LONG_LOW_CONF')
                return result

        # SMART FILTER 4: Indonesian Plate Pattern Scoring
        pattern_score = 0.0
        for pattern, score, desc in self.plate_patterns:
            if re.match(pattern, text.strip().upper()):
                pattern_score = score
                result['reasons'].append(f'PATTERN_MATCH: {desc}')
                break

        # SMART FILTER 5: Character composition analysis
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)

        composition_score = 0.0
        if has_letters and has_numbers:
            composition_score = 0.8
            result['reasons'].append('MIXED_ALPHANUM')
        elif has_letters or has_numbers:
            composition_score = 0.4
            result['reasons'].append('SINGLE_TYPE')

        # SMART FILTER 6: Size and aspect ratio validation
        area = w * h
        aspect_ratio = candidate['aspect_ratio']

        size_score = 0.0
        if 500 <= area <= 10000:  # Good size range
            size_score += 0.3
        if 2.0 <= aspect_ratio <= 6.0:  # Good aspect ratio
            size_score += 0.3

        # CALCULATE FINAL QUALITY SCORE
        quality_score = (
            (confidence / 100.0) * 0.3 +
            pattern_score * 0.3 +
            composition_score * 0.2 +
            zone_priority * 0.1 +
            size_score * 0.1
        )

        result['quality_score'] = quality_score

        # DECISION LOGIC
        if quality_score >= 0.4:  # Good quality
            result['should_include'] = True
            result['is_likely_plate'] = quality_score >= 0.6
            result['reasons'].append(f'ACCEPTED_QUALITY: {quality_score:.2f}')
        elif quality_score >= 0.25 and confidence >= 50:  # Medium quality but high confidence
            result['should_include'] = True
            result['is_likely_plate'] = False
            result['reasons'].append(f'ACCEPTED_CONFIDENCE: conf={confidence:.1f}, qual={quality_score:.2f}')
        else:
            result['reasons'].append(f'REJECTED_LOW_QUALITY: {quality_score:.2f}')

        return result

    def _select_best_detections(self, detections: List[BalancedPlateDetection]) -> List[BalancedPlateDetection]:
        """Select best detections with overlap removal"""
        if not detections:
            return detections

        # Remove overlapping detections
        unique_detections = self._remove_overlapping_detections(detections)

        # Separate likely plates from uncertain detections
        likely_plates = [d for d in unique_detections if d.is_likely_plate]
        uncertain_detections = [d for d in unique_detections if not d.is_likely_plate]

        # Sort each group
        likely_plates.sort(key=lambda d: d.quality_score, reverse=True)
        uncertain_detections.sort(key=lambda d: d.confidence, reverse=True)

        # Return best detections (prioritize likely plates)
        final_detections = likely_plates[:2] + uncertain_detections[:2]
        return final_detections[:3]  # Max 3 detections

    def _remove_overlapping_detections(self, detections: List[BalancedPlateDetection]) -> List[BalancedPlateDetection]:
        """Remove overlapping detections keeping the best ones"""
        if len(detections) <= 1:
            return detections

        # Sort by quality score
        sorted_detections = sorted(detections, key=lambda d: d.quality_score, reverse=True)

        unique = []
        for detection in sorted_detections:
            is_overlap = False

            for existing in unique:
                overlap = self._calculate_bbox_overlap(detection.bbox, existing.bbox)
                if overlap > 0.3:  # 30% overlap threshold
                    is_overlap = True
                    break

            if not is_overlap:
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

    def draw_detections(self, frame: np.ndarray, detections: List[BalancedPlateDetection]) -> np.ndarray:
        """Draw balanced detections dengan quality indicators"""
        result = frame.copy()

        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox

            # Color coding
            if detection.is_likely_plate:
                if detection.quality_score >= 0.8:
                    color = (0, 255, 0)  # Bright green - high confidence plate
                else:
                    color = (0, 200, 100)  # Green - likely plate
            else:
                color = (0, 165, 255)  # Orange - uncertain

            thickness = 3 if detection.is_likely_plate else 2

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

            # Label dengan quality info
            plate_indicator = "🎯PLATE" if detection.is_likely_plate else "❓UNCERTAIN"
            label = f"{plate_indicator}: {detection.text} ({detection.confidence:.1f}%)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Background untuk text
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(result, (x, y - text_h - 15), (x + text_w, y), color, -1)
            cv2.putText(result, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Quality score
            quality_info = f"Quality: {detection.quality_score:.2f}"
            cv2.putText(result, quality_info, (x, y + h + 20), font, 0.5, color, 1)

        return result

    def get_statistics(self) -> Dict:
        """Get balanced detector statistics"""
        return {
            "detector_type": "BALANCED_CCTV",
            "ui_patterns": len(self.ui_patterns),
            "detection_zones": len(self.detection_zones),
            "plate_patterns": len(self.plate_patterns),
            "balanced_filtering": True
        }

if __name__ == "__main__":
    # Test balanced detector
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)

        if image is not None:
            detector = BalancedCCTVDetector()
            detections = detector.detect_plates(image)

            print(f"⚖️ BALANCED detected {len(detections)} candidates:")
            for i, det in enumerate(detections):
                plate_type = "LIKELY PLATE" if det.is_likely_plate else "UNCERTAIN"
                print(f"   {i+1}. [{plate_type}] '{det.text}' ({det.confidence:.1f}%)")
                print(f"       Quality score: {det.quality_score:.2f}")
                print(f"       Reasons: {det.detection_details.get('reasons', [])}")
                print()

            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("balanced_cctv_result.jpg", result)
            print("💾 Result saved: balanced_cctv_result.jpg")

            # Show statistics
            stats = detector.get_statistics()
            print(f"📊 Statistics: {stats}")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python balanced_cctv_detector.py <image_path>")