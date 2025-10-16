#!/usr/bin/env python3
"""
CCTV-Optimized Plate Detector
Specialized untuk kondisi CCTV dengan plat jarak jauh dan kondisi pencahayaan yang challenging
"""

import cv2
import numpy as np
import pytesseract
import logging
import time
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class CCTVPlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    region_name: str = "unknown"
    preprocessing_method: str = "standard"
    ocr_method: str = "standard"
    raw_confidence: float = 0.0

class CCTVPlateDetector:
    """
    Detector khusus untuk kondisi CCTV dengan optimasi untuk:
    - Plat nomor jarak jauh
    - Pencahayaan bervariasi
    - Sudut kamera yang tidak ideal
    - Resolusi terbatas
    """

    def __init__(self):
        """Initialize CCTV plate detector"""
        self.logger = logging.getLogger(__name__)

        # CCTV-specific regions untuk scanning
        # Berdasarkan analisis, plat biasanya berada di area bawah-tengah frame
        self.search_regions = [
            # Format: (x_ratio, y_ratio, width_ratio, height_ratio, name)
            (0.25, 0.45, 0.50, 0.25, "center_bottom_focus"),
            (0.20, 0.40, 0.60, 0.35, "wide_center_search"),
            (0.30, 0.50, 0.40, 0.20, "tight_center_focus"),
            (0.15, 0.35, 0.70, 0.45, "extended_search"),
        ]

        # OCR configurations optimized untuk CCTV
        self.ocr_configs = [
            {
                'name': 'cctv_block',
                'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                'weight': 1.5
            },
            {
                'name': 'cctv_line',
                'config': '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                'weight': 1.3
            },
            {
                'name': 'cctv_word',
                'config': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                'weight': 1.1
            },
            {
                'name': 'cctv_indonesia',
                'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l ind+eng',
                'weight': 1.2
            }
        ]

        # Indonesian plate patterns untuk validation
        self.indonesian_patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',  # Standard: B 1234 ABC
            r'^[A-Z]\s*\d{3,4}\s*[A-Z]{2,3}$',       # Common: B 1234 AB
            r'^[A-Z]{2}\s*\d{3,4}\s*[A-Z]$',         # Regional: AB 1234 C
        ]

        # Performance tracking
        self.total_detections = 0
        self.successful_detections = 0

        self.logger.info("🎯 CCTV Plate Detector initialized")

    def detect_plates(self, image: np.ndarray) -> List[CCTVPlateDetection]:
        """
        Main detection method untuk CCTV images
        """
        start_time = time.time()
        detections = []

        if image is None or image.size == 0:
            return detections

        height, width = image.shape[:2]
        self.logger.info(f"🔍 Processing CCTV image: {width}x{height}")

        # Process each search region
        for region_info in self.search_regions:
            x_ratio, y_ratio, w_ratio, h_ratio, region_name = region_info

            # Calculate actual coordinates
            x = int(width * x_ratio)
            y = int(height * y_ratio)
            w = int(width * w_ratio)
            h = int(height * h_ratio)

            # Extract region
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            self.logger.debug(f"🔍 Analyzing region: {region_name} ({x},{y},{w},{h})")

            # Detect plates in this region
            region_detections = self._detect_in_region(roi, (x, y), region_name)

            # Adjust coordinates back to full image
            for detection in region_detections:
                det_x, det_y, det_w, det_h = detection.bbox
                detection.bbox = (det_x + x, det_y + y, det_w, det_h)
                detections.append(detection)

        # Post-process detections
        detections = self._post_process_detections(detections)

        # Update statistics
        self.total_detections += len(detections)
        self.successful_detections += len([d for d in detections if d.confidence >= 30])

        detection_time = time.time() - start_time
        self.logger.info(f"✅ CCTV detection complete: {len(detections)} plates in {detection_time:.2f}s")

        return detections

    def _detect_in_region(self, roi: np.ndarray, offset: Tuple[int, int], region_name: str) -> List[CCTVPlateDetection]:
        """Detect plates within a specific region"""
        detections = []

        # Apply multiple preprocessing methods
        preprocessed_images = self._preprocess_for_cctv(roi)

        for preprocessing_method, processed_image in preprocessed_images:
            # Find contours dengan relaxed parameters
            candidates = self._find_plate_candidates(processed_image, preprocessing_method)

            # Process each candidate
            for candidate_info in candidates:
                x, y, w, h, area, aspect_ratio = candidate_info

                # Extract candidate region
                candidate_roi = roi[y:y+h, x:x+w]
                if candidate_roi.size == 0:
                    continue

                # Apply OCR dengan multiple methods
                ocr_results = self._apply_ocr_methods(candidate_roi)

                # Select best OCR result
                best_result = self._select_best_ocr_result(ocr_results)

                if best_result and best_result['confidence'] >= 25:  # Relaxed threshold
                    detection = CCTVPlateDetection(
                        text=best_result['text'],
                        confidence=best_result['confidence'],
                        bbox=(x, y, w, h),
                        region_name=region_name,
                        preprocessing_method=preprocessing_method,
                        ocr_method=best_result['method'],
                        raw_confidence=best_result['raw_confidence']
                    )
                    detections.append(detection)

                    self.logger.debug(f"✅ Found plate: '{detection.text}' ({detection.confidence:.1f}%) in {region_name}")

        return detections

    def _preprocess_for_cctv(self, roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Apply multiple preprocessing methods optimized for CCTV conditions"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        preprocessed = []

        # Method 1: CLAHE + Gaussian (good for variable lighting)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blur1 = cv2.GaussianBlur(enhanced, (3, 3), 0)
        preprocessed.append(("clahe_gaussian", blur1))

        # Method 2: Bilateral filter (preserves edges)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed.append(("bilateral", bilateral))

        # Method 3: Morphological processing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        preprocessed.append(("morphological", morph))

        # Method 4: Histogram equalization
        equalized = cv2.equalizeHist(gray)
        blurred_eq = cv2.GaussianBlur(equalized, (3, 3), 0)
        preprocessed.append(("histogram_eq", blurred_eq))

        return preprocessed

    def _find_plate_candidates(self, processed_image: np.ndarray, method_name: str) -> List[Tuple]:
        """Find potential plate regions using contour analysis"""
        candidates = []

        # Apply adaptive threshold (try both normal and inverted)
        thresh_normal = cv2.adaptiveThreshold(
            processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        thresh_inv = cv2.adaptiveThreshold(
            processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        for thresh_name, thresh in [('normal', thresh_normal), ('inverted', thresh_inv)]:
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # Filter berdasarkan karakteristik plat nomor Indonesia
                if self._is_valid_plate_candidate(w, h, area):
                    aspect_ratio = w / h if h > 0 else 0
                    candidates.append((x, y, w, h, area, aspect_ratio))

        # Sort by area (larger candidates processed first)
        candidates.sort(key=lambda x: x[4], reverse=True)

        # Return top candidates untuk efficiency
        return candidates[:10]  # Limit to top 10 per method

    def _is_valid_plate_candidate(self, width: int, height: int, area: int) -> bool:
        """Validate if dimensions match Indonesian license plate characteristics"""
        # Size validation - very permissive untuk CCTV conditions
        if area < 200 or area > 15000:  # Reasonable area range
            return False

        if width < 25 or height < 8:  # Minimum readable size
            return False

        if width > 400 or height > 150:  # Maximum reasonable size
            return False

        # Aspect ratio validation - permissive untuk various angles
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 1.5 or aspect_ratio > 8.0:  # Very permissive range
            return False

        return True

    def _apply_ocr_methods(self, candidate_roi: np.ndarray) -> List[Dict]:
        """Apply multiple OCR methods to a candidate region"""
        results = []

        # Upscale candidate untuk better OCR
        scale_factor = max(3, 60 // max(candidate_roi.shape[:2]))
        upscaled = cv2.resize(
            candidate_roi,
            (candidate_roi.shape[1] * scale_factor, candidate_roi.shape[0] * scale_factor),
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale if needed
        if len(upscaled.shape) == 3:
            upscaled_gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        else:
            upscaled_gray = upscaled

        # Apply additional enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        final_image = clahe.apply(upscaled_gray)

        # Try each OCR configuration
        for ocr_config in self.ocr_configs:
            try:
                # Get detailed OCR data
                data = pytesseract.image_to_data(
                    final_image,
                    config=ocr_config['config'],
                    output_type=pytesseract.Output.DICT
                )

                # Extract text with confidence filtering
                words = []
                confidences = []

                for i in range(len(data['text'])):
                    conf = int(data['conf'][i])
                    text = data['text'][i].strip()

                    if conf > 20 and text:  # Relaxed confidence threshold
                        words.append(text)
                        confidences.append(conf)

                if words:
                    combined_text = ' '.join(words)
                    avg_confidence = np.mean(confidences)

                    # Apply Indonesian plate validation boost
                    validation_boost = self._validate_indonesian_pattern(combined_text)
                    final_confidence = min(100.0, avg_confidence + validation_boost)

                    results.append({
                        'text': combined_text,
                        'confidence': final_confidence,
                        'raw_confidence': avg_confidence,
                        'method': ocr_config['name'],
                        'weight': ocr_config['weight']
                    })

            except Exception as e:
                self.logger.debug(f"OCR method {ocr_config['name']} failed: {e}")
                continue

        return results

    def _validate_indonesian_pattern(self, text: str) -> float:
        """Validate text against Indonesian license plate patterns and return confidence boost"""
        if not text or len(text) < 3:
            return 0.0

        boost = 0.0
        clean_text = text.replace(' ', '').upper()

        # Pattern matching boost
        import re
        for pattern in self.indonesian_patterns:
            if re.match(pattern, text.upper().strip()):
                boost += 15.0  # Significant boost for pattern match
                break

        # Character composition analysis
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)

        if has_letters and has_numbers:
            boost += 8.0  # Mixed alphanumeric bonus

        # Length analysis (Indonesian plates typically 6-9 characters)
        if 5 <= len(clean_text) <= 10:
            boost += 5.0

        # Regional code analysis
        regional_codes = ['B', 'D', 'F', 'E', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W']
        if any(clean_text.startswith(code) for code in regional_codes):
            boost += 10.0

        return boost

    def _select_best_ocr_result(self, ocr_results: List[Dict]) -> Optional[Dict]:
        """Select the best OCR result based on confidence and method weight"""
        if not ocr_results:
            return None

        # Calculate weighted scores
        best_result = None
        best_score = 0.0

        for result in ocr_results:
            # Weighted score = confidence * method_weight
            weighted_score = result['confidence'] * result['weight']

            if weighted_score > best_score:
                best_score = weighted_score
                best_result = result

        return best_result

    def _post_process_detections(self, detections: List[CCTVPlateDetection]) -> List[CCTVPlateDetection]:
        """Post-process and filter detections"""
        if not detections:
            return detections

        # Remove duplicates based on spatial overlap
        filtered = self._remove_overlapping_detections(detections)

        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)

        # Limit results untuk performance
        return filtered[:5]  # Top 5 detections

    def _remove_overlapping_detections(self, detections: List[CCTVPlateDetection]) -> List[CCTVPlateDetection]:
        """Remove overlapping detections keeping the best ones"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        filtered = []
        for detection in sorted_detections:
            bbox1 = detection.bbox
            is_overlap = False

            for existing in filtered:
                bbox2 = existing.bbox
                if self._calculate_overlap(bbox1, bbox2) > 0.3:  # 30% overlap threshold
                    is_overlap = True
                    break

            if not is_overlap:
                filtered.append(detection)

        return filtered

    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
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

    def draw_detections(self, frame: np.ndarray, detections: List[CCTVPlateDetection]) -> np.ndarray:
        """Draw CCTV detections with distinctive styling"""
        result = frame.copy()

        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox

            # Color coding based on confidence
            if detection.confidence >= 70:
                color = (0, 255, 0)  # Green for high confidence
            elif detection.confidence >= 50:
                color = (0, 165, 255)  # Orange for medium confidence
            else:
                color = (0, 100, 255)  # Red for low confidence

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)

            # Draw label
            label = f"CCTV-PLATE: {detection.text} ({detection.confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw background
            cv2.rectangle(result, (x, y - text_h - 10), (x + text_w, y), color, -1)

            # Draw text
            cv2.putText(result, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Draw region indicator
            region_label = f"[{detection.region_name}]"
            cv2.putText(result, region_label, (x, y + h + 20), font, 0.5, color, 1)

        return result

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        success_rate = (self.successful_detections / self.total_detections * 100) if self.total_detections > 0 else 0

        return {
            "detector_type": "CCTV_OPTIMIZED",
            "total_detections": self.total_detections,
            "successful_detections": self.successful_detections,
            "success_rate": round(success_rate, 1),
            "search_regions": len(self.search_regions),
            "ocr_methods": len(self.ocr_configs)
        }

if __name__ == "__main__":
    # Test CCTV detector
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)

        if image is not None:
            detector = CCTVPlateDetector()
            detections = detector.detect_plates(image)

            print(f"🎯 CCTV detected {len(detections)} license plates:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%) from {det.region_name} via {det.ocr_method}")

            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("cctv_plate_result.jpg", result)
            print("💾 Result saved: cctv_plate_result.jpg")

            # Show statistics
            stats = detector.get_statistics()
            print(f"📊 Statistics: {stats}")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python cctv_plate_detector.py <image_path>")