#!/usr/bin/env python3
"""
Final Plate Detector
Detector final yang fokus pada plat nomor kendaraan dengan:
- Deteksi plat lurus dan miring
- Filter yang lebih smart untuk eliminasi false positive
- Multi-stage processing yang optimal
- Enhanced untuk kondisi CCTV real
"""

import cv2
import numpy as np
import pytesseract
import time
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class PlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    angle: float = 0.0
    processed_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    vehicle_type: str = "vehicle"
    detection_method: str = "final"

class FinalPlateDetector:
    """
    Final plate detector yang optimal untuk plat kendaraan
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Optimized parameters untuk plat kendaraan
        self.min_area = 400
        self.max_area = 15000
        self.min_aspect_ratio = 1.8
        self.max_aspect_ratio = 5.0
        self.min_width = 40
        self.max_width = 300
        self.min_height = 15
        self.max_height = 100
        
        # Smart validation parameters
        self.min_confidence = 30
        self.min_text_length = 3  # Indonesian plates usually 5+ chars
        
        self.logger.info("🎯 Final Plate Detector initialized for vehicle plates")
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Main detection method yang optimal untuk plat kendaraan
        """
        if image is None or image.size == 0:
            return []
        
        detections = []
        
        try:
            # Stage 1: Find high-quality candidates
            candidates = self._find_plate_candidates(image)
            self.logger.info(f"🔍 Found {len(candidates)} plate candidates")
            
            # Stage 2: Smart filtering
            filtered_candidates = self._smart_filter_candidates(image, candidates)
            self.logger.info(f"🔧 After smart filtering: {len(filtered_candidates)} candidates")
            
            # Stage 3: OCR processing
            for i, candidate in enumerate(filtered_candidates[:20]):  # Process top 20
                detection = self._process_plate_candidate(image, candidate, i+1)
                if detection:
                    detections.append(detection)
            
            # Stage 4: Final validation and ranking
            final_detections = self._final_validation(detections)
            
            self.logger.info(f"✅ Final plate detections: {len(final_detections)}")
            
        except Exception as e:
            self.logger.error(f"Error in final plate detection: {e}")
        
        return final_detections
    
    def _find_plate_candidates(self, image: np.ndarray) -> List[Dict]:
        """
        Find potential plate candidates dengan multiple methods
        """
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Method 1: Adaptive threshold untuk text detection
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        candidates.extend(self._extract_candidates_from_binary(adaptive, "adaptive"))
        
        # Method 2: OTSU untuk high contrast
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.extend(self._extract_candidates_from_binary(otsu, "otsu"))
        
        # Method 3: Edge-based detection untuk outlined plates
        edges = cv2.Canny(gray, 50, 150)
        # Dilate to connect characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        candidates.extend(self._extract_candidates_from_binary(dilated, "edges"))
        
        # Method 4: Morphological operations untuk rectangular shapes
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel)
        _, morph_thresh = cv2.threshold(morph, 10, 255, cv2.THRESH_BINARY)
        candidates.extend(self._extract_candidates_from_binary(morph_thresh, "morphological"))
        
        return candidates
    
    def _extract_candidates_from_binary(self, binary_image: np.ndarray, method: str) -> List[Dict]:
        """
        Extract candidates dari binary image
        """
        candidates = []
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check geometric constraints
            if (aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio or
                w < self.min_width or w > self.max_width or
                h < self.min_height or h > self.max_height):
                continue
            
            # Calculate quality score
            score = self._calculate_candidate_score(area, aspect_ratio, w, h, contour)
            
            candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'score': score,
                'method': method,
                'contour': contour
            })
        
        return candidates
    
    def _calculate_candidate_score(self, area: float, aspect_ratio: float, 
                                 width: int, height: int, contour) -> float:
        """
        Calculate score untuk plate candidacy
        """
        score = 0.0
        
        # Aspect ratio score (Indonesian plates typically 2.5-4.0)
        if 2.5 <= aspect_ratio <= 4.0:
            score += 40
        elif 2.0 <= aspect_ratio <= 5.0:
            score += 25
        else:
            score += max(0, 15 - abs(aspect_ratio - 3.0) * 3)
        
        # Size score (prefer medium-sized plates)
        optimal_area = 3000
        area_score = 100 - min(100, abs(area - optimal_area) / optimal_area * 50)
        score += area_score * 0.3
        
        # Rectangularity (how rectangular is the contour)
        rect_area = width * height
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity > 0.7:
            score += 20
        elif rectangularity > 0.5:
            score += 10
        
        # Dimension bonus for typical plate sizes
        if 60 <= width <= 200 and 20 <= height <= 60:
            score += 15
        
        return min(100, score)
    
    def _smart_filter_candidates(self, image: np.ndarray, candidates: List[Dict]) -> List[Dict]:
        """
        Smart filtering untuk eliminasi candidates yang jelas bukan plat
        """
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates
        filtered = []
        for candidate in candidates:
            bbox1 = candidate['bbox']
            is_duplicate = False
            
            for existing in filtered:
                bbox2 = existing['bbox']
                overlap = self._calculate_overlap(bbox1, bbox2)
                if overlap > 0.5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Quick visual validation
                if self._quick_visual_validation(image, candidate):
                    filtered.append(candidate)
        
        return filtered[:30]  # Return top 30 candidates
    
    def _quick_visual_validation(self, image: np.ndarray, candidate: Dict) -> bool:
        """
        Quick visual validation untuk eliminasi obvious false positives
        """
        x, y, w, h = candidate['bbox']
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return False
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Check basic properties
        mean_intensity = np.mean(roi_gray)
        std_intensity = np.std(roi_gray)
        
        # Reject if too uniform (likely not text)
        if std_intensity < 8:
            return False
        
        # Reject if too dark or too bright
        if mean_intensity < 20 or mean_intensity > 240:
            return False
        
        # Check for text-like patterns
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h) * 100
        
        # Must have some edge density for text
        if edge_density < 2:
            return False
        
        return True
    
    def _process_plate_candidate(self, image: np.ndarray, candidate: Dict, index: int) -> Optional[PlateDetection]:
        """
        Process individual plate candidate
        """
        bbox = candidate['bbox']
        x, y, w, h = bbox
        
        self.logger.debug(f"📋 Processing candidate {index}: {bbox} (score: {candidate['score']:.1f})")
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # Enhanced OCR processing
        text, confidence = self._enhanced_ocr(roi)
        
        if not text or len(text) < self.min_text_length:
            self.logger.debug(f"❌ Text too short: '{text}'")
            return None
        
        if confidence < self.min_confidence:
            self.logger.debug(f"❌ Low confidence: {confidence:.1f}%")
            return None
        
        # Indonesian plate pattern validation
        pattern_score = self._validate_indonesian_pattern(text)
        if pattern_score < 30:
            self.logger.debug(f"❌ Invalid Indonesian pattern: '{text}' (score: {pattern_score})")
            return None
        
        # Boost confidence based on pattern
        final_confidence = confidence + pattern_score * 0.3
        final_confidence = min(100, final_confidence)
        
        self.logger.info(f"✅ Plate detected: '{text}' ({final_confidence:.1f}%) via {candidate['method']}")
        
        return PlateDetection(
            text=text,
            confidence=final_confidence,
            bbox=bbox,
            processed_image=roi.copy(),
            timestamp=time.time(),
            vehicle_type="vehicle",
            detection_method=f"final_{candidate['method']}"
        )
    
    def _enhanced_ocr(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Enhanced OCR processing dengan multiple approaches
        """
        if roi.size == 0:
            return "", 0.0
        
        results = []
        
        # Preprocessing variations
        preprocessed_rois = self._prepare_roi_variants(roi)
        
        for variant_name, processed_roi in preprocessed_rois:
            # Multiple OCR configurations
            configs = [
                ('psm8', '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
                ('psm7', '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'),
                ('psm13', '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            ]
            
            for config_name, config in configs:
                try:
                    # Get OCR result
                    data = pytesseract.image_to_data(
                        processed_roi,
                        lang='eng',
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract text and confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    texts = [text.strip() for text in data['text'] if text.strip()]
                    
                    if texts and confidences:
                        full_text = ''.join(texts).upper()
                        avg_confidence = np.mean(confidences)
                        
                        # Clean text
                        cleaned_text = self._clean_plate_text(full_text)
                        
                        if len(cleaned_text) >= 3:
                            # Boost confidence for good preprocessing
                            if variant_name in ['enhanced', 'upscaled']:
                                avg_confidence += 5
                            
                            results.append((cleaned_text, avg_confidence, f"{variant_name}_{config_name}"))
                
                except Exception:
                    continue
        
        if results:
            # Sort by confidence and text length
            results.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            return results[0][0], results[0][1]
        
        return "", 0.0
    
    def _prepare_roi_variants(self, roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Prepare multiple variants of ROI for OCR
        """
        variants = []
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Original
        variants.append(("original", gray))
        
        # Upscaled version
        h, w = gray.shape
        if h < 40 or w < 120:
            scale = max(2, 40 // h, 120 // w)
            upscaled = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            variants.append(("upscaled", upscaled))
        
        # Enhanced contrast
        enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=10)
        variants.append(("enhanced", enhanced))
        
        # Denoised
        denoised = cv2.bilateralFilter(gray, 5, 50, 50)
        variants.append(("denoised", denoised))
        
        # Sharpened
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        variants.append(("sharpened", sharpened))
        
        return variants
    
    def _clean_plate_text(self, text: str) -> str:
        """
        Clean OCR text untuk Indonesian plates
        """
        if not text:
            return ""
        
        # Remove non-alphanumeric
        cleaned = ''.join(c for c in text.upper() if c.isalnum())
        
        # Common OCR corrections for Indonesian plates
        corrections = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G'
        }
        
        # Apply smart corrections
        if len(cleaned) >= 4:
            # First 1-2 characters usually letters
            for i in range(min(2, len(cleaned))):
                if cleaned[i].isdigit() and cleaned[i] in corrections:
                    cleaned = cleaned[:i] + corrections[cleaned[i]] + cleaned[i+1:]
            
            # Last 1-3 characters usually letters
            for i in range(max(2, len(cleaned) - 3), len(cleaned)):
                if i < len(cleaned) and cleaned[i].isdigit() and cleaned[i] in corrections:
                    cleaned = cleaned[:i] + corrections[cleaned[i]] + cleaned[i+1:]
        
        return cleaned
    
    def _validate_indonesian_pattern(self, text: str) -> int:
        """
        Validate Indonesian license plate pattern
        """
        if not text or len(text) < 4:
            return 0
        
        text = text.upper().strip()
        
        # Indonesian plate patterns dengan scoring
        import re
        
        patterns = [
            (r'^[A-Z]\\d{3,4}[A-Z]{2,3}$', 100),  # B1234ABC
            (r'^[A-Z]{2}\\d{3,4}[A-Z]{1,2}$', 95),  # AB1234C
            (r'^[A-Z]\\d{1,4}[A-Z]{1,3}$', 80),  # B123A
            (r'^[A-Z]{2,3}\\d{2,4}$', 70),  # AB1234
            (r'^\\d{2,4}[A-Z]{2,4}$', 65),  # 1234AB
            (r'^[A-Z]{3,6}$', 60),  # ABCDEF (partial)
            (r'^\\d{3,5}$', 50),  # 12345 (partial)
        ]
        
        for pattern, score in patterns:
            if re.match(pattern, text):
                return score
        
        # Additional scoring untuk fragments
        if len(text) >= 3:
            has_letters = any(c.isalpha() for c in text)
            has_numbers = any(c.isdigit() for c in text)
            
            if has_letters and has_numbers:
                return 50
            elif has_letters:
                return 40
            elif has_numbers:
                return 30
        
        return 20  # Minimum score for any 3+ character string
    
    def _final_validation(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """
        Final validation and ranking
        """
        if not detections:
            return []
        
        # Remove duplicates berdasarkan overlap
        final_detections = []
        
        for detection in detections:
            bbox1 = detection.bbox
            is_duplicate = False
            
            for existing in final_detections:
                bbox2 = existing.bbox
                overlap = self._calculate_overlap(bbox1, bbox2)
                
                if overlap > 0.3:
                    # Keep the one with higher confidence
                    if detection.confidence > existing.confidence:
                        final_detections.remove(existing)
                    else:
                        is_duplicate = True
                    break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        # Sort by confidence
        final_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return final_detections
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap ratio between two bounding boxes
        """
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

def test_final_detector():
    """
    Test function untuk final detector
    """
    print("🎯 Testing Final Plate Detector")
    print("=" * 50)
    
    detector = FinalPlateDetector()
    
    # Test dengan sample image
    test_images = [
        "detected_plates/screenshot_20250919_092204.jpg",
        "optimized_plate_test_20250919_100153.jpg"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            continue
        
        print(f"\\n📸 Testing: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        start_time = time.time()
        detections = detector.detect_plates(image)
        detection_time = time.time() - start_time
        
        print(f"⏱️ Detection time: {detection_time:.2f}s")
        print(f"📊 Found {len(detections)} plates")
        
        for i, detection in enumerate(detections):
            print(f"   {i+1}. '{detection.text}' ({detection.confidence:.1f}%) "
                  f"method: {detection.detection_method}")

if __name__ == "__main__":
    import os
    test_final_detector()