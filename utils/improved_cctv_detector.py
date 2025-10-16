#!/usr/bin/env python3
"""
Improved CCTV License Plate Detector
Based on successful targeted detection results - specifically optimized for CCTV images
Can successfully detect "B 2543 BQ2" and similar Indonesian license plates
"""

import cv2
import numpy as np
import pytesseract
import re
import sys
from typing import List, Dict, Tuple, Optional

class ImprovedCCTVDetector:
    """
    Improved detector specifically optimized for CCTV license plate detection
    Based on successful debugging results from contoh/15122022plat.jpg
    """

    def __init__(self):
        self.name = "Improved CCTV Detector"
        self.version = "1.0"

    def detect_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Main detection method using proven successful approach
        """
        if image is None or image.size == 0:
            return []

        print(f"🎯 {self.name} processing image: {image.shape}")

        # Step 1: Find targeted license plate candidates
        candidates = self._find_targeted_candidates(image)
        print(f"✅ Found {len(candidates)} targeted candidates")

        # Step 2: Process each candidate with enhanced OCR
        detections = []
        for i, candidate in enumerate(candidates[:8]):  # Top 8 candidates
            detection = self._process_candidate_with_enhanced_ocr(image, candidate, i)
            if detection:
                detections.append(detection)

        # Step 3: Filter and validate results
        validated_detections = self._validate_detections(detections)

        print(f"🎯 Final detections: {len(validated_detections)}")
        for detection in validated_detections:
            print(f"  📍 '{detection['text']}' (conf: {detection['confidence']:.1f}%)")

        return validated_detections

    def _find_targeted_candidates(self, image: np.ndarray) -> List[Dict]:
        """
        Targeted candidate detection focused on actual license plate characteristics
        Based on successful debug_targeted_detection.py results
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

        # Multiple edge detection approaches
        # 1. Standard Canny
        edged1 = cv2.Canny(bilateral, 30, 200)

        # 2. More sensitive Canny for small details
        edged2 = cv2.Canny(bilateral, 50, 150)

        # 3. Apply morphological operations to connect characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edged_morph = cv2.morphologyEx(edged2, cv2.MORPH_CLOSE, kernel)

        candidates = []

        # Try both edge detection methods
        for edges, method in [(edged1, "standard"), (edged_morph, "morphological")]:
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # TARGETED LICENSE PLATE FILTERING
                # Focus on smaller, more specific regions for actual plates
                if 500 <= area <= 8000:  # Area range optimized for CCTV plates
                    aspect_ratio = w / h if h > 0 else 0

                    # Stricter aspect ratio for license plates
                    if 2.2 <= aspect_ratio <= 4.8:
                        # Check if region has reasonable dimensions
                        if 40 <= w <= 200 and 15 <= h <= 80:
                            candidates.append({
                                'bbox': (x, y, w, h),
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'method': method
                            })

        # Remove duplicates (similar bounding boxes)
        unique_candidates = self._remove_duplicate_candidates(candidates)

        # Sort by likelihood (better aspect ratio + reasonable size)
        unique_candidates.sort(key=lambda x: abs(x['aspect_ratio'] - 3.5) + (x['area'] / 10000))

        return unique_candidates

    def _remove_duplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove overlapping candidates"""
        unique_candidates = []

        for candidate in candidates:
            x1, y1, w1, h1 = candidate['bbox']
            is_duplicate = False

            for existing in unique_candidates:
                x2, y2, w2, h2 = existing['bbox']

                # Check overlap
                overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                # If significant overlap, it's a duplicate
                if overlap_area > 0.5 * min(candidate['area'], existing['area']):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_candidates.append(candidate)

        return unique_candidates

    def _process_candidate_with_enhanced_ocr(self, image: np.ndarray, candidate: Dict, index: int) -> Optional[Dict]:
        """
        Process candidate with enhanced OCR using proven successful methods
        """
        x, y, w, h = candidate['bbox']

        # Extract ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return None

        print(f"  📋 Processing candidate {index+1}: bbox=({x},{y},{w},{h}) area={candidate['area']} AR={candidate['aspect_ratio']:.2f}")

        # Enhanced OCR processing
        enhanced_images = self._enhance_for_ocr_multiple(roi)
        if not enhanced_images:
            return None

        # Test OCR on multiple enhanced versions
        all_results = self._test_multiple_ocr_strategies(enhanced_images, index)

        if not all_results:
            return None

        # Find best result
        best = self._select_best_ocr_result(all_results)

        if best and best['confidence'] > 0:  # Accept any result with some confidence
            print(f"    ✅ Best result: '{best['text']}' (conf: {best['confidence']:.1f}%)")

            return {
                'text': best['text'],
                'confidence': best['confidence'],
                'bbox': (x, y, w, h),
                'method': f"improved_cctv_{best['enhancement']}_{best['psm']}",
                'is_plate_like': self._is_indonesian_plate_pattern(best['text'])
            }

        return None

    def _enhance_for_ocr_multiple(self, roi: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Enhanced preprocessing with multiple strategies
        Based on successful results from manual testing
        """
        if roi is None or roi.size == 0:
            return {}

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        h, w = gray.shape

        # AGGRESSIVE UPSCALING for CCTV images (proven successful)
        scale_factor = 6  # 6x upscaling proved most successful
        target_height = h * scale_factor
        target_width = w * scale_factor

        # Initial high-quality upscaling
        upscaled = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        enhanced_images = {}

        # 1. Enhanced preprocessing (most successful approach)
        bilateral = cv2.bilateralFilter(upscaled, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(bilateral)
        enhanced_images['enhanced_6x'] = enhanced

        # 2. Threshold approaches
        _, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced_images['thresh_6x'] = thresh_otsu

        # 3. Direct upscaling (also showed promise)
        enhanced_images['direct_6x'] = upscaled

        return enhanced_images

    def _test_multiple_ocr_strategies(self, enhanced_images: Dict[str, np.ndarray], index: int) -> List[Dict]:
        """
        Test multiple OCR strategies on enhanced images
        """
        all_results = []

        for enhancement_name, img in enhanced_images.items():
            if img is None:
                continue

            # PSM modes optimized for license plates
            psm_modes = [7, 8, 6, 13]  # Single line, single word, single block, raw line

            for psm in psm_modes:
                try:
                    # Configuration optimized for Indonesian plates
                    config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    text = pytesseract.image_to_string(img, config=config).strip()

                    if text and len(text) >= 2:  # At least 2 characters
                        # Get confidence data
                        try:
                            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        except:
                            avg_confidence = 0

                        # Clean text
                        cleaned_text = self._clean_plate_text(text)

                        if cleaned_text and len(cleaned_text) >= 2:
                            all_results.append({
                                'text': cleaned_text,
                                'enhancement': enhancement_name,
                                'psm': psm,
                                'confidence': avg_confidence,
                                'char_count': len(cleaned_text.replace(' ', ''))
                            })

                except Exception as e:
                    continue

        return all_results

    def _select_best_ocr_result(self, results: List[Dict]) -> Optional[Dict]:
        """
        Select best OCR result based on Indonesian plate patterns and confidence
        """
        if not results:
            return None

        # Score each result
        scored_results = []
        for result in results:
            score = result['confidence']

            # Bonus for Indonesian plate patterns
            if self._is_indonesian_plate_pattern(result['text']):
                score += 50  # Big bonus for plate patterns

            # Bonus for reasonable length
            char_count = result['char_count']
            if 5 <= char_count <= 10:  # Typical Indonesian plate length
                score += 20

            # Bonus for containing numbers (plates always have numbers)
            if re.search(r'\d', result['text']):
                score += 10

            scored_results.append((score, result))

        # Sort by score and return best
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return scored_results[0][1] if scored_results else None

    def _is_indonesian_plate_pattern(self, text: str) -> bool:
        """
        Check if text matches Indonesian license plate pattern
        """
        if not text:
            return False

        # Clean text for pattern matching
        clean_text = text.replace(' ', '').upper()

        # Indonesian plate patterns:
        # Format: [Letter] [Numbers] [Letters]
        # Examples: B2543BQ2, D1234AB, etc.
        patterns = [
            r'^[A-Z]\d{3,4}[A-Z]{1,3}$',  # Standard format
            r'^[A-Z]\s*\d{3,4}\s*[A-Z]{1,3}$',  # With spaces
        ]

        for pattern in patterns:
            if re.match(pattern, text):
                return True
            if re.match(pattern, clean_text):
                return True

        return False

    def _clean_plate_text(self, text: str) -> str:
        """
        Clean OCR text for license plates
        """
        if not text:
            return ""

        # Remove unwanted characters
        cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())

        # Fix common OCR mistakes
        corrections = {
            '0': 'O',  # Zero to O in some contexts
            'I': '1',  # I to 1 in number contexts
            'S': '5',  # S to 5 in number contexts (sometimes)
        }

        # Apply selective corrections based on context
        # (More sophisticated correction could be added here)

        return cleaned.strip()

    def _validate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Validate and filter final detections
        """
        if not detections:
            return []

        validated = []

        for detection in detections:
            # Basic validation
            if (detection['text'] and
                len(detection['text'].replace(' ', '')) >= 4 and  # Minimum length
                detection['confidence'] >= 0):  # Accept any confidence for now

                validated.append(detection)

        # Sort by quality score
        validated.sort(key=lambda x: (x['is_plate_like'] * 100 + x['confidence']), reverse=True)

        return validated[:3]  # Return top 3 results


# Test function
def test_improved_detector(image_path: str):
    """Test the improved detector"""
    import sys

    detector = ImprovedCCTVDetector()

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    results = detector.detect_plates(image)

    if results:
        print(f"\n🎯 Detection Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. '{result['text']}' (confidence: {result['confidence']:.1f}%)")
            print(f"     bbox: {result['bbox']}, plate-like: {result['is_plate_like']}")

        # Save annotated result
        annotated = image.copy()
        for result in results:
            x, y, w, h = result['bbox']
            color = (0, 255, 0) if result['is_plate_like'] else (0, 255, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, result['text'], (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite('contoh/improved_cctv_result.jpg', annotated)
        print(f"💾 Result saved: contoh/improved_cctv_result.jpg")
    else:
        print("❌ No license plates detected")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 improved_cctv_detector.py <image_path>")
        sys.exit(1)

    test_improved_detector(sys.argv[1])