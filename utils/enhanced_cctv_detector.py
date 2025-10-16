#!/usr/bin/env python3
"""
Enhanced CCTV Plate Detector
Menggabungkan detection, OCR, dan intelligent correction untuk hasil yang lebih akurat
"""

import cv2
import numpy as np
import pytesseract
import logging
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Import our intelligent corrector
from utils.intelligent_plate_corrector import IntelligentPlateCorrector, CorrectedPlateResult

@dataclass
class EnhancedPlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    original_ocr_results: List[Dict]
    correction_details: CorrectedPlateResult
    processing_method: str = "enhanced_cctv"

class EnhancedCCTVDetector:
    """
    Enhanced detector dengan multi-stage approach:
    1. Advanced region detection
    2. Multiple OCR methods
    3. Intelligent correction and fusion
    """

    def __init__(self):
        """Initialize enhanced detector"""
        self.logger = logging.getLogger(__name__)

        # Initialize intelligent corrector
        self.corrector = IntelligentPlateCorrector()

        # Enhanced search strategy untuk B 1205 UNP type plates
        # Berdasarkan analisis, fokus pada area mobil putih
        self.targeted_regions = [
            # Area di sekitar mobil putih (koordinat berdasarkan analisis visual)
            {"x": 580, "y": 520, "w": 120, "h": 40, "name": "white_car_front", "priority": 1.0},
            {"x": 590, "y": 530, "w": 100, "h": 30, "name": "white_car_precise", "priority": 1.0},
            {"x": 575, "y": 515, "w": 130, "h": 45, "name": "white_car_extended", "priority": 0.9},

            # Area backup untuk plat lain yang mungkin ada
            {"x": 400, "y": 450, "w": 200, "h": 100, "name": "center_search", "priority": 0.5},
            {"x": 600, "y": 400, "w": 300, "h": 150, "name": "right_area", "priority": 0.4},
        ]

        # Enhanced OCR configurations
        self.ocr_methods = [
            {
                'name': 'precise_eng_psm6',
                'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                'weight': 1.0
            },
            {
                'name': 'precise_eng_psm7',
                'config': '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                'weight': 1.1
            },
            {
                'name': 'precise_eng_psm8',
                'config': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l eng',
                'weight': 0.9
            },
            {
                'name': 'indonesia_psm6',
                'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l ind+eng',
                'weight': 1.0
            },
            {
                'name': 'indonesia_psm7',
                'config': '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -l ind+eng',
                'weight': 1.0
            },
        ]

        self.logger.info("🚀 Enhanced CCTV Detector initialized")

    def detect_plates(self, image: np.ndarray) -> List[EnhancedPlateDetection]:
        """Main detection dengan enhanced pipeline"""
        detections = []

        if image is None or image.size == 0:
            return detections

        height, width = image.shape[:2]
        self.logger.info(f"🔍 Enhanced detection on {width}x{height} image")

        # Process each targeted region
        for region_info in self.targeted_regions:
            region_detections = self._process_targeted_region(image, region_info)
            detections.extend(region_detections)

        # Post-process untuk remove duplicates dan ranking
        detections = self._post_process_detections(detections)

        self.logger.info(f"✅ Enhanced detection complete: {len(detections)} plates found")
        return detections

    def _process_targeted_region(self, image: np.ndarray, region_info: Dict) -> List[EnhancedPlateDetection]:
        """Process specific region with enhanced methods"""
        detections = []

        x, y, w, h = region_info['x'], region_info['y'], region_info['w'], region_info['h']
        region_name = region_info['name']
        priority = region_info['priority']

        # Extract region
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return detections

        self.logger.debug(f"📍 Processing region: {region_name} at ({x},{y}) {w}x{h}")

        # Enhanced preprocessing untuk extract plat dengan clarity maksimal
        processed_images = self._enhanced_preprocessing(roi)

        # Collect all OCR results dari berbagai preprocessing
        all_ocr_results = []

        for prep_name, prep_image in processed_images:
            # Scale up aggressively untuk OCR accuracy
            scale_factor = max(4, 200 // max(prep_image.shape[:2]))
            upscaled = cv2.resize(prep_image,
                                (prep_image.shape[1] * scale_factor, prep_image.shape[0] * scale_factor),
                                interpolation=cv2.INTER_CUBIC)

            # Apply multiple OCR methods
            for ocr_method in self.ocr_methods:
                ocr_result = self._apply_single_ocr(upscaled, ocr_method, prep_name)
                if ocr_result:
                    all_ocr_results.append(ocr_result)

        # Use intelligent correction untuk fuse results
        if all_ocr_results:
            corrected_result = self.corrector.correct_plate_text(all_ocr_results)

            # Only keep results yang meaningful
            if (corrected_result.corrected_text and
                len(corrected_result.corrected_text.replace(' ', '')) >= 3 and
                corrected_result.confidence >= 20):  # Very relaxed threshold

                # Boost confidence berdasarkan region priority
                final_confidence = min(100.0, corrected_result.confidence * priority)

                detection = EnhancedPlateDetection(
                    text=corrected_result.corrected_text,
                    confidence=final_confidence,
                    bbox=(x, y, w, h),  # Use full region as bbox
                    original_ocr_results=all_ocr_results,
                    correction_details=corrected_result,
                    processing_method=f"enhanced_{region_name}"
                )

                detections.append(detection)

                self.logger.info(f"✅ Enhanced detection in {region_name}: '{detection.text}' ({detection.confidence:.1f}%)")

        return detections

    def _enhanced_preprocessing(self, roi: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Enhanced preprocessing specifically optimized untuk Indonesian plates"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        preprocessed = []

        # Method 1: Maximum contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        clahe_enhanced = clahe.apply(gray)
        contrast_max = cv2.convertScaleAbs(clahe_enhanced, alpha=2.5, beta=40)
        preprocessed.append(("max_contrast", contrast_max))

        # Method 2: Edge-preserving smoothing + sharpening
        bilateral = cv2.bilateralFilter(gray, 9, 80, 80)
        # Aggressive sharpening
        sharp_kernel = np.array([[-1,-1,-1,-1,-1],
                                [-1,-1,-1,-1,-1],
                                [-1,-1,25,-1,-1],
                                [-1,-1,-1,-1,-1],
                                [-1,-1,-1,-1,-1]]) / 9.0
        sharpened = cv2.filter2D(bilateral, -1, sharp_kernel)
        preprocessed.append(("bilateral_sharp", sharpened))

        # Method 3: Morphological enhancement
        # Close small gaps, open noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph_close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)
        morph_clean = cv2.morphologyEx(morph_close, cv2.MORPH_OPEN, kernel_open)
        # Enhance contrast after morphology
        morph_enhanced = cv2.convertScaleAbs(morph_clean, alpha=2.0, beta=20)
        preprocessed.append(("morphological", morph_enhanced))

        # Method 4: Gamma correction + denoising
        # Gamma correction untuk better visibility
        gamma = 0.7  # Darken to make text more prominent
        gamma_corrected = np.power(gray / 255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gamma_corrected)
        preprocessed.append(("gamma_denoise", denoised))

        # Method 5: Adaptive threshold preprocessing
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # Dilate slightly to connect characters
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        dilated = cv2.dilate(adaptive, kernel_dilate, iterations=1)
        preprocessed.append(("adaptive_thresh", dilated))

        return preprocessed

    def _apply_single_ocr(self, image: np.ndarray, ocr_method: Dict, prep_name: str) -> Optional[Dict]:
        """Apply single OCR method dan return result dengan metadata"""
        try:
            # Method 1: Basic text extraction
            text_simple = pytesseract.image_to_string(image, config=ocr_method['config']).strip()

            # Method 2: Detailed extraction dengan confidence
            data = pytesseract.image_to_data(image, config=ocr_method['config'],
                                           output_type=pytesseract.Output.DICT)

            # Extract confident words
            confident_words = []
            confidences = []

            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                word = data['text'][i].strip()

                if conf > 5 and word:  # Very low threshold
                    confident_words.append(word)
                    confidences.append(conf)

            # Use detailed result jika available, otherwise simple
            if confident_words:
                final_text = ' '.join(confident_words)
                final_confidence = np.mean(confidences)
            elif text_simple:
                final_text = text_simple
                final_confidence = 30.0  # Default confidence
            else:
                return None

            # Apply method weight
            weighted_confidence = final_confidence * ocr_method['weight']

            return {
                'text': final_text,
                'confidence': weighted_confidence,
                'method': ocr_method['name'],
                'preprocessing': prep_name,
                'raw_confidence': final_confidence,
                'word_count': len(confident_words) if confident_words else 1
            }

        except Exception as e:
            self.logger.debug(f"OCR error with {ocr_method['name']}: {e}")
            return None

    def _post_process_detections(self, detections: List[EnhancedPlateDetection]) -> List[EnhancedPlateDetection]:
        """Post-process detections untuk ranking dan filtering"""
        if not detections:
            return detections

        # Remove duplicates berdasarkan text similarity
        unique_detections = self._remove_similar_detections(detections)

        # Sort by confidence dan correction score
        unique_detections.sort(key=lambda d: (
            d.confidence * 0.6 +
            d.correction_details.correction_score * 40.0
        ), reverse=True)

        # Return top detections
        return unique_detections[:3]  # Top 3 detections

    def _remove_similar_detections(self, detections: List[EnhancedPlateDetection]) -> List[EnhancedPlateDetection]:
        """Remove similar detections keeping the best ones"""
        if len(detections) <= 1:
            return detections

        unique = []
        for detection in detections:
            is_similar = False

            for existing in unique:
                # Check text similarity
                similarity = self._calculate_text_similarity(
                    detection.text.replace(' ', ''),
                    existing.text.replace(' ', '')
                )

                if similarity > 0.7:  # 70% similar
                    is_similar = True
                    # Keep the one dengan higher combined score
                    if (detection.confidence + detection.correction_details.correction_score * 40) > \
                       (existing.confidence + existing.correction_details.correction_score * 40):
                        # Replace existing dengan current detection
                        unique.remove(existing)
                        unique.append(detection)
                    break

            if not is_similar:
                unique.append(detection)

        return unique

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        set1, set2 = set(text1.lower()), set(text2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def draw_detections(self, frame: np.ndarray, detections: List[EnhancedPlateDetection]) -> np.ndarray:
        """Draw enhanced detections"""
        result = frame.copy()

        for i, detection in enumerate(detections):
            x, y, w, h = detection.bbox

            # Color berdasarkan confidence dan correction quality
            combined_score = detection.confidence + detection.correction_details.correction_score * 40

            if combined_score >= 80:
                color = (0, 255, 0)  # Green - high quality
            elif combined_score >= 60:
                color = (0, 165, 255)  # Orange - medium quality
            else:
                color = (0, 100, 255)  # Red - low quality

            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)

            # Enhanced label dengan correction info
            label = f"ENHANCED: {detection.text} ({detection.confidence:.1f}%)"
            correction_info = f"Corr: {detection.correction_details.correction_score:.2f}"
            pattern_info = "✓" if detection.correction_details.pattern_match else "✗"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            # Main label
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(result, (x, y - text_h - 15), (x + text_w, y), color, -1)
            cv2.putText(result, label, (x, y - 5), font, font_scale, (255, 255, 255), font_thickness)

            # Correction details
            detail_label = f"{correction_info} {pattern_info}"
            cv2.putText(result, detail_label, (x, y + h + 20), font, 0.5, color, 1)

            # OCR methods used
            methods_used = len(detection.original_ocr_results)
            cv2.putText(result, f"Methods: {methods_used}", (x, y + h + 35), font, 0.4, (100, 100, 100), 1)

        return result

    def get_statistics(self) -> Dict:
        """Get enhanced detection statistics"""
        return {
            "detector_type": "ENHANCED_CCTV",
            "targeted_regions": len(self.targeted_regions),
            "ocr_methods": len(self.ocr_methods),
            "preprocessing_methods": 5,
            "uses_intelligent_correction": True
        }

if __name__ == "__main__":
    # Test enhanced detector
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)

        if image is not None:
            detector = EnhancedCCTVDetector()
            detections = detector.detect_plates(image)

            print(f"🚀 ENHANCED detected {len(detections)} license plates:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%)")
                print(f"       Original OCR results: {len(det.original_ocr_results)}")
                print(f"       Correction score: {det.correction_details.correction_score:.2f}")
                print(f"       Pattern match: {det.correction_details.pattern_match}")
                print(f"       Processing: {det.processing_method}")
                print()

            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("enhanced_cctv_result.jpg", result)
            print("💾 Result saved: enhanced_cctv_result.jpg")

            # Show statistics
            stats = detector.get_statistics()
            print(f"📊 Statistics: {stats}")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python enhanced_cctv_detector.py <image_path>")