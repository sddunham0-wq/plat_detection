"""
Motorcycle License Plate Detector
Specialized detector untuk plat nomor motor dengan optimasi untuk:
- Multi-motorcycle detection di parking lot
- Jarak jauh dengan plat kecil
- CCTV view angle dengan berbagai sudut
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from config import MotorcycleDetectionConfig, TesseractConfig

try:
    from utils.indonesian_ocr import IndonesianPlateOCR
except ImportError:
    IndonesianPlateOCR = None

# Import PlateDetection to avoid circular import
@dataclass
class PlateDetection:
    """Data class untuk hasil deteksi plat - compatible with main detector"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    processed_image: np.ndarray
    timestamp: float
    vehicle_type: str = "motorcycle"
    detection_method: str = "motorcycle_optimized"

class MotorcyclePlateDetector:
    """
    Specialized detector untuk plat nomor motor
    Optimized untuk CCTV parking lot dengan multiple motorcycles
    """
    
    def __init__(self, confidence=0.5):
        """Initialize motorcycle detector"""
        self.logger = logging.getLogger(__name__)
        self.config = MotorcycleDetectionConfig
        self.is_enabled_flag = self.config.MOTORCYCLE_PRIORITY
        self.confidence_threshold = confidence
        
        # Initialize Indonesian OCR
        if IndonesianPlateOCR:
            try:
                self.indonesian_ocr = IndonesianPlateOCR()
                self.logger.info("Indonesian OCR initialized for motorcycle detection")
            except Exception as e:
                self.logger.warning(f"Indonesian OCR not available: {e}")
                self.indonesian_ocr = None
        else:
            self.indonesian_ocr = None
            self.logger.warning("Indonesian OCR module not found")
        
        self.logger.info(f"Motorcycle detector initialized - Enabled: {self.is_enabled_flag}")
    
    def is_enabled(self) -> bool:
        """Check if motorcycle detector is enabled"""
        return self.is_enabled_flag
    
    def preprocess_for_motorcycles(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Preprocessing khusus untuk motor dengan multiple enhancement methods
        
        Args:
            image: Input image
            
        Returns:
            List of preprocessed images
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        processed_images = []
        
        # 1. Standard preprocessing
        standard = cv2.GaussianBlur(gray, (3, 3), 0)
        processed_images.append(("standard", standard))
        
        # 2. Enhanced contrast untuk plat motor (sering gelap)
        enhanced = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        processed_images.append(("enhanced", enhanced))
        
        # 3. CLAHE untuk extreme low contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        processed_images.append(("clahe", clahe_img))
        
        # 4. Bilateral filter untuk noise reduction tapi preserve edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        processed_images.append(("bilateral", bilateral))
        
        return processed_images
    
    def detect_motorcycle_contours(self, image: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        """
        Enhanced contour detection khusus untuk motor
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            List of (contour, metadata) tuples
        """
        contours_found = []
        
        # Enhanced multiple thresholding methods for better stability
        thresh_methods = [
            ("binary", cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive_mean", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)),  # Reduced block size
            ("adaptive_gaussian", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)),  # Reduced block size
            ("binary_inv", cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),  # Additional method
        ]
        
        for method_name, thresh in thresh_methods:
            # Enhanced morphological operations untuk cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Horizontal kernel for plates
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # Additional noise removal
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))  # Vertical kernel
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel2)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check area constraints
                if self.config.MIN_CONTOUR_AREA <= area <= self.config.MAX_CONTOUR_AREA:
                    # Get bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check size constraints
                    if (w >= self.config.MIN_PLATE_WIDTH and h >= self.config.MIN_PLATE_HEIGHT and
                        w <= self.config.MAX_PLATE_WIDTH and h <= self.config.MAX_PLATE_HEIGHT):
                        
                        # Check aspect ratio
                        aspect_ratio = w / h if h > 0 else 0
                        if self.config.MIN_ASPECT_RATIO <= aspect_ratio <= self.config.MAX_ASPECT_RATIO:
                            
                            metadata = {
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'bbox': (x, y, w, h),
                                'method': method_name,
                                'score': self._calculate_motorcycle_score(area, aspect_ratio, w, h)
                            }
                            
                            contours_found.append((contour, metadata))
        
        # Sort by score dan remove duplicates
        contours_found.sort(key=lambda x: x[1]['score'], reverse=True)
        return self._remove_duplicate_contours(contours_found)
    
    def _calculate_motorcycle_score(self, area: float, aspect_ratio: float, width: int, height: int) -> float:
        """
        Calculate score untuk motorcycle plate candidacy
        
        Args:
            area: Contour area
            aspect_ratio: Width/height ratio
            width: Bounding box width
            height: Bounding box height
            
        Returns:
            Score (0-100)
        """
        score = 0.0
        
        # Area score (prefer moderate sizes typical for motor plates) - more tolerant
        optimal_area = 1500  # Reduced sweet spot untuk motor plates di CCTV
        area_tolerance = optimal_area * 1.5  # More tolerance
        area_score = 100 - min(100, abs(area - optimal_area) / area_tolerance * 100)
        score += area_score * 0.35
        
        # Aspect ratio score (motor plates are typically 1.5-4.0) - more tolerant
        if 1.8 <= aspect_ratio <= 3.5:  # Perfect range
            ratio_score = 100
        elif 1.5 <= aspect_ratio <= 4.5:  # Good range
            ratio_score = 80
        else:  # Outside range
            ratio_score = max(0, 60 - abs(aspect_ratio - 2.5) * 10)
        score += ratio_score * 0.35
        
        # Size score (prefer medium sizes) - more balanced
        size_score = min(100, (width * height) / 3000 * 100)
        score += size_score * 0.2
        
        # Bonus for typical motorcycle plate dimensions
        if 40 <= width <= 120 and 15 <= height <= 40:
            score += 10  # Bonus untuk ukuran tipikal
        
        return max(0, min(100, score))
    
    def _remove_duplicate_contours(self, contours_with_meta: List[Tuple[np.ndarray, dict]]) -> List[Tuple[np.ndarray, dict]]:
        """Remove overlapping/duplicate contours"""
        if len(contours_with_meta) <= 1:
            return contours_with_meta
        
        filtered = []
        overlap_threshold = 0.7  # Increased from 0.5 to 0.7 - less aggressive duplicate removal
        
        for i, (contour1, meta1) in enumerate(contours_with_meta):
            bbox1 = meta1['bbox']
            is_duplicate = False
            
            for j, (contour2, meta2) in enumerate(filtered):
                bbox2 = meta2['bbox']
                overlap = self._calculate_bbox_overlap(bbox1, bbox2)
                
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append((contour1, meta1))
        
        return filtered
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
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
    
    def _analyze_plate_texture(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> dict:
        """
        Analyze texture properties to distinguish real plates from false positives
        
        Args:
            image: Original image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Dict with texture analysis scores
        """
        x, y, w, h = bbox
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return {'text_likelihood': 0, 'edge_density': 0, 'texture_variance': 0, 'is_valid': False}
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # 1. Edge Density Analysis - plates have high edge density due to text
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h) * 100
        
        # 2. Texture Variance - text areas have high intensity variance
        texture_variance = np.std(roi_gray)
        
        # 3. Intensity Distribution - plates have bimodal distribution (text + background)
        hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
        hist_peaks = len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
        
        # 4. Horizontal Line Detection - plates often have horizontal text lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w//4, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / (w * h) * 100
        
        # 5. Character-like Pattern Detection
        char_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        char_patterns = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, char_kernel)
        char_score = np.sum(char_patterns > 0) / (w * h) * 100
        
        # 6. Surface Smoothness - spion and motor parts are usually smoother
        laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        
        # Calculate text likelihood score (0-100)
        text_likelihood = 0
        
        # Edge density contribution (plates typically have 5-25% edge density)
        if 5 <= edge_density <= 25:
            text_likelihood += 30
        elif 2 <= edge_density <= 30:
            text_likelihood += 20
        
        # Texture variance contribution (text creates variance)
        if texture_variance > 15:
            text_likelihood += 25
        elif texture_variance > 10:
            text_likelihood += 15
        
        # Horizontal lines contribution
        if horizontal_score > 2:
            text_likelihood += 20
        elif horizontal_score > 1:
            text_likelihood += 10
        
        # Character patterns contribution
        if char_score > 8:
            text_likelihood += 15
        elif char_score > 5:
            text_likelihood += 10
        
        # Surface roughness contribution (plates are not too smooth)
        if laplacian_var > 100:
            text_likelihood += 10
        elif laplacian_var > 50:
            text_likelihood += 5
        
        # Histogram peaks (text usually creates 2-4 peaks)
        if 2 <= hist_peaks <= 4:
            text_likelihood += 10
        
        # Determine if this looks like a valid plate
        is_valid = (
            text_likelihood >= 50 and  # Minimum text likelihood
            edge_density >= 3 and      # Minimum edge density
            texture_variance >= 8 and  # Minimum texture variance
            laplacian_var >= 30        # Minimum surface roughness
        )
        
        return {
            'text_likelihood': min(100, text_likelihood),
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'horizontal_score': horizontal_score,
            'char_score': char_score,
            'laplacian_var': laplacian_var,
            'hist_peaks': hist_peaks,
            'is_valid': is_valid
        }
    
    def _detect_false_positive_patterns(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> dict:
        """
        Detect common false positive patterns (spion, motor parts, etc.)
        
        Args:
            image: Original image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Dict with false positive indicators
        """
        x, y, w, h = bbox
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return {'is_false_positive': True, 'reason': 'empty_roi'}
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        false_positive_indicators = []
        confidence_penalty = 0
        
        # 1. Check for circular/elliptical shapes (spion)
        contours, _ = cv2.findContours(cv2.Canny(roi_gray, 50, 150), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Check circularity
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:  # High circularity suggests circular object
                    false_positive_indicators.append('circular_shape')
                    confidence_penalty += 30
        
        # 2. Check for excessive smoothness (glossy surfaces)
        mean_intensity = np.mean(roi_gray)
        intensity_std = np.std(roi_gray)
        if intensity_std < 5 and mean_intensity > 100:  # Very smooth and bright
            false_positive_indicators.append('too_smooth_bright')
            confidence_penalty += 25
        
        # 3. Check for metallic reflection patterns
        # High intensity with low variation suggests metallic surface
        if mean_intensity > 150 and intensity_std < 10:
            false_positive_indicators.append('metallic_reflection')
            confidence_penalty += 20
        
        # 4. Check aspect ratio extremes (very thin or very wide)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 6 or aspect_ratio < 1.2:
            false_positive_indicators.append('extreme_aspect_ratio')
            confidence_penalty += 15
        
        # 5. Check for lack of text-like features
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h) * 100
        if edge_density < 2:  # Very low edge density
            false_positive_indicators.append('no_text_features')
            confidence_penalty += 25
        
        # 6. Check for uniform color regions (motor body parts)
        unique_colors = len(np.unique(roi_gray))
        if unique_colors < 10:  # Very few colors suggests uniform surface
            false_positive_indicators.append('uniform_surface')
            confidence_penalty += 20
        
        is_false_positive = len(false_positive_indicators) >= 2 or confidence_penalty >= 40
        
        return {
            'is_false_positive': is_false_positive,
            'indicators': false_positive_indicators,
            'confidence_penalty': confidence_penalty,
            'reason': ', '.join(false_positive_indicators) if false_positive_indicators else 'none'
        }
    
    def extract_and_enhance_plate(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract dan enhance plate region untuk optimal OCR
        
        Args:
            image: Original image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Enhanced plate image
        """
        x, y, w, h = bbox
        
        # Extract with padding
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        plate_roi = image[y1:y2, x1:x2]
        
        if plate_roi.size == 0:
            return np.zeros((20, 50), dtype=np.uint8)
        
        # Convert to grayscale if needed
        if len(plate_roi.shape) == 3:
            plate_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Upscaling untuk small plates
        if h < self.config.MIN_OCR_HEIGHT:
            scale_factor = self.config.UPSCALE_FACTOR
            if self.config.ENABLE_EXTREME_UPSCALING and h < 15:
                scale_factor = self.config.EXTREME_UPSCALE_FACTOR
            
            new_height = int(h * scale_factor)
            new_width = int(w * scale_factor)
            
            if self.config.USE_INTERPOLATION_CUBIC:
                plate_roi = cv2.resize(plate_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            else:
                plate_roi = cv2.resize(plate_roi, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Noise reduction untuk small plates
        if self.config.ENABLE_NOISE_REDUCTION:
            plate_roi = cv2.bilateralFilter(plate_roi, 5, 50, 50)
        
        # Enhance contrast
        plate_roi = cv2.convertScaleAbs(plate_roi, alpha=1.5, beta=10)
        
        # Sharpening untuk better OCR
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        plate_roi = cv2.filter2D(plate_roi, -1, kernel)
        
        return plate_roi
    
    def perform_motorcycle_ocr(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        OCR khusus untuk motorcycle plates
        
        Args:
            plate_image: Enhanced plate image
            
        Returns:
            Tuple of (text, confidence)
        """
        if plate_image.size == 0:
            return "", 0.0
        
        # Try Indonesian OCR first if available
        if self.indonesian_ocr:
            try:
                text, confidence = self.indonesian_ocr.recognize_plate(plate_image)
                if text and confidence >= 25:  # Lowered threshold for motorcycle OCR
                    self.logger.debug(f"Indonesian OCR for motorcycle: '{text}' ({confidence:.1f}%)")
                    return text, confidence
            except Exception as e:
                self.logger.debug(f"Indonesian OCR failed for motorcycle: {e}")
        
        # Fallback to standard OCR with multiple PSM modes
        import pytesseract
        
        # Multiple PSM configurations for motorcycles
        psm_configs = [
            ('--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "single_word"),
            ('--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "single_line"),
            ('--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', "raw_line"),
        ]
        
        best_text = ""
        best_confidence = 0.0
        
        for config, method in psm_configs:
            try:
                # Get detailed OCR data
                data = pytesseract.image_to_data(
                    plate_image,
                    lang='ind+eng',
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
                    cleaned_text = self._clean_motorcycle_text(full_text)
                    
                    if cleaned_text and avg_confidence > best_confidence:
                        best_text = cleaned_text
                        best_confidence = avg_confidence
                        self.logger.debug(f"Motorcycle OCR {method}: '{cleaned_text}' ({avg_confidence:.1f}%)")
                
            except Exception as e:
                self.logger.debug(f"OCR method {method} failed: {e}")
                continue
        
        return best_text, best_confidence
    
    def _clean_motorcycle_text(self, text: str) -> str:
        """Clean OCR text specifically for motorcycle plates"""
        if not text:
            return ""
        
        # Convert to uppercase dan remove spaces
        cleaned = text.upper().strip()
        
        # Remove unwanted characters
        unwanted = ['-', '_', '.', ',', ':', ';', '|', '/', '\\', ' ']
        for char in unwanted:
            cleaned = cleaned.replace(char, '')
        
        # Character corrections untuk motor plates
        corrections = {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2',
            '8': 'B', '6': 'G', '0': 'O'  # Sometimes O gets detected as 0
        }
        
        # Apply corrections based on position (smart replacement)
        result = ""
        for i, char in enumerate(cleaned):
            if char in corrections:
                # First char is usually letter for Indonesian plates
                if i == 0 and char.isdigit() and corrections[char].isalpha():
                    result += corrections[char]
                # Last chars are usually letters
                elif i >= len(cleaned) - 3 and char.isdigit() and corrections[char].isalpha():
                    result += corrections[char]
                # Middle chars are usually numbers
                elif 1 <= i < len(cleaned) - 3 and char.isalpha() and corrections[char].isdigit():
                    result += corrections[char]
                else:
                    result += char
            else:
                result += char
        
        # Validate length
        if len(result) < 4 or len(result) > 10:
            return ""
        
        return result
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Main method untuk detect motorcycle plates
        
        Args:
            image: Input image
            
        Returns:
            List of PlateDetection objects
        """
        if not self.is_enabled():
            return []
        
        detections = []
        
        try:
            # Multiple preprocessing methods
            processed_images = self.preprocess_for_motorcycles(image)
            self.logger.info(f"🔄 Preprocessing generated {len(processed_images)} methods")
            
            all_contours = []
            
            # Process each preprocessed image
            for method_name, processed_img in processed_images:
                contours_with_meta = self.detect_motorcycle_contours(processed_img)
                self.logger.debug(f"   Method '{method_name}': {len(contours_with_meta)} contours found")
                
                # Add method info to metadata
                for contour, meta in contours_with_meta:
                    meta['preprocess_method'] = method_name
                    all_contours.append((contour, meta))
            
            self.logger.info(f"🔍 Total contours from all methods: {len(all_contours)}")
            
            # Sort all contours by score
            all_contours.sort(key=lambda x: x[1]['score'], reverse=True)
            
            # Remove global duplicates
            unique_contours = self._remove_duplicate_contours(all_contours)
            self.logger.info(f"🔧 After duplicate removal: {len(unique_contours)} unique contours")
            
            # Limit to top candidates untuk performance - increased for better detection
            top_contours = unique_contours[:25]  # Process top 25 candidates for better stability
            
            self.logger.info(f"🎯 Processing top {len(top_contours)} motorcycle plate candidates")
            
            # Process each candidate
            for i, (contour, meta) in enumerate(top_contours):
                bbox = meta['bbox']
                x, y, w, h = bbox
                self.logger.debug(f"📋 Candidate {i+1}: bbox=({x},{y},{w},{h}) area={meta.get('area', 0):.0f} score={meta.get('score', 0):.2f}")
                
                # PRE-OCR VALIDATION: Analyze texture and detect false positives
                texture_analysis = self._analyze_plate_texture(image, bbox)
                false_positive_check = self._detect_false_positive_patterns(image, bbox)
                
                self.logger.debug(f"   Texture analysis: text_likelihood={texture_analysis['text_likelihood']:.1f}% "
                                f"edge_density={texture_analysis['edge_density']:.1f}% "
                                f"texture_variance={texture_analysis['texture_variance']:.1f}")
                
                # Early rejection for obvious false positives
                if false_positive_check['is_false_positive']:
                    self.logger.debug(f"❌ REJECTED (Pre-OCR): False positive detected - {false_positive_check['reason']}")
                    continue
                
                # Early rejection for poor texture characteristics
                if not texture_analysis['is_valid']:
                    self.logger.debug(f"❌ REJECTED (Pre-OCR): Poor text likelihood ({texture_analysis['text_likelihood']:.1f}%)")
                    continue
                
                self.logger.debug(f"✅ PASSED Pre-OCR validation: text_likelihood={texture_analysis['text_likelihood']:.1f}%")
                
                # Extract and enhance plate
                plate_image = self.extract_and_enhance_plate(image, bbox)
                
                # Perform OCR
                text, confidence = self.perform_motorcycle_ocr(plate_image)
                self.logger.debug(f"📝 OCR result: '{text}' confidence={confidence:.1f}%")
                
                # Apply Indonesian plate validation - lowered threshold for stability
                if text and confidence >= 25:  # Lowered from MIN_CONFIDENCE for better real-world detection
                    # Import validation functions from config
                    from config import validate_indonesian_plate, calculate_plate_confidence_boost, IndonesianPlateConfig
                    
                    # Strict validation if enabled
                    if IndonesianPlateConfig.ENABLE_STRICT_PATTERN_VALIDATION:
                        if not validate_indonesian_plate(text):
                            self.logger.debug(f"Rejected invalid plate pattern: '{text}' (confidence: {confidence:.1f}%)")
                            continue
                    
                    # Apply confidence boost for valid patterns
                    original_confidence = confidence
                    confidence = calculate_plate_confidence_boost(text, confidence)
                    
                    # ENHANCED SCORING: Apply texture-based confidence adjustment
                    texture_boost = min(25, texture_analysis['text_likelihood'] / 4)  # Max 25% boost
                    confidence_penalty = false_positive_check['confidence_penalty']
                    
                    # Apply texture boost and false positive penalty
                    adjusted_confidence = confidence + texture_boost - confidence_penalty
                    adjusted_confidence = max(0, min(100, adjusted_confidence))
                    
                    if confidence != original_confidence:
                        self.logger.debug(f"Confidence boosted: '{text}' {original_confidence:.1f}% -> {confidence:.1f}%")
                    
                    if adjusted_confidence != confidence:
                        self.logger.debug(f"Texture adjusted: '{text}' {confidence:.1f}% -> {adjusted_confidence:.1f}% "
                                        f"(texture_boost: +{texture_boost:.1f}%, penalty: -{confidence_penalty:.1f}%)")
                        confidence = adjusted_confidence
                
                if text and confidence >= 25:  # Consistent lowered threshold for stability
                    # Extract plate region for processed_image
                    x, y, w, h = bbox
                    plate_region = image[y:y+h, x:x+w]
                    
                    detection = PlateDetection(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        processed_image=plate_region.copy(),
                        timestamp=time.time(),
                        vehicle_type="motorcycle",
                        detection_method="motorcycle_optimized"
                    )
                    
                    detections.append(detection)
                    
                    self.logger.info(f"✅ Motorcycle plate detected: '{text}' ({confidence:.1f}%) "
                                   f"bbox={bbox} area={meta.get('area', 0):.0f}")
                else:
                    if text:
                        self.logger.debug(f"❌ Rejected: '{text}' confidence={confidence:.1f}% < threshold=25")
                    else:
                        self.logger.debug(f"❌ No text found in candidate {i+1}")
            
            # Sort by confidence
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error in motorcycle plate detection: {e}")
        
        return detections
    
    def get_detection_stats(self) -> dict:
        """Get detection statistics"""
        return {
            'enabled': self.is_enabled(),
            'min_area': self.config.MIN_CONTOUR_AREA,
            'max_area': self.config.MAX_CONTOUR_AREA,
            'min_aspect_ratio': self.config.MIN_ASPECT_RATIO,
            'max_aspect_ratio': self.config.MAX_ASPECT_RATIO,
            'min_confidence': self.config.MIN_CONFIDENCE,
            'upscale_factor': self.config.UPSCALE_FACTOR,
            'extreme_upscaling': self.config.ENABLE_EXTREME_UPSCALING
        }

def test_motorcycle_detector():
    """Test function untuk motorcycle detector"""
    print("🏍️ Testing Motorcycle Plate Detector")
    print("=" * 50)
    
    detector = MotorcyclePlateDetector()
    
    # Create test image with small motorcycle plates
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # Add small motorcycle plate
    cv2.rectangle(test_img, (200, 300), (280, 340), (255, 255, 255), -1)
    cv2.rectangle(test_img, (200, 300), (280, 340), (0, 0, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_img, 'B1234A', (205, 325), font, 0.6, (0, 0, 0), 1)
    
    # Test detection
    detections = detector.detect_plates(test_img)
    
    print(f"✅ Found {len(detections)} motorcycle plates")
    for i, detection in enumerate(detections):
        print(f"   Plate {i+1}: '{detection.text}' ({detection.confidence:.1f}%) "
              f"bbox={detection.bbox} source={detection.source}")
    
    # Show stats
    stats = detector.get_detection_stats()
    print(f"\n📊 Detector Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    

if __name__ == "__main__":
    test_motorcycle_detector()