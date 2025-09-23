"""
Enhanced License Plate Detector untuk jarak jauh dan kondisi kurang jelas
Menggunakan teknik advanced image processing dan ensemble OCR
"""

import cv2
import numpy as np
import pytesseract
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from scipy import ndimage
from skimage import restoration, filters, exposure, morphology
from skimage.metrics import structural_similarity as ssim

@dataclass
class EnhancedPlateDetection:
    """Enhanced data class untuk hasil deteksi plat"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    processed_image: np.ndarray
    enhanced_image: np.ndarray
    timestamp: float
    vehicle_type: str = "unknown"
    detection_method: str = "enhanced"
    quality_score: float = 0.0
    enhancement_applied: List[str] = None

class EnhancedPlateDetector:
    """
    Enhanced License Plate Detector dengan algoritma canggih
    untuk deteksi plat jarak jauh dan kondisi sulit
    """
    
    def __init__(self, config=None):
        """Initialize enhanced detector"""
        # Import config
        if config is None:
            from config import TesseractConfig, DetectionConfig, MotorcycleDetectionConfig
            self.tesseract_config = TesseractConfig
            self.detection_config = DetectionConfig
            self.motorcycle_config = MotorcycleDetectionConfig
        else:
            self.tesseract_config = config.TesseractConfig
            self.detection_config = config.DetectionConfig
            self.motorcycle_config = config.MotorcycleDetectionConfig
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_detections = 0
        self.successful_ocr = 0
        self.enhancement_stats = {}
        
        # Quality assessment models
        self.blur_threshold = 100.0
        self.contrast_threshold = 50.0
        
        self.logger.info("Enhanced License Plate Detector initialized")
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality untuk menentukan enhancement strategy
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Dict: Quality metrics
        """
        # Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Contrast assessment
        contrast_score = image.std()
        
        # Brightness assessment
        brightness_score = image.mean()
        
        # Noise estimation using local variance (alternative method)
        try:
            from skimage.filters.rank import variance
            noise_score = variance(image, morphology.disk(1)).mean()
        except (ImportError, AttributeError):
            # Alternative noise estimation
            kernel = np.ones((3,3), np.uint8)
            local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel/9)
            noise_score = np.mean((image.astype(np.float32) - local_mean) ** 2)
        
        # Overall quality score (0-100)
        quality_score = min(100, (blur_score / 2) + (contrast_score / 2))
        
        return {
            'blur_score': blur_score,
            'contrast_score': contrast_score,
            'brightness_score': brightness_score,
            'noise_score': noise_score,
            'overall_quality': quality_score
        }
    
    def super_resolution_esrgan(self, image: np.ndarray, scale_factor: float = 4.0) -> np.ndarray:
        """
        Super-resolution menggunakan ESRGAN-like approach
        Simplified version tanpa deep learning untuk real-time performance
        
        Args:
            image: Input low-resolution image
            scale_factor: Upscaling factor
            
        Returns:
            np.ndarray: Super-resolved image
        """
        # Bicubic upscaling as base
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Edge-preserving smoothing
        upscaled = cv2.bilateralFilter(upscaled, 9, 75, 75)
        
        # Sharpening dengan unsharp mask
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(upscaled, 1.5, gaussian, -0.5, 0)
        
        # Detail enhancement
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(unsharp_mask, -1, kernel_sharpen)
        
        return sharpened
    
    def adaptive_enhancement(self, image: np.ndarray, quality_metrics: Dict) -> np.ndarray:
        """
        Adaptive image enhancement berdasarkan quality assessment
        
        Args:
            image: Input image
            quality_metrics: Quality metrics dari assess_image_quality
            
        Returns:
            np.ndarray: Enhanced image
        """
        enhanced = image.copy()
        applied_enhancements = []
        
        # Low contrast enhancement
        if quality_metrics['contrast_score'] < self.contrast_threshold:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            applied_enhancements.append("CLAHE")
        
        # Blur mitigation
        if quality_metrics['blur_score'] < self.blur_threshold:
            # Wiener deconvolution simulation
            kernel = np.ones((3, 3)) / 9
            enhanced = restoration.wiener(enhanced, kernel, 0.1)
            enhanced = (enhanced * 255).astype(np.uint8)
            applied_enhancements.append("Deblur")
        
        # Noise reduction untuk low quality images
        if quality_metrics['noise_score'] > 20:
            enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            applied_enhancements.append("Denoise")
        
        # Brightness adjustment
        if quality_metrics['brightness_score'] < 80:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
            applied_enhancements.append("Brightness")
        elif quality_metrics['brightness_score'] > 200:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.8, beta=-20)
            applied_enhancements.append("Brightness")
        
        # Gamma correction untuk better visibility
        gamma = 1.5 if quality_metrics['brightness_score'] < 120 else 0.8
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, lookup_table)
        applied_enhancements.append("Gamma")
        
        # Store enhancement stats
        self.enhancement_stats[time.time()] = applied_enhancements
        
        return enhanced
    
    def multi_scale_detection(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Multi-scale detection untuk menangkap plat di berbagai ukuran
        
        Args:
            image: Input preprocessed image
            
        Returns:
            List[Tuple]: List of bounding boxes dari berbagai skala
        """
        all_candidates = []
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multiple scales
        
        for scale in scales:
            # Resize image
            height, width = image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if new_width < 50 or new_height < 50:
                continue
                
            scaled_image = cv2.resize(image, (new_width, new_height))
            
            # Find contours pada skala ini
            contours, _ = cv2.findContours(scaled_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Adjusted area thresholds berdasarkan scale
                min_area = self.motorcycle_config.MIN_CONTOUR_AREA * (scale ** 2)
                max_area = self.motorcycle_config.MAX_CONTOUR_AREA * (scale ** 2)
                
                if min_area <= area <= max_area:
                    # Get bounding rectangle dan scale back
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Scale coordinates back to original size
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    
                    # Validate aspect ratio
                    aspect_ratio = orig_w / orig_h if orig_h > 0 else 0
                    if (self.motorcycle_config.MIN_ASPECT_RATIO <= aspect_ratio <= 
                        self.motorcycle_config.MAX_ASPECT_RATIO):
                        
                        all_candidates.append((orig_x, orig_y, orig_w, orig_h))
        
        # Remove duplicates menggunakan Non-Maximum Suppression
        if all_candidates:
            candidates_array = np.array(all_candidates)
            indices = cv2.dnn.NMSBoxes(
                candidates_array.tolist(),
                [1.0] * len(candidates_array),  # Dummy scores
                0.5,  # Score threshold
                0.4   # NMS threshold
            )
            
            if len(indices) > 0:
                return [all_candidates[i] for i in indices.flatten()]
        
        return all_candidates
    
    def ensemble_ocr(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Ensemble OCR menggunakan multiple methods dan voting
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            Tuple[str, float]: (best_text, confidence)
        """
        ocr_results = []
        
        # Method 1: Standard OCR dengan preprocessing
        processed1 = self.standard_preprocess(plate_image)
        text1, conf1 = self._single_ocr(processed1, self.tesseract_config.OCR_CONFIG)
        if text1:
            ocr_results.append((text1, conf1, "standard"))
        
        # Method 2: Enhanced preprocessing
        processed2 = self.enhanced_preprocess(plate_image)
        text2, conf2 = self._single_ocr(processed2, self.tesseract_config.OCR_CONFIG)
        if text2:
            ocr_results.append((text2, conf2, "enhanced"))
        
        # Method 3: Different PSM mode
        psm_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text3, conf3 = self._single_ocr(processed1, psm_config)
        if text3:
            ocr_results.append((text3, conf3, "psm7"))
        
        # Method 4: Character-level detection
        text4, conf4 = self.character_level_ocr(plate_image)
        if text4:
            ocr_results.append((text4, conf4, "character"))
        
        # Voting mechanism
        if not ocr_results:
            return "", 0.0
        
        # Find most confident result
        best_result = max(ocr_results, key=lambda x: x[1])
        
        # If confidence is low, try consensus
        if best_result[1] < 60 and len(ocr_results) > 1:
            # Check for consensus among results
            texts = [result[0] for result in ocr_results]
            from collections import Counter
            text_counts = Counter(texts)
            
            if text_counts.most_common(1)[0][1] > 1:  # If there's agreement
                consensus_text = text_counts.most_common(1)[0][0]
                # Average confidence for consensus
                consensus_conf = np.mean([r[1] for r in ocr_results if r[0] == consensus_text])
                return consensus_text, consensus_conf
        
        return best_result[0], best_result[1]
    
    def standard_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Standard preprocessing pipeline"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Basic enhancement
        processed = cv2.GaussianBlur(gray, (3, 3), 0)
        processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        return processed
    
    def enhanced_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing dengan quality assessment"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Quality assessment
        quality_metrics = self.assess_image_quality(gray)
        
        # Adaptive enhancement
        enhanced = self.adaptive_enhancement(gray, quality_metrics)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def character_level_ocr(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Character-level OCR untuk better accuracy pada low quality images
        
        Args:
            plate_image: Input plate image
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence)
        """
        try:
            # Preprocess untuk character segmentation
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image.copy()
            
            # Enhanced preprocessing
            processed = self.enhanced_preprocess(gray)
            
            # Character segmentation using contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter dan sort contours by x-coordinate
            char_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by character-like dimensions
                if 5 <= w <= 50 and 10 <= h <= 80:
                    char_contours.append((x, y, w, h))
            
            # Sort by x-coordinate (left to right)
            char_contours.sort(key=lambda x: x[0])
            
            # Extract each character
            characters = []
            confidences = []
            
            for x, y, w, h in char_contours:
                char_roi = processed[y:y+h, x:x+w]
                
                # Resize character untuk better OCR
                if char_roi.size > 0:
                    char_resized = cv2.resize(char_roi, (30, 60), interpolation=cv2.INTER_CUBIC)
                    
                    # OCR single character
                    char_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    char_text = pytesseract.image_to_string(char_resized, config=char_config).strip()
                    
                    if char_text and len(char_text) == 1:
                        characters.append(char_text)
                        confidences.append(80.0)  # Default confidence untuk character level
            
            if characters:
                full_text = ''.join(characters)
                avg_confidence = np.mean(confidences)
                return full_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"Character-level OCR error: {str(e)}")
            return "", 0.0
    
    def _single_ocr(self, processed_image: np.ndarray, ocr_config: str) -> Tuple[str, float]:
        """Single OCR attempt dengan given config"""
        try:
            ocr_result = pytesseract.image_to_data(
                processed_image,
                config=ocr_config,
                lang=self.tesseract_config.LANGUAGE,
                output_type=pytesseract.Output.DICT
            )
            
            texts = []
            confidences = []
            
            for i, text in enumerate(ocr_result['text']):
                if text.strip():
                    conf = float(ocr_result['conf'][i])
                    if conf > 20:  # Very low threshold untuk ensemble
                        texts.append(text.strip())
                        confidences.append(conf)
            
            if texts:
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                # Clean text
                cleaned_text = self.clean_plate_text(full_text)
                return cleaned_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"Single OCR error: {str(e)}")
            return "", 0.0
    
    def clean_plate_text(self, text: str) -> str:
        """Enhanced text cleaning untuk Indonesian plates"""
        import re
        
        # Remove non-alphanumeric
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Length validation
        if len(cleaned) < 5 or len(cleaned) > 10:
            return ""
        
        # Indonesian plate patterns
        patterns = [
            r'^[A-Z]{1,2}\d{3,4}[A-Z]{1,3}$',  # Standard
            r'^[A-Z]\d{4}[A-Z]{2,3}$',         # Alternative
            r'^[A-Z]{2}\d{3,4}[A-Z]{1,2}$'    # Regional
        ]
        
        for pattern in patterns:
            if re.match(pattern, cleaned):
                return cleaned
        
        # Return jika reasonable length
        if 5 <= len(cleaned) <= 10:
            return cleaned
        
        return ""
    
    def detect_enhanced_plates(self, frame: np.ndarray, apply_super_resolution: bool = True) -> List[EnhancedPlateDetection]:
        """
        Main enhanced detection function
        
        Args:
            frame: Input video frame
            apply_super_resolution: Whether to apply super-resolution
            
        Returns:
            List[EnhancedPlateDetection]: Enhanced detection results
        """
        detections = []
        current_time = time.time()
        
        try:
            original_frame = frame.copy()
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame.copy()
            
            # Quality assessment
            quality_metrics = self.assess_image_quality(gray_frame)
            
            # Apply super-resolution jika quality rendah dan diminta
            if apply_super_resolution and quality_metrics['overall_quality'] < 50:
                enhanced_frame = self.super_resolution_esrgan(gray_frame, scale_factor=3.0)
                scale_factor = 3.0
            else:
                enhanced_frame = gray_frame
                scale_factor = 1.0
            
            # Adaptive enhancement
            processed_frame = self.adaptive_enhancement(enhanced_frame, quality_metrics)
            
            # Multi-scale detection
            candidates = self.multi_scale_detection(processed_frame)
            
            for x, y, w, h in candidates:
                # Adjust coordinates jika ada super-resolution
                if scale_factor != 1.0:
                    actual_x = int(x / scale_factor)
                    actual_y = int(y / scale_factor)
                    actual_w = int(w / scale_factor)
                    actual_h = int(h / scale_factor)
                else:
                    actual_x, actual_y, actual_w, actual_h = x, y, w, h
                
                # Ensure bounds
                actual_x = max(0, actual_x)
                actual_y = max(0, actual_y)
                actual_w = min(actual_w, original_frame.shape[1] - actual_x)
                actual_h = min(actual_h, original_frame.shape[0] - actual_y)
                
                if actual_w <= 0 or actual_h <= 0:
                    continue
                
                # Extract plate region dari original frame
                plate_region = original_frame[actual_y:actual_y+actual_h, actual_x:actual_x+actual_w]
                
                if plate_region.size == 0:
                    continue
                
                # Enhanced plate region dari processed frame
                if scale_factor != 1.0:
                    enhanced_plate = processed_frame[y:y+h, x:x+w]
                else:
                    enhanced_plate = processed_frame[actual_y:actual_y+actual_h, actual_x:actual_x+actual_w]
                
                # Ensemble OCR
                plate_text, confidence = self.ensemble_ocr(plate_region)
                
                if plate_text and confidence > 20:  # Lower threshold untuk enhanced detector
                    detection = EnhancedPlateDetection(
                        text=plate_text,
                        confidence=confidence,
                        bbox=(actual_x, actual_y, actual_w, actual_h),
                        processed_image=plate_region.copy(),
                        enhanced_image=enhanced_plate.copy() if enhanced_plate.size > 0 else plate_region.copy(),
                        timestamp=current_time,
                        vehicle_type="unknown",
                        detection_method="enhanced",
                        quality_score=quality_metrics['overall_quality'],
                        enhancement_applied=self.enhancement_stats.get(current_time, [])
                    )
                    
                    detections.append(detection)
                    self.total_detections += 1
                    self.successful_ocr += 1
                    
                    self.logger.info(
                        f"Enhanced detection: {plate_text} "
                        f"(confidence: {confidence:.1f}%, quality: {quality_metrics['overall_quality']:.1f})"
                    )
        
        except Exception as e:
            self.logger.error(f"Error in enhanced plate detection: {str(e)}")
        
        return detections
    
    def get_statistics(self) -> Dict[str, any]:
        """Get enhanced detection statistics"""
        success_rate = (self.successful_ocr / self.total_detections * 100) if self.total_detections > 0 else 0
        
        return {
            "total_detections": self.total_detections,
            "successful_ocr": self.successful_ocr,
            "success_rate": round(success_rate, 2),
            "enhancement_stats": self.enhancement_stats
        }

# Test function
def test_enhanced_detector():
    """Test enhanced detector"""
    print("Testing Enhanced License Plate Detector...")
    
    # Create test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.rectangle(test_image, (200, 200), (400, 250), (255, 255, 255), -1)
    cv2.putText(test_image, "B 1234 ABC", (210, 235), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add noise untuk simulate poor conditions
    noise = np.random.normal(0, 25, test_image.shape).astype(np.uint8)
    noisy_image = cv2.add(test_image, noise)
    
    detector = EnhancedPlateDetector()
    detections = detector.detect_enhanced_plates(noisy_image)
    
    print(f"Found {len(detections)} enhanced plates:")
    for det in detections:
        print(f"- Text: {det.text}, Confidence: {det.confidence:.1f}%, Quality: {det.quality_score:.1f}")
    
    stats = detector.get_statistics()
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_enhanced_detector()