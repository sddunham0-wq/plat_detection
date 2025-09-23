"""
Indonesian OCR Utility
Optimized OCR specifically for Indonesian license plates
"""

import cv2
import numpy as np
import pytesseract
import re
import logging
from typing import Tuple, List, Optional
from config import TesseractConfig, IndonesianPlateConfig

class IndonesianPlateOCR:
    """
    Specialized OCR for Indonesian license plates
    """
    
    def __init__(self):
        """Initialize Indonesian OCR"""
        self.logger = logging.getLogger(__name__)
        self.tesseract_config = TesseractConfig
        self.plate_config = IndonesianPlateConfig
        
        # Setup Tesseract path
        if self.tesseract_config.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_config.TESSERACT_PATH
        
        # Compile regex patterns
        self.compiled_patterns = [
            re.compile(pattern) for pattern in self.plate_config.PLATE_PATTERNS
        ]
        
        self.logger.info("Indonesian Plate OCR initialized")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image specifically for Indonesian plates
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # For clean synthetic images, minimal processing
        # Check if image is already clean (high contrast, minimal noise)
        if self._is_clean_image(gray):
            return gray
        
        # Enhanced contrast untuk plat Indonesia (hitam text, putih background)
        enhanced = cv2.convertScaleAbs(
            gray, 
            alpha=self.plate_config.CONTRAST_ENHANCEMENT,
            beta=0
        )
        
        # Gentle noise reduction only if needed
        if self._has_noise(enhanced):
            kernel = np.ones(self.plate_config.NOISE_REDUCTION_KERNEL, np.uint8)
            denoised = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        else:
            denoised = enhanced
        
        return denoised
    
    def _is_clean_image(self, gray_image: np.ndarray) -> bool:
        """Check if image is already clean and high contrast"""
        # Calculate standard deviation (measure of contrast)
        std_dev = np.std(gray_image)
        
        # Check if mostly black and white (typical clean plate)
        unique_values = len(np.unique(gray_image))
        
        return std_dev > 50 and unique_values < 20  # High contrast, few colors
    
    def _has_noise(self, image: np.ndarray) -> bool:
        """Check if image has significant noise"""
        # Simple noise detection using Laplacian variance
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance < 100  # Low variance indicates noise/blur
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR text dengan Indonesian-specific rules
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to uppercase
        cleaned = text.upper().strip()
        
        # Remove unwanted characters
        for char in self.plate_config.REMOVE_CHARS:
            cleaned = cleaned.replace(char, '')
        
        # Replace multiple spaces
        if self.plate_config.REPLACE_MULTIPLE_SPACES:
            cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Apply character corrections only if confidence is low or pattern doesn't match
        # Skip corrections for high-confidence, well-formed results
        if len(cleaned) >= 5:  # Only for reasonable length strings
            for wrong, correct in self.plate_config.CHAR_CORRECTIONS.items():
                if wrong in cleaned:
                    # Smart replacement berdasarkan posisi
                    cleaned = self._smart_character_replacement(cleaned, wrong, correct)
        
        return cleaned.strip()
    
    def _smart_character_replacement(self, text: str, wrong: str, correct: str) -> str:
        """
        Smart character replacement berdasarkan konteks Indonesian plates
        
        Args:
            text: Input text
            wrong: Character to replace
            correct: Replacement character
            
        Returns:
            Text with smart replacements
        """
        # For letter-to-number corrections (O->0, I->1)
        if wrong.isalpha() and correct.isdigit():
            # Only replace if surrounded by digits or at number position
            pattern = rf'(?<=\d){wrong}(?=\d)|(?<=\d){wrong}$|^{wrong}(?=\d)'
            text = re.sub(pattern, correct, text)
        
        # For number-to-letter corrections (8->B, 6->G)
        elif wrong.isdigit() and correct.isalpha():
            # Only replace if at letter position (beginning or after space)
            pattern = rf'^{wrong}(?=\d)|(?<=\s){wrong}(?=\d)'
            text = re.sub(pattern, correct, text)
        
        return text
    
    def validate_plate_pattern(self, text: str) -> Tuple[bool, float]:
        """
        Validate text against Indonesian plate patterns
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, confidence_boost)
        """
        if not text or len(text) < self.plate_config.MIN_PLATE_LENGTH:
            return False, 0.0
        
        if len(text) > self.plate_config.MAX_PLATE_LENGTH:
            return False, 0.0
        
        confidence_boost = 0.0
        
        # Check against patterns
        for pattern in self.compiled_patterns:
            if pattern.match(text):
                confidence_boost += self.plate_config.PATTERN_MATCH_BOOST
                
                # Additional boost for regional code
                first_chars = text[:2] if len(text) > 1 else text[:1]
                if first_chars in self.plate_config.REGIONAL_CODES:
                    confidence_boost += self.plate_config.REGIONAL_CODE_BOOST
                
                return True, confidence_boost
        
        return False, 0.0
    
    def try_multiple_psm(self, image: np.ndarray, language: str) -> Tuple[str, float]:
        """
        Try multiple PSM modes for better accuracy
        
        Args:
            image: Preprocessed image
            language: OCR language
            
        Returns:
            Tuple of (best_text, best_confidence)
        """
        best_text = ""
        best_confidence = 0.0
        
        # Get all PSM configurations
        psm_configs = []
        if hasattr(self.tesseract_config, 'USE_MULTIPLE_PSM') and self.tesseract_config.USE_MULTIPLE_PSM:
            for psm in self.tesseract_config.PSM_PRIORITY:
                config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                psm_configs.append((psm, config))
        else:
            # Fallback to main config
            psm_configs = [(7, self.tesseract_config.OCR_CONFIG)]
        
        for psm_num, config in psm_configs:
            try:
                # Get OCR result with detailed info
                data = pytesseract.image_to_data(
                    image,
                    lang=language,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text and confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                texts = [text.strip() for text in data['text'] if text.strip()]
                
                if texts and confidences:
                    full_text = ' '.join(texts)
                    avg_confidence = np.mean(confidences)
                    
                    # Clean text
                    cleaned_text = self.clean_text(full_text)
                    
                    # Validate pattern and boost confidence
                    is_valid, confidence_boost = self.validate_plate_pattern(cleaned_text)
                    final_confidence = avg_confidence + confidence_boost
                    
                    self.logger.debug(f"PSM {psm_num}: '{cleaned_text}' ({final_confidence:.1f}%)")
                    
                    if final_confidence > best_confidence and cleaned_text:
                        best_text = cleaned_text
                        best_confidence = final_confidence
                
            except Exception as e:
                self.logger.debug(f"PSM {psm_num} failed: {str(e)}")
                continue
        
        return best_text, best_confidence
    
    def recognize_plate(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Main OCR function untuk Indonesian plates
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Try Indonesian language first
        ind_text, ind_confidence = self.try_multiple_psm(
            processed_image, 
            self.tesseract_config.LANGUAGE
        )
        
        # Check if Indonesian result is acceptable
        if (ind_text and 
            ind_confidence >= self.tesseract_config.INDONESIAN_MIN_CONFIDENCE):
            
            self.logger.debug(f"Indonesian OCR success: '{ind_text}' ({ind_confidence:.1f}%)")
            return ind_text, ind_confidence
        
        # Try fallback language if auto language is enabled
        if (hasattr(self.tesseract_config, 'ENABLE_AUTO_LANGUAGE') and 
            self.tesseract_config.ENABLE_AUTO_LANGUAGE and
            ind_confidence < self.tesseract_config.LANGUAGE_SWITCH_THRESHOLD):
            
            self.logger.info(f"Trying fallback language (Indonesian confidence was {ind_confidence:.1f}%)")
            
            fallback_text, fallback_confidence = self.try_multiple_psm(
                processed_image,
                self.tesseract_config.FALLBACK_LANGUAGE
            )
            
            # Compare results and choose better one
            if fallback_confidence > ind_confidence and fallback_text:
                self.logger.info(f"Using fallback result: '{fallback_text}' ({fallback_confidence:.1f}%)")
                return fallback_text, fallback_confidence
        
        # Return Indonesian result even if low confidence
        return ind_text, ind_confidence
    
    def get_statistics(self) -> dict:
        """Get OCR statistics"""
        return {
            'language': self.tesseract_config.LANGUAGE,
            'fallback_language': self.tesseract_config.FALLBACK_LANGUAGE,
            'min_confidence': self.tesseract_config.INDONESIAN_MIN_CONFIDENCE,
            'switch_threshold': self.tesseract_config.LANGUAGE_SWITCH_THRESHOLD,
            'pattern_validation': self.plate_config.ENABLE_PATTERN_VALIDATION,
            'multiple_psm': getattr(self.tesseract_config, 'USE_MULTIPLE_PSM', False)
        }

def test_indonesian_ocr():
    """Test function untuk Indonesian OCR"""
    print("🇮🇩 Testing Indonesian Plate OCR")
    print("=" * 40)
    
    # Initialize OCR
    ocr = IndonesianPlateOCR()
    
    # Create test image
    test_img = np.ones((60, 200, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_img, 'B1234ABC', (10, 40), font, 1.2, (0, 0, 0), 2)
    
    # Test recognition
    result_text, confidence = ocr.recognize_plate(test_img)
    
    print(f"✅ Result: '{result_text}'")
    print(f"📊 Confidence: {confidence:.1f}%")
    
    # Test pattern validation
    is_valid, boost = ocr.validate_plate_pattern(result_text)
    print(f"🎯 Pattern Valid: {is_valid}")
    print(f"⚡ Confidence Boost: +{boost:.1f}%")
    
    # Show statistics
    stats = ocr.get_statistics()
    print(f"\n📋 OCR Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    test_indonesian_ocr()