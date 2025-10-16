"""
OCR Ensemble Module untuk koordinasi multiple OCR engines
Meningkatkan accuracy dengan voting dan consensus mechanisms
"""

import cv2
import numpy as np
import pytesseract
import logging
import re
import time
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import difflib

@dataclass
class OCRResult:
    """Data class untuk single OCR result"""
    text: str
    confidence: float
    method: str
    processing_time: float
    raw_data: dict = None

class OCREnsemble:
    """
    OCR Ensemble system menggunakan multiple methods dan voting
    """
    
    def __init__(self, config=None):
        """Initialize OCR Ensemble"""
        # Import config
        if config is None:
            from config import TesseractConfig
            self.tesseract_config = TesseractConfig
        else:
            self.tesseract_config = config.TesseractConfig
        
        self.logger = logging.getLogger(__name__)
        
        # OCR Methods configuration - OPTIMIZED for CCTV conditions
        self.ocr_methods = {
            'cctv_block': {
                'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'language': 'eng',
                'weight': 1.5  # Higher weight for CCTV conditions
            },
            'single_line': {
                'config': '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'language': 'ind+eng',
                'weight': 1.3  # Increased weight for distant plates
            },
            'single_word': {
                'config': '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'language': 'eng',
                'weight': 1.1  # Slightly increased
            },
            'raw_line': {
                'config': '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'language': 'ind+eng',
                'weight': 1.0  # Increased from 0.8
            },
            'tesseract_eng': {
                'config': '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'language': 'eng',
                'weight': 1.2
            },
            'tesseract_indonesia': {
                'config': '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'language': 'ind',
                'weight': 1.1
            }
        }
        
        # Indonesian license plate patterns
        self.plate_patterns = [
            r'^[A-Z]{1,2}\d{3,4}[A-Z]{1,3}$',  # Standard: B1234ABC
            r'^[A-Z]\d{4}[A-Z]{2,3}$',         # Alternative: B1234AB
            r'^[A-Z]{2}\d{3,4}[A-Z]{1,2}$',   # Regional: AB1234C
            r'^[A-Z]\d{1,3}[A-Z]{3}$',        # Special: B123ABC
        ]
        
        # Character substitution rules untuk common OCR errors
        self.char_substitutions = {
            '0': ['O', 'Q', 'D'],
            'O': ['0', 'Q', 'D'],
            '1': ['I', 'L', '|'],
            'I': ['1', 'L', '|'],
            'S': ['5', '8'],
            '5': ['S', '8'],
            '8': ['B', '3'],
            'B': ['8', '3'],
            '6': ['G', 'C'],
            'G': ['6', 'C'],
            'Z': ['2', '7'],
            '2': ['Z', '7']
        }
        
        # Performance tracking
        self.total_ocr_calls = 0
        self.successful_consensus = 0
        self.method_performance = defaultdict(list)
        
        self.logger.info("OCR Ensemble initialized")
    
    def preprocess_for_method(self, image: np.ndarray, method: str) -> np.ndarray:
        """
        Preprocess image khusus untuk OCR method tertentu
        
        Args:
            image: Input image
            method: OCR method name
            
        Returns:
            np.ndarray: Preprocessed image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'standard':
            # Standard preprocessing
            processed = cv2.GaussianBlur(gray, (3, 3), 0)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        elif method == 'single_line':
            # Enhanced untuk single line
            processed = cv2.bilateralFilter(gray, 9, 75, 75)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 15, 5)
            # Morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        elif method == 'single_word':
            # Optimized untuk single word
            processed = cv2.medianBlur(gray, 3)
            processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Dilation untuk connect characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            processed = cv2.dilate(processed, kernel, iterations=1)
        
        elif method == 'raw_line':
            # Minimal preprocessing untuk raw detection
            processed = cv2.GaussianBlur(gray, (1, 1), 0)
        
        elif method == 'character_level':
            # Enhanced untuk character segmentation
            processed = cv2.bilateralFilter(gray, 9, 75, 75)
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            # Sharpening
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
        
        else:
            processed = gray
        
        return processed
    
    def single_ocr_attempt(self, image: np.ndarray, method: str) -> OCRResult:
        """
        Single OCR attempt dengan specific method
        
        Args:
            image: Preprocessed image
            method: OCR method name
            
        Returns:
            OCRResult: OCR result
        """
        start_time = time.time()
        
        try:
            method_config = self.ocr_methods[method]
            
            # Preprocess image untuk method ini
            processed_image = self.preprocess_for_method(image, method)
            
            # Resize jika terlalu kecil
            height, width = processed_image.shape
            if height < 30 or width < 60:
                scale_factor = max(30 / height, 60 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                processed_image = cv2.resize(processed_image, (new_width, new_height), 
                                           interpolation=cv2.INTER_CUBIC)
            
            # OCR dengan data output untuk detailed analysis
            ocr_data = pytesseract.image_to_data(
                processed_image,
                config=method_config['config'],
                lang=method_config['language'],
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text dan confidence
            texts = []
            confidences = []

            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    conf = float(ocr_data['conf'][i])
                    if conf > 10:  # Very low threshold untuk ensemble
                        texts.append(text.strip())
                        confidences.append(conf)

            # Fallback: jika image_to_data tidak menghasilkan text, coba image_to_string
            if not texts:
                try:
                    fallback_text = pytesseract.image_to_string(
                        processed_image,
                        config=method_config['config'],
                        lang=method_config['language']
                    ).strip()

                    if fallback_text:
                        # Clean text
                        clean_text = ''.join(c for c in fallback_text if c.isalnum())
                        if clean_text:
                            texts = [clean_text]
                            confidences = [70.0]  # Default confidence untuk fallback

                except Exception as fallback_error:
                    self.logger.debug(f'Fallback OCR juga gagal: {fallback_error}')
            
            if texts:
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                # Clean text
                cleaned_text = self.clean_plate_text(full_text)
                
                processing_time = time.time() - start_time
                
                # Apply method weight
                weighted_confidence = avg_confidence * method_config['weight']
                
                result = OCRResult(
                    text=cleaned_text,
                    confidence=weighted_confidence,
                    method=method,
                    processing_time=processing_time,
                    raw_data=ocr_data
                )
                
                # Track performance
                self.method_performance[method].append(avg_confidence)
                
                return result
            else:
                processing_time = time.time() - start_time
                return OCRResult("", 0.0, method, processing_time)
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"OCR method {method} failed: {str(e)}")
            return OCRResult("", 0.0, method, processing_time)
    
    def character_level_ocr(self, image: np.ndarray) -> OCRResult:
        """
        Character-level OCR dengan segmentation
        
        Args:
            image: Input plate image
            
        Returns:
            OCRResult: Character-level OCR result
        """
        start_time = time.time()
        
        try:
            # Preprocess
            processed = self.preprocess_for_method(image, 'character_level')
            
            # Character segmentation
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter dan sort contours
            char_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by character-like dimensions
                if 5 <= w <= 50 and 10 <= h <= 80:
                    char_boxes.append((x, y, w, h))
            
            # Sort by x-coordinate
            char_boxes.sort(key=lambda x: x[0])
            
            # OCR each character
            characters = []
            char_confidences = []
            
            for x, y, w, h in char_boxes:
                char_roi = processed[y:y+h, x:x+w]
                
                if char_roi.size > 0:
                    # Resize untuk better OCR
                    char_resized = cv2.resize(char_roi, (30, 60), interpolation=cv2.INTER_CUBIC)
                    
                    # OCR single character
                    char_config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    try:
                        char_text = pytesseract.image_to_string(
                            char_resized, 
                            config=char_config,
                            lang='eng'
                        ).strip()
                        
                        if char_text and len(char_text) == 1 and char_text.isalnum():
                            characters.append(char_text.upper())
                            char_confidences.append(75.0)  # Default confidence
                    except:
                        continue
            
            processing_time = time.time() - start_time
            
            if characters:
                full_text = ''.join(characters)
                avg_confidence = np.mean(char_confidences) if char_confidences else 0.0
                cleaned_text = self.clean_plate_text(full_text)
                
                return OCRResult(
                    text=cleaned_text,
                    confidence=avg_confidence,
                    method="character_level",
                    processing_time=processing_time
                )
            else:
                return OCRResult("", 0.0, "character_level", processing_time)
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Character-level OCR failed: {str(e)}")
            return OCRResult("", 0.0, "character_level", processing_time)
    
    def clean_plate_text(self, text: str) -> str:
        """
        Enhanced text cleaning untuk Indonesian plates
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned plate text
        """
        # Remove non-alphanumeric
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Length validation
        if len(cleaned) < 5 or len(cleaned) > 10:
            return ""
        
        # Pattern validation
        for pattern in self.plate_patterns:
            if re.match(pattern, cleaned):
                return cleaned
        
        # Return jika reasonable length
        if 5 <= len(cleaned) <= 10:
            return cleaned

        return ""

    def smart_indonesian_validation(self, text: str, confidence: float) -> Tuple[bool, float]:
        """
        Smart validation untuk Indonesian plates dengan selective rules

        Args:
            text: Cleaned plate text
            confidence: Original confidence score

        Returns:
            Tuple[bool, float]: (is_valid, adjusted_confidence)
        """
        from config import IndonesianPlateConfig

        # Skip validation untuk low confidence atau short text
        if (confidence < IndonesianPlateConfig.SMART_VALIDATION_MIN_CONFIDENCE or
            len(text) < IndonesianPlateConfig.SMART_VALIDATION_MIN_LENGTH):
            return True, confidence  # Allow through without validation

        adjusted_confidence = confidence

        # Check Indonesian plate patterns
        pattern_match = False
        for pattern in IndonesianPlateConfig.PLATE_PATTERNS:
            if re.match(pattern, text):
                pattern_match = True
                adjusted_confidence += IndonesianPlateConfig.PATTERN_MATCH_BOOST
                break

        # Check regional codes
        regional_match = False
        if IndonesianPlateConfig.ENABLE_REGIONAL_CODE_BOOST:
            for code in IndonesianPlateConfig.REGIONAL_CODES:
                if text.startswith(code):
                    regional_match = True
                    adjusted_confidence += IndonesianPlateConfig.REGIONAL_CODE_BOOST
                    break

        # Smart validation logic
        if IndonesianPlateConfig.ENABLE_SMART_VALIDATION:
            # High confidence plates pass regardless of pattern
            if confidence >= 70.0:
                return True, adjusted_confidence

            # Medium confidence plates need pattern OR regional code
            if confidence >= 40.0 and (pattern_match or regional_match):
                return True, adjusted_confidence

            # Low-medium confidence plates need both pattern AND regional code
            if confidence >= 25.0 and pattern_match and regional_match:
                return True, adjusted_confidence

            # Very low confidence plates are rejected
            if confidence < 25.0 and not (pattern_match and regional_match):
                return False, adjusted_confidence

        return True, adjusted_confidence
    
    def apply_character_corrections(self, text: str) -> List[str]:
        """
        Generate possible corrections untuk common OCR errors
        
        Args:
            text: Input text dengan possible errors
            
        Returns:
            List[str]: List of possible corrections
        """
        corrections = [text]  # Original text
        
        # Generate substitutions
        for i, char in enumerate(text):
            if char in self.char_substitutions:
                for substitute in self.char_substitutions[char]:
                    corrected = text[:i] + substitute + text[i+1:]
                    if corrected not in corrections:
                        corrections.append(corrected)
        
        # Filter corrections by pattern matching
        valid_corrections = []
        for correction in corrections:
            for pattern in self.plate_patterns:
                if re.match(pattern, correction):
                    valid_corrections.append(correction)
                    break
        
        # Return valid corrections atau original list jika tidak ada yang valid
        return valid_corrections if valid_corrections else corrections
    
    def consensus_voting(self, ocr_results: List[OCRResult]) -> Tuple[str, float]:
        """
        Consensus voting dari multiple OCR results
        
        Args:
            ocr_results: List of OCR results
            
        Returns:
            Tuple[str, float]: (consensus_text, consensus_confidence)
        """
        if not ocr_results:
            return "", 0.0
        
        # Filter out empty results
        valid_results = [r for r in ocr_results if r.text]
        
        if not valid_results:
            return "", 0.0
        
        # Method 1: Direct consensus
        text_votes = {}
        for result in valid_results:
            text = result.text
            if text in text_votes:
                text_votes[text].append(result)
            else:
                text_votes[text] = [result]
        
        # Find consensus
        if len(text_votes) == 1:
            # Perfect consensus
            consensus_text = list(text_votes.keys())[0]
            consensus_confidence = np.mean([r.confidence for r in valid_results])
            self.successful_consensus += 1
            return consensus_text, consensus_confidence
        
        # Method 2: Weighted voting
        weighted_scores = {}
        for text, results in text_votes.items():
            weighted_score = sum(r.confidence * self.ocr_methods[r.method]['weight'] for r in results)
            weighted_scores[text] = weighted_score
        
        best_text = max(weighted_scores, key=weighted_scores.get)
        best_confidence = np.mean([r.confidence for r in text_votes[best_text]])
        
        # Method 3: Similarity-based consensus jika no clear winner
        if len(text_votes) > 2:
            # Find most similar texts
            texts = list(text_votes.keys())
            similarity_groups = []
            
            for i, text1 in enumerate(texts):
                for j, text2 in enumerate(texts[i+1:], i+1):
                    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
                    if similarity > 0.7:  # 70% similarity threshold
                        # Merge groups
                        found_group = False
                        for group in similarity_groups:
                            if text1 in group or text2 in group:
                                group.update([text1, text2])
                                found_group = True
                                break
                        if not found_group:
                            similarity_groups.append({text1, text2})
            
            # Find largest similarity group
            if similarity_groups:
                largest_group = max(similarity_groups, key=len)
                # Choose best text from group based on confidence
                group_scores = {}
                for text in largest_group:
                    if text in weighted_scores:
                        group_scores[text] = weighted_scores[text]
                
                if group_scores:
                    best_text = max(group_scores, key=group_scores.get)
                    best_confidence = np.mean([r.confidence for r in text_votes[best_text]])
        
        # Method 4: Character correction attempt
        corrected_candidates = []
        for text in text_votes.keys():
            corrections = self.apply_character_corrections(text)
            for correction in corrections:
                if any(re.match(pattern, correction) for pattern in self.plate_patterns):
                    corrected_candidates.append((correction, text_votes[text]))
        
        if corrected_candidates:
            # Choose best corrected candidate
            best_corrected = max(corrected_candidates, 
                               key=lambda x: np.mean([r.confidence for r in x[1]]))
            best_text = best_corrected[0]
            best_confidence = np.mean([r.confidence for r in best_corrected[1]])
        
        return best_text, best_confidence
    
    def _create_exposure_variants(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Create multiple exposure/brightness variants for challenging lighting conditions

        Args:
            image: Input grayscale image

        Returns:
            List of (variant_image, variant_name) tuples
        """
        variants = []

        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Original (baseline)
            variants.append((gray.copy(), "original"))

            # Variant 1: Brightened (for dark plates)
            brightened = cv2.convertScaleAbs(gray, alpha=1.3, beta=30)
            variants.append((brightened, "bright"))

            # Variant 2: Darkened (for overexposed plates)
            darkened = cv2.convertScaleAbs(gray, alpha=0.8, beta=-20)
            variants.append((darkened, "dark"))

            # Variant 3: High contrast (gamma correction < 1.0)
            gamma_low = np.power(gray / 255.0, 0.7) * 255.0
            gamma_low = gamma_low.astype(np.uint8)
            variants.append((gamma_low, "gamma_low"))

            # Variant 4: Low contrast (gamma correction > 1.0)
            gamma_high = np.power(gray / 255.0, 1.3) * 255.0
            gamma_high = gamma_high.astype(np.uint8)
            variants.append((gamma_high, "gamma_high"))

            # Variant 5: Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            equalized = clahe.apply(gray)
            variants.append((equalized, "equalized"))

        except Exception as e:
            self.logger.warning(f"Exposure variant creation failed: {e}, using original only")
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            variants = [(gray, "original")]

        return variants

    def ensemble_ocr(self, image: np.ndarray, methods: List[str] = None, use_exposure_bracketing: bool = True) -> Tuple[str, float, Dict]:
        """
        Main ensemble OCR function with exposure bracketing

        Args:
            image: Input plate image
            methods: List of methods to use (None untuk semua)
            use_exposure_bracketing: Try multiple exposure variants (slower but more accurate)

        Returns:
            Tuple[str, float, Dict]: (text, confidence, details)
        """
        start_time = time.time()
        self.total_ocr_calls += 1

        # Default methods - FIXED untuk match available methods
        if methods is None:
            methods = ['cctv_block', 'single_line', 'single_word', 'character_level']

        # Collect OCR results
        ocr_results = []

        # === EXPOSURE BRACKETING: Try multiple brightness/gamma variants ===
        if use_exposure_bracketing:
            variants = self._create_exposure_variants(image)
            self.logger.debug(f"Testing {len(variants)} exposure variants")
        else:
            # Just use original image
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            variants = [(gray, "original")]

        # Try OCR on each variant
        best_variant_results = []
        for variant_img, variant_name in variants:
            variant_results = []

            # Standard OCR methods on this variant
            for method in methods:
                if method in self.ocr_methods:
                    result = self.single_ocr_attempt(variant_img, method)
                    if result.text:  # Only add non-empty results
                        # Tag with variant name
                        result.method = f"{method}_{variant_name}"
                        variant_results.append(result)

            # Character-level OCR on this variant
            if 'character_level' in methods:
                char_result = self.character_level_ocr(variant_img)
                if char_result.text:
                    char_result.method = f"character_level_{variant_name}"
                    variant_results.append(char_result)

            # Keep best result from this variant
            if variant_results:
                best_variant_result = max(variant_results, key=lambda r: r.confidence)
                best_variant_results.append(best_variant_result)
                self.logger.debug(f"Variant '{variant_name}': '{best_variant_result.text}' ({best_variant_result.confidence:.1f}%)")

        # Use all best variant results for consensus
        ocr_results = best_variant_results
        
        # Consensus voting
        consensus_text, consensus_confidence = self.consensus_voting(ocr_results)

        # Apply smart Indonesian validation
        if consensus_text:
            is_valid, adjusted_confidence = self.smart_indonesian_validation(consensus_text, consensus_confidence)
            if not is_valid:
                # Reject invalid plates
                consensus_text = ""
                consensus_confidence = 0.0
            else:
                # Use adjusted confidence
                consensus_confidence = adjusted_confidence

        total_time = time.time() - start_time
        
        # Prepare details
        details = {
            'total_methods': len(methods),
            'successful_methods': len(ocr_results),
            'processing_time': total_time,
            'individual_results': [
                {
                    'method': r.method,
                    'text': r.text,
                    'confidence': r.confidence,
                    'time': r.processing_time
                } for r in ocr_results
            ]
        }
        
        self.logger.info(f"Ensemble OCR: '{consensus_text}' "
                        f"({consensus_confidence:.1f}%) "
                        f"from {len(ocr_results)} methods")
        
        return consensus_text, consensus_confidence, details
    
    def get_performance_stats(self) -> Dict:
        """Get ensemble performance statistics"""
        total_calls = self.total_ocr_calls
        consensus_rate = (self.successful_consensus / total_calls * 100) if total_calls > 0 else 0
        
        method_stats = {}
        for method, performances in self.method_performance.items():
            if performances:
                method_stats[method] = {
                    'calls': len(performances),
                    'avg_confidence': round(np.mean(performances), 2),
                    'success_rate': round(len([p for p in performances if p > 50]) / len(performances) * 100, 2)
                }
        
        return {
            'total_calls': total_calls,
            'consensus_rate': round(consensus_rate, 2),
            'method_stats': method_stats
        }

# Test function
def test_ocr_ensemble():
    """Test OCR ensemble"""
    print("Testing OCR Ensemble...")
    
    # Create test image
    test_image = np.ones((40, 120, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "B1234ABC", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add noise untuk simulate poor conditions
    noise = np.random.normal(0, 20, test_image.shape).astype(np.uint8)
    noisy_image = cv2.add(test_image, noise)
    
    # Test ensemble
    ensemble = OCREnsemble()
    text, confidence, details = ensemble.ensemble_ocr(noisy_image)
    
    print(f"Ensemble Result: '{text}' (confidence: {confidence:.1f}%)")
    print(f"Processing details: {details}")
    
    # Performance stats
    stats = ensemble.get_performance_stats()
    print(f"Performance stats: {stats}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    test_ocr_ensemble()