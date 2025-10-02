"""
License Plate Detector untuk Live Video Stream
Deteksi dan ekstraksi plat nomor dari frame video
"""

import cv2
import numpy as np
import pytesseract
import re
import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class PlateDetection:
    """Data class untuk hasil deteksi plat"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    processed_image: np.ndarray
    timestamp: float
    vehicle_type: str = "unknown"  # "motorcycle", "car", "bus", "truck"
    detection_method: str = "general"  # "general", "motorcycle_optimized"

class LicensePlateDetector:
    """
    Main class untuk deteksi plat nomor
    Menggunakan OpenCV untuk deteksi dan Tesseract untuk OCR
    """
    
    def __init__(self, config=None, use_enhanced=False):
        """
        Initialize detector dengan konfigurasi
        
        Args:
            config: Configuration object (dari config.py)
            use_enhanced: Whether to use enhanced detection methods
        """
        # Import config jika tidak diberikan
        if config is None:
            from config import TesseractConfig, DetectionConfig, MotorcycleDetectionConfig, EnhancedDetectionConfig
            self.tesseract_config = TesseractConfig
            self.detection_config = DetectionConfig
            self.motorcycle_config = MotorcycleDetectionConfig
            self.enhanced_config = EnhancedDetectionConfig
        else:
            self.tesseract_config = config.TesseractConfig
            self.detection_config = config.DetectionConfig
            self.motorcycle_config = config.MotorcycleDetectionConfig
            self.enhanced_config = getattr(config, 'EnhancedDetectionConfig', None)
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        # Enhanced detection setup
        self.use_enhanced = use_enhanced and self.enhanced_config and self.enhanced_config.ENABLE_ENHANCED_DETECTION
        self.enhanced_detector = None
        self.super_resolution = None
        self.ocr_ensemble = None
        
        if self.use_enhanced:
            try:
                from .enhanced_plate_detector import EnhancedPlateDetector
                from .super_resolution import AdaptiveSuperResolution
                from .ocr_ensemble import OCREnsemble
                
                self.enhanced_detector = EnhancedPlateDetector(config)
                self.super_resolution = AdaptiveSuperResolution()
                self.ocr_ensemble = OCREnsemble(config)
                
                self.logger.info("Enhanced detection modules loaded successfully")
            except ImportError as e:
                self.logger.warning(f"Enhanced detection modules not available: {str(e)}")
                self.use_enhanced = False
        
        # Setup Tesseract path
        if hasattr(pytesseract, 'pytesseract'):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_config.TESSERACT_PATH
        
        # Performance tracking
        self.total_detections = 0
        self.successful_ocr = 0
        self.last_detection_time = {}  # Untuk duplicate detection
        
        # Temporal smoothing dan stability tracking
        self.detection_history = {}  # Track detections over time
        self.stable_detections = {}  # Stable/confirmed detections
        self.detection_tracker_id = 0  # ID counter untuk tracking
        
        self.logger.info("License Plate Detector initialized")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing gambar untuk deteksi plat nomor yang lebih baik
        
        Args:
            image: Input image (BGR)
            
        Returns:
            np.ndarray: Processed grayscale image
        """
        # Convert ke grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter untuk preserve edges sambil mengurangi noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply Gaussian blur untuk mengurangi noise lebih lanjut
        blurred = cv2.GaussianBlur(
            filtered, 
            self.detection_config.GAUSSIAN_BLUR_KERNEL, 
            0
        )
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply sharpening filter untuk meningkatkan edge definition
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.detection_config.ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            self.detection_config.ADAPTIVE_THRESHOLD_C
        )
        
        # Morphological operations untuk membersihkan noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def find_license_plate_candidates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Enhanced kandidat area plat nomor berdasarkan kontur dengan filtering yang lebih baik
        
        Args:
            image: Preprocessed image
            
        Returns:
            List[Tuple]: List of bounding boxes (x, y, w, h)
        """
        candidates = []
        
        # Apply edge detection untuk meningkatkan deteksi kontur
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Dilate edges untuk menghubungkan garis yang terputus
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (area < self.detection_config.MIN_CONTOUR_AREA or 
                area > self.detection_config.MAX_CONTOUR_AREA):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Enhanced aspect ratio filtering untuk plat nomor Indonesia
            if (aspect_ratio < self.detection_config.MIN_ASPECT_RATIO or 
                aspect_ratio > self.detection_config.MAX_ASPECT_RATIO):
                continue
            
            # Enhanced size filtering
            if w < 80 or h < 25 or w > 400 or h > 120:
                continue
            
            # Calculate contour properties untuk filtering yang lebih baik
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            # Rectangularity check (seberapa mirip dengan persegi panjang)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Filter by rectangularity (plat nomor harus cukup persegi panjang)
            if rectangularity < 0.7:  # Minimal 70% rectangularity
                continue
            
            # Solidity check (ratio area contour terhadap convex hull)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Filter by solidity (plat nomor harus solid, tidak berlubang)
            if solidity < 0.8:  # Minimal 80% solidity
                continue
            
            # Extent check (ratio area terhadap bounding rectangle)
            extent = area / rect_area if rect_area > 0 else 0
            if extent < 0.7:  # Minimal 70% extent
                continue
            
            # Calculate confidence score berdasarkan berbagai faktor
            confidence = self._calculate_candidate_confidence(
                area, aspect_ratio, rectangularity, solidity, extent, w, h
            )
            
            candidates.append((x, y, w, h, confidence))
        
        # Sort by confidence score (higher first)
        candidates.sort(key=lambda x: x[4], reverse=True)
        
        # Return top candidates (remove confidence from tuple)
        return [(x, y, w, h) for x, y, w, h, _ in candidates[:5]]
    
    def _calculate_candidate_confidence(self, area: float, aspect_ratio: float, 
                                      rectangularity: float, solidity: float, 
                                      extent: float, width: int, height: int) -> float:
        """
        Calculate confidence score untuk kandidat plat nomor
        
        Args:
            area: Area contour
            aspect_ratio: Width/height ratio
            rectangularity: Seberapa persegi panjang
            solidity: Seberapa solid (tidak berlubang)
            extent: Ratio area terhadap bounding rectangle
            width: Lebar bounding box
            height: Tinggi bounding box
            
        Returns:
            float: Confidence score (0-100)
        """
        confidence = 0.0
        
        # Aspect ratio score (ideal untuk plat Indonesia: 3.0-4.5)
        if 3.0 <= aspect_ratio <= 4.5:
            confidence += 30.0
        elif 2.0 <= aspect_ratio < 3.0 or 4.5 < aspect_ratio <= 5.5:
            confidence += 20.0
        elif 1.5 <= aspect_ratio < 2.0 or 5.5 < aspect_ratio <= 6.0:
            confidence += 10.0
        
        # Size score (ideal size untuk plat nomor)
        if 120 <= width <= 300 and 30 <= height <= 80:
            confidence += 25.0
        elif 100 <= width <= 350 and 25 <= height <= 100:
            confidence += 15.0
        elif 80 <= width <= 400 and 20 <= height <= 120:
            confidence += 10.0
        
        # Rectangularity score
        if rectangularity >= 0.9:
            confidence += 20.0
        elif rectangularity >= 0.8:
            confidence += 15.0
        elif rectangularity >= 0.7:
            confidence += 10.0
        
        # Solidity score
        if solidity >= 0.95:
            confidence += 15.0
        elif solidity >= 0.9:
            confidence += 12.0
        elif solidity >= 0.8:
            confidence += 8.0
        
        # Extent score
        if extent >= 0.9:
            confidence += 10.0
        elif extent >= 0.8:
            confidence += 8.0
        elif extent >= 0.7:
            confidence += 5.0
        
        return min(confidence, 100.0)  # Cap at 100
    
    def find_motorcycle_plate_candidates(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Enhanced kandidat plat motor dengan filtering yang lebih akurat
        
        Args:
            image: Preprocessed image
            
        Returns:
            List[Tuple]: List of bounding boxes (x, y, w, h) untuk motor
        """
        candidates = []
        
        # Apply more aggressive edge detection untuk plat motor kecil
        edges = cv2.Canny(image, 30, 100, apertureSize=3)
        
        # Apply closing operation untuk menghubungkan garis yang terputus
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, hierarchy = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by area (lebih toleran untuk motor kecil/jauh)
            if (area < self.motorcycle_config.MIN_CONTOUR_AREA or 
                area > self.motorcycle_config.MAX_CONTOUR_AREA):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Enhanced size filtering untuk motor
            if (w < self.motorcycle_config.MIN_PLATE_WIDTH or 
                w > self.motorcycle_config.MAX_PLATE_WIDTH or
                h < self.motorcycle_config.MIN_PLATE_HEIGHT or
                h > self.motorcycle_config.MAX_PLATE_HEIGHT):
                continue
            
            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            
            # Enhanced aspect ratio filtering (motor bisa lebih persegi)
            if (aspect_ratio < self.motorcycle_config.MIN_ASPECT_RATIO or 
                aspect_ratio > self.motorcycle_config.MAX_ASPECT_RATIO):
                continue
            
            # Calculate additional properties untuk motorcycle plates
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Rectangularity check (lebih toleran untuk motor)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Motorcycle plates bisa kurang persegi karena sudut pandang
            if rectangularity < 0.6:  # Lebih toleran dari 0.7
                continue
            
            # Solidity check
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Lebih toleran untuk solidity motor
            if solidity < 0.7:  # Lebih toleran dari 0.8
                continue
            
            # Calculate confidence khusus untuk motor
            confidence = self._calculate_motorcycle_candidate_confidence(
                area, aspect_ratio, rectangularity, solidity, w, h
            )
            
            # Filter by minimum confidence
            if confidence < 20.0:  # Minimum confidence untuk motorcycle
                continue
            
            candidates.append((x, y, w, h, confidence))
        
        # Sort by confidence score (higher first)
        candidates.sort(key=lambda x: x[4], reverse=True)
        
        # Return top candidates (remove confidence from tuple)
        return [(x, y, w, h) for x, y, w, h, _ in candidates[:8]]  # Return lebih banyak candidates untuk motor
    
    def _calculate_motorcycle_candidate_confidence(self, area: float, aspect_ratio: float, 
                                                 rectangularity: float, solidity: float, 
                                                 width: int, height: int) -> float:
        """
        Calculate confidence score khusus untuk kandidat plat motor
        
        Args:
            area: Area contour
            aspect_ratio: Width/height ratio
            rectangularity: Seberapa persegi panjang
            solidity: Seberapa solid
            width: Lebar bounding box
            height: Tinggi bounding box
            
        Returns:
            float: Confidence score (0-100)
        """
        confidence = 0.0
        
        # Aspect ratio score untuk motor (lebih toleran, 1.5-4.0)
        if 2.0 <= aspect_ratio <= 3.5:
            confidence += 30.0
        elif 1.5 <= aspect_ratio < 2.0 or 3.5 < aspect_ratio <= 4.5:
            confidence += 25.0
        elif 1.2 <= aspect_ratio < 1.5 or 4.5 < aspect_ratio <= 5.0:
            confidence += 15.0
        
        # Size score untuk motor (lebih toleran untuk plat kecil)
        if 50 <= width <= 200 and 15 <= height <= 60:
            confidence += 30.0
        elif 30 <= width <= 250 and 10 <= height <= 80:
            confidence += 20.0
        elif 20 <= width <= 300 and 8 <= height <= 100:
            confidence += 15.0
        
        # Rectangularity score (lebih toleran)
        if rectangularity >= 0.8:
            confidence += 20.0
        elif rectangularity >= 0.7:
            confidence += 15.0
        elif rectangularity >= 0.6:
            confidence += 10.0
        
        # Solidity score (lebih toleran)
        if solidity >= 0.9:
            confidence += 20.0
        elif solidity >= 0.8:
            confidence += 15.0
        elif solidity >= 0.7:
            confidence += 10.0
        
        return min(confidence, 100.0)  # Cap at 100
    
    def _check_indonesian_plate_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Check apakah area memiliki karakteristik warna plat nomor Indonesia
        (Hitam pada putih atau putih pada hitam)
        
        Args:
            image: Original BGR image
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            float: Color confidence score (0-100)
        """
        try:
            x, y, w, h = bbox
            
            # Extract region
            if len(image.shape) == 3:
                region = image[y:y+h, x:x+w]
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray_region = image[y:y+h, x:x+w]
            
            if gray_region.size == 0:
                return 0.0
            
            # Analyze intensity distribution
            mean_intensity = np.mean(gray_region)
            std_intensity = np.std(gray_region)
            
            # Check for high contrast (indikasi teks pada background)
            if std_intensity > 30:  # High contrast threshold
                color_score = 30.0
            elif std_intensity > 20:
                color_score = 20.0
            elif std_intensity > 15:
                color_score = 10.0
            else:
                color_score = 0.0
            
            # Check for white background (Indonesian plates typically white)
            if mean_intensity > 150:  # Bright background
                color_score += 20.0
            elif mean_intensity > 120:
                color_score += 10.0
            
            # Apply binary threshold dan check untuk text-like patterns
            _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count white vs black pixels ratio
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            total_pixels = white_pixels + black_pixels
            
            if total_pixels > 0:
                white_ratio = white_pixels / total_pixels
                # Indonesian plates should have mostly white background with black text
                if 0.6 <= white_ratio <= 0.9:  # Good balance
                    color_score += 25.0
                elif 0.5 <= white_ratio < 0.6 or 0.9 < white_ratio <= 0.95:
                    color_score += 15.0
            
            # Check edge density (plat nomor should have good edge definition)
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / (w * h)
            
            if edge_density > 0.1:  # Good edge density
                color_score += 15.0
            elif edge_density > 0.05:
                color_score += 10.0
            
            return min(color_score, 100.0)
            
        except Exception as e:
            self.logger.warning(f"Error in color check: {str(e)}")
            return 0.0
    
    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap ratio antara dua bounding boxes
        
        Args:
            bbox1: (x, y, w, h)
            bbox2: (x, y, w, h)
            
        Returns:
            float: Overlap ratio (0-1)
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
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _update_detection_tracking(self, detections: List[PlateDetection], current_time: float) -> List[PlateDetection]:
        """
        Update detection tracking dan apply temporal smoothing
        
        Args:
            detections: Current frame detections
            current_time: Current timestamp
            
        Returns:
            List[PlateDetection]: Filtered and stabilized detections
        """
        stable_detections = []
        
        # Match current detections dengan history
        for detection in detections:
            best_match_id = None
            best_overlap = 0.0
            
            # Cari match terbaik dalam detection history
            for track_id, history in self.detection_history.items():
                if len(history) == 0:
                    continue
                    
                last_detection = history[-1]
                overlap = self._calculate_bbox_overlap(detection.bbox, last_detection['bbox'])
                
                # Check jika text sama atau similar dan overlap cukup
                if (overlap > 0.5 and 
                    (detection.text == last_detection['text'] or 
                     self._text_similarity(detection.text, last_detection['text']) > 0.8)):
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match_id = track_id
            
            # Update existing track atau create new track
            if best_match_id is not None:
                # Update existing track
                self.detection_history[best_match_id].append({
                    'detection': detection,
                    'bbox': detection.bbox,
                    'text': detection.text,
                    'confidence': detection.confidence,
                    'timestamp': current_time
                })
                
                # Limit history length
                if len(self.detection_history[best_match_id]) > 10:
                    self.detection_history[best_match_id].pop(0)
                    
                # Check if detection is stable (appeared in multiple frames)
                track_history = self.detection_history[best_match_id]
                if len(track_history) >= 3:  # Minimal 3 detections untuk stability
                    # Calculate average confidence dan position
                    avg_confidence = sum(h['confidence'] for h in track_history[-3:]) / 3
                    
                    # Use most recent detection dengan boosted confidence
                    stable_detection = detection
                    stable_detection.confidence = min(avg_confidence + 10.0, 100.0)  # Boost stable detections
                    stable_detections.append(stable_detection)
                    
            else:
                # Create new track
                self.detection_tracker_id += 1
                self.detection_history[self.detection_tracker_id] = [{
                    'detection': detection,
                    'bbox': detection.bbox,
                    'text': detection.text,
                    'confidence': detection.confidence,
                    'timestamp': current_time
                }]
                
                # New detections dengan confidence yang cukup tinggi langsung di-accept
                if detection.confidence > 60.0:
                    stable_detections.append(detection)
        
        # Clean up old tracks
        current_track_ids = list(self.detection_history.keys())
        for track_id in current_track_ids:
            if len(self.detection_history[track_id]) > 0:
                last_time = self.detection_history[track_id][-1]['timestamp']
                if current_time - last_time > 10.0:  # Remove tracks older than 10 seconds
                    del self.detection_history[track_id]
        
        return stable_detections
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity antara dua text strings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity ratio (0-1)
        """
        if not text1 or not text2:
            return 0.0
            
        if text1 == text2:
            return 1.0
        
        # Simple character-based similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
            
        matches = 0
        for i in range(min(len(text1), len(text2))):
            if text1[i] == text2[i]:
                matches += 1
        
        return matches / max_len
    
    def extract_text_from_plate(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Ekstraksi teks dari gambar plat menggunakan Tesseract dengan multi-language
        
        Args:
            plate_image: Cropped plate image
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence)
        """
        try:
            # Resize untuk OCR yang lebih baik
            height, width = plate_image.shape[:2]
            if height < 50:
                scale_factor = 50 / height
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                plate_image = cv2.resize(plate_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Additional preprocessing untuk OCR
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(plate_image, cv2.MORPH_CLOSE, kernel)
            
            # Use optimized Indonesian OCR
            try:
                from utils.indonesian_ocr import IndonesianPlateOCR
                
                if not hasattr(self, '_indonesian_ocr'):
                    self._indonesian_ocr = IndonesianPlateOCR()
                
                cleaned_text, avg_confidence = self._indonesian_ocr.recognize_plate(processed)
                
                if cleaned_text and avg_confidence >= self.tesseract_config.INDONESIAN_MIN_CONFIDENCE:
                    self.logger.debug(f"Indonesian OCR success: {cleaned_text} ({avg_confidence:.1f}%)")
                else:
                    self.logger.debug(f"Indonesian OCR low confidence: {cleaned_text} ({avg_confidence:.1f}%)")
                
            except ImportError:
                # Fallback to original method if Indonesian OCR not available
                self.logger.warning("Indonesian OCR not available, using standard OCR")
                
                # Try with Indonesian language first
                cleaned_text, avg_confidence = self._try_ocr_with_language(
                    processed, self.tesseract_config.LANGUAGE, self.tesseract_config.INDONESIAN_MIN_CONFIDENCE)
                
                # Fallback ke English jika confidence rendah atau gagal
                if (not cleaned_text or avg_confidence < self.tesseract_config.LANGUAGE_SWITCH_THRESHOLD) and \
                   hasattr(self.tesseract_config, 'ENABLE_AUTO_LANGUAGE') and self.tesseract_config.ENABLE_AUTO_LANGUAGE:
                    
                    self.logger.info(f"Fallback to English OCR (confidence was {avg_confidence:.1f}%)")
                    fallback_text, fallback_confidence = self._try_ocr_with_language(
                        processed, self.tesseract_config.FALLBACK_LANGUAGE, self.tesseract_config.MIN_CONFIDENCE)
                    
                    # Gunakan hasil yang lebih baik
                    if fallback_confidence > avg_confidence:
                        cleaned_text, avg_confidence = fallback_text, fallback_confidence
                        self.logger.info(f"Using English result: {cleaned_text} ({fallback_confidence:.1f}%)")
            
            return cleaned_text, avg_confidence
                
        except Exception as e:
            self.logger.error(f"OCR Error: {str(e)}")
            return "", 0.0
    
    def _try_ocr_with_language(self, processed_image: np.ndarray, language: str, min_confidence: float) -> Tuple[str, float]:
        """
        Try OCR with specific language
        
        Args:
            processed_image: Preprocessed image
            language: Language code for Tesseract
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence)
        """
        try:
            # OCR dengan konfigurasi khusus
            ocr_result = pytesseract.image_to_data(
                processed_image,
                config=self.tesseract_config.OCR_CONFIG,
                lang=language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text dan confidence
            texts = []
            confidences = []
            
            for i, text in enumerate(ocr_result['text']):
                if text.strip():
                    conf = float(ocr_result['conf'][i])
                    if conf > min_confidence:
                        texts.append(text.strip())
                        confidences.append(conf)
            
            if texts:
                # Join semua text yang ditemukan
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                # Clean up text (remove non-alphanumeric)
                cleaned_text = self.clean_plate_text(full_text)
                
                return cleaned_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"OCR Error with language {language}: {str(e)}")
            return "", 0.0
    
    def extract_text_from_motorcycle_plate(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Ekstraksi teks dari plat motor dengan optimasi khusus
        
        Args:
            plate_image: Cropped motorcycle plate image
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence)
        """
        try:
            # Upscale image untuk OCR yang lebih baik pada plat kecil
            height, width = plate_image.shape[:2]
            
            # Extreme upscaling untuk plat sangat kecil (jarak jauh)
            if hasattr(self.motorcycle_config, 'ENABLE_EXTREME_UPSCALING') and self.motorcycle_config.ENABLE_EXTREME_UPSCALING:
                if height < 20 or width < 40:  # Plat sangat kecil
                    scale_factor = self.motorcycle_config.EXTREME_UPSCALE_FACTOR
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    plate_image = cv2.resize(plate_image, (new_width, new_height), 
                                           interpolation=cv2.INTER_CUBIC)
                    
                    # Additional denoising untuk extreme upscaling
                    if hasattr(self.motorcycle_config, 'ENABLE_NOISE_REDUCTION') and self.motorcycle_config.ENABLE_NOISE_REDUCTION:
                        plate_image = cv2.medianBlur(plate_image, 3)
            
            # Regular upscaling jika terlalu kecil
            elif height < self.motorcycle_config.MIN_OCR_HEIGHT:
                scale_factor = self.motorcycle_config.UPSCALE_FACTOR
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                plate_image = cv2.resize(plate_image, (new_width, new_height), 
                                       interpolation=cv2.INTER_CUBIC)
            
            # Enhanced preprocessing untuk plat motor
            # Bilateral filter untuk preserve edges
            processed = cv2.bilateralFilter(plate_image, 9, 75, 75)
            
            # Morphological operations khusus untuk plat kecil
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            # Sharpening untuk teks yang lebih jelas
            kernel_sharpen = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                     [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel_sharpen)
            
            # Try with Indonesian language first untuk motor
            motor_ocr_config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            cleaned_text, avg_confidence = self._try_motorcycle_ocr_with_language(
                processed, self.tesseract_config.LANGUAGE, motor_ocr_config, self.motorcycle_config.MIN_CONFIDENCE)
            
            # Fallback ke English jika confidence rendah atau gagal
            if (not cleaned_text or avg_confidence < self.tesseract_config.LANGUAGE_SWITCH_THRESHOLD) and \
               hasattr(self.tesseract_config, 'ENABLE_AUTO_LANGUAGE') and self.tesseract_config.ENABLE_AUTO_LANGUAGE:
                
                self.logger.info(f"Motorcycle fallback to English OCR (confidence was {avg_confidence:.1f}%)")
                fallback_text, fallback_confidence = self._try_motorcycle_ocr_with_language(
                    processed, self.tesseract_config.FALLBACK_LANGUAGE, motor_ocr_config, self.motorcycle_config.MIN_CONFIDENCE)
                
                # Gunakan hasil yang lebih baik
                if fallback_confidence > avg_confidence:
                    cleaned_text, avg_confidence = fallback_text, fallback_confidence
                    self.logger.info(f"Using English motorcycle result: {cleaned_text} ({fallback_confidence:.1f}%)")
            
            return cleaned_text, avg_confidence
                
        except Exception as e:
            self.logger.error(f"Motorcycle OCR Error: {str(e)}")
            return "", 0.0
    
    def clean_plate_text(self, text: str) -> str:
        """
        Bersihkan dan validasi teks plat nomor
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned plate text
        """
        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Filter length
        if (len(cleaned) < self.detection_config.MIN_PLATE_LENGTH or 
            len(cleaned) > self.detection_config.MAX_PLATE_LENGTH):
            return ""
        
        # Basic format validation untuk plat Indonesia
        # Format: B1234ABC atau B1234AB
        pattern = r'^[A-Z]{1,2}\d{3,4}[A-Z]{1,3}$'
        if re.match(pattern, cleaned):
            return cleaned
        
        # Jika tidak match exact pattern, return jika panjangnya reasonable
        if 5 <= len(cleaned) <= 10:
            return cleaned
        
        return ""
    
    def clean_motorcycle_plate_text(self, text: str) -> str:
        """
        Bersihkan dan validasi teks plat nomor motor
        
        Args:
            text: Raw OCR text
            
        Returns:
            str: Cleaned motorcycle plate text
        """
        # Remove non-alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Filter length (plat motor biasanya sama dengan mobil)
        if (len(cleaned) < self.detection_config.MIN_PLATE_LENGTH or 
            len(cleaned) > self.detection_config.MAX_PLATE_LENGTH):
            return ""
        
        # Pattern validation untuk plat motor Indonesia
        # Format umum: B1234XYZ, B1234ABC, dll
        patterns = [
            r'^[A-Z]{1,2}\d{3,4}[A-Z]{1,3}$',  # Standard format
            r'^[A-Z]\d{4}[A-Z]{2,3}$',         # Alternative format
            r'^[A-Z]{2}\d{3,4}[A-Z]{1,2}$'    # Regional format
        ]
        
        for pattern in patterns:
            if re.match(pattern, cleaned):
                return cleaned
        
        # Jika tidak match exact pattern, return jika panjangnya reasonable
        if 5 <= len(cleaned) <= 10:
            return cleaned
        
        return ""
    
    def _try_motorcycle_ocr_with_language(self, processed_image: np.ndarray, language: str, 
                                        ocr_config: str, min_confidence: float) -> Tuple[str, float]:
        """
        Try motorcycle OCR with specific language
        
        Args:
            processed_image: Preprocessed image
            language: Language code for Tesseract
            ocr_config: OCR configuration string
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple[str, float]: (extracted_text, confidence)
        """
        try:
            ocr_result = pytesseract.image_to_data(
                processed_image,
                config=ocr_config,
                lang=language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text dan confidence
            texts = []
            confidences = []
            
            for i, text in enumerate(ocr_result['text']):
                if text.strip():
                    conf = float(ocr_result['conf'][i])
                    if conf > min_confidence:
                        texts.append(text.strip())
                        confidences.append(conf)
            
            if texts:
                # Join semua text yang ditemukan
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                
                # Clean up text khusus untuk motor
                cleaned_text = self.clean_motorcycle_plate_text(full_text)
                
                return cleaned_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"Motorcycle OCR Error with language {language}: {str(e)}")
            return "", 0.0
    
    def is_duplicate_detection(self, plate_text: str, current_time: float) -> bool:
        """
        Cek apakah ini duplicate detection dari plat yang sama
        
        Args:
            plate_text: Plate text
            current_time: Current timestamp
            
        Returns:
            bool: True jika duplicate
        """
        if plate_text in self.last_detection_time:
            time_diff = current_time - self.last_detection_time[plate_text]
            if time_diff < self.detection_config.DUPLICATE_THRESHOLD:
                return True
        
        self.last_detection_time[plate_text] = current_time
        return False
    
    def apply_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Apply Region of Interest untuk fokus area deteksi
        
        Args:
            image: Full frame image
            
        Returns:
            Tuple[np.ndarray, Tuple]: (ROI image, ROI coordinates)
        """
        height, width = image.shape[:2]
        
        # Calculate ROI coordinates
        roi_x_percent, roi_y_percent, roi_w_percent, roi_h_percent = self.detection_config.ROI_AREA
        
        roi_x = int(width * roi_x_percent)
        roi_y = int(height * roi_y_percent)
        roi_w = int(width * roi_w_percent)
        roi_h = int(height * roi_h_percent)
        
        # Ensure coordinates are within bounds
        roi_x = max(0, min(roi_x, width))
        roi_y = max(0, min(roi_y, height))
        roi_w = min(roi_w, width - roi_x)
        roi_h = min(roi_h, height - roi_y)
        
        roi_coords = (roi_x, roi_y, roi_w, roi_h)
        roi_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        return roi_image, roi_coords
    
    def apply_motorcycle_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Apply Region of Interest khusus untuk deteksi plat motor
        
        Args:
            image: Full frame image
            
        Returns:
            Tuple[np.ndarray, Tuple]: (ROI image, ROI coordinates)
        """
        height, width = image.shape[:2]
        
        # Calculate ROI coordinates untuk motor (area lebih luas)
        roi_x_percent, roi_y_percent, roi_w_percent, roi_h_percent = self.motorcycle_config.ROI_AREA
        
        roi_x = int(width * roi_x_percent)
        roi_y = int(height * roi_y_percent)
        roi_w = int(width * roi_w_percent)
        roi_h = int(height * roi_h_percent)
        
        # Ensure coordinates are within bounds
        roi_x = max(0, min(roi_x, width))
        roi_y = max(0, min(roi_y, height))
        roi_w = min(roi_w, width - roi_x)
        roi_h = min(roi_h, height - roi_y)
        
        roi_coords = (roi_x, roi_y, roi_w, roi_h)
        roi_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        return roi_image, roi_coords
    
    def detect_plates(self, frame: np.ndarray, apply_roi: bool = True, use_enhanced: bool = None) -> List[PlateDetection]:
        """
        Main function untuk deteksi plat nomor dalam frame
        
        Args:
            frame: Input video frame (BGR)
            apply_roi: Apakah menggunakan ROI
            use_enhanced: Force use enhanced detection (None for auto)
            
        Returns:
            List[PlateDetection]: List of detected plates
        """
        # Determine whether to use enhanced detection
        if use_enhanced is None:
            use_enhanced = self.use_enhanced
        
        # Try enhanced detection first if available and enabled
        if use_enhanced and self.enhanced_detector:
            try:
                enhanced_detections = self.enhanced_detector.detect_enhanced_plates(
                    frame, 
                    apply_super_resolution=self.enhanced_config.USE_SUPER_RESOLUTION
                )
                
                # Convert enhanced detections to standard format
                standard_detections = []
                for edet in enhanced_detections:
                    detection = PlateDetection(
                        text=edet.text,
                        confidence=edet.confidence + self.enhanced_config.ENHANCED_CONFIDENCE_BOOST,
                        bbox=edet.bbox,
                        processed_image=edet.processed_image,
                        timestamp=edet.timestamp,
                        vehicle_type=edet.vehicle_type,
                        detection_method=edet.detection_method
                    )
                    standard_detections.append(detection)
                
                if standard_detections:
                    self.logger.info(f"Enhanced detection found {len(standard_detections)} plates")
                    return standard_detections
                
            except Exception as e:
                self.logger.warning(f"Enhanced detection failed, falling back to standard: {str(e)}")
        
        # Standard detection (fallback or default)
        detections = []
        current_time = time.time()
        
        try:
            original_frame = frame.copy()
            
            # Apply ROI jika diminta
            if apply_roi:
                roi_frame, roi_coords = self.apply_roi(frame)
                roi_x, roi_y, roi_w, roi_h = roi_coords
            else:
                roi_frame = frame
                roi_x, roi_y = 0, 0
            
            # Preprocessing
            processed_frame = self.preprocess_image(roi_frame)
            
            # Find plate candidates
            candidates = self.find_license_plate_candidates(processed_frame)
            
            for x, y, w, h in candidates:
                # Adjust coordinates jika menggunakan ROI
                actual_x = x + roi_x
                actual_y = y + roi_y
                
                # Enhanced validation dengan color checking
                bbox = (actual_x, actual_y, w, h)
                color_confidence = self._check_indonesian_plate_color(original_frame, bbox)
                
                # Filter berdasarkan color confidence (minimal threshold)
                if color_confidence < 15.0:  # Minimal color confidence
                    continue
                
                # Extract plate region dari original frame
                plate_region = original_frame[actual_y:actual_y+h, actual_x:actual_x+w]
                
                # OCR processing with ensemble if available
                if self.use_enhanced and self.ocr_ensemble and self.enhanced_config.USE_OCR_ENSEMBLE:
                    try:
                        plate_text, confidence, _ = self.ocr_ensemble.ensemble_ocr(plate_region)
                    except:
                        plate_text, confidence = self.extract_text_from_plate(plate_region)
                else:
                    plate_text, confidence = self.extract_text_from_plate(plate_region)
                
                # Boost confidence berdasarkan color analysis
                if plate_text:
                    confidence = min(confidence + (color_confidence * 0.3), 100.0)
                
                # Apply Indonesian plate validation for general detector too
                if plate_text:
                    from config import validate_indonesian_plate, calculate_plate_confidence_boost, IndonesianPlateConfig
                    
                    # Strict validation if enabled
                    if IndonesianPlateConfig.ENABLE_STRICT_PATTERN_VALIDATION:
                        if not validate_indonesian_plate(plate_text):
                            self.logger.debug(f"General detector rejected invalid pattern: '{plate_text}' (confidence: {confidence:.1f}%)")
                            continue
                    
                    # Apply confidence boost for valid patterns
                    original_confidence = confidence
                    confidence = calculate_plate_confidence_boost(plate_text, confidence)
                    
                    if confidence != original_confidence:
                        self.logger.debug(f"General detector confidence boosted: '{plate_text}' {original_confidence:.1f}% -> {confidence:.1f}%")
                
                if plate_text and not self.is_duplicate_detection(plate_text, current_time):
                    detection = PlateDetection(
                        text=plate_text,
                        confidence=confidence,
                        bbox=(actual_x, actual_y, w, h),
                        processed_image=plate_region.copy(),
                        timestamp=current_time,
                        vehicle_type="car",  # Default to car for general detection
                        detection_method="general"
                    )
                    
                    detections.append(detection)
                    self.total_detections += 1
                    self.successful_ocr += 1
                    
                    self.logger.info(f"Detected plate: {plate_text} (confidence: {confidence:.1f}%)")
        
        except Exception as e:
            self.logger.error(f"Error in plate detection: {str(e)}")
        
        # Apply temporal smoothing untuk stability
        stable_detections = self._update_detection_tracking(detections, current_time)
        
        return stable_detections
    
    def detect_motorcycle_plates(self, frame: np.ndarray, vehicle_regions: List[Tuple[int, int, int, int]] = None) -> List[PlateDetection]:
        """
        Deteksi plat nomor motor dengan optimasi khusus
        
        Args:
            frame: Input video frame (BGR)
            vehicle_regions: List of vehicle bounding boxes untuk focus detection
            
        Returns:
            List[PlateDetection]: List of detected motorcycle plates
        """
        detections = []
        current_time = time.time()
        
        try:
            original_frame = frame.copy()
            
            # Jika ada vehicle regions, fokus pada area tersebut
            if vehicle_regions:
                search_frames = []
                frame_offsets = []
                
                for x, y, w, h in vehicle_regions:
                    # Expand region sedikit untuk capture area plat
                    expand_factor = 0.1  # 10% expansion
                    exp_w = int(w * expand_factor)
                    exp_h = int(h * expand_factor)
                    
                    x1 = max(0, x - exp_w)
                    y1 = max(0, y - exp_h)
                    x2 = min(frame.shape[1], x + w + exp_w)
                    y2 = min(frame.shape[0], y + h + exp_h)
                    
                    vehicle_frame = frame[y1:y2, x1:x2]
                    search_frames.append(vehicle_frame)
                    frame_offsets.append((x1, y1))
            else:
                # Use motorcycle ROI
                roi_frame, roi_coords = self.apply_motorcycle_roi(frame)
                search_frames = [roi_frame]
                frame_offsets = [(roi_coords[0], roi_coords[1])]
            
            # Process each search frame
            for search_frame, (offset_x, offset_y) in zip(search_frames, frame_offsets):
                if search_frame.size == 0:
                    continue
                
                # Preprocessing khusus untuk motor
                processed_frame = self.preprocess_image(search_frame)
                
                # Find motorcycle plate candidates
                candidates = self.find_motorcycle_plate_candidates(processed_frame)
                
                for x, y, w, h in candidates:
                    # Adjust coordinates
                    actual_x = x + offset_x
                    actual_y = y + offset_y
                    
                    # Enhanced validation untuk motorcycle dengan color checking
                    bbox = (actual_x, actual_y, w, h)
                    color_confidence = self._check_indonesian_plate_color(original_frame, bbox)
                    
                    # Lebih toleran untuk motor karena bisa lebih kecil/jauh
                    if color_confidence < 10.0:  # Threshold lebih rendah untuk motor
                        continue
                    
                    # Extract plate region dari original frame
                    plate_region = original_frame[actual_y:actual_y+h, actual_x:actual_x+w]
                    
                    # OCR processing khusus motor
                    plate_text, confidence = self.extract_text_from_motorcycle_plate(plate_region)
                    
                    # Boost confidence berdasarkan color analysis untuk motor
                    if plate_text:
                        confidence = min(confidence + (color_confidence * 0.4), 100.0)  # Boost lebih besar untuk motor
                    
                    if plate_text and not self.is_duplicate_detection(plate_text, current_time):
                        detection = PlateDetection(
                            text=plate_text,
                            confidence=confidence,
                            bbox=(actual_x, actual_y, w, h),
                            processed_image=plate_region.copy(),
                            timestamp=current_time,
                            vehicle_type="motorcycle",
                            detection_method="motorcycle_optimized"
                        )
                        
                        detections.append(detection)
                        self.total_detections += 1
                        self.successful_ocr += 1
                        
                        self.logger.info(f"Detected motorcycle plate: {plate_text} (confidence: {confidence:.1f}%)")
        
        except Exception as e:
            self.logger.error(f"Error in motorcycle plate detection: {str(e)}")
        
        # Apply temporal smoothing untuk motorcycle stability 
        stable_detections = self._update_detection_tracking(detections, current_time)
        
        return stable_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[PlateDetection], 
                       show_roi: bool = True) -> np.ndarray:
        """
        Gambar hasil deteksi di frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_roi: Tampilkan ROI box
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw ROI box
        if show_roi:
            roi_frame, roi_coords = self.apply_roi(frame)
            roi_x, roi_y, roi_w, roi_h = roi_coords
            cv2.rectangle(annotated_frame, (roi_x, roi_y), 
                         (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
            cv2.putText(annotated_frame, "ROI", (roi_x, roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw detections
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Choose color based on vehicle type and detection method
            if detection.vehicle_type == "motorcycle":
                color = (0, 255, 255)  # Bright yellow for motorcycle plates
                thickness = 3  # Thicker for better visibility
                label_prefix = "🏍️"
            else:
                color = (0, 255, 0)  # Green for regular plates
                thickness = 2
                label_prefix = "🚗"
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw detection method indicator for motorcycles
            if detection.detection_method == "motorcycle_optimized":
                # Add small corner indicator
                corner_size = 8
                cv2.rectangle(annotated_frame, (x, y), (x + corner_size, y + corner_size), 
                             (0, 255, 255), -1)
            
            # Draw text with vehicle type indicator
            text = f"{label_prefix} {detection.text} ({detection.confidence:.1f}%)"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_frame, 
                         (x, y - text_height - baseline - 5),
                         (x + text_width, y - 5),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_frame
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get detection statistics
        
        Returns:
            Dict: Statistics dictionary
        """
        success_rate = (self.successful_ocr / self.total_detections * 100) if self.total_detections > 0 else 0
        
        return {
            "total_detections": self.total_detections,
            "successful_ocr": self.successful_ocr,
            "success_rate": round(success_rate, 2)
        }

def test_plate_detector():
    """
    Test function untuk plate detector
    """
    print("Testing License Plate Detector...")
    
    # Create dummy test image (bisa diganti dengan real image)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 50
    
    # Add some noise/patterns
    cv2.rectangle(test_image, (200, 200), (400, 250), (255, 255, 255), -1)
    cv2.putText(test_image, "B 1234 ABC", (210, 235), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Initialize detector
    detector = LicensePlateDetector()
    
    # Detect plates
    detections = detector.detect_plates(test_image)
    
    print(f"Found {len(detections)} plates:")
    for det in detections:
        print(f"- Text: {det.text}, Confidence: {det.confidence:.1f}%")
    
    # Show statistics
    stats = detector.get_statistics()
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    # Setup logging untuk testing
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_plate_detector()