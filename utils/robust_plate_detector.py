#!/usr/bin/env python3
"""
Robust Plate Detector
Deteksi plat nomor yang robust untuk berbagai kondisi:
- Plat lurus dan miring
- Berbagai kondisi cahaya
- Ukuran plat bervariasi
- Anti-false positive yang tidak terlalu ketat
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
    vehicle_type: str = "unknown"
    detection_method: str = "robust"

class RobustPlateDetector:
    """
    Robust plate detector yang bisa handle berbagai kondisi
    """
    
    def __init__(self, streaming_mode=False):
        self.logger = logging.getLogger(__name__)
        self.streaming_mode = streaming_mode  # Mode untuk real-time streaming
        
        if streaming_mode:
            # STREAMING MODE - Balanced optimization untuk closer camera
            self.min_area = 400  # More permissive untuk catch smaller plates
            self.max_area = 20000  # Expanded range
            self.min_aspect_ratio = 1.8  # Slightly more permissive
            self.max_aspect_ratio = 5.0  # Slightly more permissive
            self.min_width = 40  # More permissive
            self.max_width = 350
            self.min_height = 15  # More permissive
            self.max_height = 120
            self.min_confidence = 10  # Very low untuk more detections
            self.min_text_likelihood = 20  # Very low untuk more detections
            self.max_candidates = 15  # More candidates untuk better coverage
        else:
            # FULL MODE - Parameter untuk accuracy
            self.min_area = 200
            self.max_area = 25000
            self.min_aspect_ratio = 1.5
            self.max_aspect_ratio = 6.0
            self.min_width = 30
            self.max_width = 400
            self.min_height = 12
            self.max_height = 150
            self.min_confidence = 20
            self.min_text_likelihood = 30
            self.max_candidates = 30
        
        # Statistics tracking
        self.total_detections = 0
        self.successful_ocr = 0
        self.failed_ocr = 0
        self.false_positives = 0
        self.min_edge_density = 1.0  # More permissive untuk closer camera
        self.min_texture_variance = 3  # More permissive untuk closer camera
        
        self.logger.info("🔧 Robust Plate Detector initialized with permissive parameters")
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Main detection method yang robust untuk berbagai kondisi
        """
        if image is None or image.size == 0:
            return []
        
        detections = []
        
        try:
            # Multi-stage detection - optimize for mode
            candidates = []
            
            if self.streaming_mode:
                # STREAMING MODE: Hanya metode tercepat
                horizontal_candidates = self._detect_horizontal_plates(image)
                candidates.extend(horizontal_candidates)
                
                # Hanya edge detection jika horizontal kurang dari 5 candidates
                if len(horizontal_candidates) < 5:
                    edge_candidates = self._detect_edge_based_plates(image)
                    candidates.extend(edge_candidates)
            else:
                # FULL MODE: Semua metode untuk accuracy maksimal
                # Stage 1: Standard horizontal detection
                horizontal_candidates = self._detect_horizontal_plates(image)
                candidates.extend(horizontal_candidates)
                
                # Stage 2: Rotated plate detection
                rotated_candidates = self._detect_rotated_plates(image)
                candidates.extend(rotated_candidates)
                
                # Stage 3: Enhanced edge-based detection
                edge_candidates = self._detect_edge_based_plates(image)
                candidates.extend(edge_candidates)
            
            self.logger.info(f"🔍 Found {len(candidates)} total candidates from all methods")
            
            # Remove duplicates
            unique_candidates = self._remove_duplicate_candidates(candidates)
            self.logger.info(f"🔧 After duplicate removal: {len(unique_candidates)} unique candidates")
            
            # Process each candidate - limit based on mode
            max_process = self.max_candidates
            for i, candidate in enumerate(unique_candidates[:max_process]):
                detection = self._process_candidate(image, candidate, i+1)
                if detection:
                    detections.append(detection)
                    # Update statistics
                    self.total_detections += 1
                    if detection.text and len(detection.text) >= 3:
                        self.successful_ocr += 1
                    else:
                        self.failed_ocr += 1
                
                # Early exit in streaming mode jika sudah ada hasil bagus
                if self.streaming_mode and len(detections) >= 3:
                    break
            
            # Filter dan stabilkan detections
            detections = self._stabilize_detections(detections)
            
            # Sort by confidence
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            self.logger.info(f"✅ Final detections: {len(detections)}")
            
        except Exception as e:
            self.logger.error(f"Error in robust plate detection: {e}")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[PlateDetection], 
                       show_roi: bool = True) -> np.ndarray:
        """
        Gambar hasil deteksi di frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_roi: Tampilkan ROI box
            
        Returns:
            np.ndarray: Frame dengan deteksi tergambar
        """
        result = frame.copy()
        
        # Sort detections by confidence untuk prioritas visual
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        for i, detection in enumerate(sorted_detections):
            x, y, w, h = detection.bbox
            
            # BOUNDING BOX PLAT - SANGAT MENONJOL
            if i == 0:  # Detection terbaik - warna merah terang
                plate_color = (0, 0, 255)  # MERAH TERANG untuk plat terbaik
                thickness = 4  # Sangat tebal
            else:  # Detection lainnya - warna orange
                plate_color = (0, 165, 255)  # ORANGE untuk plat lainnya  
                thickness = 3
            
            # Double border untuk efek menonjol
            # Border luar (hitam)
            cv2.rectangle(result, (x-2, y-2), (x + w + 2, y + h + 2), (0, 0, 0), thickness+1)
            # Border dalam (warna plat)
            cv2.rectangle(result, (x, y), (x + w, y + h), plate_color, thickness)
            
            # Corner markers untuk lebih menonjol
            corner_size = 10
            corner_thickness = 2
            # Top-left corner
            cv2.line(result, (x, y), (x + corner_size, y), plate_color, corner_thickness)
            cv2.line(result, (x, y), (x, y + corner_size), plate_color, corner_thickness)
            # Top-right corner  
            cv2.line(result, (x + w, y), (x + w - corner_size, y), plate_color, corner_thickness)
            cv2.line(result, (x + w, y), (x + w, y + corner_size), plate_color, corner_thickness)
            # Bottom-left corner
            cv2.line(result, (x, y + h), (x + corner_size, y + h), plate_color, corner_thickness)
            cv2.line(result, (x, y + h), (x, y + h - corner_size), plate_color, corner_thickness)
            # Bottom-right corner
            cv2.line(result, (x + w, y + h), (x + w - corner_size, y + h), plate_color, corner_thickness)
            cv2.line(result, (x + w, y + h), (x + w, y + h - corner_size), plate_color, corner_thickness)
            
            # LABEL PLAT - SANGAT JELAS
            if detection.text:
                if i == 0:
                    label = f"🎯 PLAT: {detection.text} ({detection.confidence:.0f}%)"
                    font_scale = 0.8  # Lebih besar
                else:
                    label = f"PLAT: {detection.text} ({detection.confidence:.0f}%)"
                    font_scale = 0.7
                
                font = cv2.FONT_HERSHEY_DUPLEX  # Font yang lebih jelas
                font_thickness = 2
                
                # Get text size for background
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Background dengan border
                bg_padding = 5
                bg_x1 = x - bg_padding
                bg_y1 = y - text_h - 15 - bg_padding  
                bg_x2 = x + text_w + bg_padding
                bg_y2 = y - 5 + bg_padding
                
                # Background hitam dengan border
                cv2.rectangle(result, (bg_x1-1, bg_y1-1), (bg_x2+1, bg_y2+1), (0, 0, 0), -1)
                cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), plate_color, -1)
                
                # Text putih tebal
                cv2.putText(result, label, (x, y - 8), font, font_scale, (255, 255, 255), font_thickness)
        
        return result
    
    def _stabilize_detections(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """
        Stabilkan detections dengan filtering dan prioritas
        """
        if not detections:
            return detections
            
        stable_detections = []
        
        # Filter 1: Validasi format plat Indonesia
        for detection in detections:
            text = detection.text.strip().upper()
            
            # Skip jika text terlalu pendek 
            if len(text) < 2:
                continue
                
            # Skip jika hanya karakter aneh atau symbol
            if not any(c.isalnum() for c in text):
                continue
            
            # VALIDASI FORMAT PLAT INDONESIA
            plate_score = self._validate_indonesian_plate(text)
            
            # Skip jika tidak mirip format plat Indonesia (threshold sangat permissive)
            if plate_score < 5:  # Threshold minimum sangat rendah untuk fragment
                continue
                
            # Bonus untuk format plat yang valid
            detection.confidence += plate_score
            detection.text = text  # Normalize ke uppercase
            
            stable_detections.append(detection)
        
        # Filter 2: Validasi ukuran dan konteks visual
        filtered = []
        for detection in stable_detections:
            x, y, w, h = detection.bbox
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Size validation yang reasonable untuk plat asli
            if (1200 <= area <= 12000 and  # Area yang sesuai plat motor (lebih toleran)
                1.8 <= aspect_ratio <= 4.5 and  # Aspect ratio plat Indonesia (lebih toleran)
                50 <= w <= 300 and  # Width reasonable (lebih toleran)
                15 <= h <= 100):  # Height reasonable (lebih toleran)
                
                # Additional visual validation (lebih permissive)
                visual_score = self._validate_visual_context(detection, x, y, w, h)
                if visual_score >= 5:  # Threshold visual validation sangat permissive
                    detection.confidence += visual_score
                    filtered.append(detection)
                else:
                    # Tetap include jika plate score tinggi meski visual score rendah
                    plate_score = self._validate_indonesian_plate(detection.text)
                    if plate_score >= 40:  # High confidence format
                        filtered.append(detection)
        
        # Filter 3: Dalam streaming mode, prioritaskan confidence tinggi
        if self.streaming_mode and len(filtered) > 3:
            # Hanya ambil 3 detection terbaik untuk stabilitas
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:3]
        
        return filtered
    
    def _validate_indonesian_plate(self, text: str) -> int:
        """
        Validasi format plat nomor Indonesia dan beri skor
        Format umum: [Area][Nomor][Huruf] contoh: B1234ABC, D5678EF, etc
        """
        if not text or len(text) < 1:  # Allow single character
            return 0
            
        text = text.upper().strip()
        score = 0
        
        # Define common Indonesian area codes first
        common_areas = ['B', 'D', 'F', 'G', 'H', 'L', 'N', 'R', 'S', 'T', 'W', 'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'DA', 'DB', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DR', 'DS', 'DT']
        
        # Pattern 1: Format standar Indonesia [Huruf][Nomor][Huruf]
        import re
        
        # Pattern lengkap: 1-2 huruf + 1-4 angka + 1-3 huruf
        standard_pattern = re.match(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$', text)
        if standard_pattern:
            score += 50
            if len(text) >= 6:  # Format lengkap
                score += 20
        
        # Pattern 2: Hanya huruf area (B, D, F, etc) + angka
        area_pattern = re.match(r'^[A-Z]{1,2}[0-9]+$', text)
        if area_pattern and len(text) >= 3:
            score += 30
        
        # Pattern 3: Angka + huruf akhir
        number_letter_pattern = re.match(r'^[0-9]+[A-Z]{1,3}$', text)
        if number_letter_pattern and len(text) >= 3:
            score += 25
        
        # Pattern 4: Fragmen yang masuk akal (part of plate)
        # Huruf area saja (B, D, F, etc)
        if len(text) == 1 and text in common_areas:
            score += 25
        
        # 2-3 huruf yang bisa jadi bagian plat
        if 2 <= len(text) <= 3 and text.isalpha():
            # Check if it could be area code or suffix
            if text in common_areas or any(text.startswith(area) for area in common_areas):
                score += 20
            else:
                # Even non-standard 2-3 letter combinations might be plate fragments
                score += 15
        
        # Bonus untuk area yang umum di Indonesia (already defined above)
        
        for area in common_areas:
            if text.startswith(area):
                score += 15
                break
        
        # Penalty untuk pattern yang tidak mirip plat
        if re.match(r'^[A-Z]{3,}$', text):  # Hanya huruf semua
            score -= 20
        
        if re.match(r'^[0-9]{3,}$', text):  # Hanya angka semua  
            score -= 20
            
        # Penalty untuk single char atau double char tanpa angka
        if len(text) <= 2 and text.isalpha():
            score -= 30
            
        return max(0, score)
    
    def _validate_visual_context(self, detection: PlateDetection, x: int, y: int, w: int, h: int) -> int:
        """
        Validasi konteks visual untuk memastikan ini benar-benar plat
        """
        score = 0
        
        # Posisi validation - plat biasanya di bagian bawah motor
        image_height = 720  # Assume standard height
        relative_y = y / image_height
        
        # Bonus jika di bagian bawah image (posisi wajar untuk plat)
        if 0.4 <= relative_y <= 0.9:
            score += 15
        elif relative_y > 0.9:  # Terlalu bawah
            score -= 10
        elif relative_y < 0.3:  # Terlalu atas
            score -= 15
            
        # Aspect ratio bonus untuk bentuk plat Indonesia
        aspect_ratio = w / h if h > 0 else 0
        if 3.0 <= aspect_ratio <= 3.8:  # Sweet spot untuk plat Indonesia
            score += 20
        elif 2.8 <= aspect_ratio <= 4.2:  # Still good
            score += 10
            
        # Size consistency validation
        area = w * h
        if 2000 <= area <= 6000:  # Optimal size untuk plat motor
            score += 15
        elif 1500 <= area <= 8000:  # Acceptable size
            score += 8
            
        # Text length validation
        text_len = len(detection.text)
        if 5 <= text_len <= 8:  # Optimal length untuk plat Indonesia
            score += 15
        elif 3 <= text_len <= 9:  # Acceptable length
            score += 8
        elif text_len < 3:  # Too short
            score -= 20
            
        return score
    
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
            "failed_ocr": self.failed_ocr,
            "false_positives": self.false_positives,
            "success_rate": round(success_rate, 1)
        }
    
    def _detect_horizontal_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Deteksi plat horizontal standard
        """
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Multiple preprocessing methods
        methods = [
            ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive_mean", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("adaptive_gaussian", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
            ("edges", cv2.Canny(gray, 50, 150))
        ]
        
        for method_name, processed in methods:
            # Find contours
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                candidate = self._analyze_contour(contour, method_name, angle=0.0)
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _detect_rotated_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Deteksi plat yang miring dengan rotated rectangle
        """
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Edge detection untuk rotated rectangles
        edges = cv2.Canny(gray, 50, 150)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 5:  # Minimum points for fitEllipse
                try:
                    # Get rotated rectangle
                    rect = cv2.minAreaRect(contour)
                    (center_x, center_y), (width, height), angle = rect
                    
                    # Normalize angle
                    if width < height:
                        width, height = height, width
                        angle += 90
                    
                    # Check if it could be a plate
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    if (self.min_area <= area <= self.max_area and
                        self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                        self.min_width <= width <= self.max_width and
                        self.min_height <= height <= self.max_height):
                        
                        # Convert to regular bbox for processing
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        x, y, w, h = cv2.boundingRect(box)
                        
                        candidate = {
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'angle': angle,
                            'method': 'rotated',
                            'rotated_rect': rect,
                            'score': self._calculate_score(area, aspect_ratio, width, height)
                        }
                        candidates.append(candidate)
                        
                except Exception as e:
                    continue
        
        return candidates
    
    def _detect_edge_based_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Deteksi berbasis edge density untuk plat yang sulit
        """
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Enhanced edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            candidate = self._analyze_contour(contour, "edge_based", angle=0.0)
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _analyze_contour(self, contour, method_name: str, angle: float = 0.0) -> Optional[Dict]:
        """
        Analisa contour untuk menentukan apakah kandidat plat dengan enhanced precision
        """
        area = cv2.contourArea(contour)

        if area < self.min_area or area > self.max_area:
            return None

        # ENHANCED: Get precise bounding box dengan contour fitting
        x, y, w, h = cv2.boundingRect(contour)

        # Try to get more precise bounding box using contour analysis
        precise_bbox = self._get_precise_bbox(contour, x, y, w, h)
        if precise_bbox:
            x, y, w, h = precise_bbox

        aspect_ratio = w / h if h > 0 else 0

        if (aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio or
            w < self.min_width or w > self.max_width or
            h < self.min_height or h > self.max_height):
            return None

        # Calculate precision score
        precision_score = self._calculate_bbox_precision(contour, x, y, w, h)
        base_score = self._calculate_score(area, aspect_ratio, w, h)

        return {
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'angle': angle,
            'method': method_name,
            'score': base_score + precision_score,
            'precision_score': precision_score,
            'contour': contour
        }
    
    def _calculate_score(self, area: float, aspect_ratio: float, width: int, height: int) -> float:
        """
        Calculate score for plate candidacy - more permissive
        """
        score = 0.0
        
        # Area score - wider tolerance
        optimal_area = 2000
        area_tolerance = optimal_area * 2.0  # Doubled tolerance
        area_score = 100 - min(100, abs(area - optimal_area) / area_tolerance * 100)
        score += area_score * 0.3
        
        # Aspect ratio score - more permissive
        if 2.0 <= aspect_ratio <= 4.0:  # Perfect range
            ratio_score = 100
        elif 1.5 <= aspect_ratio <= 5.0:  # Good range
            ratio_score = 80
        else:  # Outside range but still acceptable
            ratio_score = max(0, 50 - abs(aspect_ratio - 2.5) * 5)
        score += ratio_score * 0.4
        
        # Size score
        size_score = min(100, (width * height) / 4000 * 100)
        score += size_score * 0.2
        
        # Bonus for typical plate sizes
        if 40 <= width <= 200 and 15 <= height <= 60:
            score += 10
        
        return max(0, min(100, score))

    def _get_precise_bbox(self, contour, orig_x: int, orig_y: int, orig_w: int, orig_h: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get more precise bounding box using contour analysis untuk tight fitting
        """
        try:
            # Method 1: Use minimum area rectangle untuk better fitting
            if len(contour) >= 5:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Get bounding box dari rotated rectangle
                min_x = np.min(box[:, 0])
                max_x = np.max(box[:, 0])
                min_y = np.min(box[:, 1])
                max_y = np.max(box[:, 1])

                precise_w = max_x - min_x
                precise_h = max_y - min_y

                # Only use if it's significantly better (more compact)
                area_ratio = (precise_w * precise_h) / (orig_w * orig_h)
                if 0.7 <= area_ratio <= 0.95:  # 5-30% tighter
                    return (min_x, min_y, precise_w, precise_h)

            # Method 2: Contour-based edge fitting untuk horizontal plates
            contour_points = contour.reshape(-1, 2)

            # Find actual content boundaries
            margin = 2  # Small margin
            min_x = max(orig_x, np.min(contour_points[:, 0]) - margin)
            max_x = min(orig_x + orig_w, np.max(contour_points[:, 0]) + margin)
            min_y = max(orig_y, np.min(contour_points[:, 1]) - margin)
            max_y = min(orig_y + orig_h, np.max(contour_points[:, 1]) + margin)

            tight_w = max_x - min_x
            tight_h = max_y - min_y

            # Validate the tight bbox
            if tight_w >= 20 and tight_h >= 8:  # Minimum reasonable size
                return (int(min_x), int(min_y), int(tight_w), int(tight_h))

        except Exception:
            pass

        return None

    def _calculate_bbox_precision(self, contour, x: int, y: int, w: int, h: int) -> float:
        """
        Calculate precision score untuk bounding box fit
        """
        try:
            # Create bbox mask
            bbox_area = w * h
            contour_area = cv2.contourArea(contour)

            if bbox_area == 0 or contour_area == 0:
                return 0

            # Calculate fill ratio (how much of bbox is filled by contour)
            fill_ratio = contour_area / bbox_area

            # Calculate precision score
            if fill_ratio >= 0.6:  # Very tight fit
                precision_score = 15
            elif fill_ratio >= 0.4:  # Good fit
                precision_score = 10
            elif fill_ratio >= 0.25:  # Acceptable fit
                precision_score = 5
            else:  # Poor fit
                precision_score = 0

            # Bonus for rectangular shape (plate-like)
            try:
                rect = cv2.minAreaRect(contour)
                rect_area = rect[1][0] * rect[1][1]
                rectangularity = contour_area / rect_area if rect_area > 0 else 0

                if rectangularity >= 0.8:  # Very rectangular
                    precision_score += 10
                elif rectangularity >= 0.6:  # Somewhat rectangular
                    precision_score += 5
            except:
                pass

            return precision_score

        except Exception:
            return 0
    
    def _remove_duplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Enhanced NMS untuk remove duplicates dan merge overlapping boxes
        """
        if len(candidates) <= 1:
            return candidates

        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Enhanced NMS dengan multiple strategies
        final_candidates = []
        processed_indices = set()

        for i, candidate in enumerate(candidates):
            if i in processed_indices:
                continue

            bbox1 = candidate['bbox']
            overlapping_candidates = [candidate]
            overlapping_indices = {i}

            # Find all overlapping candidates
            for j, other_candidate in enumerate(candidates[i+1:], start=i+1):
                if j in processed_indices:
                    continue

                bbox2 = other_candidate['bbox']
                overlap = self._calculate_overlap(bbox1, bbox2)

                # Different thresholds based on context
                if overlap > 0.3:  # Lower threshold untuk better consolidation
                    # Check if they're likely the same plate
                    if self._are_same_plate_candidates(candidate, other_candidate):
                        overlapping_candidates.append(other_candidate)
                        overlapping_indices.add(j)

            # If multiple overlapping candidates, merge them intelligently
            if len(overlapping_candidates) > 1:
                merged_candidate = self._merge_overlapping_candidates(overlapping_candidates)
                if merged_candidate:
                    final_candidates.append(merged_candidate)
            else:
                final_candidates.append(candidate)

            # Mark all processed indices
            processed_indices.update(overlapping_indices)

        # Additional filtering untuk remove weak duplicates
        return self._final_nms_pass(final_candidates)

    def _are_same_plate_candidates(self, cand1: Dict, cand2: Dict) -> bool:
        """
        Determine if two candidates likely represent the same plate
        """
        bbox1 = cand1['bbox']
        bbox2 = cand2['bbox']

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate center distance
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        center_distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

        # Calculate size similarity
        area1 = w1 * h1
        area2 = w2 * h2
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0

        # Average dimension for relative distance
        avg_dimension = (w1 + h1 + w2 + h2) / 4
        relative_distance = center_distance / avg_dimension if avg_dimension > 0 else float('inf')

        # Criteria for same plate
        return (relative_distance < 0.5 and size_ratio > 0.6) or \
               (relative_distance < 0.3 and size_ratio > 0.4)

    def _merge_overlapping_candidates(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Merge multiple overlapping candidates into single best candidate
        """
        if not candidates:
            return None

        # Strategy 1: Use highest score candidate as base
        best_candidate = max(candidates, key=lambda x: x['score'])

        # Strategy 2: Create merged bounding box dari all candidates
        all_bboxes = [c['bbox'] for c in candidates]
        merged_bbox = self._merge_bounding_boxes(all_bboxes)

        # Strategy 3: Choose between original and merged based on quality
        original_area = best_candidate['bbox'][2] * best_candidate['bbox'][3]
        merged_area = merged_bbox[2] * merged_bbox[3]

        # Prefer merged if it's not significantly larger (avoid over-expansion)
        area_ratio = merged_area / original_area if original_area > 0 else float('inf')

        if area_ratio <= 1.3:  # Max 30% area increase
            # Use merged bbox but keep best candidate's other properties
            result = best_candidate.copy()
            result['bbox'] = merged_bbox
            result['score'] += 5  # Bonus for being merged dari multiple detections
            result['method'] = f"merged_{result['method']}"
            return result
        else:
            # Keep original best candidate
            return best_candidate

    def _merge_bounding_boxes(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """
        Merge multiple bounding boxes into one that encompasses all
        """
        if not bboxes:
            return (0, 0, 0, 0)

        # Find bounds
        min_x = min(bbox[0] for bbox in bboxes)
        min_y = min(bbox[1] for bbox in bboxes)
        max_x = max(bbox[0] + bbox[2] for bbox in bboxes)
        max_y = max(bbox[1] + bbox[3] for bbox in bboxes)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _final_nms_pass(self, candidates: List[Dict]) -> List[Dict]:
        """
        Final NMS pass dengan stricter thresholds
        """
        if len(candidates) <= 1:
            return candidates

        candidates.sort(key=lambda x: x['score'], reverse=True)

        final = []
        for candidate in candidates:
            bbox1 = candidate['bbox']
            is_duplicate = False

            for existing in final:
                bbox2 = existing['bbox']
                overlap = self._calculate_overlap(bbox1, bbox2)

                # Stricter threshold for final pass
                if overlap > 0.7:
                    is_duplicate = True
                    break

            if not is_duplicate:
                final.append(candidate)

        return final
    
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
    
    def _process_candidate(self, image: np.ndarray, candidate: Dict, index: int) -> Optional[PlateDetection]:
        """
        Process candidate dengan smart validation
        """
        bbox = candidate['bbox']
        x, y, w, h = bbox
        
        self.logger.debug(f"📋 Processing candidate {index}: {bbox} ({candidate['method']})")
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # Smart validation - lebih permissive
        validation_result = self._smart_validation(roi, bbox)
        
        if not validation_result['is_valid']:
            self.logger.debug(f"❌ Rejected: {validation_result['reason']}")
            return None
        
        # Multi-angle OCR untuk handle rotasi
        text, confidence = self._multi_angle_ocr(roi, candidate.get('angle', 0.0))
        
        if not text or len(text) < 2:
            self.logger.debug(f"❌ No valid text found")
            return None
        
        # Apply confidence boost dari validation
        boosted_confidence = confidence + validation_result['confidence_boost']
        boosted_confidence = max(0, min(100, boosted_confidence))
        
        if boosted_confidence < self.min_confidence:
            self.logger.debug(f"❌ Low confidence: {boosted_confidence:.1f}%")
            return None
        
        self.logger.info(f"✅ Plate detected: '{text}' ({boosted_confidence:.1f}%) via {candidate['method']}")
        
        return PlateDetection(
            text=text,
            confidence=boosted_confidence,
            bbox=bbox,
            angle=candidate.get('angle', 0.0),
            processed_image=roi.copy(),
            timestamp=time.time(),
            vehicle_type="vehicle",
            detection_method=f"robust_{candidate['method']}"
        )
    
    def _smart_validation(self, roi: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Smart validation yang lebih permissive
        """
        x, y, w, h = bbox
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi
        
        # Calculate features
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (w * h) * 100
        texture_variance = np.std(roi_gray)
        mean_intensity = np.mean(roi_gray)
        
        # Smart validation dengan thresholds yang reasonable
        reasons = []
        confidence_boost = 0
        
        # Edge density check - more permissive
        if edge_density >= self.min_edge_density:
            confidence_boost += min(10, edge_density / 2)
        else:
            reasons.append(f"low_edge_density_{edge_density:.1f}")
        
        # Texture variance check - more permissive  
        if texture_variance >= self.min_texture_variance:
            confidence_boost += min(10, texture_variance / 3)
        else:
            reasons.append(f"low_texture_variance_{texture_variance:.1f}")
        
        # Check for obviously bad candidates
        is_too_uniform = texture_variance < 3 and len(np.unique(roi_gray)) < 5
        is_too_bright = mean_intensity > 200 and texture_variance < 5
        is_too_dark = mean_intensity < 30 and texture_variance < 5
        
        # More permissive validation
        is_valid = (
            edge_density >= self.min_edge_density or texture_variance >= self.min_texture_variance
        ) and not (is_too_uniform or is_too_bright or is_too_dark)
        
        # Aspect ratio bonus
        aspect_ratio = w / h if h > 0 else 0
        if 2.0 <= aspect_ratio <= 4.0:
            confidence_boost += 5
        
        return {
            'is_valid': is_valid,
            'confidence_boost': confidence_boost,
            'edge_density': edge_density,
            'texture_variance': texture_variance,
            'reason': ', '.join(reasons) if reasons else 'passed'
        }
    
    def _multi_angle_ocr(self, roi: np.ndarray, angle: float = 0.0) -> Tuple[str, float]:
        """
        OCR dengan multiple angles untuk handle rotasi
        """
        if roi.size == 0:
            return "", 0.0
        
        results = []
        
        # Test different angles - optimize for streaming
        if self.streaming_mode:
            test_angles = [0] if angle == 0 else [0, angle]  # Minimal angles untuk speed
        else:
            test_angles = [0, angle, -angle] if angle != 0 else [0, 10, -10, 15, -15]
        
        for test_angle in test_angles:
            try:
                # Rotate if needed
                if abs(test_angle) > 2:
                    rotated_roi = self._rotate_image(roi, test_angle)
                else:
                    rotated_roi = roi
                
                # Enhance for OCR
                enhanced_roi = self._enhance_for_ocr(rotated_roi)
                
                # Multiple OCR configurations
                configs = [
                    '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ]
                
                for config in configs:
                    try:
                        # Get OCR data
                        data = pytesseract.image_to_data(
                            enhanced_roi,
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
                            cleaned_text = self._clean_text(full_text)
                            
                            if len(cleaned_text) >= 2:
                                # Bonus for good angles
                                angle_bonus = 5 if abs(test_angle) < 5 else 0
                                final_confidence = avg_confidence + angle_bonus
                                
                                results.append((cleaned_text, final_confidence, abs(test_angle)))
                    
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        if results:
            # Sort by confidence, then by text length, then by angle
            results.sort(key=lambda x: (x[1], len(x[0]), -x[2]), reverse=True)
            return results[0][0], results[0][1]
        
        return "", 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle
        """
        if abs(angle) < 1:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def _enhance_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Enhance ROI for better OCR
        """
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Upscale for small images
        h, w = roi.shape
        if h < 30 or w < 80:
            scale_factor = max(3, 30 // h, 80 // w)
            roi = cv2.resize(roi, (w * scale_factor, h * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        roi = cv2.bilateralFilter(roi, 5, 50, 50)
        
        # Enhance contrast
        roi = cv2.convertScaleAbs(roi, alpha=1.2, beta=10)
        
        return roi
    
    def _clean_text(self, text: str) -> str:
        """
        Clean OCR text
        """
        if not text:
            return ""
        
        # Remove unwanted characters
        cleaned = ''.join(c for c in text.upper() if c.isalnum())
        
        # Basic corrections for common OCR errors
        corrections = {
            '0': 'O', '1': 'I', '5': 'S', '8': 'B'
        }
        
        # Apply corrections contextually
        if len(cleaned) >= 4:
            # First character usually letter
            if cleaned[0].isdigit() and cleaned[0] in corrections:
                cleaned = corrections[cleaned[0]] + cleaned[1:]
            
            # Last few characters usually letters
            for i in range(max(1, len(cleaned) - 3), len(cleaned)):
                if i < len(cleaned) and cleaned[i].isdigit() and cleaned[i] in corrections:
                    cleaned = cleaned[:i] + corrections[cleaned[i]] + cleaned[i+1:]
        
        return cleaned

def test_robust_detector():
    """
    Test function untuk robust detector
    """
    print("🔧 Testing Robust Plate Detector")
    print("=" * 50)
    
    detector = RobustPlateDetector()
    
    # Test images
    test_images = [
        "detected_plates/screenshot_20250919_092204.jpg",
        "optimized_plate_test_20250919_100153.jpg",
        "debug_plate_final_20250919_095955.jpg"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            continue
            
        print(f"\n📸 Testing: {image_path}")
        
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
                  f"angle: {detection.angle:.1f}° method: {detection.detection_method}")

if __name__ == "__main__":
    import os
    test_robust_detector()