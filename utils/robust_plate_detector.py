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

        # Performance optimizations - batch processing cache
        self.candidate_cache = {}  # Cache untuk candidate detection
        self.batch_size = 3  # Process candidates in batches untuk efficiency

        if streaming_mode:
            # STREAMING MODE - ULTRA-SENSITIVE for distant/challenging plate detection
            self.min_area = 150  # Reduced from 200 to 150 untuk very small/distant plates
            self.max_area = 20000  # Increased to 20000 untuk better coverage
            self.min_aspect_ratio = 0.5  # Reduced from 0.8 to 0.5 untuk allow vertical contours (aggressive dilation side-effect)
            self.max_aspect_ratio = 7.0  # Increased from 6.0 to 7.0 untuk wider range
            self.min_width = 15  # Reduced from 20 to 15 untuk smaller plate detection
            self.max_width = 400  # Increased back to 400 untuk better coverage
            self.min_height = 6   # Reduced from 8 to 6 untuk very distant plates
            self.max_height = 150 # Increased back to 150 untuk better range
            self.min_confidence = 20   # Reduced from 25 to 20 untuk better sensitivity while maintaining quality
            self.min_text_likelihood = 25  # Reduced from 30 to 25 untuk balanced text validation
            self.max_candidates = 10  # Increased from 5 to 10 untuk process more candidates
        else:
            # FULL MODE - SMART BALANCED parameter for realistic plate detection
            self.min_area = 600  # Increased from 100 to 600 for realistic plates
            self.max_area = 25000  # Reduced from 30000 to 25000
            self.min_aspect_ratio = 1.5  # Restored from 1.0 to 1.5
            self.max_aspect_ratio = 6.0  # Reduced from 8.0 to 6.0
            self.min_width = 30  # Increased from 15 to 30
            self.max_width = 400  # Reduced from 500 to 400
            self.min_height = 12   # Increased from 5 to 12
            self.max_height = 150 # Reduced from 200 to 150
            self.min_confidence = 25  # Increased from 10 to 25
            self.min_text_likelihood = 30  # Increased from 15 to 30
            self.max_candidates = 8  # Reduced from 12 to 8 untuk speed optimization

        # Statistics tracking
        self.total_detections = 0
        self.successful_ocr = 0
        self.failed_ocr = 0
        self.false_positives = 0
        self.min_edge_density = 0.6  # Reduced from 1.2 to 0.6 for distant/blur plates (50% reduction)
        self.min_texture_variance = 2.0  # Reduced from 4.0 to 2.0 for low-contrast plates (50% reduction)

        self.logger.info("🔧 Robust Plate Detector initialized with batch processing optimization")
    
    def detect_plates(self, image: np.ndarray, vehicle_type: str = 'general') -> List[PlateDetection]:
        """
        Main detection method dengan smart ROI untuk fokus area plat

        Args:
            image: Input image
            vehicle_type: Type of vehicle for ROI selection ('motorcycle', 'car', 'truck', 'general')
        """
        if image is None or image.size == 0:
            return []

        detections = []

        try:
            # Apply smart ROI untuk fokus area deteksi
            roi_image, roi_offset = self._apply_smart_roi(image, vehicle_type)

            # Multi-stage detection with batch processing optimization
            candidates = []

            # Clear conflicting cache for stability - fresh detection each time
            self.candidate_cache.clear()  # Clear cache untuk eliminate false positive accumulation

            if self.streaming_mode:
                # STREAMING MODE: Single best method untuk maximum stability
                horizontal_candidates = self._detect_horizontal_plates(roi_image)
                # Adjust coordinates untuk original image
                horizontal_candidates = self._adjust_candidates_coordinates(horizontal_candidates, roi_offset)
                candidates.extend(horizontal_candidates[:self.max_candidates])  # Limit immediately
            else:
                # FULL MODE: Primary horizontal detection with fallback edge detection
                # Stage 1: Standard horizontal detection (primary method)
                horizontal_candidates = self._detect_horizontal_plates(roi_image)
                horizontal_candidates = self._adjust_candidates_coordinates(horizontal_candidates, roi_offset)
                candidates.extend(horizontal_candidates)

                # Stage 2: Enhanced edge-based detection (fallback only if needed)
                if len(horizontal_candidates) < 2:  # Only use fallback if primary method finds few candidates
                    edge_candidates = self._detect_edge_based_plates(roi_image)
                    edge_candidates = self._adjust_candidates_coordinates(edge_candidates, roi_offset)
                    candidates.extend(edge_candidates[:2])  # Limit fallback candidates
            
            self.logger.info(f"🔍 Found {len(candidates)} total candidates from all methods")
            
            # Remove duplicates dengan strict filtering
            unique_candidates = self._remove_duplicate_candidates(candidates)
            self.logger.info(f"🔧 After duplicate removal: {len(unique_candidates)} unique candidates")

            # Additional quality filtering untuk eliminate low-quality candidates
            quality_candidates = []
            for candidate in unique_candidates:
                if candidate.get('score', 0) >= 5:  # Reduced from 10 to 5 untuk maximum sensitivity
                    quality_candidates.append(candidate)

            self.logger.info(f"🎯 After quality filtering: {len(quality_candidates)} quality candidates")
            unique_candidates = quality_candidates
            
            # Process each candidate - limit based on mode
            max_process = self.max_candidates
            for i, candidate in enumerate(unique_candidates[:max_process]):
                detection = self._process_candidate(image, candidate, i+1)
                if detection:
                    # Apply ROI confidence boost
                    detection = self._apply_roi_confidence_boost(detection)
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
            
            # Skip jika text kosong - accept any readable text untuk CCTV
            if len(text) < 1:
                continue
                
            # Skip jika hanya karakter aneh atau symbol
            if not any(c.isalnum() for c in text):
                continue
            
            # VALIDASI FORMAT PLAT INDONESIA
            plate_score = self._validate_indonesian_plate(text)
            
            # Very relaxed format validation untuk CCTV - accept any reasonable text
            if plate_score < 1:  # Minimal validation untuk CCTV edge cases
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
            
            # Size validation yang reasonable untuk plat asli - STABILIZED
            if (400 <= area <= 12000 and  # Realistic area range untuk actual plates
                1.5 <= aspect_ratio <= 5.0 and  # Standard plate aspect ratios
                30 <= w <= 350 and  # Realistic width range untuk plates
                12 <= h <= 100):  # Realistic height range untuk plates
                
                # Additional visual validation - STABILIZED
                visual_score = self._validate_visual_context(detection, x, y, w, h)
                if visual_score >= 8:  # Increased from 4 to 8 untuk better quality filtering
                    detection.confidence += visual_score
                    filtered.append(detection)
                else:
                    # Include only high-scoring plates dengan strict requirements
                    plate_score = self._validate_indonesian_plate(detection.text)
                    if plate_score >= 50 and visual_score >= 6:  # Stricter untuk eliminate false positives
                        filtered.append(detection)
        
        # Filter 3: Dalam streaming mode, prioritaskan confidence tinggi - PERFORMANCE OPTIMIZED
        if self.streaming_mode and len(filtered) > 3:  # Reduced to 3 untuk speed optimization
            # Ambil 3 detection terbaik untuk balance speed vs coverage
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:3]
        
        return filtered
    
    def _validate_indonesian_plate(self, text: str) -> int:
        """
        Validasi format plat nomor Indonesia dan beri skor
        Format umum: [Area][Nomor][Huruf] contoh: B1234ABC, D5678EF, etc
        """
        if not text:  # Only check for empty text, allow any length including single character
            return 0
            
        text = text.upper().strip()
        score = 0
        
        # Define common Indonesian area codes first
        common_areas = ['B', 'D', 'F', 'G', 'H', 'L', 'N', 'R', 'S', 'T', 'W', 'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE', 'BG', 'BH', 'BK', 'BL', 'BM', 'BN', 'DA', 'DB', 'DD', 'DE', 'DG', 'DH', 'DK', 'DL', 'DM', 'DN', 'DR', 'DS', 'DT']
        
        # Pattern 1: Format standar Indonesia [Huruf][Nomor][Huruf]
        import re
        
        # Pattern lengkap: 1-2 huruf + 1-4 angka + 1-3 huruf - CCTV OPTIMIZED
        standard_pattern = re.match(r'^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$', text)
        if standard_pattern:
            score += 100  # High score untuk complete Indonesian format
            if len(text) >= 6:  # Format lengkap
                score += 50

        # Pattern 2: Hanya huruf area (B, D, F, etc) + angka - CCTV OPTIMIZED
        area_pattern = re.match(r'^[A-Z]{1,2}[0-9]+$', text)
        if area_pattern and len(text) >= 2:  # Relaxed untuk CCTV partial reads
            score += 60

        # Pattern 3: Angka + huruf akhir - CCTV OPTIMIZED
        number_letter_pattern = re.match(r'^[0-9]+[A-Z]{1,3}$', text)
        if number_letter_pattern and len(text) >= 2:  # Relaxed untuk CCTV partial reads
            score += 55

        # Pattern 4: Any reasonable alphanumeric untuk CCTV edge cases
        if re.match(r'^[A-Z0-9]+$', text) and len(text) >= 2:
            # Basic alphanumeric that could be part of a plate
            score += 30
            # Bonus for likely plate fragments
            if any(c.isdigit() for c in text) and any(c.isalpha() for c in text):
                score += 20  # Mixed alphanumeric
        
        # Pattern 4: Fragmen yang masuk akal (part of plate) - EXTREME ENHANCED SCORING
        # Huruf area saja (B, D, F, etc)
        if len(text) == 1 and text in common_areas:
            score += 35  # Increased from 25 to 35

        # 2-3 huruf yang bisa jadi bagian plat - ENHANCED SCORING
        if 2 <= len(text) <= 3 and text.isalpha():
            # Check if it could be area code or suffix
            if text in common_areas or any(text.startswith(area) for area in common_areas):
                score += 30  # Increased from 20 to 30
            else:
                # Even non-standard 2-3 letter combinations might be plate fragments
                score += 25  # Increased from 15 to 25

        # NEW: Single digit bonus for number fragments
        if len(text) == 1 and text.isdigit():
            score += 15  # Bonus for single digit fragments
        
        # Bonus untuk area yang umum di Indonesia (already defined above)
        
        for area in common_areas:
            if text.startswith(area):
                score += 20  # Increased from 15 to 20
                break

        # Penalty untuk pattern yang tidak mirip plat - REDUCED PENALTIES
        if len(text) >= 4 and re.match(r'^[A-Z]{4,}$', text):  # Only penalize long letter-only strings
            score -= 10  # Reduced from 20 to 10

        if len(text) >= 4 and re.match(r'^[0-9]{4,}$', text):  # Only penalize long number-only strings
            score -= 10  # Reduced from 20 to 10

        # REMOVED: Penalty for single/double char alpha - allow fragments
            
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

    def _apply_smart_roi(self, image: np.ndarray, vehicle_type: str = 'general') -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Apply smart ROI based on vehicle type untuk fokus deteksi plat

        Args:
            image: Input image
            vehicle_type: Vehicle type for ROI selection

        Returns:
            Tuple of (ROI image, (x_offset, y_offset))
        """
        from config import DetectionConfig

        h, w = image.shape[:2]

        # Get ROI zone based on vehicle type
        if DetectionConfig.ENABLE_SMART_ROI and vehicle_type in DetectionConfig.ROI_ZONES:
            roi_zone = DetectionConfig.ROI_ZONES[vehicle_type]
        else:
            roi_zone = DetectionConfig.ROI_ZONES[DetectionConfig.DEFAULT_ROI_ZONE]

        # Calculate ROI coordinates
        x_percent, y_percent, w_percent, h_percent = roi_zone

        x1 = int(w * x_percent)
        y1 = int(h * y_percent)
        x2 = int(w * (x_percent + w_percent))
        y2 = int(h * (y_percent + h_percent))

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))

        # Extract ROI
        roi_image = image[y1:y2, x1:x2]
        roi_offset = (x1, y1)

        self.logger.debug(f"🎯 Smart ROI applied for {vehicle_type}: {(x1, y1, x2-x1, y2-y1)} from {(w, h)}")

        return roi_image, roi_offset

    def _adjust_candidates_coordinates(self, candidates: List[Dict], roi_offset: Tuple[int, int]) -> List[Dict]:
        """
        Adjust candidate coordinates from ROI back to original image coordinates

        Args:
            candidates: List of candidate dictionaries
            roi_offset: (x_offset, y_offset) from ROI

        Returns:
            List of candidates with adjusted coordinates
        """
        x_offset, y_offset = roi_offset
        adjusted_candidates = []

        for candidate in candidates:
            # Create copy to avoid modifying original
            adjusted_candidate = candidate.copy()

            # Adjust bounding box coordinates
            if 'bbox' in adjusted_candidate:
                x, y, w, h = adjusted_candidate['bbox']
                adjusted_candidate['bbox'] = (x + x_offset, y + y_offset, w, h)

            # Adjust contour points if present
            if 'contour' in adjusted_candidate and adjusted_candidate['contour'] is not None:
                adjusted_contour = adjusted_candidate['contour'].copy()
                adjusted_contour[:, 0, 0] += x_offset  # Adjust x coordinates
                adjusted_contour[:, 0, 1] += y_offset  # Adjust y coordinates
                adjusted_candidate['contour'] = adjusted_contour

            adjusted_candidates.append(adjusted_candidate)

        return adjusted_candidates

    def _apply_roi_confidence_boost(self, detection) -> any:
        """
        Apply confidence boost untuk detection dalam ROI area

        Args:
            detection: PlateDetection object

        Returns:
            PlateDetection with boosted confidence
        """
        from config import DetectionConfig

        # Apply ROI confidence boost
        if hasattr(DetectionConfig, 'ROI_CONFIDENCE_BOOST'):
            detection.confidence += DetectionConfig.ROI_CONFIDENCE_BOOST
            # Cap at 100%
            detection.confidence = min(detection.confidence, 100.0)

        return detection
    
    def _detect_horizontal_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Deteksi plat horizontal standard
        """
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Focused preprocessing methods - stability first
        methods = [
            ("adaptive_gaussian", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),  # Primary: best for plates
            ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])  # Fallback: for clear contrast
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
        
        # Edge detection untuk rotated rectangles - ULTRA-SENSITIVE
        edges = cv2.Canny(gray, 30, 120)  # Reduced from (50, 150) for distant plate detection

        # ULTRA-AGGRESSIVE morphological operations untuk connect fragmented plate characters
        # Step 1: Small closing untuk cleanup noise
        kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_cleanup)

        # Step 2: STRONG horizontal dilation untuk connect characters into plate regions
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Wide horizontal kernel
        edges = cv2.dilate(edges, kernel_horizontal, iterations=2)

        # Step 3: Additional vertical connection untuk full plate height
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        edges = cv2.dilate(edges, kernel_vertical, iterations=1)
        
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
        
        # Enhanced edge detection - ULTRA-SENSITIVE
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 20, 80)  # Reduced from (30, 100) for maximum sensitivity

        # ULTRA-AGGRESSIVE dilation untuk distant/fragmented plates
        # Horizontal connection untuk plate characters
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))  # Increased from (5,1)
        dilated = cv2.dilate(edges, kernel_horizontal, iterations=3)  # Increased from 1 to 3 iterations

        # Additional vertical connection untuk full plate height
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        dilated = cv2.dilate(dilated, kernel_vertical, iterations=1)
        
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
        
        # Apply minimal confidence filtering untuk CCTV edge cases
        if boosted_confidence < 5 or confidence < 1:  # Minimal threshold untuk CCTV detection
            self.logger.debug(f"❌ Low confidence: {boosted_confidence:.1f}% (original: {confidence:.1f}%)")
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
        edges = cv2.Canny(roi_gray, 30, 120)  # Reduced from (50, 150) for ultra-sensitivity
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
        
        # Test different angles - stabilized for accuracy
        if self.streaming_mode:
            test_angles = [0]  # Only test original angle untuk maximum stability
        else:
            test_angles = [0, 10, -10] if angle == 0 else [0, angle]  # Limited angles untuk focus
        
        for test_angle in test_angles:
            try:
                # Rotate if needed
                if abs(test_angle) > 2:
                    rotated_roi = self._rotate_image(roi, test_angle)
                else:
                    rotated_roi = roi
                
                # Enhance for OCR
                enhanced_roi = self._enhance_for_ocr(rotated_roi)
                
                # ENHANCED OCR configurations for CCTV - multiple strategies
                configs = [
                    # Primary: Uniform block - best for license plates
                    '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c classify_bln_numeric_mode=1',
                    # Secondary: Single text line - for aligned plates
                    '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    # Tertiary: Single word - for partial reads
                    '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    # Quaternary: Fully automatic - for challenging cases
                    '--psm 3 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    # Ultimate fallback: Raw line without OSD
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

                        # Fallback: jika image_to_data tidak menghasilkan text, coba image_to_string
                        if not texts:
                            try:
                                fallback_text = pytesseract.image_to_string(
                                    enhanced_roi,
                                    lang='eng',
                                    config=config
                                ).strip()
                                if fallback_text:
                                    clean_text = ''.join(c for c in fallback_text if c.isalnum())
                                    if clean_text:
                                        texts = [clean_text]
                                        confidences = [70.0]  # Default confidence untuk fallback
                            except Exception:
                                pass

                        if texts and confidences:
                            full_text = ''.join(texts).upper()
                            avg_confidence = np.mean(confidences)
                            
                            # Clean text
                            cleaned_text = self._clean_text(full_text)
                            
                            if len(cleaned_text) >= 2:  # Relaxed untuk CCTV detection, accept partial reads
                                # Bonus for good angles
                                angle_bonus = 5 if abs(test_angle) < 5 else 0
                                final_confidence = avg_confidence + angle_bonus

                                # CCTV bonus - boost confidence untuk reasonable text
                                if len(cleaned_text) >= 4:  # Longer text gets confidence boost
                                    final_confidence += 10
                                if any(c.isdigit() for c in cleaned_text) and any(c.isalpha() for c in cleaned_text):
                                    final_confidence += 5  # Mixed alphanumeric gets boost

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
        ULTRA-ENHANCED ROI preprocessing untuk challenging conditions:
        - Distant plates (small size)
        - Glass reflections/distortions
        - Poor lighting/contrast
        """
        try:
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()

            h, w = gray.shape
            original_size = (w, h)

            # === PHASE 1: ADVANCED DENOISING (for glass reflections) ===
            # Non-local Means Denoising - excellent for removing glass artifacts
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

            # === PHASE 2: SUPER-RESOLUTION UPSCALING ===
            # Use LANCZOS4 for highest quality upscaling (better than CUBIC for text)
            target_height = max(48, h * 4)  # Increased minimum to 48px
            target_width = max(144, w * 4)  # Increased minimum to 144px

            # Always upscale distant/small plates
            if h < 50 or w < 150:
                upscaled = cv2.resize(denoised, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                self.logger.debug(f"🔍 Super-resolution: {w}x{h} → {target_width}x{target_height} (LANCZOS4)")
            else:
                upscaled = denoised

            # === PHASE 3: REFLECTION REMOVAL ===
            # Enhanced bilateral filter to remove glass reflections while preserving text edges
            # Larger d and sigma values for stronger reflection removal
            reflection_removed = cv2.bilateralFilter(upscaled, d=11, sigmaColor=90, sigmaSpace=90)

            # === PHASE 4: ADAPTIVE CONTRAST ENHANCEMENT ===
            # CLAHE with optimized parameters for CCTV conditions
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced_contrast = clahe.apply(reflection_removed)

            # === PHASE 5: TEXT ENHANCEMENT ===
            # Morphological operations to enhance text clarity
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            text_enhanced = cv2.morphologyEx(enhanced_contrast, cv2.MORPH_CLOSE, kernel)

            # === PHASE 6: SHARPENING ===
            # Unsharp mask for text sharpening
            gaussian = cv2.GaussianBlur(text_enhanced, (0, 0), 2.0)
            sharpened = cv2.addWeighted(text_enhanced, 1.8, gaussian, -0.8, 0)

            # === PHASE 7: FINAL CONTRAST & BRIGHTNESS ===
            # Stronger adjustments for distant plates
            final = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=20)

            # === PHASE 8: LIGHT SMOOTHING ===
            # Very light smoothing to reduce artifacts without losing text detail
            final = cv2.GaussianBlur(final, (1, 1), 0)

            return final

        except Exception as e:
            self.logger.warning(f"Ultra-enhanced preprocessing failed: {e}, using fallback")
            # Fallback: basic preprocessing
            try:
                if len(roi.shape) == 3:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray = roi.copy()

                h, w = gray.shape

                # Basic upscaling
                if h < 40 or w < 120:
                    scale_factor = max(40//h, 120//w, 3)
                    gray = cv2.resize(gray, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)

                # Basic enhancement
                gray = cv2.convertScaleAbs(gray, alpha=1.3, beta=15)
                return gray

            except Exception as e2:
                self.logger.error(f"Fallback preprocessing failed: {e2}")
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