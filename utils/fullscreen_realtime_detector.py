#!/usr/bin/env python3
"""
Full Screen Real-Time Plate Detector
Adaptasi dari FullScreenPlateDetector untuk streaming real-time dengan optimasi performance
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from utils.ocr_ensemble import OCREnsemble
from utils.plate_detector import PlateDetection

@dataclass
class PlateCandidate:
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    text: str
    ocr_confidence: float
    visual_score: float
    combined_score: float
    roi: np.ndarray
    detection_method: str

class FullScreenRealtimeDetector:
    """
    Real-time version of FullScreenPlateDetector optimized for video streaming
    Maintains quality while optimizing for speed
    """

    def __init__(self, streaming_mode=True):
        self.streaming_mode = streaming_mode
        self.ocr_ensemble = OCREnsemble()
        self.logger = logging.getLogger(__name__)

        # Performance optimization settings
        self.frame_skip_counter = 0
        self.intensive_processing_interval = 2  # Do full processing every 2 frames (more frequent)

        # Region cache for smart processing
        self.cached_regions = []
        self.cache_timestamp = 0
        self.cache_lifetime = 1.0  # seconds

        # Statistics
        self.total_detections = 0
        self.processing_times = []

        # Enhanced detection parameters for higher sensitivity
        self.quick_window_configs = [
            (0.12, 0.03),  # Very small plate (motorcycle/distant)
            (0.15, 0.04),  # Small plate
            (0.20, 0.05),  # Medium plate (most common)
            (0.25, 0.06),  # Large plate
        ]

        self.full_window_configs = [
            (0.10, 0.025), # Tiny plate (very distant)
            (0.12, 0.03),  # Very small plate (motorcycle/distant)
            (0.15, 0.04),  # Small plate
            (0.20, 0.05),  # Medium plate
            (0.25, 0.06),  # Large plate
            (0.30, 0.08),  # Extra large plate
        ]

        self.logger.info("✅ FullScreenRealtimeDetector initialized for streaming")

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Main detection method optimized for real-time streaming
        """
        start_time = time.time()

        # Smart processing: alternate between quick and full detection
        self.frame_skip_counter += 1
        use_full_processing = (self.frame_skip_counter % self.intensive_processing_interval == 0)

        if use_full_processing:
            best_candidate = self._detect_best_plate_full(image)
            self.logger.debug("🔍 Full processing frame")
        else:
            best_candidate = self._detect_best_plate_quick(image)
            self.logger.debug("⚡ Quick processing frame")

        # Convert to PlateDetection format
        detections = []
        if best_candidate and best_candidate.text and best_candidate.combined_score > 0.15:
            # User-friendly detection logging
            confidence_percent = int(best_candidate.combined_score * 100)
            self.logger.info(f"🚗 PLATE DETECTED: {best_candidate.text} ({confidence_percent}% confidence)")

            # Debug info (only shown in debug mode)
            self.logger.debug(f"🎯 Enhanced Detection Details: '{best_candidate.text}' (score: {best_candidate.combined_score:.3f}, method: {best_candidate.detection_method})")
            detection = PlateDetection(
                text=best_candidate.text,
                confidence=best_candidate.combined_score * 100,  # Convert to percentage
                bbox=best_candidate.bbox,
                processed_image=best_candidate.roi,
                timestamp=time.time(),
                vehicle_type="unknown",
                detection_method=f"fullscreen_{best_candidate.detection_method}"
            )
            detections.append(detection)
            self.total_detections += 1

        # Track performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)

        self.logger.debug(f"🎯 Detection completed in {processing_time:.3f}s")
        return detections

    def _detect_best_plate_quick(self, image: np.ndarray) -> Optional[PlateCandidate]:
        """Quick detection for real-time performance"""

        # Use cached regions if available and recent
        current_time = time.time()
        if (self.cached_regions and
            current_time - self.cache_timestamp < self.cache_lifetime):
            candidates = self._scan_cached_regions(image)
        else:
            # Quick grid scan with reduced window configurations
            candidates = self._grid_scan_detection(image, self.quick_window_configs)
            # Cache promising regions for next frames
            self._update_region_cache(candidates, current_time)

        if not candidates:
            return None

        # Quick filtering and scoring
        filtered_candidates = self._remove_overlapping_candidates(candidates)
        scored_candidates = self._score_and_rank_candidates_quick(image, filtered_candidates)

        return scored_candidates[0] if scored_candidates else None

    def _detect_best_plate_full(self, image: np.ndarray) -> Optional[PlateCandidate]:
        """Full detection processing (every nth frame)"""

        all_candidates = []

        # Grid-based scanning with full configuration
        grid_candidates = self._grid_scan_detection(image, self.full_window_configs)
        all_candidates.extend(grid_candidates)

        # Edge-based detection (lighter version)
        edge_candidates = self._edge_based_detection_light(image)
        all_candidates.extend(edge_candidates)

        # Color-based detection (lighter version)
        color_candidates = self._color_based_detection_light(image)
        all_candidates.extend(color_candidates)

        if not all_candidates:
            return None

        # Remove overlapping candidates
        filtered_candidates = self._remove_overlapping_candidates(all_candidates)

        # Score and rank all candidates
        scored_candidates = self._score_and_rank_candidates_full(image, filtered_candidates)

        # Update cache with best regions
        if scored_candidates:
            self._update_region_cache(scored_candidates[:3], time.time())

        return scored_candidates[0] if scored_candidates else None

    def _grid_scan_detection(self, image: np.ndarray, window_configs: List[Tuple[float, float]]) -> List[PlateCandidate]:
        """Grid scan with configurable window sizes"""
        candidates = []
        height, width = image.shape[:2]

        for w_ratio, h_ratio in window_configs:
            window_w = int(width * w_ratio)
            window_h = int(height * h_ratio)

            # Increased overlap for better coverage (more sensitive)
            step_x = window_w // 6  # More overlap for better detection
            step_y = window_h // 4  # More overlap for better detection

            for y in range(0, height - window_h, step_y):
                for x in range(0, width - window_w, step_x):
                    roi = image[y:y+window_h, x:x+window_w]

                    # Quick visual assessment
                    visual_score = self._assess_visual_quality_quick(roi)

                    if visual_score > 0.25:  # Lowered threshold for better sensitivity
                        candidates.append(PlateCandidate(
                            bbox=(x, y, window_w, window_h),
                            text="",
                            ocr_confidence=0.0,
                            visual_score=visual_score,
                            combined_score=0.0,
                            roi=roi,
                            detection_method="grid_scan"
                        ))

        return candidates

    def _edge_based_detection_light(self, image: np.ndarray) -> List[PlateCandidate]:
        """Lightweight edge-based detection"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Single threshold for speed
        edges = cv2.Canny(gray, 75, 200)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300 or area > 50000:  # More sensitive for smaller plates
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if 2.0 <= aspect_ratio <= 6.0:
                roi = image[y:y+h, x:x+w]
                visual_score = self._assess_visual_quality_quick(roi)

                if visual_score > 0.2:
                    candidates.append(PlateCandidate(
                        bbox=(x, y, w, h),
                        text="",
                        ocr_confidence=0.0,
                        visual_score=visual_score,
                        combined_score=0.0,
                        roi=roi,
                        detection_method="edge_detection"
                    ))

        return candidates

    def _color_based_detection_light(self, image: np.ndarray) -> List[PlateCandidate]:
        """Lightweight color-based detection"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Single threshold for speed
        _, light_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        morph = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 250 or area > 35000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if 2.0 <= aspect_ratio <= 6.0:
                roi = image[y:y+h, x:x+w]
                visual_score = self._assess_visual_quality_quick(roi)

                if visual_score > 0.2:
                    candidates.append(PlateCandidate(
                        bbox=(x, y, w, h),
                        text="",
                        ocr_confidence=0.0,
                        visual_score=visual_score,
                        combined_score=0.0,
                        roi=roi,
                        detection_method="color_based"
                    ))

        return candidates

    def _assess_visual_quality_quick(self, roi: np.ndarray) -> float:
        """Quick visual quality assessment for real-time"""
        if roi.size == 0:
            return 0.0

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi

        # More sensitive contrast check
        contrast = gray_roi.std()
        contrast_score = min(1.0, contrast / 30.0)  # Lowered divisor for higher sensitivity

        # Quick size check
        area = gray_roi.shape[0] * gray_roi.shape[1]
        ideal_area = 3000
        size_score = 1.0 - min(1.0, abs(area - ideal_area) / ideal_area)

        return (contrast_score + size_score) / 2.0

    def _scan_cached_regions(self, image: np.ndarray) -> List[PlateCandidate]:
        """Scan cached promising regions for quick processing"""
        candidates = []

        for cached_bbox, cached_score in self.cached_regions:
            x, y, w, h = cached_bbox

            # Expand region slightly for better coverage
            expand = 10
            x = max(0, x - expand)
            y = max(0, y - expand)
            w = min(image.shape[1] - x, w + 2 * expand)
            h = min(image.shape[0] - y, h + 2 * expand)

            roi = image[y:y+h, x:x+w]
            visual_score = self._assess_visual_quality_quick(roi)

            if visual_score > 0.3:
                candidates.append(PlateCandidate(
                    bbox=(x, y, w, h),
                    text="",
                    ocr_confidence=0.0,
                    visual_score=visual_score,
                    combined_score=0.0,
                    roi=roi,
                    detection_method="cached_region"
                ))

        return candidates

    def _update_region_cache(self, candidates: List[PlateCandidate], timestamp: float):
        """Update cached regions with promising candidates"""
        self.cached_regions = []
        for candidate in candidates[:3]:  # Keep top 3
            if candidate.visual_score > 0.3:  # Lowered cache threshold for better sensitivity
                self.cached_regions.append((candidate.bbox, candidate.visual_score))
        self.cache_timestamp = timestamp

    def _remove_overlapping_candidates(self, candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Remove overlapping candidates, keep the one with higher visual score"""
        if len(candidates) <= 1:
            return candidates

        # Sort by visual score descending
        candidates.sort(key=lambda x: x.visual_score, reverse=True)

        filtered = []
        for candidate in candidates:
            x1, y1, w1, h1 = candidate.bbox

            # Check overlap with existing filtered candidates
            is_overlapping = False
            for existing in filtered:
                x2, y2, w2, h2 = existing.bbox

                # Calculate intersection
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = w1 * h1
                    area2 = w2 * h2

                    # If intersection > 30% of either area, consider overlapping
                    if inter_area > 0.3 * min(area1, area2):
                        is_overlapping = True
                        break

            if not is_overlapping:
                filtered.append(candidate)

        return filtered

    def _score_and_rank_candidates_quick(self, image: np.ndarray, candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Quick scoring for real-time performance"""
        scored_candidates = []

        for candidate in candidates[:5]:  # Limit to top 5 for speed
            # Quick OCR processing
            enhanced_roi = self._enhance_roi_for_ocr_quick(candidate.roi)

            try:
                # Use faster OCR method
                text, ocr_conf, _ = self.ocr_ensemble.paddleocr_detect(enhanced_roi)

                if text:
                    clean_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip()
                    if len(clean_text.replace(' ', '')) < 2:  # Reduced from 3 to 2 for better sensitivity
                        ocr_conf *= 0.3  # Less penalty for short text
                else:
                    clean_text = ""
                    ocr_conf = 0.0

            except Exception as e:
                clean_text = ""
                ocr_conf = 0.0

            # Quick combined scoring
            combined_score = self._calculate_combined_score_quick(
                candidate.visual_score, ocr_conf, candidate.bbox, clean_text
            )

            # Update candidate
            candidate.text = clean_text
            candidate.ocr_confidence = ocr_conf
            candidate.combined_score = combined_score

            if combined_score > 0.18:
                scored_candidates.append(candidate)

        # Sort by combined score descending
        scored_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        return scored_candidates

    def _score_and_rank_candidates_full(self, image: np.ndarray, candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Full scoring with ensemble OCR"""
        scored_candidates = []

        for candidate in candidates[:10]:  # Process more in full mode
            # Enhanced OCR processing
            enhanced_roi = self._enhance_roi_for_ocr(candidate.roi)

            try:
                text, ocr_conf, details = self.ocr_ensemble.ensemble_ocr(enhanced_roi)

                if text:
                    clean_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip()
                    if len(clean_text.replace(' ', '')) < 2 or len(clean_text.replace(' ', '')) > 15:  # Reduced minimum from 4 to 2
                        ocr_conf *= 0.3  # Less penalty for short text
                else:
                    clean_text = ""
                    ocr_conf = 0.0

            except Exception as e:
                clean_text = ""
                ocr_conf = 0.0

            # Full combined scoring
            combined_score = self._calculate_combined_score_full(
                candidate.visual_score, ocr_conf, candidate.bbox, clean_text
            )

            # Update candidate
            candidate.text = clean_text
            candidate.ocr_confidence = ocr_conf
            candidate.combined_score = combined_score

            if combined_score > 0.15:
                scored_candidates.append(candidate)

        # Sort by combined score descending
        scored_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        return scored_candidates

    def _enhance_roi_for_ocr_quick(self, roi: np.ndarray) -> np.ndarray:
        """Quick ROI enhancement for OCR"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Simple resize for OCR
        target_height = 48  # Smaller for speed
        if gray.shape[0] < target_height:
            scale = target_height / gray.shape[0]
            new_width = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_LINEAR)

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _enhance_roi_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Full ROI enhancement for OCR"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Resize untuk OCR optimal
        target_height = 64
        if gray.shape[0] < target_height:
            scale = target_height / gray.shape[0]
            new_width = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE untuk better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _calculate_combined_score_quick(self, visual_score: float, ocr_conf: float,
                                      bbox: Tuple[int, int, int, int], text: str) -> float:
        """Quick combined scoring"""
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 0

        # Simplified scoring
        scores = {
            'visual': visual_score * 0.3,
            'ocr': (ocr_conf / 100.0) * 0.4,
            'aspect': self._score_aspect_ratio(aspect_ratio) * 0.2,
            'text': self._score_text_quality(text) * 0.1
        }

        return sum(scores.values())

    def _calculate_combined_score_full(self, visual_score: float, ocr_conf: float,
                                     bbox: Tuple[int, int, int, int], text: str) -> float:
        """Full combined scoring"""
        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Full scoring with all factors
        scores = {
            'visual': visual_score * 0.25,
            'ocr': (ocr_conf / 100.0) * 0.35,
            'aspect': self._score_aspect_ratio(aspect_ratio) * 0.15,
            'size': self._score_size(area) * 0.15,
            'text_quality': self._score_text_quality(text) * 0.10
        }

        return sum(scores.values())

    def _score_aspect_ratio(self, aspect_ratio: float) -> float:
        """Score aspect ratio (Indonesian plates: ~3:1 to 4:1)"""
        ideal_ratio = 3.5
        deviation = abs(aspect_ratio - ideal_ratio) / ideal_ratio
        return max(0.0, 1.0 - deviation)

    def _score_size(self, area: int) -> float:
        """Score size (prefer medium sizes)"""
        ideal_area = 4000
        if area == 0:
            return 0.0
        deviation = abs(area - ideal_area) / ideal_area
        return max(0.0, 1.0 - deviation)

    def _score_text_quality(self, text: str) -> float:
        """Score text quality berdasarkan karakteristik plat Indonesia"""
        if not text:
            return 0.0

        clean_text = text.replace(' ', '')

        # Length score - more lenient for partial reads
        if 3 <= len(clean_text) <= 10:  # Reduced minimum from 5 to 3
            length_score = 1.0
        elif 2 <= len(clean_text) <= 12:  # Accept 2-char fragments too
            length_score = 0.8  # High score for partial reads
        else:
            length_score = 0.4  # Still give some score

        # Character composition score
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)

        if has_letters and has_numbers:
            composition_score = 1.0
        elif has_letters or has_numbers:
            composition_score = 0.7
        else:
            composition_score = 0.0

        return (length_score + composition_score) / 2.0

    def draw_detections(self, image: np.ndarray, detections: List[PlateDetection], show_roi: bool = False) -> np.ndarray:
        """
        Draw detections dengan format yang sama seperti hasil sukses static detection
        """
        result = image.copy()

        for detection in detections:
            x, y, w, h = detection.bbox

            # Draw bounding box (hijau tebal seperti successful detection)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Prepare label dengan format yang sama: "TEXT (SCORE)"
            if detection.text:
                # Format confidence as 2 decimal places like the successful detection
                confidence_display = detection.confidence / 100.0  # Convert back to 0-1 scale
                label = f"{detection.text} ({confidence_display:.2f})"
            else:
                label = f"Detected ({detection.confidence/100:.2f})"

            # Draw label dengan background (sama seperti draw_best_detection)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2

            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Position label
            label_x = x
            label_y = y - 15

            if label_y - text_height < 0:
                label_y = y + h + text_height + 15

            # Background rectangle
            cv2.rectangle(result,
                         (label_x - 5, label_y - text_height - 5),
                         (label_x + text_width + 10, label_y + baseline + 5),
                         (0, 0, 0), -1)

            # White text
            cv2.putText(result, label, (label_x, label_y),
                       font, font_scale, (255, 255, 255), font_thickness)

        return result

    def get_statistics(self) -> Dict[str, any]:
        """Get detector statistics"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0

        return {
            'detector_type': 'fullscreen_realtime',
            'total_detections': self.total_detections,
            'avg_processing_time': round(avg_processing_time, 3),
            'cache_regions': len(self.cached_regions),
            'streaming_mode': self.streaming_mode
        }