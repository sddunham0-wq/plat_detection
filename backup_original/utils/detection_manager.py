"""
Centralized Detection Manager
Unified system untuk manage semua plate detections dan eliminate duplicates
"""

import cv2
import numpy as np
import re
import time
import logging
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from utils.plate_detector import PlateDetection
from utils.scene_analyzer import scene_analyzer
from utils.validation_pipeline import validation_pipeline
from config import DetectionConfig

@dataclass
class DetectionQuality:
    """Quality scoring untuk detection"""
    geometry_score: float = 0.0      # Aspect ratio, rectangularity (30%)
    color_score: float = 0.0         # Indonesian plate colors (25%)
    text_score: float = 0.0          # Text pattern validity (35%)
    size_score: float = 0.0          # Size appropriateness (10%)
    total_score: float = 0.0         # Weighted total score
    is_valid: bool = False           # Overall validity

@dataclass
class ManagedDetection:
    """Enhanced detection dengan quality metrics"""
    detection: PlateDetection
    quality: DetectionQuality
    detection_source: str           # 'general', 'motorcycle', 'yolo', 'hybrid'
    timestamp: float
    frame_id: int
    confidence_adjusted: float = 0.0

class DetectionManager:
    """
    Centralized manager untuk semua plate detections
    - Unified counting system
    - Enhanced duplicate filtering
    - Plate-specific validation
    - Quality scoring system
    """

    def __init__(self):
        """Initialize detection manager"""
        self.logger = logging.getLogger(__name__)

        # Detection statistics - SINGLE SOURCE OF TRUTH
        self.total_detections = 0
        self.valid_detections = 0
        self.filtered_duplicates = 0
        self.filtered_non_plates = 0

        # Detection history untuk temporal filtering
        self.detection_history: List[ManagedDetection] = []
        self.plate_last_seen: Dict[str, float] = {}

        # Indonesian plate patterns
        self.indonesian_patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',  # B 1234 ABC
            r'^[A-Z]{2}\s*\d{2,4}\s*[A-Z]{2,3}$',    # AB 1234 CD
            r'^[A-Z]\s*\d{1,4}\s*[A-Z]{2,3}$',       # B 123 AB
        ]

        # Indonesian plate colors (BGR format)
        self.plate_colors = {
            'white': {'lower': (180, 180, 180), 'upper': (255, 255, 255)},
            'yellow': {'lower': (0, 200, 200), 'upper': (50, 255, 255)},
            'red': {'lower': (0, 0, 150), 'upper': (50, 50, 255)},
            'black': {'lower': (0, 0, 0), 'upper': (50, 50, 50)}
        }

        self.logger.info("Detection Manager initialized")

    def process_detections(self, detections_dict: Dict[str, List[PlateDetection]],
                          frame: np.ndarray, frame_id: int) -> List[PlateDetection]:
        """
        Process detections dengan comprehensive validation pipeline

        Args:
            detections_dict: Dictionary dengan key = source, value = detections
            frame: Original frame untuk quality analysis
            frame_id: Frame ID untuk tracking

        Returns:
            List of validated, deduplicated detections
        """
        current_time = time.time()

        # Step 1: Scene Analysis untuk early rejection
        scene_analysis = scene_analyzer.analyze_frame(frame)

        # Early rejection untuk empty scenes
        if scene_analysis.likely_empty:
            self.logger.debug("Empty scene detected - skipping detection processing")
            return []

        all_validated_detections = []

        # Step 2: Multi-stage validation pipeline
        for source, detections in detections_dict.items():
            for detection in detections:
                # Run comprehensive validation pipeline
                pipeline_result = validation_pipeline.validate_detection(
                    detection, frame, scene_analysis
                )

                if pipeline_result.is_valid:
                    # Apply scene-based confidence adjustment
                    adjusted_confidence = detection.confidence * pipeline_result.confidence_modifier

                    # Create managed detection dengan pipeline results
                    managed_detection = ManagedDetection(
                        detection=detection,
                        quality=DetectionQuality(
                            geometry_score=pipeline_result.total_score,
                            color_score=0.8,  # From pipeline
                            text_score=0.8,   # From pipeline
                            size_score=0.8,   # From pipeline
                            total_score=pipeline_result.total_score,
                            is_valid=True
                        ),
                        detection_source=source,
                        timestamp=current_time,
                        frame_id=frame_id,
                        confidence_adjusted=adjusted_confidence
                    )
                    all_validated_detections.append(managed_detection)
                else:
                    self.filtered_non_plates += 1
                    self.logger.debug(f"Detection rejected: {pipeline_result.rejection_reason}")

        # Step 3: Enhanced duplicate filtering
        unique_detections = self._enhanced_duplicate_filtering(all_validated_detections)

        # Step 4: Temporal filtering
        final_detections = self._temporal_filtering(unique_detections, current_time)

        # Update statistics
        self.total_detections += len(final_detections)
        self.valid_detections += len(final_detections)  # All are valid after pipeline

        # Convert back ke PlateDetection format
        result_detections = []
        for managed_det in final_detections:
            # Update confidence dengan adjusted value
            managed_det.detection.confidence = managed_det.confidence_adjusted
            result_detections.append(managed_det.detection)

        # Update detection history (keep last 100)
        self.detection_history.extend(final_detections)
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]

        self.logger.debug(f"Pipeline processed: {sum(len(detections) for detections in detections_dict.values())} → {len(final_detections)} valid detections")
        return result_detections

    def _calculate_quality_score(self, detection: PlateDetection, frame: np.ndarray) -> DetectionQuality:
        """
        Calculate comprehensive quality score untuk detection

        Args:
            detection: PlateDetection object
            frame: Original frame

        Returns:
            DetectionQuality object dengan detailed scoring
        """
        quality = DetectionQuality()

        # Extract bounding box
        x, y, w, h = detection.bbox

        # 1. Geometry Score (30% weight)
        aspect_ratio = w / h if h > 0 else 0

        # Indonesian plate aspect ratio: 1.5 - 5.5
        if 1.5 <= aspect_ratio <= 5.5:
            geometry_score = 1.0 - abs(aspect_ratio - 3.0) / 3.0  # Optimal around 3.0
        else:
            geometry_score = 0.0

        # Rectangularity check
        if detection.processed_image is not None:
            rectangularity = self._calculate_rectangularity(detection.processed_image)
            geometry_score = (geometry_score + rectangularity) / 2

        quality.geometry_score = max(0.0, min(1.0, geometry_score))

        # 2. Color Score (25% weight)
        if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
            roi = frame[y:y+h, x:x+w]
            quality.color_score = self._calculate_color_score(roi)
        else:
            quality.color_score = 0.0

        # 3. Text Score (35% weight)
        if detection.text:
            quality.text_score = self._calculate_text_score(detection.text)
        else:
            quality.text_score = 0.0

        # 4. Size Score (10% weight)
        frame_area = frame.shape[0] * frame.shape[1]
        detection_area = w * h
        size_ratio = detection_area / frame_area

        # Optimal size: 0.1% - 5% of frame
        if 0.001 <= size_ratio <= 0.05:
            quality.size_score = 1.0
        elif size_ratio < 0.001:
            quality.size_score = size_ratio / 0.001  # Penalty untuk terlalu kecil
        else:
            quality.size_score = max(0.0, 1.0 - (size_ratio - 0.05) / 0.05)

        # Calculate weighted total score
        quality.total_score = (
            quality.geometry_score * 0.30 +
            quality.color_score * 0.25 +
            quality.text_score * 0.35 +
            quality.size_score * 0.10
        )

        # Determine validity (STRICT criteria)
        quality.is_valid = (
            quality.total_score >= 0.8 and
            quality.geometry_score >= 0.6 and
            quality.text_score >= 0.6 and
            quality.color_score >= 0.3
        )

        return quality

    def _calculate_rectangularity(self, image: np.ndarray) -> float:
        """Calculate rectangularity score dari plate image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.0

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate rectangularity
            area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect_area = w * h

            if rect_area > 0:
                return area / rect_area
            else:
                return 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating rectangularity: {str(e)}")
            return 0.0

    def _calculate_color_score(self, roi: np.ndarray) -> float:
        """
        Calculate color score berdasarkan Indonesian plate colors

        Args:
            roi: Region of interest (detected plate area)

        Returns:
            Color score (0.0 - 1.0)
        """
        try:
            if roi.size == 0:
                return 0.0

            # Convert ke HSV untuk better color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Calculate histogram untuk dominant colors
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

            # Check untuk plate colors
            color_scores = []

            # White plate (most common)
            white_pixels = np.sum((hsv[:,:,2] > 200) & (hsv[:,:,1] < 50))
            white_ratio = white_pixels / roi.size
            color_scores.append(white_ratio)

            # Yellow plate (public transport)
            yellow_pixels = np.sum((hsv[:,:,0] >= 15) & (hsv[:,:,0] <= 35) &
                                 (hsv[:,:,1] > 100) & (hsv[:,:,2] > 100))
            yellow_ratio = yellow_pixels / roi.size
            color_scores.append(yellow_ratio * 0.8)  # Slightly lower weight

            # Red plate (government)
            red_pixels = np.sum(((hsv[:,:,0] <= 10) | (hsv[:,:,0] >= 170)) &
                              (hsv[:,:,1] > 100) & (hsv[:,:,2] > 100))
            red_ratio = red_pixels / roi.size
            color_scores.append(red_ratio * 0.9)

            # Black plate (old format)
            black_pixels = np.sum(hsv[:,:,2] < 50)
            black_ratio = black_pixels / roi.size
            color_scores.append(black_ratio * 0.7)

            return max(color_scores)

        except Exception as e:
            self.logger.warning(f"Error calculating color score: {str(e)}")
            return 0.0

    def _calculate_text_score(self, text: str) -> float:
        """
        Calculate text score berdasarkan Indonesian plate patterns

        Args:
            text: Detected text

        Returns:
            Text score (0.0 - 1.0)
        """
        if not text:
            return 0.0

        # Clean text
        cleaned_text = re.sub(r'[^A-Z0-9\s]', '', text.upper().strip())

        # Length check
        if len(cleaned_text) < 5 or len(cleaned_text) > 12:
            return 0.0

        # Pattern matching
        pattern_scores = []
        for pattern in self.indonesian_patterns:
            if re.match(pattern, cleaned_text):
                pattern_scores.append(1.0)
            else:
                # Partial matching score
                pattern_chars = len(re.findall(r'[A-Z0-9]', pattern))
                text_chars = len(re.findall(r'[A-Z0-9]', cleaned_text))
                if pattern_chars > 0:
                    partial_score = min(text_chars / pattern_chars, 1.0) * 0.6
                    pattern_scores.append(partial_score)

        if not pattern_scores:
            return 0.0

        base_score = max(pattern_scores)

        # Character composition bonus
        letters = len(re.findall(r'[A-Z]', cleaned_text))
        numbers = len(re.findall(r'[0-9]', cleaned_text))

        # Indonesian plates typically have 1-4 letters and 1-4 numbers
        if 1 <= letters <= 4 and 1 <= numbers <= 4:
            composition_bonus = 0.2
        else:
            composition_bonus = 0.0

        return min(1.0, base_score + composition_bonus)

    def _enhanced_duplicate_filtering(self, detections: List[ManagedDetection]) -> List[ManagedDetection]:
        """
        Enhanced duplicate filtering dengan improved IoU threshold

        Args:
            detections: List of ManagedDetection objects

        Returns:
            Filtered unique detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence_adjusted (descending)
        sorted_detections = sorted(detections, key=lambda x: x.confidence_adjusted, reverse=True)

        unique_detections = []

        for current_detection in sorted_detections:
            is_duplicate = False
            current_bbox = current_detection.detection.bbox

            for unique_detection in unique_detections:
                unique_bbox = unique_detection.detection.bbox

                # Calculate IoU
                iou = self._calculate_iou(current_bbox, unique_bbox)

                # Enhanced duplicate criteria
                if iou > 0.65:  # Increased from 0.30 to 0.65
                    # Additional text similarity check
                    text_similarity = self._calculate_text_similarity(
                        current_detection.detection.text,
                        unique_detection.detection.text
                    )

                    if iou > 0.8 or (iou > 0.5 and text_similarity > 0.8):
                        is_duplicate = True
                        self.filtered_duplicates += 1
                        break

            if not is_duplicate:
                unique_detections.append(current_detection)

        return unique_detections

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)

        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
        else:
            intersection = 0

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score"""
        if not text1 or not text2:
            return 0.0

        # Clean texts
        clean1 = re.sub(r'[^A-Z0-9]', '', text1.upper())
        clean2 = re.sub(r'[^A-Z0-9]', '', text2.upper())

        if clean1 == clean2:
            return 1.0

        # Levenshtein distance-based similarity
        if len(clean1) == 0 or len(clean2) == 0:
            return 0.0

        # Simple character overlap ratio
        common_chars = len(set(clean1) & set(clean2))
        total_chars = len(set(clean1) | set(clean2))

        return common_chars / total_chars if total_chars > 0 else 0.0

    def _temporal_filtering(self, detections: List[ManagedDetection],
                           current_time: float) -> List[ManagedDetection]:
        """
        Apply temporal filtering untuk avoid rapid re-detection

        Args:
            detections: Current frame detections
            current_time: Current timestamp

        Returns:
            Temporally filtered detections
        """
        filtered_detections = []
        duplicate_threshold = 12.0  # Increased from 5 to 12 seconds

        for detection in detections:
            plate_text = detection.detection.text

            if not plate_text:
                continue

            # Check temporal duplicates
            if plate_text in self.plate_last_seen:
                time_diff = current_time - self.plate_last_seen[plate_text]
                if time_diff < duplicate_threshold:
                    # Skip this detection (too recent)
                    continue

            # Accept detection dan update timestamp
            self.plate_last_seen[plate_text] = current_time
            filtered_detections.append(detection)

        # Cleanup old entries (older than 5 minutes)
        cleanup_threshold = current_time - 300.0
        self.plate_last_seen = {
            plate: timestamp for plate, timestamp in self.plate_last_seen.items()
            if timestamp > cleanup_threshold
        }

        return filtered_detections

    def get_statistics(self) -> Dict:
        """Get comprehensive detection statistics"""
        return {
            'total_detections': self.total_detections,
            'valid_detections': self.valid_detections,
            'filtered_duplicates': self.filtered_duplicates,
            'filtered_non_plates': self.filtered_non_plates,
            'success_rate': (self.valid_detections / self.total_detections * 100) if self.total_detections > 0 else 0,
            'duplicate_filter_rate': (self.filtered_duplicates / (self.total_detections + self.filtered_duplicates) * 100) if (self.total_detections + self.filtered_duplicates) > 0 else 0,
            'detection_history_count': len(self.detection_history),
            'tracked_plates_count': len(self.plate_last_seen)
        }

    def reset_statistics(self):
        """Reset semua statistics"""
        self.total_detections = 0
        self.valid_detections = 0
        self.filtered_duplicates = 0
        self.filtered_non_plates = 0
        self.detection_history.clear()
        self.plate_last_seen.clear()
        self.logger.info("Detection statistics reset")

# Singleton instance
detection_manager = DetectionManager()