"""
Multi-Stage Validation Pipeline
Comprehensive validation untuk prevent false positive plate detections
"""

import cv2
import numpy as np
import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from utils.plate_detector import PlateDetection
from utils.scene_analyzer import scene_analyzer, SceneAnalysis
from config import DetectionConfig

class ValidationStage(Enum):
    """Validation stages"""
    GEOMETRIC = "geometric"
    EDGE_QUALITY = "edge_quality"
    COLOR_VALIDATION = "color_validation"
    TEXT_PATTERN = "text_pattern"
    FINAL_CONFIDENCE = "final_confidence"

@dataclass
class ValidationResult:
    """Result dari validation stage"""
    stage: ValidationStage
    passed: bool
    score: float
    reason: str
    details: Dict = None

@dataclass
class PipelineResult:
    """Final result dari validation pipeline"""
    detection: PlateDetection
    is_valid: bool
    total_score: float
    stage_results: List[ValidationResult]
    confidence_modifier: float
    rejection_reason: Optional[str] = None

class ValidationPipeline:
    """
    Multi-stage validation pipeline untuk eliminate false positives
    """

    def __init__(self):
        """Initialize validation pipeline"""
        self.logger = logging.getLogger(__name__)

        # Indonesian plate patterns (strict)
        self.indonesian_patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',      # B 1234 ABC
            r'^[A-Z]{2}\s*\d{2,4}\s*[A-Z]{2,3}$',        # AB 1234 CD
            r'^[A-Z]\s*\d{1,4}\s*[A-Z]{2,3}$',           # B 123 AB
            r'^[DFGHINRTZ]\s*\d{1,4}\s*[A-Z]{2,3}$',      # Specific Indonesian prefixes
        ]

        # Indonesian plate colors (BGR format) - strict
        self.plate_colors = {
            'white': {'lower': (200, 200, 200), 'upper': (255, 255, 255)},
            'yellow': {'lower': (0, 180, 180), 'upper': (60, 255, 255)},
            'red': {'lower': (0, 0, 180), 'upper': (60, 60, 255)},
            'black': {'lower': (0, 0, 0), 'upper': (60, 60, 60)}
        }

        # Validation thresholds
        self.thresholds = {
            'min_geometry_score': 0.6,
            'min_edge_quality': 0.5,
            'min_color_confidence': 0.4,
            'min_text_score': 0.6,
            'min_final_confidence': 0.7
        }

        self.logger.info("ValidationPipeline initialized with strict criteria")

    def validate_detection(self, detection: PlateDetection,
                          frame: np.ndarray,
                          scene_analysis: SceneAnalysis = None) -> PipelineResult:
        """
        Run complete validation pipeline

        Args:
            detection: PlateDetection object
            frame: Original frame
            scene_analysis: Optional scene analysis

        Returns:
            PipelineResult dengan detailed validation results
        """
        try:
            stage_results = []
            total_score = 0.0
            weights = {
                ValidationStage.GEOMETRIC: 0.25,
                ValidationStage.EDGE_QUALITY: 0.20,
                ValidationStage.COLOR_VALIDATION: 0.20,
                ValidationStage.TEXT_PATTERN: 0.25,
                ValidationStage.FINAL_CONFIDENCE: 0.10
            }

            # Get scene analysis if not provided
            if scene_analysis is None:
                scene_analysis = scene_analyzer.analyze_frame(frame)

            # Early rejection untuk empty scenes
            if scene_analysis.likely_empty:
                return PipelineResult(
                    detection=detection,
                    is_valid=False,
                    total_score=0.0,
                    stage_results=[],
                    confidence_modifier=0.1,
                    rejection_reason="Empty scene detected"
                )

            # Stage 1: Geometric Validation
            geo_result = self._validate_geometry(detection, frame)
            stage_results.append(geo_result)
            if not geo_result.passed:
                return self._create_rejection_result(detection, stage_results, "Geometry validation failed")

            # Stage 2: Edge Quality Validation
            edge_result = self._validate_edge_quality(detection, frame)
            stage_results.append(edge_result)
            if not edge_result.passed:
                return self._create_rejection_result(detection, stage_results, "Edge quality insufficient")

            # Stage 3: Color Validation
            color_result = self._validate_color(detection, frame)
            stage_results.append(color_result)
            if not color_result.passed:
                return self._create_rejection_result(detection, stage_results, "Color validation failed")

            # Stage 4: Text Pattern Validation
            text_result = self._validate_text_pattern(detection)
            stage_results.append(text_result)
            if not text_result.passed:
                return self._create_rejection_result(detection, stage_results, "Text pattern invalid")

            # Stage 5: Final Confidence Validation
            conf_result = self._validate_final_confidence(detection, scene_analysis)
            stage_results.append(conf_result)

            # Calculate weighted total score
            for result in stage_results:
                weight = weights.get(result.stage, 0.0)
                total_score += result.score * weight

            # Final decision
            is_valid = (
                all(result.passed for result in stage_results) and
                total_score >= 0.7 and
                detection.confidence >= 50.0
            )

            return PipelineResult(
                detection=detection,
                is_valid=is_valid,
                total_score=total_score,
                stage_results=stage_results,
                confidence_modifier=scene_analysis.confidence_modifier,
                rejection_reason=None if is_valid else "Overall validation failed"
            )

        except Exception as e:
            self.logger.error(f"Error in validation pipeline: {str(e)}")
            return PipelineResult(
                detection=detection,
                is_valid=False,
                total_score=0.0,
                stage_results=[],
                confidence_modifier=0.1,
                rejection_reason="Pipeline error"
            )

    def _validate_geometry(self, detection: PlateDetection, frame: np.ndarray) -> ValidationResult:
        """
        Stage 1: Geometric validation (strict criteria)

        Args:
            detection: PlateDetection object
            frame: Original frame

        Returns:
            ValidationResult untuk geometric validation
        """
        try:
            x, y, w, h = detection.bbox
            score = 0.0
            details = {}

            # 1. Aspect ratio validation (strict Indonesian plate ratios)
            aspect_ratio = w / h if h > 0 else 0
            details['aspect_ratio'] = aspect_ratio

            if 2.0 <= aspect_ratio <= 5.0:  # Strict range untuk Indonesian plates
                aspect_score = 1.0 - abs(aspect_ratio - 3.5) / 3.5  # Optimal around 3.5
            else:
                aspect_score = 0.0

            # 2. Size validation relative to frame
            frame_area = frame.shape[0] * frame.shape[1]
            detection_area = w * h
            size_ratio = detection_area / frame_area
            details['size_ratio'] = size_ratio

            # Indonesian plates should be 0.5% to 3% of frame (strict)
            if 0.005 <= size_ratio <= 0.03:
                size_score = 1.0
            elif size_ratio < 0.005:
                size_score = size_ratio / 0.005  # Penalty untuk too small
            else:
                size_score = max(0.0, 1.0 - (size_ratio - 0.03) / 0.03)

            # 3. Position validation (plates usually in lower 70% of frame)
            center_y = y + h // 2
            frame_height = frame.shape[0]
            relative_y = center_y / frame_height
            details['relative_y'] = relative_y

            if relative_y > 0.3:  # Lower part of frame
                position_score = 1.0
            else:
                position_score = relative_y / 0.3

            # 4. Rectangularity check (if processed image available)
            rectangularity_score = 0.8  # Default
            if detection.processed_image is not None:
                rectangularity_score = self._calculate_rectangularity(detection.processed_image)

            details['rectangularity'] = rectangularity_score

            # Calculate weighted geometric score
            score = (
                aspect_score * 0.4 +
                size_score * 0.3 +
                position_score * 0.2 +
                rectangularity_score * 0.1
            )

            passed = (
                score >= self.thresholds['min_geometry_score'] and
                aspect_score > 0.0 and
                size_score > 0.3 and
                rectangularity_score > 0.5
            )

            return ValidationResult(
                stage=ValidationStage.GEOMETRIC,
                passed=passed,
                score=score,
                reason=f"Geometric score: {score:.3f}" + ("" if passed else " - Failed strict criteria"),
                details=details
            )

        except Exception as e:
            self.logger.warning(f"Error in geometry validation: {str(e)}")
            return ValidationResult(
                stage=ValidationStage.GEOMETRIC,
                passed=False,
                score=0.0,
                reason="Geometry validation error"
            )

    def _validate_edge_quality(self, detection: PlateDetection, frame: np.ndarray) -> ValidationResult:
        """
        Stage 2: Edge quality validation

        Args:
            detection: PlateDetection object
            frame: Original frame

        Returns:
            ValidationResult untuk edge quality
        """
        try:
            x, y, w, h = detection.bbox

            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return ValidationResult(
                    stage=ValidationStage.EDGE_QUALITY,
                    passed=False,
                    score=0.0,
                    reason="Invalid ROI"
                )

            # Convert to grayscale
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi.copy()

            details = {}

            # 1. Edge density calculation
            edges = cv2.Canny(gray_roi, 50, 150)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0

            details['edge_density'] = edge_density

            # 2. Edge structure analysis (plates should have rectangular edges)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Approximate to polygon
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Plates should approximate to 4-sided polygon
                structure_score = min(len(approx) / 4.0, 1.0) if len(approx) >= 3 else 0
            else:
                structure_score = 0.0

            details['edge_structure'] = structure_score

            # 3. Edge continuity (plates should have continuous edges)
            # Calculate edge gaps
            edge_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
            continuity_score = min(len(edge_lines) / 4.0, 1.0) if edge_lines is not None else 0

            details['edge_continuity'] = continuity_score

            # Calculate overall edge quality score
            score = (
                edge_density * 0.5 +
                structure_score * 0.3 +
                continuity_score * 0.2
            )

            passed = (
                score >= self.thresholds['min_edge_quality'] and
                edge_density >= DetectionConfig.MIN_EDGE_DENSITY
            )

            return ValidationResult(
                stage=ValidationStage.EDGE_QUALITY,
                passed=passed,
                score=score,
                reason=f"Edge quality: {score:.3f}, density: {edge_density:.3f}",
                details=details
            )

        except Exception as e:
            self.logger.warning(f"Error in edge validation: {str(e)}")
            return ValidationResult(
                stage=ValidationStage.EDGE_QUALITY,
                passed=False,
                score=0.0,
                reason="Edge validation error"
            )

    def _validate_color(self, detection: PlateDetection, frame: np.ndarray) -> ValidationResult:
        """
        Stage 3: Indonesian plate color validation

        Args:
            detection: PlateDetection object
            frame: Original frame

        Returns:
            ValidationResult untuk color validation
        """
        try:
            x, y, w, h = detection.bbox

            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0:
                return ValidationResult(
                    stage=ValidationStage.COLOR_VALIDATION,
                    passed=False,
                    score=0.0,
                    reason="Invalid ROI for color analysis"
                )

            # Convert to HSV untuk better color analysis
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            details = {}
            color_scores = []

            # Check each Indonesian plate color
            for color_name, color_range in self.plate_colors.items():
                # Create mask untuk color range
                if color_name == 'yellow':
                    # Special handling untuk yellow (use HSV)
                    lower_hsv = np.array([15, 100, 100])
                    upper_hsv = np.array([35, 255, 255])
                    mask = cv2.inRange(hsv_roi, lower_hsv, upper_hsv)
                else:
                    # BGR color range
                    lower = np.array(color_range['lower'])
                    upper = np.array(color_range['upper'])
                    mask = cv2.inRange(roi, lower, upper)

                # Calculate color ratio
                color_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                color_ratio = color_pixels / total_pixels if total_pixels > 0 else 0

                details[f'{color_name}_ratio'] = color_ratio

                # Weight different colors (white is most common)
                if color_name == 'white':
                    color_scores.append(color_ratio * 1.0)
                elif color_name == 'yellow':
                    color_scores.append(color_ratio * 0.8)
                elif color_name == 'red':
                    color_scores.append(color_ratio * 0.9)
                else:  # black
                    color_scores.append(color_ratio * 0.7)

            # Best color match
            max_color_score = max(color_scores) if color_scores else 0.0

            # Additional validation: check for text contrast
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            min_val = np.min(gray_roi)
            max_val = np.max(gray_roi)
            contrast_ratio = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0

            details['contrast_ratio'] = contrast_ratio

            # Final color score
            score = max_color_score * (1.0 + contrast_ratio) / 2.0

            passed = (
                score >= self.thresholds['min_color_confidence'] and
                contrast_ratio >= DetectionConfig.MIN_CONTRAST_RATIO
            )

            return ValidationResult(
                stage=ValidationStage.COLOR_VALIDATION,
                passed=passed,
                score=score,
                reason=f"Color score: {score:.3f}, contrast: {contrast_ratio:.3f}",
                details=details
            )

        except Exception as e:
            self.logger.warning(f"Error in color validation: {str(e)}")
            return ValidationResult(
                stage=ValidationStage.COLOR_VALIDATION,
                passed=False,
                score=0.0,
                reason="Color validation error"
            )

    def _validate_text_pattern(self, detection: PlateDetection) -> ValidationResult:
        """
        Stage 4: Indonesian text pattern validation (strict)

        Args:
            detection: PlateDetection object

        Returns:
            ValidationResult untuk text pattern
        """
        try:
            text = detection.text
            if not text:
                return ValidationResult(
                    stage=ValidationStage.TEXT_PATTERN,
                    passed=False,
                    score=0.0,
                    reason="No text detected"
                )

            # Clean and normalize text
            cleaned_text = re.sub(r'[^A-Z0-9\s]', '', text.upper().strip())

            details = {
                'original_text': text,
                'cleaned_text': cleaned_text
            }

            # Length validation (strict Indonesian format)
            if len(cleaned_text) < 6 or len(cleaned_text) > 10:
                return ValidationResult(
                    stage=ValidationStage.TEXT_PATTERN,
                    passed=False,
                    score=0.0,
                    reason=f"Invalid length: {len(cleaned_text)} chars",
                    details=details
                )

            # Pattern matching against Indonesian formats
            pattern_scores = []
            matched_patterns = []

            for i, pattern in enumerate(self.indonesian_patterns):
                if re.match(pattern, cleaned_text):
                    pattern_scores.append(1.0)
                    matched_patterns.append(i)
                else:
                    # Partial matching untuk similarity
                    pattern_chars = len(re.findall(r'[A-Z0-9]', pattern))
                    text_chars = len(re.findall(r'[A-Z0-9]', cleaned_text))
                    if pattern_chars > 0:
                        similarity = min(text_chars / pattern_chars, 1.0) * 0.6
                        pattern_scores.append(similarity)

            if not pattern_scores:
                pattern_score = 0.0
            else:
                pattern_score = max(pattern_scores)

            details['pattern_score'] = pattern_score
            details['matched_patterns'] = matched_patterns

            # Character composition validation
            letters = len(re.findall(r'[A-Z]', cleaned_text))
            numbers = len(re.findall(r'[0-9]', cleaned_text))

            details['letters_count'] = letters
            details['numbers_count'] = numbers

            # Indonesian plates: 1-3 letters + 1-4 numbers + 1-3 letters
            composition_valid = (
                1 <= letters <= 6 and  # Total letters
                1 <= numbers <= 4      # Total numbers
            )

            composition_score = 1.0 if composition_valid else 0.3

            # Indonesian city codes validation (common prefixes)
            indonesian_prefixes = ['B', 'D', 'F', 'G', 'H', 'I', 'N', 'R', 'T', 'Z',
                                 'AA', 'AB', 'AD', 'AE', 'AG', 'BA', 'BB', 'BD', 'BE']

            prefix_match = False
            for prefix in indonesian_prefixes:
                if cleaned_text.startswith(prefix):
                    prefix_match = True
                    break

            prefix_score = 1.0 if prefix_match else 0.5

            details['prefix_match'] = prefix_match

            # Calculate final text score
            score = (
                pattern_score * 0.5 +
                composition_score * 0.3 +
                prefix_score * 0.2
            )

            passed = (
                score >= self.thresholds['min_text_score'] and
                pattern_score >= 0.6 and
                composition_valid
            )

            return ValidationResult(
                stage=ValidationStage.TEXT_PATTERN,
                passed=passed,
                score=score,
                reason=f"Text score: {score:.3f}, pattern: {pattern_score:.3f}",
                details=details
            )

        except Exception as e:
            self.logger.warning(f"Error in text validation: {str(e)}")
            return ValidationResult(
                stage=ValidationStage.TEXT_PATTERN,
                passed=False,
                score=0.0,
                reason="Text validation error"
            )

    def _validate_final_confidence(self, detection: PlateDetection,
                                  scene_analysis: SceneAnalysis) -> ValidationResult:
        """
        Stage 5: Final confidence validation dengan scene context

        Args:
            detection: PlateDetection object
            scene_analysis: Scene analysis results

        Returns:
            ValidationResult untuk final confidence
        """
        try:
            base_confidence = detection.confidence
            scene_modifier = scene_analysis.confidence_modifier

            # Apply scene-based confidence adjustment
            adjusted_confidence = base_confidence * scene_modifier

            details = {
                'base_confidence': base_confidence,
                'scene_modifier': scene_modifier,
                'adjusted_confidence': adjusted_confidence,
                'scene_empty': scene_analysis.likely_empty,
                'has_vehicles': scene_analysis.has_vehicles
            }

            # Final validation criteria
            min_confidence = DetectionConfig.MIN_CONFIDENCE_THRESHOLD * 100  # Convert to percentage

            passed = (
                adjusted_confidence >= min_confidence and
                base_confidence >= 40.0 and  # Minimum base confidence
                scene_modifier >= 0.3        # Minimum scene quality
            )

            score = min(adjusted_confidence / 100.0, 1.0)

            return ValidationResult(
                stage=ValidationStage.FINAL_CONFIDENCE,
                passed=passed,
                score=score,
                reason=f"Confidence: {adjusted_confidence:.1f}% (base: {base_confidence:.1f}%)",
                details=details
            )

        except Exception as e:
            self.logger.warning(f"Error in final confidence validation: {str(e)}")
            return ValidationResult(
                stage=ValidationStage.FINAL_CONFIDENCE,
                passed=False,
                score=0.0,
                reason="Confidence validation error"
            )

    def _calculate_rectangularity(self, image: np.ndarray) -> float:
        """Calculate rectangularity score untuk plate image"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return 0.5  # Default moderate score

            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Calculate rectangularity
            area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            rect_area = w * h

            if rect_area > 0:
                return area / rect_area
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Error calculating rectangularity: {str(e)}")
            return 0.5

    def _create_rejection_result(self, detection: PlateDetection,
                               stage_results: List[ValidationResult],
                               reason: str) -> PipelineResult:
        """Create rejection result dengan stage details"""
        return PipelineResult(
            detection=detection,
            is_valid=False,
            total_score=0.0,
            stage_results=stage_results,
            confidence_modifier=0.1,
            rejection_reason=reason
        )

    def get_validation_summary(self, results: List[PipelineResult]) -> Dict:
        """Get validation summary statistics"""
        if not results:
            return {}

        total_processed = len(results)
        valid_count = sum(1 for r in results if r.is_valid)

        # Stage-wise statistics
        stage_stats = {}
        for stage in ValidationStage:
            stage_passed = 0
            stage_scores = []

            for result in results:
                for stage_result in result.stage_results:
                    if stage_result.stage == stage:
                        if stage_result.passed:
                            stage_passed += 1
                        stage_scores.append(stage_result.score)

            if stage_scores:
                stage_stats[stage.value] = {
                    'pass_rate': stage_passed / len(stage_scores) * 100,
                    'avg_score': np.mean(stage_scores),
                    'count': len(stage_scores)
                }

        return {
            'total_processed': total_processed,
            'valid_detections': valid_count,
            'validation_rate': valid_count / total_processed * 100 if total_processed > 0 else 0,
            'stage_statistics': stage_stats
        }

# Global instance
validation_pipeline = ValidationPipeline()