"""
Scene Analyzer untuk False Positive Prevention
Analyze frame untuk detect empty scenes dan prevent false plate detections
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from config import DetectionConfig

@dataclass
class SceneAnalysis:
    """Scene analysis results"""
    has_vehicles: bool = False          # Apakah ada kendaraan di scene
    background_uniformity: float = 0.0  # Uniformitas background (0-1)
    edge_density: float = 0.0           # Density dari edges dalam frame
    contrast_ratio: float = 0.0         # Contrast ratio overall
    motion_detected: bool = False       # Apakah ada motion
    likely_empty: bool = True           # Apakah scene kemungkinan kosong
    confidence_modifier: float = 1.0    # Modifier untuk detection confidence

class SceneAnalyzer:
    """
    Analyze scene untuk prevent false positive detections
    """

    def __init__(self):
        """Initialize scene analyzer"""
        self.logger = logging.getLogger(__name__)

        # Background subtractor untuk motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )

        # Frame history untuk temporal analysis
        self.frame_history = []
        self.max_history = 5

        # Scene statistics
        self.background_model = None
        self.stable_frame_count = 0

        self.logger.info("SceneAnalyzer initialized")

    def analyze_frame(self, frame: np.ndarray) -> SceneAnalysis:
        """
        Comprehensive scene analysis untuk detect empty scenes

        Args:
            frame: Input frame

        Returns:
            SceneAnalysis object dengan detailed metrics
        """
        try:
            analysis = SceneAnalysis()

            # Convert ke grayscale untuk analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # 1. Background uniformity analysis
            analysis.background_uniformity = self._calculate_background_uniformity(gray)

            # 2. Edge density analysis
            analysis.edge_density = self._calculate_edge_density(gray)

            # 3. Contrast analysis
            analysis.contrast_ratio = self._calculate_contrast_ratio(gray)

            # 4. Motion detection
            analysis.motion_detected = self._detect_motion(frame)

            # 5. Vehicle detection (basic shape analysis)
            analysis.has_vehicles = self._detect_vehicle_shapes(gray)

            # 6. Determine if scene is likely empty
            analysis.likely_empty = self._is_scene_empty(analysis)

            # 7. Calculate confidence modifier
            analysis.confidence_modifier = self._calculate_confidence_modifier(analysis)

            # Update frame history
            self._update_frame_history(gray)

            self.logger.debug(f"Scene analysis: empty={analysis.likely_empty}, "
                            f"uniformity={analysis.background_uniformity:.3f}, "
                            f"edges={analysis.edge_density:.3f}")

            return analysis

        except Exception as e:
            self.logger.error(f"Error in scene analysis: {str(e)}")
            # Return conservative analysis
            return SceneAnalysis(likely_empty=False, confidence_modifier=1.0)

    def _calculate_background_uniformity(self, gray: np.ndarray) -> float:
        """
        Calculate background uniformity (higher = more uniform/empty)

        Args:
            gray: Grayscale image

        Returns:
            Uniformity score (0-1)
        """
        try:
            # Calculate standard deviation of pixel intensities
            std_dev = np.std(gray)

            # Calculate histogram uniformity
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_normalized = hist / (gray.shape[0] * gray.shape[1])

            # Calculate entropy (lower entropy = more uniform)
            hist_nonzero = hist_normalized[hist_normalized > 0]
            entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

            # Combine metrics (normalize to 0-1)
            uniformity = 1.0 - min(std_dev / 128.0, 1.0)  # Normalize std_dev
            entropy_norm = 1.0 - min(entropy / 8.0, 1.0)  # Normalize entropy

            return (uniformity + entropy_norm) / 2.0

        except Exception as e:
            self.logger.warning(f"Error calculating uniformity: {str(e)}")
            return 0.5

    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """
        Calculate edge density dalam frame

        Args:
            gray: Grayscale image

        Returns:
            Edge density (0-1)
        """
        try:
            # Gaussian blur untuk noise reduction
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Calculate edge density
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]

            return edge_pixels / total_pixels

        except Exception as e:
            self.logger.warning(f"Error calculating edge density: {str(e)}")
            return 0.0

    def _calculate_contrast_ratio(self, gray: np.ndarray) -> float:
        """
        Calculate contrast ratio dalam frame

        Args:
            gray: Grayscale image

        Returns:
            Contrast ratio
        """
        try:
            # Calculate min and max intensities
            min_val = np.min(gray)
            max_val = np.max(gray)

            if min_val == max_val:
                return 0.0

            # Calculate contrast ratio
            contrast_ratio = (max_val - min_val) / (max_val + min_val)

            return contrast_ratio

        except Exception as e:
            self.logger.warning(f"Error calculating contrast: {str(e)}")
            return 0.0

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion dalam frame

        Args:
            frame: Input frame

        Returns:
            True if motion detected
        """
        try:
            # Apply background subtraction
            fg_mask = self.bg_subtractor.apply(frame)

            # Calculate motion area
            motion_pixels = np.sum(fg_mask > 0)
            total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
            motion_ratio = motion_pixels / total_pixels

            # Motion threshold
            return motion_ratio > 0.05  # 5% of frame

        except Exception as e:
            self.logger.warning(f"Error detecting motion: {str(e)}")
            return False

    def _detect_vehicle_shapes(self, gray: np.ndarray) -> bool:
        """
        Basic vehicle shape detection

        Args:
            gray: Grayscale image

        Returns:
            True if vehicle-like shapes detected
        """
        try:
            # Find contours
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            vehicle_like_shapes = 0

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by area (vehicles should be reasonably large)
                if area < 5000 or area > 200000:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Vehicle-like aspect ratios (cars, motorcycles)
                if 0.5 <= aspect_ratio <= 4.0:
                    # Check solidity (filled vs convex hull)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    if solidity > 0.3:  # Reasonably solid shape
                        vehicle_like_shapes += 1

            return vehicle_like_shapes > 0

        except Exception as e:
            self.logger.warning(f"Error detecting vehicles: {str(e)}")
            return False

    def _is_scene_empty(self, analysis: SceneAnalysis) -> bool:
        """
        Determine if scene is likely empty berdasarkan metrics

        Args:
            analysis: SceneAnalysis object

        Returns:
            True if scene likely empty
        """
        empty_indicators = 0

        # High background uniformity
        if analysis.background_uniformity > DetectionConfig.MAX_BACKGROUND_UNIFORMITY:
            empty_indicators += 1

        # Low edge density
        if analysis.edge_density < DetectionConfig.MIN_EDGE_DENSITY:
            empty_indicators += 1

        # Low contrast
        if analysis.contrast_ratio < DetectionConfig.MIN_CONTRAST_RATIO:
            empty_indicators += 1

        # No vehicles detected
        if not analysis.has_vehicles:
            empty_indicators += 1

        # No motion
        if not analysis.motion_detected:
            empty_indicators += 1

        # Scene is empty if majority indicators suggest so
        return empty_indicators >= 3

    def _calculate_confidence_modifier(self, analysis: SceneAnalysis) -> float:
        """
        Calculate confidence modifier berdasarkan scene analysis

        Args:
            analysis: SceneAnalysis object

        Returns:
            Confidence modifier (0.0-1.0)
        """
        if analysis.likely_empty:
            # Drastically reduce confidence untuk empty scenes
            return 0.1

        modifier = 1.0

        # Reduce confidence untuk poor conditions
        if analysis.background_uniformity > 0.8:
            modifier *= 0.7

        if analysis.edge_density < 0.1:
            modifier *= 0.8

        if analysis.contrast_ratio < 1.0:
            modifier *= 0.9

        if not analysis.has_vehicles:
            modifier *= 0.6

        return max(0.1, modifier)

    def _update_frame_history(self, gray: np.ndarray):
        """Update frame history untuk temporal analysis"""
        try:
            # Add current frame to history
            self.frame_history.append(gray.copy())

            # Maintain max history size
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)

        except Exception as e:
            self.logger.warning(f"Error updating frame history: {str(e)}")

    def get_background_model(self) -> Optional[np.ndarray]:
        """Get current background model"""
        try:
            if len(self.frame_history) >= 3:
                # Create background model from frame history
                frames = np.array(self.frame_history)
                background = np.median(frames, axis=0).astype(np.uint8)
                return background
            return None

        except Exception as e:
            self.logger.warning(f"Error creating background model: {str(e)}")
            return None

    def reset(self):
        """Reset analyzer state"""
        self.frame_history.clear()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50
        )
        self.stable_frame_count = 0
        self.logger.info("SceneAnalyzer reset")

# Global instance
scene_analyzer = SceneAnalyzer()