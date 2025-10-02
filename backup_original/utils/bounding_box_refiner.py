"""
Bounding Box Refiner
Enhanced precision untuk plate detection bounding boxes
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class RefinedBoundingBox:
    """Refined bounding box dengan additional metrics"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    edge_quality: float      # Quality of detected edges
    contour_area: float      # Area of refined contour
    aspect_ratio: float      # Width/height ratio
    rectangularity: float    # How rectangular the shape is

class BoundingBoxRefiner:
    """
    Enhanced bounding box refinement untuk plate detection
    """

    def __init__(self):
        """Initialize bounding box refiner"""
        self.logger = logging.getLogger(__name__)

        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.gaussian_kernel = 5

        # Contour filtering parameters
        self.min_contour_area = 500
        self.max_contour_area = 50000
        self.min_aspect_ratio = 1.5
        self.max_aspect_ratio = 6.0

        # Morphological operation kernels
        self.closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        self.logger.info("BoundingBoxRefiner initialized")

    def refine_bounding_box(self, image: np.ndarray,
                           original_bbox: Tuple[int, int, int, int],
                           confidence: float = 1.0) -> Optional[RefinedBoundingBox]:
        """
        Refine bounding box dengan edge detection dan contour analysis

        Args:
            image: Input image
            original_bbox: Original bounding box (x, y, w, h)
            confidence: Original detection confidence

        Returns:
            RefinedBoundingBox atau None jika refinement gagal
        """
        try:
            x, y, w, h = original_bbox

            # Expand ROI untuk better edge detection
            expansion = 20
            x_expanded = max(0, x - expansion)
            y_expanded = max(0, y - expansion)
            w_expanded = min(image.shape[1] - x_expanded, w + 2 * expansion)
            h_expanded = min(image.shape[0] - y_expanded, h + 2 * expansion)

            # Extract ROI
            roi = image[y_expanded:y_expanded + h_expanded,
                       x_expanded:x_expanded + w_expanded]

            if roi.size == 0:
                return None

            # Multi-stage refinement
            refined_contour = self._find_best_plate_contour(roi)

            if refined_contour is None:
                # Fallback ke original bbox dengan quality assessment
                return self._create_fallback_bbox(original_bbox, confidence, roi)

            # Get refined bounding box
            refined_x, refined_y, refined_w, refined_h = cv2.boundingRect(refined_contour)

            # Adjust coordinates ke original image space
            final_x = x_expanded + refined_x
            final_y = y_expanded + refined_y

            # Calculate quality metrics
            edge_quality = self._calculate_edge_quality(roi, refined_contour)
            contour_area = cv2.contourArea(refined_contour)
            aspect_ratio = refined_w / refined_h if refined_h > 0 else 0
            rectangularity = self._calculate_rectangularity(refined_contour)

            # Validate refined bbox
            if not self._validate_refined_bbox(refined_w, refined_h, aspect_ratio, contour_area):
                return self._create_fallback_bbox(original_bbox, confidence, roi)

            # Adjust confidence based on quality
            quality_factor = (edge_quality + rectangularity) / 2
            adjusted_confidence = confidence * (0.7 + 0.3 * quality_factor)

            return RefinedBoundingBox(
                x=final_x,
                y=final_y,
                width=refined_w,
                height=refined_h,
                confidence=adjusted_confidence,
                edge_quality=edge_quality,
                contour_area=contour_area,
                aspect_ratio=aspect_ratio,
                rectangularity=rectangularity
            )

        except Exception as e:
            self.logger.warning(f"Bounding box refinement failed: {str(e)}")
            return self._create_fallback_bbox(original_bbox, confidence)

    def _find_best_plate_contour(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Find best plate contour dalam ROI

        Args:
            roi: Region of interest

        Returns:
            Best contour atau None
        """
        try:
            # Convert ke grayscale
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()

            # Multi-scale edge detection
            edges_list = []

            # Scale 1: Original size
            edges1 = self._enhanced_edge_detection(gray)
            edges_list.append(edges1)

            # Scale 2: Slightly blurred for noise reduction
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges2 = self._enhanced_edge_detection(blurred)
            edges_list.append(edges2)

            # Combine edges
            combined_edges = np.zeros_like(gray)
            for edges in edges_list:
                combined_edges = cv2.bitwise_or(combined_edges, edges)

            # Find contours
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Filter dan score contours
            candidate_contours = []

            for contour in contours:
                score = self._score_plate_contour(contour, gray.shape)
                if score > 0.3:  # Minimum score threshold
                    candidate_contours.append((contour, score))

            if not candidate_contours:
                return None

            # Return best contour
            best_contour = max(candidate_contours, key=lambda x: x[1])[0]
            return best_contour

        except Exception as e:
            self.logger.warning(f"Error finding plate contour: {str(e)}")
            return None

    def _enhanced_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhanced edge detection untuk plate boundaries

        Args:
            gray: Grayscale image

        Returns:
            Edge image
        """
        # Gaussian blur untuk noise reduction
        blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel, self.gaussian_kernel), 0)

        # Adaptive Canny thresholds
        v = np.median(blurred)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))

        # Canny edge detection
        edges = cv2.Canny(blurred, lower, upper)

        # Morphological operations untuk connect broken edges
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.closing_kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, self.opening_kernel)

        return edges

    def _score_plate_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """
        Score contour berdasarkan plate characteristics

        Args:
            contour: Input contour
            image_shape: Shape of the image (height, width)

        Returns:
            Score (0.0 - 1.0)
        """
        try:
            # Basic measurements
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)

            if area < self.min_contour_area or area > self.max_contour_area:
                return 0.0

            # Aspect ratio score
            aspect_ratio = w / h if h > 0 else 0
            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                aspect_score = 1.0 - abs(aspect_ratio - 3.0) / 3.0  # Optimal around 3.0
            else:
                aspect_score = 0.0

            # Area ratio score (relative to image)
            image_area = image_shape[0] * image_shape[1]
            area_ratio = area / image_area
            if 0.01 <= area_ratio <= 0.5:  # 1% to 50% of image
                area_score = 1.0
            else:
                area_score = 0.2

            # Rectangularity score
            rectangularity = self._calculate_rectangularity(contour)

            # Solidity score (filled vs convex hull)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Position score (prefer center regions)
            center_x = x + w // 2
            center_y = y + h // 2
            img_center_x = image_shape[1] // 2
            img_center_y = image_shape[0] // 2

            distance_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
            position_score = 1.0 - (distance_from_center / max_distance)

            # Weighted final score
            final_score = (
                aspect_score * 0.25 +
                area_score * 0.20 +
                rectangularity * 0.25 +
                solidity * 0.15 +
                position_score * 0.15
            )

            return max(0.0, min(1.0, final_score))

        except Exception as e:
            self.logger.warning(f"Error scoring contour: {str(e)}")
            return 0.0

    def _calculate_rectangularity(self, contour: np.ndarray) -> float:
        """
        Calculate rectangularity of contour

        Args:
            contour: Input contour

        Returns:
            Rectangularity score (0.0 - 1.0)
        """
        try:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h

            if rect_area > 0:
                return area / rect_area
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_edge_quality(self, roi: np.ndarray, contour: np.ndarray) -> float:
        """
        Calculate edge quality for the contour

        Args:
            roi: Region of interest
            contour: Detected contour

        Returns:
            Edge quality score (0.0 - 1.0)
        """
        try:
            # Create mask dari contour
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, 2)

            # Calculate edge density
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            mask_pixels = np.sum(mask > 0)

            if mask_pixels > 0:
                edge_density = edge_pixels / mask_pixels
                return min(1.0, edge_density * 2)  # Scale factor
            else:
                return 0.0

        except Exception:
            return 0.5  # Default moderate score

    def _validate_refined_bbox(self, width: int, height: int,
                              aspect_ratio: float, area: float) -> bool:
        """
        Validate refined bounding box

        Args:
            width: Bounding box width
            height: Bounding box height
            aspect_ratio: Width/height ratio
            area: Contour area

        Returns:
            True if valid
        """
        # Size validation
        if width < 50 or height < 15:  # Too small
            return False

        if width > 800 or height > 200:  # Too large
            return False

        # Aspect ratio validation
        if aspect_ratio < 1.2 or aspect_ratio > 8.0:
            return False

        # Area validation
        if area < 300 or area > 100000:
            return False

        return True

    def _create_fallback_bbox(self, original_bbox: Tuple[int, int, int, int],
                             confidence: float, roi: np.ndarray = None) -> RefinedBoundingBox:
        """
        Create fallback refined bbox dari original

        Args:
            original_bbox: Original bounding box
            confidence: Original confidence
            roi: Region of interest (optional)

        Returns:
            RefinedBoundingBox with fallback values
        """
        x, y, w, h = original_bbox
        aspect_ratio = w / h if h > 0 else 0

        # Basic quality assessment
        edge_quality = 0.5  # Default moderate
        rectangularity = 0.7  # Assume reasonably rectangular

        if roi is not None:
            try:
                # Quick quality check
                if len(roi.shape) == 3:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray = roi

                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / roi.size
                edge_quality = min(1.0, edge_density * 5)

            except Exception:
                pass

        return RefinedBoundingBox(
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=confidence * 0.8,  # Slight penalty untuk fallback
            edge_quality=edge_quality,
            contour_area=w * h,  # Approximate
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity
        )

    def refine_multiple_bboxes(self, image: np.ndarray,
                              bboxes: List[Tuple[int, int, int, int, float]]) -> List[RefinedBoundingBox]:
        """
        Refine multiple bounding boxes

        Args:
            image: Input image
            bboxes: List of (x, y, w, h, confidence) tuples

        Returns:
            List of RefinedBoundingBox objects
        """
        refined_boxes = []

        for bbox_data in bboxes:
            if len(bbox_data) == 5:
                x, y, w, h, conf = bbox_data
                bbox = (x, y, w, h)
            else:
                x, y, w, h = bbox_data
                conf = 1.0
                bbox = (x, y, w, h)

            refined = self.refine_bounding_box(image, bbox, conf)
            if refined is not None:
                refined_boxes.append(refined)

        return refined_boxes

# Global instance
bounding_box_refiner = BoundingBoxRefiner()