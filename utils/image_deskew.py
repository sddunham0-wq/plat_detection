"""
Image Deskewing and Preprocessing Module
Handles skew detection, rotation correction, and image enhancement for tilted license plates
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class ImageDeskewer:
    """
    Image preprocessing class for handling tilted/skewed license plates
    Improves OCR accuracy through skew correction and image enhancement
    """

    def __init__(self,
                 max_skew_angle: float = 30.0,
                 enable_perspective_correction: bool = True,
                 enable_enhancement: bool = True):
        """
        Initialize ImageDeskewer

        Args:
            max_skew_angle: Maximum skew angle to detect (degrees)
            enable_perspective_correction: Enable perspective transformation
            enable_enhancement: Enable image enhancement (CLAHE, denoising, sharpening)
        """
        self.max_skew_angle = max_skew_angle
        self.enable_perspective_correction = enable_perspective_correction
        self.enable_enhancement = enable_enhancement

    def detect_skew_angle(self, image: np.ndarray) -> float:
        """
        Detect skew angle using Canny edge detection and Hough Transform

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            float: Detected skew angle in degrees
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

            # Hough Line Transform to detect lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=int(image.shape[1] * 0.3),
                maxLineGap=10
            )

            if lines is None or len(lines) == 0:
                logger.debug("No lines detected for skew angle calculation")
                return 0.0

            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                # Filter out vertical lines and extreme angles
                if abs(angle) < self.max_skew_angle and abs(abs(angle) - 90) > 45:
                    angles.append(angle)

            if not angles:
                logger.debug("No valid angles detected")
                return 0.0

            # Use median angle to reduce outlier impact
            skew_angle = np.median(angles)
            logger.debug(f"Detected skew angle: {skew_angle:.2f}°")

            return skew_angle

        except Exception as e:
            logger.error(f"Error detecting skew angle: {str(e)}")
            return 0.0

    def deskew(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Correct skew angle and straighten image

        Args:
            image: Input image to deskew
            angle: Skew angle in degrees (if None, auto-detect)

        Returns:
            np.ndarray: Deskewed image
        """
        try:
            if angle is None:
                angle = self.detect_skew_angle(image)

            # Skip rotation if angle is negligible
            if abs(angle) < 0.5:
                logger.debug("Skew angle negligible, skipping rotation")
                return image

            # Get image dimensions
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)

            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new bounding dimensions to prevent cropping
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # Adjust rotation matrix for new dimensions
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            # Perform rotation with border padding
            deskewed = cv2.warpAffine(
                image,
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            logger.info(f"Image deskewed by {angle:.2f}°")
            return deskewed

        except Exception as e:
            logger.error(f"Error deskewing image: {str(e)}")
            return image

    def correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation for plates viewed at angles

        Args:
            image: Input image

        Returns:
            np.ndarray: Perspective-corrected image
        """
        try:
            if not self.enable_perspective_correction:
                return image

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return image

            # Find largest rectangular contour
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Only apply perspective correction if we found a quadrilateral
            if len(approx) == 4:
                # Order points: top-left, top-right, bottom-right, bottom-left
                pts = approx.reshape(4, 2)
                rect = self._order_points(pts)

                # Compute width and height of new image
                (tl, tr, br, bl) = rect
                width_a = np.linalg.norm(br - bl)
                width_b = np.linalg.norm(tr - tl)
                max_width = max(int(width_a), int(width_b))

                height_a = np.linalg.norm(tr - br)
                height_b = np.linalg.norm(tl - bl)
                max_height = max(int(height_a), int(height_b))

                # Destination points
                dst = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype=np.float32)

                # Compute perspective transform matrix
                matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)

                # Apply perspective transformation
                warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

                logger.info("Perspective correction applied")
                return warped

            return image

        except Exception as e:
            logger.error(f"Error correcting perspective: {str(e)}")
            return image

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in consistent order: top-left, top-right, bottom-right, bottom-left

        Args:
            pts: Array of 4 points

        Returns:
            np.ndarray: Ordered points
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum, bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference, bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def enhance_plate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement techniques (CLAHE, denoising, sharpening)

        Args:
            image: Input image

        Returns:
            np.ndarray: Enhanced image
        """
        try:
            if not self.enable_enhancement:
                return image

            # Convert to grayscale if needed
            is_color = len(image.shape) == 3
            if is_color:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

            # Sharpen
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Convert back to BGR if input was color
            if is_color:
                enhanced_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            else:
                enhanced_image = sharpened

            logger.debug("Image enhancement applied (CLAHE, denoise, sharpen)")
            return enhanced_image

        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image

    def preprocess(self, image: np.ndarray, multi_angle_attempts: bool = True) -> List[np.ndarray]:
        """
        Full preprocessing pipeline with optional multi-angle attempts

        Args:
            image: Input image
            multi_angle_attempts: Generate multiple rotated versions for OCR

        Returns:
            List[np.ndarray]: List of preprocessed images (primary + rotated variants)
        """
        try:
            results = []

            # Step 1: Enhance image
            enhanced = self.enhance_plate_image(image)

            # Step 2: Detect and correct skew
            deskewed = self.deskew(enhanced)

            # Step 3: Perspective correction (optional)
            if self.enable_perspective_correction:
                corrected = self.correct_perspective(deskewed)
            else:
                corrected = deskewed

            # Primary preprocessed image
            results.append(corrected)

            # Step 4: Multi-angle attempts (optional)
            if multi_angle_attempts:
                angles = [-5, 5, -10, 10]  # Try small rotations
                for angle in angles:
                    rotated = self.deskew(enhanced, angle=angle)
                    results.append(rotated)

            logger.info(f"Preprocessing complete: {len(results)} image variants generated")
            return results

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            return [image]  # Return original if preprocessing fails


def preprocess_plate_for_ocr(image: np.ndarray,
                             enable_multi_angle: bool = True,
                             max_variants: int = 5) -> List[np.ndarray]:
    """
    Convenience function for preprocessing license plate images for OCR

    Args:
        image: Input plate image
        enable_multi_angle: Generate multiple rotation variants
        max_variants: Maximum number of variants to return

    Returns:
        List[np.ndarray]: List of preprocessed image variants
    """
    deskewer = ImageDeskewer(
        max_skew_angle=30.0,
        enable_perspective_correction=True,
        enable_enhancement=True
    )

    variants = deskewer.preprocess(image, multi_angle_attempts=enable_multi_angle)
    return variants[:max_variants]
