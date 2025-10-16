# -*- coding: utf-8 -*-
"""
RECTANGLE PLATE DETECTOR
Spesial untuk plat Indonesia format persegi panjang (landscape)
Optimized untuk plat mobil & motor standar Indonesia
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RectanglePlateDetector:
    """
    Detector untuk plat nomor Indonesia format PERSEGI PANJANG

    Plat Indonesia characteristics:
    - Format: Landscape (lebih lebar dari tinggi)
    - Ratio: 2.5:1 hingga 4:1 (mobil & motor)
    - Warna: Putih dengan text hitam
    - Size: Bervariasi tergantung jarak
    """

    def __init__(self):
        # ==========================================
        # RECTANGLE PLATE SPECIFIC FILTERS
        # ==========================================

        # Ukuran minimum/maximum (pixels)
        self.MIN_WIDTH = 40        # Plat dari jauh
        self.MAX_WIDTH = 600       # Plat dari dekat
        self.MIN_HEIGHT = 12       # Motor plate minimal
        self.MAX_HEIGHT = 150      # Mobil plate maksimal

        # Aspect Ratio untuk RECTANGLE PLATES
        # Indonesia plates: landscape orientation
        self.MIN_RATIO = 2.2       # Minimum landscape (motor)
        self.MAX_RATIO = 5.0       # Maximum landscape (panjang)

        # Brightness (support shadow & low light)
        self.MIN_BRIGHTNESS = 50   # Very relaxed untuk shadow

        # Area minimum (luas minimum dalam pixels²)
        self.MIN_AREA = 500        # Skip noise kecil

        # ==========================================
        # ADDITIONAL FILTERS untuk RECTANGLE
        # ==========================================

        # Rectangle quality filters
        self.MIN_SOLIDITY = 0.3    # Solidity = area / convex_hull_area
        self.MIN_EXTENT = 0.3      # Extent = contour_area / bounding_rect_area

        logger.info("Rectangle Plate Detector initialized")
        logger.info(f"  Optimized for: LANDSCAPE Indonesian plates")
        logger.info(f"  Width: {self.MIN_WIDTH}-{self.MAX_WIDTH}px")
        logger.info(f"  Height: {self.MIN_HEIGHT}-{self.MAX_HEIGHT}px")
        logger.info(f"  Ratio: {self.MIN_RATIO}-{self.MAX_RATIO} (landscape)")
        logger.info(f"  Brightness: >={self.MIN_BRIGHTNESS}")
        logger.info(f"  Min Area: {self.MIN_AREA}px²")

    def detect(self, frame):
        """
        Deteksi plat PERSEGI PANJANG di frame

        Args:
            frame: Input image (BGR)

        Returns:
            List of (x, y, w, h) bounding boxes
        """
        try:
            # 1. Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Preprocessing pipeline
            # Sharpen untuk enhance edges
            kernel_sharpen = np.array([[-1,-1,-1],
                                       [-1, 9,-1],
                                       [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

            # CLAHE untuk kontras enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharpened)

            # Bilateral filter (smooth + preserve edges)
            smooth = cv2.bilateralFilter(enhanced, 11, 17, 17)

            # 3. Edge detection
            edges = cv2.Canny(smooth, 30, 200)

            # Morphological operations untuk connect edges
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_morph)

            # 4. Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 5. Filter untuk RECTANGLE PLATES
            candidates = []

            for cnt in contours:
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(cnt)

                # === FILTER 1: Size ===
                if w < self.MIN_WIDTH or w > self.MAX_WIDTH:
                    continue
                if h < self.MIN_HEIGHT or h > self.MAX_HEIGHT:
                    continue

                # === FILTER 2: Area ===
                area = cv2.contourArea(cnt)
                if area < self.MIN_AREA:
                    continue

                # === FILTER 3: Aspect Ratio (RECTANGLE) ===
                ratio = w / h if h > 0 else 0
                if ratio < self.MIN_RATIO or ratio > self.MAX_RATIO:
                    continue

                # === FILTER 4: Brightness ===
                roi = gray[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                brightness = np.mean(roi)
                if brightness < self.MIN_BRIGHTNESS:
                    continue

                # === FILTER 5: Rectangle Quality ===
                # Solidity = contour_area / convex_hull_area
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                if solidity < self.MIN_SOLIDITY:
                    continue

                # Extent = contour_area / bounding_rect_area
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0

                if extent < self.MIN_EXTENT:
                    continue

                # === QUALITY SCORE ===
                # Score tinggi = kandidat bagus
                quality_score = self._calculate_quality_score(
                    ratio, brightness, solidity, extent, area
                )

                candidates.append({
                    'bbox': (x, y, w, h),
                    'ratio': ratio,
                    'brightness': brightness,
                    'area': area,
                    'solidity': solidity,
                    'extent': extent,
                    'quality': quality_score
                })

                logger.debug(f"Candidate: {w}x{h} ratio={ratio:.2f} bright={brightness:.0f} quality={quality_score:.2f}")

            # 6. Sort by quality score (best first)
            candidates.sort(key=lambda c: c['quality'], reverse=True)

            # 7. Extract top 3 bounding boxes
            plates = [c['bbox'] for c in candidates[:3]]

            logger.info(f"Rectangle detection: {len(plates)} plate(s) found")

            return plates

        except Exception as e:
            logger.error(f"Rectangle detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_quality_score(self, ratio, brightness, solidity, extent, area):
        """
        Calculate quality score untuk ranking candidates

        Higher score = better candidate
        """
        score = 0.0

        # 1. Ratio score (prefer 2.5-3.5 range = typical plate)
        if 2.5 <= ratio <= 3.5:
            score += 3.0  # Perfect ratio
        elif 2.2 <= ratio <= 4.0:
            score += 2.0  # Good ratio
        else:
            score += 1.0  # Acceptable

        # 2. Brightness score (prefer white plates)
        if brightness >= 100:
            score += 2.0  # Bright white
        elif brightness >= 70:
            score += 1.5  # Medium
        else:
            score += 1.0  # Dark (shadow)

        # 3. Solidity score (prefer solid rectangles)
        if solidity >= 0.7:
            score += 2.0  # Very solid
        elif solidity >= 0.5:
            score += 1.5  # Good
        else:
            score += 1.0  # Acceptable

        # 4. Extent score (prefer filled rectangles)
        if extent >= 0.7:
            score += 2.0  # Well filled
        elif extent >= 0.5:
            score += 1.5  # Good
        else:
            score += 1.0  # Acceptable

        # 5. Area bonus (prefer larger = closer plates)
        if area >= 3000:
            score += 1.0  # Large plate
        elif area >= 1500:
            score += 0.5  # Medium

        return score

    def draw(self, frame, boxes, vehicle_type="KENDARAAN"):
        """
        Draw bounding boxes dengan label

        Args:
            frame: Input frame
            boxes: List of (x,y,w,h) bounding boxes
            vehicle_type: Label untuk jenis kendaraan

        Returns:
            Annotated frame dengan bounding boxes
        """
        if not boxes:
            return frame

        # Warna hijau untuk semua boxes
        GREEN = (0, 255, 0)

        for i, (x, y, w, h) in enumerate(boxes):
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

            # Label
            ratio = w / h if h > 0 else 0
            if i == 0:
                label = f"{vehicle_type} ({ratio:.1f}:1)"
            else:
                label = f"PLATE #{i+1} ({ratio:.1f}:1)"

            # Draw label di atas box
            cv2.putText(frame, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        return frame

    def get_info(self):
        """Get detector configuration info"""
        return {
            'type': 'Rectangle Plate Detector',
            'orientation': 'Landscape',
            'min_ratio': self.MIN_RATIO,
            'max_ratio': self.MAX_RATIO,
            'min_width': self.MIN_WIDTH,
            'max_width': self.MAX_WIDTH,
            'min_brightness': self.MIN_BRIGHTNESS,
            'optimized_for': 'Indonesian license plates (rectangle/landscape)'
        }
