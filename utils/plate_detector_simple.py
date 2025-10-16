# -*- coding: utf-8 -*-
"""
PLATE DETECTOR SEDERHANA
Deteksi area plat nomor dengan filter basic
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SimplePlateDetector:
    """Detector sederhana untuk deteksi plat nomor"""

    def __init__(self):
        # Filter ukuran plat (pixel)
        # RELAXED untuk support motor plates dan far distance
        self.MIN_WIDTH = 50        # Turun dari 70 - motor plate dari jauh ~60px
        self.MAX_WIDTH = 600
        self.MIN_HEIGHT = 15       # Turun dari 20 - motor plate lebih kecil
        self.MAX_HEIGHT = 150

        # Filter aspect ratio (lebar/tinggi)
        # RELAXED untuk motor plates (ratio ~2.5)
        self.MIN_RATIO = 2.3       # Turun dari 2.8 - motor F 1818 HG ratio ~2.5
        self.MAX_RATIO = 4.2

        # Filter warna putih (brightness)
        # VERY RELAXED untuk low light and shadow conditions
        self.MIN_BRIGHTNESS = 60   # Turun dari 140 → 120 → 60 (very relaxed)

        logger.info("Simple Plate Detector initialized (VERY RELAXED filters)")
        logger.info(f"  Width: {self.MIN_WIDTH}-{self.MAX_WIDTH}px")
        logger.info(f"  Height: {self.MIN_HEIGHT}-{self.MAX_HEIGHT}px")
        logger.info(f"  Ratio: {self.MIN_RATIO}-{self.MAX_RATIO}")
        logger.info(f"  Brightness: >={self.MIN_BRIGHTNESS} (VERY relaxed for shadows)")

    def detect(self, frame):
        """
        Deteksi plat di frame

        Returns:
            List of (x, y, w, h) bounding boxes
        """
        try:
            # 1. Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharp = cv2.filter2D(gray, -1, kernel)

            # 3. CLAHE (kontras)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharp)

            # 4. Bilateral filter (smooth tapi jaga edges)
            smooth = cv2.bilateralFilter(enhanced, 11, 17, 17)

            # 5. Edge detection
            edges = cv2.Canny(smooth, 20, 200)

            # 6. Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 7. Filter contours
            plates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # Filter ukuran
                if w < self.MIN_WIDTH or w > self.MAX_WIDTH:
                    continue
                if h < self.MIN_HEIGHT or h > self.MAX_HEIGHT:
                    continue

                # Filter aspect ratio
                ratio = w / h
                if ratio < self.MIN_RATIO or ratio > self.MAX_RATIO:
                    continue

                # Filter warna putih
                roi = gray[y:y+h, x:x+w]
                brightness = np.mean(roi)
                if brightness < self.MIN_BRIGHTNESS:
                    continue

                plates.append((x, y, w, h))

            # Sort by area (terbesar dulu)
            plates.sort(key=lambda box: box[2] * box[3], reverse=True)

            # Return top 3
            return plates[:3]

        except Exception as e:
            logger.error(f"Error detecting plate: {e}")
            return []

    def draw(self, frame, boxes, vehicle_type="KENDARAAN"):
        """
        Gambar bounding box HIJAU di frame dengan label jenis kendaraan

        Args:
            frame: Input frame
            boxes: List of (x,y,w,h) bounding boxes
            vehicle_type: Jenis kendaraan (MOBIL/MOTOR)
        """
        if not boxes:
            return frame

        GREEN = (0, 255, 0)  # Warna hijau konsisten untuk semua box

        for i, (x, y, w, h) in enumerate(boxes):
            # Gambar rectangle hijau
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

            # Label jenis kendaraan di atas box
            label = f"{vehicle_type} #{i+1}" if i > 0 else vehicle_type
            cv2.putText(frame, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        return frame
