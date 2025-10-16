# -*- coding: utf-8 -*-
"""
VEHICLE ANALYZER MODULE - Deteksi Warna dan Tipe Kendaraan

Penjelasan SMK: Modul ini seperti "detective" yang bisa
tahu warna dan jenis kendaraan dari gambar plat.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VehicleAnalyzer:
    """
    Class untuk analisis warna dan tipe kendaraan dari cropped plate image
    """

    def __init__(self):
        """
        Initialize vehicle analyzer dengan color definitions
        """
        # Definisi warna dalam HSV color space
        # Penjelasan SMK: HSV lebih akurat untuk deteksi warna
        # dibanding RGB (tidak terpengaruh cahaya)

        self.color_ranges = {
            'Hitam': {
                'lower': np.array([0, 0, 0]),
                'upper': np.array([180, 255, 50])
            },
            'Putih': {
                'lower': np.array([0, 0, 200]),
                'upper': np.array([180, 30, 255])
            },
            'Abu-abu': {
                'lower': np.array([0, 0, 50]),
                'upper': np.array([180, 30, 200])
            },
            'Merah': {
                'lower': np.array([0, 100, 100]),
                'upper': np.array([10, 255, 255])
            },
            'Merah Tua': {
                'lower': np.array([170, 100, 100]),
                'upper': np.array([180, 255, 255])
            },
            'Biru': {
                'lower': np.array([100, 100, 100]),
                'upper': np.array([130, 255, 255])
            },
            'Hijau': {
                'lower': np.array([40, 50, 50]),
                'upper': np.array([80, 255, 255])
            },
            'Kuning': {
                'lower': np.array([20, 100, 100]),
                'upper': np.array([35, 255, 255])
            },
            'Orange': {
                'lower': np.array([10, 100, 100]),
                'upper': np.array([20, 255, 255])
            },
            'Silver': {
                'lower': np.array([0, 0, 180]),
                'upper': np.array([180, 30, 220])
            }
        }

        logger.info("Vehicle Analyzer initialized")

    def detect_vehicle_color(self, plate_roi, frame=None, bbox=None):
        """
        Deteksi warna kendaraan dari area sekitar plat

        Penjelasan SMK: Ambil area di ATAS plat (bagian body kendaraan),
        lalu analisis warna dominan di area tersebut.

        Args:
            plate_roi: Cropped plate image
            frame: Full frame (optional, untuk analisis area sekitar)
            bbox: Bounding box (x, y, w, h) dari plat

        Returns:
            str: Nama warna (e.g., "Hitam", "Putih", "Merah")
        """
        try:
            # Jika ada full frame, ambil area di ATAS plat
            if frame is not None and bbox is not None:
                x, y, w, h = bbox

                # Area analisis: di atas plat (body kendaraan)
                # Penjelasan SMK: Ambil 3x tinggi plat ke atas
                analysis_height = int(h * 3)
                analysis_y = max(0, y - analysis_height)

                # Pastikan tidak keluar dari frame
                analysis_y = max(0, analysis_y)
                analysis_x = max(0, x)
                analysis_w = min(w, frame.shape[1] - x)

                if analysis_y < y:
                    analysis_roi = frame[analysis_y:y, analysis_x:analysis_x+analysis_w]
                else:
                    # Fallback ke plate ROI jika tidak bisa ambil area atas
                    analysis_roi = plate_roi
            else:
                # Fallback: analisis dari plate ROI saja
                analysis_roi = plate_roi

            if analysis_roi.size == 0:
                return "Unknown"

            # Convert ke HSV color space
            hsv = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2HSV)

            # Hitung warna dominan untuk setiap kategori
            color_scores = {}

            for color_name, color_range in self.color_ranges.items():
                # Buat mask untuk warna ini
                mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])

                # Hitung persentase pixel yang match
                pixel_count = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                percentage = (pixel_count / total_pixels) * 100

                color_scores[color_name] = percentage

            # Handle special case untuk merah (2 range)
            if 'Merah' in color_scores and 'Merah Tua' in color_scores:
                color_scores['Merah'] = color_scores['Merah'] + color_scores['Merah Tua']
                del color_scores['Merah Tua']

            # Cari warna dengan skor tertinggi
            if color_scores:
                dominant_color = max(color_scores, key=color_scores.get)
                max_percentage = color_scores[dominant_color]

                # Minimal 15% untuk dianggap valid
                if max_percentage >= 15:
                    logger.debug(f"Detected color: {dominant_color} ({max_percentage:.1f}%)")
                    return dominant_color

            # Fallback: analisis dari brightness
            gray = cv2.cvtColor(analysis_roi, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)

            if avg_brightness < 50:
                return "Hitam"
            elif avg_brightness > 200:
                return "Putih"
            elif avg_brightness > 150:
                return "Silver"
            else:
                return "Abu-abu"

        except Exception as e:
            logger.error(f"Error detecting color: {e}")
            return "Unknown"

    def detect_vehicle_type(self, plate_roi, bbox=None):
        """
        Deteksi tipe kendaraan (motor/mobil) dari ukuran plat

        Penjelasan SMK: Plat motor biasanya LEBIH KECIL dari plat mobil.
        Kita gunakan ukuran untuk estimasi.

        Args:
            plate_roi: Cropped plate image
            bbox: Bounding box (x, y, w, h) dari plat

        Returns:
            str: "Motor" atau "Mobil"
        """
        try:
            if bbox is not None:
                x, y, w, h = bbox
            else:
                h, w = plate_roi.shape[:2]

            # Hitung area plat
            area = w * h

            # Hitung aspect ratio
            aspect_ratio = w / h if h > 0 else 0

            # Heuristic untuk Indonesia:
            # - Plat motor: lebih kecil, aspect ratio ~3.0-3.5
            # - Plat mobil: lebih besar, aspect ratio ~3.5-4.5

            # Threshold berdasarkan area
            # Penjelasan SMK:
            # - Area < 8000 px² → kemungkinan motor
            # - Area >= 8000 px² → kemungkinan mobil

            if area < 8000:
                vehicle_type = "Motor"
            elif area >= 15000:
                vehicle_type = "Mobil"
            else:
                # Ambiguous size, gunakan aspect ratio
                if aspect_ratio < 3.3:
                    vehicle_type = "Motor"
                else:
                    vehicle_type = "Mobil"

            logger.debug(f"Detected vehicle type: {vehicle_type} (area={area}, ratio={aspect_ratio:.2f})")
            return vehicle_type

        except Exception as e:
            logger.error(f"Error detecting vehicle type: {e}")
            return "Unknown"

    def analyze_vehicle(self, plate_roi, frame=None, bbox=None):
        """
        Analisis lengkap: warna + tipe kendaraan

        Args:
            plate_roi: Cropped plate image
            frame: Full frame (optional)
            bbox: Bounding box (x, y, w, h)

        Returns:
            dict: {'color': str, 'type': str}
        """
        try:
            color = self.detect_vehicle_color(plate_roi, frame, bbox)
            vehicle_type = self.detect_vehicle_type(plate_roi, bbox)

            return {
                'color': color,
                'type': vehicle_type
            }

        except Exception as e:
            logger.error(f"Error analyzing vehicle: {e}")
            return {
                'color': 'Unknown',
                'type': 'Unknown'
            }

# Global instance
vehicle_analyzer = VehicleAnalyzer()
