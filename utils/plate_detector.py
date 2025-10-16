# -*- coding: utf-8 -*-
"""
PLATE DETECTOR MODULE - Deteksi Area Plat Nomor

Penjelasan SMK: Modul ini seperti "mata robot" yang bisa
cari dan tandai area plat nomor di gambar/video.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PlateDetector:
    """
    Class untuk deteksi area plat nomor secara otomatis
    """

    def __init__(self, method='contour', max_detections=3):
        """
        Args:
            method: Metode deteksi ('contour' atau 'yolo')
            max_detections: Jumlah maksimal plat yang dideteksi (default 3, lebih sedikit tapi akurat)
        """
        self.method = method
        self.max_detections = max_detections  # Fokus pada plat terbaik saja

        # Parameter filter plat INDONESIA (persegi panjang putih) - LEBIH KETAT!
        self.MIN_PLATE_WIDTH = 70   # Naik dari 50 (filter plat terlalu kecil)
        self.MAX_PLATE_WIDTH = 600  # Support berbagai jarak
        self.MIN_PLATE_HEIGHT = 20  # Naik dari 15 (filter noise)
        self.MAX_PLATE_HEIGHT = 150 # Support berbagai jarak

        # Aspect ratio plat Indonesia LEBIH KETAT (2.8:1 sampai 4.2:1)
        self.MIN_ASPECT_RATIO = 2.8  # Naik dari 2.5 (lebih strict)
        self.MAX_ASPECT_RATIO = 4.2  # Turun dari 4.5 (lebih strict)

        # Parameter untuk deteksi warna putih/terang - LEBIH KETAT!
        self.MIN_WHITE_BRIGHTNESS = 140  # Naik dari 115 (harus benar-benar putih)
        self.MAX_WHITE_BRIGHTNESS = 255  # Batas atas (putih sempurna)

        # Text density untuk plat Indonesia (text hitam di background putih)
        self.MIN_TEXT_DENSITY = 0.10     # Naik dari 0.08 (harus ada text jelas)
        self.MAX_TEXT_DENSITY = 0.40     # Batas atas (tidak terlalu gelap)

        logger.info(f"Plate Detector initialized with method: {method}, max_detections: {max_detections} (OPTIMIZED)")

    def detect_plate_region(self, frame):
        """
        Cari area plat di gambar
        Return: (x, y, w, h) atau None
        """
        if self.method == 'contour':
            return self._detect_by_contour(frame)
        return None

    def _is_white_region(self, gray_roi):
        """
        Penjelasan SMK: Cek apakah area ini PUTIH/TERANG (seperti plat Indonesia)

        Cara kerja:
        - Hitung rata-rata brightness (0-255)
        - Kalau 140-255 artinya putih (plat Indonesia)
        - Kalau <140 artinya gelap (bukan plat)

        Filter LEBIH KETAT: Hanya terima yang benar-benar putih!
        """
        avg_brightness = np.mean(gray_roi)

        # Cek apakah dalam range putih yang valid
        is_white = (avg_brightness >= self.MIN_WHITE_BRIGHTNESS and
                    avg_brightness <= self.MAX_WHITE_BRIGHTNESS)

        # Optional: Cek standard deviation (plat putih punya variance rendah di background)
        std_dev = np.std(gray_roi)
        has_uniform_background = std_dev < 60  # Background putih relatif uniform

        return is_white and has_uniform_background

    def _has_text_characters(self, gray_roi):
        """
        Penjelasan SMK: Cek apakah ada HURUF/ANGKA di dalam area

        Cara kerja:
        - Cari area GELAP (huruf hitam) di background PUTIH
        - Hitung berapa % area yang gelap
        - Kalau 10-40% artinya ada text (plat valid)
        - Kalau <10% → terlalu kosong (bukan plat)
        - Kalau >40% → terlalu gelap (bukan plat putih)
        """
        # Threshold untuk cari text (huruf hitam di background putih)
        _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Hitung % area yang berisi text
        text_pixels = np.sum(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]
        text_density = text_pixels / total_pixels

        # Filter: text density harus dalam range yang wajar
        has_enough_text = text_density >= self.MIN_TEXT_DENSITY
        not_too_dark = text_density <= self.MAX_TEXT_DENSITY

        return has_enough_text and not_too_dark

    def _detect_by_contour(self, frame):
        """
        Deteksi plat dengan preprocessing ULTRA TAJAM untuk zoom out

        Penjelasan SMK: Preprocessing ekstra kuat biar plat kecil/jauh
        tetap terdeteksi dengan jelas!
        """
        try:
            # Step 1: Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Step 2: SHARPENING - Tambah ketajaman detail!
            # Penjelasan SMK: Seperti "focus" kamera, biar detail lebih tajam
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

            # Step 3: CLAHE AGRESIF - Adaptive Histogram Equalization
            # Penjelasan SMK: "Auto-brightness" lebih kuat untuk plat kecil
            # clipLimit naik dari 2.0 ke 3.0 untuk kontras lebih tinggi
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)

            # Step 4: Bilateral Filter - Kurangi noise tapi jaga edges
            # Penjelasan SMK: "Smooth" tapi garis-garis tetap tajam
            bilateral = cv2.bilateralFilter(enhanced, 11, 17, 17)

            # Step 5: Morphological Opening - Hilangkan noise kecil
            # Penjelasan SMK: Hapus "titik-titik kecil" yang ganggu
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(bilateral, cv2.MORPH_OPEN, kernel)

            # Step 6: Edge detection LEBIH SENSITIF untuk plat kecil
            # Penjelasan SMK: Turunkan threshold biar deteksi edge lebih sensitif
            # Threshold turun dari 30→20 untuk deteksi plat lebih kecil
            edges = cv2.Canny(morph, 20, 200)

            # Step 7: Morphological Closing - Sambung garis yang putus
            # Penjelasan SMK: "Sambung" garis yang terputus biar jadi kotak utuh
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)

            # Step 8: Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            plate_candidates = []

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter 1: Ukuran plat
                if w < self.MIN_PLATE_WIDTH or w > self.MAX_PLATE_WIDTH:
                    continue
                if h < self.MIN_PLATE_HEIGHT or h > self.MAX_PLATE_HEIGHT:
                    continue

                # Filter 2: Aspect ratio (bentuk persegi panjang)
                aspect_ratio = w / h
                if aspect_ratio < self.MIN_ASPECT_RATIO or aspect_ratio > self.MAX_ASPECT_RATIO:
                    continue

                # Filter 3: WARNA PUTIH/TERANG (khas plat Indonesia!)
                # Penjelasan SMK: Cek apakah area ini putih seperti plat
                try:
                    roi = gray[y:y+h, x:x+w]
                    if not self._is_white_region(roi):
                        continue  # Skip kalau tidak putih
                except:
                    continue

                # Filter 4: ADA TEXT/HURUF di dalamnya
                # Penjelasan SMK: Cek apakah ada huruf/angka hitam
                try:
                    if not self._has_text_characters(roi):
                        continue  # Skip kalau tidak ada text
                except:
                    continue

                # Kandidat VALID - persegi panjang putih dengan text!
                # Hitung confidence score untuk ranking
                area = w * h

                # Calculate confidence score (0.0 - 1.0)
                confidence_score = self._calculate_plate_confidence(roi, aspect_ratio, area)

                plate_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence_score
                })

            # Return MULTIPLE bounding boxes (bukan cuma 1!)
            if plate_candidates:
                # Sort by CONFIDENCE dulu, lalu area (yang terbaik dulu)
                plate_candidates.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)

                # Ambil top N candidates (maksimal self.max_detections)
                top_candidates = plate_candidates[:self.max_detections]

                # Extract bbox dari setiap candidate
                bboxes = [candidate['bbox'] for candidate in top_candidates]

                # Log confidence scores
                for i, candidate in enumerate(top_candidates):
                    logger.debug(f"Plate #{i+1}: confidence={candidate['confidence']:.2f}, "
                               f"area={candidate['area']}, aspect={candidate['aspect_ratio']:.2f}")

                logger.info(f"{len(bboxes)} plate(s) detected (sorted by confidence)")
                return bboxes  # Return LIST of bboxes, sorted by quality!

            return []  # Return empty list kalau tidak ada
            
        except Exception as e:
            logger.error(f"Error in contour detection: {e}")
            return None

    def _calculate_plate_confidence(self, roi, aspect_ratio, area):
        """
        Hitung confidence score untuk ranking plate candidates

        Args:
            roi: Region of interest (cropped plate area)
            aspect_ratio: Width/height ratio
            area: Total area (pixels)

        Returns:
            confidence: 0.0 - 1.0 (higher is better)
        """
        confidence = 0.0

        try:
            # Factor 1: Aspect ratio (40%) - ideal plat Indonesia sekitar 3.5:1
            ideal_aspect = 3.5
            aspect_deviation = abs(aspect_ratio - ideal_aspect)
            aspect_score = max(0, 1.0 - (aspect_deviation / 2.0))  # Penalty untuk deviation
            confidence += aspect_score * 0.40

            # Factor 2: Brightness uniformity (30%) - plat putih punya background uniform
            avg_brightness = np.mean(roi)
            std_dev = np.std(roi)

            # Ideal: brightness tinggi (putih), std dev rendah (uniform)
            brightness_score = min(avg_brightness / 255.0, 1.0)  # Normalize ke 0-1
            uniformity_score = max(0, 1.0 - (std_dev / 100.0))   # Lower std = better

            confidence += (brightness_score * 0.15 + uniformity_score * 0.15)

            # Factor 3: Text density (20%) - plat harus punya text density optimal
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            text_pixels = np.sum(binary == 255)
            total_pixels = binary.shape[0] * binary.shape[1]
            text_density = text_pixels / total_pixels

            # Ideal text density: 20-25% (huruf hitam di background putih)
            ideal_density = 0.225
            density_deviation = abs(text_density - ideal_density)
            density_score = max(0, 1.0 - (density_deviation * 5))  # Penalty untuk deviation
            confidence += density_score * 0.20

            # Factor 4: Size (10%) - plat lebih besar = lebih jelas
            # Normalize: 70px (min) = 0.0, 400px (optimal) = 1.0
            size_score = min((area ** 0.5) / 400.0, 1.0)
            confidence += size_score * 0.10

        except Exception as e:
            logger.debug(f"Error calculating confidence: {e}")
            confidence = 0.5  # Fallback

        return min(confidence, 1.0)  # Cap at 1.0

    def draw_detections(self, frame, bboxes, color=(0, 255, 0), thickness=2):
        """
        Gambar MULTIPLE kotak di semua plat yang terdeteksi

        Penjelasan SMK: Seperti "tandai semua plat" yang ketemu,
        bukan cuma 1 plat tapi semua plat di frame!

        Args:
            frame: Gambar asli
            bboxes: LIST of (x, y, w, h) - bisa banyak bbox!
            color: Warna kotak
            thickness: Ketebalan garis
        """
        if not bboxes:  # Kalau list kosong
            return frame

        # Warna berbeda untuk setiap plat
        colors = [
            (0, 255, 0),    # Hijau
            (255, 0, 0),    # Biru
            (0, 255, 255),  # Kuning
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
        ]

        # Gambar setiap bounding box
        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox

            # Pakai warna berbeda untuk setiap plat
            box_color = colors[i % len(colors)]

            # Gambar rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)

            # Label dengan nomor
            label = f"PLATE #{i+1}"
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Tampilkan jumlah total plat terdeteksi
        total_text = f"Total: {len(bboxes)} plate(s)"
        cv2.putText(frame, total_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
