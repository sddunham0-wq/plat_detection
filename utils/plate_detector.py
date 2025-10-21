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

        # Parameter filter plat INDONESIA (persegi panjang putih) - OPTIMIZED!
        self.MIN_PLATE_WIDTH = 45   # Turun ke 45 (terima plat lebih kecil/jauh)
        self.MAX_PLATE_WIDTH = 600  # Support berbagai jarak
        self.MIN_PLATE_HEIGHT = 12  # Turun ke 12 (terima plat lebih kecil)
        self.MAX_PLATE_HEIGHT = 150 # Support berbagai jarak

        # Aspect ratio plat Indonesia - OPTIMIZED! (terima berbagai angle)
        self.MIN_ASPECT_RATIO = 2.0  # Turun ke 2.0 (terima plat agak miring)
        self.MAX_ASPECT_RATIO = 5.5  # Naik ke 5.5 (terima plat panjang/angle)

        # Parameter untuk deteksi warna - DUAL COLOR SUPPORT! ★
        # White plate (background putih, text hitam) - mobil pribadi
        self.MIN_WHITE_BRIGHTNESS = 100  # Turun ke 100 (terima plat di shadow/cahaya kurang)
        self.MAX_WHITE_BRIGHTNESS = 255  # Batas atas (putih sempurna)

        # Black plate (background hitam, text putih) - pemerintah/TNI/Polri ★ BARU!
        self.MIN_BLACK_BRIGHTNESS = 0    # Batas bawah (hitam sempurna)
        self.MAX_BLACK_BRIGHTNESS = 80   # Batas atas (hitam-abu gelap)

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

    def _is_valid_plate_color(self, gray_roi):
        """
        DUAL COLOR DETECTION ★ - Support plat PUTIH dan HITAM!

        Penjelasan SMK: Cek apakah area ini PUTIH atau HITAM (plat Indonesia)

        Jenis plat Indonesia:
        1. PUTIH (brightness 100-255) → Mobil pribadi (text hitam, bg putih)
        2. HITAM (brightness 0-80) → Pemerintah/TNI/Polri (text putih, bg hitam) ★ BARU!
        3. KUNING (brightness ~150-200) → Angkutan umum
        4. MERAH (brightness ~100-150) → Sementara/diplomatic

        Returns:
            True jika warna valid untuk plat Indonesia
        """
        avg_brightness = np.mean(gray_roi)

        # Check 1: White plate (background putih) - mobil pribadi
        # ★ BUG FIX: Turunkan threshold dari 100 ke 90 untuk handle plat putih kotor/berdebu
        is_white_plate = (avg_brightness >= 90 and
                         avg_brightness <= self.MAX_WHITE_BRIGHTNESS)

        # Check 2: Black plate (background hitam) - pemerintah/TNI/Polri
        # ★ BUG FIX: Naikkan threshold dari 80 ke 89 untuk non-overlapping (0-89 black, 90-255 white)
        is_black_plate = (avg_brightness >= self.MIN_BLACK_BRIGHTNESS and
                         avg_brightness < 90)

        # Accept either white OR black plate
        is_valid_color = is_white_plate or is_black_plate

        if not is_valid_color:
            return False

        # Additional check: Background uniformity
        # Plat punya background yang relatif uniform (tidak banyak variasi)
        std_dev = np.std(gray_roi)
        has_uniform_background = std_dev < 60

        # Debug log
        if is_black_plate:
            logger.debug(f"BLACK PLATE detected: brightness={avg_brightness:.1f}, std={std_dev:.1f}")
        elif is_white_plate:
            logger.debug(f"WHITE PLATE detected: brightness={avg_brightness:.1f}, std={std_dev:.1f}")

        return is_valid_color and has_uniform_background

    def _has_plate_border(self, gray_roi):
        """
        Deteksi border/frame plat (tepi hitam yang mengelilingi area putih/hitam)

        Penjelasan SMK: Plat Indonesia punya frame hitam di tepi.
        Cek apakah ada garis hitam di pinggir atas/bawah/kiri/kanan.

        Returns:
            True jika ada border yang jelas
        """
        try:
            h, w = gray_roi.shape

            # Sample tepi plat (5 pixel dari edge)
            border_width = min(5, h // 10, w // 20)

            top_border = gray_roi[:border_width, :]
            bottom_border = gray_roi[-border_width:, :]
            left_border = gray_roi[:, :border_width]
            right_border = gray_roi[:, -border_width:]

            # Hitung avg brightness di setiap border
            borders = [top_border, bottom_border, left_border, right_border]
            border_brightness = [np.mean(b) for b in borders]

            # Plat punya border gelap (hitam) di minimal 2 sisi
            dark_borders = sum(1 for b in border_brightness if b < 100)

            has_border = dark_borders >= 2
            logger.debug(f"Border check: {dark_borders}/4 dark borders, has_border={has_border}")

            return has_border

        except Exception as e:
            logger.debug(f"Border detection error: {e}")
            return True  # Fallback: assume has border

    def _has_text_characters(self, gray_roi):
        """
        DUAL TEXT DETECTION ★ - Support plat PUTIH dan HITAM!

        Penjelasan SMK: Cek apakah ada HURUF/ANGKA di dalam area

        Cara kerja:
        1. Deteksi tipe plat (putih atau hitam) dari brightness
        2. WHITE PLATE: Cari text HITAM di background PUTIH
        3. BLACK PLATE: Cari text PUTIH di background HITAM ★ BARU!
        4. Hitung text density (10-40%)

        Returns:
            True jika ada text dengan density yang wajar
        """
        # Detect plate type first
        avg_brightness = np.mean(gray_roi)
        # ★ BUG FIX: Update thresholds to match _is_valid_plate_color (non-overlapping)
        is_white_plate = avg_brightness >= 90  # White background (relaxed from 100)
        is_black_plate = avg_brightness < 90   # Black background (up to 89)

        # Apply appropriate threshold based on plate type
        if is_white_plate:
            # White plate: detect BLACK text on WHITE background
            # BINARY_INV = invert (text jadi putih di hasil binary)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            logger.debug("Text detection: WHITE plate mode (black text on white bg)")

        elif is_black_plate:
            # Black plate: detect WHITE text on BLACK background ★ BARU!
            # BINARY = no invert (text putih sudah jadi putih di hasil binary)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            logger.debug("Text detection: BLACK plate mode (white text on black bg)")

        else:
            # Medium brightness (kuning/merah) - try default
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            logger.debug(f"Text detection: DEFAULT mode (brightness={avg_brightness:.1f})")

        # Hitung % area yang berisi text
        text_pixels = np.sum(binary == 255)
        total_pixels = binary.shape[0] * binary.shape[1]
        text_density = text_pixels / total_pixels

        # Filter: text density harus dalam range yang wajar
        has_enough_text = text_density >= self.MIN_TEXT_DENSITY   # >= 10%
        not_too_dark = text_density <= self.MAX_TEXT_DENSITY      # <= 40%

        logger.debug(f"Text density: {text_density:.2%} (valid: {has_enough_text and not_too_dark})")

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

                # Filter 3: WARNA VALID (putih atau hitam) - DUAL COLOR! ★
                # Penjelasan SMK: Cek apakah warna valid untuk plat Indonesia
                # Support: putih (mobil pribadi) DAN hitam (pemerintah/TNI/Polri)
                try:
                    # ★ BUG FIX: Add bounds checking untuk prevent out-of-bounds error
                    img_h, img_w = gray.shape[:2]
                    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                        logger.warning(f"Invalid bbox: ({x},{y},{w},{h}) for image {img_w}x{img_h}")
                        continue

                    roi = gray[y:y+h, x:x+w]
                    if not self._is_valid_plate_color(roi):
                        continue  # Skip kalau warna tidak valid
                except Exception as e:
                    logger.debug(f"Color check error: {e}")
                    continue

                # Filter 4: ADA BORDER PLAT (frame hitam di tepi) ★ BARU!
                # Penjelasan SMK: Plat asli punya frame hitam, bumper mobil tidak
                try:
                    if not self._has_plate_border(roi):
                        logger.debug(f"Rejected: no plate border detected")
                        continue  # Skip kalau tidak ada border
                except:
                    pass  # Ignore error, lanjut ke text check

                # Filter 5: ADA TEXT/HURUF di dalamnya
                # Penjelasan SMK: Cek apakah ada huruf/angka
                try:
                    if not self._has_text_characters(roi):
                        continue  # Skip kalau tidak ada text
                except:
                    continue

                # Kandidat VALID - plat dengan border + text! (putih atau hitam)
                # Hitung confidence score untuk ranking
                area = w * h

                # Detect plate type ★ BARU!
                avg_brightness = np.mean(roi)
                plate_type = "BLACK" if avg_brightness <= 80 else ("WHITE" if avg_brightness >= 100 else "OTHER")

                # Calculate confidence score (0.0 - 1.0)
                confidence_score = self._calculate_plate_confidence(roi, aspect_ratio, area)

                plate_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'confidence': confidence_score,
                    'plate_type': plate_type  # ★ BARU! Track plate type
                })

            # Return MULTIPLE bounding boxes (bukan cuma 1!)
            if plate_candidates:
                # Sort by CONFIDENCE dulu, lalu area (yang terbaik dulu)
                plate_candidates.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)

                # Ambil top N candidates (maksimal self.max_detections)
                top_candidates = plate_candidates[:self.max_detections]

                # Extract bbox dari setiap candidate
                bboxes = [candidate['bbox'] for candidate in top_candidates]

                # Log confidence scores with PLATE TYPE ★ BARU!
                for i, candidate in enumerate(top_candidates):
                    logger.debug(f"Plate #{i+1} ({candidate['plate_type']}): confidence={candidate['confidence']:.2f}, "
                               f"area={candidate['area']}, aspect={candidate['aspect_ratio']:.2f}")

                # Count plate types ★ BARU!
                white_count = sum(1 for c in top_candidates if c['plate_type'] == 'WHITE')
                black_count = sum(1 for c in top_candidates if c['plate_type'] == 'BLACK')
                other_count = sum(1 for c in top_candidates if c['plate_type'] == 'OTHER')

                # Enhanced logging with plate type breakdown ★ BARU!
                type_info = []
                if white_count > 0:
                    type_info.append(f"{white_count} white")
                if black_count > 0:
                    type_info.append(f"{black_count} black")
                if other_count > 0:
                    type_info.append(f"{other_count} other")

                logger.info(f"{len(bboxes)} plate(s) detected ({', '.join(type_info)}) - sorted by confidence")
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

            # Factor 2: Brightness uniformity (30%) - DUAL COLOR SUPPORT! ★
            avg_brightness = np.mean(roi)
            std_dev = np.std(roi)

            # Detect plate type first
            # ★ BUG FIX: Update thresholds to match _is_valid_plate_color (non-overlapping)
            is_white_plate = avg_brightness >= 90  # White background (relaxed from 100)
            is_black_plate = avg_brightness < 90   # Black background (up to 89)

            # Apply appropriate brightness scoring based on plate type
            if is_white_plate:
                # White plate: higher brightness = better
                brightness_score = min(avg_brightness / 255.0, 1.0)
            elif is_black_plate:
                # Black plate: lower brightness = better ★ BARU!
                brightness_score = max(0, 1.0 - (avg_brightness / 80.0))
            else:
                # Medium brightness (yellow/red plates) - neutral score
                brightness_score = 0.5

            uniformity_score = max(0, 1.0 - (std_dev / 100.0))   # Lower std = better

            confidence += (brightness_score * 0.15 + uniformity_score * 0.15)

            # Factor 3: Text density (20%) - plat harus punya text density optimal
            # ★ BUG FIX: Detect plate type first untuk correct threshold mode
            if is_white_plate:
                # White plate: detect BLACK text on WHITE background
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            elif is_black_plate:
                # Black plate: detect WHITE text on BLACK background ★ FIXED!
                _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                # Default: white plate mode
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
        # ★ BUG FIX: Defensive checks untuk berbagai input types
        if bboxes is None:
            return frame

        if not isinstance(bboxes, list):
            return frame

        if len(bboxes) == 0:
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
            try:
                x, y, w, h = bbox

                # ★ Validasi coordinate (tidak boleh negative atau di luar frame)
                frame_h, frame_w = frame.shape[:2]
                if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                    logger.warning(f"Skipping invalid bbox: ({x},{y},{w},{h})")
                    continue

                # Pakai warna berbeda untuk setiap plat
                box_color = colors[i % len(colors)]

                # Gambar rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)

                # Label dengan nomor
                label = f"PLATE #{i+1}"
                cv2.putText(frame, label, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            except Exception as e:
                logger.debug(f"Error drawing bbox #{i}: {e}")
                continue

        # Tampilkan jumlah total plat terdeteksi
        total_text = f"Total: {len(bboxes)} plate(s)"
        cv2.putText(frame, total_text, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
