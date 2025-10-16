# -*- coding: utf-8 -*-
"""
OCR PROCESSOR SEDERHANA
Baca teks plat nomor dengan Tesseract OCR
"""

import cv2
import numpy as np
import pytesseract
import re
import logging

logger = logging.getLogger(__name__)

class SimpleOCRProcessor:
    """OCR sederhana untuk baca plat nomor"""

    def __init__(self):
        # Whitelist karakter Indonesia (huruf + angka + spasi)
        self.whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '

        # PSM modes untuk fallback
        self.psm_modes = [7, 8, 6]  # Single line, word, block

        logger.info("Simple OCR Processor initialized")

    def preprocess(self, img):
        """Preprocessing gambar untuk OCR"""
        try:
            # Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Resize lebih besar
            h, w = gray.shape
            if w < 400:
                scale = 400 / w
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # CLAHE untuk kontras
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Bilateral filter
            smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Threshold
            _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            logger.error(f"Preprocess error: {e}")
            return img

    def clean_text(self, text):
        """Bersihkan hasil OCR"""
        # Remove newlines dan multiple spaces
        text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')
        text = re.sub(r'\s+', ' ', text)

        # Uppercase
        text = text.upper()

        # Remove karakter aneh (keep huruf, angka, spasi)
        text = re.sub(r'[^A-Z0-9\s]', '', text)

        return text.strip()

    def format_indonesian_plate(self, text):
        """
        Format plat Indonesia: F 1234 ABC

        Returns:
            (formatted_text, confidence)
        """
        if not text:
            return text, 0.0

        # Remove spaces dulu
        text_clean = text.replace(' ', '')

        # Pattern Indonesia: [Huruf][Angka][Huruf]
        pattern = r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$'
        match = re.match(pattern, text_clean)

        if match:
            area = match.group(1)
            number = match.group(2)
            series = match.group(3)

            # Format dengan spasi
            formatted = f"{area} {number} {series}"
            return formatted, 1.0

        return text, 0.3

    def read_plate(self, plate_img):
        """
        Baca teks dari gambar plat dengan multiple fallback

        Returns:
            (text, confidence)
        """
        try:
            # Validate input
            if plate_img is None or plate_img.size == 0:
                return None, 0.0

            # Check minimum size
            h, w = plate_img.shape[:2]
            if w < 50 or h < 15:
                logger.warning(f"Plate too small: {w}x{h}")
                return None, 0.0

            # Preprocess dengan error handling
            try:
                processed = self.preprocess(plate_img)
            except Exception as e:
                logger.error(f"Preprocess error: {e}")
                # Fallback: use original
                processed = plate_img

            best_text = None
            best_conf = 0.0
            attempts = 0

            # Try different PSM modes dengan retry
            for psm in self.psm_modes:
                try:
                    config = f"--psm {psm} --oem 3 -c tessedit_char_whitelist={self.whitelist}"

                    # OCR dengan timeout implicit
                    raw_text = pytesseract.image_to_string(processed, config=config)
                    attempts += 1

                    # Clean
                    text = self.clean_text(raw_text)

                    if not text or len(text) < 3:
                        continue

                    # Format
                    formatted, conf = self.format_indonesian_plate(text)

                    # Keep best result
                    if conf > best_conf:
                        best_text = formatted
                        best_conf = conf

                        # Early exit jika sudah dapat hasil bagus
                        if conf >= 0.9:
                            break

                except pytesseract.TesseractError as te:
                    logger.warning(f"Tesseract error PSM {psm}: {te}")
                    continue
                except Exception as e:
                    logger.debug(f"OCR PSM {psm} error: {e}")
                    continue

            # Return best result
            if best_text and best_conf > 0.3:
                logger.info(f"✅ OCR: {best_text} ({best_conf:.2f}) in {attempts} attempts")
                return best_text, best_conf
            else:
                logger.warning(f"❌ OCR failed after {attempts} attempts")
                return None, 0.0

        except Exception as e:
            logger.error(f"Read plate critical error: {e}")
            return None, 0.0

    def read_plate_with_confidence(self, plate_img):
        """Alias untuk compatibility"""
        return self.read_plate(plate_img)
