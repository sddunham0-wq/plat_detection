# -*- coding: utf-8 -*-
"""
OCR PROCESSOR - FIXED VERSION
Lebih simpel, lebih reliable, multiple fallback
"""

import cv2
import numpy as np
import pytesseract
import re
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR Processor dengan multiple fallback strategy"""

    def __init__(self):
        # Multiple PSM modes untuk fallback
        self.psm_modes = [
            ('PSM 7 - Single Line', '--psm 7 --oem 3'),
            ('PSM 8 - Single Word', '--psm 8 --oem 3'),
            ('PSM 6 - Block of Text', '--psm 6 --oem 3'),
            ('PSM 13 - Raw Line', '--psm 13 --oem 3'),
        ]

        # Whitelist karakter (huruf + angka + spasi untuk format Indonesia)
        self.char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '

        logger.info("OCR Processor initialized with multiple PSM fallback")

    def preprocess_simple(self, img):
        """Preprocessing SIMPLE - resize + grayscale + threshold"""
        try:
            # Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Resize KE 400px width (lebih besar = lebih baik)
            h, w = gray.shape
            if w < 400:
                scale = 400 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Simple threshold (Otsu)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            logger.error(f"Error in simple preprocessing: {e}")
            return img

    def preprocess_advanced(self, img):
        """Preprocessing ADVANCED - dengan CLAHE dan denoising"""
        try:
            # Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Resize
            h, w = gray.shape
            if w < 400:
                scale = 400 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # CLAHE untuk kontras
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Bilateral filter
            bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Threshold
            _, binary = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Denoise
            denoised = cv2.medianBlur(binary, 3)

            return denoised

        except Exception as e:
            logger.error(f"Error in advanced preprocessing: {e}")
            return self.preprocess_simple(img)

    def clean_text(self, text, preserve_spaces=True):
        """
        Bersihkan text OCR

        Args:
            text: Raw OCR text
            preserve_spaces: True = keep spaces (untuk format F 1234 ABC)
                           False = remove all spaces (untuk format F1234ABC)
        """
        # Remove newline, tabs, multiple spaces
        text = text.strip().replace('\n', '').replace('\r', '').replace('\t', '')

        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)

        # Convert to uppercase
        text = text.upper()

        if preserve_spaces:
            # Keep spaces, remove hanya karakter aneh
            text = re.sub(r'[^A-Z0-9\s]', '', text)
            # Format: "F 1234 ABC" atau "F1234ABC" (kedua-duanya valid)
        else:
            # Remove ALL spaces (old behavior)
            text = text.replace(' ', '')
            text = re.sub(r'[^A-Z0-9]', '', text)

        return text.strip()

    def auto_correct_plate(self, text):
        """
        Auto-correction untuk common OCR errors

        Penjelasan SMK: OCR sering salah baca huruf yang mirip angka
        Kita koreksi otomatis berdasarkan posisi karakter
        """
        if not text or len(text) < 3:
            return text

        # Convert to list untuk mudah edit
        chars = list(text)

        # Deteksi pola plat Indonesia: [huruf][angka][huruf]
        # Format: B1234ABC atau AA1234CC

        # PART 1: Huruf DEPAN (1-2 karakter pertama)
        # Kalau ada angka di depan, convert ke huruf
        for i in range(min(2, len(chars))):
            # Common errors di posisi huruf depan
            if chars[i] == '0':
                chars[i] = 'O'
            elif chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '5':
                chars[i] = 'S'
            elif chars[i] == '8':
                chars[i] = 'B'

        # PART 2: Angka TENGAH (setelah huruf depan)
        # Mulai dari karakter ke-2 atau ke-3
        start_digit = 1 if chars[0].isalpha() else 2

        # Cari dimana angka dimulai
        digit_start = -1
        for i in range(start_digit, len(chars)):
            if chars[i].isdigit():
                digit_start = i
                break

        # PART 3: Huruf BELAKANG (1-3 karakter terakhir)
        # Kalau ada angka di belakang, convert ke huruf
        if len(chars) >= 3:
            # 3 karakter terakhir biasanya huruf
            for i in range(max(0, len(chars) - 3), len(chars)):
                # Skip kalau masih di bagian angka
                if digit_start >= 0 and i <= digit_start + 4:  # Max 4 digit
                    continue

                # Common errors di posisi huruf belakang
                if chars[i] == '0':
                    chars[i] = 'O'
                elif chars[i] == '1':
                    chars[i] = 'I'
                elif chars[i] == '5':
                    chars[i] = 'S'
                elif chars[i] == '8':
                    chars[i] = 'B'

        corrected = ''.join(chars)

        if corrected != text:
            logger.debug(f"Auto-corrected: {text} → {corrected}")

        return corrected

    def format_indonesian_plate(self, text):
        """
        Format plat Indonesia dengan spasi yang benar

        Input: "F1234ABC" atau "F 1234 ABC" atau "F1234 ABC"
        Output: "F 1234 ABC" (format standar)

        Returns:
            formatted_text: Plat dengan format standar
            confidence: 0.0-1.0 (seberapa yakin ini format Indonesia)
        """
        if not text:
            return text, 0.0

        # Remove ALL spaces dulu untuk normalisasi
        text_no_space = text.replace(' ', '')

        # Pattern plat Indonesia: [Huruf][Angka][Huruf]
        # Contoh: F1234ABC, B1234XYZ, AA1234BB
        pattern = r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$'
        match = re.match(pattern, text_no_space)

        if match:
            area_code = match.group(1)    # "F" atau "AA"
            number = match.group(2)       # "1234"
            series = match.group(3)       # "ABC"

            # Format dengan spasi: "F 1234 ABC"
            formatted = f"{area_code} {number} {series}"

            # Hitung confidence berdasarkan format
            confidence = 1.0

            # Penalty kalau format aneh
            if len(number) < 2:  # Angka terlalu pendek
                confidence -= 0.2
            if len(series) < 1:  # Series terlalu pendek
                confidence -= 0.2

            return formatted, max(confidence, 0.5)
        else:
            # Tidak match pattern Indonesia
            return text, 0.3

    def is_valid_plate(self, text):
        """
        Validasi plat Indonesia dengan format yang benar

        Valid formats:
        - "F 1234 ABC" (dengan spasi)
        - "F1234ABC" (tanpa spasi)
        - "AA 1234 BB" (2 huruf depan)
        """
        if not text or len(text) < 2:
            return False

        # Coba format dengan spasi
        formatted, confidence = self.format_indonesian_plate(text)

        # Valid kalau confidence >= 0.5 (format match)
        return confidence >= 0.5

    def ocr_single_mode(self, img, psm_config):
        """OCR dengan 1 PSM mode"""
        try:
            config = f"{psm_config} -c tessedit_char_whitelist={self.char_whitelist}"

            raw_text = pytesseract.image_to_string(img, config=config)
            cleaned = self.clean_text(raw_text)

            return cleaned

        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return ""

    def read_plate_text(self, plate_img):
        """
        Baca plat dengan MULTIPLE FALLBACK strategy

        Strategy:
        1. Try advanced preprocessing + PSM 7
        2. Try simple preprocessing + PSM 7
        3. Try advanced preprocessing + PSM 8
        4. Try simple preprocessing + PSM 8
        5. Try all other PSM modes
        6. Return best result dengan format Indonesia (F 1234 ABC)
        """
        try:
            results = []

            # Preprocessing options
            preprocessed_images = [
                ('Advanced', self.preprocess_advanced(plate_img)),
                ('Simple', self.preprocess_simple(plate_img)),
            ]

            # Try each preprocessing + PSM combination
            for prep_name, processed_img in preprocessed_images:
                for psm_name, psm_config in self.psm_modes:

                    text = self.ocr_single_mode(processed_img, psm_config)

                    if text:
                        # Auto-correct common OCR errors
                        corrected_text = self.auto_correct_plate(text)

                        # Format dengan spasi standar Indonesia
                        formatted_text, format_confidence = self.format_indonesian_plate(corrected_text)

                        results.append({
                            'text': formatted_text,
                            'original': text,
                            'length': len(formatted_text.replace(' ', '')),  # Length tanpa spasi
                            'valid': self.is_valid_plate(formatted_text),
                            'format_confidence': format_confidence,
                            'method': f"{prep_name} + {psm_name}"
                        })

            # Sort results: valid first, format confidence, then by length
            results.sort(key=lambda x: (not x['valid'], -x['format_confidence'], -x['length']))

            # Log all results
            logger.debug(f"OCR tried {len(results)} combinations")
            for i, r in enumerate(results[:3]):
                logger.debug(f"  [{i+1}] {r['text']} (valid={r['valid']}, conf={r['format_confidence']:.2f}, method={r['method']})")

            # Return best result
            if results:
                best = results[0]
                if best['valid']:
                    logger.info(f"✅ OCR SUCCESS: {best['text']} (method: {best['method']})")
                    return best['text']
                elif best['length'] >= 3:
                    logger.warning(f"⚠️ OCR PARTIAL: {best['text']} (not fully valid)")
                    return best['text']

            logger.warning(f"❌ OCR FAILED - No readable text")
            return None

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None

    def read_plate_with_confidence(self, plate_img):
        """Baca plat dengan confidence score dan format Indonesia"""
        try:
            # Get text dengan format Indonesia
            text = self.read_plate_text(plate_img)

            if not text:
                return None, 0.0

            # Try to get confidence from OCR data
            try:
                processed = self.preprocess_advanced(plate_img)
                config = f"--psm 7 --oem 3 -c tessedit_char_whitelist={self.char_whitelist}"

                ocr_data = pytesseract.image_to_data(
                    processed,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )

                # Calculate average confidence
                confidences = [float(c) for c in ocr_data['conf'] if float(c) > 0]

                if confidences:
                    avg_conf = sum(confidences) / len(confidences) / 100.0
                else:
                    avg_conf = 0.5

            except:
                # Fallback confidence based on text quality and format
                _, format_conf = self.format_indonesian_plate(text)

                if self.is_valid_plate(text):
                    avg_conf = 0.7 * format_conf  # Combine validation + format confidence
                else:
                    avg_conf = 0.4

            return text, avg_conf

        except Exception as e:
            logger.error(f"Error getting confidence: {e}")
            return None, 0.0
