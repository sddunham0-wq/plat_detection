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

# ★ EASYOCR INTEGRATION
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("✅ EasyOCR available for fallback")
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("⚠️  EasyOCR not available, using Tesseract only")

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

        # ★ Initialize EasyOCR sebagai fallback
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                # GPU=False untuk kompatibilitas, lang=['en'] untuk plat Indonesia
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("✅ EasyOCR reader initialized (English)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None

        logger.info("OCR Processor initialized with multiple PSM fallback")

    def preprocess_simple(self, img):
        """Preprocessing SIMPLE - resize + grayscale + threshold"""
        try:
            # Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # ★ AGGRESSIVE UPSCALING untuk plat kecil
            # Penjelasan SMK: Perbesar gambar 6x untuk OCR lebih akurat
            # Target: 120px → 720px (6x), 150px → 600px (4x)
            h, w = gray.shape
            target_width = 600  # Naik dari 400 ke 600
            if w < target_width:
                scale = target_width / w
                scale = min(scale, 6.0)  # Max 6x scaling (naik dari 4x)
                new_w = int(w * scale)
                new_h = int(h * scale)

                # Gunakan INTER_LANCZOS4 untuk upscaling terbaik
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                logger.debug(f"📏 Upscaled: {w}x{h} → {new_w}x{new_h} ({scale:.1f}x)")

            # ★ SHARPENING untuk detail lebih tajam
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(gray, -1, kernel_sharpen)

            # Simple threshold (Otsu)
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

            # ★ AGGRESSIVE UPSCALING (sama seperti simple)
            h, w = gray.shape
            target_width = 600  # Naik dari 400
            if w < target_width:
                scale = target_width / w
                scale = min(scale, 6.0)  # Max 6x scaling
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                logger.debug(f"📏 Advanced upscaled: {w}x{h} → {new_w}x{new_h} ({scale:.1f}x)")

            # ★ CLAHE LEBIH AGRESIF untuk kontras tinggi
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Naik dari 2.5
            enhanced = clahe.apply(gray)

            # ★ SHARPENING sebelum bilateral
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

            # Bilateral filter
            bilateral = cv2.bilateralFilter(sharpened, 9, 75, 75)

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
            # ★ UPSCALE IMAGE BEFORE OCR (FIX UNTUK CROP KECIL!)
            # Penjelasan SMK: Zoom gambar supaya lebih besar dan jelas untuk Tesseract
            # Gambar kecil (120x64) → Zoom 4x → Jadi besar (480x256) → Lebih mudah dibaca!
            h, w = img.shape[:2]
            target_height = 256  # Target minimal untuk OCR yang bagus
            scale_factor = max(4.0, target_height / h)  # Minimum 4x zoom

            if scale_factor > 1.0:
                # cv2.INTER_CUBIC = Interpolasi berkualitas tinggi (smooth, tidak pixelated)
                img = cv2.resize(img, None,
                                fx=scale_factor,
                                fy=scale_factor,
                                interpolation=cv2.INTER_CUBIC)
                logger.debug(f"📐 [Tesseract] Upscaled from {w}x{h} to {img.shape[1]}x{img.shape[0]} ({scale_factor:.1f}x)")

            config = f"{psm_config} -c tessedit_char_whitelist={self.char_whitelist}"

            raw_text = pytesseract.image_to_string(img, config=config)
            cleaned = self.clean_text(raw_text)

            return cleaned

        except Exception as e:
            logger.debug(f"OCR error: {e}")
            return ""

    def ocr_with_easyocr(self, img):
        """
        OCR menggunakan EasyOCR sebagai fallback

        Penjelasan SMK: EasyOCR lebih bagus untuk plat nomor Indonesia
        karena trained dengan deep learning (lebih pintar)

        Returns:
            text: Hasil OCR atau None kalau gagal
        """
        if not self.easyocr_reader:
            return None

        try:
            # EasyOCR butuh BGR image (bukan grayscale/binary)
            if len(img.shape) == 2:
                # Convert grayscale to BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img

            # ★ UPSCALE IMAGE BEFORE OCR (FIX UNTUK CROP KECIL!)
            # Penjelasan SMK: Zoom gambar supaya lebih besar dan jelas untuk OCR
            # Gambar kecil (120x64) → Zoom 4x → Jadi besar (480x256) → Lebih mudah dibaca!
            h, w = img_bgr.shape[:2]
            target_height = 256  # Target minimal untuk OCR yang bagus
            scale_factor = max(4.0, target_height / h)  # Minimum 4x zoom

            if scale_factor > 1.0:
                # cv2.INTER_CUBIC = Interpolasi berkualitas tinggi (smooth, tidak pixelated)
                img_bgr = cv2.resize(img_bgr, None,
                                    fx=scale_factor,
                                    fy=scale_factor,
                                    interpolation=cv2.INTER_CUBIC)
                logger.debug(f"📐 Upscaled from {w}x{h} to {img_bgr.shape[1]}x{img_bgr.shape[0]} ({scale_factor:.1f}x)")

            # Read text dengan EasyOCR
            results = self.easyocr_reader.readtext(img_bgr, detail=1)

            if not results:
                return None

            # Ambil text dengan confidence tertinggi
            best_result = max(results, key=lambda x: x[2])  # x[2] = confidence
            text = best_result[1]  # x[1] = text
            confidence = best_result[2]  # x[2] = confidence

            # Clean text
            cleaned = self.clean_text(text)

            logger.debug(f"EasyOCR: '{cleaned}' (conf: {confidence:.2f})")

            return cleaned if confidence > 0.3 else None

        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")
            return None

    def read_plate_text(self, plate_img):
        """
        Baca plat dengan EASYOCR FIRST strategy (REVERSED ORDER)

        NEW Strategy (OPTIMIZED untuk akurasi tinggi):
        1. ★ TRY EASYOCR FIRST (paling akurat untuk plat Indonesia!)
        2. Fallback ke Tesseract kalau EasyOCR gagal
        3. Return best result dengan format Indonesia (B 1234 ABC)

        Kenapa EasyOCR first?
        - Deep learning based → lebih pintar
        - Trained untuk Asian text → cocok untuk Indonesia
        - Lebih akurat untuk plat yang sudah jelas

        Tesseract hanya sebagai fallback untuk edge cases.
        """
        try:
            # ★ STRATEGY 1: TRY EASYOCR FIRST (HIGHEST ACCURACY!)
            if self.easyocr_reader:
                logger.info("🔍 Trying EasyOCR (primary method)...")
                easyocr_text = self.ocr_with_easyocr(plate_img)

                if easyocr_text:
                    # Auto-correct dan format
                    corrected = self.auto_correct_plate(easyocr_text)
                    formatted, format_conf = self.format_indonesian_plate(corrected)

                    if self.is_valid_plate(formatted):
                        logger.info(f"✅ EASYOCR SUCCESS: {formatted} (primary)")
                        return formatted
                    elif len(formatted.replace(' ', '')) >= 3:
                        logger.info(f"⚠️ EASYOCR PARTIAL: {formatted} (confidence: {format_conf:.2f})")
                        # Return EasyOCR result even if not fully valid (biasanya lebih akurat)
                        return formatted

            # ★ STRATEGY 2: FALLBACK TO TESSERACT (kalau EasyOCR gagal/tidak available)
            logger.info("🔄 Trying Tesseract fallback...")
            results = []

            # Preprocessing options - LIGHT preprocessing untuk gambar yang sudah jelas
            preprocessed_images = [
                ('Simple', self.preprocess_simple(plate_img)),
                ('Advanced', self.preprocess_advanced(plate_img)),
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
                            'length': len(formatted_text.replace(' ', '')),
                            'valid': self.is_valid_plate(formatted_text),
                            'format_confidence': format_confidence,
                            'method': f"Tesseract {prep_name} + {psm_name}"
                        })

            # Sort results: valid first, format confidence, then by length
            results.sort(key=lambda x: (not x['valid'], -x['format_confidence'], -x['length']))

            # Log all results
            logger.debug(f"Tesseract tried {len(results)} combinations")
            for i, r in enumerate(results[:3]):
                logger.debug(f"  [{i+1}] {r['text']} (valid={r['valid']}, conf={r['format_confidence']:.2f}, method={r['method']})")

            # Return best Tesseract result
            if results:
                best = results[0]
                if best['valid']:
                    logger.info(f"✅ TESSERACT SUCCESS: {best['text']} (fallback method)")
                    return best['text']
                elif best['length'] >= 3:
                    logger.warning(f"⚠️ TESSERACT PARTIAL: {best['text']} (fallback)")
                    return best['text']

            logger.warning(f"❌ ALL OCR FAILED - No readable text")
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
