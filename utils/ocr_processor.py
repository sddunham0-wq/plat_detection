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
        # Multiple PSM modes untuk fallback dengan BAHASA INDONESIA
        self.psm_modes = [
            ('PSM 7 - Single Line', '--psm 7 --oem 3 -l ind'),
            ('PSM 8 - Single Word', '--psm 8 --oem 3 -l ind'),
            ('PSM 6 - Block of Text', '--psm 6 --oem 3 -l ind'),
            ('PSM 13 - Raw Line', '--psm 13 --oem 3 -l ind'),
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
            # Penjelasan SMK: Perbesar gambar 8x untuk OCR lebih akurat
            # Target: 120px → 960px (8x), 150px → 720px (4.8x)
            h, w = gray.shape
            target_width = 720  # Increased from 600 to 720 for better OCR quality
            if w < target_width:
                scale = target_width / w
                scale = min(scale, 8.0)  # Max 8x scaling (increased from 6x)
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
            target_width = 720  # Increased from 600 to 720
            if w < target_width:
                scale = target_width / w
                scale = min(scale, 8.0)  # Max 8x scaling (increased from 6x)
                new_w = int(w * scale)
                new_h = int(h * scale)
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                logger.debug(f"📏 Advanced upscaled: {w}x{h} → {new_w}x{new_h} ({scale:.1f}x)")

            # ★ FIX: GENTLE CLAHE untuk preserve text quality
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Lowered from 2.5 to 2.0
            enhanced = clahe.apply(gray)

            # ★ FIX: LIGHT SHARPENING - Preserve text, avoid noise
            # Penjelasan SMK: Kernel ringan untuk edges lebih jelas tanpa over-processing
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1, 7, -1],  # Center lowered from 8 to 7
                                       [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)

            # Bilateral filter - lighter untuk preserve detail
            bilateral = cv2.bilateralFilter(sharpened, 9, 50, 50)  # Lowered from 75,75 to 50,50

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

        FIXED: Remove spasi dulu sebelum koreksi untuk akurasi lebih baik!
        Ini mencegah salah koreksi angka yang ada di nomor seri.
        """
        if not text or len(text) < 3:
            return text

        # ★ PERBAIKAN 1: Remove spasi dulu untuk processing lebih akurat
        # Penjelasan: Kalau ada spasi, indexing jadi ngaco!
        # Contoh: "B 0123 ABC" → ['B', ' ', '0', '1', '2', '3', ...]
        # Angka 0 ada di index 2, bukan di bagian huruf depan!
        text_no_space = text.replace(' ', '')

        if len(text_no_space) < 3:
            return text

        # Convert to list untuk mudah edit
        chars = list(text_no_space)

        # Deteksi pola plat Indonesia: [huruf][angka][huruf]
        # Format: B1234ABC atau AA1234CC

        # ★ PERBAIKAN 2: Deteksi berapa huruf di depan (1 atau 2?)
        # Contoh: "B1234ABC" → prefix = 1, "AA1234BB" → prefix = 2
        prefix_length = 2 if len(chars) > 1 and chars[1].isalpha() else 1

        # PART 1: Huruf DEPAN (1-2 karakter pertama)
        # Kalau ada angka di posisi huruf depan, convert ke huruf
        for i in range(prefix_length):
            if i >= len(chars):
                break

            # ★ PERBAIKAN 3: Hanya koreksi kalau memang bukan huruf!
            # Ini mencegah koreksi huruf yang sudah benar
            if not chars[i].isalpha():
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
        # Cari dimana angka dimulai dan berakhir
        digit_start = -1
        digit_end = -1

        for i in range(prefix_length, len(chars)):
            if chars[i].isdigit():
                if digit_start == -1:
                    digit_start = i
                digit_end = i
            elif digit_start != -1:
                # Sudah ketemu angka, sekarang ketemu huruf lagi (series)
                break

        # PART 3: Huruf BELAKANG (setelah angka)
        # Kalau ada angka di posisi huruf belakang, convert ke huruf
        if digit_end >= 0:
            suffix_start = digit_end + 1

            for i in range(suffix_start, len(chars)):
                # ★ PERBAIKAN 4: Hanya koreksi kalau memang bukan huruf!
                # Kalau sudah huruf, skip
                if not chars[i].isalpha():
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

        if corrected != text_no_space:
            logger.debug(f"Auto-corrected: {text_no_space} → {corrected}")

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
        # ★ FIXED: Minimal 3 digit untuk nomor plat (plat Indonesia standar: 3-4 digit)
        # BEFORE: \d{1,4} → terima "A 1 B" (terlalu lemah!)
        # AFTER: \d{3,4} → hanya terima "B 123 AB" atau "B 1234 ABC" (lebih ketat!)
        pattern = r'^([A-Z]{1,2})(\d{3,4})([A-Z]{1,3})$'
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
        """
        OCR dengan 1 PSM mode menggunakan BAHASA INDONESIA

        FIXED: Remove double upscaling - preprocessing sudah handle upscaling!
        """
        try:
            config = f"{psm_config} -c tessedit_char_whitelist={self.char_whitelist}"

            # Gunakan bahasa Indonesia (ind) untuk Tesseract
            raw_text = pytesseract.image_to_string(img, lang='ind', config=config)
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

            # ★ FIX: Read text dengan EasyOCR (no allowlist - too restrictive)
            # Strategy: Let EasyOCR read freely, then validate dengan is_valid_plate()
            # Whitelist di EasyOCR terlalu strict, bikin OCR gagal baca text yang jelas
            results = self.easyocr_reader.readtext(
                img_bgr,
                detail=1,
                paragraph=False  # Read individual text blocks, not paragraphs
            )

            if not results:
                return None

            # Ambil text dengan confidence tertinggi
            best_result = max(results, key=lambda x: x[2])  # x[2] = confidence
            text = best_result[1]  # x[1] = text
            confidence = best_result[2]  # x[2] = confidence

            # Clean text
            cleaned = self.clean_text(text)

            logger.info(f"📝 EasyOCR RAW: '{text}' → CLEANED: '{cleaned}' (conf: {confidence:.2f})")

            # ★ DEBUGGING: Save preprocessed image untuk analisis
            try:
                import os
                import time
                os.makedirs('debug_ocr', exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                debug_path = f"debug_ocr/easyocr_{cleaned}_{timestamp}.jpg"
                cv2.imwrite(debug_path, img_bgr)
                logger.debug(f"💾 Debug saved: {debug_path}")
            except:
                pass

            # ★ ADJUST: Lower threshold dari 0.6 ke 0.5 - balance between accuracy and recall
            # Rationale: 0.6 too strict, banyak valid plates rejected
            #            0.5 lebih balance, masih reject garbage (<0.3)
            return cleaned if confidence > 0.5 else None

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
                    logger.info(f"TESSERACT SUCCESS: {best['text']} (fallback method)")
                    return best['text']
                elif best['length'] >= 3:
                    logger.warning(f"TESSERACT PARTIAL: {best['text']} (fallback)")
                    return best['text']

            logger.warning(f"ALL OCR FAILED - No readable text")
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

            # Try to get confidence from OCR data menggunakan bahasa Indonesia
            try:
                processed = self.preprocess_advanced(plate_img)
                config = f"--psm 7 --oem 3 -l ind -c tessedit_char_whitelist={self.char_whitelist}"

                ocr_data = pytesseract.image_to_data(
                    processed,
                    lang='ind',
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
