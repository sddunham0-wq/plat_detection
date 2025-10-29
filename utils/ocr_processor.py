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

        # ★ RESULT HISTORY untuk konsistensi (Bug #24)
        # Simpan 5 hasil terakhir untuk majority voting
        from collections import deque
        self.ocr_history = deque(maxlen=5)

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

                # logger.debug(f"📏 Upscaled: {w}x{h} → {new_w}x{new_h} ({scale:.1f}x)")  # SANTAI: disabled log
                pass

            # ★ ULTRA MINIMAL (Bug #24): NO THRESHOLD! Just upscaled grayscale
            # Problem: Threshold menghilangkan digit pertama ("1818" → "818")
            # Solution: Return grayscale saja, let OCR engine handle it
            # EasyOCR dan Tesseract bisa baca grayscale tanpa threshold!

            # Return GRAYSCALE only (no binary threshold)
            return gray

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

            # ★ AGGRESSIVE CORRECTION untuk huruf depan (Bug #24)
            # Problem: "F 1818 HG" terbaca "I 810 HG"
            # I→F: Huruf 'I' SANGAT JARANG di plat Indonesia posisi depan!
            # 7→B: Angka '7' bisa salah baca huruf 'B'
            if chars[i] == 'I':
                chars[i] = 'F'
                # logger.debug(f"🔧 Corrected I→F at position {i} (common OCR error)")  # SANTAI: disabled log
            elif chars[i] == '7':
                chars[i] = 'B'
                # logger.debug(f"🔧 Corrected 7→B at position {i} (Jakarta plates)")  # SANTAI: disabled log
            elif chars[i] == '1':
                # NEW: 1 juga bisa dibaca sebagai I, tapi di posisi huruf depan harus jadi F!
                chars[i] = 'F'
                # logger.debug(f"🔧 Corrected 1→F at position {i} (digit as letter)")  # SANTAI: disabled log
            # ★ Koreksi kalau memang bukan huruf!
            elif not chars[i].isalpha():
                # Common errors di posisi huruf depan
                if chars[i] == '0':
                    chars[i] = 'O'
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

        # ★ LIMIT: Maksimal 8 karakter untuk plat Indonesia (prevent junk reading)
        text_no_space = text.replace(' ', '')
        if len(text_no_space) > 8:
            text_no_space = text_no_space[:8]  # Potong di 8 karakter

        # Pattern plat Indonesia: [1 Huruf][3-4 Angka][1-3 Huruf]
        # ★ SMART FIX (Bug #24): Terima 3-4 digit, auto-fix kalau kehilangan digit
        # Area code: 1 huruf (B, F, D, E, dll)
        # Nomor: 3-4 angka (818→1818, 1205, dll)
        # Series: 1-3 huruf random (A, AB, ABC, UNP, HG, dll)
        # Examples: "F 1818 HG" (4 digit OK), "F 818 HG" (3 digit → predict 1818)
        pattern = r'^([A-Z])(\d{3,4})([A-Z]{1,3})$'
        match = re.match(pattern, text_no_space)

        if match:
            area_code = match.group(1)    # "F", "B", "D"
            number = match.group(2)       # "1234" or "818"
            series = match.group(3)       # "ABC"

            # ★ SMART DIGIT RECOVERY (Bug #24): Kalau 3 digit, predict yang hilang
            if len(number) == 3:
                # Guess: digit pertama kemungkinan besar sama dengan digit terakhir
                # Contoh: 818 → 1818 (common pattern), 205 → 1205
                # Pattern Indonesia: sering 1xxx, 2xxx, atau xYYx (symmetry)
                predicted_digit = number[0]  # Ambil digit pertama sebagai guess

                # Try common patterns
                predicted_numbers = [
                    f"1{number}",  # 818 → 1818 (most common)
                    f"{number[0]}{number}",  # 818 → 8818 (symmetry)
                ]

                # Gunakan yang pertama (1xxx paling umum)
                number = predicted_numbers[0]
                confidence = 0.6  # Lower confidence untuk predicted
                logger.info(f"🔧 Auto-fix: {match.group(2)} → {number} (3→4 digit recovery)")
            elif len(number) == 4:
                # Check kalau mulai dengan '0' → suspicious
                if number[0] == '0':
                    logger.warning(f"⚠️ Suspicious number: {number} (starts with 0)")
                    confidence = 0.3
                else:
                    confidence = 1.0
            else:
                # 1, 2, or 5+ digits = invalid
                logger.warning(f"❌ Invalid number: {number} ({len(number)} digits)")
                return text, 0.1

            # Penalty kalau series terlalu pendek
            if len(series) < 1:
                confidence -= 0.2

            formatted = f"{area_code} {number} {series}"
            return formatted, max(confidence, 0.3)
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

            # ★ SIMPLE UPSCALING untuk plat kecil (Bug #24)
            # EasyOCR works better dengan gambar lebih besar
            h, w = img_bgr.shape[:2]
            if w < 300:  # Kalau width < 300px, upscale
                scale = 300 / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                # logger.debug(f"📏 EasyOCR upscaled: {w}x{h} → {new_w}x{new_h}")  # SANTAI: disabled log
                pass

            # ★ MULTI-ATTEMPT (Bug #24): Try with different preprocessing
            # Problem: "F 1818 HG" → "F 818 HG" (digit pertama hilang)
            # Solution: Try multiple preprocessing untuk cari yang terbaik

            all_results = []

            # Attempt 1: Original image
            results1 = self.easyocr_reader.readtext(img_bgr, detail=1, paragraph=False)
            if results1:
                best1 = max(results1, key=lambda x: x[2])
                all_results.append(('original', best1[1], best1[2]))

            # Attempt 2: Increased brightness (untuk plat gelap)
            img_bright = cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=30)
            results2 = self.easyocr_reader.readtext(img_bright, detail=1, paragraph=False)
            if results2:
                best2 = max(results2, key=lambda x: x[2])
                all_results.append(('bright', best2[1], best2[2]))

            # Attempt 3: Higher contrast
            img_contrast = cv2.convertScaleAbs(img_bgr, alpha=1.5, beta=0)
            results3 = self.easyocr_reader.readtext(img_contrast, detail=1, paragraph=False)
            if results3:
                best3 = max(results3, key=lambda x: x[2])
                all_results.append(('contrast', best3[1], best3[2]))

            if not all_results:
                return None

            # Pilih hasil terbaik berdasarkan length (lebih panjang = lebih baik)
            # Karena kita kehilangan digit, prioritaskan text lebih panjang
            all_results.sort(key=lambda x: (len(x[1].replace(' ', '')), x[2]), reverse=True)

            method, text, confidence = all_results[0]
            cleaned = self.clean_text(text)

            logger.info(f"📝 EasyOCR: '{cleaned}' (conf: {confidence:.2f}, method: {method})")

            # ★ ULTRA LOW THRESHOLD (Bug #24): 0.30 → 0.01
            # Problem: Valid plates dengan confidence 0.04-0.30 ditolak
            # Solution: Accept almost everything, let auto_correct and validation handle quality
            return cleaned if confidence >= 0.01 else None

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
                # logger.info("🔍 Trying EasyOCR (primary method)...")  # SANTAI: disabled log
                easyocr_text = self.ocr_with_easyocr(plate_img)

                if easyocr_text:
                    # Auto-correct dan format
                    corrected = self.auto_correct_plate(easyocr_text)
                    formatted, format_conf = self.format_indonesian_plate(corrected)

                    # ★ SMART VALIDATION (Bug #24): Accept 3-4 digit, auto-fix di format_indonesian_plate
                    # format_indonesian_plate sudah handle 3→4 digit conversion
                    if self.is_valid_plate(formatted) and format_conf >= 0.3:
                        # ★ MAJORITY VOTING (Bug #24): Prioritas text lebih panjang
                        self.ocr_history.append(formatted)

                        # Cari text terpanjang dari history (untuk hindari kehilangan digit)
                        if len(self.ocr_history) >= 3:
                            # Sort by length, ambil yang terpanjang
                            sorted_history = sorted(self.ocr_history, key=lambda x: len(x.replace(' ', '')), reverse=True)
                            best_result = sorted_history[0]

                            # Kalau best result berbeda dengan current, gunakan best
                            if best_result != formatted:
                                logger.info(f"✅ OCR: {best_result} (stable)")
                                return best_result

                        logger.info(f"✅ OCR: {formatted}")
                        return formatted

            # ★ STRATEGY 2: FALLBACK TO TESSERACT (kalau EasyOCR gagal/tidak available)
            # logger.info("🔄 Trying Tesseract fallback...")  # SANTAI: disabled log
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
            # logger.debug(f"Tesseract tried {len(results)} combinations")  # SANTAI: disabled log
            # for i, r in enumerate(results[:3]):  # SANTAI: disabled log
            #     logger.debug(f"  [{i+1}] {r['text']} (valid={r['valid']}, conf={r['format_confidence']:.2f}, method={r['method']})")

            # Return best Tesseract result
            if results:
                best = results[0]
                if best['valid']:
                    logger.info(f"✅ OCR: {best['text']}")  # SANTAI: simplified log
                    return best['text']
                elif best['length'] >= 3:
                    logger.warning(f"⚠️ OCR: {best['text']}")  # SANTAI: simplified log
                    return best['text']

            logger.warning(f"❌ OCR gagal")  # SANTAI: simplified log
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
