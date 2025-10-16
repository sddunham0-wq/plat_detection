import re
from typing import Tuple, Dict, List

class IndonesianPlateValidator:
    """
    Validator untuk format plat nomor Indonesia
    Membantu mengurangi false positive dengan validasi format yang tepat
    """

    def __init__(self):
        # Pola regex untuk berbagai format plat Indonesia
        # Supports both "F 1234 ABC" (with spaces) dan "F1234ABC" (no spaces)
        self.patterns = [
            r'^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$',     # Format dengan spasi: F 1234 ABC
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',         # Format tanpa spasi: F1234ABC
            r'^[A-Z]{2}\s\d{1,4}\s[A-Z]{1,2}$',       # Format 2 huruf dengan spasi: AB 1234 CD
            r'^[A-Z]{2}\d{1,4}[A-Z]{1,2}$',           # Format 2 huruf tanpa spasi: AB1234CD
            r'^[A-Z]\s\d{1,4}\s[A-Z]{1,3}$',          # Format 1 huruf dengan spasi: F 1234 ABC
            r'^[A-Z]\d{1,4}[A-Z]{1,3}$'               # Format 1 huruf tanpa spasi: F1234ABC
        ]

        # Daftar kode wilayah valid di Indonesia
        self.valid_area_codes = {
            # Sumatera
            'BL', 'BB', 'BN', 'BA', 'BK', 'BM', 'BP', 'BG', 'BU', 'BD',
            # Jawa
            'B', 'D', 'F', 'E', 'H', 'K', 'T', 'G', 'L', 'M', 'N', 'P', 'R', 'S', 'W', 'AA', 'AB', 'AD', 'AE', 'AG',
            # Kalimantan
            'DA', 'KB', 'KT', 'KU', 'KH',
            # Sulawesi
            'DD', 'DT', 'DN', 'DL', 'DM', 'DR', 'DC',
            # Bali & Nusa Tenggara
            'DK', 'EA', 'EB', 'ED',
            # Maluku & Papua
            'DE', 'DG', 'PA', 'PB'
        }

    def clean_text(self, text: str, normalize_spaces: bool = True) -> str:
        """
        Bersihkan teks hasil OCR dari karakter yang tidak perlu

        Args:
            text: Raw OCR text
            normalize_spaces: True = normalize ke format standar (F 1234 ABC)
                            False = keep original spacing
        """
        if not text:
            return ""

        # Hapus karakter khusus kecuali alphanumeric dan space
        text = re.sub(r'[^\w\s]', '', text)

        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().upper()

        # Normalize ke format standar Indonesia jika diminta
        if normalize_spaces:
            # Pattern: [huruf][angka][huruf] → format dengan spasi
            pattern = r'^([A-Z]{1,2})\s*(\d{1,4})\s*([A-Z]{1,3})$'
            match = re.match(pattern, text)
            if match:
                area = match.group(1)
                number = match.group(2)
                series = match.group(3)
                text = f"{area} {number} {series}"

        return text

    def is_valid_format(self, plate_text: str) -> bool:
        """
        Cek apakah format plat nomor sesuai standar Indonesia
        """
        cleaned_text = self.clean_text(plate_text)

        if not cleaned_text:
            return False

        # Cek dengan semua pola yang valid
        for pattern in self.patterns:
            if re.match(pattern, cleaned_text):
                return True

        return False

    def is_valid_area_code(self, plate_text: str) -> bool:
        """
        Cek apakah kode wilayah valid
        """
        cleaned_text = self.clean_text(plate_text)

        if not cleaned_text:
            return False

        # Ekstrak kode wilayah (huruf di awal)
        area_match = re.match(r'^([A-Z]{1,2})', cleaned_text)
        if area_match:
            area_code = area_match.group(1)
            return area_code in self.valid_area_codes

        return False

    def calculate_confidence(self, plate_text: str, detection_confidence: float = 1.0) -> float:
        """
        Hitung confidence score berdasarkan berbagai faktor
        """
        cleaned_text = self.clean_text(plate_text)

        if not cleaned_text:
            return 0.0

        confidence_factors = []

        # Faktor 1: Format yang benar (40%)
        if self.is_valid_format(cleaned_text):
            confidence_factors.append(0.4)
        else:
            confidence_factors.append(0.0)

        # Faktor 2: Kode wilayah valid (30%)
        if self.is_valid_area_code(cleaned_text):
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.0)

        # Faktor 3: Panjang teks reasonable (15%)
        text_length = len(cleaned_text.replace(' ', ''))
        if 5 <= text_length <= 10:  # Plat Indonesia umumnya 5-10 karakter
            confidence_factors.append(0.15)
        else:
            confidence_factors.append(0.0)

        # Faktor 4: Rasio huruf vs angka (15%)
        letters = len(re.findall(r'[A-Z]', cleaned_text))
        numbers = len(re.findall(r'\d', cleaned_text))
        if letters >= 2 and numbers >= 1:  # Minimal 2 huruf dan 1 angka
            confidence_factors.append(0.15)
        else:
            confidence_factors.append(0.0)

        # Total confidence
        total_confidence = sum(confidence_factors) * detection_confidence

        return min(total_confidence, 1.0)

    def validate_plate(self, plate_text: str, min_confidence: float = 0.6) -> Dict:
        """
        Validasi lengkap plat nomor dengan hasil detail
        """
        cleaned_text = self.clean_text(plate_text)
        confidence = self.calculate_confidence(plate_text)

        result = {
            'original_text': plate_text,
            'cleaned_text': cleaned_text,
            'is_valid': confidence >= min_confidence,
            'confidence': confidence,
            'format_valid': self.is_valid_format(cleaned_text),
            'area_code_valid': self.is_valid_area_code(cleaned_text),
            'details': {
                'length': len(cleaned_text.replace(' ', '')),
                'has_letters': len(re.findall(r'[A-Z]', cleaned_text)) > 0,
                'has_numbers': len(re.findall(r'\d', cleaned_text)) > 0,
                'area_code': re.match(r'^([A-Z]{1,2})', cleaned_text).group(1) if re.match(r'^([A-Z]{1,2})', cleaned_text) else None
            }
        }

        return result

    def get_suggestions(self, plate_text: str) -> List[str]:
        """
        Berikan saran perbaikan untuk teks yang tidak valid
        """
        suggestions = []
        cleaned_text = self.clean_text(plate_text)

        if not cleaned_text:
            suggestions.append("Teks kosong atau tidak terbaca")
            return suggestions

        # Analisis masalah
        if not self.is_valid_format(cleaned_text):
            suggestions.append(f"Format tidak sesuai standar Indonesia. Contoh: B 1234 ABC")

        if not self.is_valid_area_code(cleaned_text):
            area_match = re.match(r'^([A-Z]{1,2})', cleaned_text)
            if area_match:
                suggestions.append(f"Kode wilayah '{area_match.group(1)}' tidak valid")
            else:
                suggestions.append("Kode wilayah tidak terdeteksi")

        length = len(cleaned_text.replace(' ', ''))
        if length < 5:
            suggestions.append("Teks terlalu pendek untuk plat nomor")
        elif length > 10:
            suggestions.append("Teks terlalu panjang untuk plat nomor")

        return suggestions

# Instance global validator
plate_validator = IndonesianPlateValidator()

# Fungsi helper untuk penggunaan mudah
def validate_indonesian_plate(plate_text: str, min_confidence: float = 0.6) -> bool:
    """
    Fungsi sederhana untuk validasi cepat
    """
    return plate_validator.validate_plate(plate_text, min_confidence)['is_valid']

def clean_plate_text(plate_text: str) -> str:
    """
    Fungsi sederhana untuk membersihkan teks
    """
    return plate_validator.clean_text(plate_text)

def get_plate_confidence(plate_text: str) -> float:
    """
    Fungsi sederhana untuk mendapatkan confidence score
    """
    return plate_validator.calculate_confidence(plate_text)