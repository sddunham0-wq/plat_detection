# -*- coding: utf-8 -*-
"""
PLATE VALIDATOR SEDERHANA
Validasi format plat nomor Indonesia
"""

import re

class SimplePlateValidator:
    """Validator sederhana untuk plat Indonesia"""

    def __init__(self):
        # Pattern plat Indonesia
        self.patterns = [
            r'^[A-Z]{1,2}\s\d{1,4}\s[A-Z]{1,3}$',   # F 1234 ABC (dengan spasi)
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',       # F1234ABC (tanpa spasi)
        ]

        # Kode wilayah valid (simplified)
        self.valid_codes = {
            # Jakarta
            'B',
            # Bandung
            'D',
            # Bogor
            'F',
            # Bekasi
            'T',
            # Tangerang
            'A', 'AA', 'AB',
            # dll (add more as needed)
        }

    def is_valid_format(self, plate_text):
        """Cek format plat Indonesia"""
        if not plate_text:
            return False

        # Uppercase dan clean
        text = plate_text.upper().strip()

        # Cek pattern
        for pattern in self.patterns:
            if re.match(pattern, text):
                return True

        return False

    def is_valid_area_code(self, plate_text):
        """Cek kode wilayah valid"""
        if not plate_text:
            return False

        # Extract area code (huruf di awal)
        match = re.match(r'^([A-Z]{1,2})', plate_text)
        if match:
            area_code = match.group(1)
            # Allow all jika valid_codes kosong, atau cek dari list
            return True  # Simplified: allow semua
        return False

    def clean_text(self, text):
        """Bersihkan teks"""
        if not text:
            return ""

        # Remove karakter aneh
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().upper()
