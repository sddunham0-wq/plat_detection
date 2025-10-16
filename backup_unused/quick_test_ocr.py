#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK TEST OCR - Test cepat dengan teks sample

Penjelasan SMK: Script ini buat test OCR tanpa perlu kamera
Pakai gambar yang dibuat dari teks langsung
"""

import cv2
import numpy as np
from utils.ocr_processor import OCRProcessor

def create_test_plate_image(text, width=400, height=100):
    """Buat gambar plat nomor test dengan text"""

    # Buat background putih
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Font dan ukuran
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3

    # Hitung ukuran text
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Posisi text di tengah
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Tulis text hitam
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    return img

def test_ocr():
    """Test OCR dengan berbagai sample text"""

    print("="*80)
    print("🔍 QUICK TEST OCR - Tanpa Kamera")
    print("="*80)

    # Initialize OCR
    ocr = OCRProcessor()
    print("✅ OCR Processor initialized\n")

    # Test cases - format plat Indonesia
    test_plates = [
        'B 1234 ABC',   # Jakarta
        'D 5678 XYZ',   # Bandung
        'F 9999 AAA',   # Bogor
        'L 1111 BBB',   # Surabaya
        'AA 2222 CC',   # 2 huruf depan
        'B 123 AB',     # Angka pendek
    ]

    print("📋 TEST CASES:")
    print("-" * 80)

    success_count = 0
    for i, plate_text in enumerate(test_plates, 1):
        print(f"\n[{i}/{len(test_plates)}] Testing: '{plate_text}'")
        print("-" * 80)

        # Buat gambar test
        img = create_test_plate_image(plate_text)
        print(f"📐 Image size: {img.shape[1]}x{img.shape[0]}")

        # OCR
        result_text, confidence = ocr.read_plate_with_confidence(img)

        # Expected (cleaned)
        expected = plate_text.replace(' ', '').upper()

        if result_text:
            is_correct = result_text == expected
            status = "✅ CORRECT" if is_correct else "⚠️ PARTIAL"

            print(f"{status}")
            print(f"   Expected : {expected}")
            print(f"   Got      : {result_text}")
            print(f"   Confidence: {confidence:.2f}")

            if is_correct:
                success_count += 1
        else:
            print(f"❌ FAILED - No text detected")

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Total tests: {len(test_plates)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {len(test_plates) - success_count}")
    print(f"Success rate: {success_count/len(test_plates)*100:.1f}%")
    print("="*80)

    if success_count == len(test_plates):
        print("\n🎉 ALL TESTS PASSED! OCR working perfectly!")
    elif success_count > 0:
        print("\n⚠️ SOME TESTS PASSED - OCR working but needs improvement")
    else:
        print("\n❌ ALL TESTS FAILED - Check tesseract installation")
        print("Run: brew install tesseract")

if __name__ == '__main__':
    test_ocr()
