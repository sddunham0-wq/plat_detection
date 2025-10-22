#!/usr/bin/env python3
"""
Test script untuk verify bug fixes di OCR processor

Penjelasan SMK: Script ini test apakah perbaikan kita berhasil!
"""

import sys
sys.path.append('.')

from utils.ocr_processor import OCRProcessor

def test_whitelist():
    """
    Test Bug #1: Whitelist harus ada spasi

    Penjelasan: Whitelist adalah "daftar karakter yang boleh lewat".
    Kalau spasi tidak ada, Tesseract akan buang semua spasi!
    """
    print("=" * 60)
    print("TEST #1: WHITELIST CHECK")
    print("=" * 60)

    ocr = OCRProcessor()

    print(f"Whitelist: '{ocr.char_whitelist}'")
    print(f"Length: {len(ocr.char_whitelist)}")

    has_space = ' ' in ocr.char_whitelist
    print(f"Has space: {has_space}")

    if has_space:
        print("✅ PASS: Whitelist contains space")
        print("   → Tesseract tidak akan filter spasi!\n")
        return True
    else:
        print("❌ FAIL: Whitelist missing space!")
        print("   → Tesseract akan buang semua spasi!\n")
        return False

def test_auto_correction():
    """
    Test Bug #2: Auto-correction tidak boleh ubah angka di nomor seri

    Penjelasan: Function ini harus pintar membedakan:
    - Angka di posisi HURUF depan → dikoreksi (contoh: 8 → B)
    - Angka di posisi NOMOR seri → TIDAK dikoreksi (contoh: 0 tetap 0)
    - Angka di posisi HURUF belakang → dikoreksi (contoh: 8 → B)
    """
    print("=" * 60)
    print("TEST #2: AUTO-CORRECTION LOGIC")
    print("=" * 60)

    ocr = OCRProcessor()

    test_cases = [
        # (input, expected_output, description)
        ("B0123ABC", "B0123ABC", "Angka 0 di nomor TIDAK diubah"),
        ("B5678ABC", "B5678ABC", "Angka 5 di nomor TIDAK diubah"),
        ("81234ABC", "B1234ABC", "Angka 8 di huruf depan diubah jadi B"),
        ("B1234A8C", "B1234ABC", "Angka 8 di huruf belakang diubah jadi B"),
        ("0123ABC", "O123ABC", "Angka 0 di huruf depan diubah jadi O"),
        ("AA1234BB", "AA1234BB", "Format 2 huruf depan tidak berubah"),
        ("B 0123 ABC", "B0123ABC", "Spasi di-remove, angka 0 tetap 0"),
    ]

    all_pass = True
    for input_text, expected, description in test_cases:
        result = ocr.auto_correct_plate(input_text)
        passed = result == expected
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"{status}: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Output:   '{result}'")
        print(f"  Expected: '{expected}'")

        if not passed:
            print(f"  ⚠️  MISMATCH!")
            all_pass = False
        print()

    return all_pass

def test_validation():
    """
    Test Bug #3: Validation harus reject plat terlalu pendek

    Penjelasan: Plat Indonesia minimal 3 digit nomor.
    Kalau cuma 1-2 digit, kemungkinan besar OCR error!
    """
    print("=" * 60)
    print("TEST #3: FORMAT VALIDATION")
    print("=" * 60)

    ocr = OCRProcessor()

    test_cases = [
        # (input, should_be_valid, description)
        ("A 1 B", False, "Plat 1 digit harus INVALID"),
        ("B 12 AB", False, "Plat 2 digit harus INVALID"),
        ("B 123 AB", True, "Plat 3 digit harus VALID"),
        ("B 1234 ABC", True, "Plat 4 digit harus VALID"),
        ("AA 123 BB", True, "Format AA + 3 digit harus VALID"),
        ("AA 1234 B", True, "Format AA + 4 digit harus VALID"),
    ]

    all_pass = True
    for input_text, should_be_valid, description in test_cases:
        is_valid = ocr.is_valid_plate(input_text)
        passed = is_valid == should_be_valid
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"{status}: {description}")
        print(f"  Input:    '{input_text}'")
        print(f"  Valid:    {is_valid}")
        print(f"  Expected: {should_be_valid}")

        if not passed:
            print(f"  ⚠️  MISMATCH!")
            all_pass = False
        print()

    return all_pass

def test_format_indonesian():
    """
    Test format_indonesian_plate() function

    Penjelasan: Function ini harus bisa format plat dengan spasi yang benar
    """
    print("=" * 60)
    print("TEST #4: FORMAT INDONESIAN PLATE")
    print("=" * 60)

    ocr = OCRProcessor()

    test_cases = [
        # (input, expected_output, min_confidence)
        ("B1234ABC", "B 1234 ABC", 0.5),
        ("B 1234 ABC", "B 1234 ABC", 0.5),
        ("AA1234BB", "AA 1234 BB", 0.5),
        ("F123AB", "F 123 AB", 0.5),
    ]

    all_pass = True
    for input_text, expected, min_conf in test_cases:
        formatted, confidence = ocr.format_indonesian_plate(input_text)
        passed = (formatted == expected) and (confidence >= min_conf)
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"{status}: Format '{input_text}'")
        print(f"  Output:     '{formatted}'")
        print(f"  Expected:   '{expected}'")
        print(f"  Confidence: {confidence:.2f} (min: {min_conf})")

        if not passed:
            print(f"  ⚠️  MISMATCH!")
            all_pass = False
        print()

    return all_pass

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "OCR PROCESSOR BUG FIXES - TEST SUITE" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    results = []

    # Test Bug #1
    results.append(("Whitelist", test_whitelist()))

    # Test Bug #2
    results.append(("Auto-Correction", test_auto_correction()))

    # Test Bug #3
    results.append(("Validation", test_validation()))

    # Test Bonus
    results.append(("Format Indonesian", test_format_indonesian()))

    # Summary
    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_pass = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    print()
    if all_pass:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Semua bug sudah diperbaiki dengan benar!")
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  Ada bug yang belum terperbaiki, cek output di atas!")
    print()

    return 0 if all_pass else 1

if __name__ == '__main__':
    exit(main())
