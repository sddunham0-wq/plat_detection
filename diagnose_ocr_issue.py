#!/usr/bin/env python3
"""
Diagnose OCR Issue
Analisis kenapa OCR masih baca teks ngaco
"""

import cv2
import os
import sys
from utils.ocr_processor import OCRProcessor

def analyze_recent_crops():
    """Analisis crop terakhir untuk diagnosis"""

    ocr = OCRProcessor()
    crop_dir = "gambarplat"

    # Get 5 crop terakhir
    crops = sorted([f for f in os.listdir(crop_dir) if f.startswith('crop_')],
                   reverse=True)[:5]

    print("=" * 80)
    print("🔍 DIAGNOSIS OCR ISSUE - Analisis 5 Crop Terakhir")
    print("=" * 80)

    for idx, crop_file in enumerate(crops, 1):
        crop_path = os.path.join(crop_dir, crop_file)

        print(f"\n{'='*80}")
        print(f"📷 CROP {idx}: {crop_file}")
        print(f"{'='*80}")

        # Load image
        img = cv2.imread(crop_path)
        if img is None:
            print(f"❌ Error loading {crop_file}")
            continue

        h, w = img.shape[:2]
        print(f"📐 Size: {w}x{h} pixels (area: {w*h} px²)")
        print(f"   Aspect ratio: {w/h:.2f}")

        # Crop 65% bagian atas (seperti di real_plate_detection)
        upper_height = int(h * 0.65)
        roi_upper = img[:upper_height, :]

        print(f"✂️  Upper 65%: {roi_upper.shape[1]}x{roi_upper.shape[0]} pixels")

        # Test OCR dengan confidence
        text, conf = ocr.read_plate_with_confidence(roi_upper)

        print(f"\n📝 OCR RESULT:")
        print(f"   Text: '{text}'")
        print(f"   Confidence: {conf:.2%}")
        print(f"   Valid format: {ocr.is_valid_plate(text)}")

        # Check validation
        if text:
            is_valid = ocr.is_valid_plate(text)
            meets_threshold = conf >= 0.45

            print(f"\n✅ VALIDATION:")
            print(f"   Format valid: {is_valid}")
            print(f"   Meets threshold (0.45): {meets_threshold}")
            print(f"   Should PASS: {is_valid and meets_threshold}")

            if not is_valid:
                print(f"   ⚠️  PROBLEM: Invalid format - '{text}' bukan plat Indonesia")
            elif not meets_threshold:
                print(f"   ⚠️  PROBLEM: Confidence {conf:.2%} < 0.45")
            else:
                print(f"   ✅ PASS - Plat valid!")
        else:
            print(f"   ❌ OCR returned empty text")

        # Test fallback OCR
        print(f"\n🔄 FALLBACK OCR (Tesseract):")
        fallback_text = ocr.read_plate_fallback(roi_upper)
        print(f"   Text: '{fallback_text}'")
        print(f"   Valid: {ocr.is_valid_plate(fallback_text)}")

    print(f"\n{'='*80}")
    print("📊 SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")

    # Recommendations
    print("\n🔍 POSSIBLE ISSUES:")
    print("1. YOLO mendeteksi area yang BUKAN plat (false positive)")
    print("2. Crop terlalu kecil atau blur untuk OCR")
    print("3. Preprocessing belum optimal untuk kondisi lighting")
    print("4. Format validation terlalu permisif (membiarkan garbage text)")

    print("\n💡 SOLUTIONS:")
    print("1. Naikkan YOLO confidence threshold dari 0.35 ke 0.45-0.50")
    print("2. Tambah minimum size requirement untuk crop")
    print("3. Improve preprocessing (contrast, sharpening)")
    print("4. Tambah stricter validation untuk OCR result")
    print("5. Reject text dengan confidence < 0.50 (instead of 0.45)")

if __name__ == "__main__":
    try:
        analyze_recent_crops()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
