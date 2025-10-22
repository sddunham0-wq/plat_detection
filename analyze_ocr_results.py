#!/usr/bin/env python3
"""
Analyze OCR results from recent crops
"""

import os
import cv2
from utils.ocr_processor import OCRProcessor
import glob

def analyze_recent_crops():
    """Analyze OCR performance on recent crops"""

    ocr = OCRProcessor()

    # Get recent crop files
    crop_files = sorted(glob.glob('gambarplat/crop_*.jpg'), reverse=True)[:10]

    print("=" * 70)
    print("OCR ANALYSIS - Recent Crops")
    print("=" * 70)
    print()

    results = []

    for i, img_path in enumerate(crop_files, 1):
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            # Try OCR
            text, conf = ocr.read_plate_with_confidence(img)

            # Check validation
            is_valid = ocr.is_valid_plate(text) if text else False

            results.append({
                'file': os.path.basename(img_path),
                'size': f'{w}x{h}',
                'area': w * h,
                'text': text or "FAILED",
                'confidence': conf,
                'valid': is_valid
            })

            print(f"{i}. {os.path.basename(img_path)}")
            print(f"   Size: {w}x{h} ({w*h} pixels)")
            print(f"   OCR:  '{text}' (conf: {conf:.2f})")
            print(f"   Valid: {'✅' if is_valid else '❌'} {is_valid}")
            print()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

    # Statistics
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)

    total = len(results)
    successful = sum(1 for r in results if r['text'] != "FAILED")
    valid_plates = sum(1 for r in results if r['valid'])

    print(f"Total tested: {total}")
    print(f"OCR successful: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"Valid plates: {valid_plates}/{total} ({valid_plates/total*100:.1f}%)")

    if successful > 0:
        avg_conf = sum(r['confidence'] for r in results if r['text'] != "FAILED") / successful
        print(f"Average confidence: {avg_conf:.2f}")

    print()
    print("=" * 70)
    print("COMMON ISSUES")
    print("=" * 70)

    # Analyze failures
    failed = [r for r in results if not r['valid']]
    if failed:
        print(f"❌ {len(failed)} plates failed validation:")
        for r in failed[:5]:
            print(f"   • {r['file']}: '{r['text']}' (conf: {r['confidence']:.2f})")
    else:
        print("✅ All plates passed validation!")

    print()

if __name__ == '__main__':
    analyze_recent_crops()
