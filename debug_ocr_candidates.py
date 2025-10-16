#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
import sys
import re

def enhance_for_ocr(roi):
    """Enhanced preprocessing for OCR"""
    if roi is None or roi.size == 0:
        return None

    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    h, w = gray.shape
    print(f"    📐 Original ROI size: {w}x{h}")

    # Apply bilateral filter to reduce noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(bilateral)

    # SUPER-RESOLUTION UPSCALING for CCTV images
    target_height = max(40, h * 4)  # Minimum 40px height, 4x upscaling
    target_width = max(120, w * 4)   # Minimum 120px width, 4x upscaling

    upscaled = cv2.resize(clahe_img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    print(f"    🔍 Upscaled to: {target_width}x{target_height}")

    # Apply morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(upscaled, cv2.MORPH_CLOSE, kernel)

    # Apply threshold
    _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def test_ocr_on_roi(enhanced_roi, roi_index):
    """Test multiple OCR strategies on enhanced ROI"""
    if enhanced_roi is None:
        return []

    results = []

    # Multiple PSM modes for different text layouts
    psm_modes = [7, 8, 6, 13]  # Single text line, single word, single block, raw line

    for psm in psm_modes:
        try:
            config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(enhanced_roi, config=config).strip()

            if text and len(text) >= 2:  # At least 2 characters
                # Get confidence data
                data = pytesseract.image_to_data(enhanced_roi, config=config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                results.append({
                    'text': text,
                    'psm': psm,
                    'confidence': avg_confidence,
                    'char_count': len(text)
                })

                print(f"      PSM {psm}: '{text}' (conf: {avg_confidence:.1f}%)")

        except Exception as e:
            print(f"      PSM {psm}: Error - {e}")

    return results

def debug_ocr_candidates(image_path):
    """Debug OCR on detected candidates"""
    print(f"🔍 Testing OCR on candidates from: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return

    # Get candidates using same logic as debug_contour_detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bilateral, 30, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < 100:
            continue

        aspect_ratio = w / h if h > 0 else 0

        if 1.5 <= aspect_ratio <= 6.0 and area >= 100:
            candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })

    # Sort by area (largest first)
    candidates.sort(key=lambda x: x['area'], reverse=True)

    print(f"🎯 Testing OCR on top {min(10, len(candidates))} candidates:")

    all_ocr_results = []

    for i, candidate in enumerate(candidates[:10]):
        x, y, w, h = candidate['bbox']
        print(f"\n  📋 Candidate {i+1}: bbox=({x},{y},{w},{h}) area={candidate['area']} AR={candidate['aspect_ratio']:.2f}")

        # Extract ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            print("    ❌ Empty ROI")
            continue

        # Save original ROI
        cv2.imwrite(f'contoh/candidate_{i}_roi.jpg', roi)

        # Enhance for OCR
        enhanced = enhance_for_ocr(roi)
        if enhanced is not None:
            cv2.imwrite(f'contoh/candidate_{i}_enhanced.jpg', enhanced)

            # Test OCR
            ocr_results = test_ocr_on_roi(enhanced, i)

            if ocr_results:
                # Find best result (highest confidence with reasonable length)
                best = max(ocr_results, key=lambda x: x['confidence'] if x['char_count'] >= 4 else 0)
                all_ocr_results.append({
                    'candidate': i+1,
                    'bbox': candidate['bbox'],
                    'best_text': best['text'],
                    'confidence': best['confidence'],
                    'all_results': ocr_results
                })
                print(f"    ✅ Best: '{best['text']}' (conf: {best['confidence']:.1f}%)")
            else:
                print("    ❌ No readable text found")

    # Summary
    print(f"\n📊 OCR Results Summary:")
    if all_ocr_results:
        print(f"  Found text in {len(all_ocr_results)} candidates:")
        for result in all_ocr_results:
            x, y, w, h = result['bbox']
            print(f"    Candidate {result['candidate']}: '{result['best_text']}' at ({x},{y},{w},{h})")
    else:
        print("  ❌ No readable text found in any candidate")

    return all_ocr_results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 debug_ocr_candidates.py <image_path>")
        sys.exit(1)

    debug_ocr_candidates(sys.argv[1])