#!/usr/bin/env python3

import cv2
import numpy as np
import pytesseract
import sys
import re

def detect_plate_candidates(image):
    """Improved contour detection focused on license plate characteristics"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    # Multiple edge detection approaches
    # 1. Standard Canny
    edged1 = cv2.Canny(bilateral, 30, 200)

    # 2. More sensitive Canny for small details
    edged2 = cv2.Canny(bilateral, 50, 150)

    # 3. Apply morphological operations to connect characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    edged_morph = cv2.morphologyEx(edged2, cv2.MORPH_CLOSE, kernel)

    candidates = []

    # Try both edge detection methods
    for edges, method in [(edged1, "standard"), (edged_morph, "morphological")]:
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Indonesian license plate characteristics:
            # - Width: typically 30-50cm (in image: varies by distance)
            # - Height: typically 10-15cm
            # - Aspect ratio: 2.5-4.5
            # - Minimum readable size in pixels

            # Focus on smaller, more specific regions
            if 500 <= area <= 8000:  # Smaller area range for actual plates
                aspect_ratio = w / h if h > 0 else 0

                # Stricter aspect ratio for license plates
                if 2.2 <= aspect_ratio <= 4.8:
                    # Check if region has reasonable dimensions
                    if 40 <= w <= 200 and 15 <= h <= 80:
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'method': method
                        })

    # Remove duplicates (similar bounding boxes)
    unique_candidates = []
    for candidate in candidates:
        x1, y1, w1, h1 = candidate['bbox']
        is_duplicate = False

        for existing in unique_candidates:
            x2, y2, w2, h2 = existing['bbox']

            # Check overlap
            overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y

            # If significant overlap, it's a duplicate
            if overlap_area > 0.5 * min(candidate['area'], existing['area']):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_candidates.append(candidate)

    return unique_candidates

def enhance_for_ocr_v2(roi):
    """Enhanced preprocessing specifically for license plates"""
    if roi is None or roi.size == 0:
        return None

    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    h, w = gray.shape
    print(f"    📐 Original ROI size: {w}x{h}")

    # Special handling for very small ROIs
    if h < 20 or w < 60:
        # Aggressive upscaling for very small plates
        scale_factor = max(3, 40 // h)  # Scale to at least 40px height
        target_height = h * scale_factor
        target_width = w * scale_factor
    else:
        # Standard upscaling
        target_height = max(40, h * 3)
        target_width = max(120, w * 3)

    # Initial upscaling with high quality interpolation
    upscaled = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    print(f"    🔍 Upscaled to: {target_width}x{target_height}")

    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(upscaled, 9, 75, 75)

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(bilateral)

    # Try multiple threshold approaches
    # 1. OTSU threshold
    _, thresh_otsu = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. Adaptive threshold
    thresh_adaptive = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 3. Try inverted if original doesn't work well
    _, thresh_otsu_inv = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return {
        'otsu': thresh_otsu,
        'adaptive': thresh_adaptive,
        'otsu_inv': thresh_otsu_inv,
        'original': clahe_img
    }

def test_multiple_ocr(enhanced_images, roi_index):
    """Test OCR on multiple enhanced versions"""
    all_results = []

    for method, img in enhanced_images.items():
        if img is None:
            continue

        print(f"      Testing {method} enhancement:")

        # PSM modes optimized for license plates
        psm_modes = [7, 8, 13]  # Single line, single word, raw line

        for psm in psm_modes:
            try:
                config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(img, config=config).strip()

                if text and len(text) >= 2:
                    # Get confidence
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                    # Check if it looks like Indonesian plate pattern
                    plate_pattern = re.match(r'^[A-Z]\s*\d+\s*[A-Z]{1,3}$', text.replace(' ', ''))
                    is_plate_like = plate_pattern is not None

                    result = {
                        'text': text,
                        'method': method,
                        'psm': psm,
                        'confidence': avg_confidence,
                        'char_count': len(text.replace(' ', '')),
                        'is_plate_like': is_plate_like
                    }

                    all_results.append(result)
                    plate_indicator = "🎯" if is_plate_like else ""
                    print(f"        PSM {psm}: '{text}' (conf: {avg_confidence:.1f}%) {plate_indicator}")

                    # Save the enhanced image if it produced good results
                    if avg_confidence > 30 or is_plate_like:
                        cv2.imwrite(f'contoh/candidate_{roi_index}_{method}_psm{psm}.jpg', img)

            except Exception as e:
                print(f"        PSM {psm}: Error - {e}")

    return all_results

def debug_targeted_detection(image_path):
    """Debug detection with focus on actual license plate regions"""
    print(f"🎯 Targeted license plate detection for: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return

    # Get targeted candidates
    candidates = detect_plate_candidates(image)
    print(f"✅ Found {len(candidates)} targeted candidates")

    # Create debug image
    debug_image = image.copy()

    # Sort by likelihood (smaller area, better aspect ratio)
    candidates.sort(key=lambda x: abs(x['aspect_ratio'] - 3.5) + (x['area'] / 10000))

    all_results = []

    for i, candidate in enumerate(candidates[:8]):  # Test top 8
        x, y, w, h = candidate['bbox']
        print(f"\n  📋 Candidate {i+1}: bbox=({x},{y},{w},{h}) area={candidate['area']} AR={candidate['aspect_ratio']:.2f} method={candidate['method']}")

        # Draw candidate on debug image
        color = (0, 255, 0) if candidate['method'] == 'standard' else (255, 0, 0)
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(debug_image, f"#{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Extract and process ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        cv2.imwrite(f'contoh/targeted_candidate_{i}_roi.jpg', roi)

        # Enhanced OCR processing
        enhanced_images = enhance_for_ocr_v2(roi)
        if enhanced_images:
            ocr_results = test_multiple_ocr(enhanced_images, i)

            if ocr_results:
                # Find best result
                best = max(ocr_results, key=lambda x: (x['is_plate_like'] * 100) + x['confidence'])
                all_results.append({
                    'candidate': i+1,
                    'bbox': candidate['bbox'],
                    'best_result': best,
                    'all_results': ocr_results
                })

                indicator = "🎯 PLATE-LIKE!" if best['is_plate_like'] else "✅"
                print(f"    {indicator} Best: '{best['text']}' ({best['method']}, PSM{best['psm']}, conf: {best['confidence']:.1f}%)")

    # Save debug image
    cv2.imwrite('contoh/targeted_debug.jpg', debug_image)
    print(f"\n💾 Debug image saved: contoh/targeted_debug.jpg")

    # Summary
    print(f"\n📊 Targeted Detection Summary:")
    plate_candidates = [r for r in all_results if r['best_result']['is_plate_like']]

    if plate_candidates:
        print(f"  🎯 Found {len(plate_candidates)} PLATE-LIKE candidates:")
        for result in plate_candidates:
            x, y, w, h = result['bbox']
            best = result['best_result']
            print(f"    Candidate {result['candidate']}: '{best['text']}' at ({x},{y},{w},{h}) conf={best['confidence']:.1f}%")
    else:
        print(f"  Found text in {len(all_results)} candidates, but none match plate patterns")
        if all_results:
            print("  Best text results:")
            for result in all_results[:3]:
                best = result['best_result']
                print(f"    Candidate {result['candidate']}: '{best['text']}' (conf: {best['confidence']:.1f}%)")

    return all_results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 debug_targeted_detection.py <image_path>")
        sys.exit(1)

    debug_targeted_detection(sys.argv[1])