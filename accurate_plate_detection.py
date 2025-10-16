#!/usr/bin/env python3
"""
Accurate plate detection untuk kondisi real-world seperti gambar P 2543 BP2
Fokus pada deteksi plat Indonesia dengan berbagai kondisi pencahayaan
"""

import cv2
import numpy as np
import sys
import os
from utils.ocr_ensemble import OCREnsemble

def detect_license_plate_regions(image):
    """Deteksi region plat nomor dengan pendekatan yang lebih fleksibel"""

    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Multiple approaches to find plate regions
    candidates = []

    # Approach 1: Edge-based detection
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Multiple edge detection with different thresholds
    for low_thresh in [50, 75, 100]:
        for high_thresh in [150, 200, 250]:
            edges = cv2.Canny(blurred, low_thresh, high_thresh)

            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 20000:  # Filter by reasonable size
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Indonesian license plates are typically 2.5:1 to 4.5:1 ratio
                if 2.0 <= aspect_ratio <= 5.0:
                    candidates.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'method': f'edge_{low_thresh}_{high_thresh}',
                        'score': area * aspect_ratio
                    })

    # Approach 2: Text-like region detection
    # Create MSER detector for text regions
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    for region in regions:
        if len(region) < 50:  # Too few points
            continue

        # Get bounding box of region
        x, y, w, h = cv2.boundingRect(region)
        area = w * h

        if area < 300 or area > 15000:
            continue

        aspect_ratio = w / h if h > 0 else 0

        if 2.0 <= aspect_ratio <= 5.0:
            candidates.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'method': 'mser',
                'score': area * aspect_ratio * 1.1  # Slight boost for MSER
            })

    # Approach 3: Color-based detection (lighter regions)
    # Detect lighter regions that could be license plates

    # Threshold for lighter regions
    for threshold in [120, 140, 160, 180]:
        _, light_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        morph = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 400 or area > 25000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            if 2.0 <= aspect_ratio <= 5.0:
                candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'method': f'light_{threshold}',
                    'score': area * aspect_ratio * 0.9  # Slightly lower priority
                })

    # Remove overlapping candidates and sort by score
    filtered_candidates = remove_overlapping_regions(candidates)
    filtered_candidates.sort(key=lambda x: x['score'], reverse=True)

    return filtered_candidates[:10]  # Return top 10 candidates

def remove_overlapping_regions(candidates):
    """Remove overlapping regions, keeping the one with higher score"""
    filtered = []

    # Sort by score first
    candidates.sort(key=lambda x: x['score'], reverse=True)

    for candidate in candidates:
        x1, y1, w1, h1 = candidate['bbox']

        # Check if this candidate overlaps significantly with any existing filtered candidate
        is_overlapping = False
        for filtered_candidate in filtered:
            x2, y2, w2, h2 = filtered_candidate['bbox']

            # Calculate intersection area
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = w1 * h1
                area2 = w2 * h2

                # If intersection is more than 30% of either region, consider it overlapping
                if inter_area > 0.3 * min(area1, area2):
                    is_overlapping = True
                    break

        if not is_overlapping:
            filtered.append(candidate)

    return filtered

def enhance_roi_for_ocr(image, bbox):
    """Enhance ROI untuk OCR yang lebih akurat"""
    x, y, w, h = bbox

    # Add padding
    padding = 8
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return []

    # Convert to grayscale
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi.copy()

    # Resize if too small
    min_height = 32
    min_width = 80

    if gray_roi.shape[0] < min_height or gray_roi.shape[1] < min_width:
        scale_h = min_height / gray_roi.shape[0] if gray_roi.shape[0] < min_height else 1
        scale_w = min_width / gray_roi.shape[1] if gray_roi.shape[1] < min_width else 1
        scale = max(scale_h, scale_w, 2.0)  # At least 2x scale

        new_width = int(gray_roi.shape[1] * scale)
        new_height = int(gray_roi.shape[0] * scale)
        gray_roi = cv2.resize(gray_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Multiple enhancement techniques
    enhanced_versions = []

    # 1. Original (grayscale to BGR)
    enhanced_versions.append(('original', cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)))

    # 2. Histogram equalization
    eq_roi = cv2.equalizeHist(gray_roi)
    enhanced_versions.append(('equalized', cv2.cvtColor(eq_roi, cv2.COLOR_GRAY2BGR)))

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    clahe_roi = clahe.apply(gray_roi)
    enhanced_versions.append(('clahe', cv2.cvtColor(clahe_roi, cv2.COLOR_GRAY2BGR)))

    # 4. Gaussian blur + Otsu threshold
    blur_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    _, otsu_roi = cv2.threshold(blur_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(('otsu', cv2.cvtColor(otsu_roi, cv2.COLOR_GRAY2BGR)))

    # 5. Adaptive threshold
    adaptive_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    enhanced_versions.append(('adaptive', cv2.cvtColor(adaptive_roi, cv2.COLOR_GRAY2BGR)))

    # 6. Bilateral filter + threshold
    bilateral = cv2.bilateralFilter(gray_roi, 9, 75, 75)
    _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(('bilateral_thresh', cv2.cvtColor(bilateral_thresh, cv2.COLOR_GRAY2BGR)))

    # 7. Inverted threshold (for dark text on light background)
    _, inv_thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    enhanced_versions.append(('inverted', cv2.cvtColor(inv_thresh, cv2.COLOR_GRAY2BGR)))

    return enhanced_versions

def accurate_plate_detection(image_path, output_dir="contoh"):
    """Deteksi plat nomor yang akurat untuk kondisi real-world"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Tidak dapat memuat gambar: {image_path}")
        return []

    print(f"📷 Memuat gambar: {image.shape}")

    # Detect potential plate regions
    print("🔍 Mencari kandidat region plat nomor...")
    candidates = detect_license_plate_regions(image)

    print(f"  📊 Ditemukan {len(candidates)} kandidat region")

    # Initialize OCR
    ocr_ensemble = OCREnsemble()

    detections = []
    result_image = image.copy()
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Test each candidate
    for i, candidate in enumerate(candidates):
        x, y, w, h = candidate['bbox']
        print(f"\n  🔍 Testing kandidat {i+1}: {w}x{h} pixels")
        print(f"      Aspect ratio: {candidate['aspect_ratio']:.2f}")
        print(f"      Method: {candidate['method']}")
        print(f"      Score: {candidate['score']:.1f}")

        # Enhance ROI for OCR
        enhanced_versions = enhance_roi_for_ocr(image, candidate['bbox'])

        best_result = None
        best_confidence = 0
        best_method = None

        # Test each enhancement method
        for method_name, enhanced_roi in enhanced_versions:
            try:
                text, conf, details = ocr_ensemble.ensemble_ocr(enhanced_roi)

                # Filter out very short or very long results
                if text and 3 <= len(text.replace(' ', '')) <= 15 and conf > 30:
                    print(f"        {method_name}: '{text}' ({conf:.1f}%)")

                    if conf > best_confidence:
                        best_result = text
                        best_confidence = conf
                        best_method = method_name

            except Exception as e:
                print(f"        {method_name}: Error - {e}")

        if best_result and best_confidence > 45:  # Minimum confidence threshold
            print(f"    🎯 BEST: '{best_result}' (confidence: {best_confidence:.1f}%, method: {best_method})")

            detections.append({
                'text': best_result,
                'confidence': best_confidence,
                'bbox': candidate['bbox'],
                'method': best_method,
                'detection_method': candidate['method']
            })

            # Draw green rectangle for successful detection
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result_image, f"{best_result} ({best_confidence:.1f}%)",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save the best enhanced ROI
            for method_name, enhanced_roi in enhanced_versions:
                if method_name == best_method:
                    roi_path = os.path.join(output_dir, f"{base_name}_best_roi_{i+1}_{best_method}.jpg")
                    cv2.imwrite(roi_path, enhanced_roi)
                    print(f"    💾 Best ROI disimpan: {roi_path}")
                    break
        else:
            print(f"    ❌ Tidak ada hasil OCR yang memadai (best: {best_confidence:.1f}%)")
            # Draw red rectangle for failed detection
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Save result image
    output_path = os.path.join(output_dir, f"{base_name}_accurate_detection.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Hasil deteksi disimpan: {output_path}")

    # Summary
    if detections:
        print(f"\n✅ BERHASIL mendeteksi {len(detections)} plat nomor:")
        for i, det in enumerate(detections):
            print(f"   {i+1}. '{det['text']}' (confidence: {det['confidence']:.1f}%)")
            print(f"      Enhancement: {det['method']}, Detection: {det['detection_method']}")

        # Return the best detection
        best_detection = max(detections, key=lambda x: x['confidence'])
        print(f"\n🏆 DETEKSI TERBAIK: '{best_detection['text']}' ({best_detection['confidence']:.1f}%)")
    else:
        print("\n❌ Tidak ada plat nomor yang berhasil dideteksi")

    return detections

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "contoh"
        accurate_plate_detection(image_path, output_dir)
    else:
        print("Usage: python accurate_plate_detection.py <image_path> [output_dir]")
        print("Example: python accurate_plate_detection.py contoh/15122022plat.jpg contoh")