#!/usr/bin/env python3
"""
Enhanced single image plate detection dengan multiple approaches
"""

import cv2
import numpy as np
import sys
import os
from utils.ocr_ensemble import OCREnsemble

def find_plate_regions(image):
    """Cari region yang kemungkinan berisi plat nomor"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Multiple techniques to find plate-like regions
    regions = []

    # Method 1: Contour-based detection
    # Apply bilateral filter to reduce noise but keep edges sharp
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    # Find edges
    edges = cv2.Canny(bilateral, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate contour
        approx = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)

        # Calculate area and bounding rect
        area = cv2.contourArea(contour)
        if area < 500:  # Too small
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Check aspect ratio (plates are typically wider than tall)
        aspect_ratio = w / h if h > 0 else 0

        # Indonesian license plates typically have aspect ratio between 2:1 to 4:1
        if 1.5 <= aspect_ratio <= 6.0 and area > 500:
            regions.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'method': 'contour',
                'score': area * aspect_ratio  # Simple scoring
            })

    # Method 2: Template matching approach - look for rectangular regions
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))

    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to connect text
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours in morphological result
    contours2, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours2:
        area = cv2.contourArea(contour)
        if area < 300:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        if 1.5 <= aspect_ratio <= 6.0:
            regions.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'method': 'morphological',
                'score': area * aspect_ratio * 1.2  # Slight boost for morphological
            })

    # Sort by score (descending)
    regions.sort(key=lambda x: x['score'], reverse=True)

    # Remove overlapping regions
    filtered_regions = []
    for region in regions:
        x1, y1, w1, h1 = region['bbox']

        is_overlap = False
        for existing in filtered_regions:
            x2, y2, w2, h2 = existing['bbox']

            # Check overlap
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                is_overlap = True
                break

        if not is_overlap:
            filtered_regions.append(region)

    return filtered_regions[:5]  # Return top 5 candidates

def enhanced_ocr_on_region(image, bbox, ocr_ensemble):
    """Enhanced OCR pada region tertentu"""
    x, y, w, h = bbox

    # Extract ROI with some padding
    padding = 10
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None, 0, None

    # Multiple preprocessing approaches for better OCR
    results = []

    # Original ROI
    text, conf, details = ocr_ensemble.ensemble_ocr(roi)
    if text and conf > 30:
        results.append((text, conf, 'original'))

    # Convert to grayscale if not already
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi.copy()

    # Resize for better OCR (scale up small regions)
    if gray_roi.shape[0] < 50 or gray_roi.shape[1] < 100:
        scale_factor = max(2, 100 // min(gray_roi.shape[0], gray_roi.shape[1]))
        new_height = gray_roi.shape[0] * scale_factor
        new_width = gray_roi.shape[1] * scale_factor
        gray_roi = cv2.resize(gray_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Apply histogram equalization
    eq_roi = cv2.equalizeHist(gray_roi)
    eq_bgr = cv2.cvtColor(eq_roi, cv2.COLOR_GRAY2BGR)
    text, conf, details = ocr_ensemble.ensemble_ocr(eq_bgr)
    if text and conf > 30:
        results.append((text, conf, 'equalized'))

    # Apply threshold
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_bgr = cv2.cvtColor(thresh_roi, cv2.COLOR_GRAY2BGR)
    text, conf, details = ocr_ensemble.ensemble_ocr(thresh_bgr)
    if text and conf > 30:
        results.append((text, conf, 'threshold'))

    # Apply adaptive threshold
    adaptive_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    adaptive_bgr = cv2.cvtColor(adaptive_roi, cv2.COLOR_GRAY2BGR)
    text, conf, details = ocr_ensemble.ensemble_ocr(adaptive_bgr)
    if text and conf > 30:
        results.append((text, conf, 'adaptive'))

    # Return best result
    if results:
        best_result = max(results, key=lambda x: x[1])  # Highest confidence
        return best_result[0], best_result[1], best_result[2]

    return None, 0, None

def detect_plates_enhanced(image_path, output_dir="contoh"):
    """Enhanced plate detection dengan multiple approaches"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Tidak dapat memuat gambar: {image_path}")
        return

    print(f"📷 Memuat gambar: {image.shape}")

    # Find potential plate regions
    print("🔍 Mencari region kandidat plat...")
    regions = find_plate_regions(image)

    print(f"  📊 Ditemukan {len(regions)} region kandidat")

    # Initialize OCR
    ocr_ensemble = OCREnsemble()

    # Test OCR on each region
    detections = []
    result_image = image.copy()

    for i, region in enumerate(regions):
        x, y, w, h = region['bbox']
        print(f"\n  🔍 Testing region {i+1}: {w}x{h} (aspect: {region['aspect_ratio']:.2f}, method: {region['method']})")

        # Enhanced OCR on this region
        text, conf, method = enhanced_ocr_on_region(image, region['bbox'], ocr_ensemble)

        if text and conf > 40:  # Lower threshold for candidates
            print(f"    🎯 OCR: '{text}' (confidence: {conf:.1f}%, method: {method})")
            detections.append({
                'text': text,
                'confidence': conf,
                'bbox': region['bbox'],
                'method': method,
                'region_method': region['method']
            })

            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_image, f"{text} ({conf:.1f}%)",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"    ❌ OCR gagal atau confidence rendah: '{text}' ({conf:.1f}%)")

            # Draw failed regions in red
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Save result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_enhanced_detection.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Hasil deteksi disimpan: {output_path}")

    # Summary
    if detections:
        print(f"\n✅ Berhasil mendeteksi {len(detections)} plat nomor:")
        for i, det in enumerate(detections):
            print(f"   {i+1}. '{det['text']}' (confidence: {det['confidence']:.1f}%)")
            print(f"      Method: {det['method']} via {det['region_method']}")
    else:
        print("\n❌ Tidak ada plat nomor yang berhasil dideteksi")

    return detections

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "contoh"
        detect_plates_enhanced(image_path, output_dir)
    else:
        print("Usage: python enhanced_single_detection.py <image_path> [output_dir]")