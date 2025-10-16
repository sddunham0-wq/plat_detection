#!/usr/bin/env python3
"""
Targeted plate detection untuk fokus pada area plat yang jelas terlihat
"""

import cv2
import numpy as np
import sys
import os
from utils.ocr_ensemble import OCREnsemble

def detect_white_rectangular_regions(image):
    """Deteksi region putih berbentuk persegi panjang (karakteristik plat nomor Indonesia)"""

    # Convert to HSV for better white detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for white color in HSV
    # White has low saturation and high value
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 30, 255])

    # Create mask for white regions
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Also try grayscale thresholding for white regions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine both masks
    combined_mask = cv2.bitwise_or(white_mask, white_thresh)

    # Morphological operations to clean up and connect regions
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_rect)

    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area (plat nomor tidak terlalu kecil atau besar)
        if area < 800 or area > 50000:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 0

        # Indonesian license plates aspect ratio typically 3:1 to 5:1
        if 2.5 <= aspect_ratio <= 6.0:

            # Calculate how rectangular the contour is
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0

            # Plat nomor should be quite rectangular
            if fill_ratio > 0.7:
                plate_candidates.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'fill_ratio': fill_ratio,
                    'score': area * aspect_ratio * fill_ratio
                })

    # Sort by score
    plate_candidates.sort(key=lambda x: x['score'], reverse=True)

    return plate_candidates

def extract_and_enhance_plate_roi(image, bbox):
    """Extract dan enhance ROI plat untuk OCR yang lebih baik"""

    x, y, w, h = bbox

    # Add some padding
    padding = 5
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    # Convert to grayscale
    if len(roi.shape) == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = roi.copy()

    # Resize if too small
    min_height = 40
    min_width = 120

    if gray_roi.shape[0] < min_height or gray_roi.shape[1] < min_width:
        scale_h = min_height / gray_roi.shape[0] if gray_roi.shape[0] < min_height else 1
        scale_w = min_width / gray_roi.shape[1] if gray_roi.shape[1] < min_width else 1
        scale = max(scale_h, scale_w)

        new_width = int(gray_roi.shape[1] * scale)
        new_height = int(gray_roi.shape[0] * scale)
        gray_roi = cv2.resize(gray_roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Multiple enhancement techniques
    enhanced_rois = []

    # 1. Original
    enhanced_rois.append(('original', cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)))

    # 2. Histogram equalization
    eq_roi = cv2.equalizeHist(gray_roi)
    enhanced_rois.append(('equalized', cv2.cvtColor(eq_roi, cv2.COLOR_GRAY2BGR)))

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_roi = clahe.apply(gray_roi)
    enhanced_rois.append(('clahe', cv2.cvtColor(clahe_roi, cv2.COLOR_GRAY2BGR)))

    # 4. Otsu threshold
    _, otsu_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_rois.append(('otsu', cv2.cvtColor(otsu_roi, cv2.COLOR_GRAY2BGR)))

    # 5. Adaptive threshold
    adaptive_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
    enhanced_rois.append(('adaptive', cv2.cvtColor(adaptive_roi, cv2.COLOR_GRAY2BGR)))

    # 6. Bilateral filter + threshold
    bilateral = cv2.bilateralFilter(gray_roi, 9, 75, 75)
    _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_rois.append(('bilateral', cv2.cvtColor(bilateral_thresh, cv2.COLOR_GRAY2BGR)))

    return enhanced_rois

def targeted_plate_detection(image_path, output_dir="contoh"):
    """Targeted detection fokus pada plat putih berbentuk persegi panjang"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Tidak dapat memuat gambar: {image_path}")
        return

    print(f"📷 Memuat gambar: {image.shape}")

    # Detect white rectangular regions
    print("🔍 Mencari region putih berbentuk persegi panjang...")
    candidates = detect_white_rectangular_regions(image)

    print(f"  📊 Ditemukan {len(candidates)} kandidat plat")

    # Initialize OCR
    ocr_ensemble = OCREnsemble()

    detections = []
    result_image = image.copy()
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i, candidate in enumerate(candidates[:3]):  # Test top 3 candidates
        x, y, w, h = candidate['bbox']
        print(f"\n  🔍 Testing kandidat {i+1}: {w}x{h} pixels")
        print(f"      Aspect ratio: {candidate['aspect_ratio']:.2f}")
        print(f"      Fill ratio: {candidate['fill_ratio']:.2f}")

        # Extract and enhance ROI
        enhanced_rois = extract_and_enhance_plate_roi(image, candidate['bbox'])

        if not enhanced_rois:
            continue

        best_result = None
        best_confidence = 0

        # Test each enhancement method
        for method_name, enhanced_roi in enhanced_rois:
            text, conf, details = ocr_ensemble.ensemble_ocr(enhanced_roi)

            if text and conf > best_confidence:
                best_result = (text, conf, method_name)
                best_confidence = conf

        if best_result and best_confidence > 50:
            text, conf, method = best_result
            print(f"    🎯 Best OCR: '{text}' (confidence: {conf:.1f}%, method: {method})")

            detections.append({
                'text': text,
                'confidence': conf,
                'bbox': candidate['bbox'],
                'method': method
            })

            # Draw green bounding box for successful detection
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result_image, f"{text} ({conf:.1f}%)",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Save the best enhanced ROI
            if best_result:
                for method_name, enhanced_roi in enhanced_rois:
                    if method_name == method:
                        roi_path = os.path.join(output_dir, f"{base_name}_roi_{i+1}_{method}.jpg")
                        cv2.imwrite(roi_path, enhanced_roi)
                        print(f"    💾 ROI disimpan: {roi_path}")
                        break
        else:
            print(f"    ❌ OCR gagal atau confidence rendah")
            # Draw red bounding box for failed detection
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save result image
    output_path = os.path.join(output_dir, f"{base_name}_targeted_detection.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Hasil deteksi disimpan: {output_path}")

    # Summary
    if detections:
        print(f"\n✅ Berhasil mendeteksi {len(detections)} plat nomor:")
        for i, det in enumerate(detections):
            print(f"   {i+1}. '{det['text']}' (confidence: {det['confidence']:.1f}%)")
            print(f"      Enhancement method: {det['method']}")
    else:
        print("\n❌ Tidak ada plat nomor yang berhasil dideteksi")

    return detections

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "contoh"
        targeted_plate_detection(image_path, output_dir)
    else:
        print("Usage: python targeted_plate_detection.py <image_path> [output_dir]")