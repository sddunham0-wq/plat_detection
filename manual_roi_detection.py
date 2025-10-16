#!/usr/bin/env python3
"""
Manual ROI detection untuk fokus pada area plat nomor yang terlihat jelas
Berdasarkan analisis visual gambar P 2543 BP2
"""

import cv2
import numpy as np
import sys
import os
from utils.ocr_ensemble import OCREnsemble

def create_focused_regions(image):
    """Buat region fokus berdasarkan analisis visual plat nomor"""

    height, width = image.shape[:2]

    # Berdasarkan gambar, plat nomor ada di area tengah-bawah
    # Mari buat beberapa region kandidat di area tersebut

    regions = []

    # Region 1: Area tengah-bawah (dimana plat nomor terlihat)
    center_x = width // 2
    center_y = int(height * 0.7)  # 70% dari atas

    # Berbagai ukuran region di sekitar area tersebut
    region_configs = [
        # (width_ratio, height_ratio, center_offset_x, center_offset_y)
        (0.3, 0.08, 0, 0),      # Region utama
        (0.25, 0.06, 0, 0),     # Lebih kecil
        (0.35, 0.1, 0, 0),      # Lebih besar
        (0.3, 0.08, -50, 0),    # Shift kiri
        (0.3, 0.08, 50, 0),     # Shift kanan
        (0.3, 0.08, 0, -20),    # Shift atas
        (0.3, 0.08, 0, 20),     # Shift bawah
    ]

    for i, (w_ratio, h_ratio, offset_x, offset_y) in enumerate(region_configs):
        region_w = int(width * w_ratio)
        region_h = int(height * h_ratio)

        region_center_x = center_x + offset_x
        region_center_y = center_y + offset_y

        x = max(0, region_center_x - region_w // 2)
        y = max(0, region_center_y - region_h // 2)
        w = min(region_w, width - x)
        h = min(region_h, height - y)

        if w > 50 and h > 20:  # Minimum size check
            regions.append({
                'bbox': (x, y, w, h),
                'name': f'focused_region_{i+1}',
                'description': f'Focused area {i+1} around expected plate location'
            })

    # Juga tambahkan beberapa region horizontal strip di berbagai ketinggian
    for y_ratio in [0.6, 0.65, 0.7, 0.75, 0.8]:
        strip_y = int(height * y_ratio)
        strip_h = int(height * 0.05)  # 5% tinggi
        strip_w = int(width * 0.6)    # 60% lebar
        strip_x = (width - strip_w) // 2

        if strip_y + strip_h < height:
            regions.append({
                'bbox': (strip_x, strip_y, strip_w, strip_h),
                'name': f'horizontal_strip_{y_ratio}',
                'description': f'Horizontal strip at {y_ratio*100:.0f}% height'
            })

    return regions

def enhance_plate_roi(roi):
    """Enhance ROI khusus untuk plat nomor Indonesia"""

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # Resize jika terlalu kecil
    min_height = 40
    min_width = 120

    if gray.shape[0] < min_height or gray.shape[1] < min_width:
        scale_h = min_height / gray.shape[0] if gray.shape[0] < min_height else 1
        scale_w = min_width / gray.shape[1] if gray.shape[1] < min_width else 1
        scale = max(scale_h, scale_w, 3.0)  # Minimum 3x scale

        new_width = int(gray.shape[1] * scale)
        new_height = int(gray.shape[0] * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    enhanced_versions = []

    # 1. Original scaled
    enhanced_versions.append(('scaled_original', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))

    # 2. Histogram equalization
    eq = cv2.equalizeHist(gray)
    enhanced_versions.append(('equalized', cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)))

    # 3. CLAHE dengan parameter agresif
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2,2))
    clahe_result = clahe.apply(gray)
    enhanced_versions.append(('clahe_aggressive', cv2.cvtColor(clahe_result, cv2.COLOR_GRAY2BGR)))

    # 4. Bilateral filter + adaptive threshold
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 7, 4)
    enhanced_versions.append(('bilateral_adaptive', cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)))

    # 5. Otsu threshold with blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    enhanced_versions.append(('blur_otsu', cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)))

    # 6. Inverted Otsu (untuk kasus dark text on light background)
    _, inv_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    enhanced_versions.append(('inverted_otsu', cv2.cvtColor(inv_otsu, cv2.COLOR_GRAY2BGR)))

    # 7. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    enhanced_versions.append(('morphological', cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)))

    # 8. Unsharp masking untuk sharpening
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
    unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
    enhanced_versions.append(('unsharp_mask', cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)))

    return enhanced_versions

def manual_roi_detection(image_path, output_dir="contoh"):
    """Detection dengan fokus manual pada region yang diperkirakan"""

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Tidak dapat memuat gambar: {image_path}")
        return []

    print(f"📷 Memuat gambar: {image.shape}")

    # Create focused regions
    print("🎯 Membuat region fokus berdasarkan analisis visual...")
    regions = create_focused_regions(image)

    print(f"  📊 Dibuat {len(regions)} region fokus")

    # Initialize OCR
    ocr_ensemble = OCREnsemble()

    detections = []
    result_image = image.copy()
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Test each focused region
    for i, region in enumerate(regions):
        x, y, w, h = region['bbox']
        print(f"\n  🔍 Testing {region['name']}: {w}x{h} pixels")
        print(f"      {region['description']}")

        # Extract ROI
        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        # Enhance ROI
        enhanced_versions = enhance_plate_roi(roi)

        best_result = None
        best_confidence = 0
        best_method = None
        best_enhanced_roi = None

        # Test each enhancement
        for method_name, enhanced_roi in enhanced_versions:
            try:
                text, conf, details = ocr_ensemble.ensemble_ocr(enhanced_roi)

                # Clean up text - remove obvious garbage
                if text:
                    # Remove very short results
                    clean_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip()

                    if len(clean_text) >= 4 and conf > 30:
                        print(f"        {method_name}: '{clean_text}' ({conf:.1f}%)")

                        if conf > best_confidence:
                            best_result = clean_text
                            best_confidence = conf
                            best_method = method_name
                            best_enhanced_roi = enhanced_roi

            except Exception as e:
                print(f"        {method_name}: Error - {e}")

        if best_result and best_confidence > 40:
            print(f"    🎯 BEST: '{best_result}' (confidence: {best_confidence:.1f}%, method: {best_method})")

            detections.append({
                'text': best_result,
                'confidence': best_confidence,
                'bbox': region['bbox'],
                'method': best_method,
                'region_name': region['name']
            })

            # Draw green rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_image, f"{best_result} ({best_confidence:.1f}%)",
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save the best enhanced ROI
            if best_enhanced_roi is not None:
                roi_path = os.path.join(output_dir, f"{base_name}_roi_{region['name']}_{best_method}.jpg")
                cv2.imwrite(roi_path, best_enhanced_roi)
                print(f"    💾 Best ROI disimpan: {roi_path}")

            # Juga simpan ROI asli untuk referensi
            original_roi_path = os.path.join(output_dir, f"{base_name}_original_roi_{region['name']}.jpg")
            cv2.imwrite(original_roi_path, roi)
            print(f"    💾 Original ROI disimpan: {original_roi_path}")

        else:
            print(f"    ❌ Tidak ada hasil yang memadai (best: {best_confidence:.1f}%)")
            # Draw blue rectangle for tested regions
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    # Save result image
    output_path = os.path.join(output_dir, f"{base_name}_manual_roi_detection.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"\n💾 Hasil deteksi disimpan: {output_path}")

    # Summary
    if detections:
        print(f"\n✅ BERHASIL mendeteksi {len(detections)} plat nomor:")
        for i, det in enumerate(detections):
            print(f"   {i+1}. '{det['text']}' (confidence: {det['confidence']:.1f}%)")
            print(f"      Region: {det['region_name']}, Method: {det['method']}")

        # Show best detection
        best_detection = max(detections, key=lambda x: x['confidence'])
        print(f"\n🏆 DETEKSI TERBAIK: '{best_detection['text']}' ({best_detection['confidence']:.1f}%)")
        print(f"    Region: {best_detection['region_name']}")
        print(f"    Enhancement: {best_detection['method']}")
    else:
        print("\n❌ Tidak ada plat nomor yang berhasil dideteksi")

    return detections

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "contoh"
        manual_roi_detection(image_path, output_dir)
    else:
        print("Usage: python manual_roi_detection.py <image_path> [output_dir]")
        print("Example: python manual_roi_detection.py contoh/15122022plat.jpg contoh")