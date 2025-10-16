#!/usr/bin/env python3
"""
Script untuk mendeteksi plat nomor pada gambar tunggal
"""

import cv2
import numpy as np
import sys
import os
from utils.robust_plate_detector import RobustPlateDetector
from utils.hybrid_plate_detector import HybridPlateDetector

def detect_plates_in_image(image_path, output_dir="contoh"):
    """Deteksi plat nomor pada gambar dan simpan hasilnya"""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Tidak dapat memuat gambar: {image_path}")
        return

    print(f"📷 Memuat gambar: {image.shape}")

    # Try multiple detectors
    detectors = [
        ("RobustPlateDetector", RobustPlateDetector()),
        ("HybridPlateDetector", HybridPlateDetector())
    ]

    all_results = []

    for detector_name, detector in detectors:
        print(f"\n🔍 Mencoba dengan {detector_name}...")

        try:
            detections = detector.detect_plates(image)
            print(f"  🎯 Ditemukan {len(detections)} plat nomor:")

            for i, det in enumerate(detections):
                print(f"    {i+1}. '{det.text}' (confidence: {det.confidence:.1f}%)")
                print(f"       BBox: {det.bbox}")

            if detections:
                # Draw detections
                result_image = detector.draw_detections(image, detections)

                # Save result
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_{detector_name.lower()}_result.jpg")
                cv2.imwrite(output_path, result_image)
                print(f"  💾 Hasil disimpan: {output_path}")

                all_results.extend(detections)

        except Exception as e:
            print(f"  ❌ Error dengan {detector_name}: {e}")

    if not all_results:
        print("\n❌ Tidak ada plat nomor yang terdeteksi dengan semua detektor")
        print("   Mencoba dengan parameter yang lebih sensitif...")

        # Try with more sensitive parameters
        try_alternative_detection(image, image_path, output_dir)
    else:
        print(f"\n✅ Total {len(all_results)} deteksi berhasil")

def try_alternative_detection(image, image_path, output_dir):
    """Coba deteksi alternatif dengan parameter yang lebih sensitif"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply various preprocessing
    preprocessed_images = []
    preprocessed_images.append(("original_gray", gray))

    # Histogram equalization
    eq = cv2.equalizeHist(gray)
    preprocessed_images.append(("equalized", eq))

    # Gaussian blur + threshold
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("threshold", thresh))

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(("morphological", morph))

    print("\n🔍 Mencoba deteksi alternatif dengan preprocessing...")

    from utils.ocr_ensemble import OCREnsemble
    ocr = OCREnsemble()

    found_plates = []

    for name, processed_img in preprocessed_images:
        print(f"  📊 Testing {name}...")

        # Try OCR on whole image
        if len(processed_img.shape) == 3:
            text, conf, details = ocr.ensemble_ocr(processed_img)
        else:
            # Convert grayscale to BGR for OCR
            bgr_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            text, conf, details = ocr.ensemble_ocr(bgr_img)

        if text and conf > 50:
            print(f"    🎯 OCR: '{text}' (confidence: {conf:.1f}%)")
            found_plates.append((name, text, conf))

            # Save the processed image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{name}_ocr.jpg")

            if len(processed_img.shape) == 2:
                cv2.imwrite(output_path, processed_img)
            else:
                cv2.imwrite(output_path, processed_img)
            print(f"    💾 Disimpan: {output_path}")

    if found_plates:
        print(f"\n✅ Ditemukan {len(found_plates)} kandidat plat dengan preprocessing:")
        for name, text, conf in found_plates:
            print(f"   - {text} ({conf:.1f}%) via {name}")
    else:
        print("\n❌ Tidak ditemukan plat nomor dengan semua metode")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "contoh"
        detect_plates_in_image(image_path, output_dir)
    else:
        print("Usage: python detect_single_image.py <image_path> [output_dir]")
        print("Example: python detect_single_image.py contoh/15122022plat.jpg contoh")