#!/usr/bin/env python3

import cv2
import numpy as np
from utils.robust_plate_detector import RobustPlateDetector

def test_contour_detection():
    # Load the debug ROI image that contains clear plate text
    image_path = "debug_roi_thresh.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"📷 Loaded image: {image.shape}")

    detector = RobustPlateDetector()

    # Test the contour detection method directly
    # Let's see what candidates are found

    # Apply smart ROI first
    h, w = image.shape[:2]
    general_roi = (0.05, 0.25, 0.9, 0.6)  # From config
    x1 = int(general_roi[0] * w)
    y1 = int(general_roi[1] * h)
    x2 = int((general_roi[0] + general_roi[2]) * w)
    y2 = int((general_roi[1] + general_roi[3]) * h)

    roi_image = image[y1:y2, x1:x2]
    print(f"🎯 ROI extracted: {roi_image.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing like in the detector
    # Morphological tophat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # Threshold
    thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"🔍 Found {len(contours)} contours")

    # Check which contours pass the filtering
    valid_candidates = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Basic filtering from the detector
        min_area = 200  # minimum area
        max_area = roi_image.shape[0] * roi_image.shape[1] * 0.8
        aspect_ratio = w / h if h > 0 else 0

        print(f"  Contour {i}: area={area}, bbox=({x},{y},{w},{h}), aspect={aspect_ratio:.2f}")

        if (area >= min_area and area <= max_area and
            1.5 <= aspect_ratio <= 6.0 and
            w >= 50 and h >= 15):
            valid_candidates.append((x, y, w, h))
            print(f"    ✅ Valid candidate!")
        else:
            print(f"    ❌ Filtered out")

    print(f"\n🎯 Valid candidates: {len(valid_candidates)}")

    # If we have candidates, test OCR on them
    if valid_candidates:
        for i, (x, y, w, h) in enumerate(valid_candidates):
            candidate_roi = roi_image[y:y+h, x:x+w]
            print(f"\n🧠 Testing OCR on candidate {i+1}:")

            # Test with our fixed OCR ensemble
            from utils.ocr_ensemble import OCREnsemble
            ocr_ensemble = OCREnsemble()

            # Test OCR
            text, confidence = ocr_ensemble.recognize(candidate_roi)
            print(f"  OCR result: '{text}' (confidence: {confidence:.1f}%)")

            # Also test the multi-angle OCR
            text_multi, conf_multi = detector._multi_angle_ocr(candidate_roi)
            print(f"  Multi-angle OCR: '{text_multi}' (confidence: {conf_multi:.1f}%)")

if __name__ == "__main__":
    test_contour_detection()