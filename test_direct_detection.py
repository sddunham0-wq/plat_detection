#!/usr/bin/env python3

import cv2
import numpy as np
from utils.ocr_ensemble import OCREnsemble

def test_direct_ocr():
    # Load the debug ROI image that contains clear plate text
    image_path = "debug_roi_thresh.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"📷 Loaded image: {image.shape}")

    # Test OCR directly on the entire image (since it's already a plate ROI)
    ocr_ensemble = OCREnsemble()

    print("\n🧠 Testing OCR directly on the full image:")
    text, confidence, details = ocr_ensemble.ensemble_ocr(image)
    print(f"  OCR result: '{text}' (confidence: {confidence:.1f}%)")

    # Also test with grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("\n🧠 Testing OCR on grayscale:")
    text_gray, conf_gray, details_gray = ocr_ensemble.ensemble_ocr(gray)
    print(f"  OCR result: '{text_gray}' (confidence: {conf_gray:.1f}%)")

    # Test with thresholding
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print("\n🧠 Testing OCR on thresholded image:")
    text_thresh, conf_thresh, details_thresh = ocr_ensemble.ensemble_ocr(thresh)
    print(f"  OCR result: '{text_thresh}' (confidence: {conf_thresh:.1f}%)")

if __name__ == "__main__":
    test_direct_ocr()