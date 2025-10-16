#!/usr/bin/env python3

import cv2
import numpy as np
from utils.robust_plate_detector import RobustPlateDetector

def test_detection():
    # Load the debug ROI image that contains clear plate text
    image_path = "debug_roi_thresh.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"📷 Loaded image: {image.shape}")

    # Test robust detector directly
    detector = RobustPlateDetector()

    # Test with different vehicle types
    vehicle_types = ['general', 'motorcycle', 'car', None]

    for vehicle_type in vehicle_types:
        print(f"\n🚗 Testing vehicle_type: {vehicle_type}")
        detections = detector.detect_plates(image, vehicle_type=vehicle_type)
        print(f"📊 Found {len(detections)} detections")

        for i, detection in enumerate(detections):
            print(f"  🎯 Detection {i+1}: '{detection.text}' (conf: {detection.confidence:.1f}%)")

if __name__ == "__main__":
    test_detection()