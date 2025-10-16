#!/usr/bin/env python3
"""
Debug script untuk test stabilized plate detector
"""

import cv2
import logging
import sys
import os

# Add project root to path
sys.path.append('/Users/andra/Documents/DWI/project-plat-detection-pai')

# Import our detectors
from utils.robust_plate_detector import RobustPlateDetector

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_detector(image_path):
    """Test the stabilized detector"""
    print(f"🔧 Testing Stabilized Plate Detector on: {image_path}")
    print("=" * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"📸 Image loaded: {image.shape}")

    # Test robust detector in streaming mode (our optimized mode)
    detector = RobustPlateDetector(streaming_mode=True)

    print(f"🎯 Detector settings:")
    print(f"   Min area: {detector.min_area}")
    print(f"   Max area: {detector.max_area}")
    print(f"   Min aspect ratio: {detector.min_aspect_ratio}")
    print(f"   Max aspect ratio: {detector.max_aspect_ratio}")
    print(f"   Min confidence: {detector.min_confidence}")
    print(f"   Min text likelihood: {detector.min_text_likelihood}")
    print(f"   Max candidates: {detector.max_candidates}")

    # Run detection with debug ROI
    print(f"\n🔍 Running detection...")

    # Test without ROI first to see if we get candidates
    print(f"🧪 Testing full image detection (no ROI)...")

    # Temporarily disable ROI for testing
    from config import DetectionConfig
    original_enable_roi = DetectionConfig.ENABLE_SMART_ROI
    DetectionConfig.ENABLE_SMART_ROI = False

    detections = detector.detect_plates(image)

    # Restore ROI setting
    DetectionConfig.ENABLE_SMART_ROI = original_enable_roi

    print(f"\n📊 Results:")
    print(f"   Total detections: {len(detections)}")

    for i, detection in enumerate(detections):
        print(f"   {i+1}. Text: '{detection.text}'")
        print(f"       Confidence: {detection.confidence:.1f}%")
        print(f"       Bbox: {detection.bbox}")
        print(f"       Method: {detection.detection_method}")

    # Show statistics
    stats = detector.get_statistics()
    print(f"\n📈 Statistics: {stats}")

    # Save result image
    if detections:
        result = detector.draw_detections(image, detections)
        output_path = "debug_stabilized_result.jpg"
        cv2.imwrite(output_path, result)
        print(f"💾 Result saved: {output_path}")

    return detections

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_detector(image_path)
    else:
        print("Usage: python debug_stabilized_detector.py <image_path>")

        # Test with available images
        test_images = [
            "better_test_plate.jpg",
            "balanced_cctv_result.jpg",
            "contoh/image.png",
            "contoh/qwerty.png"
        ]

        for img in test_images:
            if os.path.exists(img):
                print(f"\n" + "="*60)
                test_detector(img)
                break