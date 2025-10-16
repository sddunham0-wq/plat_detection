#!/usr/bin/env python3
"""
Test script for image deskewing functionality
Tests OCR accuracy improvement on tilted license plates
"""

import cv2
import sys
import logging
from utils.yolo_plate_detector import YOLOPlateDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_deskewing(image_path):
    """Test deskewing on a license plate image"""

    print("\n" + "="*70)
    print("🧪 Testing Image Deskewing for Tilted License Plates")
    print("="*70)

    # Load image
    print(f"\n📷 Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return False

    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

    # Test 1: Detection WITHOUT deskewing
    print("\n" + "-"*70)
    print("Test 1: Detection WITHOUT deskewing")
    print("-"*70)

    detector_no_deskew = YOLOPlateDetector(enable_deskew=False)
    detections_no_deskew = detector_no_deskew.detect_plates(image)

    print(f"\n📊 Results WITHOUT deskewing:")
    print(f"   Total detections: {len(detections_no_deskew)}")

    for i, det in enumerate(detections_no_deskew, 1):
        print(f"   {i}. Plate: '{det.text}' | Confidence: {det.confidence:.1f}%")

    # Test 2: Detection WITH deskewing
    print("\n" + "-"*70)
    print("Test 2: Detection WITH deskewing (ENABLED)")
    print("-"*70)

    detector_with_deskew = YOLOPlateDetector(enable_deskew=True)
    detections_with_deskew = detector_with_deskew.detect_plates(image)

    print(f"\n📊 Results WITH deskewing:")
    print(f"   Total detections: {len(detections_with_deskew)}")

    for i, det in enumerate(detections_with_deskew, 1):
        print(f"   {i}. Plate: '{det.text}' | Confidence: {det.confidence:.1f}%")

    # Comparison
    print("\n" + "="*70)
    print("📈 Comparison Summary")
    print("="*70)

    if len(detections_no_deskew) > 0 and len(detections_with_deskew) > 0:
        no_deskew_text = detections_no_deskew[0].text
        with_deskew_text = detections_with_deskew[0].text

        print(f"\nWithout deskewing: '{no_deskew_text}'")
        print(f"With deskewing:    '{with_deskew_text}'")

        if no_deskew_text != with_deskew_text:
            print(f"\n✅ IMPROVEMENT DETECTED! OCR result changed.")
            print(f"   This indicates deskewing is working.")
        else:
            print(f"\n⚠️ Same result. Either:")
            print(f"   - Plate was already straight (deskewing not needed)")
            print(f"   - Deskewing didn't improve this particular case")

    # Draw results
    result_no_deskew = detector_no_deskew.draw_detections(image.copy(), detections_no_deskew)
    result_with_deskew = detector_with_deskew.draw_detections(image.copy(), detections_with_deskew)

    # Save results
    cv2.imwrite("test_no_deskew.jpg", result_no_deskew)
    cv2.imwrite("test_with_deskew.jpg", result_with_deskew)

    print(f"\n💾 Results saved:")
    print(f"   - test_no_deskew.jpg (without deskewing)")
    print(f"   - test_with_deskew.jpg (with deskewing)")

    print("\n" + "="*70)
    print("✅ Testing completed!")
    print("="*70 + "\n")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_deskewing.py <image_path>")
        print("\nExample:")
        print("  python3 test_deskewing.py contoh/15122022plat.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    test_deskewing(image_path)
