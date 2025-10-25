#!/usr/bin/env python3
"""
Test detection pada screenshot yang tidak muncul bbox
"""

import cv2
import sys
from utils.yolo_plate_detector import YOLOPlateDetector

def test_screenshot(image_path):
    print(f"📷 Testing detection on: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"✅ Image loaded: {image.shape}")

    # Initialize detector with lowered confidence
    print("\n🔧 Initializing YOLO detector (confidence=0.3)...")
    detector = YOLOPlateDetector(confidence=0.3, streaming_mode=False)

    if not detector.enabled:
        print("❌ YOLO detector not enabled")
        return

    print("✅ Detector initialized")

    # Detect plates
    print("\n🔍 Running plate detection...")
    detections = detector.detect_plates(image)

    print(f"\n📊 Detection Results:")
    print(f"  Total detections: {len(detections)}")

    if len(detections) == 0:
        print("\n❌ No plates detected!")
        print("\nPossible reasons:")
        print("  1. YOLO model confidence still too high")
        print("  2. Plate angle/distance not in training data")
        print("  3. Model not loaded properly")
        return

    # Show detection details
    for i, det in enumerate(detections, 1):
        print(f"\n  Detection #{i}:")
        print(f"    Plate: {det.text}")
        print(f"    Confidence: {det.confidence:.1f}%")
        print(f"    Bbox: {det.bbox}")
        print(f"    Method: {det.detection_method}")

    # Draw bboxes on image
    output_image = image.copy()
    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_image, f"{det.text} ({det.confidence:.0f}%)",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save output
    output_path = image_path.replace('.png', '_detected.png')
    cv2.imwrite(output_path, output_image)
    print(f"\n✅ Output saved to: {output_path}")

if __name__ == "__main__":
    image_path = "contogimg/Screenshot 2025-10-25 104330.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    test_screenshot(image_path)
