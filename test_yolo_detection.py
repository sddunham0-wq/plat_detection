#!/usr/bin/env python3
"""
Test YOLO Plate Detection
Test dengan image atau webcam
"""

import cv2
import sys
from utils.yolo_plate_detector import YOLOPlateDetector

def test_image(image_path):
    """Test YOLO detection dengan single image"""

    print(f"\n{'='*60}")
    print(f"🧪 Testing YOLO Detection")
    print(f"📁 Image: {image_path}")
    print(f"{'='*60}\n")

    # Initialize detector
    print("🔧 Loading YOLO model...")
    try:
        detector = YOLOPlateDetector(
            model_path='models/best.pt',
            conf_threshold=0.25
        )
        print("✅ YOLO model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        return

    # Load image
    print(f"📷 Loading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    print(f"✅ Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels\n")

    # Detect dengan confidence
    print("🔍 Running YOLO detection...")
    detections = detector.detect_with_confidence(frame)

    print(f"\n📊 Detection Results:")
    print(f"{'─'*60}")
    print(f"Total detections: {len(detections)}\n")

    if detections:
        for i, det in enumerate(detections, 1):
            bbox = det['bbox']
            conf = det['confidence']
            x, y, w, h = bbox
            print(f"Detection #{i}:")
            print(f"  📍 Position: x={x}, y={y}")
            print(f"  📏 Size: w={w}px, h={h}px")
            print(f"  ✨ Confidence: {conf:.3f} ({conf*100:.1f}%)")
            print()
    else:
        print("⚠️  No plates detected")
        print("💡 Try lowering conf_threshold to 0.15\n")

    # Draw boxes
    result = detector.draw(frame, detections, "PLAT")

    # Save result
    output_path = image_path.replace('.', '_yolo.')
    cv2.imwrite(output_path, result)
    print(f"💾 Result saved: {output_path}\n")

    # Display
    print("👁️  Displaying result (press any key to close)...")
    cv2.imshow('YOLO Detection Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ Test completed!")

def test_camera():
    """Test YOLO detection dengan webcam"""

    print(f"\n{'='*60}")
    print(f"🧪 Testing YOLO Detection - Camera Mode")
    print(f"{'='*60}\n")

    # Initialize detector
    print("🔧 Loading YOLO model...")
    try:
        detector = YOLOPlateDetector(
            model_path='models/best.pt',
            conf_threshold=0.25
        )
        print("✅ YOLO model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        return

    # Open camera
    print("📹 Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    print("✅ Camera opened")
    print("\n📋 Instructions:")
    print("  - Point camera at license plate")
    print("  - Green box will appear on detection")
    print("  - Press 'q' to quit\n")
    print("🎬 Starting live detection...\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame")
            break

        frame_count += 1

        # Detect (every frame)
        boxes = detector.detect(frame)

        # Draw results
        result = detector.draw(frame, boxes, "PLAT")

        # Add FPS counter
        cv2.putText(result, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show
        cv2.imshow('YOLO Camera Test - Press Q to quit', result)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Camera test completed")
    print(f"📊 Total frames processed: {frame_count}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Test with image path
        test_image(sys.argv[1])
    else:
        # Test with camera
        test_camera()
