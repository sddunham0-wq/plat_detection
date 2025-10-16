#!/usr/bin/env python3
"""
Test Plate Detection dengan Stream Kamera Baru
Analisis kualitas bounding box dan optimasi parameter
"""

import cv2
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hybrid_plate_detector import HybridPlateDetector
from config import CCTVConfig

def test_plate_detection_stream():
    """
    Test deteksi plat dengan stream kamera baru
    """
    print("🎯 Testing Plate Detection dengan Kamera Baru")
    print("=" * 55)

    # Initialize detector
    detector = HybridPlateDetector()
    print(f"✅ Detector initialized - YOLO: {detector.yolo_enabled}")

    # Test URL
    stream_url = CCTVConfig.DEFAULT_RTSP_URL
    print(f"🔗 Stream URL: {stream_url}")

    # Open video capture
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Failed to open camera stream")
        return

    # Get stream info
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Stream Info: {width}x{height} @ {fps:.1f} FPS")
    print("\\n🔍 Starting detection test...")
    print("Press 'q' to quit, 's' to save frame with detection")

    frame_count = 0
    detection_count = 0
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            frame_count += 1

            # Skip frames untuk testing (process every 5th frame)
            if frame_count % 5 != 0:
                continue

            start_time = time.time()

            # Run detection
            detections = detector.detect_plates(frame)

            detection_time = time.time() - start_time

            if detections:
                detection_count += 1
                total_detections += len(detections)

                print(f"\\n📍 Frame {frame_count}: {len(detections)} plate(s) detected ({detection_time:.3f}s)")

                for i, detection in enumerate(detections, 1):
                    x, y, w, h = detection.bbox
                    confidence = detection.confidence
                    text = detection.text
                    method = detection.detection_method

                    print(f"   Plate {i}: '{text}' (conf: {confidence:.2f}, method: {method})")
                    print(f"   BBox: ({x}, {y}, {w}, {h}) - Size: {w}x{h}")

                    # Analyze bounding box quality
                    aspect_ratio = w / h if h > 0 else 0
                    area = w * h
                    print(f"   Quality: aspect_ratio={aspect_ratio:.2f}, area={area}")

                # Draw detections on frame
                annotated_frame = detector.draw_detections(frame, detections)

                # Resize untuk display
                display_frame = cv2.resize(annotated_frame, (1280, 720))
                cv2.imshow('Plate Detection Test', display_frame)

            else:
                # Show frame without detections setiap 30 frames
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: No plates detected ({detection_time:.3f}s)")
                    display_frame = cv2.resize(frame, (1280, 720))
                    cv2.imshow('Plate Detection Test', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and detections:
                # Save frame with detections
                filename = f"detected_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"💾 Saved frame: {filename}")

            # Auto-stop after testing period
            if frame_count >= 150:  # About 30 seconds at 5fps processing
                break

    except KeyboardInterrupt:
        print("\\n⏹️ Detection test stopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Print statistics
        print(f"\\n📊 Detection Statistics:")
        print(f"   Total frames processed: {frame_count // 5}")
        print(f"   Frames with detections: {detection_count}")
        print(f"   Total plates detected: {total_detections}")
        print(f"   Detection rate: {(detection_count/(frame_count//5)*100):.1f}%")

        if detection_count > 0:
            print(f"   Average plates per detection: {total_detections/detection_count:.1f}")

def analyze_detection_parameters():
    """
    Analyze dan suggest parameter optimizations
    """
    print("\\n🔧 Detection Parameter Analysis:")
    print("=" * 40)

    # Read current parameters
    from utils.robust_plate_detector import RobustPlateDetector
    robust_detector = RobustPlateDetector()

    print("Current Parameters:")
    print(f"  Min contour area: {robust_detector.min_contour_area}")
    print(f"  Max contour area: {robust_detector.max_contour_area}")
    print(f"  Min aspect ratio: {robust_detector.min_aspect_ratio}")
    print(f"  Max aspect ratio: {robust_detector.max_aspect_ratio}")

    # Suggestions untuk closer camera
    print("\\nSuggested optimizations for closer camera:")
    print("  📏 Increase min_contour_area (plates should be larger)")
    print("  📐 Tighten aspect ratio range (more precise detection)")
    print("  🎯 Adjust ROI untuk focus area tertentu")
    print("  ⚡ Fine-tune confidence thresholds")

if __name__ == "__main__":
    try:
        test_plate_detection_stream()
        analyze_detection_parameters()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()