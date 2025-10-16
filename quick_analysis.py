#!/usr/bin/env python3
"""
Quick Analysis untuk Bounding Box Quality
Analisis cepat parameter deteksi dengan kamera yang lebih dekat
"""

import cv2
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hybrid_plate_detector import HybridPlateDetector
from utils.robust_plate_detector import RobustPlateDetector
from config import CCTVConfig

def analyze_current_settings():
    """
    Analisis setting deteksi saat ini
    """
    print("🔧 Current Detection Settings Analysis")
    print("=" * 50)

    # Initialize detector
    robust_detector = RobustPlateDetector()

    print(f"📋 Current Parameters:")
    print(f"   Min area: {robust_detector.min_area}")
    print(f"   Max area: {robust_detector.max_area}")
    print(f"   Aspect ratio: {robust_detector.min_aspect_ratio} - {robust_detector.max_aspect_ratio}")
    print(f"   Size limits: {robust_detector.min_width}x{robust_detector.min_height} to {robust_detector.max_width}x{robust_detector.max_height}")
    print(f"   Mode: {'STREAMING' if robust_detector.streaming_mode else 'FULL'}")

    return robust_detector

def test_quick_detection():
    """
    Test cepat deteksi dengan beberapa frame
    """
    print("\n🎬 Quick Detection Test...")

    # Open camera
    cap = cv2.VideoCapture(CCTVConfig.DEFAULT_RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    detector = HybridPlateDetector()

    # Test 5 frames quickly
    total_detections = 0
    frames_tested = 0
    best_detection = None

    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            continue

        frames_tested += 1
        start_time = time.time()
        detections = detector.detect_plates(frame)
        detection_time = time.time() - start_time

        if detections:
            total_detections += len(detections)
            print(f"Frame {i+1}: {len(detections)} detections in {detection_time:.3f}s")

            # Analyze best detection
            for detection in detections:
                x, y, w, h = detection.bbox
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                print(f"   Text: '{detection.text}' (conf: {detection.confidence:.2f})")
                print(f"   BBox: ({x}, {y}, {w}, {h}) → area: {area}, ratio: {aspect_ratio:.2f}")

                if best_detection is None or detection.confidence > best_detection.confidence:
                    best_detection = detection
        else:
            print(f"Frame {i+1}: No detections ({detection_time:.3f}s)")

        time.sleep(0.5)  # Brief pause

    cap.release()

    print(f"\n📊 Quick Results:")
    print(f"   Frames tested: {frames_tested}")
    print(f"   Total detections: {total_detections}")
    print(f"   Detection rate: {(total_detections > 0)}")

    return best_detection

def suggest_optimizations_for_closer_camera():
    """
    Suggest optimizations berdasarkan jarak kamera yang lebih dekat
    """
    print("\n🎯 Optimizations for Closer Camera Distance")
    print("=" * 50)

    print("📏 Expected Changes dengan Closer Camera:")
    print("   • Plat nomor akan terlihat lebih besar di frame")
    print("   • Detail karakter akan lebih jelas")
    print("   • Bounding box seharusnya lebih akurat")
    print("   • Less background noise dalam detection area")

    print("\n🔧 Recommended Parameter Adjustments:")
    print("   • Increase min_area dari 200 ke 800-1200 (filter small noise)")
    print("   • Tighten aspect_ratio range ke 2.0-4.5 (more precise)")
    print("   • Increase min_width dari 30 ke 60 (larger plates expected)")
    print("   • Increase min_height dari 12 ke 25 (larger plates expected)")
    print("   • Consider using STREAMING mode untuk real-time (faster)")

    print("\n⚡ Performance Optimizations:")
    print("   • Set streaming_mode=True untuk faster processing")
    print("   • Reduce max_candidates untuk speed")
    print("   • Focus on horizontal detection first")
    print("   • Consider ROI untuk specific area")

def create_optimized_detector():
    """
    Create detector dengan parameter yang dioptimasi untuk closer camera
    """
    print("\n🛠️ Creating Optimized Detector for Closer Camera...")

    # Create optimized detector
    detector = RobustPlateDetector(streaming_mode=True)  # Use streaming for speed

    # Adjust parameters for closer camera
    detector.min_area = 1000  # Larger minimum area
    detector.max_area = 12000  # Reasonable maximum
    detector.min_aspect_ratio = 2.0  # Tighter range
    detector.max_aspect_ratio = 4.5
    detector.min_width = 60  # Larger minimum width
    detector.min_height = 25  # Larger minimum height
    detector.max_width = 250  # Reasonable maximum
    detector.max_height = 80

    print("✅ Optimized parameters:")
    print(f"   Area: {detector.min_area} - {detector.max_area}")
    print(f"   Aspect ratio: {detector.min_aspect_ratio} - {detector.max_aspect_ratio}")
    print(f"   Size: {detector.min_width}x{detector.min_height} to {detector.max_width}x{detector.max_height}")

    return detector

if __name__ == "__main__":
    try:
        # Analyze current settings
        current_detector = analyze_current_settings()

        # Test current detection
        best_detection = test_quick_detection()

        # Suggest optimizations
        suggest_optimizations_for_closer_camera()

        # Create optimized detector
        optimized_detector = create_optimized_detector()

        print("\n✅ Analysis Complete!")
        if best_detection:
            print(f"🎯 Best detection found: '{best_detection.text}' (conf: {best_detection.confidence:.2f})")
        else:
            print("⚠️  No detections found - parameter optimization needed")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()