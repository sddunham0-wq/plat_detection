#!/usr/bin/env python3
"""
Analisis Bounding Box Quality untuk Plate Detection
Dengan kamera yang lebih dekat, kita perlu mengoptimasi parameter deteksi
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

def analyze_detection_parameters():
    """
    Analisis parameter deteksi saat ini dan suggest optimizations
    """
    print("🔧 Detection Parameter Analysis")
    print("=" * 50)

    # Initialize detectors
    hybrid_detector = HybridPlateDetector()
    robust_detector = RobustPlateDetector()

    print("\n📋 Current Parameters:")
    print(f"   Min area: {robust_detector.min_area}")
    print(f"   Max area: {robust_detector.max_area}")
    print(f"   Min aspect ratio: {robust_detector.min_aspect_ratio}")
    print(f"   Max aspect ratio: {robust_detector.max_aspect_ratio}")
    print(f"   Mode: {'STREAMING' if robust_detector.streaming_mode else 'FULL'}")
    print(f"   Min width: {robust_detector.min_width}, Max width: {robust_detector.max_width}")
    print(f"   Min height: {robust_detector.min_height}, Max height: {robust_detector.max_height}")

    # Analyze with closer camera distance
    print("\n🎯 Optimizations for Closer Camera:")
    print("   📏 Larger plates expected → increase min_area")
    print("   📐 Better precision → tighten aspect ratio range")
    print("   🔍 Higher resolution → adjust detection sensitivity")

    return robust_detector

def test_multiple_frames():
    """
    Test deteksi pada multiple frames untuk analysis
    """
    print("\n🎬 Testing Multiple Frames...")

    # Open camera
    cap = cv2.VideoCapture(CCTVConfig.DEFAULT_RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    detector = HybridPlateDetector()

    # Statistics
    total_frames = 0
    frames_with_detections = 0
    total_detections = 0
    detection_areas = []
    aspect_ratios = []

    print("📊 Analyzing 20 frames...")

    for i in range(20):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i+1}")
            continue

        total_frames += 1

        # Run detection
        start_time = time.time()
        detections = detector.detect_plates(frame)
        detection_time = time.time() - start_time

        if detections:
            frames_with_detections += 1
            total_detections += len(detections)

            print(f"\n📍 Frame {i+1}: {len(detections)} plate(s) detected ({detection_time:.3f}s)")

            for j, detection in enumerate(detections, 1):
                x, y, w, h = detection.bbox
                confidence = detection.confidence
                text = detection.text

                # Calculate metrics
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                detection_areas.append(area)
                aspect_ratios.append(aspect_ratio)

                print(f"   Plate {j}: '{text}' (conf: {confidence:.2f})")
                print(f"   BBox: ({x}, {y}, {w}, {h})")
                print(f"   Area: {area}, Aspect Ratio: {aspect_ratio:.2f}")

                # Quality assessment
                if area < 1000:
                    print("   ⚠️  Small detection - might be noise")
                elif area > 50000:
                    print("   ⚠️  Large detection - might be false positive")
                else:
                    print("   ✅ Good size detection")

                if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                    print("   ⚠️  Unusual aspect ratio")
                else:
                    print("   ✅ Good aspect ratio")
        else:
            if i % 5 == 0:  # Print every 5th frame
                print(f"Frame {i+1}: No detections ({detection_time:.3f}s)")

        time.sleep(0.2)  # Brief pause

    cap.release()

    # Final statistics
    print(f"\n📊 Analysis Results:")
    print(f"   Total frames: {total_frames}")
    print(f"   Frames with detections: {frames_with_detections}")
    print(f"   Total plates detected: {total_detections}")
    print(f"   Detection rate: {(frames_with_detections/total_frames*100):.1f}%")

    if detection_areas:
        print(f"   Average detection area: {np.mean(detection_areas):.0f}")
        print(f"   Area range: {min(detection_areas):.0f} - {max(detection_areas):.0f}")

    if aspect_ratios:
        print(f"   Average aspect ratio: {np.mean(aspect_ratios):.2f}")
        print(f"   Aspect ratio range: {min(aspect_ratios):.2f} - {max(aspect_ratios):.2f}")

    return {
        'detection_rate': frames_with_detections/total_frames if total_frames > 0 else 0,
        'avg_area': np.mean(detection_areas) if detection_areas else 0,
        'avg_aspect_ratio': np.mean(aspect_ratios) if aspect_ratios else 0,
        'areas': detection_areas,
        'aspect_ratios': aspect_ratios
    }

def suggest_optimizations(stats):
    """
    Suggest parameter optimizations berdasarkan analysis
    """
    print("\n🎯 Optimization Suggestions:")
    print("=" * 40)

    detection_rate = stats['detection_rate']
    avg_area = stats['avg_area']
    avg_aspect_ratio = stats['avg_aspect_ratio']

    if detection_rate < 0.3:
        print("📉 Low detection rate - consider:")
        print("   • Lowering confidence thresholds")
        print("   • Reducing min_area")
        print("   • Expanding aspect ratio range")

    if avg_area > 0:
        if avg_area < 2000:
            print("📏 Small average detection area:")
            print(f"   • Current avg: {avg_area:.0f}")
            print("   • Consider lowering min_area to 800-1200")

        elif avg_area > 20000:
            print("📏 Large average detection area:")
            print(f"   • Current avg: {avg_area:.0f}")
            print("   • Consider raising min_area to 2000-3000")

        else:
            print("✅ Good average detection area")

    if avg_aspect_ratio > 0:
        if avg_aspect_ratio < 2.0 or avg_aspect_ratio > 5.0:
            print("📐 Unusual aspect ratios detected:")
            print(f"   • Current avg: {avg_aspect_ratio:.2f}")
            print("   • Consider tightening range to 2.0-4.5")
        else:
            print("✅ Good aspect ratio range")

    # Camera-specific suggestions
    print("\n🎥 Camera Distance Optimizations:")
    print("   • Closer camera = larger plates in frame")
    print("   • Increase min_area untuk filter noise")
    print("   • Tighten aspect ratio untuk better precision")
    print("   • Consider ROI untuk focus area tertentu")

if __name__ == "__main__":
    try:
        # Analyze current parameters
        robust_detector = analyze_detection_parameters()

        # Test with multiple frames
        stats = test_multiple_frames()

        # Suggest optimizations
        suggest_optimizations(stats)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()