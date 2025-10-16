#!/usr/bin/env python3
"""
Optimized Plate Detector untuk Closer Camera Distance
Detector yang dioptimasi khusus untuk kamera dengan jarak yang lebih dekat
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

class OptimizedCloseCameraDetector(RobustPlateDetector):
    """
    Detector yang dioptimasi untuk kamera dengan jarak dekat
    """

    def __init__(self):
        # Initialize with streaming mode untuk performance
        super().__init__(streaming_mode=True)

        # Override parameters untuk closer camera
        self.min_area = 800          # Larger minimum area (filter noise)
        self.max_area = 15000        # Reasonable maximum for closer camera
        self.min_aspect_ratio = 2.0  # Tighter range for precision
        self.max_aspect_ratio = 4.5  # Tighter range for precision
        self.min_width = 50          # Larger minimum width
        self.min_height = 20         # Larger minimum height
        self.max_width = 300         # Reasonable maximum
        self.max_height = 100        # Reasonable maximum

        # More permissive confidence untuk lebih banyak detections
        self.min_confidence = 15     # Lower confidence threshold
        self.min_text_likelihood = 25  # Lower text likelihood

        # Reduce candidates untuk speed
        self.max_candidates = 8

        # More permissive texture parameters
        self.min_edge_density = 1.0  # More permissive
        self.min_texture_variance = 3  # More permissive

        print("🎯 Optimized Close Camera Detector initialized")
        print(f"   Area range: {self.min_area} - {self.max_area}")
        print(f"   Aspect ratio: {self.min_aspect_ratio} - {self.max_aspect_ratio}")
        print(f"   Size: {self.min_width}x{self.min_height} to {self.max_width}x{self.max_height}")

def test_optimized_detector():
    """
    Test optimized detector dengan live stream
    """
    print("\n🧪 Testing Optimized Detector...")
    print("=" * 50)

    # Initialize optimized detector
    detector = OptimizedCloseCameraDetector()

    # Open camera
    cap = cv2.VideoCapture(CCTVConfig.DEFAULT_RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    # Test statistics
    total_frames = 0
    frames_with_detections = 0
    total_detections = 0
    detection_times = []

    print("📊 Testing 10 frames with optimized parameters...")

    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i+1}")
            continue

        total_frames += 1

        # Run detection
        start_time = time.time()
        detections = detector.detect_plates(frame)
        detection_time = time.time() - start_time
        detection_times.append(detection_time)

        if detections:
            frames_with_detections += 1
            total_detections += len(detections)

            print(f"\n📍 Frame {i+1}: {len(detections)} plate(s) detected ({detection_time:.3f}s)")

            for j, detection in enumerate(detections, 1):
                x, y, w, h = detection.bbox
                area = w * h
                aspect_ratio = w / h if h > 0 else 0

                print(f"   Plate {j}: '{detection.text}' (conf: {detection.confidence:.2f})")
                print(f"   BBox: ({x}, {y}, {w}, {h}) → area: {area}, ratio: {aspect_ratio:.2f}")

                # Quality indicators
                quality_indicators = []
                if 800 <= area <= 15000:
                    quality_indicators.append("✅ Good size")
                else:
                    quality_indicators.append("⚠️ Size issue")

                if 2.0 <= aspect_ratio <= 4.5:
                    quality_indicators.append("✅ Good ratio")
                else:
                    quality_indicators.append("⚠️ Ratio issue")

                print(f"   Quality: {', '.join(quality_indicators)}")
        else:
            print(f"Frame {i+1}: No detections ({detection_time:.3f}s)")

        time.sleep(0.3)  # Brief pause

    cap.release()

    # Results
    avg_detection_time = np.mean(detection_times) if detection_times else 0
    detection_rate = frames_with_detections / total_frames if total_frames > 0 else 0

    print(f"\n📊 Optimized Detector Results:")
    print(f"   Total frames: {total_frames}")
    print(f"   Frames with detections: {frames_with_detections}")
    print(f"   Total detections: {total_detections}")
    print(f"   Detection rate: {detection_rate:.1%}")
    print(f"   Average detection time: {avg_detection_time:.3f}s")

    return {
        'detection_rate': detection_rate,
        'total_detections': total_detections,
        'avg_time': avg_detection_time
    }

def apply_optimization_to_hybrid():
    """
    Apply optimizations ke HybridPlateDetector
    """
    print("\n🔧 Applying Optimizations to Hybrid Detector...")

    # Create optimized hybrid detector
    class OptimizedHybridDetector(HybridPlateDetector):
        def __init__(self):
            super().__init__(streaming_mode=True)

            # Override robust detector dengan optimized version
            self.robust_detector = OptimizedCloseCameraDetector()

            print("✅ Hybrid detector optimized untuk closer camera")

    return OptimizedHybridDetector()

def compare_detectors():
    """
    Compare original vs optimized detector
    """
    print("\n⚖️  Comparing Original vs Optimized Detector...")
    print("=" * 60)

    # Test original
    print("🔍 Testing Original Detector...")
    original_detector = RobustPlateDetector()

    # Test optimized
    print("\n🎯 Testing Optimized Detector...")
    optimized_detector = OptimizedCloseCameraDetector()

    # Quick comparison with single frame
    cap = cv2.VideoCapture(CCTVConfig.DEFAULT_RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Original detector
            start = time.time()
            original_detections = original_detector.detect_plates(frame)
            original_time = time.time() - start

            # Optimized detector
            start = time.time()
            optimized_detections = optimized_detector.detect_plates(frame)
            optimized_time = time.time() - start

            print(f"\n📊 Comparison Results:")
            print(f"   Original: {len(original_detections)} detections in {original_time:.3f}s")
            print(f"   Optimized: {len(optimized_detections)} detections in {optimized_time:.3f}s")

            if len(optimized_detections) > len(original_detections):
                print("✅ Optimized detector found more detections!")
            elif len(optimized_detections) == len(original_detections):
                print("➡️  Same number of detections")
            else:
                print("⚠️  Optimized found fewer detections")

        cap.release()

if __name__ == "__main__":
    try:
        # Test optimized detector
        results = test_optimized_detector()

        # Apply to hybrid detector
        optimized_hybrid = apply_optimization_to_hybrid()

        # Compare detectors
        compare_detectors()

        print(f"\n🎯 Final Recommendations:")
        if results['detection_rate'] > 0.3:
            print("✅ Optimized parameters working well!")
            print("   → Apply these parameters to production detector")
        else:
            print("⚠️  Detection rate still low")
            print("   → Consider further parameter tuning")
            print("   → Check if vehicles are actually in frame")
            print("   → Consider ROI implementation")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()