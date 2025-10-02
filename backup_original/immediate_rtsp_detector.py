#!/usr/bin/env python3
"""
Immediate RTSP Plate Detector
Detector real-time yang langsung mendeteksi plat dari RTSP stream
dengan focus pada immediate response dan stabilitas
"""

import cv2
import numpy as np
import time
import sys
import os
import threading
from collections import deque

# Import existing detector yang sudah proven work
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from utils.plate_detector import LicensePlateDetector
except ImportError:
    try:
        from utils.robust_plate_detector import RobustPlateDetector as LicensePlateDetector
    except ImportError:
        print("❌ No suitable detector found")
        sys.exit(1)

class ImmediateRTSPDetector:
    """
    Real-time RTSP plate detector dengan immediate response
    """

    def __init__(self, rtsp_url=None, confidence_threshold=40):
        """Initialize detector dengan existing proven components"""
        self.rtsp_url = rtsp_url or "rtsp://admin:H4nd4l9165!@192.168.1.195:554/85"
        self.confidence_threshold = confidence_threshold

        # Use existing proven plate detector
        self.plate_detector = LicensePlateDetector()

        # Immediate response settings
        self.frame_skip = 2  # Process every 2nd frame untuk speed
        self.detection_buffer = deque(maxlen=3)  # Small buffer untuk consistency

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.total_processing_time = 0
        self.last_detection_text = ""
        self.last_detection_time = 0

        # Threading untuk non-blocking detection
        self.processing = False
        self.latest_frame = None
        self.latest_detections = []

        print("🎯 Immediate RTSP Detector initialized")
        print(f"📡 RTSP URL: {self.rtsp_url}")
        print(f"🎚️  Confidence threshold: {confidence_threshold}%")

    def start_detection(self):
        """Start immediate detection process"""
        print("🚀 Starting immediate detection...")
        print("=" * 60)

        # Initialize video capture
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print("❌ Cannot connect to RTSP stream")
            return

        # Optimize capture settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)

        print("✅ Connected to RTSP stream")
        print("🎬 Press 'q' to quit, 's' to show stats")
        print("-" * 60)

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Frame read failed")
                    continue

                self.frame_count += 1

                # Skip frames untuk performance
                if self.frame_count % self.frame_skip != 0:
                    continue

                # Process frame immediately
                processing_start = time.time()
                detections = self._process_frame_immediate(frame)
                processing_time = time.time() - processing_start
                self.total_processing_time += processing_time

                # Handle detections immediately
                for detection in detections:
                    self._handle_immediate_detection(detection, processing_time)

                # Show frame dengan annotations
                display_frame = self._annotate_frame(frame, detections, processing_time)
                cv2.imshow('Immediate RTSP Detection', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_stats(time.time() - start_time)

        except KeyboardInterrupt:
            print("\n⏹️  Detection stopped by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()

        # Final stats
        elapsed = time.time() - start_time
        self._print_final_stats(elapsed)

    def _process_frame_immediate(self, frame):
        """Process frame dengan focus pada immediate detection"""
        try:
            # Use existing detector yang sudah proven
            detections = self.plate_detector.detect_plates(frame)

            # Filter by confidence
            valid_detections = []
            for detection in detections:
                if hasattr(detection, 'confidence') and detection.confidence >= self.confidence_threshold:
                    if hasattr(detection, 'text') and len(detection.text) >= 4:
                        valid_detections.append(detection)
                elif hasattr(detection, 'text') and len(detection.text) >= 5:
                    # Even without high confidence, accept if text looks valid
                    valid_detections.append(detection)

            return valid_detections

        except Exception as e:
            print(f"⚠️  Processing error: {e}")
            return []

    def _handle_immediate_detection(self, detection, processing_time):
        """Handle detection dengan immediate feedback"""
        current_time = time.time()

        # Prevent duplicate immediate announcements
        if (detection.text != self.last_detection_text or
            current_time - self.last_detection_time > 5):  # 5 second cooldown

            self.detection_count += 1
            self.last_detection_text = detection.text
            self.last_detection_time = current_time

            # Immediate console feedback
            confidence = getattr(detection, 'confidence', 0)
            print(f"🚗 IMMEDIATE DETECTION: {detection.text}")
            print(f"   📊 Confidence: {confidence:.1f}%")
            print(f"   ⚡ Processing: {processing_time:.3f}s")
            print(f"   🎬 Frame: {self.frame_count}")

            # Performance indicator
            if processing_time <= 0.2:
                print(f"   ✅ FAST RESPONSE")
            elif processing_time <= 0.5:
                print(f"   ⚠️  MODERATE RESPONSE")
            else:
                print(f"   ❌ SLOW RESPONSE")

            print("-" * 40)

    def _annotate_frame(self, frame, detections, processing_time):
        """Annotate frame dengan detection results"""
        annotated = frame.copy()

        # Draw detections
        for detection in detections:
            if hasattr(detection, 'bbox'):
                x, y, w, h = detection.bbox
            else:
                # Fallback jika bbox format berbeda
                continue

            # Color based pada confidence/quality
            confidence = getattr(detection, 'confidence', 50)
            if confidence >= 70:
                color = (0, 255, 0)  # Green - high confidence
            elif confidence >= 50:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence but valid

            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

            # Add text
            label = f"{detection.text}"
            if hasattr(detection, 'confidence'):
                label += f" ({detection.confidence:.0f}%)"

            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Add real-time stats
        elapsed = time.time() - getattr(self, 'start_time', time.time())
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        avg_processing = self.total_processing_time / max(1, self.frame_count // self.frame_skip)

        stats_text = [
            f"FPS: {fps:.1f}",
            f"Detections: {self.detection_count}",
            f"Processing: {processing_time:.3f}s",
            f"Avg: {avg_processing:.3f}s"
        ]

        for i, stat in enumerate(stats_text):
            cv2.putText(annotated, stat, (10, 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Response time indicator
        if processing_time <= 0.2:
            response_text = "IMMEDIATE"
            response_color = (0, 255, 0)
        elif processing_time <= 0.5:
            response_text = "FAST"
            response_color = (0, 255, 255)
        else:
            response_text = "SLOW"
            response_color = (0, 0, 255)

        cv2.putText(annotated, response_text, (10, annotated.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, response_color, 3)

        return annotated

    def _print_stats(self, elapsed):
        """Print current statistics"""
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        processed_frames = self.frame_count // self.frame_skip
        avg_processing = self.total_processing_time / max(1, processed_frames)

        print(f"\n📊 CURRENT STATS (after {elapsed:.1f}s):")
        print(f"   🎬 Total frames: {self.frame_count}")
        print(f"   ⚡ FPS: {fps:.1f}")
        print(f"   🔍 Processed frames: {processed_frames}")
        print(f"   🚗 Total detections: {self.detection_count}")
        print(f"   ⏱️  Avg processing: {avg_processing:.3f}s")
        print(f"   📈 Detection rate: {self.detection_count/processed_frames:.3f} per processed frame")
        print("-" * 40)

    def _print_final_stats(self, elapsed):
        """Print final comprehensive statistics"""
        processed_frames = self.frame_count // self.frame_skip
        avg_processing = self.total_processing_time / max(1, processed_frames)
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("📊 IMMEDIATE DETECTION FINAL RESULTS")
        print("=" * 60)

        print(f"🎬 Session Duration: {elapsed:.1f} seconds")
        print(f"📹 Total Frames: {self.frame_count}")
        print(f"🔍 Processed Frames: {processed_frames}")
        print(f"⚡ Average FPS: {fps:.2f}")

        print(f"\n🎯 DETECTION PERFORMANCE:")
        print(f"   Total detections: {self.detection_count}")
        print(f"   Detection rate: {self.detection_count/processed_frames:.3f} per frame")
        print(f"   Average processing time: {avg_processing:.3f}s")

        print(f"\n⚡ IMMEDIATE RESPONSE ANALYSIS:")
        target_time = 0.3
        if avg_processing <= target_time:
            print(f"   ✅ EXCELLENT: Response time {avg_processing:.3f}s <= {target_time}s")
        elif avg_processing <= 0.5:
            print(f"   ⚠️  GOOD: Response time {avg_processing:.3f}s (acceptable)")
        else:
            print(f"   ❌ NEEDS IMPROVEMENT: Response time {avg_processing:.3f}s")

        print(f"\n🎯 IMMEDIATE DETECTION VERDICT:")
        if self.detection_count > 0 and avg_processing <= 0.5:
            print("   ✅ DETECTION IS WORKING AND RESPONSIVE")
        elif self.detection_count > 0:
            print("   ⚠️  DETECTION WORKS BUT COULD BE FASTER")
        else:
            print("   ❌ NO DETECTIONS - CHECK CAMERA ANGLE OR SETTINGS")

        print("=" * 60)

def main():
    """Main function untuk immediate detection test"""
    import argparse

    parser = argparse.ArgumentParser(description='Immediate RTSP Plate Detection')
    parser.add_argument('--rtsp', type=str,
                       default="rtsp://admin:H4nd4l9165!@192.168.1.195:554/85",
                       help='RTSP stream URL')
    parser.add_argument('--confidence', type=int, default=40,
                       help='Minimum confidence threshold')

    args = parser.parse_args()

    print("🚀 IMMEDIATE RTSP PLATE DETECTOR")
    print("=" * 60)
    print("🎯 Goal: Immediate detection when plates are visible")
    print("⚡ Method: Optimized existing detector + immediate feedback")
    print("=" * 60)

    detector = ImmediateRTSPDetector(args.rtsp, args.confidence)
    detector.start_detection()

if __name__ == "__main__":
    main()