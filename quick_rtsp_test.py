#!/usr/bin/env python3
"""
Quick RTSP Test Script
Script simpel untuk test RTSP stream dan deteksi cepat

Usage:
    python quick_rtsp_test.py

Controls:
    'q' = quit
    's' = screenshot
    'c' = change confidence threshold
    'v' = toggle vehicles only mode
"""

import cv2
import numpy as np
import time
import sys
import os

# Set PYTHONPATH untuk import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_plate_detector import OptimizedPlateDetector, OptimizedRTSPStream

def main():
    print("🚀 Quick RTSP Detection Test")
    print("=" * 50)

    # RTSP URL kamera Anda
    rtsp_url = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"

    print(f"📡 Connecting to RTSP: {rtsp_url}")

    # Initialize components
    try:
        detector = OptimizedPlateDetector(confidence_threshold=0.6, enable_gpu=True)
        stream = OptimizedRTSPStream(rtsp_url, buffer_size=3, fps_limit=10)

        if not stream.start():
            print("❌ Failed to start RTSP stream")
            return

        print("✅ RTSP stream connected!")
        print("✅ Detector initialized!")
        print("\nControls:")
        print("  'q' = quit")
        print("  's' = screenshot")
        print("  'c' = change confidence")
        print("  'v' = toggle vehicles only")
        print("\n🎬 Starting detection...")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Main loop
    frame_count = 0
    last_time = time.time()
    fps = 0
    vehicles_only = True
    confidence_levels = [0.3, 0.5, 0.6, 0.7, 0.8]
    current_conf_idx = 2  # Start with 0.6

    try:
        while True:
            # Get frame
            ret, frame = stream.get_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Calculate FPS
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time) if current_time - last_time > 0 else 0
                frame_count = 0
                last_time = current_time

            # Detection
            detection_start = time.time()
            detections = detector.detect_vehicles_stable(frame)
            detection_time = (time.time() - detection_start) * 1000

            # Draw results
            annotated_frame = detector.draw_detections(frame, detections)

            # Add overlay info
            info_lines = [
                f"FPS: {fps:.1f}",
                f"Detection: {detection_time:.1f}ms",
                f"Confidence: {detector.confidence_threshold}",
                f"Vehicles Only: {vehicles_only}",
                f"Detections: {len(detections)}",
                f"Confirmed: {len([d for d in detections if d.is_confirmed])}",
                f"",
                f"Controls: q=quit, s=screenshot, c=confidence, v=vehicles"
            ]

            # Draw info overlay
            y_start = 30
            for i, line in enumerate(info_lines):
                if line.strip() == "":
                    y_start += 10
                    continue

                color = (0, 255, 0) if not line.startswith("Controls") else (255, 255, 0)
                cv2.putText(annotated_frame, line, (10, y_start + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show frame
            cv2.imshow('Quick RTSP Detection Test', annotated_frame)

            # Log detections
            if detections:
                confirmed = [d for d in detections if d.is_confirmed]
                if confirmed:
                    for det in confirmed:
                        print(f"🚗 CONFIRMED: {det.vehicle_type} ID:{det.track_id} "
                              f"Conf:{det.confidence:.1f}% Stability:{det.stability_score:.1f}")

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n⏹️ Quitting...")
                break
            elif key == ord('s'):
                # Screenshot
                timestamp = int(time.time())
                filename = f"rtsp_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('c'):
                # Change confidence
                current_conf_idx = (current_conf_idx + 1) % len(confidence_levels)
                detector.confidence_threshold = confidence_levels[current_conf_idx]
                print(f"🎯 Confidence threshold changed to: {detector.confidence_threshold}")
            elif key == ord('v'):
                # Toggle vehicles only (this affects the display, actual detection is always vehicles)
                vehicles_only = not vehicles_only
                print(f"🚙 Vehicles only mode: {vehicles_only}")

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error during detection: {e}")
    finally:
        # Cleanup
        stream.stop()
        cv2.destroyAllWindows()

        # Final stats
        stream_stats = stream.get_stats()
        detector_stats = detector.get_statistics()

        print("\n📊 Final Statistics:")
        print(f"   Stream FPS: {stream_stats['fps_actual']:.1f}")
        print(f"   Drop Rate: {stream_stats['drop_rate_percent']:.1f}%")
        print(f"   Total Detections: {detector_stats['total_detections']}")
        print(f"   Confirmed Detections: {detector_stats['confirmed_detections']}")
        print(f"   Confirmation Rate: {detector_stats['confirmation_rate_percent']:.1f}%")
        print(f"   Active Tracks: {detector_stats['active_tracks']}")

        print("\n✅ Test completed!")

if __name__ == "__main__":
    main()