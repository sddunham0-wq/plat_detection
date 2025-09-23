"""
Test Script untuk Optimized Plate Detection
Script lengkap untuk test semua fitur dan validasi performa

Usage:
1. Test dengan webcam: python test_optimized_detection.py --source 0
2. Test dengan RTSP: python test_optimized_detection.py --source rtsp://url
3. Test dengan video file: python test_optimized_detection.py --source video.mp4
4. Test mode stress: python test_optimized_detection.py --source rtsp://url --stress
"""

import cv2
import numpy as np
import time
import argparse
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import threading
import queue

# Import optimized detector
from optimized_plate_detector import OptimizedPlateDetector, OptimizedRTSPStream, StableDetection

class PerformanceMonitor:
    """Monitor performa real-time dengan metrics tracking"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {
            'fps': [],
            'detection_time': [],
            'frame_processing_time': [],
            'memory_usage': [],
            'cpu_usage': []
        }
        self.start_time = time.time()

    def update(self, fps: float, detection_time: float, frame_time: float):
        """Update metrics"""
        self.metrics['fps'].append(fps)
        self.metrics['detection_time'].append(detection_time)
        self.metrics['frame_processing_time'].append(frame_time)

        # Keep only last window_size values
        for key in self.metrics:
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key].pop(0)

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        stats = {}

        for key, values in self.metrics.items():
            if values:
                stats[f'{key}_avg'] = np.mean(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_std'] = np.std(values)

        stats['uptime'] = time.time() - self.start_time
        return stats

class DetectionLogger:
    """Log detection results untuk analisis"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.detections = []

    def log_detection(self, detection: StableDetection, frame_id: int, timestamp: float):
        """Log detection event"""
        log_entry = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'track_id': detection.track_id,
            'vehicle_type': detection.vehicle_type,
            'confidence': detection.confidence,
            'stability_score': detection.stability_score,
            'is_confirmed': detection.is_confirmed,
            'bbox': detection.bbox,
            'frame_count': detection.frame_count
        }
        self.detections.append(log_entry)

    def save_log(self):
        """Save log ke file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.detections, f, indent=2)
            print(f"📄 Detection log saved: {self.log_file}")
        except Exception as e:
            print(f"❌ Failed to save log: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get detection summary"""
        if not self.detections:
            return {}

        total_detections = len(self.detections)
        confirmed_detections = len([d for d in self.detections if d['is_confirmed']])
        unique_tracks = len(set(d['track_id'] for d in self.detections if d['track_id']))

        vehicle_types = {}
        for detection in self.detections:
            vtype = detection['vehicle_type']
            vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

        avg_confidence = np.mean([d['confidence'] for d in self.detections])
        avg_stability = np.mean([d['stability_score'] for d in self.detections])

        return {
            'total_detections': total_detections,
            'confirmed_detections': confirmed_detections,
            'confirmation_rate': confirmed_detections / total_detections * 100 if total_detections > 0 else 0,
            'unique_tracks': unique_tracks,
            'vehicle_types': vehicle_types,
            'avg_confidence': avg_confidence,
            'avg_stability': avg_stability
        }

class StressTestRunner:
    """Stress test untuk evaluate performance limits"""

    def __init__(self, detector: OptimizedPlateDetector):
        self.detector = detector
        self.stress_results = {}

    def run_confidence_test(self, frame: np.ndarray) -> Dict[str, Any]:
        """Test different confidence thresholds"""
        original_threshold = self.detector.confidence_threshold
        results = {}

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            self.detector.confidence_threshold = threshold

            start_time = time.time()
            detections = self.detector.detect_vehicles_stable(frame)
            detection_time = time.time() - start_time

            results[threshold] = {
                'detection_count': len(detections),
                'confirmed_count': len([d for d in detections if d.is_confirmed]),
                'detection_time': detection_time
            }

        # Restore original threshold
        self.detector.confidence_threshold = original_threshold

        return results

    def run_fps_stress_test(self, stream, duration: int = 30) -> Dict[str, Any]:
        """Test FPS under continuous load"""
        start_time = time.time()
        frame_count = 0
        total_detection_time = 0
        peak_detections = 0

        while time.time() - start_time < duration:
            ret, frame = stream.get_frame()
            if not ret or frame is None:
                continue

            frame_count += 1

            # Detection timing
            det_start = time.time()
            detections = self.detector.detect_vehicles_stable(frame)
            detection_time = time.time() - det_start

            total_detection_time += detection_time
            peak_detections = max(peak_detections, len(detections))

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        avg_detection_time = total_detection_time / frame_count if frame_count > 0 else 0

        return {
            'duration': elapsed,
            'total_frames': frame_count,
            'avg_fps': avg_fps,
            'avg_detection_time': avg_detection_time,
            'peak_detections': peak_detections,
            'detection_overhead': (avg_detection_time / (1/avg_fps)) * 100 if avg_fps > 0 else 0
        }

def create_test_overlay(frame: np.ndarray, stats: Dict[str, Any], detections: List[StableDetection]) -> np.ndarray:
    """Create comprehensive test overlay"""
    overlay = frame.copy()

    # Background untuk text
    cv2.rectangle(overlay, (10, 10), (400, 300), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (400, 300), (0, 255, 0), 2)

    # Text lines
    lines = [
        f"🚀 OPTIMIZED PLATE DETECTION TEST",
        f"",
        f"📊 PERFORMANCE:",
        f"  FPS: {stats.get('fps_avg', 0):.1f} ± {stats.get('fps_std', 0):.1f}",
        f"  Detection Time: {stats.get('detection_time_avg', 0)*1000:.1f}ms",
        f"  Frame Time: {stats.get('frame_processing_time_avg', 0)*1000:.1f}ms",
        f"",
        f"🚗 DETECTIONS:",
        f"  Active: {len(detections)}",
        f"  Confirmed: {len([d for d in detections if d.is_confirmed])}",
        f"  Tracking: {len([d for d in detections if not d.is_confirmed])}",
        f"",
        f"⏱️  UPTIME: {stats.get('uptime', 0):.1f}s",
        f"",
        f"Controls: 'q'=quit, 's'=screenshot, 't'=stress test"
    ]

    y_offset = 30
    for line in lines:
        if line.startswith("🚀"):
            color = (0, 255, 255)  # Yellow header
            font_scale = 0.6
        elif line.startswith(("📊", "🚗", "⏱️")):
            color = (0, 255, 0)  # Green section headers
            font_scale = 0.5
        elif line.strip() == "":
            y_offset += 10
            continue
        else:
            color = (255, 255, 255)  # White text
            font_scale = 0.4

        cv2.putText(overlay, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, 1, cv2.LINE_AA)
        y_offset += 20

    return overlay

def run_test(args):
    """Main test runner"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('detection_test.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    print("🚀 Starting Optimized Plate Detection Test")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.confidence}")
    print(f"GPU: {args.gpu}")

    # Initialize components
    detector = OptimizedPlateDetector(
        confidence_threshold=args.confidence,
        enable_gpu=args.gpu
    )

    # Setup stream
    stream = None
    cap = None

    if args.source.startswith(('rtsp://', 'http://')):
        # RTSP stream
        stream = OptimizedRTSPStream(args.source, buffer_size=2, fps_limit=15)
        if not stream.start():
            print("❌ Failed to start RTSP stream")
            return
        print("✅ RTSP stream started")

    else:
        # Webcam or video file
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Failed to open source: {args.source}")
            return
        print(f"✅ Video source opened: {args.source}")

    # Initialize monitoring
    performance_monitor = PerformanceMonitor()

    # Setup detection logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"detection_log_{timestamp}.json"
    detection_logger = DetectionLogger(log_file)

    # Setup stress testing
    stress_tester = StressTestRunner(detector) if args.stress else None

    try:
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        current_fps = 0

        print("✅ Detection started. Press 'q' to quit, 's' for screenshot, 't' for stress test")

        while True:
            frame_start = time.time()

            # Get frame
            if stream:
                ret, frame = stream.get_frame()
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                if args.source.isdigit() or not args.source.startswith(('rtsp://', 'http://')):
                    break  # End of video file
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_counter += 1

            # Calculate FPS
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_counter / (current_time - last_fps_time)
                fps_counter = 0
                last_fps_time = current_time

            # Detection timing
            detection_start = time.time()
            detections = detector.detect_vehicles_stable(frame)
            detection_time = time.time() - detection_start

            # Log detections
            for detection in detections:
                detection_logger.log_detection(detection, frame_count, current_time)

            # Draw results
            annotated_frame = detector.draw_detections(frame, detections)

            # Update performance monitoring
            frame_time = time.time() - frame_start
            performance_monitor.update(current_fps, detection_time, frame_time)

            # Create test overlay
            stats = performance_monitor.get_stats()
            final_frame = create_test_overlay(annotated_frame, stats, detections)

            # Display
            cv2.imshow('Optimized Plate Detection Test', final_frame)

            # Log significant detections
            confirmed_detections = [d for d in detections if d.is_confirmed]
            if confirmed_detections:
                for det in confirmed_detections:
                    logger.info(f"🚗 CONFIRMED: {det.vehicle_type} ID:{det.track_id} "
                              f"Conf:{det.confidence:.1f}% Stability:{det.stability_score:.1f}")

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_file = f"test_screenshot_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(screenshot_file, final_frame)
                print(f"📸 Screenshot saved: {screenshot_file}")
            elif key == ord('t') and stress_tester:
                # Run stress test
                print("🔥 Running stress test...")
                stress_results = stress_tester.run_fps_stress_test(stream or cap, duration=10)
                print(f"💪 Stress test results: {stress_results}")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")

    except Exception as e:
        logger.error(f"Test error: {e}")

    finally:
        # Cleanup
        if stream:
            stream.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

        # Save detection log
        detection_logger.save_log()

        # Final report
        print("\n" + "="*60)
        print("📊 FINAL TEST REPORT")
        print("="*60)

        # Performance stats
        final_stats = performance_monitor.get_stats()
        print(f"⏱️  Total Runtime: {final_stats.get('uptime', 0):.1f}s")
        print(f"🎬 Total Frames: {frame_count}")
        print(f"📈 Average FPS: {final_stats.get('fps_avg', 0):.1f} ± {final_stats.get('fps_std', 0):.1f}")
        print(f"⚡ Avg Detection Time: {final_stats.get('detection_time_avg', 0)*1000:.1f}ms")
        print(f"🔄 Avg Frame Time: {final_stats.get('frame_processing_time_avg', 0)*1000:.1f}ms")

        # Detection stats
        detection_summary = detection_logger.get_summary()
        if detection_summary:
            print(f"\n🚗 DETECTION SUMMARY:")
            print(f"   Total Detections: {detection_summary['total_detections']}")
            print(f"   Confirmed: {detection_summary['confirmed_detections']}")
            print(f"   Confirmation Rate: {detection_summary['confirmation_rate']:.1f}%")
            print(f"   Unique Tracks: {detection_summary['unique_tracks']}")
            print(f"   Avg Confidence: {detection_summary['avg_confidence']:.1f}%")
            print(f"   Avg Stability: {detection_summary['avg_stability']:.1f}")

            print(f"\n🚙 VEHICLE TYPES:")
            for vtype, count in detection_summary['vehicle_types'].items():
                print(f"   {vtype}: {count}")

        # Detector stats
        detector_stats = detector.get_statistics()
        print(f"\n🎯 DETECTOR STATS:")
        print(f"   Device: {detector_stats['device']}")
        print(f"   Confidence Threshold: {detector_stats['confidence_threshold']}")
        print(f"   Total Detections: {detector_stats['total_detections']}")
        print(f"   Confirmed Detections: {detector_stats['confirmed_detections']}")
        print(f"   Active Tracks: {detector_stats['active_tracks']}")

        # Stream stats (if RTSP)
        if stream:
            stream_stats = stream.get_stats()
            print(f"\n📡 STREAM STATS:")
            print(f"   Total Frames: {stream_stats['total_frames']}")
            print(f"   Dropped Frames: {stream_stats['dropped_frames']}")
            print(f"   Drop Rate: {stream_stats['drop_rate_percent']:.1f}%")
            print(f"   Stream FPS: {stream_stats['fps_actual']:.1f}")

        print("="*60)
        print(f"📄 Detection log saved: {log_file}")
        print("✅ Test completed successfully!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Test Optimized Plate Detection')

    parser.add_argument('--source', default='0',
                       help='Video source (0 for webcam, RTSP URL, or video file)')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Detection confidence threshold (default: 0.7)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Enable GPU acceleration (default: True)')
    parser.add_argument('--stress', action='store_true',
                       help='Enable stress testing mode')

    args = parser.parse_args()

    # Validate source
    if not args.source.startswith(('rtsp://', 'http://')) and not args.source.isdigit() and not os.path.exists(args.source):
        print(f"❌ Invalid source: {args.source}")
        return

    run_test(args)

if __name__ == "__main__":
    main()