#!/usr/bin/env python3
"""
Final Stable Detector
Solusi final untuk deteksi plat yang stabil menggunakan komponen existing
dengan parameter yang dioptimasi untuk immediate response
"""

import cv2
import time
import sys
import os

# Import detector yang sudah ada dan proven work
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def run_optimized_detection():
    """Run detection dengan parameter yang sudah dioptimasi"""
    print("🎯 FINAL STABLE PLATE DETECTOR")
    print("=" * 60)
    print("🔧 Using existing proven components with optimized settings")
    print("⚡ Focus: Immediate detection + High stability")
    print("=" * 60)

    # Import dan setup detector yang sudah ada
    try:
        from utils.hybrid_plate_detector import HybridPlateDetector
        detector = HybridPlateDetector(streaming_mode=True)
        print("✅ Using Hybrid Plate Detector")
    except ImportError:
        try:
            from utils.plate_detector import LicensePlateDetector
            detector = LicensePlateDetector()
            print("✅ Using License Plate Detector")
        except ImportError:
            print("❌ No detector available")
            return

    # RTSP settings
    rtsp_url = "rtsp://admin:H4nd4l9165!@192.168.1.195:554/85"

    print(f"🎥 Connecting to RTSP: {rtsp_url}")

    # Initialize capture
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ Cannot connect to RTSP stream")
        return

    # Optimize capture settings untuk stability
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)

    print("✅ Connected successfully!")
    print("🎬 Starting optimized detection...")
    print("📋 Press 'q' to quit, 'r' to reset stats")
    print("-" * 60)

    # Detection settings
    frame_skip = 1  # Process every frame untuk immediate response
    confidence_threshold = 40  # Lowered untuk catch more plates

    # Statistics
    frame_count = 0
    detection_count = 0
    valid_detection_count = 0
    start_time = time.time()
    processing_times = []
    detected_plates = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame read failed")
                continue

            frame_count += 1

            # Skip frames untuk performance jika diperlukan
            if frame_count % frame_skip != 0:
                continue

            # Process frame
            process_start = time.time()

            try:
                detections = detector.detect_plates(frame)
                processing_time = time.time() - process_start
                processing_times.append(processing_time)

                # Filter valid detections
                valid_detections = []
                for detection in detections:
                    # Check apakah detection valid
                    text = getattr(detection, 'text', '')
                    confidence = getattr(detection, 'confidence', 0)

                    # Validation rules
                    if (len(text) >= 5 and
                        len(text) <= 12 and
                        confidence >= confidence_threshold):

                        valid_detections.append(detection)
                        detection_count += 1

                        # Track unique plates
                        if text not in detected_plates:
                            detected_plates.add(text)
                            valid_detection_count += 1

                            # IMMEDIATE FEEDBACK untuk new plate
                            print(f"🚗 NEW PLATE DETECTED: {text}")
                            print(f"   📊 Confidence: {confidence:.1f}%")
                            print(f"   ⚡ Processing: {processing_time:.3f}s")
                            print(f"   🎬 Frame: {frame_count}")
                            print(f"   📈 Total unique plates: {len(detected_plates)}")

                            # Response quality indicator
                            if processing_time <= 0.3:
                                print(f"   ✅ EXCELLENT RESPONSE TIME")
                            elif processing_time <= 0.5:
                                print(f"   ⚠️  GOOD RESPONSE TIME")
                            else:
                                print(f"   ❌ SLOW RESPONSE TIME")

                            print("-" * 50)

                # Annotate frame
                display_frame = frame.copy()
                for detection in valid_detections:
                    if hasattr(detection, 'bbox'):
                        x, y, w, h = detection.bbox

                        # Color coding
                        confidence = getattr(detection, 'confidence', 0)
                        if confidence >= 70:
                            color = (0, 255, 0)  # Green - high confidence
                        elif confidence >= 50:
                            color = (0, 255, 255)  # Yellow - medium
                        else:
                            color = (0, 165, 255)  # Orange - low but valid

                        # Draw detection
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                        # Label
                        label = f"{detection.text} ({confidence:.0f}%)"
                        cv2.putText(display_frame, label, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                print(f"⚠️  Processing error: {e}")
                processing_time = 0

            # Real-time stats overlay
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            avg_processing = sum(processing_times) / len(processing_times) if processing_times else 0

            stats = [
                f"FPS: {fps:.1f}",
                f"Frames: {frame_count}",
                f"Detections: {detection_count}",
                f"Unique Plates: {len(detected_plates)}",
                f"Processing: {processing_time:.3f}s",
                f"Avg Processing: {avg_processing:.3f}s"
            ]

            for i, stat in enumerate(stats):
                cv2.putText(display_frame, stat, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Performance indicator
            if avg_processing <= 0.3:
                perf_text = "EXCELLENT"
                perf_color = (0, 255, 0)
            elif avg_processing <= 0.5:
                perf_text = "GOOD"
                perf_color = (0, 255, 255)
            else:
                perf_text = "NEEDS IMPROVEMENT"
                perf_color = (0, 0, 255)

            cv2.putText(display_frame, perf_text, (10, display_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, perf_color, 2)

            # Resize untuk display
            height, width = display_frame.shape[:2]
            if width > 900:
                scale = 900 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))

            cv2.imshow('Final Stable Detection', display_frame)

            # Periodic console stats
            if frame_count % 100 == 0:
                print(f"📊 Status Update (Frame {frame_count}):")
                print(f"   FPS: {fps:.1f}")
                print(f"   Unique plates detected: {len(detected_plates)}")
                print(f"   Average processing time: {avg_processing:.3f}s")
                print(f"   Detection rate: {detection_count/frame_count:.4f} per frame")
                print("-" * 40)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset stats
                detection_count = 0
                valid_detection_count = 0
                detected_plates.clear()
                processing_times.clear()
                start_time = time.time()
                frame_count = 0
                print("📊 Statistics reset!")

    except KeyboardInterrupt:
        print("\n⏹️  Detection stopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Final comprehensive results
    elapsed = time.time() - start_time
    avg_processing = sum(processing_times) / len(processing_times) if processing_times else 0

    print("\n" + "=" * 60)
    print("📊 FINAL STABLE DETECTION RESULTS")
    print("=" * 60)

    print(f"🎬 Session Summary:")
    print(f"   Duration: {elapsed:.1f} seconds")
    print(f"   Total frames: {frame_count}")
    print(f"   Average FPS: {frame_count/elapsed:.2f}")

    print(f"\n🎯 Detection Performance:")
    print(f"   Total detections: {detection_count}")
    print(f"   Unique plates found: {len(detected_plates)}")
    print(f"   Detection rate: {detection_count/frame_count:.4f} per frame")

    print(f"\n⚡ Processing Performance:")
    print(f"   Average processing time: {avg_processing:.3f}s")
    print(f"   Fastest processing: {min(processing_times):.3f}s" if processing_times else "N/A")
    print(f"   Slowest processing: {max(processing_times):.3f}s" if processing_times else "N/A")

    print(f"\n🚗 Detected Plates:")
    for plate in sorted(detected_plates):
        print(f"   {plate}")

    print(f"\n🎯 FINAL VERDICT:")
    if len(detected_plates) > 0 and avg_processing <= 0.5:
        print("   ✅ DETECTION IS STABLE AND WORKING WELL!")
        print("   🎉 System ready for production use")
    elif len(detected_plates) > 0:
        print("   ⚠️  DETECTION WORKS BUT PERFORMANCE COULD BE BETTER")
        print("   🔧 Consider optimizing processing pipeline")
    else:
        print("   ❌ NO PLATES DETECTED")
        print("   🔍 Check camera angle, lighting, or detection parameters")

    print("=" * 60)

if __name__ == "__main__":
    run_optimized_detection()