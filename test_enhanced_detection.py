#!/usr/bin/env python3
"""
Test Enhanced Plate Detection
Test akurasi tinggi untuk deteksi kendaraan dan plat nomor
"""
import cv2
import os
import sys
import time
import numpy as np
from enhanced_plate_detector import EnhancedPlateDetector

def test_enhanced_detection():
    """Test enhanced detection dengan berbagai skenario"""
    print("🚀 Testing Enhanced Plate Detection System")
    print("=" * 50)

    # Initialize enhanced detector
    try:
        detector = EnhancedPlateDetector('enhanced_detection_config.ini')
        print("✅ Enhanced detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

    # Test scenarios
    test_scenarios = [
        {
            'name': 'Webcam Real-time Test',
            'type': 'webcam',
            'source': 0
        },
        {
            'name': 'Image File Test',
            'type': 'image',
            'source': 'test_images/'
        },
        {
            'name': 'Video File Test',
            'type': 'video',
            'source': 'test_videos/'
        }
    ]

    results = {}

    for scenario in test_scenarios:
        print(f"\n🧪 Running {scenario['name']}...")
        try:
            if scenario['type'] == 'webcam':
                result = test_webcam(detector, scenario['source'])
            elif scenario['type'] == 'image':
                result = test_images(detector, scenario['source'])
            elif scenario['type'] == 'video':
                result = test_videos(detector, scenario['source'])

            results[scenario['name']] = result
            print(f"✅ {scenario['name']} completed")

        except Exception as e:
            print(f"❌ {scenario['name']} failed: {e}")
            results[scenario['name']] = {'error': str(e)}

    # Print final results
    print_test_results(results)

    return True

def test_webcam(detector, source):
    """Test dengan webcam"""
    print(f"📹 Testing with webcam (source: {source})")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Cannot open webcam {source}")

    # Test parameters
    test_duration = 30  # seconds
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    processing_times = []

    print("🎬 Recording for 30 seconds...")
    print("   Press 'q' to quit early")
    print("   Press 's' to show statistics")

    while time.time() - start_time < test_duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # Process frame dengan enhanced detector
        results = detector.process_frame_enhanced(frame)

        processing_time = time.time() - frame_start
        processing_times.append(processing_time)

        frame_count += 1
        detection_count += len(results)

        # Draw results
        output_frame = detector.draw_enhanced_results(frame, results)

        # Add test info
        elapsed = int(time.time() - start_time)
        info_text = f"Test: {elapsed}s | Frames: {frame_count} | Detections: {detection_count}"
        cv2.putText(output_frame, info_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Enhanced Detection Test', output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stats = detector.get_performance_stats()
            print(f"\n📊 Current Statistics:")
            for k, v in stats.items():
                print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

    cap.release()
    cv2.destroyAllWindows()

    # Calculate results
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    detection_rate = detection_count / frame_count if frame_count > 0 else 0

    return {
        'total_frames': frame_count,
        'total_detections': detection_count,
        'test_duration': total_time,
        'avg_fps': avg_fps,
        'avg_processing_time': avg_processing_time,
        'detection_rate': detection_rate,
        'max_processing_time': max(processing_times) if processing_times else 0,
        'min_processing_time': min(processing_times) if processing_times else 0
    }

def test_images(detector, image_dir):
    """Test dengan image files"""
    print(f"🖼️ Testing with images from: {image_dir}")

    if not os.path.exists(image_dir):
        print(f"⚠️ Directory {image_dir} not found, creating sample test...")
        return test_sample_image(detector)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        pattern = os.path.join(image_dir, f"*{ext}")
        image_files.extend([f for f in os.listdir(image_dir)
                           if f.lower().endswith(ext.lower())])

    if not image_files:
        print(f"⚠️ No images found in {image_dir}")
        return test_sample_image(detector)

    results = []
    total_detections = 0

    for i, image_file in enumerate(image_files[:10]):  # Test max 10 images
        image_path = os.path.join(image_dir, image_file)
        print(f"  📸 Processing {i+1}/{min(len(image_files), 10)}: {image_file}")

        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            # Process dengan enhanced detector
            start_time = time.time()
            detections = detector.process_frame_enhanced(frame)
            processing_time = time.time() - start_time

            # Draw results
            output_frame = detector.draw_enhanced_results(frame, detections)

            # Save result
            output_path = os.path.join(image_dir, f"result_{image_file}")
            cv2.imwrite(output_path, output_frame)

            total_detections += len(detections)

            result = {
                'file': image_file,
                'detections': len(detections),
                'processing_time': processing_time,
                'plates_found': [d['plate_text'] for d in detections]
            }
            results.append(result)

            # Show result (optional)
            # cv2.imshow('Enhanced Detection Result', output_frame)
            # cv2.waitKey(1000)  # Show for 1 second

        except Exception as e:
            print(f"    ❌ Error processing {image_file}: {e}")

    # cv2.destroyAllWindows()

    return {
        'total_images': len(image_files),
        'processed_images': len(results),
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / len(results) if results else 0,
        'avg_processing_time': np.mean([r['processing_time'] for r in results]) if results else 0,
        'results': results
    }

def test_sample_image(detector):
    """Test dengan sample image (jika tidak ada test images)"""
    print("📸 Creating sample test image...")

    # Create sample image with vehicle-like rectangle
    sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_frame.fill(50)  # Gray background

    # Draw sample "vehicle"
    cv2.rectangle(sample_frame, (200, 200), (400, 350), (100, 100, 100), -1)
    # Draw sample "plate"
    cv2.rectangle(sample_frame, (250, 300), (350, 330), (255, 255, 255), -1)
    cv2.putText(sample_frame, "B1234CD", (255, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Process dengan enhanced detector
    start_time = time.time()
    results = detector.process_frame_enhanced(sample_frame)
    processing_time = time.time() - start_time

    # Draw results
    output_frame = detector.draw_enhanced_results(sample_frame, results)

    # Save sample result
    cv2.imwrite('sample_test_result.jpg', output_frame)

    return {
        'sample_test': True,
        'detections': len(results),
        'processing_time': processing_time,
        'plates_found': [r['plate_text'] for r in results]
    }

def test_videos(detector, video_dir):
    """Test dengan video files"""
    print(f"🎥 Testing with videos from: {video_dir}")

    if not os.path.exists(video_dir):
        print(f"⚠️ Directory {video_dir} not found, skipping video test")
        return {'skipped': True, 'reason': 'Directory not found'}

    # Get video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for ext in video_extensions:
        video_files.extend([f for f in os.listdir(video_dir)
                           if f.lower().endswith(ext.lower())])

    if not video_files:
        print(f"⚠️ No videos found in {video_dir}")
        return {'skipped': True, 'reason': 'No video files found'}

    results = []

    for i, video_file in enumerate(video_files[:3]):  # Test max 3 videos
        video_path = os.path.join(video_dir, video_file)
        print(f"  🎬 Processing {i+1}/{min(len(video_files), 3)}: {video_file}")

        try:
            result = process_video_file(detector, video_path)
            results.append(result)

        except Exception as e:
            print(f"    ❌ Error processing {video_file}: {e}")

    return {
        'total_videos': len(video_files),
        'processed_videos': len(results),
        'results': results
    }

def process_video_file(detector, video_path):
    """Process single video file"""
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    detection_count = 0
    processing_times = []

    # Process every 10th frame untuk speed
    frame_skip = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            start_time = time.time()
            results = detector.process_frame_enhanced(frame)
            processing_time = time.time() - start_time

            processing_times.append(processing_time)
            detection_count += len(results)

        frame_count += 1

        # Process max 100 frames
        if frame_count >= 100:
            break

    cap.release()

    return {
        'file': os.path.basename(video_path),
        'total_frames': total_frames,
        'processed_frames': frame_count,
        'detections': detection_count,
        'avg_processing_time': np.mean(processing_times) if processing_times else 0,
        'original_fps': fps
    }

def print_test_results(results):
    """Print comprehensive test results"""
    print("\n" + "="*60)
    print("📊 ENHANCED DETECTION TEST RESULTS")
    print("="*60)

    for test_name, result in results.items():
        print(f"\n🧪 {test_name}:")

        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            continue

        if 'skipped' in result:
            print(f"   ⏭️ Skipped: {result['reason']}")
            continue

        # Print relevant metrics
        if 'total_frames' in result:
            print(f"   📊 Total Frames: {result['total_frames']}")
            print(f"   🎯 Total Detections: {result['total_detections']}")
            print(f"   ⚡ Avg FPS: {result.get('avg_fps', 0):.2f}")
            print(f"   ⏱️ Avg Processing Time: {result.get('avg_processing_time', 0):.3f}s")
            print(f"   📈 Detection Rate: {result.get('detection_rate', 0):.2f} detections/frame")

        if 'total_images' in result:
            print(f"   📸 Images Processed: {result['processed_images']}/{result['total_images']}")
            print(f"   🎯 Total Detections: {result['total_detections']}")
            print(f"   📈 Avg Detections/Image: {result['avg_detections_per_image']:.2f}")

        if 'sample_test' in result:
            print(f"   🧪 Sample Test: {result['detections']} detections found")
            print(f"   📝 Plates: {result['plates_found']}")

    print("\n✅ All tests completed!")
    print("📁 Check output files for detailed results")

if __name__ == "__main__":
    print("🚀 Enhanced Plate Detection Test Suite")
    print("Testing akurasi tinggi YOLO + OCR untuk deteksi plat nomor")
    print()

    success = test_enhanced_detection()

    if success:
        print("\n🎉 Test suite completed successfully!")
    else:
        print("\n❌ Test suite failed!")
        sys.exit(1)