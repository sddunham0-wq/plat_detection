#!/usr/bin/env python3
"""
Test Enhanced Detection dengan gambar real mobil.png
Testing akurasi tinggi untuk deteksi plat nomor pada gambar nyata
"""
import cv2
import os
import sys
import time
import numpy as np
from enhanced_plate_detector import EnhancedPlateDetector

def test_real_image_detection():
    """Test enhanced detection dengan gambar mobil.png"""
    print("🚗 Testing Enhanced Detection pada Gambar Real")
    print("=" * 50)

    # Path ke gambar
    image_path = "contoh/mobil.png"

    if not os.path.exists(image_path):
        print(f"❌ Gambar tidak ditemukan: {image_path}")
        return False

    # Initialize enhanced detector
    try:
        print("📡 Initializing enhanced detector...")
        detector = EnhancedPlateDetector('enhanced_detection_config.ini')
        print("✅ Enhanced detector loaded successfully")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

    # Load gambar
    print(f"📸 Loading image: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ Cannot load image: {image_path}")
        return False

    print(f"✅ Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")

    # Process dengan enhanced detector
    print("🔍 Processing dengan Enhanced Detection...")
    start_time = time.time()

    results = detector.process_frame_enhanced(frame)

    processing_time = time.time() - start_time
    print(f"⏱️ Processing time: {processing_time:.3f} seconds")

    # Analyze results
    print(f"\n📊 Detection Results:")
    print(f"   🎯 Total detections: {len(results)}")

    if len(results) == 0:
        print("   ⚠️ No license plates detected")

        # Try dengan threshold yang lebih rendah
        print("\n🔧 Trying with lower confidence threshold...")
        original_threshold = detector.enhanced_conf_threshold
        detector.enhanced_conf_threshold = 0.1

        results_low = detector.process_frame_enhanced(frame)
        detector.enhanced_conf_threshold = original_threshold

        print(f"   🎯 Low threshold results: {len(results_low)}")

        if len(results_low) > 0:
            results = results_low

    for i, result in enumerate(results):
        print(f"\n   🚗 Detection {i+1}:")
        print(f"      Vehicle Type: {result['vehicle_type']}")
        print(f"      Plate Text: '{result['plate_text']}'")
        print(f"      Confidence: {result['confidence']:.3f}")
        print(f"      Detection Method: {result['detection_method']}")
        print(f"      OCR Config: {result['ocr_config']}")
        print(f"      Enhancement: {result['enhancement_applied']}")
        print(f"      Vehicle BBox: {result['vehicle_bbox']}")
        print(f"      Plate BBox: {result['plate_bbox']}")

    # Draw results dan save
    print(f"\n🎨 Drawing detection results...")
    output_frame = detector.draw_enhanced_results(frame, results)

    # Save hasil
    output_path = "enhanced_detection_result_mobil.jpg"
    cv2.imwrite(output_path, output_frame)
    print(f"💾 Result saved to: {output_path}")

    # Show image dengan deteksi (optional - untuk display)
    print(f"\n📺 Displaying result...")
    try:
        # Resize untuk display jika terlalu besar
        display_frame = output_frame.copy()
        height, width = display_frame.shape[:2]

        if width > 1200:
            scale = 1200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))

        cv2.imshow('Enhanced Detection Result', display_frame)
        print("   Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"   ⚠️ Display error (normal pada server): {e}")

    # Performance stats
    stats = detector.get_performance_stats()
    if stats:
        print(f"\n📈 Performance Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

    return True

def analyze_detection_quality(results):
    """Analyze quality dari detection results"""
    if not results:
        return

    print(f"\n🔬 Detection Quality Analysis:")

    total_confidence = sum(r['confidence'] for r in results)
    avg_confidence = total_confidence / len(results)

    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Confidence Range: {min(r['confidence'] for r in results):.3f} - {max(r['confidence'] for r in results):.3f}")

    # Method distribution
    methods = [r['detection_method'] for r in results]
    method_counts = {method: methods.count(method) for method in set(methods)}

    print(f"   Detection Methods:")
    for method, count in method_counts.items():
        print(f"      {method}: {count}")

    # OCR config distribution
    ocr_configs = [r['ocr_config'] for r in results]
    ocr_counts = {config: ocr_configs.count(config) for config in set(ocr_configs)}

    print(f"   OCR Configurations:")
    for config, count in ocr_counts.items():
        print(f"      {config}: {count}")

def main():
    """Main function"""
    print("🚀 Enhanced License Plate Detection - Real Image Test")
    print("Testing dengan gambar mobil.png dari folder contoh/")
    print()

    success = test_real_image_detection()

    if success:
        print("\n🎉 Test completed successfully!")
        print("📁 Check 'enhanced_detection_result_mobil.jpg' for hasil visual")
    else:
        print("\n❌ Test failed!")
        return False

    return True

if __name__ == "__main__":
    main()