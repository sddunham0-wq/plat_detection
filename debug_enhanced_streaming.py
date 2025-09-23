#!/usr/bin/env python3
"""
Debug Enhanced Streaming
Debug mengapa enhanced detection tidak menemukan plat di streaming mode
"""
import cv2
import time
import sys
import os
from enhanced_plate_detector import EnhancedPlateDetector

def debug_enhanced_detection():
    """Debug enhanced detection step by step"""
    print("🔍 Debugging Enhanced Detection")
    print("=" * 50)

    image_path = "contoh/mobil.png"

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False

    # Load image
    frame = cv2.imread(image_path)
    print(f"✅ Image loaded: {frame.shape}")

    try:
        # Test 1: Default config
        print(f"\n🧪 Test 1: Default Enhanced Config")
        detector1 = EnhancedPlateDetector('enhanced_detection_config.ini')

        # Manually set very low thresholds
        detector1.enhanced_conf_threshold = 0.05
        detector1.use_secondary = False
        detector1.use_tertiary = False

        results1 = detector1.process_frame_enhanced(frame)
        print(f"   Results: {len(results1)} detections")

        for result in results1:
            print(f"      - '{result['plate_text']}' (conf: {result['confidence']:.3f})")

        # Test 2: Streaming config
        print(f"\n🧪 Test 2: Streaming Enhanced Config")
        detector2 = EnhancedPlateDetector('enhanced_detection_streaming_config.ini')

        # Even lower thresholds
        detector2.enhanced_conf_threshold = 0.01

        results2 = detector2.process_frame_enhanced(frame)
        print(f"   Results: {len(results2)} detections")

        for result in results2:
            print(f"      - '{result['plate_text']}' (conf: {result['confidence']:.3f})")

        # Test 3: Manual vehicle detection first
        print(f"\n🧪 Test 3: Manual Vehicle Detection Debug")

        # Test YOLO vehicle detection
        vehicle_detections = detector2.detect_vehicles_enhanced(frame)
        print(f"   Vehicle detections: {len(vehicle_detections)}")

        for i, vehicle in enumerate(vehicle_detections):
            print(f"      Vehicle {i+1}: {vehicle['vehicle_type']} (conf: {vehicle['confidence']:.3f})")
            print(f"         BBox: {vehicle['bbox']}")

        # Test 4: Manual ROI extraction
        print(f"\n🧪 Test 4: Manual ROI Extraction")

        if vehicle_detections:
            for vehicle in vehicle_detections[:1]:  # Test first vehicle
                print(f"   Testing vehicle: {vehicle['vehicle_type']}")

                plate_regions = detector2.extract_plate_regions_enhanced(frame, vehicle)
                print(f"   Plate regions found: {len(plate_regions)}")

                for j, region in enumerate(plate_regions):
                    print(f"      Region {j+1}: {region['size']} pixels")

                    # Try direct OCR on region
                    if region['region'].size > 0:
                        ocr_result = detector2.perform_ocr_enhanced(region['region'])
                        if ocr_result:
                            print(f"         OCR: '{ocr_result['text']}' (conf: {ocr_result['confidence']:.3f})")

        # Test 5: Compare dengan original working method
        print(f"\n🧪 Test 5: Original Working Method Comparison")

        # Load original detector for comparison
        import sys
        sys.path.append('.')

        try:
            from test_optimized_detection import test_optimized_detection
            print("   Running original optimized detection...")
            # This would show if the original method still works
        except Exception as e:
            print(f"   Cannot run original method: {e}")

        # Generate debug output
        if results1 or results2:
            best_results = results1 if len(results1) > len(results2) else results2
            best_detector = detector1 if len(results1) > len(results2) else detector2

            if best_results:
                output_frame = best_detector.draw_enhanced_results(frame, best_results)
                cv2.imwrite('debug_enhanced_detection.jpg', output_frame)
                print(f"\n💾 Debug result saved: debug_enhanced_detection.jpg")
                return True

        print(f"\n❌ No detections found dengan any method")
        return False

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_streaming_config_diff():
    """Debug perbedaan antara config yang working vs streaming"""
    print(f"\n🔧 Debugging Config Differences")
    print("-" * 40)

    try:
        import configparser

        # Load working config
        working_config = configparser.ConfigParser()
        working_config.read('enhanced_detection_config.ini')

        # Load streaming config
        streaming_config = configparser.ConfigParser()
        streaming_config.read('enhanced_detection_streaming_config.ini')

        # Compare key settings
        comparisons = [
            ('YOLO_ENHANCED', 'ENHANCED_CONF_THRESHOLD'),
            ('YOLO_ENHANCED', 'USE_MULTI_PASS'),
            ('PREPROCESSING', 'USE_CLAHE'),
            ('PREPROCESSING', 'USE_BILATERAL_FILTER'),
            ('OCR_ENHANCED', 'PSM_MODES'),
        ]

        print("Config comparison:")
        for section, key in comparisons:
            working_val = working_config.get(section, key, fallback='NOT_SET')
            streaming_val = streaming_config.get(section, key, fallback='NOT_SET')

            status = "✅" if working_val == streaming_val else "⚠️"
            print(f"   {status} {section}.{key}:")
            print(f"      Working: {working_val}")
            print(f"      Streaming: {streaming_val}")

    except Exception as e:
        print(f"Config comparison failed: {e}")

def main():
    """Main debug function"""
    print("🔍 Enhanced Detection Streaming Debug")
    print("Debugging mengapa enhanced detection tidak bekerja di streaming")
    print()

    # Debug detection
    success = debug_enhanced_detection()

    # Debug config differences
    debug_streaming_config_diff()

    if success:
        print(f"\n✅ Debug completed - detections found!")
    else:
        print(f"\n❌ Debug completed - no detections found")
        print(f"💡 Possible issues:")
        print(f"   1. Confidence threshold terlalu tinggi")
        print(f"   2. OCR configuration tidak optimal")
        print(f"   3. Preprocessing berbeda dengan working version")
        print(f"   4. Vehicle detection tidak menemukan kendaraan")

if __name__ == "__main__":
    main()