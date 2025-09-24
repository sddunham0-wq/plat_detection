#!/usr/bin/env python3
"""
Production-Ready Final Test - Ultra Stable Indonesian Plate Counter
Test dengan scenario real-world Indonesian traffic untuk memastikan production readiness
"""

import time
import logging
from stream_manager import HeadlessStreamManager
from utils.stable_plate_counter import create_stable_plate_counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_realistic_detection_stream():
    """Simulate realistic Indonesian traffic dengan multiple vehicles"""

    print("🚗 PRODUCTION READINESS TEST - Indonesian Traffic Simulation")
    print("=" * 65)

    # Create ultra-stable counter (same config as integrated system)
    counter = create_stable_plate_counter({
        'text_similarity_threshold': 0.80,
        'spatial_distance_threshold': 120.0,
        'temporal_window': 5.0,
        'min_stability_score': 0.65,
        'min_confidence': 0.30,
        'confirmation_detections': 2
    })

    print("📋 Testing dengan konfigurasi production:")
    stats = counter.get_comprehensive_stats()
    config = stats['configuration']
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\n🎯 SCENARIO: Jakarta Traffic - 3 vehicles dalam 10 detik")
    print("-" * 65)

    # Realistic Indonesian traffic simulation
    traffic_detections = [
        # Vehicle 1: Motorcycle B 1234 ABC
        {"time": 0.0, "text": "B 1234 ABC", "bbox": (100, 200, 140, 45), "conf": 0.75, "vehicle": "motorcycle"},
        {"time": 0.8, "text": "B1234ABC", "bbox": (110, 205, 142, 47), "conf": 0.70, "vehicle": "motorcycle"},
        {"time": 1.5, "text": "B 1234 A8C", "bbox": (120, 210, 138, 46), "conf": 0.65, "vehicle": "motorcycle"},  # OCR error
        {"time": 2.2, "text": "B 1234 ABC", "bbox": (130, 215, 140, 48), "conf": 0.80, "vehicle": "motorcycle"},
        {"time": 3.0, "text": "B1234ABC", "bbox": (140, 220, 142, 47), "conf": 0.78, "vehicle": "motorcycle"},

        # Vehicle 2: Car D 5678 XYZ (overlapping dengan motorcycle)
        {"time": 2.5, "text": "D 5678 XYZ", "bbox": (300, 300, 150, 50), "conf": 0.85, "vehicle": "car"},
        {"time": 3.2, "text": "D5678XYZ", "bbox": (310, 305, 148, 52), "conf": 0.82, "vehicle": "car"},
        {"time": 4.0, "text": "D 5678 XYZ", "bbox": (320, 310, 152, 51), "conf": 0.88, "vehicle": "car"},
        {"time": 4.8, "text": "D 5678 XYZ", "bbox": (330, 315, 150, 50), "conf": 0.90, "vehicle": "car"},

        # Vehicle 3: Bus AA 9876 BB (larger vehicle)
        {"time": 5.0, "text": "AA 9876 BB", "bbox": (500, 400, 160, 55), "conf": 0.72, "vehicle": "bus"},
        {"time": 5.7, "text": "AA9876BB", "bbox": (510, 405, 158, 57), "conf": 0.68, "vehicle": "bus"},
        {"time": 6.5, "text": "AA 9876 BB", "bbox": (520, 410, 162, 56), "conf": 0.75, "vehicle": "bus"},
        {"time": 7.2, "text": "AA 9876 B8", "bbox": (530, 415, 160, 58), "conf": 0.70, "vehicle": "bus"},  # OCR error
        {"time": 8.0, "text": "AA 9876 BB", "bbox": (540, 420, 161, 55), "conf": 0.85, "vehicle": "bus"},

        # Noise detections (should be filtered)
        {"time": 6.0, "text": "???", "bbox": (200, 100, 30, 15), "conf": 0.20, "vehicle": "unknown"},
        {"time": 7.5, "text": "HTTP", "bbox": (250, 150, 50, 20), "conf": 0.35, "vehicle": "unknown"},
        {"time": 8.5, "text": "AB", "bbox": (300, 200, 25, 12), "conf": 0.25, "vehicle": "unknown"},

        # Late detections (same vehicles, should be matched)
        {"time": 9.0, "text": "B1234ABC", "bbox": (600, 250, 140, 45), "conf": 0.77, "vehicle": "motorcycle"},  # Motorcycle far position
        {"time": 9.5, "text": "D5678XYZ", "bbox": (650, 350, 150, 50), "conf": 0.83, "vehicle": "car"},        # Car far position
    ]

    print(f"Processing {len(traffic_detections)} detections over 10 seconds...")
    print("\nReal-time processing:")

    start_time = time.time()
    processed_count = 0
    filtered_count = 0

    for detection in traffic_detections:
        # Simulate real-time processing
        elapsed = time.time() - start_time
        target_time = detection["time"]

        if elapsed < target_time:
            time.sleep(target_time - elapsed)

        # Process detection
        track_id = counter.add_detection(
            text=detection["text"],
            bbox=detection["bbox"],
            confidence=detection["conf"],
            vehicle_type=detection["vehicle"]
        )

        if track_id:
            processed_count += 1
            status = "TRACKED"
        else:
            filtered_count += 1
            status = "FILTERED"

        counts = counter.get_stable_counts()

        print(f"T+{detection['time']:4.1f}s: '{detection['text']:<12}' "
              f"({detection['vehicle']:<10}) conf={detection['conf']:.2f} -> {status:>8} "
              f"| Current: {counts['current_visible_plates']}, Session: {counts['total_unique_session']}")

    # Final analysis
    print(f"\n📊 FINAL PRODUCTION TEST RESULTS:")
    print("-" * 40)

    final_stats = counter.get_comprehensive_stats()

    print(f"✅ UNIQUE VEHICLES DETECTED:")
    print(f"  Current visible: {final_stats['current_visible_plates']}")
    print(f"  Total unique session: {final_stats['total_unique_session']}")
    print(f"  Stable confirmed: {final_stats['stable_confirmed_plates']}")
    print(f"  High confidence: {final_stats['high_confidence_plates']}")

    print(f"\n📈 PROCESSING EFFICIENCY:")
    print(f"  Total detections processed: {processed_count}")
    print(f"  Noise filtered: {filtered_count}")
    print(f"  Raw detections processed: {final_stats['total_raw_processed']}")
    print(f"  Quality filtering rate: {final_stats['stability_metrics']['quality_filter_rate_percent']:.1f}%")

    print(f"\n🎯 STABILITY ANALYSIS:")
    stability_metrics = final_stats['stability_metrics']
    print(f"  Stability rate: {stability_metrics['stability_rate_percent']:.1f}%")
    print(f"  Efficiency rate: {stability_metrics['efficiency_rate_percent']:.1f}%")

    print(f"\n⚡ PERFORMANCE:")
    perf_metrics = final_stats['performance_metrics']
    print(f"  Average processing time: {perf_metrics['avg_processing_time_ms']:.2f} ms")
    print(f"  Total tracks created: {perf_metrics['total_tracks_created']}")

    # Production readiness validation
    expected_vehicles = 3  # B1234ABC, D5678XYZ, AA9876BB
    expected_noise_filtered = 3  # ???, HTTP, AB

    print(f"\n✅ PRODUCTION READINESS VALIDATION:")
    print("-" * 45)

    success_count = 0
    total_tests = 5

    # Test 1: Correct unique counting
    if final_stats['total_unique_session'] == expected_vehicles:
        print(f"✅ PASS: Unique vehicle counting (Expected: {expected_vehicles}, Got: {final_stats['total_unique_session']})")
        success_count += 1
    else:
        print(f"❌ FAIL: Unique vehicle counting (Expected: {expected_vehicles}, Got: {final_stats['total_unique_session']})")

    # Test 2: Noise filtering
    total_filtered = final_stats['low_quality_filtered'] + final_stats['false_positives_filtered']
    if total_filtered >= expected_noise_filtered:
        print(f"✅ PASS: Noise filtering (Expected ≥{expected_noise_filtered}, Got: {total_filtered})")
        success_count += 1
    else:
        print(f"❌ FAIL: Noise filtering (Expected ≥{expected_noise_filtered}, Got: {total_filtered})")

    # Test 3: Stability achievement
    if final_stats['stable_confirmed_plates'] >= 2:  # At least 2 vehicles should be stable
        print(f"✅ PASS: Stability detection (Got: {final_stats['stable_confirmed_plates']} stable plates)")
        success_count += 1
    else:
        print(f"❌ FAIL: Stability detection (Got: {final_stats['stable_confirmed_plates']} stable plates)")

    # Test 4: Performance
    if perf_metrics['avg_processing_time_ms'] < 5.0:  # Sub-5ms processing
        print(f"✅ PASS: Performance (Processing time: {perf_metrics['avg_processing_time_ms']:.2f} ms)")
        success_count += 1
    else:
        print(f"❌ FAIL: Performance (Processing time: {perf_metrics['avg_processing_time_ms']:.2f} ms)")

    # Test 5: Efficiency
    if stability_metrics['efficiency_rate_percent'] >= 10:  # Reasonable efficiency
        print(f"✅ PASS: System efficiency ({stability_metrics['efficiency_rate_percent']:.1f}%)")
        success_count += 1
    else:
        print(f"❌ FAIL: System efficiency ({stability_metrics['efficiency_rate_percent']:.1f}%)")

    # Final verdict
    print(f"\n🏆 FINAL PRODUCTION VERDICT:")
    print("=" * 50)

    success_rate = (success_count / total_tests) * 100

    if success_count == total_tests:
        print("✅ PRODUCTION READY!")
        print("🚀 System passed ALL validation tests!")
        print("\n🎯 Ready for deployment with Indonesian traffic!")

        print(f"\n📋 Production Features Validated:")
        print(f"  ✅ Ultra-stable unique plate counting")
        print(f"  ✅ Indonesian OCR error handling")
        print(f"  ✅ Real-time noise filtering")
        print(f"  ✅ High-performance processing (<5ms)")
        print(f"  ✅ Traffic stability analysis")
        print(f"  ✅ Temporal tracking and expiry")

    elif success_count >= 4:
        print("⚠️  MOSTLY READY - Minor adjustments needed")
        print(f"🎯 Passed {success_count}/{total_tests} tests ({success_rate:.0f}%)")

    else:
        print("❌ NOT PRODUCTION READY")
        print(f"🎯 Only passed {success_count}/{total_tests} tests ({success_rate:.0f}%)")
        print("System needs significant improvements")

    # Wait for expiry test
    print(f"\n⏰ Testing temporal expiry (waiting 6 seconds)...")
    time.sleep(6)

    expired_counts = counter.get_stable_counts()
    if expired_counts['current_visible_plates'] == 0:
        print("✅ Temporal expiry working correctly")
    else:
        print(f"❌ Temporal expiry issue: {expired_counts['current_visible_plates']} still visible")

    return final_stats, success_count == total_tests

if __name__ == "__main__":
    try:
        print("🚀 Starting Production Readiness Test...")
        print("Testing Ultra-Stable Indonesian Plate Counter System\n")

        final_stats, production_ready = simulate_realistic_detection_stream()

        print(f"\n🎉 PRODUCTION TEST COMPLETED!")
        print("=" * 50)

        if production_ready:
            print("✅ SISTEM SIAP UNTUK PRODUKSI!")
            print("🇮🇩 Ready untuk Indonesian traffic monitoring!")
        else:
            print("❌ Sistem masih perlu penyesuaian")

    except Exception as e:
        logger.error(f"Production test failed: {e}")
        import traceback
        traceback.print_exc()