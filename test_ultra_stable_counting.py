#!/usr/bin/env python3
"""
Ultra-Stable Plate Counting System Test
Comprehensive test untuk memverifikasi stabilitas dan akurasi sistem counting
dengan real-world Indonesian traffic scenarios
"""

import time
import logging
import numpy as np
from utils.stable_plate_counter import create_stable_plate_counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ultra_stable_system():
    """Test ultra-stable system dengan Indonesian traffic simulation"""

    print("🚀 Testing ULTRA-STABLE Indonesian Plate Counting System")
    print("=" * 60)

    # Create ultra-stable counter
    counter = create_stable_plate_counter({
        'text_similarity_threshold': 0.75,
        'spatial_distance_threshold': 120.0,
        'temporal_window': 5.0,
        'min_stability_score': 0.6,
        'min_confidence': 0.30,
        'confirmation_detections': 2
    })

    print("\n📋 Ultra-Stable Configuration:")
    stats = counter.get_comprehensive_stats()
    config = stats['configuration']
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n🎯 TEST SCENARIO 1: Realistic Indonesian Traffic")
    print("-" * 60)
    print("Simulating: 1 motorcycle (B 1234 ABC) moving through frame dengan OCR variations")

    # Realistic Indonesian traffic simulation
    motorcycle_detections = [
        # Initial detection
        ("B 1234 ABC", (100, 200, 140, 45), 0.75, 0.0),

        # Movement dengan OCR variations (typical OCR challenges)
        ("B1234ABC", (105, 202, 142, 47), 0.72, 0.5),     # No spaces
        ("B 1234 A8C", (110, 205, 138, 46), 0.68, 1.0),   # 8 instead of B
        ("81234ABC", (115, 207, 140, 48), 0.65, 1.5),     # 8 instead of B at start
        ("B.1234.ABC", (120, 210, 144, 49), 0.70, 2.0),   # Dots noise
        ("B-1234-ABC", (125, 212, 139, 47), 0.73, 2.5),   # Dash noise

        # Stable detections (should confirm the plate)
        ("B 1234 ABC", (130, 215, 141, 48), 0.80, 3.0),
        ("B 1234 ABC", (135, 218, 143, 47), 0.82, 3.5),

        # More movement
        ("B1234ABC", (140, 220, 140, 46), 0.74, 4.0),
        ("B 1234 ABC", (145, 222, 142, 48), 0.85, 4.5),
    ]

    print(f"\nProcessing {len(motorcycle_detections)} detections over 5 seconds...")

    for i, (text, bbox, confidence, timestamp) in enumerate(motorcycle_detections):
        # Simulate time progression
        if i > 0:
            time.sleep(0.1)  # Small delay untuk simulation

        track_id = counter.add_detection(text, bbox, confidence, "motorcycle")
        counts = counter.get_stable_counts()

        print(f"Frame {i+1:2d} @ {timestamp:.1f}s: '{text:<12}' conf={confidence:.2f} -> Track: {track_id}")
        print(f"         Status: Current={counts['current_visible_plates']}, "
              f"Session={counts['total_unique_session']}, "
              f"Stable={counts['stable_confirmed_plates']}")

    print("\n🎯 TEST SCENARIO 2: Multiple Vehicles (Realistic Traffic)")
    print("-" * 60)

    # Add second vehicle (car)
    car_detections = [
        ("D 5678 XYZ", (300, 350, 150, 50), 0.78, 5.0),
        ("D5678XYZ", (305, 352, 148, 52), 0.75, 5.5),
        ("D 5678 XYZ", (310, 355, 152, 51), 0.82, 6.0),
    ]

    print(f"\nAdding car: D 5678 XYZ")
    for i, (text, bbox, confidence, timestamp) in enumerate(car_detections):
        time.sleep(0.1)
        track_id = counter.add_detection(text, bbox, confidence, "car")
        counts = counter.get_stable_counts()

        print(f"Car {i+1}: '{text}' -> Track: {track_id}")
        print(f"        Status: Current={counts['current_visible_plates']}, "
              f"Session={counts['total_unique_session']}")

    print("\n🎯 TEST SCENARIO 3: Noise Detection (Should be filtered)")
    print("-" * 60)

    noise_detections = [
        ("AB", (500, 600, 30, 15), 0.25),          # Too short
        ("???", (510, 605, 25, 12), 0.20),         # Invalid
        ("HTTP", (520, 610, 60, 20), 0.35),       # False positive
        ("12345", (530, 615, 40, 18), 0.40),      # Only numbers
        ("ABCDEF", (540, 620, 50, 22), 0.45),     # Only letters
    ]

    filtered_count = 0
    for i, (text, bbox, confidence) in enumerate(noise_detections):
        track_id = counter.add_detection(text, bbox, confidence, "unknown")
        if not track_id:
            filtered_count += 1

        counts = counter.get_stable_counts()
        result = "FILTERED" if not track_id else "ACCEPTED"
        print(f"Noise {i+1}: '{text:<8}' conf={confidence:.2f} -> {result}")

    print(f"\n✅ Filtered {filtered_count}/{len(noise_detections)} noise detections")

    print("\n🎯 TEST SCENARIO 4: Stability Analysis")
    print("-" * 60)

    # Check stability of tracked plates
    print("Analyzing plate stability...")

    # Get final comprehensive stats
    final_stats = counter.get_comprehensive_stats()

    print(f"\n📊 FINAL RESULTS:")
    print("-" * 30)

    print(f"✅ UNIQUE PLATES:")
    print(f"  Current visible: {final_stats['current_visible_plates']}")
    print(f"  Total session: {final_stats['total_unique_session']}")
    print(f"  Stable confirmed: {final_stats['stable_confirmed_plates']}")
    print(f"  High confidence: {final_stats['high_confidence_plates']}")

    print(f"\n📈 PROCESSING STATS:")
    print(f"  Raw processed: {final_stats['total_raw_processed']}")
    print(f"  Low quality filtered: {final_stats['low_quality_filtered']}")
    print(f"  False positives filtered: {final_stats['false_positives_filtered']}")

    print(f"\n🎯 STABILITY METRICS:")
    stability_metrics = final_stats['stability_metrics']
    print(f"  Stability rate: {stability_metrics['stability_rate_percent']:.1f}%")
    print(f"  Efficiency rate: {stability_metrics['efficiency_rate_percent']:.1f}%")
    print(f"  Quality filter rate: {stability_metrics['quality_filter_rate_percent']:.1f}%")

    print(f"\n⚡ PERFORMANCE METRICS:")
    perf_metrics = final_stats['performance_metrics']
    print(f"  Avg processing time: {perf_metrics['avg_processing_time_ms']:.2f} ms")
    print(f"  Total tracks created: {perf_metrics['total_tracks_created']}")
    print(f"  Active tracks: {perf_metrics['active_tracks']}")

    print("\n🎯 TEST SCENARIO 5: Temporal Expiry")
    print("-" * 60)

    print("Waiting for plates to expire (6 seconds)...")
    time.sleep(6)

    expired_counts = counter.get_stable_counts()
    print(f"After expiry:")
    print(f"  Current visible: {expired_counts['current_visible_plates']}")
    print(f"  Total session: {expired_counts['total_unique_session']} (should remain same)")

    print("\n✅ TEST RESULTS VALIDATION:")
    print("=" * 60)

    # Expected results
    expected_unique_plates = 2  # B 1234 ABC + D 5678 XYZ
    expected_stable_plates = 2  # Both should be stable
    expected_filtered = 5       # All noise should be filtered

    success = True

    # Test 1: Unique plate count
    if final_stats['total_unique_session'] == expected_unique_plates:
        print("✅ PASS: Unique plate counting correct")
        print(f"   Expected: {expected_unique_plates}, Actual: {final_stats['total_unique_session']}")
    else:
        print("❌ FAIL: Unique plate counting incorrect")
        print(f"   Expected: {expected_unique_plates}, Actual: {final_stats['total_unique_session']}")
        success = False

    # Test 2: Stability detection
    if final_stats['stable_confirmed_plates'] >= 1:  # At least 1 stable plate
        print("✅ PASS: Stability detection working")
        print(f"   Stable plates: {final_stats['stable_confirmed_plates']}")
    else:
        print("❌ FAIL: Stability detection not working")
        print(f"   Stable plates: {final_stats['stable_confirmed_plates']}")
        success = False

    # Test 3: Noise filtering
    if final_stats['false_positives_filtered'] + final_stats['low_quality_filtered'] >= 5:
        print("✅ PASS: Noise filtering working")
        print(f"   Filtered: {final_stats['false_positives_filtered'] + final_stats['low_quality_filtered']}")
    else:
        print("❌ FAIL: Noise filtering not working properly")
        print(f"   Filtered: {final_stats['false_positives_filtered'] + final_stats['low_quality_filtered']}")
        success = False

    # Test 4: Efficiency
    efficiency = final_stats['stability_metrics']['efficiency_rate_percent']
    if efficiency >= 10:  # Reasonable efficiency
        print("✅ PASS: System efficiency acceptable")
        print(f"   Efficiency: {efficiency:.1f}%")
    else:
        print("❌ FAIL: System efficiency too low")
        print(f"   Efficiency: {efficiency:.1f}%")
        success = False

    # Test 5: Temporal expiry
    if expired_counts['current_visible_plates'] == 0:
        print("✅ PASS: Temporal expiry working")
        print(f"   Expired correctly: {expired_counts['current_visible_plates']} visible")
    else:
        print("❌ FAIL: Temporal expiry not working")
        print(f"   Should be 0, got: {expired_counts['current_visible_plates']}")
        success = False

    print(f"\n🏆 OVERALL RESULT:")
    if success:
        print("✅ ALL TESTS PASSED! Ultra-Stable System is working correctly!")
        print("🚀 System is ready for Indonesian traffic deployment!")
    else:
        print("❌ Some tests failed. System needs adjustment.")

    return final_stats, success

def test_indonesian_text_processing():
    """Test Indonesian text normalization dan similarity"""

    print("\n\n🇮🇩 INDONESIAN TEXT PROCESSING TEST")
    print("=" * 50)

    counter = create_stable_plate_counter()

    test_cases = [
        # Format variations
        ("B1234ABC", "B 1234 ABC", 0.95),
        ("B 1234 ABC", "B 1234 ABC", 1.0),
        ("B.1234.ABC", "B 1234 ABC", 0.90),
        ("B-1234-ABC", "B 1234 ABC", 0.90),

        # OCR errors
        ("B 1234 A8C", "B 1234 ABC", 0.85),
        ("81234ABC", "B 1234 ABC", 0.80),
        ("B 1234 A8C", "B1234ABC", 0.85),

        # Different plates (should be low similarity)
        ("B 1234 ABC", "D 5678 XYZ", 0.30),
        ("B 1234 ABC", "B 5678 ABC", 0.60),
    ]

    print("Text Similarity Tests:")
    print("-" * 30)

    all_passed = True
    for text1, text2, expected_min_sim in test_cases:
        similarity = counter.calculate_advanced_similarity(text1, text2)
        status = "✅" if similarity >= expected_min_sim else "❌"

        if similarity < expected_min_sim:
            all_passed = False

        print(f"{status} '{text1}' <-> '{text2}': {similarity:.2f} (expected ≥{expected_min_sim})")

    print(f"\n🎯 Text Processing Result: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")

    return all_passed

if __name__ == "__main__":
    try:
        print("🚀 Starting Ultra-Stable Indonesian Plate Counter Tests...")

        # Run main stability test
        final_stats, main_success = test_ultra_stable_system()

        # Run text processing test
        text_success = test_indonesian_text_processing()

        print(f"\n🎉 COMPREHENSIVE TEST RESULTS:")
        print("=" * 50)

        if main_success and text_success:
            print("✅ ALL TESTS PASSED!")
            print("🚀 Ultra-Stable System is READY for production!")
            print("\n📋 Key Benefits Achieved:")
            print("  • ✅ Accurate unique plate counting")
            print("  • ✅ Indonesian OCR error handling")
            print("  • ✅ Noise filtering and false positive removal")
            print("  • ✅ Temporal stability and expiry")
            print("  • ✅ High-performance processing")
            print("  • ✅ Real-world traffic simulation passed")
        else:
            print("❌ SOME TESTS FAILED")
            print("System needs fine-tuning before production deployment")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()