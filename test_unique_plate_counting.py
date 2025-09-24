#!/usr/bin/env python3
"""
Test Unique Plate Counting System
Verify that PlateCounterManager correctly handles duplicate detections and counts unique plates only
"""

import time
import logging
from utils.plate_counter_manager import create_plate_counter_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_unique_counting():
    """Test unique plate counting with simulated duplicate detections"""

    print("🧪 Testing Unique Plate Counting System")
    print("=" * 50)

    # Create optimized plate counter for Indonesian plates
    counter = create_plate_counter_manager()

    print("\n📋 Configuration:")
    stats = counter.get_statistics()
    config = stats['configuration']
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n🎯 Test Case 1: Same plate multiple detections (should count as 1)")
    print("-" * 50)

    # Simulate same plate detected multiple times
    test_detections = [
        ("B 1234 ABC", (100, 200, 150, 50), 0.85),  # Original detection
        ("B1234ABC", (105, 205, 145, 48), 0.82),    # Same plate, slightly different position & format
        ("B 1234 ABC", (108, 202, 148, 52), 0.88),  # Same plate, small movement
        ("B 1234 A8C", (110, 200, 150, 50), 0.75),  # Same plate with OCR error (8 instead of B)
        ("81234ABC", (112, 198, 152, 51), 0.78),    # Same plate with OCR error (8 instead of B, no spaces)
    ]

    for i, (text, bbox, confidence) in enumerate(test_detections):
        plate_id = counter.add_or_update_detection(text, bbox, confidence)
        counts = counter.get_current_counts()

        print(f"Detection {i+1}: '{text}' -> Plate ID: {plate_id}")
        print(f"  Current unique plates: {counts['current_visible_plates']}")
        print(f"  Total session plates: {counts['total_unique_plates_session']}")

    # Small delay to simulate time passage
    time.sleep(0.5)

    print("\n🎯 Test Case 2: Different plate (should count as +1)")
    print("-" * 50)

    # Add completely different plate
    different_plate_detections = [
        ("D 5678 XYZ", (300, 400, 160, 55), 0.90),  # New plate
        ("D5678XYZ", (305, 402, 158, 53), 0.87),    # Same new plate, different format
    ]

    for i, (text, bbox, confidence) in enumerate(different_plate_detections):
        plate_id = counter.add_or_update_detection(text, bbox, confidence)
        counts = counter.get_current_counts()

        print(f"Detection {i+1}: '{text}' -> Plate ID: {plate_id}")
        print(f"  Current unique plates: {counts['current_visible_plates']}")
        print(f"  Total session plates: {counts['total_unique_plates_session']}")

    print("\n🎯 Test Case 3: Low quality detections (should be filtered)")
    print("-" * 50)

    # Add low quality detections that should be filtered
    low_quality_detections = [
        ("AB", (500, 600, 50, 20), 0.30),           # Too short, low confidence
        ("???", (510, 610, 45, 18), 0.25),         # Invalid text
        ("HTTP", (520, 620, 80, 25), 0.40),        # False positive text
    ]

    for i, (text, bbox, confidence) in enumerate(low_quality_detections):
        plate_id = counter.add_or_update_detection(text, bbox, confidence)
        counts = counter.get_current_counts()

        result = "ACCEPTED" if plate_id else "FILTERED"
        print(f"Low quality {i+1}: '{text}' -> {result}")
        print(f"  Current unique plates: {counts['current_visible_plates']}")
        print(f"  Total session plates: {counts['total_unique_plates_session']}")

    print("\n📊 Final Statistics:")
    print("-" * 50)

    final_stats = counter.get_statistics()

    print(f"✅ UNIQUE PLATES COUNT:")
    print(f"  Current visible: {final_stats['current_visible_plates']}")
    print(f"  Total session: {final_stats['total_unique_plates_session']}")
    print(f"  Confirmed plates: {final_stats['confirmed_visible_plates']}")

    print(f"\n📈 Processing Statistics:")
    print(f"  Raw detections processed: {final_stats['raw_detections_processed']}")
    print(f"  False positives filtered: {final_stats['false_positives_filtered']}")
    print(f"  Duplicates filtered: {final_stats['duplicates_filtered']}")

    print(f"\n🎯 Accuracy Metrics:")
    accuracy = final_stats['accuracy_metrics']
    print(f"  Unique plate extraction rate: {accuracy['unique_plate_extraction_rate_percent']}%")
    print(f"  False positive filter rate: {accuracy['false_positive_filter_rate_percent']}%")
    print(f"  Deduplication rate: {accuracy['deduplication_rate_percent']}%")
    print(f"  Confirmation rate: {accuracy['confirmation_rate_percent']}%")

    print(f"\n⚡ Performance Metrics:")
    perf = final_stats['performance_metrics']
    print(f"  Average processing time: {perf['avg_processing_time_ms']:.2f} ms")
    print(f"  Average detections per plate: {perf['avg_detections_per_plate']:.1f}")

    # Test plate expiry
    print(f"\n🎯 Test Case 4: Plate expiry (wait {counter.plate_expiry_time + 1}s)")
    print("-" * 50)

    print("Waiting for plates to expire...")
    time.sleep(counter.plate_expiry_time + 1)  # Wait longer than expiry time

    # Get counts after expiry
    expired_counts = counter.get_current_counts()
    print(f"After expiry - Current visible: {expired_counts['current_visible_plates']}")
    print(f"After expiry - Total session: {expired_counts['total_unique_plates_session']}")
    print(f"Expired plates: {expired_counts['expired_plates']}")

    print("\n✅ Test Results Summary:")
    print("=" * 50)

    if final_stats['total_unique_plates_session'] == 2:  # Should be exactly 2 unique plates
        print("✅ PASS: Unique counting works correctly!")
        print(f"   Expected: 2 unique plates")
        print(f"   Actual: {final_stats['total_unique_plates_session']} unique plates")
    else:
        print("❌ FAIL: Unique counting not working correctly")
        print(f"   Expected: 2 unique plates")
        print(f"   Actual: {final_stats['total_unique_plates_session']} unique plates")

    if final_stats['false_positives_filtered'] >= 3:  # Should filter at least the 3 low quality ones
        print("✅ PASS: False positive filtering works!")
        print(f"   Filtered: {final_stats['false_positives_filtered']} false positives")
    else:
        print("❌ FAIL: False positive filtering not working")
        print(f"   Filtered: {final_stats['false_positives_filtered']} false positives")

    if expired_counts['current_visible_plates'] == 0:  # Should expire after timeout
        print("✅ PASS: Plate expiry works correctly!")
        print(f"   Current visible after expiry: {expired_counts['current_visible_plates']}")
    else:
        print("❌ FAIL: Plate expiry not working")
        print(f"   Current visible after expiry: {expired_counts['current_visible_plates']}")

    return final_stats

def test_indonesian_plate_normalization():
    """Test Indonesian plate text normalization"""

    print("\n\n🇮🇩 Testing Indonesian Plate Normalization")
    print("=" * 50)

    counter = create_plate_counter_manager()

    test_cases = [
        ("B1234ABC", "B 1234 ABC"),          # Add spaces
        ("B 1234 A8C", "B 1234 ABC"),        # Fix OCR error 8->B
        ("81234ABC", "B 1234 ABC"),          # Fix OCR error 8->B at start
        ("B.1234.ABC", "B 1234 ABC"),        # Remove noise
        ("B-1234-ABC", "B 1234 ABC"),        # Remove noise
        ("D5678XYZ", "D 5678 XYZ"),          # Standard format
        ("DD1234AB", "DD 1234 AB"),          # Double letter area
    ]

    print("Original -> Normalized:")
    print("-" * 30)

    for original, expected in test_cases:
        normalized = counter.normalize_text(original)
        status = "✅" if normalized == expected else "❌"
        print(f"{status} '{original}' -> '{normalized}'")
        if normalized != expected:
            print(f"   Expected: '{expected}'")

if __name__ == "__main__":
    try:
        # Run unique counting test
        stats = test_unique_counting()

        # Run Indonesian normalization test
        test_indonesian_plate_normalization()

        print(f"\n🎉 Testing completed!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()