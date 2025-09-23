#!/usr/bin/env python3
"""
Test balanced configuration untuk avoid both under-counting dan over-counting
"""

import sys
import os
from dataclasses import dataclass
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import PlateCounterManager

@dataclass
class TestDetection:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    should_count: bool  # Expected result

def test_balanced_configuration():
    """Test dengan balanced config yang avoid over-counting tapi masih catch real plates"""
    print("⚖️ Testing Balanced Configuration")
    print("Goal: Avoid over-counting (200+) tapi masih detect real plates")

    # Balanced config seperti yang sudah di-fix
    counter = PlateCounterManager(
        similarity_threshold=0.85,
        spatial_proximity_distance=60.0,
        plate_expiry_time=3.0,
        confirmation_threshold=3,  # Need 3 hits (reduce false positives)
        confidence_filter_min=0.45  # 45% threshold (balanced)
    )

    # Test cases dengan different confidence levels
    test_cases = [
        # Very low confidence - should be filtered (avoid noise)
        TestDetection("BB", (100, 100, 150, 50), 0.27, False),  # Too low
        TestDetection("AB", (200, 100, 150, 50), 0.35, False),  # Too low + too short
        TestDetection("12", (300, 100, 150, 50), 0.40, False),  # Too low + too short

        # Medium confidence - should be accepted
        TestDetection("B123", (400, 100, 150, 50), 0.45, True),   # Exactly at threshold
        TestDetection("B1234", (500, 100, 150, 50), 0.55, True),  # Medium
        TestDetection("B123ABC", (600, 100, 150, 50), 0.65, True), # Good

        # High confidence - definitely accepted
        TestDetection("B1234ABC", (700, 100, 150, 50), 0.75, True),
        TestDetection("D5678XYZ", (800, 100, 150, 50), 0.85, True),
    ]

    print(f"\n📊 Testing {len(test_cases)} cases...")

    # Simulate multiple detections of same plates (untuk test confirmation threshold)
    for frame in range(4):  # 4 frames untuk test confirmation
        print(f"\n--- Frame {frame + 1} ---")

        for i, case in enumerate(test_cases):
            # Add small variation untuk simulate real conditions
            varied_bbox = (
                case.bbox[0] + (frame % 3) - 1,
                case.bbox[1] + (frame % 2),
                case.bbox[2],
                case.bbox[3]
            )

            plate_id = counter.add_or_update_detection(
                detection_text=case.text,
                detection_bbox=varied_bbox,
                confidence=case.confidence + (frame * 0.01),  # Slight confidence variation
                tracking_id=None,
                vehicle_type="car"
            )

            if frame == 0:  # Log first frame only
                status = "✅ ADDED" if plate_id else "❌ FILTERED"
                expected = "✅ EXPECTED" if case.should_count else "❌ EXPECTED"
                print(f"   {case.text} (conf={case.confidence:.2f}) → {status} | {expected}")

    # Get results
    counts = counter.get_current_counts()
    confirmed_plates = counter.get_confirmed_plates()

    print(f"\n🏁 BALANCED CONFIG RESULTS:")
    print(f"   Total unique plates: {counts['total_unique_plates_session']}")
    print(f"   Confirmed plates: {len(confirmed_plates)}")
    print(f"   Raw detections: {counts['raw_detections_processed']}")
    print(f"   False positives filtered: {counts['false_positives_filtered']}")
    print(f"   Duplicates filtered: {counts['duplicates_filtered']}")

    # Expected: 5 plates should be counted (yang confidence >= 0.45 dan length >= 3)
    expected_valid = len([case for case in test_cases if case.should_count])
    actual_confirmed = len(confirmed_plates)

    print(f"\n📈 Analysis:")
    print(f"   Expected valid plates: {expected_valid}")
    print(f"   Actual confirmed plates: {actual_confirmed}")

    if 0 < actual_confirmed <= expected_valid:
        print(f"   ✅ GOOD: Reasonable count, tidak over-counting!")
        if actual_confirmed == expected_valid:
            print(f"   🎯 PERFECT: Exact match dengan expected!")
        else:
            print(f"   📝 NOTE: Slightly conservative (lebih baik daripada over-count)")
    elif actual_confirmed == 0:
        print(f"   ⚠️ WARNING: Under-counting, mungkin threshold masih terlalu ketat")
    else:
        print(f"   ❌ PROBLEM: Over-counting detected!")

    # Show confirmed plates details
    if confirmed_plates:
        print(f"\n✅ Confirmed Plates:")
        for plate in confirmed_plates:
            print(f"   - {plate.text} (hits: {plate.detection_count}, conf: {plate.confidence:.2f})")

if __name__ == "__main__":
    test_balanced_configuration()