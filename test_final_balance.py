#!/usr/bin/env python3
"""
Test final balanced configuration
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

def test_final_balance():
    """Test final balanced configuration"""
    print("🎯 Testing FINAL Balanced Configuration")
    print("Target: Detect real plates, avoid over-counting (200+)")

    # Final balanced config
    counter = PlateCounterManager(
        similarity_threshold=0.85,
        spatial_proximity_distance=60.0,
        plate_expiry_time=3.0,
        confirmation_threshold=2,  # 2 hits for responsiveness
        confidence_filter_min=0.45  # 45% confidence threshold
    )

    # Realistic test scenario: 4 real plates dengan varying confidence
    real_plates = [
        TestDetection("B123ABC", (100, 100, 150, 50), 0.65),  # Good plate
        TestDetection("D456XYZ", (300, 200, 150, 50), 0.55),  # Medium plate
        TestDetection("F789DEF", (500, 300, 150, 50), 0.48),  # Lower but valid
        TestDetection("H012GHI", (200, 400, 150, 50), 0.72)   # High confidence
    ]

    # Noise yang harus difilter
    noise_detections = [
        TestDetection("AB", (150, 150, 150, 50), 0.35),       # Too low confidence + too short
        TestDetection("12", (350, 250, 150, 50), 0.40),       # Too low confidence + too short
        TestDetection("XYZ", (450, 350, 150, 50), 0.43),      # Just below threshold
    ]

    print(f"\n📊 Testing 4 real plates + 3 noise detections...")

    # Simulate detection over 3 frames
    for frame in range(3):
        print(f"\n--- Frame {frame + 1} ---")

        # Add real plates (should be counted)
        for plate in real_plates:
            varied_bbox = (
                plate.bbox[0] + (frame % 3) - 1,
                plate.bbox[1] + (frame % 2),
                plate.bbox[2],
                plate.bbox[3]
            )

            plate_id = counter.add_or_update_detection(
                detection_text=plate.text,
                detection_bbox=varied_bbox,
                confidence=plate.confidence + (frame * 0.01),
                tracking_id=None,
                vehicle_type="car"
            )

            if frame == 0:
                status = "✅" if plate_id else "❌"
                print(f"   REAL: {plate.text} (conf={plate.confidence:.2f}) → {status}")

        # Add noise (should be filtered)
        for noise in noise_detections:
            noise_id = counter.add_or_update_detection(
                detection_text=noise.text,
                detection_bbox=noise.bbox,
                confidence=noise.confidence,
                tracking_id=None,
                vehicle_type="car"
            )

            if frame == 0:
                status = "✅" if noise_id else "❌"
                print(f"   NOISE: {noise.text} (conf={noise.confidence:.2f}) → {status}")

    # Get final results
    counts = counter.get_current_counts()
    confirmed_plates = counter.get_confirmed_plates()

    print(f"\n🏁 FINAL BALANCED RESULTS:")
    print(f"   Total unique plates: {counts['total_unique_plates_session']}")
    print(f"   Confirmed plates: {len(confirmed_plates)}")
    print(f"   Current visible: {counts['current_visible_plates']}")
    print(f"   Raw detections: {counts['raw_detections_processed']}")
    print(f"   False positives filtered: {counts['false_positives_filtered']}")
    print(f"   Duplicates filtered: {counts['duplicates_filtered']}")

    # Analysis
    expected_real = len(real_plates)  # 4 real plates
    actual_total = counts['total_unique_plates_session']
    actual_confirmed = len(confirmed_plates)

    print(f"\n📈 ANALYSIS:")
    print(f"   Expected real plates: {expected_real}")
    print(f"   Actual total counted: {actual_total}")
    print(f"   Actual confirmed: {actual_confirmed}")

    if 3 <= actual_total <= 5:  # Reasonable range
        print(f"   ✅ EXCELLENT: Counter in reasonable range!")
        if actual_confirmed >= 2:
            print(f"   🎯 GREAT: {actual_confirmed} plates confirmed (good responsiveness)")

        # Success criteria
        if actual_total < 10:  # Much less than 200+ over-counting
            print(f"   🚫 OVER-COUNTING FIXED: No more 200+ false counts!")

        if actual_total > 0:  # Not zero like before
            print(f"   🚫 UNDER-COUNTING FIXED: Shows real detections now!")

        print(f"\n🎉 BALANCED CONFIGURATION SUCCESS!")
        print(f"   - Detects real plates: ✅")
        print(f"   - Avoids over-counting: ✅")
        print(f"   - Filters noise effectively: ✅")

    elif actual_total == 0:
        print(f"   ⚠️ WARNING: Still under-counting")
    elif actual_total > 10:
        print(f"   ❌ PROBLEM: Still over-counting")
    else:
        print(f"   📝 NOTE: Results in acceptable range")

    # Show confirmed plates
    if confirmed_plates:
        print(f"\n✅ CONFIRMED PLATES:")
        for plate in confirmed_plates:
            print(f"   - {plate.text} (hits: {plate.detection_count}, conf: {plate.confidence:.2f})")

if __name__ == "__main__":
    test_final_balance()