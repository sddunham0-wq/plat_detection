#!/usr/bin/env python3
"""
Test realistic streaming scenario untuk final validation
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import PlateCounterManager

@dataclass
class RealisticPlate:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    appears_in_frames: int

def test_realistic_streaming():
    """Test realistic streaming scenario seperti user experience"""
    print("🎬 Testing REALISTIC Streaming Scenario")
    print("Simulating real user case: 1 plate visible di CCTV untuk beberapa frames")

    # Final config dari stream_manager.py
    counter = PlateCounterManager(
        similarity_threshold=0.85,
        spatial_proximity_distance=60.0,
        plate_expiry_time=3.0,
        confirmation_threshold=2,
        confidence_filter_min=0.45
    )

    # Realistic scenario: 1 plate yang terlihat consistently
    visible_plate = RealisticPlate(
        text="B1234ABC",
        bbox=(300, 200, 150, 50),
        confidence=0.62,
        appears_in_frames=8
    )

    # Occasional noise
    occasional_noise = [
        RealisticPlate("AB", (100, 100, 150, 50), 0.35, 2),  # Low confidence noise
        RealisticPlate("XY", (500, 300, 150, 50), 0.42, 1),  # Below threshold
    ]

    print(f"\n📺 Simulating 10 frames of CCTV footage...")
    print(f"   Real plate: '{visible_plate.text}' should appear in {visible_plate.appears_in_frames} frames")
    print(f"   Noise: Should be filtered out")

    # Simulate 10 frames
    for frame_num in range(10):
        print(f"\n--- Frame {frame_num + 1} ---")

        # Add main plate if it should appear
        if frame_num < visible_plate.appears_in_frames:
            # Add realistic variations (movement, confidence changes)
            variation_x = (frame_num % 3) - 1  # Small movement
            variation_y = (frame_num % 2)
            conf_variation = 0.62 + (frame_num * 0.02)  # Gradually better

            varied_bbox = (
                visible_plate.bbox[0] + variation_x,
                visible_plate.bbox[1] + variation_y,
                visible_plate.bbox[2],
                visible_plate.bbox[3]
            )

            plate_id = counter.add_or_update_detection(
                detection_text=visible_plate.text,
                detection_bbox=varied_bbox,
                confidence=min(conf_variation, 0.95),  # Cap at 95%
                tracking_id=1,  # Consistent tracking
                vehicle_type="car"
            )

            if plate_id:
                print(f"   ✅ Main plate detected: {visible_plate.text} (conf={conf_variation:.2f}) → ID={plate_id}")
            else:
                print(f"   ❌ Main plate filtered: {visible_plate.text}")

        # Occasionally add noise
        for noise in occasional_noise:
            if frame_num < noise.appears_in_frames:
                noise_id = counter.add_or_update_detection(
                    detection_text=noise.text,
                    detection_bbox=noise.bbox,
                    confidence=noise.confidence,
                    tracking_id=None,
                    vehicle_type="car"
                )

                status = "✅ ADDED" if noise_id else "❌ FILTERED"
                print(f"   Noise '{noise.text}' (conf={noise.confidence:.2f}) → {status}")

        # Get real-time counts (like UI would show)
        counts = counter.get_current_counts()
        print(f"   📊 Current UI would show: {counts['total_unique_plates_session']} detections")

        time.sleep(0.1)  # Simulate frame rate

    # Final results - what user sees
    final_counts = counter.get_current_counts()
    confirmed_plates = counter.get_confirmed_plates()

    print(f"\n📱 WHAT USER SEES IN UI:")
    print(f"   Detections: {final_counts['total_unique_plates_session']}")
    print(f"   (Previously was showing 200+ ❌ atau 0 ❌)")

    print(f"\n🏁 DETAILED RESULTS:")
    print(f"   Total unique plates: {final_counts['total_unique_plates_session']}")
    print(f"   Confirmed plates: {len(confirmed_plates)}")
    print(f"   Current visible: {final_counts['current_visible_plates']}")
    print(f"   Raw detections processed: {final_counts['raw_detections_processed']}")
    print(f"   False positives filtered: {final_counts['false_positives_filtered']}")
    print(f"   Duplicates filtered: {final_counts['duplicates_filtered']}")

    # Success criteria untuk user scenario
    displayed_count = final_counts['total_unique_plates_session']

    print(f"\n🎯 USER EXPERIENCE ANALYSIS:")

    if displayed_count == 1:
        print(f"   🎉 PERFECT: Shows exactly 1 detection!")
        print(f"   ✅ User problem SOLVED!")
        print(f"      - Before: 200+ detections (over-counting)")
        print(f"      - Before: 0 detections (under-counting)")
        print(f"      - Now: {displayed_count} detection (accurate!)")

    elif 1 <= displayed_count <= 3:
        print(f"   ✅ EXCELLENT: Shows {displayed_count} detections")
        print(f"   📝 Reasonable range, much better than 200+ atau 0")

    elif displayed_count == 0:
        print(f"   ⚠️ UNDER-COUNTING: Still showing 0 (user's original problem)")

    elif displayed_count > 10:
        print(f"   ❌ OVER-COUNTING: Still showing {displayed_count} (similar to 200+ problem)")

    else:
        print(f"   📊 ACCEPTABLE: Shows {displayed_count} detections")

    # Show confirmation details
    if confirmed_plates:
        print(f"\n✅ CONFIRMED DETECTIONS (what matters for accuracy):")
        for plate in confirmed_plates:
            print(f"   - {plate.text} (detected {plate.detection_count} times, conf: {plate.confidence:.2f})")
    else:
        print(f"\n📝 NOTE: No confirmed plates yet (need more consistent detection)")

    print(f"\n🎬 STREAMING TEST COMPLETED!")

if __name__ == "__main__":
    test_realistic_streaming()