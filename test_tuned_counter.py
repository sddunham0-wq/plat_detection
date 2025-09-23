#!/usr/bin/env python3
"""
Test tuned PlateCounterManager dengan confidence rendah untuk Indonesia
"""

import sys
import os
from dataclasses import dataclass
from typing import Tuple

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import create_plate_counter_manager

@dataclass
class FakePlateDetection:
    """Fake detection dengan confidence rendah untuk test"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    vehicle_type: str = "car"

def test_tuned_counter():
    """Test counter dengan confidence rendah seperti real Indonesia CCTV"""
    print("🇮🇩 Testing Tuned PlateCounterManager untuk Indonesia")

    # Load tuned config
    counter = create_plate_counter_manager()  # Will load from config file

    # Test dengan detections seperti real Indonesia CCTV (confidence rendah)
    low_confidence_plates = [
        FakePlateDetection("BB", (100, 100, 150, 50), 0.27, "bus"),  # Real dari debug log
        FakePlateDetection("D5", (300, 200, 150, 50), 0.31, "car"),  # Partial plate
        FakePlateDetection("F901", (500, 300, 150, 50), 0.35, "truck"), # Partial plate
        FakePlateDetection("H34", (200, 400, 150, 50), 0.42, "car")  # Medium confidence
    ]

    print(f"📊 Testing with {len(low_confidence_plates)} low confidence detections...")

    # Test Frame 1
    print(f"\n--- Frame 1 ---")
    for detection in low_confidence_plates:
        plate_id = counter.add_or_update_detection(
            detection_text=detection.text,
            detection_bbox=detection.bbox,
            confidence=detection.confidence,
            tracking_id=None,
            vehicle_type=detection.vehicle_type
        )

        if plate_id:
            print(f"   ✅ Plate '{detection.text}' (conf={detection.confidence:.2f}) → ID={plate_id}")
        else:
            print(f"   ❌ Plate '{detection.text}' (conf={detection.confidence:.2f}) → FILTERED")

    # Get counts
    counts = counter.get_current_counts()
    print(f"\n📈 Results:")
    print(f"   total_detections: {counts['total_unique_plates_session']}")
    print(f"   current_visible: {counts['current_visible_plates']}")
    print(f"   raw_processed: {counts['raw_detections_processed']}")
    print(f"   false_positives_filtered: {counts['false_positives_filtered']}")

    # Test dengan detections yang lebih bagus
    better_plates = [
        FakePlateDetection("B1234ABC", (400, 100, 150, 50), 0.85, "car"),
        FakePlateDetection("D5678XYZ", (600, 200, 150, 50), 0.75, "truck")
    ]

    print(f"\n--- Frame 2 (Better detections) ---")
    for detection in better_plates:
        plate_id = counter.add_or_update_detection(
            detection_text=detection.text,
            detection_bbox=detection.bbox,
            confidence=detection.confidence,
            tracking_id=None,
            vehicle_type=detection.vehicle_type
        )

        if plate_id:
            print(f"   ✅ Plate '{detection.text}' (conf={detection.confidence:.2f}) → ID={plate_id}")
        else:
            print(f"   ❌ Plate '{detection.text}' (conf={detection.confidence:.2f}) → FILTERED")

    # Final results
    final_counts = counter.get_current_counts()
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   total_detections: {final_counts['total_unique_plates_session']}")
    print(f"   current_visible: {final_counts['current_visible_plates']}")
    print(f"   raw_processed: {final_counts['raw_detections_processed']}")
    print(f"   false_positives_filtered: {final_counts['false_positives_filtered']}")

    if final_counts['total_unique_plates_session'] > 0:
        print(f"\n✅ SUCCESS: Counter now accepts low confidence Indonesia plates!")
        print(f"   Detected {final_counts['total_unique_plates_session']} unique plates")
        print("   This should fix the issue where detections show 0 padahal ada plates!")
    else:
        print(f"\n❌ STILL FAILING: Counter masih filter semua detections")

if __name__ == "__main__":
    test_tuned_counter()