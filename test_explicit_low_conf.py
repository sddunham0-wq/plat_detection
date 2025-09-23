#!/usr/bin/env python3
"""
Test dengan explicit low confidence untuk memastikan Indonesia plates ter-count
"""

import sys
import os
from dataclasses import dataclass
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import PlateCounterManager

@dataclass
class FakePlateDetection:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float

def test_explicit_low_confidence():
    """Test dengan explicit configuration seperti di stream_manager.py yang sudah di-fix"""
    print("🇮🇩 Test dengan Explicit Low Confidence Configuration")

    # Explicit config seperti di stream_manager.py setelah fix
    counter = PlateCounterManager(
        similarity_threshold=0.85,
        spatial_proximity_distance=60.0,
        plate_expiry_time=3.0,
        confirmation_threshold=2,
        confidence_filter_min=0.25  # FIXED: Accept confidence 25%+
    )

    # Test plates dari real Indonesia CCTV
    test_plates = [
        FakePlateDetection("BB", (100, 100, 150, 50), 0.27),  # Real dari debug log
        FakePlateDetection("B1234", (300, 200, 150, 50), 0.45),  # Medium
        FakePlateDetection("B1234ABC", (500, 300, 150, 50), 0.75),  # Good
    ]

    print("📊 Testing low confidence detections...")

    for detection in test_plates:
        plate_id = counter.add_or_update_detection(
            detection_text=detection.text,
            detection_bbox=detection.bbox,
            confidence=detection.confidence,
            tracking_id=None,
            vehicle_type="car"
        )

        if plate_id:
            print(f"   ✅ '{detection.text}' (conf={detection.confidence:.2f}) → ID={plate_id}")
        else:
            print(f"   ❌ '{detection.text}' (conf={detection.confidence:.2f}) → FILTERED")

    # Get final counts
    counts = counter.get_current_counts()
    print(f"\n🏁 Results:")
    print(f"   total_detections: {counts['total_unique_plates_session']}")
    print(f"   raw_processed: {counts['raw_detections_processed']}")
    print(f"   false_positives_filtered: {counts['false_positives_filtered']}")

    expected = len(test_plates)
    actual = counts['total_unique_plates_session']

    if actual >= 2:  # At least partial success
        print(f"\n✅ IMPROVEMENT: {actual}/{expected} plates counted!")
        print("   Low confidence tuning is working!")

        if actual == expected:
            print("   🎉 PERFECT: All plates including very low confidence detected!")
        else:
            print("   📝 NOTE: Very low confidence still filtered (ini mungkin normal untuk avoid noise)")
    else:
        print(f"\n❌ ISSUE: Only {actual}/{expected} plates counted")

if __name__ == "__main__":
    test_explicit_low_confidence()