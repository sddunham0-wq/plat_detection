#!/usr/bin/env python3
"""
Test PlateCounterManager dengan fake detections untuk memastikan counter bekerja
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Tuple

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import create_plate_counter_manager

@dataclass
class FakePlateDetection:
    """Fake detection class untuk simulate plate detection"""
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    vehicle_type: str = "car"

def test_counter_with_fake_data():
    """Test counter dengan data palsu untuk memastikan logic bekerja"""
    print("🧪 Testing PlateCounterManager dengan fake detections")

    # Initialize counter seperti di stream_manager.py
    counter_config = {
        'similarity_threshold': 0.85,
        'spatial_proximity_distance': 60.0,
        'plate_expiry_time': 3.0,
        'confirmation_threshold': 2,
        'confidence_filter_min': 0.6
    }
    counter = create_plate_counter_manager(counter_config)

    # Test data yang sama seperti kasus user: 4 plat di stream
    fake_plates = [
        FakePlateDetection("B1234ABC", (100, 100, 150, 50), 0.8, "car"),
        FakePlateDetection("D5678XYZ", (300, 200, 150, 50), 0.9, "truck"),
        FakePlateDetection("F9012DEF", (500, 300, 150, 50), 0.7, "bus"),
        FakePlateDetection("H3456GHI", (200, 400, 150, 50), 0.85, "car")
    ]

    print(f"📊 Simulating detection of {len(fake_plates)} plates...")

    # Simulate multiple frames seperti di stream
    for frame_num in range(5):
        print(f"\n--- Frame {frame_num + 1} ---")

        # Process each plate detection seperti di stream_manager.py
        for detection in fake_plates:
            try:
                # Extract vehicle type dari detection (sama seperti di stream_manager.py line 364)
                vehicle_type = getattr(detection, 'vehicle_type', 'unknown')

                # Add detection ke counter manager (sama seperti di stream_manager.py line 367)
                plate_id = counter.add_or_update_detection(
                    detection_text=detection.text,
                    detection_bbox=detection.bbox,
                    confidence=detection.confidence,
                    tracking_id=None,  # Will be updated setelah tracking
                    vehicle_type=vehicle_type
                )

                # Set plate_id untuk reference (sama seperti di stream_manager.py line 377)
                if plate_id:
                    detection.plate_counter_id = plate_id
                    print(f"   ✅ Plate '{detection.text}' added to counter: ID={plate_id}")
                else:
                    print(f"   ❌ Plate '{detection.text}' not added (filtered)")

            except Exception as e:
                print(f"   ❌ Error adding detection to counter: {e}")

        # Get accurate counts dari PlateCounterManager (sama seperti di stream_manager.py line 493)
        accurate_counts = counter.get_current_counts()

        # Update stats seperti di stream_manager.py line 508
        total_detections = accurate_counts['total_unique_plates_session']
        current_unique_plates = accurate_counts['current_visible_plates']

        print(f"📈 Counter Stats:")
        print(f"   total_detections: {total_detections}")  # Ini yang ditampilkan di UI
        print(f"   current_unique_plates: {current_unique_plates}")
        print(f"   raw_processed: {accurate_counts['raw_detections_processed']}")
        print(f"   duplicates_filtered: {accurate_counts['duplicates_filtered']}")

        time.sleep(0.1)  # Simulate frame rate

    # Final results
    final_counts = counter.get_current_counts()
    print("\n" + "="*50)
    print("🏁 FINAL RESULTS")
    print("="*50)
    print(f"Expected unique plates: 4")
    print(f"Actual total_detections: {final_counts['total_unique_plates_session']}")
    print(f"Current visible plates: {final_counts['current_visible_plates']}")
    print(f"Raw detections processed: {final_counts['raw_detections_processed']}")
    print(f"Duplicates filtered: {final_counts['duplicates_filtered']}")

    # Test hasil
    expected = 4
    actual = final_counts['total_unique_plates_session']

    if actual == expected:
        print(f"\n✅ SUCCESS: Counter working correctly!")
        print(f"   Expected: {expected} plates")
        print(f"   Got: {actual} plates")
        print("   This means PlateCounterManager integration is working!")
        print("   The issue is that HybridPlateDetector is not detecting any plates from the RTSP stream.")
    else:
        print(f"\n❌ FAILED: Counter not working correctly")
        print(f"   Expected: {expected} plates")
        print(f"   Got: {actual} plates")

if __name__ == "__main__":
    test_counter_with_fake_data()