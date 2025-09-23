#!/usr/bin/env python3
"""
Debug script untuk check apakah PlateCounterManager menerima detections dari stream
"""

import sys
import os
import time
import cv2
import logging

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import create_plate_counter_manager
from utils.yolo_detector import YOLOObjectDetector
from utils.hybrid_plate_detector import HybridPlateDetector

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_counter_with_rtsp():
    """Test counter dengan RTSP stream langsung"""
    print("🔍 Debug Counter Integration dengan RTSP Stream")

    # Initialize components
    rtsp_url = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"

    try:
        # Initialize detector
        yolo_detector = YOLOObjectDetector('yolov8n.pt', confidence=0.4)
        plate_detector = HybridPlateDetector()

        # Initialize counter dengan debug mode
        counter_config = {
            'similarity_threshold': 0.8,
            'spatial_proximity_distance': 60.0,
            'plate_expiry_time': 3.0,
            'confirmation_threshold': 2,
            'confidence_filter_min': 0.5  # Lower untuk catch more
        }
        counter = create_plate_counter_manager(counter_config)

        # Connect RTSP
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("❌ Failed to connect to RTSP")
            return

        print("✅ Connected to RTSP")
        print("🎥 Processing frames... Press 'q' to quit")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Detect objects
            object_detections = yolo_detector.detect_objects(frame, vehicles_only=True)
            print(f"\n--- Frame {frame_count} ---")
            print(f"📊 Objects detected: {len(object_detections)}")

            # Process each vehicle for plates
            plate_detections = []
            for obj_detection in object_detections:
                print(f"🚗 Vehicle: {obj_detection.class_name} conf={obj_detection.confidence:.2f}")

                try:
                    # Extract ROI dan detect plates
                    x, y, w, h = obj_detection.bbox
                    vehicle_roi = frame[y:y+h, x:x+w]

                    if vehicle_roi.size > 0:
                        plates = plate_detector.detect_plates_in_roi(vehicle_roi, (x, y))

                        for plate in plates:
                            print(f"   📄 Plate detected: '{plate.text}' conf={plate.confidence:.2f}")

                            # Add ke counter
                            plate_id = counter.add_or_update_detection(
                                detection_text=plate.text,
                                detection_bbox=plate.bbox,
                                confidence=plate.confidence,
                                tracking_id=None,
                                vehicle_type=obj_detection.class_name
                            )

                            if plate_id:
                                print(f"   ✅ Added to counter: ID={plate_id}")
                            else:
                                print(f"   ❌ Not added to counter (filtered)")

                            plate_detections.append(plate)

                except Exception as e:
                    print(f"   ❌ Error processing vehicle: {e}")

            # Get counter stats
            counts = counter.get_current_counts()
            print(f"📈 Counter Stats:")
            print(f"   Current visible: {counts['current_visible_plates']}")
            print(f"   Total unique: {counts['total_unique_plates_session']}")
            print(f"   Raw processed: {counts['raw_detections_processed']}")
            print(f"   Duplicates filtered: {counts['duplicates_filtered']}")
            print(f"   False positives: {counts['false_positives_filtered']}")

            # Show frame dengan detections
            annotated_frame = frame.copy()

            # Draw vehicles
            for obj in object_detections:
                x, y, w, h = obj.bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{obj.class_name} {obj.confidence:.1f}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw plates
            for plate in plate_detections:
                x, y, w, h = plate.bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"{plate.text} {plate.confidence:.1f}",
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Add counter info
            info_text = [
                f"Frame: {frame_count}",
                f"Objects: {len(object_detections)}",
                f"Plates detected: {len(plate_detections)}",
                f"Counter visible: {counts['current_visible_plates']}",
                f"Counter total: {counts['total_unique_plates_session']}"
            ]

            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Debug Counter Integration', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Process only first 50 frames untuk debugging
            if frame_count >= 50:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Final stats
        final_counts = counter.get_current_counts()
        final_stats = counter.get_statistics()

        print("\n" + "="*50)
        print("🏁 FINAL DEBUG RESULTS")
        print("="*50)
        print(f"Frames processed: {frame_count}")
        print(f"Current visible plates: {final_counts['current_visible_plates']}")
        print(f"Total unique plates: {final_counts['total_unique_plates_session']}")
        print(f"Confirmed plates: {final_counts['confirmed_visible_plates']}")
        print(f"Raw detections processed: {final_counts['raw_detections_processed']}")
        print(f"Duplicates filtered: {final_counts['duplicates_filtered']}")
        print(f"False positives filtered: {final_counts['false_positives_filtered']}")
        print(f"Deduplication rate: {final_stats['accuracy_metrics']['deduplication_rate_percent']:.1f}%")

        if final_counts['total_unique_plates_session'] == 0:
            print("\n❌ PROBLEM: No plates counted!")
            print("Possible causes:")
            print("1. No plates detected in video")
            print("2. All detections filtered by confidence threshold")
            print("3. Integration issue between detector and counter")
        else:
            print(f"\n✅ SUCCESS: {final_counts['total_unique_plates_session']} unique plates counted")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    test_counter_with_rtsp()