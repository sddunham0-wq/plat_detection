#!/usr/bin/env python3
"""
Test HybridPlateDetector untuk debug mengapa tidak detect plates dari RTSP
"""

import sys
import os
import cv2
import time
import logging

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hybrid_plate_detector import HybridPlateDetector
from utils.yolo_detector import YOLOObjectDetector

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_hybrid_detector_rtsp():
    """Test HybridPlateDetector dengan RTSP stream"""
    print("🔍 Testing HybridPlateDetector dengan RTSP Stream")

    rtsp_url = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"

    try:
        # Initialize detector dengan debug mode
        print("📦 Initializing HybridPlateDetector...")
        plate_detector = HybridPlateDetector()
        yolo_detector = YOLOObjectDetector('yolov8n.pt', confidence=0.3)  # Lower confidence

        # Connect RTSP
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("❌ Failed to connect to RTSP")
            return

        print("✅ Connected to RTSP")
        print("🎥 Testing plate detection... Press 'q' to quit")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            print(f"\n--- Frame {frame_count} ---")

            # Test 1: Full frame detection
            print("🔍 Testing full frame detection...")
            try:
                plate_detections = plate_detector.detect_plates(frame)
                print(f"   📊 Full frame plates detected: {len(plate_detections)}")
                for plate in plate_detections:
                    print(f"   📄 Plate: '{plate.text}' conf={plate.confidence:.2f} bbox={plate.bbox}")
            except Exception as e:
                print(f"   ❌ Full frame detection failed: {e}")

            # Test 2: Vehicle ROI detection
            print("🚗 Testing vehicle ROI detection...")
            try:
                vehicles = yolo_detector.detect_objects(frame, vehicles_only=True)
                print(f"   📊 Vehicles detected: {len(vehicles)}")

                for i, vehicle in enumerate(vehicles):
                    print(f"   🚗 Vehicle {i+1}: {vehicle.class_name} conf={vehicle.confidence:.2f}")

                    # Extract ROI
                    x, y, w, h = vehicle.bbox
                    # Add padding
                    x = max(0, x - 10)
                    y = max(0, y - 10)
                    w = min(frame.shape[1] - x, w + 20)
                    h = min(frame.shape[0] - y, h + 20)

                    vehicle_roi = frame[y:y+h, x:x+w]

                    if vehicle_roi.size > 0:
                        # Test plate detection di ROI
                        roi_plates = plate_detector.detect_plates(vehicle_roi)
                        print(f"      📄 Plates in ROI: {len(roi_plates)}")

                        for plate in roi_plates:
                            # Adjust coordinates ke full frame
                            adjusted_bbox = (
                                plate.bbox[0] + x,
                                plate.bbox[1] + y,
                                plate.bbox[2],
                                plate.bbox[3]
                            )
                            print(f"         Plate: '{plate.text}' conf={plate.confidence:.2f}")

                        # Save ROI untuk debug jika ada masalah
                        if frame_count <= 3:  # Save first 3 ROIs
                            roi_filename = f"debug_vehicle_roi_frame{frame_count}_vehicle{i+1}.jpg"
                            cv2.imwrite(roi_filename, vehicle_roi)
                            print(f"         💾 ROI saved: {roi_filename}")

            except Exception as e:
                print(f"   ❌ Vehicle ROI detection failed: {e}")

            # Show frame dengan annotations
            annotated_frame = frame.copy()

            # Draw vehicles
            try:
                vehicles = yolo_detector.detect_objects(frame, vehicles_only=True)
                for vehicle in vehicles:
                    x, y, w, h = vehicle.bbox
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{vehicle.class_name} {vehicle.confidence:.1f}",
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except:
                pass

            # Draw plates
            try:
                plates = plate_detector.detect_plates(frame)
                for plate in plates:
                    x, y, w, h = plate.bbox
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{plate.text} {plate.confidence:.1f}",
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            except:
                pass

            # Add info
            cv2.putText(annotated_frame, f"Frame {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('HybridPlateDetector Test', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Test hanya 10 frames
            if frame_count >= 10:
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "="*50)
        print("🏁 HYBRID DETECTOR TEST COMPLETED")
        print("="*50)
        print("Check hasil untuk identify masalah:")
        print("1. Jika tidak ada plates detected sama sekali → Parameter terlalu ketat")
        print("2. Jika plates detected tapi OCR salah → OCR tuning needed")
        print("3. Jika vehicles detected tapi no plates in ROI → ROI extraction issue")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    test_hybrid_detector_rtsp()