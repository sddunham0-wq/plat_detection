#!/usr/bin/env python3
"""
Test Script untuk validasi accuracy PlateCounterManager
Mengatasi masalah over-counting detection yang melaporkan 20+ deteksi padahal hanya 4 plat

Test scenarios:
1. Baseline test - pastikan counter akurat dengan input controlled
2. Duplicate detection test - pastikan tidak ada double counting
3. Real RTSP stream test - test dengan stream asli
4. Stress test - test dengan banyak deteksi simultan
"""

import sys
import os
import time
import cv2
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add project root ke path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plate_counter_manager import PlateCounterManager, create_plate_counter_manager
from utils.yolo_detector import YOLOObjectDetector
import configparser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlateCounterTester:
    """Test suite untuk validasi akurasi PlateCounterManager"""

    def __init__(self):
        self.results = {}
        self.config = self.load_config()
        self.counter_manager = create_plate_counter_manager(self.config)

        # RTSP URL dari quick_rtsp_test.py
        self.rtsp_url = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"

        # Test data untuk controlled testing
        self.test_plates = [
            {"text": "B1234ABC", "bbox": (100, 100, 150, 50), "confidence": 0.8},
            {"text": "D5678XYZ", "bbox": (300, 200, 150, 50), "confidence": 0.9},
            {"text": "F9012DEF", "bbox": (500, 300, 150, 50), "confidence": 0.7},
            {"text": "H3456GHI", "bbox": (200, 400, 150, 50), "confidence": 0.85}
        ]

    def load_config(self) -> Dict:
        """Load configuration dari plate_counter_config.ini"""
        config_path = "config/plate_counter_config.ini"
        config = configparser.ConfigParser()

        try:
            config.read(config_path)
            return {
                'similarity_threshold': config.getfloat('PlateCounterSettings', 'similarity_threshold'),
                'spatial_proximity_distance': config.getfloat('PlateCounterSettings', 'spatial_proximity_distance'),
                'plate_expiry_time': config.getfloat('PlateCounterSettings', 'plate_expiry_time'),
                'confirmation_threshold': config.getint('PlateCounterSettings', 'confirmation_threshold'),
                'confidence_filter_min': config.getfloat('PlateCounterSettings', 'confidence_filter_min')
            }
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {}

    def test_1_baseline_accuracy(self) -> Dict[str, Any]:
        """Test 1: Baseline accuracy dengan controlled input"""
        logger.info("🧪 Test 1: Baseline Accuracy Test")
        self.counter_manager.reset_session_stats()

        # Simulate 4 unique plates detected multiple times
        frames_to_test = 10
        expected_unique = 4

        for frame_num in range(frames_to_test):
            logger.info(f"   Frame {frame_num + 1}/{frames_to_test}")

            for plate in self.test_plates:
                # Add small position variation untuk simulate real detection
                varied_bbox = (
                    plate["bbox"][0] + (frame_num % 3) - 1,  # Small x variation
                    plate["bbox"][1] + (frame_num % 2),      # Small y variation
                    plate["bbox"][2],                         # Same width
                    plate["bbox"][3]                          # Same height
                )

                self.counter_manager.add_or_update_detection(
                    detection_text=plate["text"],
                    detection_bbox=varied_bbox,
                    confidence=plate["confidence"] + (frame_num * 0.01),  # Slight confidence variation
                    tracking_id=None,
                    vehicle_type="car"
                )

        counts = self.counter_manager.get_current_counts()
        stats = self.counter_manager.get_statistics()

        result = {
            'test_name': 'Baseline Accuracy',
            'frames_processed': frames_to_test,
            'expected_unique_plates': expected_unique,
            'actual_unique_plates': counts['total_unique_plates_session'],
            'confirmed_plates': counts['confirmed_plates_total'],
            'raw_detections': counts['raw_detections_processed'],
            'duplicates_filtered': counts['duplicates_filtered'],
            'false_positives_filtered': counts['false_positives_filtered'],
            'accuracy_rate': (counts['total_unique_plates_session'] / expected_unique * 100) if expected_unique > 0 else 0,
            'deduplication_rate': stats['accuracy_metrics']['deduplication_rate_percent'],
            'passed': counts['total_unique_plates_session'] == expected_unique
        }

        logger.info(f"   ✅ Expected: {expected_unique}, Got: {counts['total_unique_plates_session']}")
        logger.info(f"   📊 Raw detections: {counts['raw_detections_processed']}")
        logger.info(f"   🔍 Duplicates filtered: {counts['duplicates_filtered']}")
        logger.info(f"   ✅ Test {'PASSED' if result['passed'] else 'FAILED'}")

        return result

    def test_2_duplicate_filtering(self) -> Dict[str, Any]:
        """Test 2: Duplicate detection filtering"""
        logger.info("🧪 Test 2: Duplicate Detection Filtering")
        self.counter_manager.reset_session_stats()

        # Simulate same plate detected rapidly (should be filtered)
        plate = self.test_plates[0]
        rapid_detections = 20  # Simulate 20 rapid detections of same plate

        for i in range(rapid_detections):
            self.counter_manager.add_or_update_detection(
                detection_text=plate["text"],
                detection_bbox=plate["bbox"],
                confidence=plate["confidence"],
                tracking_id=1,
                vehicle_type="car"
            )
            time.sleep(0.05)  # 50ms interval - should trigger duplicate filtering

        counts = self.counter_manager.get_current_counts()
        expected_unique = 1  # Should only count as 1 unique plate

        result = {
            'test_name': 'Duplicate Filtering',
            'rapid_detections_sent': rapid_detections,
            'expected_unique_plates': expected_unique,
            'actual_unique_plates': counts['total_unique_plates_session'],
            'duplicates_filtered': counts['duplicates_filtered'],
            'raw_detections': counts['raw_detections_processed'],
            'filter_effectiveness': (counts['duplicates_filtered'] / rapid_detections * 100) if rapid_detections > 0 else 0,
            'passed': counts['total_unique_plates_session'] == expected_unique and counts['duplicates_filtered'] > 0
        }

        logger.info(f"   📤 Sent: {rapid_detections} rapid detections")
        logger.info(f"   ✅ Unique plates: {counts['total_unique_plates_session']}")
        logger.info(f"   🚫 Duplicates filtered: {counts['duplicates_filtered']}")
        logger.info(f"   ✅ Test {'PASSED' if result['passed'] else 'FAILED'}")

        return result

    def test_3_spatial_filtering(self) -> Dict[str, Any]:
        """Test 3: Spatial proximity filtering"""
        logger.info("🧪 Test 3: Spatial Proximity Filtering")
        self.counter_manager.reset_session_stats()

        # Test plates at different distances
        base_plate = self.test_plates[0]

        # Same plate at different positions
        positions = [
            (100, 100, 150, 50),  # Original position
            (105, 105, 150, 50),  # Close position (should be same plate)
            (200, 200, 150, 50),  # Far position (should be different)
        ]

        for i, bbox in enumerate(positions):
            self.counter_manager.add_or_update_detection(
                detection_text=base_plate["text"],
                detection_bbox=bbox,
                confidence=base_plate["confidence"],
                tracking_id=None,
                vehicle_type="car"
            )

        counts = self.counter_manager.get_current_counts()
        # Should recognize positions 1&2 as same plate, position 3 as different
        expected_unique = 2

        result = {
            'test_name': 'Spatial Filtering',
            'positions_tested': len(positions),
            'expected_unique_plates': expected_unique,
            'actual_unique_plates': counts['total_unique_plates_session'],
            'spatial_proximity_distance': self.config.get('spatial_proximity_distance', 60.0),
            'passed': counts['total_unique_plates_session'] == expected_unique
        }

        logger.info(f"   📍 Positions tested: {len(positions)}")
        logger.info(f"   ✅ Unique plates: {counts['total_unique_plates_session']}")
        logger.info(f"   ✅ Test {'PASSED' if result['passed'] else 'FAILED'}")

        return result

    def test_4_rtsp_stream_accuracy(self) -> Dict[str, Any]:
        """Test 4: Real RTSP stream accuracy test"""
        logger.info("🧪 Test 4: Real RTSP Stream Accuracy")

        try:
            # Initialize YOLO detector
            detector = YOLOObjectDetector('yolov8n.pt', confidence=0.6)

            # Connect to RTSP stream
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                return {
                    'test_name': 'RTSP Stream Accuracy',
                    'error': 'Failed to connect to RTSP stream',
                    'passed': False
                }

            self.counter_manager.reset_session_stats()

            frames_to_process = 30  # Process 30 frames
            frame_count = 0
            start_time = time.time()

            logger.info(f"   🎥 Processing {frames_to_process} frames from RTSP stream...")

            while frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect vehicles and plates
                detections = detector.detect_objects(frame, vehicles_only=True)

                for detection in detections:
                    if hasattr(detection, 'plates') and detection.plates:
                        for plate_text in detection.plates:
                            self.counter_manager.add_or_update_detection(
                                detection_text=plate_text,
                                detection_bbox=detection.bbox,
                                confidence=detection.confidence,
                                tracking_id=getattr(detection, 'track_id', None),
                                vehicle_type=detection.label
                            )

                # Log progress every 10 frames
                if frame_count % 10 == 0:
                    counts = self.counter_manager.get_current_counts()
                    logger.info(f"   Frame {frame_count}: {counts['current_visible_plates']} plates visible")

            cap.release()

            # Final results
            counts = self.counter_manager.get_current_counts()
            stats = self.counter_manager.get_statistics()
            processing_time = time.time() - start_time

            result = {
                'test_name': 'RTSP Stream Accuracy',
                'frames_processed': frame_count,
                'processing_time_seconds': round(processing_time, 2),
                'current_visible_plates': counts['current_visible_plates'],
                'total_unique_plates': counts['total_unique_plates_session'],
                'confirmed_plates': counts['confirmed_plates_total'],
                'raw_detections': counts['raw_detections_processed'],
                'duplicates_filtered': counts['duplicates_filtered'],
                'false_positives_filtered': counts['false_positives_filtered'],
                'deduplication_rate': stats['accuracy_metrics']['deduplication_rate_percent'],
                'confirmation_rate': stats['accuracy_metrics']['confirmation_rate_percent'],
                'avg_processing_time_ms': stats['performance_metrics']['avg_processing_time_ms'],
                'passed': True  # Success if no errors
            }

            logger.info(f"   🎥 Processed {frame_count} frames in {processing_time:.2f}s")
            logger.info(f"   👁️ Currently visible: {counts['current_visible_plates']} plates")
            logger.info(f"   📊 Total unique: {counts['total_unique_plates_session']} plates")
            logger.info(f"   🚫 Duplicates filtered: {counts['duplicates_filtered']}")
            logger.info(f"   ✅ Test COMPLETED")

            return result

        except Exception as e:
            logger.error(f"   ❌ RTSP test failed: {e}")
            return {
                'test_name': 'RTSP Stream Accuracy',
                'error': str(e),
                'passed': False
            }

    def test_5_stress_test(self) -> Dict[str, Any]:
        """Test 5: Stress test dengan banyak deteksi"""
        logger.info("🧪 Test 5: Stress Test")
        self.counter_manager.reset_session_stats()

        start_time = time.time()
        stress_detections = 1000
        expected_unique = len(self.test_plates)  # 4 unique plates

        logger.info(f"   ⚡ Sending {stress_detections} detections...")

        for i in range(stress_detections):
            plate = self.test_plates[i % len(self.test_plates)]

            # Add random variations
            varied_bbox = (
                plate["bbox"][0] + (i % 10) - 5,
                plate["bbox"][1] + (i % 8) - 4,
                plate["bbox"][2],
                plate["bbox"][3]
            )

            self.counter_manager.add_or_update_detection(
                detection_text=plate["text"],
                detection_bbox=varied_bbox,
                confidence=plate["confidence"],
                tracking_id=i % 10,  # Simulate different tracking IDs
                vehicle_type="car"
            )

        processing_time = time.time() - start_time
        counts = self.counter_manager.get_current_counts()
        stats = self.counter_manager.get_statistics()

        result = {
            'test_name': 'Stress Test',
            'stress_detections_sent': stress_detections,
            'processing_time_seconds': round(processing_time, 2),
            'detections_per_second': round(stress_detections / processing_time),
            'expected_unique_plates': expected_unique,
            'actual_unique_plates': counts['total_unique_plates_session'],
            'duplicates_filtered': counts['duplicates_filtered'],
            'avg_processing_time_ms': stats['performance_metrics']['avg_processing_time_ms'],
            'deduplication_efficiency': round(counts['duplicates_filtered'] / stress_detections * 100, 1),
            'passed': counts['total_unique_plates_session'] == expected_unique
        }

        logger.info(f"   ⚡ Processed {stress_detections} detections in {processing_time:.2f}s")
        logger.info(f"   🚀 {result['detections_per_second']} detections/second")
        logger.info(f"   ✅ Unique plates: {counts['total_unique_plates_session']}")
        logger.info(f"   🚫 Duplicates filtered: {counts['duplicates_filtered']} ({result['deduplication_efficiency']}%)")
        logger.info(f"   ✅ Test {'PASSED' if result['passed'] else 'FAILED'}")

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run semua test cases"""
        logger.info("🚀 Starting PlateCounterManager Accuracy Tests")
        logger.info("=" * 60)

        test_results = {}

        # Run all tests
        test_results['test_1'] = self.test_1_baseline_accuracy()
        test_results['test_2'] = self.test_2_duplicate_filtering()
        test_results['test_3'] = self.test_3_spatial_filtering()
        test_results['test_4'] = self.test_4_rtsp_stream_accuracy()
        test_results['test_5'] = self.test_5_stress_test()

        # Summary
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get('passed', False))

        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': round(passed_tests / total_tests * 100, 1),
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config
        }

        final_result = {
            'summary': summary,
            'test_results': test_results
        }

        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info(f"   Total tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {total_tests - passed_tests}")
        logger.info(f"   Success rate: {summary['success_rate']}%")

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - Counter accuracy fixed!")
        else:
            logger.warning("⚠️ Some tests failed - review results")

        return final_result

    def save_results(self, results: Dict[str, Any]):
        """Save test results ke file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_counter_test_results_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 Results saved to: {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main function untuk run tests"""
    print("🧪 PlateCounterManager Accuracy Test Suite")
    print("Mengatasi masalah over-counting detection (20+ deteksi untuk 4 plat)")
    print()

    tester = PlateCounterTester()

    try:
        results = tester.run_all_tests()
        tester.save_results(results)

        # Exit dengan status berdasarkan hasil test
        success_rate = results['summary']['success_rate']
        exit_code = 0 if success_rate == 100 else 1

        print(f"\n🏁 Tests completed with {success_rate}% success rate")
        return exit_code

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())