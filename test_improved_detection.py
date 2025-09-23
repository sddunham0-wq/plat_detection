#!/usr/bin/env python3
"""
Test Script untuk Improved Detection System
Validate peningkatan akurasi dan precision detection
"""

import cv2
import numpy as np
import time
import logging
import os
import json
from typing import Dict, List, Tuple
from dataclasses import asdict

# Import modules
from utils.detection_manager import detection_manager, DetectionManager
from utils.bounding_box_refiner import bounding_box_refiner
from utils.plate_detector import LicensePlateDetector
from utils.motorcycle_plate_detector import MotorcyclePlateDetector
from config import *

class ImprovedDetectionTester:
    """
    Comprehensive testing untuk improved detection system
    """

    def __init__(self):
        """Initialize tester"""
        self.logger = logging.getLogger(__name__)

        # Setup detectors
        self.general_detector = LicensePlateDetector()
        try:
            self.motorcycle_detector = MotorcyclePlateDetector()
            if not self.motorcycle_detector.is_enabled():
                self.motorcycle_detector = None
        except Exception as e:
            self.logger.warning(f"Motorcycle detector unavailable: {str(e)}")
            self.motorcycle_detector = None

        # Reset detection manager stats
        detection_manager.reset_statistics()

        # Test results storage
        self.test_results = {
            'detection_accuracy': {},
            'duplicate_filtering': {},
            'bbox_precision': {},
            'quality_scoring': {},
            'performance_metrics': {}
        }

        self.logger.info("ImprovedDetectionTester initialized")

    def run_comprehensive_test(self, test_images: List[str] = None) -> Dict:
        """
        Run comprehensive test suite

        Args:
            test_images: List of test image paths

        Returns:
            Complete test results
        """
        print("🧪 Starting Comprehensive Detection System Test")
        print("=" * 60)

        # Test 1: Detection Accuracy
        print("\n1️⃣ Testing Detection Accuracy...")
        self.test_results['detection_accuracy'] = self._test_detection_accuracy(test_images)

        # Test 2: Duplicate Filtering
        print("\n2️⃣ Testing Duplicate Filtering...")
        self.test_results['duplicate_filtering'] = self._test_duplicate_filtering()

        # Test 3: Bounding Box Precision
        print("\n3️⃣ Testing Bounding Box Precision...")
        self.test_results['bbox_precision'] = self._test_bbox_precision(test_images)

        # Test 4: Quality Scoring
        print("\n4️⃣ Testing Quality Scoring System...")
        self.test_results['quality_scoring'] = self._test_quality_scoring()

        # Test 5: Performance Metrics
        print("\n5️⃣ Testing Performance Metrics...")
        self.test_results['performance_metrics'] = self._test_performance()

        # Generate comprehensive report
        self._generate_test_report()

        return self.test_results

    def _test_detection_accuracy(self, test_images: List[str] = None) -> Dict:
        """Test detection accuracy dengan new system"""
        results = {
            'plates_detected': 0,
            'valid_plates': 0,
            'false_positives': 0,
            'accuracy_rate': 0.0,
            'precision_rate': 0.0,
            'processing_times': [],
            'detection_details': []
        }

        # Use test images atau create synthetic ones
        if not test_images:
            test_images = self._create_synthetic_test_images()

        for i, image_path in enumerate(test_images):
            try:
                print(f"   Testing image {i+1}/{len(test_images)}: {image_path}")

                # Load image
                if isinstance(image_path, str) and os.path.exists(image_path):
                    frame = cv2.imread(image_path)
                else:
                    frame = image_path  # Assume it's numpy array

                if frame is None:
                    continue

                start_time = time.time()

                # Test dengan improved system
                detections_dict = {}

                # General detector
                general_detections = self.general_detector.detect_plates(frame)
                detections_dict['general'] = general_detections

                # Motorcycle detector jika tersedia
                if self.motorcycle_detector:
                    motorcycle_detections = self.motorcycle_detector.detect_plates(frame)
                    detections_dict['motorcycle'] = motorcycle_detections
                else:
                    detections_dict['motorcycle'] = []

                # Process dengan DetectionManager
                final_detections = detection_manager.process_detections(detections_dict, frame, i+1)

                processing_time = time.time() - start_time
                results['processing_times'].append(processing_time)

                # Analyze results
                total_detected = len(final_detections)
                valid_detected = sum(1 for d in final_detections if self._is_valid_plate_detection(d))

                results['plates_detected'] += total_detected
                results['valid_plates'] += valid_detected
                results['false_positives'] += (total_detected - valid_detected)

                # Store detail per image
                detail = {
                    'image_index': i+1,
                    'total_detected': total_detected,
                    'valid_detected': valid_detected,
                    'false_positives': total_detected - valid_detected,
                    'processing_time': processing_time,
                    'detections': [
                        {
                            'text': d.text,
                            'confidence': d.confidence,
                            'bbox': d.bbox,
                            'valid': self._is_valid_plate_detection(d)
                        } for d in final_detections
                    ]
                }
                results['detection_details'].append(detail)

                print(f"      Detected: {total_detected}, Valid: {valid_detected}, Time: {processing_time:.3f}s")

            except Exception as e:
                self.logger.error(f"Error testing image {i+1}: {str(e)}")

        # Calculate final metrics
        if results['plates_detected'] > 0:
            results['accuracy_rate'] = (results['valid_plates'] / results['plates_detected']) * 100
            results['precision_rate'] = results['accuracy_rate']

        avg_time = np.mean(results['processing_times']) if results['processing_times'] else 0
        results['avg_processing_time'] = avg_time

        print(f"   ✅ Accuracy: {results['accuracy_rate']:.1f}%")
        print(f"   ✅ Valid Plates: {results['valid_plates']}/{results['plates_detected']}")
        print(f"   ✅ Avg Time: {avg_time:.3f}s")

        return results

    def _test_duplicate_filtering(self) -> Dict:
        """Test duplicate filtering effectiveness"""
        results = {
            'total_input_detections': 0,
            'filtered_duplicates': 0,
            'final_detections': 0,
            'filtering_rate': 0.0,
            'test_scenarios': []
        }

        print("   Testing duplicate scenarios...")

        # Create test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Scenario 1: Same detection from multiple sources
        print("   - Testing same detection dari multiple sources...")
        from utils.plate_detector import PlateDetection

        # Create overlapping detections
        detection1 = PlateDetection("B1234ABC", 85.0, (100, 100, 200, 60), None, time.time())
        detection2 = PlateDetection("B1234ABC", 82.0, (105, 102, 198, 58), None, time.time())
        detection3 = PlateDetection("B1234ABC", 88.0, (98, 99, 202, 62), None, time.time())

        detections_dict = {
            'general': [detection1, detection2],
            'motorcycle': [detection3]
        }

        before_count = sum(len(dets) for dets in detections_dict.values())
        final_detections = detection_manager.process_detections(detections_dict, test_frame, 1)
        after_count = len(final_detections)

        scenario1 = {
            'name': 'Multiple source overlap',
            'input_count': before_count,
            'output_count': after_count,
            'filtered': before_count - after_count
        }
        results['test_scenarios'].append(scenario1)

        # Scenario 2: Temporal duplicates
        print("   - Testing temporal duplicate filtering...")
        time.sleep(0.1)  # Small delay
        temporal_detection = PlateDetection("B1234ABC", 86.0, (100, 100, 200, 60), None, time.time())
        temporal_dict = {'general': [temporal_detection], 'motorcycle': []}

        before_temporal = 1
        final_temporal = detection_manager.process_detections(temporal_dict, test_frame, 2)
        after_temporal = len(final_temporal)

        scenario2 = {
            'name': 'Temporal duplicates',
            'input_count': before_temporal,
            'output_count': after_temporal,
            'filtered': before_temporal - after_temporal
        }
        results['test_scenarios'].append(scenario2)

        # Calculate overall metrics
        total_input = sum(s['input_count'] for s in results['test_scenarios'])
        total_output = sum(s['output_count'] for s in results['test_scenarios'])
        total_filtered = total_input - total_output

        results['total_input_detections'] = total_input
        results['final_detections'] = total_output
        results['filtered_duplicates'] = total_filtered
        results['filtering_rate'] = (total_filtered / total_input * 100) if total_input > 0 else 0

        print(f"   ✅ Filtering Rate: {results['filtering_rate']:.1f}%")
        print(f"   ✅ Duplicates Filtered: {total_filtered}/{total_input}")

        return results

    def _test_bbox_precision(self, test_images: List[str] = None) -> Dict:
        """Test bounding box precision improvement"""
        results = {
            'total_bboxes_tested': 0,
            'refinement_success_rate': 0.0,
            'average_improvement': 0.0,
            'edge_quality_scores': [],
            'rectangularity_scores': [],
            'refinement_details': []
        }

        print("   Testing bounding box refinement...")

        # Use test images atau create ones
        if not test_images:
            test_images = self._create_synthetic_test_images(count=3)

        for i, image_path in enumerate(test_images):
            try:
                if isinstance(image_path, str) and os.path.exists(image_path):
                    frame = cv2.imread(image_path)
                else:
                    frame = image_path

                if frame is None:
                    continue

                # Get initial detections
                general_detections = self.general_detector.detect_plates(frame)

                for j, detection in enumerate(general_detections):
                    # Test refinement
                    refined_bbox = bounding_box_refiner.refine_bounding_box(
                        frame, detection.bbox, detection.confidence
                    )

                    if refined_bbox:
                        results['total_bboxes_tested'] += 1
                        results['edge_quality_scores'].append(refined_bbox.edge_quality)
                        results['rectangularity_scores'].append(refined_bbox.rectangularity)

                        # Calculate improvement
                        original_area = detection.bbox[2] * detection.bbox[3]
                        refined_area = refined_bbox.width * refined_bbox.height
                        area_diff = abs(refined_area - original_area) / original_area

                        detail = {
                            'image_index': i+1,
                            'detection_index': j+1,
                            'original_bbox': detection.bbox,
                            'refined_bbox': (refined_bbox.x, refined_bbox.y,
                                           refined_bbox.width, refined_bbox.height),
                            'edge_quality': refined_bbox.edge_quality,
                            'rectangularity': refined_bbox.rectangularity,
                            'confidence_change': refined_bbox.confidence - detection.confidence,
                            'area_change_ratio': area_diff
                        }
                        results['refinement_details'].append(detail)

            except Exception as e:
                self.logger.error(f"Error testing bbox refinement: {str(e)}")

        # Calculate metrics
        if results['total_bboxes_tested'] > 0:
            results['refinement_success_rate'] = len(results['refinement_details']) / results['total_bboxes_tested'] * 100
            results['average_edge_quality'] = np.mean(results['edge_quality_scores'])
            results['average_rectangularity'] = np.mean(results['rectangularity_scores'])

        print(f"   ✅ Refinement Success: {results['refinement_success_rate']:.1f}%")
        print(f"   ✅ Avg Edge Quality: {results.get('average_edge_quality', 0):.3f}")

        return results

    def _test_quality_scoring(self) -> Dict:
        """Test quality scoring system"""
        results = {
            'geometry_scores': [],
            'color_scores': [],
            'text_scores': [],
            'total_scores': [],
            'score_distribution': {},
            'validation_accuracy': 0.0
        }

        print("   Testing quality scoring components...")

        # Create test frame dengan known plates
        test_frame = self._create_test_frame_with_plates()

        # Test quality scoring
        detections_dict = {
            'general': self.general_detector.detect_plates(test_frame),
            'motorcycle': []
        }

        # Process dengan quality scoring
        final_detections = detection_manager.process_detections(detections_dict, test_frame, 1)

        # Get DetectionManager untuk access internal scoring
        test_manager = DetectionManager()

        for detection in final_detections:
            try:
                # Calculate quality scores
                quality = test_manager._calculate_quality_score(detection, test_frame)

                results['geometry_scores'].append(quality.geometry_score)
                results['color_scores'].append(quality.color_score)
                results['text_scores'].append(quality.text_score)
                results['total_scores'].append(quality.total_score)

            except Exception as e:
                self.logger.warning(f"Error calculating quality score: {str(e)}")

        # Score distribution
        if results['total_scores']:
            score_ranges = {'0.0-0.3': 0, '0.3-0.6': 0, '0.6-0.8': 0, '0.8-1.0': 0}
            for score in results['total_scores']:
                if score < 0.3:
                    score_ranges['0.0-0.3'] += 1
                elif score < 0.6:
                    score_ranges['0.3-0.6'] += 1
                elif score < 0.8:
                    score_ranges['0.6-0.8'] += 1
                else:
                    score_ranges['0.8-1.0'] += 1

            results['score_distribution'] = score_ranges
            results['average_quality'] = np.mean(results['total_scores'])

        print(f"   ✅ Avg Quality Score: {results.get('average_quality', 0):.3f}")
        print(f"   ✅ High Quality (>0.6): {sum([score > 0.6 for score in results['total_scores']])}/{len(results['total_scores'])}")

        return results

    def _test_performance(self) -> Dict:
        """Test performance metrics"""
        results = {
            'processing_times': [],
            'memory_usage': [],
            'detection_counts': [],
            'fps_measurements': [],
            'resource_efficiency': {}
        }

        print("   Testing performance metrics...")

        # Performance test dengan multiple frames
        test_frames = self._create_synthetic_test_images(count=10)

        start_total = time.time()

        for i, frame in enumerate(test_frames):
            frame_start = time.time()

            # Process frame
            detections_dict = {
                'general': self.general_detector.detect_plates(frame),
                'motorcycle': []
            }

            final_detections = detection_manager.process_detections(detections_dict, frame, i+1)

            frame_time = time.time() - frame_start
            results['processing_times'].append(frame_time)
            results['detection_counts'].append(len(final_detections))

            # Calculate FPS untuk frame ini
            fps = 1.0 / frame_time if frame_time > 0 else 0
            results['fps_measurements'].append(fps)

        total_time = time.time() - start_total

        # Calculate metrics
        results['average_processing_time'] = np.mean(results['processing_times'])
        results['average_fps'] = np.mean(results['fps_measurements'])
        results['total_processing_time'] = total_time
        results['frames_processed'] = len(test_frames)

        # Get DetectionManager statistics
        detection_stats = detection_manager.get_statistics()
        results['detection_manager_stats'] = detection_stats

        print(f"   ✅ Avg FPS: {results['average_fps']:.1f}")
        print(f"   ✅ Avg Processing Time: {results['average_processing_time']:.3f}s")

        return results

    def _is_valid_plate_detection(self, detection) -> bool:
        """Validate apakah detection adalah valid plate"""
        if not detection.text or len(detection.text) < 5:
            return False

        # Check aspect ratio
        x, y, w, h = detection.bbox
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 1.5 or aspect_ratio > 6.0:
            return False

        # Check confidence
        if detection.confidence < 30.0:
            return False

        return True

    def _create_synthetic_test_images(self, count: int = 5) -> List[np.ndarray]:
        """Create synthetic test images dengan plate-like objects"""
        images = []

        for i in range(count):
            # Create base image
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

            # Add plate-like rectangles
            for j in range(2):  # 2 plates per image
                x = np.random.randint(50, 400)
                y = np.random.randint(50, 300)
                w = np.random.randint(120, 250)
                h = np.random.randint(30, 80)

                # Draw white rectangle (plate-like)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

                # Add some text-like noise
                for k in range(3):
                    text_x = x + np.random.randint(10, w - 30)
                    text_y = y + np.random.randint(15, h - 5)
                    cv2.rectangle(img, (text_x, text_y - 10), (text_x + 20, text_y + 5), (0, 0, 0), -1)

            images.append(img)

        return images

    def _create_test_frame_with_plates(self) -> np.ndarray:
        """Create test frame dengan known plate characteristics"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add realistic plate
        cv2.rectangle(frame, (200, 200), (400, 260), (255, 255, 255), -1)  # White plate
        cv2.rectangle(frame, (200, 200), (400, 260), (0, 0, 0), 3)          # Black border

        # Add text-like elements
        cv2.putText(frame, "B 1234 XYZ", (210, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame

    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 IMPROVED DETECTION SYSTEM TEST REPORT")
        print("=" * 60)

        # Detection Accuracy
        accuracy = self.test_results['detection_accuracy']
        print(f"\n🎯 DETECTION ACCURACY:")
        print(f"   Total Detections: {accuracy['plates_detected']}")
        print(f"   Valid Detections: {accuracy['valid_plates']}")
        print(f"   Accuracy Rate: {accuracy['accuracy_rate']:.1f}%")
        print(f"   False Positives: {accuracy['false_positives']}")

        # Duplicate Filtering
        duplicates = self.test_results['duplicate_filtering']
        print(f"\n🔄 DUPLICATE FILTERING:")
        print(f"   Input Detections: {duplicates['total_input_detections']}")
        print(f"   Final Detections: {duplicates['final_detections']}")
        print(f"   Filtering Rate: {duplicates['filtering_rate']:.1f}%")

        # Bounding Box Precision
        bbox = self.test_results['bbox_precision']
        print(f"\n📦 BOUNDING BOX PRECISION:")
        print(f"   Refinement Success: {bbox['refinement_success_rate']:.1f}%")
        print(f"   Avg Edge Quality: {bbox.get('average_edge_quality', 0):.3f}")

        # Quality Scoring
        quality = self.test_results['quality_scoring']
        print(f"\n⭐ QUALITY SCORING:")
        print(f"   Average Quality: {quality.get('average_quality', 0):.3f}")
        if 'score_distribution' in quality:
            print(f"   Score Distribution: {quality['score_distribution']}")

        # Performance
        performance = self.test_results['performance_metrics']
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Average FPS: {performance['average_fps']:.1f}")
        print(f"   Avg Processing Time: {performance['average_processing_time']:.3f}s")

        # DetectionManager Stats
        dm_stats = detection_manager.get_statistics()
        print(f"\n📈 DETECTION MANAGER STATS:")
        print(f"   Total Detections: {dm_stats['total_detections']}")
        print(f"   Valid Detections: {dm_stats['valid_detections']}")
        print(f"   Success Rate: {dm_stats['success_rate']:.1f}%")
        print(f"   Duplicates Filtered: {dm_stats['filtered_duplicates']}")
        print(f"   Non-plates Filtered: {dm_stats['filtered_non_plates']}")

        # Overall Assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        overall_score = (
            accuracy['accuracy_rate'] * 0.3 +
            duplicates['filtering_rate'] * 0.2 +
            bbox['refinement_success_rate'] * 0.2 +
            (quality.get('average_quality', 0) * 100) * 0.15 +
            min(performance['average_fps'] / 10 * 100, 100) * 0.15
        )
        print(f"   Overall System Score: {overall_score:.1f}/100")

        if overall_score >= 80:
            print("   ✅ EXCELLENT - System ready for production")
        elif overall_score >= 60:
            print("   ⚠️  GOOD - Minor optimizations recommended")
        else:
            print("   ❌ NEEDS IMPROVEMENT - Major issues detected")

        # Save results to file
        with open('detection_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: detection_test_results.json")

def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("🚀 Starting Improved Detection System Test Suite")

    # Create tester
    tester = ImprovedDetectionTester()

    # Run comprehensive test
    results = tester.run_comprehensive_test()

    print("\n✅ Test Suite Completed!")

if __name__ == "__main__":
    main()