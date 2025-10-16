#!/usr/bin/env python3
"""
Zero False Positive Test Suite
Comprehensive testing untuk validate false positive elimination
"""

import cv2
import numpy as np
import time
import logging
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict

# Import modules
from utils.detection_manager import detection_manager
from utils.scene_analyzer import scene_analyzer
from utils.validation_pipeline import validation_pipeline
from utils.plate_detector import LicensePlateDetector
from utils.motorcycle_plate_detector import MotorcyclePlateDetector
from config import *

class ZeroFalsePositiveTester:
    """
    Comprehensive testing untuk zero false positive validation
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

        # Reset systems
        detection_manager.reset_statistics()
        scene_analyzer.reset()

        # Test scenarios
        self.test_scenarios = [
            "empty_parking_lot",
            "plain_wall",
            "trees_and_plants",
            "building_facade",
            "road_without_vehicles",
            "static_objects",
            "uniform_background",
            "high_contrast_patterns",
            "text_on_buildings",
            "shadows_and_lighting"
        ]

        self.logger.info("ZeroFalsePositiveTester initialized")

    def run_comprehensive_test(self) -> Dict:
        """
        Run comprehensive zero false positive test

        Returns:
            Complete test results
        """
        print("🚫 Starting Zero False Positive Test Suite")
        print("=" * 60)

        results = {
            'empty_scene_tests': {},
            'non_plate_object_tests': {},
            'edge_case_tests': {},
            'performance_metrics': {},
            'overall_assessment': {}
        }

        # Test 1: Empty Scene Detection
        print("\n1️⃣ Testing Empty Scene Detection...")
        results['empty_scene_tests'] = self._test_empty_scenes()

        # Test 2: Non-Plate Object Rejection
        print("\n2️⃣ Testing Non-Plate Object Rejection...")
        results['non_plate_object_tests'] = self._test_non_plate_objects()

        # Test 3: Edge Cases
        print("\n3️⃣ Testing Edge Cases...")
        results['edge_case_tests'] = self._test_edge_cases()

        # Test 4: Performance Impact
        print("\n4️⃣ Testing Performance Impact...")
        results['performance_metrics'] = self._test_performance_impact()

        # Test 5: Overall Assessment
        print("\n5️⃣ Overall Assessment...")
        results['overall_assessment'] = self._calculate_overall_assessment(results)

        # Generate report
        self._generate_zero_fp_report(results)

        return results

    def _test_empty_scenes(self) -> Dict:
        """Test detection pada empty scenes"""
        results = {
            'scenarios_tested': 0,
            'false_positives': 0,
            'scene_analysis_accuracy': 0.0,
            'scenario_details': []
        }

        empty_scenarios = [
            self._create_empty_parking_lot(),
            self._create_plain_wall(),
            self._create_uniform_background(),
            self._create_empty_road(),
            self._create_building_facade()
        ]

        for i, (scenario_name, frame) in enumerate(empty_scenarios):
            print(f"   Testing {scenario_name}...")

            # Test scene analysis
            scene_analysis = scene_analyzer.analyze_frame(frame)

            # Test detection pipeline
            detections_dict = {
                'general': self.general_detector.detect_plates(frame),
                'motorcycle': []
            }

            if self.motorcycle_detector:
                detections_dict['motorcycle'] = self.motorcycle_detector.detect_plates(frame)

            # Process dengan DetectionManager
            final_detections = detection_manager.process_detections(detections_dict, frame, i+1)

            # Analyze results
            false_positive_count = len(final_detections)
            scene_correctly_identified = scene_analysis.likely_empty

            results['scenarios_tested'] += 1
            results['false_positives'] += false_positive_count

            scenario_detail = {
                'scenario': scenario_name,
                'false_positives': false_positive_count,
                'scene_empty_detected': scene_correctly_identified,
                'scene_metrics': {
                    'uniformity': scene_analysis.background_uniformity,
                    'edge_density': scene_analysis.edge_density,
                    'contrast_ratio': scene_analysis.contrast_ratio,
                    'has_vehicles': scene_analysis.has_vehicles
                },
                'detections': [
                    {
                        'text': d.text,
                        'confidence': d.confidence,
                        'bbox': d.bbox
                    } for d in final_detections
                ]
            }
            results['scenario_details'].append(scenario_detail)

            status = "✅ PASS" if false_positive_count == 0 else f"❌ FAIL ({false_positive_count} FP)"
            scene_status = "✅" if scene_correctly_identified else "❌"
            print(f"      Detection: {status}, Scene Analysis: {scene_status}")

        # Calculate accuracy
        correct_scene_analysis = sum(1 for detail in results['scenario_details']
                                   if detail['scene_empty_detected'])
        results['scene_analysis_accuracy'] = (correct_scene_analysis / results['scenarios_tested'] * 100) if results['scenarios_tested'] > 0 else 0

        print(f"   ✅ Empty Scene Analysis Accuracy: {results['scene_analysis_accuracy']:.1f}%")
        print(f"   ✅ Total False Positives: {results['false_positives']}")

        return results

    def _test_non_plate_objects(self) -> Dict:
        """Test rejection of non-plate objects"""
        results = {
            'objects_tested': 0,
            'incorrect_detections': 0,
            'object_details': []
        }

        non_plate_scenarios = [
            self._create_scene_with_plants(),
            self._create_scene_with_signs(),
            self._create_scene_with_windows(),
            self._create_scene_with_rectangular_objects(),
            self._create_scene_with_text_on_building()
        ]

        for i, (scenario_name, frame) in enumerate(non_plate_scenarios):
            print(f"   Testing {scenario_name}...")

            # Get detections
            detections_dict = {
                'general': self.general_detector.detect_plates(frame),
                'motorcycle': []
            }

            if self.motorcycle_detector:
                detections_dict['motorcycle'] = self.motorcycle_detector.detect_plates(frame)

            # Process dengan validation pipeline
            final_detections = detection_manager.process_detections(detections_dict, frame, i+1)

            # Count incorrect detections (should be 0)
            incorrect_count = len(final_detections)

            results['objects_tested'] += 1
            results['incorrect_detections'] += incorrect_count

            object_detail = {
                'scenario': scenario_name,
                'incorrect_detections': incorrect_count,
                'raw_detections': len(detections_dict['general']) + len(detections_dict['motorcycle']),
                'filtered_detections': [
                    {
                        'text': d.text,
                        'confidence': d.confidence,
                        'bbox': d.bbox
                    } for d in final_detections
                ]
            }
            results['object_details'].append(object_detail)

            status = "✅ PASS" if incorrect_count == 0 else f"❌ FAIL ({incorrect_count} incorrect)"
            print(f"      Result: {status}")

        print(f"   ✅ Objects Tested: {results['objects_tested']}")
        print(f"   ✅ Incorrect Detections: {results['incorrect_detections']}")

        return results

    def _test_edge_cases(self) -> Dict:
        """Test edge cases yang might trigger false positives"""
        results = {
            'edge_cases_tested': 0,
            'failures': 0,
            'case_details': []
        }

        edge_cases = [
            self._create_low_contrast_scene(),
            self._create_high_noise_scene(),
            self._create_motion_blur_scene(),
            self._create_extreme_lighting_scene(),
            self._create_partial_occlusion_scene()
        ]

        for i, (case_name, frame) in enumerate(edge_cases):
            print(f"   Testing {case_name}...")

            # Test dengan robust validation
            detections_dict = {
                'general': self.general_detector.detect_plates(frame),
                'motorcycle': []
            }

            final_detections = detection_manager.process_detections(detections_dict, frame, i+1)

            # Edge cases should either detect correctly atau not detect at all
            # False positives are failures
            failure_count = len([d for d in final_detections if not self._is_realistic_plate_text(d.text)])

            results['edge_cases_tested'] += 1
            results['failures'] += failure_count

            case_detail = {
                'case': case_name,
                'total_detections': len(final_detections),
                'unrealistic_detections': failure_count,
                'detections': [
                    {
                        'text': d.text,
                        'confidence': d.confidence,
                        'realistic': self._is_realistic_plate_text(d.text)
                    } for d in final_detections
                ]
            }
            results['case_details'].append(case_detail)

            status = "✅ PASS" if failure_count == 0 else f"❌ FAIL ({failure_count} unrealistic)"
            print(f"      Result: {status}")

        print(f"   ✅ Edge Cases Tested: {results['edge_cases_tested']}")
        print(f"   ✅ Failures: {results['failures']}")

        return results

    def _test_performance_impact(self) -> Dict:
        """Test performance impact dari validation pipeline"""
        results = {
            'baseline_performance': {},
            'enhanced_performance': {},
            'performance_overhead': 0.0
        }

        print("   Testing performance impact...")

        # Create test frames
        test_frames = [
            self._create_empty_parking_lot()[1],
            self._create_scene_with_plants()[1],
            self._create_uniform_background()[1]
        ]

        # Test baseline (without validation)
        baseline_times = []
        for frame in test_frames:
            start_time = time.time()

            # Basic detection only
            general_detections = self.general_detector.detect_plates(frame)

            baseline_time = time.time() - start_time
            baseline_times.append(baseline_time)

        # Test dengan validation pipeline
        enhanced_times = []
        for i, frame in enumerate(test_frames):
            start_time = time.time()

            # Full pipeline
            detections_dict = {'general': self.general_detector.detect_plates(frame), 'motorcycle': []}
            final_detections = detection_manager.process_detections(detections_dict, frame, i+1)

            enhanced_time = time.time() - start_time
            enhanced_times.append(enhanced_time)

        # Calculate metrics
        avg_baseline = np.mean(baseline_times)
        avg_enhanced = np.mean(enhanced_times)
        overhead = ((avg_enhanced - avg_baseline) / avg_baseline * 100) if avg_baseline > 0 else 0

        results['baseline_performance'] = {
            'avg_time': avg_baseline,
            'times': baseline_times
        }
        results['enhanced_performance'] = {
            'avg_time': avg_enhanced,
            'times': enhanced_times
        }
        results['performance_overhead'] = overhead

        print(f"   ✅ Baseline Avg: {avg_baseline:.3f}s")
        print(f"   ✅ Enhanced Avg: {avg_enhanced:.3f}s")
        print(f"   ✅ Overhead: {overhead:.1f}%")

        return results

    def _calculate_overall_assessment(self, results: Dict) -> Dict:
        """Calculate overall assessment"""
        assessment = {
            'total_false_positives': 0,
            'total_scenarios': 0,
            'false_positive_rate': 0.0,
            'scene_analysis_accuracy': 0.0,
            'validation_effectiveness': 0.0,
            'grade': 'F',
            'recommendations': []
        }

        # Calculate totals
        assessment['total_false_positives'] = (
            results['empty_scene_tests']['false_positives'] +
            results['non_plate_object_tests']['incorrect_detections'] +
            results['edge_case_tests']['failures']
        )

        assessment['total_scenarios'] = (
            results['empty_scene_tests']['scenarios_tested'] +
            results['non_plate_object_tests']['objects_tested'] +
            results['edge_case_tests']['edge_cases_tested']
        )

        # Calculate rates
        if assessment['total_scenarios'] > 0:
            assessment['false_positive_rate'] = (assessment['total_false_positives'] / assessment['total_scenarios']) * 100

        assessment['scene_analysis_accuracy'] = results['empty_scene_tests']['scene_analysis_accuracy']

        # Calculate validation effectiveness
        detection_manager_stats = detection_manager.get_statistics()
        if detection_manager_stats['filtered_non_plates'] > 0:
            assessment['validation_effectiveness'] = (
                detection_manager_stats['filtered_non_plates'] /
                (detection_manager_stats['filtered_non_plates'] + assessment['total_false_positives'])
            ) * 100

        # Assign grade
        if assessment['false_positive_rate'] == 0:
            assessment['grade'] = 'A+'
        elif assessment['false_positive_rate'] <= 5:
            assessment['grade'] = 'A'
        elif assessment['false_positive_rate'] <= 10:
            assessment['grade'] = 'B'
        elif assessment['false_positive_rate'] <= 20:
            assessment['grade'] = 'C'
        else:
            assessment['grade'] = 'F'

        # Generate recommendations
        if assessment['false_positive_rate'] > 0:
            assessment['recommendations'].append("Increase validation thresholds")
        if assessment['scene_analysis_accuracy'] < 90:
            assessment['recommendations'].append("Improve scene analysis algorithms")
        if results['performance_metrics']['performance_overhead'] > 50:
            assessment['recommendations'].append("Optimize validation pipeline performance")

        return assessment

    # Scene creation methods
    def _create_empty_parking_lot(self) -> Tuple[str, np.ndarray]:
        """Create empty parking lot scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        # Add parking lines
        for i in range(0, 640, 80):
            cv2.line(frame, (i, 200), (i, 400), (255, 255, 255), 2)
        return ("empty_parking_lot", frame)

    def _create_plain_wall(self) -> Tuple[str, np.ndarray]:
        """Create plain wall scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
        # Add some texture
        noise = np.random.randint(-20, 20, (480, 640, 3))
        frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        return ("plain_wall", frame)

    def _create_uniform_background(self) -> Tuple[str, np.ndarray]:
        """Create uniform background"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
        return ("uniform_background", frame)

    def _create_empty_road(self) -> Tuple[str, np.ndarray]:
        """Create empty road scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        # Add road markings
        cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 3)
        return ("empty_road", frame)

    def _create_building_facade(self) -> Tuple[str, np.ndarray]:
        """Create building facade"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 160
        # Add windows
        for i in range(100, 500, 120):
            for j in range(50, 350, 100):
                cv2.rectangle(frame, (i, j), (i+80, j+60), (50, 50, 50), -1)
        return ("building_facade", frame)

    def _create_scene_with_plants(self) -> Tuple[str, np.ndarray]:
        """Create scene dengan plants"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        # Add plant-like shapes
        for i in range(5):
            x = np.random.randint(50, 550)
            y = np.random.randint(100, 350)
            cv2.circle(frame, (x, y), np.random.randint(20, 60), (0, 100, 0), -1)
        return ("scene_with_plants", frame)

    def _create_scene_with_signs(self) -> Tuple[str, np.ndarray]:
        """Create scene dengan rectangular signs"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 140
        # Add rectangular signs
        cv2.rectangle(frame, (100, 150), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (100, 150), (250, 200), (0, 0, 0), 3)
        cv2.putText(frame, "STOP", (130, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return ("scene_with_signs", frame)

    def _create_scene_with_windows(self) -> Tuple[str, np.ndarray]:
        """Create scene dengan windows"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 160
        # Add windows
        for i in range(3):
            x = 150 + i * 150
            cv2.rectangle(frame, (x, 100), (x+100, 200), (200, 200, 255), -1)
            cv2.rectangle(frame, (x, 100), (x+100, 200), (0, 0, 0), 3)
        return ("scene_with_windows", frame)

    def _create_scene_with_rectangular_objects(self) -> Tuple[str, np.ndarray]:
        """Create scene dengan rectangular objects"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 130
        # Add various rectangular objects
        rectangles = [(100, 200, 150, 50), (300, 180, 120, 40), (450, 220, 100, 35)]
        for x, y, w, h in rectangles:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (180, 180, 180), -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
        return ("scene_with_rectangular_objects", frame)

    def _create_scene_with_text_on_building(self) -> Tuple[str, np.ndarray]:
        """Create scene dengan text on building"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 150
        # Add building dengan text
        cv2.rectangle(frame, (200, 100), (400, 300), (120, 120, 120), -1)
        cv2.putText(frame, "BUILDING", (210, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return ("scene_with_text_on_building", frame)

    def _create_low_contrast_scene(self) -> Tuple[str, np.ndarray]:
        """Create low contrast scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add low contrast elements
        cv2.rectangle(frame, (200, 200), (400, 250), (140, 140, 140), -1)
        return ("low_contrast_scene", frame)

    def _create_high_noise_scene(self) -> Tuple[str, np.ndarray]:
        """Create high noise scene"""
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        return ("high_noise_scene", frame)

    def _create_motion_blur_scene(self) -> Tuple[str, np.ndarray]:
        """Create motion blur scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 130
        # Add motion blur effect
        kernel = np.ones((1, 15), np.float32) / 15
        frame = cv2.filter2D(frame, -1, kernel)
        return ("motion_blur_scene", frame)

    def _create_extreme_lighting_scene(self) -> Tuple[str, np.ndarray]:
        """Create extreme lighting scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        # Add bright spot
        cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
        return ("extreme_lighting_scene", frame)

    def _create_partial_occlusion_scene(self) -> Tuple[str, np.ndarray]:
        """Create partial occlusion scene"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 140
        # Add occluding elements
        cv2.rectangle(frame, (200, 150), (400, 300), (80, 80, 80), -1)
        return ("partial_occlusion_scene", frame)

    def _is_realistic_plate_text(self, text: str) -> bool:
        """Check if text is realistic Indonesian plate"""
        if not text or len(text) < 5:
            return False

        # Basic Indonesian plate pattern check
        import re
        patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',
            r'^[A-Z]{2}\s*\d{2,4}\s*[A-Z]{2,3}$'
        ]

        cleaned = re.sub(r'[^A-Z0-9\s]', '', text.upper())
        return any(re.match(pattern, cleaned) for pattern in patterns)

    def _generate_zero_fp_report(self, results: Dict):
        """Generate comprehensive zero false positive report"""
        print("\n" + "=" * 60)
        print("🚫 ZERO FALSE POSITIVE TEST REPORT")
        print("=" * 60)

        # Empty Scene Tests
        empty_tests = results['empty_scene_tests']
        print(f"\n🔴 EMPTY SCENE TESTS:")
        print(f"   Scenarios Tested: {empty_tests['scenarios_tested']}")
        print(f"   False Positives: {empty_tests['false_positives']}")
        print(f"   Scene Analysis Accuracy: {empty_tests['scene_analysis_accuracy']:.1f}%")

        # Non-Plate Object Tests
        object_tests = results['non_plate_object_tests']
        print(f"\n🟠 NON-PLATE OBJECT TESTS:")
        print(f"   Objects Tested: {object_tests['objects_tested']}")
        print(f"   Incorrect Detections: {object_tests['incorrect_detections']}")

        # Edge Case Tests
        edge_tests = results['edge_case_tests']
        print(f"\n🟡 EDGE CASE TESTS:")
        print(f"   Cases Tested: {edge_tests['edge_cases_tested']}")
        print(f"   Failures: {edge_tests['failures']}")

        # Performance Impact
        perf = results['performance_metrics']
        print(f"\n⚡ PERFORMANCE IMPACT:")
        print(f"   Baseline: {perf['baseline_performance']['avg_time']:.3f}s")
        print(f"   Enhanced: {perf['enhanced_performance']['avg_time']:.3f}s")
        print(f"   Overhead: {perf['performance_overhead']:.1f}%")

        # Overall Assessment
        assessment = results['overall_assessment']
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print(f"   Total False Positives: {assessment['total_false_positives']}")
        print(f"   Total Scenarios: {assessment['total_scenarios']}")
        print(f"   False Positive Rate: {assessment['false_positive_rate']:.1f}%")
        print(f"   Grade: {assessment['grade']}")

        if assessment['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in assessment['recommendations']:
                print(f"   - {rec}")

        # Detection Manager Stats
        dm_stats = detection_manager.get_statistics()
        print(f"\n📊 DETECTION MANAGER STATS:")
        print(f"   Total Detections: {dm_stats['total_detections']}")
        print(f"   Filtered Non-plates: {dm_stats['filtered_non_plates']}")
        print(f"   Success Rate: {dm_stats['success_rate']:.1f}%")

        # Final verdict
        if assessment['false_positive_rate'] == 0:
            print(f"\n🎉 PERFECT SCORE - Zero false positives achieved!")
        elif assessment['false_positive_rate'] <= 5:
            print(f"\n✅ EXCELLENT - Very low false positive rate")
        elif assessment['false_positive_rate'] <= 15:
            print(f"\n⚠️  GOOD - Acceptable false positive rate")
        else:
            print(f"\n❌ NEEDS IMPROVEMENT - High false positive rate")

        # Save results
        with open('zero_fp_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: zero_fp_test_results.json")

def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    print("🚀 Starting Zero False Positive Test Suite")

    # Create tester
    tester = ZeroFalsePositiveTester()

    # Run comprehensive test
    results = tester.run_comprehensive_test()

    print("\n✅ Zero False Positive Test Suite Completed!")

if __name__ == "__main__":
    main()