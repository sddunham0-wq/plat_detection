"""
Enhanced Hybrid Stream Manager - Ultra-Stable Integration
Integrasi Enhanced Hybrid Detector dengan Ultra-Stable Counting System
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import existing components
from utils.enhanced_hybrid_detector import EnhancedHybridDetector, EnhancedDetectionResult
from utils.stable_plate_counter import create_stable_plate_counter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from enhanced_detection_config import get_config, validate_config

@dataclass
class EnhancedStreamStats:
    """Enhanced statistics untuk hybrid detection system"""
    total_frames_processed: int = 0
    detection_method_stats: Dict[str, int] = None
    ultra_stable_stats: Dict = None
    performance_metrics: Dict = None

    def __post_init__(self):
        if self.detection_method_stats is None:
            self.detection_method_stats = {
                'plate_yolo': 0,
                'ocr_fallback': 0,
                'hybrid_validated': 0,
                'total_detections': 0
            }
        if self.ultra_stable_stats is None:
            self.ultra_stable_stats = {}
        if self.performance_metrics is None:
            self.performance_metrics = {
                'avg_processing_time': 0.0,
                'avg_fps': 0.0,
                'detection_efficiency': 0.0
            }

class EnhancedHybridStreamManager:
    """
    Enhanced Stream Manager dengan Dual YOLO + Ultra-Stable Counter
    """

    def __init__(self, source: str, config_preset: str = 'cctv_monitoring'):
        """
        Initialize Enhanced Hybrid Stream Manager

        Args:
            source: Video source (RTSP URL, camera index, or file path)
            config_preset: Configuration preset to use
        """
        self.source = source
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = get_config(config_preset)
        if not validate_config(self.config):
            raise ValueError(f"Invalid configuration for preset: {config_preset}")

        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.frame_count = 0

        # Initialize Enhanced Hybrid Detector
        self.hybrid_detector = None
        self._init_hybrid_detector()

        # Initialize Ultra-Stable Plate Counter
        self.stable_counter = None
        self._init_stable_counter()

        # Statistics and performance tracking
        self.stats = EnhancedStreamStats()
        self.processing_times = []
        self.detection_history = []

        # Threading
        self.processing_thread = None
        self.stats_lock = threading.Lock()

        self.logger.info(f"Enhanced Hybrid Stream Manager initialized with preset: {config_preset}")

    def _init_hybrid_detector(self):
        """Initialize Enhanced Hybrid Detector"""
        try:
            self.hybrid_detector = EnhancedHybridDetector(self.config)

            # Log enabled components
            components = []
            if self.config['vehicle_yolo']['enabled']:
                components.append("Vehicle YOLO")
            if self.config['license_plate_yolo']['enabled']:
                components.append("License Plate YOLO")
            if self.config['ocr']['enabled']:
                components.append("OCR Fallback")

            self.logger.info(f"✅ Hybrid Detector initialized with: {', '.join(components)}")

        except Exception as e:
            self.logger.error(f"Failed to initialize hybrid detector: {e}")
            self.hybrid_detector = None

    def _init_stable_counter(self):
        """Initialize Ultra-Stable Plate Counter"""
        try:
            # Ultra-Stable Counter configuration
            stable_config = {
                'text_similarity_threshold': 0.80,
                'spatial_distance_threshold': 120.0,
                'temporal_window': 5.0,
                'min_stability_score': 0.65,
                'min_confidence': 0.30,
                'confirmation_detections': 2
            }

            self.stable_counter = create_stable_plate_counter(stable_config)
            self.logger.info("✅ Ultra-Stable Counter initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize stable counter: {e}")
            self.stable_counter = None

    def start_stream(self):
        """Start video stream processing"""
        if self.is_running:
            self.logger.warning("Stream is already running")
            return False

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open video source: {self.source}")
            return False

        # Configure capture properties
        self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)  # Minimize buffer lag

        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        self.logger.info("✅ Enhanced hybrid stream started")
        return True

    def stop_stream(self):
        """Stop video stream processing"""
        self.is_running = False

        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.logger.info("🛑 Enhanced hybrid stream stopped")

    def _processing_loop(self):
        """Main processing loop"""
        skip_counter = 0
        skip_factor = self.config['performance'].get('skip_factor', 2)
        enable_skipping = self.config['performance'].get('frame_skipping', True)

        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame, attempting to reconnect...")
                    self._reconnect_stream()
                    continue

                self.frame_count += 1

                # Frame skipping for performance
                if enable_skipping:
                    skip_counter += 1
                    if skip_counter < skip_factor:
                        continue
                    skip_counter = 0

                # Process frame
                start_time = time.time()
                self._process_frame(frame)
                processing_time = time.time() - start_time

                # Update performance metrics
                self._update_performance_metrics(processing_time)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)  # Brief pause on error

    def _process_frame(self, frame: np.ndarray):
        """Process single frame dengan enhanced hybrid detection"""
        try:
            # Enhanced Hybrid Detection
            enhanced_detections = []
            if self.hybrid_detector and self.hybrid_detector.is_enabled():
                enhanced_detections = self.hybrid_detector.detect_license_plates(frame)

            # Update detection method statistics
            with self.stats_lock:
                for detection in enhanced_detections:
                    method = detection.detection_method
                    if method in self.stats.detection_method_stats:
                        self.stats.detection_method_stats[method] += 1
                    self.stats.detection_method_stats['total_detections'] += 1

            # Ultra-Stable Counting Integration
            if self.stable_counter:
                for detection in enhanced_detections:
                    # Convert to stable counter format
                    track_id = self.stable_counter.add_detection(
                        text=detection.text,
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        vehicle_type=detection.vehicle_type if detection.vehicle_type != 'unknown' else 'car'
                    )

                    if track_id:
                        # Update detection dengan track information
                        detection.track_id = track_id
                        self.detection_history.append({
                            'timestamp': time.time(),
                            'text': detection.text,
                            'confidence': detection.confidence,
                            'method': detection.detection_method,
                            'track_id': track_id,
                            'is_valid': detection.is_valid_format
                        })

            # Keep detection history manageable
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-500:]

        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")

    def _update_performance_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.processing_times.append(processing_time)

        # Keep only last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]

        with self.stats_lock:
            self.stats.total_frames_processed += 1

            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                self.stats.performance_metrics['avg_processing_time'] = avg_time
                self.stats.performance_metrics['avg_fps'] = 1.0 / avg_time if avg_time > 0 else 0

                # Calculate detection efficiency
                total_detections = self.stats.detection_method_stats['total_detections']
                efficiency = (total_detections / self.stats.total_frames_processed) * 100 if self.stats.total_frames_processed > 0 else 0
                self.stats.performance_metrics['detection_efficiency'] = efficiency

    def _reconnect_stream(self):
        """Attempt to reconnect to video stream"""
        max_attempts = 3
        for attempt in range(max_attempts):
            self.logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")

            if self.cap:
                self.cap.release()

            time.sleep(2)  # Wait before reconnect

            self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFER_SIZE, 1)
                self.logger.info("✅ Stream reconnected successfully")
                return True

        self.logger.error("❌ Failed to reconnect stream after maximum attempts")
        self.is_running = False
        return False

    def get_frame_with_annotations(self) -> Optional[np.ndarray]:
        """Get current frame dengan annotations"""
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Draw enhanced detection annotations
        if self.hybrid_detector:
            # Get current detections untuk frame ini
            detections = self.hybrid_detector.detect_license_plates(frame)
            annotated_frame = self.hybrid_detector.draw_detections(frame, detections)

            # Add performance info
            self._draw_performance_overlay(annotated_frame)

            return annotated_frame

        return frame

    def _draw_performance_overlay(self, frame: np.ndarray):
        """Draw performance overlay on frame"""
        try:
            with self.stats_lock:
                perf = self.stats.performance_metrics
                detection_stats = self.stats.detection_method_stats

                # Ultra-stable counter stats
                stable_stats = {}
                if self.stable_counter:
                    stable_stats = self.stable_counter.get_stable_counts()

            # Performance info
            fps = perf.get('avg_fps', 0)
            efficiency = perf.get('detection_efficiency', 0)

            # Detection method breakdown
            total_detections = detection_stats.get('total_detections', 0)
            plate_yolo_count = detection_stats.get('plate_yolo', 0)
            ocr_count = detection_stats.get('ocr_fallback', 0)

            # Ultra-stable counter info
            unique_plates = stable_stats.get('total_unique_session', 0)
            current_visible = stable_stats.get('current_visible_plates', 0)

            # Draw overlay
            overlay_y = 30
            cv2.putText(frame, f"FPS: {fps:.1f} | Efficiency: {efficiency:.1f}%",
                       (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            overlay_y += 25
            cv2.putText(frame, f"Total Detections: {total_detections} (YOLO: {plate_yolo_count}, OCR: {ocr_count})",
                       (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            overlay_y += 25
            cv2.putText(frame, f"UNIQUE PLATES: {unique_plates} | Current: {current_visible}",
                       (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        except Exception as e:
            self.logger.debug(f"Error drawing overlay: {e}")

    def get_comprehensive_statistics(self) -> Dict:
        """Get comprehensive statistics dari semua components"""
        with self.stats_lock:
            base_stats = {
                'enhanced_stream_manager': {
                    'total_frames_processed': self.stats.total_frames_processed,
                    'detection_methods': self.stats.detection_method_stats.copy(),
                    'performance': self.stats.performance_metrics.copy(),
                    'recent_detections': len(self.detection_history)
                }
            }

        # Add hybrid detector statistics
        if self.hybrid_detector:
            hybrid_stats = self.hybrid_detector.get_statistics()
            base_stats.update(hybrid_stats)

        # Add ultra-stable counter statistics
        if self.stable_counter:
            stable_stats = self.stable_counter.get_comprehensive_stats()
            base_stats['ultra_stable_counter'] = stable_stats

        return base_stats

    def get_unique_plate_count(self) -> int:
        """Get current unique plate count (main metric)"""
        if self.stable_counter:
            counts = self.stable_counter.get_stable_counts()
            return counts.get('total_unique_session', 0)
        return 0

    def get_current_visible_plates(self) -> int:
        """Get currently visible plates"""
        if self.stable_counter:
            counts = self.stable_counter.get_stable_counts()
            return counts.get('current_visible_plates', 0)
        return 0

    def reset_statistics(self):
        """Reset all statistics"""
        with self.stats_lock:
            self.stats = EnhancedStreamStats()
            self.processing_times = []
            self.detection_history = []

        if self.hybrid_detector:
            self.hybrid_detector.reset_statistics()

        if self.stable_counter:
            self.stable_counter.reset_statistics()

        self.logger.info("📊 All statistics reset")

    def get_detection_history(self, limit: int = 50) -> List[Dict]:
        """Get recent detection history"""
        return self.detection_history[-limit:] if self.detection_history else []

    def is_system_ready(self) -> bool:
        """Check if all systems are ready"""
        return (
            self.is_running and
            self.hybrid_detector and
            self.hybrid_detector.is_enabled() and
            self.stable_counter is not None
        )

    def get_system_status(self) -> Dict:
        """Get detailed system status"""
        return {
            'stream_active': self.is_running,
            'video_source': self.source,
            'frame_count': self.frame_count,
            'hybrid_detector_ready': self.hybrid_detector and self.hybrid_detector.is_enabled(),
            'stable_counter_ready': self.stable_counter is not None,
            'components_status': {
                'vehicle_yolo': self.config['vehicle_yolo']['enabled'],
                'license_plate_yolo': self.config['license_plate_yolo']['enabled'],
                'enhanced_hybrid': self.config['enhanced_hybrid']['enabled'],
                'ocr_fallback': self.config['ocr']['enabled']
            },
            'detection_priority': self.config['enhanced_hybrid'].get('detection_priority', 'unknown'),
            'system_ready': self.is_system_ready()
        }

def create_enhanced_stream_manager(source: str,
                                 preset: str = 'cctv_monitoring') -> EnhancedHybridStreamManager:
    """
    Factory function untuk create Enhanced Hybrid Stream Manager

    Args:
        source: Video source (RTSP, camera index, file)
        preset: Configuration preset

    Returns:
        EnhancedHybridStreamManager instance
    """
    return EnhancedHybridStreamManager(source, preset)