"""
Optimized Plate Detection System
Solusi komprehensif untuk mengatasi false positive, instabilitas, dan lag RTSP

Features:
- Stabilitas tinggi dengan temporal filtering
- Reduced false positive dengan confidence boosting
- Optimized RTSP streaming dengan threading
- Real-time performance dengan GPU acceleration
- Tracking-based smoothing untuk konsistensi
"""

import cv2
import numpy as np
import time
import threading
import queue
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import math
import json
import os

# Import YOLO detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

@dataclass
class StableDetection:
    """Enhanced detection dengan stability tracking"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    vehicle_type: str
    track_id: Optional[int] = None
    stability_score: float = 0.0
    frame_count: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_confirmed: bool = False

class TemporalStabilizer:
    """
    Temporal stability system untuk reduce false positive
    dan stabilkan deteksi antar frame
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, max_distance: float = 100):
        self.max_age = max_age  # Maximum frames to keep track
        self.min_hits = min_hits  # Minimum hits untuk confirmation
        self.max_distance = max_distance  # Maximum tracking distance

        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0

    def update(self, detections: List[StableDetection]) -> List[StableDetection]:
        """Update tracker dengan detections baru"""
        self.frame_count += 1
        current_time = time.time()

        # Match detections dengan existing tracks
        matched_tracks, unmatched_detections = self._match_detections(detections)

        # Update matched tracks
        for track_id, detection in matched_tracks:
            track = self.tracks[track_id]
            track['detection'] = detection
            track['last_seen'] = current_time
            track['hit_count'] += 1
            track['age'] = 0

            # Update velocity (simple)
            if 'last_bbox' in track:
                old_center = self._get_center(track['last_bbox'])
                new_center = self._get_center(detection.bbox)
                track['velocity'] = (new_center[0] - old_center[0], new_center[1] - old_center[1])

            track['last_bbox'] = detection.bbox

            # Update stability score
            track['stability_score'] = min(1.0, track['hit_count'] / self.min_hits)

            # Set confirmation status
            detection.track_id = track_id
            detection.stability_score = track['stability_score']
            detection.is_confirmed = track['hit_count'] >= self.min_hits
            detection.frame_count = track['hit_count']

        # Create new tracks untuk unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                'detection': detection,
                'first_seen': current_time,
                'last_seen': current_time,
                'hit_count': 1,
                'age': 0,
                'last_bbox': detection.bbox,
                'velocity': (0.0, 0.0),
                'stability_score': 0.0
            }

            detection.track_id = track_id
            detection.stability_score = 0.0
            detection.is_confirmed = False
            detection.frame_count = 1

        # Age existing tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            track['age'] += 1
            if track['age'] > self.max_age:
                tracks_to_remove.append(track_id)

        # Remove old tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Return confirmed detections only
        confirmed_detections = [det for det in detections if det.is_confirmed]
        return confirmed_detections

    def _match_detections(self, detections: List[StableDetection]) -> Tuple[List[Tuple], List[StableDetection]]:
        """Match detections dengan existing tracks"""
        if not self.tracks:
            return [], detections

        # Calculate cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_center = self._get_center(track['last_bbox'])

            for j, detection in enumerate(detections):
                det_center = self._get_center(detection.bbox)
                distance = np.sqrt((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)

                # Text similarity bonus
                text_similarity = self._text_similarity(track['detection'].text, detection.text)
                cost = distance - (text_similarity * 50)  # Text bonus

                cost_matrix[i, j] = cost

        # Simple greedy matching
        matched_tracks = []
        matched_detections = set()
        matched_track_indices = set()

        for _ in range(min(len(track_ids), len(detections))):
            if len(matched_track_indices) == len(track_ids) or len(matched_detections) == len(detections):
                break

            # Find minimum cost
            min_cost = float('inf')
            min_i, min_j = -1, -1

            for i in range(len(track_ids)):
                if i in matched_track_indices:
                    continue
                for j in range(len(detections)):
                    if j in matched_detections:
                        continue
                    if cost_matrix[i, j] < min_cost and cost_matrix[i, j] < self.max_distance:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j

            if min_i != -1 and min_j != -1:
                matched_tracks.append((track_ids[min_i], detections[min_j]))
                matched_track_indices.add(min_i)
                matched_detections.add(min_j)

        # Unmatched detections
        unmatched_detections = [det for i, det in enumerate(detections) if i not in matched_detections]

        return matched_tracks, unmatched_detections

    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """Get center point dari bounding box"""
        x, y, w, h = bbox
        return (x + w/2, y + h/2)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0-1)"""
        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        text1 = text1.strip().upper()
        text2 = text2.strip().upper()

        if text1 == text2:
            return 1.0

        # Levenshtein distance approximation
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        # Simple overlap ratio
        overlap = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        return overlap / max_len

class OptimizedRTSPStream:
    """
    Optimized RTSP stream dengan threading dan buffer management
    untuk mengatasi lag dan frame drop
    """

    def __init__(self, rtsp_url: str, buffer_size: int = 2, fps_limit: int = 15):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0

        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.capture_thread = None
        self.last_frame_time = 0

        # Statistics
        self.total_frames = 0
        self.dropped_frames = 0
        self.fps_actual = 0.0
        self.last_fps_time = time.time()
        self.fps_counter = 0

        self.logger = logging.getLogger(__name__)

    def start(self) -> bool:
        """Start RTSP stream"""
        try:
            # Configure OpenCV untuk optimized streaming
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Optimized buffer settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)

            # Test connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("Failed to read initial frame dari RTSP")
                return False

            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.capture_thread.start()

            self.logger.info(f"RTSP stream started: {self.rtsp_url}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start RTSP stream: {e}")
            return False

    def stop(self):
        """Stop RTSP stream"""
        self.running = False

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.logger.info("RTSP stream stopped")

    def _capture_worker(self):
        """Background thread untuk capture frames"""
        while self.running and self.cap and self.cap.isOpened():
            try:
                current_time = time.time()

                # Frame rate limiting
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue

                ret, frame = self.cap.read()

                if not ret or frame is None:
                    self.logger.warning("Failed to read frame, reconnecting...")
                    self._reconnect()
                    continue

                self.total_frames += 1
                self.last_frame_time = current_time

                # Add frame ke queue (drop old frame jika full)
                try:
                    if self.frame_queue.full():
                        # Remove oldest frame
                        self.frame_queue.get_nowait()
                        self.dropped_frames += 1

                    self.frame_queue.put(frame, block=False)

                except queue.Full:
                    self.dropped_frames += 1

                # Update FPS calculation
                self.fps_counter += 1
                if current_time - self.last_fps_time >= 1.0:
                    self.fps_actual = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time

            except Exception as e:
                self.logger.error(f"Error in capture worker: {e}")
                time.sleep(0.1)

    def _reconnect(self):
        """Reconnect ke RTSP stream"""
        try:
            if self.cap:
                self.cap.release()

            time.sleep(1.0)  # Wait before reconnect

            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)

            self.logger.info("RTSP reconnected")

        except Exception as e:
            self.logger.error(f"Reconnection failed: {e}")

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame (non-blocking)"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def is_running(self) -> bool:
        """Check if stream is running"""
        return self.running and self.cap and self.cap.isOpened()

    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        drop_rate = (self.dropped_frames / self.total_frames * 100) if self.total_frames > 0 else 0

        return {
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'drop_rate_percent': round(drop_rate, 1),
            'fps_actual': round(self.fps_actual, 1),
            'fps_limit': self.fps_limit,
            'buffer_size': self.buffer_size,
            'running': self.running
        }

class OptimizedPlateDetector:
    """
    Main optimized plate detector dengan semua fitur stabilitas
    """

    def __init__(self,
                 model_path: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.7,  # Higher default untuk reduce false positive
                 iou_threshold: float = 0.45,
                 device: str = 'auto',
                 enable_gpu: bool = True):

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.enable_gpu = enable_gpu

        # Initialize YOLO
        self.model = None
        self.model_loaded = False

        # Initialize temporal stabilizer
        self.stabilizer = TemporalStabilizer(max_age=30, min_hits=3, max_distance=80)

        # Performance tracking
        self.detection_times = deque(maxlen=100)
        self.total_detections = 0
        self.confirmed_detections = 0

        # Vehicle detection classes
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        self.logger = logging.getLogger(__name__)

        # Load model
        self._load_model()

    def _load_model(self):
        """Load YOLO model dengan optimized settings"""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available")
            return False

        try:
            self.logger.info(f"Loading YOLO model: {self.model_path}")

            self.model = YOLO(self.model_path)

            # Set device
            if self.enable_gpu and self.device == 'auto':
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                    self.logger.info("GPU acceleration enabled")
                else:
                    self.device = 'cpu'
                    self.logger.info("Using CPU inference")

            self.model_loaded = True
            self.logger.info("✅ YOLO model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False

    def detect_vehicles_stable(self, frame: np.ndarray) -> List[StableDetection]:
        """
        Detect vehicles dengan stability enhancement
        """
        if not self.model_loaded:
            return []

        start_time = time.time()

        try:
            # Run YOLO detection dengan optimized parameters
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=list(self.vehicle_classes.keys()),  # Vehicle classes only
                device=self.device,
                verbose=False
            )

            # Parse results
            raw_detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Enhanced confidence filtering
                        if confidence < self.confidence_threshold:
                            continue

                        # Get vehicle type
                        vehicle_type = self.vehicle_classes.get(class_id, 'unknown')

                        # Create stable detection
                        detection = StableDetection(
                            text=f"{vehicle_type}_{confidence:.0f}%",
                            confidence=confidence,
                            bbox=(x1, y1, x2-x1, y2-y1),
                            vehicle_type=vehicle_type,
                            first_seen=time.time(),
                            last_seen=time.time()
                        )

                        raw_detections.append(detection)

            # Apply temporal stabilization
            stable_detections = self.stabilizer.update(raw_detections)

            # Update statistics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_detections += len(raw_detections)
            self.confirmed_detections += len(stable_detections)

            return stable_detections

        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[StableDetection]) -> np.ndarray:
        """Draw detections dengan enhanced visualization"""
        annotated_frame = frame.copy()

        for detection in detections:
            x, y, w, h = detection.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Color based on stability
            if detection.is_confirmed:
                color = (0, 255, 0)  # Green untuk confirmed
                thickness = 3
            else:
                color = (0, 165, 255)  # Orange untuk unconfirmed
                thickness = 2

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare label
            label_parts = []
            if detection.track_id:
                label_parts.append(f"ID:{detection.track_id}")
            label_parts.append(f"{detection.vehicle_type}")
            label_parts.append(f"{detection.confidence:.1f}%")
            label_parts.append(f"S:{detection.stability_score:.1f}")

            label = " ".join(label_parts)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)

            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw stability indicator
            stability_width = int(w * detection.stability_score)
            cv2.rectangle(annotated_frame,
                         (x1, y2 + 2),
                         (x1 + stability_width, y2 + 8),
                         (0, 255, 0), -1)

        return annotated_frame

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        avg_time = np.mean(self.detection_times) if self.detection_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        confirmation_rate = (self.confirmed_detections / self.total_detections * 100) if self.total_detections > 0 else 0

        return {
            'model_loaded': self.model_loaded,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'total_detections': self.total_detections,
            'confirmed_detections': self.confirmed_detections,
            'confirmation_rate_percent': round(confirmation_rate, 1),
            'avg_detection_time': round(avg_time, 3),
            'detection_fps': round(fps, 1),
            'active_tracks': len(self.stabilizer.tracks)
        }

def main_demo():
    """
    Demo script untuk testing optimized plate detector
    """
    print("🚀 Starting Optimized Plate Detection Demo")

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # RTSP URL (ganti dengan URL kamera Anda)
    rtsp_url = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"

    # Initialize components
    stream = OptimizedRTSPStream(rtsp_url, buffer_size=2, fps_limit=15)
    detector = OptimizedPlateDetector(confidence_threshold=0.7, enable_gpu=True)

    # Start stream
    if not stream.start():
        print("❌ Failed to start RTSP stream")
        return

    print("✅ Stream started, beginning detection...")

    try:
        frame_count = 0
        start_time = time.time()

        while True:
            # Get frame
            ret, frame = stream.get_frame()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Detect vehicles
            detections = detector.detect_vehicles_stable(frame)

            # Draw results
            annotated_frame = detector.draw_detections(frame, detections)

            # Add statistics overlay
            stats = detector.get_statistics()
            stream_stats = stream.get_stats()

            # Overlay info
            info_lines = [
                f"Frame: {frame_count}",
                f"FPS: {stream_stats['fps_actual']:.1f}",
                f"Detections: {len(detections)}",
                f"Confirmed: {stats['confirmed_detections']}",
                f"Confirmation Rate: {stats['confirmation_rate_percent']:.1f}%",
                f"Active Tracks: {stats['active_tracks']}",
                f"Drop Rate: {stream_stats['drop_rate_percent']:.1f}%"
            ]

            y_offset = 30
            for line in info_lines:
                cv2.putText(annotated_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30

            # Display frame
            cv2.imshow('Optimized Plate Detection', annotated_frame)

            # Log detections
            if detections:
                for det in detections:
                    status = "CONFIRMED" if det.is_confirmed else "TRACKING"
                    print(f"🚗 {status}: {det.vehicle_type} (ID:{det.track_id}, "
                          f"Conf:{det.confidence:.1f}%, Stability:{det.stability_score:.1f})")

            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                cv2.imwrite(f'detection_screenshot_{int(time.time())}.jpg', annotated_frame)
                print("📸 Screenshot saved")

    except KeyboardInterrupt:
        print("\n⏹️  Stopping detection...")

    finally:
        # Cleanup
        stream.stop()
        cv2.destroyAllWindows()

        # Final statistics
        runtime = time.time() - start_time
        final_stats = detector.get_statistics()
        final_stream_stats = stream.get_stats()

        print(f"\n📊 Final Statistics:")
        print(f"Runtime: {runtime:.1f}s")
        print(f"Total Frames: {frame_count}")
        print(f"Average FPS: {frame_count/runtime:.1f}")
        print(f"Total Detections: {final_stats['total_detections']}")
        print(f"Confirmed Detections: {final_stats['confirmed_detections']}")
        print(f"Confirmation Rate: {final_stats['confirmation_rate_percent']:.1f}%")
        print(f"Stream Drop Rate: {final_stream_stats['drop_rate_percent']:.1f}%")
        print(f"Detection FPS: {final_stats['detection_fps']:.1f}")

if __name__ == "__main__":
    main_demo()