"""
Stream Manager untuk Headless Video Processing
Menghandle video stream dan detection untuk web browser
"""

import cv2
import base64
import threading
import time
import logging
import numpy as np
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from utils.video_stream import VideoStream, RTSPStream, WebcamStream
from utils.plate_detector import LicensePlateDetector, PlateDetection
from utils.robust_plate_detector import RobustPlateDetector
from utils.yolo_plate_detector import YOLOPlateDetector
from utils.hybrid_plate_detector import HybridPlateDetector
from utils.yolo_detector import YOLOObjectDetector, create_yolo_detector
from utils.tracking_manager import TrackingManager
from database import PlateDatabase
from config import TrackingConfig

@dataclass
class StreamFrame:
    """Frame data untuk web streaming"""
    image_base64: str
    timestamp: float
    frame_id: int
    detections: list
    object_detections: list
    fps: float
    processing_time: float

class HeadlessStreamManager:
    """
    Manager untuk headless video streaming ke browser
    """
    
    def __init__(self, source: str, database: PlateDatabase = None, enable_yolo: bool = True, enable_tracking: bool = True):
        """
        Initialize stream manager
        
        Args:
            source: Video source (RTSP URL, webcam index, file)
            database: Database instance untuk save results
            enable_yolo: Enable YOLOv8 object detection
            enable_tracking: Enable object tracking system
        """
        self.source = source
        self.database = database or PlateDatabase()
        
        # Components
        self.video_stream = None
        self.plate_detector = HybridPlateDetector(streaming_mode=True)  # ✅ HYBRID DETECTION (YOLO+OpenCV) UNTUK AKURASI MAKSIMAL
        self.yolo_detector = None
        self.yolo_enabled = enable_yolo
        self.tracking_manager = None
        self.tracking_enabled = enable_tracking and TrackingConfig.ENABLE_TRACKING
        
        # Threading
        self.running = False
        self.stream_thread = None
        self.lock = threading.Lock()
        
        # Current frame data
        self.current_frame = None
        self.frame_callbacks = []
        self.detection_callbacks = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,  # Total plat nomor yang berhasil dibaca (akumulatif)
            'total_detection_events': 0,  # Total detection events (akumulatif)
            'current_objects': 0,  # Objek saat ini di frame (current)
            'current_vehicles': 0,  # Kendaraan saat ini di frame (current)
            'fps': 0.0,
            'avg_processing_time': 0.0,
            'last_detection_time': None,
            'start_time': time.time(),
            'yolo_enabled': False,
            'tracking_enabled': False
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking manager
        if self.tracking_enabled:
            self.logger.info("Initializing tracking system...")
            tracking_config = {
                'max_disappeared': TrackingConfig.MAX_DISAPPEARED_FRAMES,
                'max_distance': TrackingConfig.MAX_TRACKING_DISTANCE,
                'min_hits': TrackingConfig.MIN_HITS_FOR_CONFIRMATION,
                'iou_threshold': TrackingConfig.IOU_THRESHOLD
            }
            
            self.tracking_manager = TrackingManager(
                tracking_config=tracking_config,
                plate_confirmation_threshold=TrackingConfig.PLATE_CONFIRMATION_THRESHOLD,
                max_plate_age=TrackingConfig.MAX_PLATE_AGE
            )
            self.stats['tracking_enabled'] = True
            self.logger.info("✅ Tracking system initialized")
        
        # Start YOLOv8 loading in background untuk faster startup
        if self.yolo_enabled:
            self.logger.info("Starting YOLOv8 background loading...")
            self._start_yolo_background_loading()
        
        self.logger.info("HeadlessStreamManager initialized")
    
    def _start_yolo_background_loading(self):
        """Start loading YOLOv8 in background thread"""
        def load_yolo():
            try:
                self.logger.info("🔄 Loading YOLOv8 model in background...")
                self.yolo_detector = create_yolo_detector('yolov8n.pt', 0.5)
                if self.yolo_detector:
                    # Auto-enable YOLOv8 setelah loading selesai
                    self.yolo_detector.enable()
                    self.stats['yolo_enabled'] = True
                    self.logger.info("✅ YOLOv8 loaded and enabled! Object detection active.")
                else:
                    self.logger.warning("❌ Failed to load YOLOv8")
            except Exception as e:
                self.logger.error(f"Error loading YOLOv8: {str(e)}")
        
        # Start loading in daemon thread
        loading_thread = threading.Thread(target=load_yolo, daemon=True)
        loading_thread.start()
    
    def add_frame_callback(self, callback: Callable[[StreamFrame], None]):
        """Add callback untuk new frames"""
        self.frame_callbacks.append(callback)
    
    def add_detection_callback(self, callback: Callable[[list], None]):
        """Add callback untuk detections"""
        self.detection_callbacks.append(callback)
    
    def enable_yolo(self):
        """Enable YOLOv8 object detection dengan lazy loading"""
        if not self.yolo_detector:
            self.logger.info("Loading YOLOv8 model (this may take a moment)...")
            self.yolo_detector = create_yolo_detector('yolov8n.pt', 0.5)
        
        if self.yolo_detector:
            self.yolo_detector.enable()
            self.stats['yolo_enabled'] = True
            self.logger.info("YOLOv8 object detection enabled")
            return True
        else:
            self.logger.error("Failed to enable YOLOv8 - detector not available")
            return False
    
    def disable_yolo(self):
        """Disable YOLOv8 object detection"""
        if self.yolo_detector:
            self.yolo_detector.disable()
        self.stats['yolo_enabled'] = False
        self.logger.info("YOLOv8 object detection disabled")
    
    def is_yolo_enabled(self) -> bool:
        """Check if YOLOv8 is enabled"""
        return self.yolo_detector and self.yolo_detector.is_enabled()
    
    def enable_sequential_detection(self, grid_zones=(3, 3), cycle_time=2.0):
        """Enable sequential detection mode"""
        if self.yolo_detector and hasattr(self.yolo_detector, 'enable_sequential_mode'):
            self.yolo_detector.enable_sequential_mode(grid_zones, cycle_time)
            self.logger.info(f"Sequential detection enabled: {grid_zones} grid, {cycle_time}s cycle")
            return True
        return False
    
    def disable_sequential_detection(self):
        """Disable sequential detection mode"""
        if self.yolo_detector and hasattr(self.yolo_detector, 'disable_sequential_mode'):
            self.yolo_detector.disable_sequential_mode()
            self.logger.info("Sequential detection disabled")
            return True
        return False
    
    def get_sequential_info(self):
        """Get sequential detection info"""
        if self.yolo_detector and hasattr(self.yolo_detector, 'get_current_zone_info'):
            return self.yolo_detector.get_current_zone_info()
        return {'sequential_mode': False}
    
    def start(self) -> bool:
        """Start streaming"""
        try:
            # Initialize video stream
            if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                self.video_stream = RTSPStream(self.source, buffer_size=10)
            elif isinstance(self.source, int) or self.source.isdigit():
                self.video_stream = WebcamStream(int(self.source), buffer_size=10)
            else:
                self.video_stream = VideoStream(self.source, buffer_size=10)
            
            # Start video stream
            if not self.video_stream.start():
                self.logger.error("Failed to start video stream")
                return False
            
            # Start processing thread
            self.running = True
            self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.stream_thread.start()
            
            self.logger.info("HeadlessStreamManager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting stream manager: {str(e)}")
            return False
    
    def stop(self):
        """Stop streaming"""
        self.logger.info("Stopping HeadlessStreamManager...")
        
        self.running = False
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)
        
        if self.video_stream:
            self.video_stream.stop()
        
        self.logger.info("HeadlessStreamManager stopped")
    
    def _stream_worker(self):
        """Main streaming worker thread"""
        frame_count = 0
        start_time = time.time()
        processing_times = []
        
        while self.running and self.video_stream.is_running():
            try:
                # Get latest frame
                ret, frame = self.video_stream.get_latest_frame()
                
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                process_start = time.time()
                
                # Detect objects dengan YOLOv8 (jika enabled)
                object_detections = []
                if self.is_yolo_enabled():
                    # Use sequential detection if enabled, otherwise normal detection
                    if hasattr(self.yolo_detector, 'sequential_mode') and self.yolo_detector.sequential_mode:
                        object_detections = self.yolo_detector.detect_objects_sequential(frame, vehicles_only=False)
                    else:
                        object_detections = self.yolo_detector.detect_objects(frame, vehicles_only=False)
                
                # Detect plates
                plate_detections = self.plate_detector.detect_plates(frame)
                
                # Process dengan tracking system jika enabled
                tracked_objects = []
                tracked_plates = []
                
                if self.tracking_enabled and self.tracking_manager:
                    tracked_objects, tracked_plates = self.tracking_manager.process_frame(
                        object_detections, plate_detections
                    )
                
                # Prepare annotated frame
                annotated_frame = frame.copy()
                
                # Draw zone overlay first (if sequential mode)
                if self.is_yolo_enabled() and hasattr(self.yolo_detector, 'sequential_mode'):
                    annotated_frame = self.yolo_detector.draw_zone_overlay(annotated_frame)
                
                # Draw tracking results jika enabled
                if self.tracking_enabled and tracked_objects:
                    annotated_frame = self.tracking_manager.draw_tracking_results(
                        annotated_frame, tracked_objects, tracked_plates,
                        show_ids=TrackingConfig.SHOW_TRACKING_IDS
                    )
                else:
                    # Fallback ke regular drawing
                    # Draw object detections (sebagai background)
                    if object_detections:
                        annotated_frame = self.yolo_detector.draw_detections(
                            annotated_frame, object_detections, show_confidence=True
                        )
                    
                    # Draw plate detections on top
                    annotated_frame = self.plate_detector.draw_detections(
                        annotated_frame, plate_detections, show_roi=True
                    )
                
                # Calculate processing time
                processing_time = time.time() - process_start
                processing_times.append(processing_time)
                
                # Keep only last 30 processing times for average
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Update statistics
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                avg_processing_time = sum(processing_times) / len(processing_times)
                
                # Count vehicles in object detections
                vehicle_count = sum(1 for obj in object_detections if obj.is_vehicle)
                
                with self.lock:
                    # Update detection events counter (akumulatif untuk historical tracking)
                    if len(object_detections) > 0:
                        self.stats['total_detection_events'] += 1
                    
                    # Use tracking results untuk statistics jika available
                    current_vehicles = len([obj for obj in tracked_objects if obj.is_vehicle]) if tracked_objects else vehicle_count
                    confirmed_plates = len([plate for plate in tracked_plates if plate.confirmed]) if tracked_plates else 0
                    
                    self.stats.update({
                        'total_frames': frame_count,
                        'total_detections': self.stats['total_detections'] + len(plate_detections),  # Plat nomor (akumulatif)
                        'current_objects': len(tracked_objects) if tracked_objects else len(object_detections),
                        'current_vehicles': current_vehicles,
                        'confirmed_plates': confirmed_plates,
                        'fps': round(current_fps, 1),
                        'avg_processing_time': round(avg_processing_time, 3),
                        'last_detection_time': time.time() if plate_detections else self.stats['last_detection_time']
                    })
                
                # Convert frame to base64
                frame_base64 = self._frame_to_base64(annotated_frame)
                
                # Create stream frame
                stream_frame = StreamFrame(
                    image_base64=frame_base64,
                    timestamp=time.time(),
                    frame_id=frame_count,
                    detections=[{
                        'text': det.text,
                        'confidence': det.confidence,
                        'bbox': det.bbox
                    } for det in plate_detections],
                    object_detections=[{
                        'class_name': obj.class_name,
                        'confidence': obj.confidence,
                        'bbox': obj.bbox,
                        'is_vehicle': obj.is_vehicle
                    } for obj in object_detections],
                    fps=current_fps,
                    processing_time=processing_time
                )
                
                # Update current frame
                with self.lock:
                    self.current_frame = stream_frame
                
                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(stream_frame)
                    except Exception as e:
                        self.logger.error(f"Frame callback error: {str(e)}")
                
                # Handle detections - prioritize confirmed tracked plates
                final_detections = []
                
                if self.tracking_enabled and tracked_plates:
                    # Use confirmed tracked plates untuk database save
                    confirmed_tracked_plates = [plate for plate in tracked_plates if plate.confirmed]
                    for tracked_plate in confirmed_tracked_plates:
                        # Create PlateDetection object dari TrackedPlate
                        detection = PlateDetection(
                            text=tracked_plate.text,
                            confidence=tracked_plate.confidence,
                            bbox=tracked_plate.bbox,
                            processed_image=None,  # Could be enhanced later
                            timestamp=time.time()
                        )
                        final_detections.append(detection)
                else:
                    # Fallback ke regular plate detections
                    final_detections = plate_detections
                
                if final_detections:
                    # Save to database
                    for detection in final_detections:
                        try:
                            self.database.save_detection(
                                detection,
                                source_info=str(self.source),
                                save_image=True
                            )
                        except Exception as e:
                            self.logger.error(f"Database save error: {str(e)}")
                    
                    # Call detection callbacks
                    for callback in self.detection_callbacks:
                        try:
                            callback(final_detections)
                        except Exception as e:
                            self.logger.error(f"Detection callback error: {str(e)}")
                    
                    # Log detection dengan tracking info
                    for det in final_detections:
                        if self.tracking_enabled and hasattr(det, 'vehicle_id'):
                            vehicle_info = f" -> Vehicle {det.vehicle_id}" if det.vehicle_id else ""
                            self.logger.info(f"🚗 TRACKED PLATE: {det.text} (confidence: {det.confidence:.1f}%){vehicle_info}")
                        else:
                            self.logger.info(f"🚗 DETECTED: {det.text} (confidence: {det.confidence:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error in stream worker: {str(e)}")
                time.sleep(0.1)
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame ke base64 string"""
        try:
            # Resize frame untuk web (optional)
            height, width = frame.shape[:2]
            if width > 1280:  # Max width untuk web
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode ke JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Convert ke base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return frame_base64
            
        except Exception as e:
            self.logger.error(f"Error converting frame to base64: {str(e)}")
            return ""
    
    def get_current_frame(self) -> Optional[StreamFrame]:
        """Get current frame data"""
        with self.lock:
            return self.current_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        with self.lock:
            stats = self.stats.copy()
        
        # Add detector statistics
        detector_stats = self.plate_detector.get_statistics()
        stats.update(detector_stats)
        
        # Add YOLOv8 statistics jika available
        if self.yolo_detector:
            yolo_stats = self.yolo_detector.get_statistics()
            for key, value in yolo_stats.items():
                stats[f'yolo_{key}'] = value
        
        # Add tracking statistics jika available
        if self.tracking_manager:
            tracking_stats = self.tracking_manager.get_statistics()
            for key, value in tracking_stats.items():
                stats[f'tracking_{key}'] = value
        
        # Add uptime
        stats['uptime'] = round(time.time() - stats['start_time'], 1)
        
        return stats
    
    def is_running(self) -> bool:
        """Check if streaming is running"""
        return self.running and self.video_stream and self.video_stream.is_running()

def test_stream_manager():
    """Test function untuk stream manager"""
    print("Testing HeadlessStreamManager...")
    
    # Test dengan webcam atau RTSP
    source = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"  # NEW working camera URL ✅
    
    manager = HeadlessStreamManager(source)
    
    def frame_callback(stream_frame):
        print(f"Frame {stream_frame.frame_id}: {len(stream_frame.detections)} detections, FPS: {stream_frame.fps:.1f}")
    
    def detection_callback(detections):
        for det in detections:
            print(f"  - Detected: {det.text} ({det.confidence:.1f}%)")
    
    manager.add_frame_callback(frame_callback)
    manager.add_detection_callback(detection_callback)
    
    if manager.start():
        print("Stream manager started, running for 10 seconds...")
        time.sleep(10)
        
        stats = manager.get_statistics()
        print(f"Final statistics: {stats}")
        
        manager.stop()
        print("Test completed")
    else:
        print("Failed to start stream manager")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_stream_manager()