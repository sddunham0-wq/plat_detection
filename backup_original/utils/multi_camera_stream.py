"""
Multi-Camera Stream Manager untuk Live CCTV Plate Detection
Handle multiple camera sources secara simultaneous dengan detection coordination
"""

import cv2
import threading
import time
import logging
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
import queue
import numpy as np

from .video_stream import VideoStream, WebcamStream, RTSPStream
from .camera_manager import CameraManager, CameraInfo
from .plate_detector import LicensePlateDetector, PlateDetection

@dataclass
class MultiFrameResult:
    """Result dari multi-camera frame processing"""
    camera_id: str
    camera_name: str
    frame: np.ndarray
    detections: List[PlateDetection]
    timestamp: float
    frame_id: int

class MultiCameraStream:
    """
    Manager untuk handle multiple camera streams dengan synchronized detection
    """
    
    def __init__(self, max_cameras: int = 4):
        """
        Initialize multi-camera stream manager
        
        Args:
            max_cameras: Maximum number of cameras to handle simultaneously
        """
        self.max_cameras = max_cameras
        self.logger = logging.getLogger(__name__)
        
        # Camera management
        self.camera_manager = CameraManager()
        self.active_streams: Dict[str, VideoStream] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}
        self.stream_configs: Dict[str, Dict] = {}
        
        # Detection management
        self.plate_detector = LicensePlateDetector()
        self.detection_threads: Dict[str, threading.Thread] = {}
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.result_queues: Dict[str, queue.Queue] = {}
        
        # Control
        self.running = False
        self.frame_callbacks: List[Callable[[MultiFrameResult], None]] = []
        self.detection_callbacks: List[Callable[[str, List[PlateDetection]], None]] = []
        
        # Statistics
        self.frame_counts: Dict[str, int] = {}
        self.start_time = time.time()
        
        self.logger.info("MultiCameraStream initialized")
    
    def discover_cameras(self) -> List[CameraInfo]:
        """
        Discover available cameras di system
        
        Returns:
            List[CameraInfo]: List of available cameras
        """
        self.logger.info("Discovering available cameras...")
        cameras = self.camera_manager.enumerate_cameras()
        available = self.camera_manager.get_available_cameras()
        
        self.logger.info(f"Found {len(available)} available cameras out of {len(cameras)} detected")
        return available
    
    def add_camera_source(self, camera_id: str, source: Any, config: Dict = None) -> bool:
        """
        Add camera source to multi-camera system
        
        Args:
            camera_id: Unique identifier for this camera
            source: Camera source (int untuk webcam, str untuk RTSP/file)
            config: Camera-specific configuration
            
        Returns:
            bool: True if successfully added
        """
        if len(self.active_streams) >= self.max_cameras:
            self.logger.warning(f"Maximum cameras ({self.max_cameras}) already reached")
            return False
        
        if camera_id in self.active_streams:
            self.logger.warning(f"Camera {camera_id} already exists")
            return False
        
        try:
            # Create appropriate video stream
            video_stream = self._create_video_stream(source, config or {})
            
            if video_stream:
                self.active_streams[camera_id] = video_stream
                self.stream_configs[camera_id] = config or {}
                self.frame_queues[camera_id] = queue.Queue(maxsize=30)
                self.result_queues[camera_id] = queue.Queue(maxsize=30)
                self.frame_counts[camera_id] = 0
                
                self.logger.info(f"Added camera source: {camera_id}")
                return True
            
        except Exception as e:
            self.logger.error(f"Error adding camera {camera_id}: {str(e)}")
        
        return False
    
    def add_laptop_camera(self, camera_id: str = "laptop", config: Dict = None) -> bool:
        """
        Convenience method untuk add laptop camera
        
        Args:
            camera_id: ID untuk laptop camera
            config: Camera configuration
            
        Returns:
            bool: True if successfully added
        """
        laptop_cam = self.camera_manager.get_laptop_camera()
        if laptop_cam:
            default_config = {
                'name': laptop_cam.name,
                'resolution': laptop_cam.resolution,
                'fps_limit': 10
            }
            if config:
                default_config.update(config)
            
            return self.add_camera_source(camera_id, laptop_cam.index, default_config)
        else:
            self.logger.warning("No laptop camera detected")
            return False
    
    def add_rtsp_camera(self, camera_id: str, rtsp_url: str, config: Dict = None) -> bool:
        """
        Convenience method untuk add RTSP camera
        
        Args:
            camera_id: ID untuk RTSP camera
            rtsp_url: RTSP URL
            config: Camera configuration
            
        Returns:
            bool: True if successfully added
        """
        default_config = {
            'name': f'RTSP Camera ({camera_id})',
            'fps_limit': 10
        }
        if config:
            default_config.update(config)
        
        return self.add_camera_source(camera_id, rtsp_url, default_config)
    
    def remove_camera_source(self, camera_id: str) -> bool:
        """
        Remove camera source from multi-camera system
        
        Args:
            camera_id: Camera ID to remove
            
        Returns:
            bool: True if successfully removed
        """
        if camera_id not in self.active_streams:
            self.logger.warning(f"Camera {camera_id} not found")
            return False
        
        try:
            # Stop stream jika running
            if self.running:
                self._stop_camera_processing(camera_id)
            
            # Cleanup resources
            stream = self.active_streams[camera_id]
            stream.stop()
            
            del self.active_streams[camera_id]
            del self.stream_configs[camera_id]
            if camera_id in self.frame_queues:
                del self.frame_queues[camera_id]
            if camera_id in self.result_queues:
                del self.result_queues[camera_id]
            if camera_id in self.frame_counts:
                del self.frame_counts[camera_id]
            
            self.logger.info(f"Removed camera source: {camera_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing camera {camera_id}: {str(e)}")
            return False
    
    def _create_video_stream(self, source: Any, config: Dict) -> Optional[VideoStream]:
        """Create appropriate video stream based on source type"""
        try:
            if isinstance(source, int):
                # Webcam
                resolution = config.get('resolution', (640, 480))
                return WebcamStream(
                    camera_index=source,
                    resolution=resolution,
                    buffer_size=config.get('buffer_size', 30)
                )
            elif isinstance(source, str):
                if source.startswith(('rtsp://', 'http://')):
                    # RTSP stream
                    return RTSPStream(
                        source,
                        buffer_size=config.get('buffer_size', 30)
                    )
                else:
                    # Video file
                    return VideoStream(
                        source,
                        buffer_size=config.get('buffer_size', 30)
                    )
            
        except Exception as e:
            self.logger.error(f"Error creating video stream: {str(e)}")
        
        return None
    
    def start(self) -> bool:
        """
        Start all camera streams dan detection processing
        
        Returns:
            bool: True if successfully started
        """
        if self.running:
            self.logger.warning("MultiCameraStream already running")
            return False
        
        if not self.active_streams:
            self.logger.error("No camera sources added")
            return False
        
        try:
            self.running = True
            self.start_time = time.time()
            
            # Start all video streams
            for camera_id, stream in self.active_streams.items():
                if not stream.start():
                    self.logger.error(f"Failed to start stream for camera {camera_id}")
                    self.stop()
                    return False
                
                # Start processing threads untuk setiap camera
                self._start_camera_processing(camera_id)
            
            self.logger.info(f"MultiCameraStream started with {len(self.active_streams)} cameras")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting MultiCameraStream: {str(e)}")
            self.stop()
            return False
    
    def _start_camera_processing(self, camera_id: str):
        """Start processing threads untuk specific camera"""
        # Frame capture thread
        capture_thread = threading.Thread(
            target=self._frame_capture_worker,
            args=(camera_id,),
            name=f"FrameCapture-{camera_id}",
            daemon=True
        )
        capture_thread.start()
        
        # Detection processing thread
        detection_thread = threading.Thread(
            target=self._detection_worker,
            args=(camera_id,),
            name=f"Detection-{camera_id}",
            daemon=True
        )
        detection_thread.start()
        
        self.detection_threads[camera_id] = detection_thread
    
    def _stop_camera_processing(self, camera_id: str):
        """Stop processing threads untuk specific camera"""
        # Threads akan stop automatically karena self.running = False
        if camera_id in self.detection_threads:
            thread = self.detection_threads[camera_id]
            if thread.is_alive():
                thread.join(timeout=2.0)
            del self.detection_threads[camera_id]
    
    def _frame_capture_worker(self, camera_id: str):
        """Worker thread untuk capture frames dari specific camera"""
        stream = self.active_streams[camera_id]
        frame_queue = self.frame_queues[camera_id]
        fps_limit = self.stream_configs[camera_id].get('fps_limit', 10)
        frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0
        
        last_frame_time = 0
        
        while self.running and stream.is_running():
            try:
                current_time = time.time()
                
                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.01)
                    continue
                
                ret, frame = stream.read()
                if ret and frame is not None:
                    # Add frame to queue untuk processing
                    try:
                        frame_queue.put_nowait((frame, current_time, self.frame_counts[camera_id]))
                        self.frame_counts[camera_id] += 1
                        last_frame_time = current_time
                    except queue.Full:
                        # Drop frame jika queue full
                        pass
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"Error in frame capture for {camera_id}: {str(e)}")
                time.sleep(0.1)
    
    def _detection_worker(self, camera_id: str):
        """Worker thread untuk detection processing pada specific camera"""
        frame_queue = self.frame_queues[camera_id]
        result_queue = self.result_queues[camera_id]
        camera_name = self.stream_configs[camera_id].get('name', camera_id)
        
        while self.running:
            try:
                # Get frame from queue
                frame_data = frame_queue.get(timeout=1.0)
                frame, timestamp, frame_id = frame_data
                
                # Run plate detection
                detections = self.plate_detector.detect_plates(frame)
                
                # Create result
                result = MultiFrameResult(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    frame=frame.copy(),
                    detections=detections,
                    timestamp=timestamp,
                    frame_id=frame_id
                )
                
                # Add to result queue
                try:
                    result_queue.put_nowait(result)
                except queue.Full:
                    # Drop result jika queue full
                    pass
                
                # Call callbacks
                self._call_frame_callbacks(result)
                if detections:
                    self._call_detection_callbacks(camera_id, detections)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in detection worker for {camera_id}: {str(e)}")
                time.sleep(0.1)
    
    def _call_frame_callbacks(self, result: MultiFrameResult):
        """Call all registered frame callbacks"""
        for callback in self.frame_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Error in frame callback: {str(e)}")
    
    def _call_detection_callbacks(self, camera_id: str, detections: List[PlateDetection]):
        """Call all registered detection callbacks"""
        for callback in self.detection_callbacks:
            try:
                callback(camera_id, detections)
            except Exception as e:
                self.logger.error(f"Error in detection callback: {str(e)}")
    
    def add_frame_callback(self, callback: Callable[[MultiFrameResult], None]):
        """Add callback untuk processed frames"""
        self.frame_callbacks.append(callback)
    
    def add_detection_callback(self, callback: Callable[[str, List[PlateDetection]], None]):
        """Add callback untuk detections"""
        self.detection_callbacks.append(callback)
    
    def get_latest_results(self) -> Dict[str, Optional[MultiFrameResult]]:
        """
        Get latest results dari semua cameras
        
        Returns:
            Dict[str, MultiFrameResult]: Latest result per camera
        """
        results = {}
        
        for camera_id, result_queue in self.result_queues.items():
            latest_result = None
            
            # Get all results, keep the latest
            while not result_queue.empty():
                try:
                    latest_result = result_queue.get_nowait()
                except queue.Empty:
                    break
            
            results[camera_id] = latest_result
        
        return results
    
    def get_camera_list(self) -> List[Dict]:
        """Get list of active cameras dengan info"""
        cameras = []
        
        for camera_id, stream in self.active_streams.items():
            config = self.stream_configs[camera_id]
            cameras.append({
                'id': camera_id,
                'name': config.get('name', camera_id),
                'running': stream.is_running(),
                'frame_count': self.frame_counts.get(camera_id, 0),
                'fps': stream.get_fps() if hasattr(stream, 'get_fps') else 0
            })
        
        return cameras
    
    def get_statistics(self) -> Dict:
        """Get multi-camera statistics"""
        elapsed_time = time.time() - self.start_time
        total_frames = sum(self.frame_counts.values())
        
        return {
            'running': self.running,
            'active_cameras': len(self.active_streams),
            'total_frames': total_frames,
            'elapsed_time': elapsed_time,
            'overall_fps': total_frames / elapsed_time if elapsed_time > 0 else 0,
            'per_camera_stats': {
                camera_id: {
                    'frame_count': count,
                    'fps': count / elapsed_time if elapsed_time > 0 else 0
                }
                for camera_id, count in self.frame_counts.items()
            }
        }
    
    def stop(self):
        """Stop all camera streams dan processing"""
        if not self.running:
            return
        
        self.logger.info("Stopping MultiCameraStream...")
        self.running = False
        
        # Stop all processing threads
        for camera_id in list(self.active_streams.keys()):
            self._stop_camera_processing(camera_id)
        
        # Stop all video streams
        for stream in self.active_streams.values():
            stream.stop()
        
        # Clear queues
        for queue_dict in [self.frame_queues, self.result_queues]:
            for q in queue_dict.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
        
        self.logger.info("MultiCameraStream stopped")
    
    def __del__(self):
        """Destructor untuk cleanup"""
        self.stop()

def test_multi_camera_stream():
    """Test function untuk multi-camera stream"""
    print("Testing Multi-Camera Stream...")
    
    multi_stream = MultiCameraStream()
    
    # Discover cameras
    cameras = multi_stream.discover_cameras()
    print(f"Discovered {len(cameras)} cameras")
    
    # Add laptop camera jika ada
    if multi_stream.add_laptop_camera():
        print("✅ Added laptop camera")
    else:
        print("❌ No laptop camera available")
    
    # Test start/stop
    if multi_stream.start():
        print("✅ Multi-camera stream started")
        
        # Run for a few seconds
        time.sleep(3)
        
        # Get statistics
        stats = multi_stream.get_statistics()
        print(f"📊 Statistics: {stats}")
        
        multi_stream.stop()
        print("✅ Multi-camera stream stopped")
    else:
        print("❌ Failed to start multi-camera stream")

if __name__ == "__main__":
    test_multi_camera_stream()