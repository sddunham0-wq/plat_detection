"""
Video Stream Handler untuk Live CCTV
Menangani berbagai sumber video: RTSP, webcam, file video
"""

import cv2
import time
import threading
import queue
import logging
from typing import Optional, Tuple, Union, Dict

class VideoStream:
    """
    Class untuk handle berbagai sumber video stream
    - RTSP IP Camera
    - USB Webcam  
    - File Video
    """
    
    def __init__(self, source: Union[str, int], buffer_size: int = 30):
        """
        Initialize video stream
        
        Args:
            source: Video source (RTSP URL, webcam index, atau file path)
            buffer_size: Maksimal frame di buffer
        """
        self.source = source
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.fps = 30
        self.frame_count = 0
        self.error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """
        Mulai video stream
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            # Enhanced video capture initialization untuk RTSP streams
            # Use FFMPEG backend for RTSP streams
            if isinstance(self.source, str) and self.source.startswith('rtsp://'):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.source)

            # Enhanced buffer settings for RTSP (minimal latency)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Force frame size untuk Dahua/Hikvision cameras
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            # Additional optimizations for better streaming performance
            self.cap.set(cv2.CAP_PROP_FPS, 25)  # Match camera FPS
            
            # Cek apakah berhasil terbuka
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {self.source}")
                return False
                
            # Set resolution jika diperlukan
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Get actual FPS
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # Default FPS
                
            self.logger.info(f"Video stream started - Source: {self.source}, FPS: {self.fps}")
            
            # Start thread untuk membaca frame
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting video stream: {str(e)}")
            return False
    
    def _capture_frames(self):
        """
        Thread function untuk capture frame secara kontinyu
        """
        while self.running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.error_count += 1
                    if self.error_count > 5:
                        self.logger.error("Too many read errors, stopping stream")
                        break
                    time.sleep(0.1)
                    continue
                    
                self.error_count = 0  # Reset error count
                self.frame_count += 1
                
                # Optimized queue management for low latency streaming
                # Clear multiple old frames if queue is getting full
                if self.frame_queue.qsize() >= self.buffer_size - 1:
                    # Remove all old frames to keep only the latest
                    cleared_count = 0
                    while not self.frame_queue.empty() and cleared_count < self.buffer_size - 1:
                        try:
                            self.frame_queue.get_nowait()
                            cleared_count += 1
                        except queue.Empty:
                            break
                
                # Masukkan frame baru ke queue
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip jika masih penuh
                    
            except Exception as e:
                self.logger.error(f"Error capturing frame: {str(e)}")
                time.sleep(0.1)
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Baca frame terbaru dari stream
        
        Returns:
            Tuple[bool, cv2.Mat]: (success, frame)
        """
        try:
            if not self.running:
                return False, None
                
            # Ambil frame dari queue
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
            
        except queue.Empty:
            return False, None
        except Exception as e:
            self.logger.error(f"Error reading frame: {str(e)}")
            return False, None
    
    def get_latest_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Ambil frame terbaru dan buang frame lama (untuk real-time processing)
        
        Returns:
            Tuple[bool, cv2.Mat]: (success, latest_frame)
        """
        latest_frame = None
        found = False
        
        # Ambil semua frame yang ada, simpan yang terakhir
        while not self.frame_queue.empty():
            try:
                latest_frame = self.frame_queue.get_nowait()
                found = True
            except queue.Empty:
                break
        
        return found, latest_frame
    
    def is_running(self) -> bool:
        """Check apakah stream masih berjalan"""
        return self.running and self.thread is not None and self.thread.is_alive()
    
    def get_fps(self) -> float:
        """Get FPS dari video stream"""
        return self.fps
    
    def get_frame_count(self) -> int:
        """Get jumlah frame yang sudah dibaca"""
        return self.frame_count
    
    def get_queue_size(self) -> int:
        """Get ukuran queue saat ini"""
        return self.frame_queue.qsize()
    
    def stop(self):
        """Stop video stream dan cleanup resources"""
        self.logger.info("Stopping video stream...")
        
        self.running = False
        
        # Wait for thread to finish
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        # Release video capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("Video stream stopped")
    
    def __del__(self):
        """Destructor untuk cleanup"""
        self.stop()

class RTSPStream(VideoStream):
    """
    Specialized class untuk RTSP streams dengan reconnection handling
    """
    
    def __init__(self, rtsp_url: str, username: str = None, password: str = None, **kwargs):
        """
        Initialize RTSP stream
        
        Args:
            rtsp_url: RTSP URL
            username: Username untuk autentikasi (optional)
            password: Password untuk autentikasi (optional)
        """
        if username and password:
            # Insert credentials ke URL
            if "://" in rtsp_url:
                protocol, rest = rtsp_url.split("://", 1)
                rtsp_url = f"{protocol}://{username}:{password}@{rest}"
        
        super().__init__(rtsp_url, **kwargs)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
    
    def start(self) -> bool:
        """Start RTSP stream dengan retry logic"""
        for attempt in range(self.max_reconnect_attempts):
            if super().start():
                self.reconnect_attempts = 0
                return True
            
            if attempt < self.max_reconnect_attempts - 1:
                self.logger.warning(f"RTSP connection failed, retrying in {self.reconnect_delay}s... (attempt {attempt + 1})")
                time.sleep(self.reconnect_delay)
        
        self.logger.error("Failed to connect to RTSP stream after multiple attempts")
        return False

class WebcamStream(VideoStream):
    """
    Enhanced webcam stream class dengan better camera support dan info detection
    """
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = None, 
                 camera_name: str = None, backend: int = None, **kwargs):
        """
        Initialize webcam stream
        
        Args:
            camera_index: Index kamera (biasanya 0 untuk kamera utama)
            resolution: Resolusi yang diinginkan (width, height)
            camera_name: Human-readable name untuk camera
            backend: Specific backend to use (cv2.CAP_*)
        """
        super().__init__(camera_index, **kwargs)
        self.resolution = resolution
        self.camera_name = camera_name or f"Camera {camera_index}"
        self.backend = backend
        self.actual_resolution = None
        self.camera_capabilities = {}
    
    def start(self) -> bool:
        """Start webcam dengan pengaturan resolusi dan backend"""
        # Try specific backend jika diberikan
        if self.backend is not None:
            self.cap = cv2.VideoCapture(self.source, self.backend)
        
        if super().start():
            # Apply laptop camera optimization jika ini laptop camera
            if self.source == 0:  # Laptop camera
                self._apply_laptop_optimization()
            
            # Set resolution jika diberikan
            if self.resolution:
                width, height = self.resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Verify actual resolution
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.actual_resolution = (actual_width, actual_height)
                self.logger.info(f"Webcam '{self.camera_name}' resolution set to: {actual_width}x{actual_height}")
            
            # Get camera capabilities
            self._detect_camera_capabilities()
            
            return True
        return False
    
    def _detect_camera_capabilities(self):
        """Detect camera capabilities dan properties"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        try:
            self.camera_capabilities = {
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                'auto_exposure': self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                'backend_name': self._get_backend_name()
            }
            
            self.logger.debug(f"Camera capabilities detected: {self.camera_capabilities}")
            
        except Exception as e:
            self.logger.warning(f"Could not detect camera capabilities: {str(e)}")
    
    def _get_backend_name(self) -> str:
        """Get name of current backend being used"""
        try:
            backend_names = {
                cv2.CAP_DSHOW: "DirectShow",
                cv2.CAP_MSMF: "Media Foundation", 
                cv2.CAP_VFW: "Video for Windows",
                cv2.CAP_AVFOUNDATION: "AVFoundation",
                cv2.CAP_QT: "QuickTime",
                cv2.CAP_V4L2: "Video4Linux2",
                cv2.CAP_GSTREAMER: "GStreamer"
            }
            
            if self.backend is not None:
                return backend_names.get(self.backend, "Unknown")
            else:
                return "Auto-detected"
                
        except Exception:
            return "Unknown"
    
    def get_camera_info(self) -> Dict:
        """
        Get comprehensive camera information
        
        Returns:
            Dict: Camera information dan capabilities
        """
        return {
            'index': self.source if isinstance(self.source, int) else -1,
            'name': self.camera_name,
            'resolution': self.actual_resolution or self.resolution,
            'requested_resolution': self.resolution,
            'fps': self.fps,
            'capabilities': self.camera_capabilities,
            'is_running': self.is_running(),
            'frame_count': self.frame_count
        }
    
    def set_camera_property(self, prop: int, value: float) -> bool:
        """
        Set camera property (brightness, contrast, etc.)
        
        Args:
            prop: OpenCV property constant (cv2.CAP_PROP_*)
            value: Property value
            
        Returns:
            bool: True if successfully set
        """
        if self.cap is None or not self.cap.isOpened():
            return False
        
        try:
            result = self.cap.set(prop, value)
            if result:
                # Update capabilities cache
                self._detect_camera_capabilities()
            return result
        except Exception as e:
            self.logger.error(f"Error setting camera property {prop}: {str(e)}")
            return False
    
    def get_camera_property(self, prop: int) -> Optional[float]:
        """
        Get camera property value
        
        Args:
            prop: OpenCV property constant
            
        Returns:
            float or None: Property value
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        try:
            return self.cap.get(prop)
        except Exception as e:
            self.logger.error(f"Error getting camera property {prop}: {str(e)}")
            return None
    
    def adjust_exposure(self, exposure_value: float) -> bool:
        """
        Adjust camera exposure
        
        Args:
            exposure_value: Exposure value (-13 to -1 typical range)
            
        Returns:
            bool: True if successfully adjusted
        """
        # First disable auto exposure
        self.set_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
        return self.set_camera_property(cv2.CAP_PROP_EXPOSURE, exposure_value)
    
    def enable_auto_exposure(self) -> bool:
        """Enable automatic exposure control"""
        return self.set_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
    
    def _apply_laptop_optimization(self):
        """
        Apply laptop camera specific optimizations
        """
        try:
            # Import config inside method to avoid circular imports
            from config import LaptopCameraConfig, get_laptop_camera_settings
            
            if not LaptopCameraConfig.ENABLE_LAPTOP_CAMERA:
                return
            
            self.logger.info("Applying laptop camera optimizations...")
            
            # Get detection-optimized settings
            resolution, fps, settings = get_laptop_camera_settings('detection')
            
            # Apply settings
            optimizations_applied = 0
            total_optimizations = 0
            
            # Auto exposure
            total_optimizations += 1
            if settings.get('auto_exposure'):
                if self.set_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75):
                    optimizations_applied += 1
            else:
                if self.set_camera_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
                    optimizations_applied += 1
            
            # Brightness
            if settings.get('brightness') is not None:
                total_optimizations += 1
                if self.set_camera_property(cv2.CAP_PROP_BRIGHTNESS, settings['brightness']):
                    optimizations_applied += 1
            
            # Contrast
            if settings.get('contrast') is not None:
                total_optimizations += 1
                if self.set_camera_property(cv2.CAP_PROP_CONTRAST, settings['contrast']):
                    optimizations_applied += 1
            
            # Exposure (if manual)
            if not settings.get('auto_exposure', True) and settings.get('exposure') is not None:
                total_optimizations += 1
                if self.set_camera_property(cv2.CAP_PROP_EXPOSURE, settings['exposure']):
                    optimizations_applied += 1
            
            # Buffer size optimization untuk laptop camera
            total_optimizations += 1
            if self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1):
                optimizations_applied += 1
            
            success_rate = optimizations_applied / total_optimizations if total_optimizations > 0 else 0
            self.logger.info(f"Laptop camera optimization: {optimizations_applied}/{total_optimizations} settings applied ({success_rate*100:.1f}%)")
            
        except Exception as e:
            self.logger.warning(f"Error applying laptop camera optimizations: {str(e)}")
    
    def test_camera_quality(self, duration: float = 2.0) -> Dict:
        """
        Test camera quality dengan capture beberapa frames
        
        Args:
            duration: Test duration in seconds
            
        Returns:
            Dict: Quality test results
        """
        if not self.is_running():
            return {'error': 'Camera not running'}
        
        start_time = time.time()
        frame_count = 0
        valid_frames = 0
        total_size = 0
        
        while time.time() - start_time < duration:
            ret, frame = self.read()
            if ret and frame is not None:
                frame_count += 1
                
                # Check frame validity
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    valid_frames += 1
                    total_size += frame.nbytes
            
            time.sleep(0.1)  # 10 FPS test rate
        
        actual_duration = time.time() - start_time
        
        return {
            'test_duration': actual_duration,
            'total_frames': frame_count,
            'valid_frames': valid_frames,
            'frame_rate': frame_count / actual_duration if actual_duration > 0 else 0,
            'success_rate': valid_frames / frame_count if frame_count > 0 else 0,
            'avg_frame_size_mb': (total_size / valid_frames / (1024*1024)) if valid_frames > 0 else 0,
            'quality_score': (valid_frames / frame_count) * min(frame_count / (duration * 10), 1.0) if frame_count > 0 else 0
        }

def test_video_sources():
    """
    Function untuk test berbagai video sources
    """
    print("Testing video sources...")
    
    # Test webcam
    print("\n1. Testing webcam (index 0):")
    webcam = WebcamStream(0)
    if webcam.start():
        time.sleep(2)
        ret, frame = webcam.read()
        if ret:
            print(f"✅ Webcam OK - Frame size: {frame.shape}")
        else:
            print("❌ Webcam - No frame received")
        webcam.stop()
    else:
        print("❌ Webcam - Failed to start")
    
    # Test RTSP (contoh, ganti dengan RTSP URL Anda)
    print("\n2. Testing RTSP stream:")
    rtsp_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"
    rtsp = RTSPStream(rtsp_url)
    if rtsp.start():
        time.sleep(3)
        ret, frame = rtsp.read()
        if ret:
            print(f"✅ RTSP OK - Frame size: {frame.shape}")
        else:
            print("❌ RTSP - No frame received")
        rtsp.stop()
    else:
        print("❌ RTSP - Failed to connect")

if __name__ == "__main__":
    # Setup logging untuk testing
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_video_sources()