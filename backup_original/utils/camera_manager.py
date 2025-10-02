"""
Camera Manager untuk Live CCTV Plate Detection
Auto-detection dan management untuk multiple cameras (laptop, USB, IP camera)
"""

import cv2
import platform
import subprocess
import logging
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class CameraInfo:
    """Information tentang camera yang terdeteksi"""
    index: int
    name: str
    resolution: Tuple[int, int]
    fps: float
    backend: str
    available: bool
    error_message: str = ""

class CameraManager:
    """
    Manager untuk detect dan handle multiple cameras
    Support laptop camera, USB camera, dan IP camera
    """
    
    def __init__(self):
        """Initialize camera manager"""
        self.logger = logging.getLogger(__name__)
        self.detected_cameras: List[CameraInfo] = []
        self.platform = platform.system()
        self.logger.info("CameraManager initialized")
    
    def enumerate_cameras(self, max_cameras: int = 10) -> List[CameraInfo]:
        """
        Enumerate semua available cameras di system
        
        Args:
            max_cameras: Maximum number of cameras to scan
            
        Returns:
            List[CameraInfo]: List of detected cameras
        """
        self.logger.info(f"Enumerating cameras (max: {max_cameras})...")
        self.detected_cameras = []
        
        for camera_index in range(max_cameras):
            camera_info = self._probe_camera(camera_index)
            if camera_info:
                self.detected_cameras.append(camera_info)
                self.logger.info(f"Found camera {camera_index}: {camera_info.name}")
        
        self.logger.info(f"Camera enumeration complete: {len(self.detected_cameras)} cameras found")
        return self.detected_cameras
    
    def _probe_camera(self, index: int) -> Optional[CameraInfo]:
        """
        Probe single camera untuk get info
        
        Args:
            index: Camera index to probe
            
        Returns:
            CameraInfo atau None jika camera tidak available
        """
        try:
            # Try different backends based on platform
            backends = self._get_camera_backends()
            
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                
                if cap.isOpened():
                    # Test if we can actually read frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Get camera name (platform specific)
                        camera_name = self._get_camera_name(index)
                        backend_name = self._get_backend_name(backend)
                        
                        cap.release()
                        
                        return CameraInfo(
                            index=index,
                            name=camera_name,
                            resolution=(width, height),
                            fps=fps if fps > 0 else 30.0,
                            backend=backend_name,
                            available=True
                        )
                
                cap.release()
                
        except Exception as e:
            self.logger.debug(f"Error probing camera {index}: {str(e)}")
            return CameraInfo(
                index=index,
                name=f"Camera {index}",
                resolution=(0, 0),
                fps=0.0,
                backend="unknown",
                available=False,
                error_message=str(e)
            )
        
        return None
    
    def _get_camera_backends(self) -> List[int]:
        """Get list of camera backends berdasarkan platform"""
        if self.platform == "Windows":
            return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]
        elif self.platform == "Darwin":  # macOS
            return [cv2.CAP_AVFOUNDATION, cv2.CAP_QT]
        elif self.platform == "Linux":
            return [cv2.CAP_V4L2, cv2.CAP_GSTREAMER]
        else:
            return [cv2.CAP_ANY]
    
    def _get_backend_name(self, backend: int) -> str:
        """Convert backend constant ke readable name"""
        backend_names = {
            cv2.CAP_DSHOW: "DirectShow",
            cv2.CAP_MSMF: "Media Foundation",
            cv2.CAP_VFW: "Video for Windows",
            cv2.CAP_AVFOUNDATION: "AVFoundation",
            cv2.CAP_QT: "QuickTime",
            cv2.CAP_V4L2: "Video4Linux2",
            cv2.CAP_GSTREAMER: "GStreamer",
            cv2.CAP_ANY: "Any"
        }
        return backend_names.get(backend, "Unknown")
    
    def _get_camera_name(self, index: int) -> str:
        """
        Get human-readable camera name berdasarkan platform
        
        Args:
            index: Camera index
            
        Returns:
            str: Camera name
        """
        try:
            if self.platform == "Darwin":  # macOS
                return self._get_macos_camera_name(index)
            elif self.platform == "Linux":
                return self._get_linux_camera_name(index)
            elif self.platform == "Windows":
                return self._get_windows_camera_name(index)
            else:
                return f"Camera {index}"
        except:
            return f"Camera {index}"
    
    def _get_macos_camera_name(self, index: int) -> str:
        """Get camera name di macOS using system_profiler"""
        try:
            result = subprocess.run([
                'system_profiler', 'SPCameraDataType'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                output = result.stdout
                if 'FaceTime' in output:
                    return f"FaceTime HD Camera ({index})"
                elif 'USB' in output and 'Camera' in output:
                    return f"USB Camera ({index})"
            
        except Exception as e:
            self.logger.debug(f"Error getting macOS camera name: {e}")
        
        # Default names based on typical macOS setups
        if index == 0:
            return "Built-in Camera (FaceTime HD)"
        else:
            return f"External Camera {index}"
    
    def _get_linux_camera_name(self, index: int) -> str:
        """Get camera name di Linux using v4l2-ctl"""
        try:
            device_path = f"/dev/video{index}"
            result = subprocess.run([
                'v4l2-ctl', '--device', device_path, '--info'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Card type' in line:
                        card_name = line.split(':')[-1].strip()
                        return f"{card_name} ({index})"
            
        except Exception as e:
            self.logger.debug(f"Error getting Linux camera name: {e}")
        
        # Fallback to generic names
        if index == 0:
            return "Built-in Webcam"
        else:
            return f"USB Camera {index}"
    
    def _get_windows_camera_name(self, index: int) -> str:
        """Get camera name di Windows"""
        # Windows camera naming bisa complex, use generic names
        if index == 0:
            return "Integrated Webcam"
        else:
            return f"USB Camera {index}"
    
    def get_available_cameras(self) -> List[CameraInfo]:
        """Get list of available cameras (yang bisa digunakan)"""
        return [cam for cam in self.detected_cameras if cam.available]
    
    def get_laptop_camera(self) -> Optional[CameraInfo]:
        """
        Get laptop/built-in camera (biasanya index 0)
        
        Returns:
            CameraInfo or None if not found
        """
        available_cameras = self.get_available_cameras()
        
        # Prioritas: built-in camera biasanya index 0
        for camera in available_cameras:
            if camera.index == 0:
                return camera
        
        # Fallback: camera pertama yang available
        return available_cameras[0] if available_cameras else None
    
    def test_camera_capture(self, camera_info: CameraInfo, duration: float = 2.0) -> bool:
        """
        Test apakah camera bisa capture frames dengan baik
        
        Args:
            camera_info: Camera to test
            duration: Test duration in seconds
            
        Returns:
            bool: True jika camera berfungsi dengan baik
        """
        try:
            backends = self._get_camera_backends()
            
            for backend in backends:
                cap = cv2.VideoCapture(camera_info.index, backend)
                
                if cap.isOpened():
                    start_time = time.time()
                    frame_count = 0
                    
                    while time.time() - start_time < duration:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frame_count += 1
                        else:
                            break
                        
                        time.sleep(0.1)  # 10 FPS test rate
                    
                    cap.release()
                    
                    # Test berhasil jika minimal dapat 5 frames
                    success = frame_count >= 5
                    if success:
                        self.logger.info(f"Camera {camera_info.index} test passed: {frame_count} frames in {duration}s")
                    else:
                        self.logger.warning(f"Camera {camera_info.index} test failed: only {frame_count} frames")
                    
                    return success
                
                cap.release()
                
        except Exception as e:
            self.logger.error(f"Error testing camera {camera_info.index}: {str(e)}")
        
        return False
    
    def get_camera_info_dict(self) -> List[Dict]:
        """
        Get camera info dalam format dictionary (untuk JSON serialization)
        
        Returns:
            List[Dict]: Camera info dalam format dict
        """
        return [
            {
                'index': cam.index,
                'name': cam.name,
                'resolution': cam.resolution,
                'fps': cam.fps,
                'backend': cam.backend,
                'available': cam.available,
                'error_message': cam.error_message
            }
            for cam in self.detected_cameras
        ]
    
    def print_camera_summary(self):
        """Print summary of detected cameras"""
        print("\n" + "="*60)
        print("📷 CAMERA DETECTION SUMMARY")
        print("="*60)
        
        if not self.detected_cameras:
            print("❌ No cameras detected")
            return
        
        for camera in self.detected_cameras:
            status = "✅" if camera.available else "❌"
            print(f"{status} Camera {camera.index}: {camera.name}")
            print(f"   Resolution: {camera.resolution[0]}x{camera.resolution[1]}")
            print(f"   FPS: {camera.fps:.1f}")
            print(f"   Backend: {camera.backend}")
            if not camera.available and camera.error_message:
                print(f"   Error: {camera.error_message}")
            print()
        
        available_count = len(self.get_available_cameras())
        print(f"📊 Summary: {available_count}/{len(self.detected_cameras)} cameras available")
        print("="*60)

def test_camera_manager():
    """Test function untuk camera manager"""
    print("Testing Camera Manager...")
    
    manager = CameraManager()
    
    # Enumerate cameras
    cameras = manager.enumerate_cameras()
    manager.print_camera_summary()
    
    # Test laptop camera
    laptop_cam = manager.get_laptop_camera()
    if laptop_cam:
        print(f"\n🔍 Testing laptop camera: {laptop_cam.name}")
        success = manager.test_camera_capture(laptop_cam)
        print(f"Test result: {'✅ PASS' if success else '❌ FAIL'}")
    else:
        print("\n❌ No laptop camera detected")

if __name__ == "__main__":
    test_camera_manager()