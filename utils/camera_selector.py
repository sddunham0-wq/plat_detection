"""
Camera Selector untuk Live CCTV Plate Detection
Interactive camera selection dengan auto-discovery dan testing
"""

import cv2
import time
import logging
from typing import List, Optional, Dict
from utils.camera_manager import CameraManager, CameraInfo
from utils.video_stream import WebcamStream

class CameraSelector:
    """
    Interactive camera selector dengan testing dan validation
    """
    
    def __init__(self):
        """Initialize camera selector"""
        self.logger = logging.getLogger(__name__)
        self.camera_manager = CameraManager()
        self.available_cameras: List[CameraInfo] = []
        
    def discover_cameras(self) -> List[CameraInfo]:
        """
        Discover semua available cameras
        
        Returns:
            List[CameraInfo]: List of available cameras
        """
        print("🔍 Discovering cameras...")
        self.available_cameras = self.camera_manager.enumerate_cameras()
        return self.available_cameras
    
    def show_camera_menu(self) -> Optional[int]:
        """
        Show interactive camera selection menu
        
        Returns:
            int: Selected camera index or None if cancelled
        """
        if not self.available_cameras:
            self.discover_cameras()
        
        available_cams = self.camera_manager.get_available_cameras()
        
        if not available_cams:
            print("❌ No cameras detected! Please check your camera connections.")
            return None
        
        print("\n" + "="*60)
        print("📷 CAMERA SELECTION MENU")
        print("="*60)
        
        # Show camera options
        for i, camera in enumerate(available_cams):
            status_icon = "🔴" if camera.index == 0 else "📹"
            camera_type = "(Laptop Camera)" if camera.index == 0 else "(External Camera)"
            
            print(f"{i+1}. {status_icon} {camera.name} {camera_type}")
            print(f"   Index: {camera.index} | Resolution: {camera.resolution[0]}x{camera.resolution[1]} | FPS: {camera.fps:.1f}")
            print(f"   Backend: {camera.backend}")
            print()
        
        print("0. Cancel")
        print("="*60)
        
        # Get user selection
        while True:
            try:
                choice = input("Select camera (enter number): ").strip()
                
                if choice == "0":
                    print("❌ Camera selection cancelled")
                    return None
                
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(available_cams):
                    selected_camera = available_cams[choice_idx]
                    
                    # Test selected camera
                    print(f"\n🔍 Testing {selected_camera.name}...")
                    if self.test_camera_quick(selected_camera):
                        print(f"✅ Camera {selected_camera.index} selected: {selected_camera.name}")
                        return selected_camera.index
                    else:
                        print(f"❌ Camera {selected_camera.index} test failed. Please try another camera.")
                        continue
                else:
                    print("❌ Invalid selection. Please try again.")
                    
            except ValueError:
                print("❌ Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled by user")
                return None
    
    def get_laptop_camera(self) -> Optional[int]:
        """
        Get laptop camera (biasanya index 0) tanpa menu
        
        Returns:
            int: Laptop camera index or None if not available
        """
        if not self.available_cameras:
            self.discover_cameras()
        
        laptop_cam = self.camera_manager.get_laptop_camera()
        
        if laptop_cam:
            print(f"📹 Found laptop camera: {laptop_cam.name}")
            
            # Test laptop camera
            print("🔍 Testing laptop camera...")
            if self.test_camera_quick(laptop_cam):
                print(f"✅ Laptop camera ready: {laptop_cam.name}")
                return laptop_cam.index
            else:
                print("❌ Laptop camera test failed")
                return None
        else:
            print("❌ No laptop camera detected")
            return None
    
    def test_camera_quick(self, camera_info: CameraInfo, duration: float = 2.0) -> bool:
        """
        Quick test untuk camera functionality
        
        Args:
            camera_info: Camera to test
            duration: Test duration in seconds
            
        Returns:
            bool: True if camera works properly
        """
        try:
            # Create webcam stream for testing
            webcam = WebcamStream(
                camera_index=camera_info.index,
                resolution=(640, 480)
            )
            
            if webcam.start():
                print(f"   ⏳ Testing for {duration} seconds...")
                
                start_time = time.time()
                frame_count = 0
                
                while time.time() - start_time < duration:
                    ret, frame = webcam.read()
                    if ret and frame is not None:
                        frame_count += 1
                    time.sleep(0.1)  # 10 FPS test rate
                
                webcam.stop()
                
                # Test berhasil jika minimal dapat 5 frames
                success = frame_count >= 5
                if success:
                    print(f"   ✅ Test passed: {frame_count} frames captured")
                else:
                    print(f"   ❌ Test failed: only {frame_count} frames captured")
                
                return success
            else:
                print("   ❌ Failed to start camera")
                return False
                
        except Exception as e:
            self.logger.error(f"Error testing camera {camera_info.index}: {str(e)}")
            print(f"   ❌ Test error: {str(e)}")
            return False
    
    def show_camera_info(self, camera_index: int) -> bool:
        """
        Show detailed camera information
        
        Args:
            camera_index: Camera index to show info for
            
        Returns:
            bool: True if camera info shown successfully
        """
        try:
            # Find camera info
            camera_info = None
            for cam in self.available_cameras:
                if cam.index == camera_index and cam.available:
                    camera_info = cam
                    break
            
            if not camera_info:
                print(f"❌ Camera {camera_index} not found or not available")
                return False
            
            print("\n" + "="*50)
            print(f"📷 CAMERA INFO - {camera_info.name}")
            print("="*50)
            print(f"Index: {camera_info.index}")
            print(f"Name: {camera_info.name}")
            print(f"Resolution: {camera_info.resolution[0]}x{camera_info.resolution[1]}")
            print(f"FPS: {camera_info.fps:.1f}")
            print(f"Backend: {camera_info.backend}")
            print(f"Status: {'✅ Available' if camera_info.available else '❌ Not Available'}")
            
            # Test camera dengan detailed info
            print("\n🔍 Running detailed camera test...")
            
            webcam = WebcamStream(camera_index=camera_index)
            if webcam.start():
                time.sleep(1)  # Wait for camera to stabilize
                
                # Get actual camera info
                actual_info = webcam.get_camera_info()
                print(f"Actual Resolution: {actual_info.get('resolution', 'Unknown')}")
                print(f"Actual FPS: {actual_info.get('fps', 'Unknown')}")
                
                # Test quality
                quality_result = webcam.test_camera_quality(duration=3.0)
                print(f"Quality Test Results:")
                print(f"  Frame Rate: {quality_result.get('frame_rate', 0):.1f} FPS")
                print(f"  Success Rate: {quality_result.get('success_rate', 0)*100:.1f}%")
                print(f"  Quality Score: {quality_result.get('quality_score', 0)*100:.1f}%")
                
                webcam.stop()
                print("✅ Camera test completed")
            else:
                print("❌ Failed to test camera")
            
            print("="*50)
            return True
            
        except Exception as e:
            self.logger.error(f"Error showing camera info: {str(e)}")
            print(f"❌ Error getting camera info: {str(e)}")
            return False
    
    def auto_select_best_camera(self) -> Optional[int]:
        """
        Auto-select best available camera based on quality and type
        
        Returns:
            int: Best camera index or None if no cameras available
        """
        if not self.available_cameras:
            self.discover_cameras()
        
        available_cams = self.camera_manager.get_available_cameras()
        
        if not available_cams:
            return None
        
        print("🤖 Auto-selecting best camera...")
        
        # Score cameras based on criteria
        scored_cameras = []
        
        for camera in available_cams:
            score = 0
            
            # Prioritas laptop camera (index 0)
            if camera.index == 0:
                score += 50
            
            # Resolution score (higher is better, but not too high)
            width, height = camera.resolution
            if 640 <= width <= 1920 and 480 <= height <= 1080:
                score += 30
            elif width >= 640 and height >= 480:
                score += 20
            
            # FPS score
            if 15 <= camera.fps <= 30:
                score += 20
            elif camera.fps > 30:
                score += 10
            
            # Test camera untuk additional score
            if self.test_camera_quick(camera, duration=1.0):
                score += 30
            
            scored_cameras.append((score, camera))
        
        # Sort by score (highest first)
        scored_cameras.sort(key=lambda x: x[0], reverse=True)
        
        if scored_cameras:
            best_score, best_camera = scored_cameras[0]
            print(f"✅ Auto-selected: {best_camera.name} (score: {best_score})")
            return best_camera.index
        
        return None

def interactive_camera_selection() -> Optional[int]:
    """
    Function helper untuk interactive camera selection
    
    Returns:
        int: Selected camera index or None
    """
    selector = CameraSelector()
    return selector.show_camera_menu()

def get_laptop_camera() -> Optional[int]:
    """
    Function helper untuk get laptop camera
    
    Returns:
        int: Laptop camera index or None
    """
    selector = CameraSelector()
    return selector.get_laptop_camera()

def auto_select_camera() -> Optional[int]:
    """
    Function helper untuk auto camera selection
    
    Returns:
        int: Best camera index or None
    """
    selector = CameraSelector()
    return selector.auto_select_best_camera()

if __name__ == "__main__":
    # Test camera selector
    print("Testing Camera Selector...")
    
    selector = CameraSelector()
    
    # Test auto selection
    print("\n1. Testing auto selection:")
    auto_cam = selector.auto_select_best_camera()
    if auto_cam is not None:
        print(f"Auto-selected camera: {auto_cam}")
    
    # Test laptop camera
    print("\n2. Testing laptop camera:")
    laptop_cam = selector.get_laptop_camera()
    if laptop_cam is not None:
        print(f"Laptop camera: {laptop_cam}")
    
    # Test interactive menu
    print("\n3. Testing interactive menu:")
    selected_cam = selector.show_camera_menu()
    if selected_cam is not None:
        print(f"User selected camera: {selected_cam}")
        
        # Show detailed info
        selector.show_camera_info(selected_cam)