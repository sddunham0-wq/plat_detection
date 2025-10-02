"""
Camera Configuration Utilities untuk Live CCTV Plate Detection
Auto-configuration dan optimization untuk different camera types
"""

import cv2
import logging
import platform
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class CameraSettings:
    """Camera settings configuration"""
    resolution: Tuple[int, int]
    fps: int
    auto_exposure: bool
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    saturation: Optional[float] = None
    exposure: Optional[float] = None
    focus: Optional[float] = None

class CameraConfigurator:
    """
    Auto-configuration utility untuk optimize camera settings
    based on camera type dan detection requirements
    """
    
    def __init__(self):
        """Initialize camera configurator"""
        self.logger = logging.getLogger(__name__)
        self.platform = platform.system()
        
        # Predefined configurations untuk different scenarios
        self.presets = {
            'laptop_default': CameraSettings(
                resolution=(640, 480),
                fps=15,
                auto_exposure=True,
                brightness=0.0,
                contrast=0.0
            ),
            'laptop_quality': CameraSettings(
                resolution=(1280, 720),
                fps=10,
                auto_exposure=False,
                brightness=10.0,
                contrast=20.0,
                exposure=-6.0
            ),
            'laptop_performance': CameraSettings(
                resolution=(640, 480),
                fps=20,
                auto_exposure=True,
                brightness=0.0,
                contrast=10.0
            ),
            'usb_camera_default': CameraSettings(
                resolution=(640, 480),
                fps=15,
                auto_exposure=True
            ),
            'detection_optimized': CameraSettings(
                resolution=(640, 480),
                fps=10,
                auto_exposure=False,
                brightness=15.0,
                contrast=25.0,
                exposure=-5.0
            )
        }
    
    def get_optimal_settings(self, camera_index: int, scenario: str = 'detection') -> CameraSettings:
        """
        Get optimal camera settings berdasarkan camera type dan scenario
        
        Args:
            camera_index: Camera index
            scenario: Usage scenario ('detection', 'quality', 'performance')
            
        Returns:
            CameraSettings: Optimal settings untuk camera
        """
        # Detect camera type
        camera_type = self._detect_camera_type(camera_index)
        
        # Select preset based on camera type and scenario
        if camera_type == 'laptop':
            if scenario == 'quality':
                preset_key = 'laptop_quality'
            elif scenario == 'performance':
                preset_key = 'laptop_performance'
            else:
                preset_key = 'laptop_default'
        else:
            if scenario == 'detection':
                preset_key = 'detection_optimized'
            else:
                preset_key = 'usb_camera_default'
        
        settings = self.presets.get(preset_key, self.presets['laptop_default'])
        
        self.logger.info(f"Selected preset '{preset_key}' for camera {camera_index} ({camera_type})")
        return settings
    
    def _detect_camera_type(self, camera_index: int) -> str:
        """
        Detect camera type berdasarkan index dan properties
        
        Args:
            camera_index: Camera index
            
        Returns:
            str: Camera type ('laptop', 'usb', 'ip', 'unknown')
        """
        if camera_index == 0:
            return 'laptop'  # Biasanya built-in camera
        elif camera_index < 4:
            return 'usb'     # External USB camera
        else:
            return 'unknown'
    
    def apply_settings(self, cap: cv2.VideoCapture, settings: CameraSettings) -> bool:
        """
        Apply camera settings ke VideoCapture object
        
        Args:
            cap: OpenCV VideoCapture object
            settings: CameraSettings to apply
            
        Returns:
            bool: True if settings applied successfully
        """
        if not cap.isOpened():
            return False
        
        try:
            success_count = 0
            total_attempts = 0
            
            # Set resolution
            total_attempts += 2
            if cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0]):
                success_count += 1
            if cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1]):
                success_count += 1
            
            # Set FPS
            total_attempts += 1
            if cap.set(cv2.CAP_PROP_FPS, settings.fps):
                success_count += 1
            
            # Set auto exposure
            total_attempts += 1
            if settings.auto_exposure:
                if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75):  # Auto exposure on
                    success_count += 1
            else:
                if cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):  # Manual exposure
                    success_count += 1
            
            # Set optional properties
            optional_props = [
                (cv2.CAP_PROP_BRIGHTNESS, settings.brightness),
                (cv2.CAP_PROP_CONTRAST, settings.contrast),
                (cv2.CAP_PROP_SATURATION, settings.saturation),
                (cv2.CAP_PROP_EXPOSURE, settings.exposure),
                (cv2.CAP_PROP_FOCUS, settings.focus)
            ]
            
            for prop, value in optional_props:
                if value is not None:
                    total_attempts += 1
                    if cap.set(prop, value):
                        success_count += 1
            
            success_rate = success_count / total_attempts if total_attempts > 0 else 0
            
            self.logger.info(f"Applied camera settings: {success_count}/{total_attempts} properties set ({success_rate*100:.1f}%)")
            
            return success_rate > 0.5  # Consider successful if >50% of properties were set
            
        except Exception as e:
            self.logger.error(f"Error applying camera settings: {str(e)}")
            return False
    
    def optimize_for_detection(self, cap: cv2.VideoCapture) -> bool:
        """
        Apply detection-optimized settings
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            bool: True if optimization successful
        """
        detection_settings = self.presets['detection_optimized']
        return self.apply_settings(cap, detection_settings)
    
    def get_current_settings(self, cap: cv2.VideoCapture) -> Dict:
        """
        Get current camera settings
        
        Args:
            cap: OpenCV VideoCapture object
            
        Returns:
            Dict: Current camera properties
        """
        if not cap.isOpened():
            return {}
        
        try:
            properties = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': cap.get(cv2.CAP_PROP_SATURATION),
                'exposure': cap.get(cv2.CAP_PROP_EXPOSURE),
                'auto_exposure': cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                'focus': cap.get(cv2.CAP_PROP_FOCUS),
                'buffer_size': cap.get(cv2.CAP_PROP_BUFFERSIZE)
            }
            
            return properties
            
        except Exception as e:
            self.logger.error(f"Error getting camera settings: {str(e)}")
            return {}
    
    def print_settings_comparison(self, cap: cv2.VideoCapture, settings: CameraSettings):
        """
        Print comparison between desired and actual settings
        
        Args:
            cap: OpenCV VideoCapture object
            settings: Desired settings
        """
        current = self.get_current_settings(cap)
        
        print("\n" + "="*50)
        print("📹 CAMERA SETTINGS COMPARISON")
        print("="*50)
        print(f"{'Property':<15} {'Desired':<12} {'Actual':<12} {'Status'}")
        print("-" * 50)
        
        # Resolution
        desired_res = f"{settings.resolution[0]}x{settings.resolution[1]}"
        actual_res = f"{current.get('width', 0)}x{current.get('height', 0)}"
        status = "✅" if desired_res == actual_res else "⚠️"
        print(f"{'Resolution':<15} {desired_res:<12} {actual_res:<12} {status}")
        
        # FPS
        desired_fps = f"{settings.fps}"
        actual_fps = f"{current.get('fps', 0):.1f}"
        status = "✅" if abs(settings.fps - current.get('fps', 0)) < 2 else "⚠️"
        print(f"{'FPS':<15} {desired_fps:<12} {actual_fps:<12} {status}")
        
        # Auto Exposure
        desired_auto = "Yes" if settings.auto_exposure else "No"
        actual_auto = "Yes" if current.get('auto_exposure', 0) > 0.5 else "No"
        status = "✅" if desired_auto == actual_auto else "⚠️"
        print(f"{'Auto Exposure':<15} {desired_auto:<12} {actual_auto:<12} {status}")
        
        # Optional properties
        optional_props = [
            ('Brightness', settings.brightness, current.get('brightness')),
            ('Contrast', settings.contrast, current.get('contrast')),
            ('Exposure', settings.exposure, current.get('exposure'))
        ]
        
        for prop_name, desired, actual in optional_props:
            if desired is not None:
                desired_str = f"{desired:.1f}"
                actual_str = f"{actual:.1f}" if actual is not None else "N/A"
                if actual is not None:
                    status = "✅" if abs(desired - actual) < 5 else "⚠️"
                else:
                    status = "❌"
                print(f"{prop_name:<15} {desired_str:<12} {actual_str:<12} {status}")
        
        print("="*50)
    
    def benchmark_camera(self, camera_index: int, test_duration: float = 5.0) -> Dict:
        """
        Benchmark camera performance dengan different settings
        
        Args:
            camera_index: Camera index to benchmark
            test_duration: Test duration in seconds
            
        Returns:
            Dict: Benchmark results
        """
        results = {}
        
        for preset_name, settings in self.presets.items():
            if 'laptop' in preset_name or preset_name == 'detection_optimized':
                print(f"\n🔍 Testing preset: {preset_name}")
                
                try:
                    cap = cv2.VideoCapture(camera_index)
                    if cap.isOpened():
                        # Apply settings
                        self.apply_settings(cap, settings)
                        
                        # Benchmark
                        import time
                        start_time = time.time()
                        frame_count = 0
                        
                        while time.time() - start_time < test_duration:
                            ret, frame = cap.read()
                            if ret:
                                frame_count += 1
                            time.sleep(0.01)  # Small sleep to prevent overload
                        
                        actual_duration = time.time() - start_time
                        avg_fps = frame_count / actual_duration
                        
                        results[preset_name] = {
                            'fps': avg_fps,
                            'frames': frame_count,
                            'duration': actual_duration,
                            'resolution': settings.resolution,
                            'success': True
                        }
                        
                        print(f"   Result: {avg_fps:.1f} FPS ({frame_count} frames)")
                        
                    cap.release()
                    
                except Exception as e:
                    results[preset_name] = {
                        'error': str(e),
                        'success': False
                    }
                    print(f"   Error: {str(e)}")
        
        return results

def get_laptop_camera_config(scenario: str = 'detection') -> CameraSettings:
    """
    Helper function untuk get optimal laptop camera settings
    
    Args:
        scenario: Usage scenario
        
    Returns:
        CameraSettings: Optimal settings
    """
    configurator = CameraConfigurator()
    return configurator.get_optimal_settings(0, scenario)

def apply_laptop_optimization(cap: cv2.VideoCapture) -> bool:
    """
    Helper function untuk apply laptop camera optimization
    
    Args:
        cap: OpenCV VideoCapture object
        
    Returns:
        bool: True if optimization successful
    """
    configurator = CameraConfigurator()
    settings = configurator.get_optimal_settings(0, 'detection')
    return configurator.apply_settings(cap, settings)

if __name__ == "__main__":
    # Test camera configurator
    print("Testing Camera Configurator...")
    
    configurator = CameraConfigurator()
    
    # Test dengan camera 0 (laptop camera)
    print("\n📹 Testing laptop camera configuration...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera opened successfully")
            
            # Get current settings
            current = configurator.get_current_settings(cap)
            print(f"Current resolution: {current.get('width')}x{current.get('height')}")
            
            # Apply detection optimization
            settings = configurator.get_optimal_settings(0, 'detection')
            success = configurator.apply_settings(cap, settings)
            
            if success:
                print("✅ Settings applied successfully")
                configurator.print_settings_comparison(cap, settings)
            else:
                print("⚠️ Some settings may not have been applied")
            
            cap.release()
        else:
            print("❌ Failed to open camera")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Benchmark test
    print("\n🏃 Running camera benchmark...")
    results = configurator.benchmark_camera(0, test_duration=3.0)
    
    print("\n📊 BENCHMARK RESULTS:")
    for preset, result in results.items():
        if result.get('success'):
            print(f"{preset}: {result['fps']:.1f} FPS")
        else:
            print(f"{preset}: Failed - {result.get('error', 'Unknown error')}")