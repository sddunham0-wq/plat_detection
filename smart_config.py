#!/usr/bin/env python3
"""
Smart Configuration System
Auto-tuning config berdasarkan camera type dan environment

Menggantikan complex config system yang lama dengan smart defaults
yang automatically optimize berdasarkan scenario detection.
"""

import os
from typing import Dict, Optional


class SmartConfig:
    """
    Smart configuration yang auto-adjust berdasarkan scenario

    Supported scenarios:
    - rtsp_cctv: CCTV cameras dengan jarak jauh
    - laptop_camera: Built-in laptop cameras
    - webcam: USB webcams
    - video_file: Video file processing
    """

    @staticmethod
    def get_config_for_scenario(camera_type: str = "auto") -> Dict:
        """
        Get optimized config untuk specific camera type

        Args:
            camera_type: "rtsp_cctv", "laptop_camera", "webcam", "video_file", atau "auto"

        Returns:
            Dict: Optimized configuration
        """

        configs = {
            "rtsp_cctv": SmartConfig._get_rtsp_config(),
            "laptop_camera": SmartConfig._get_laptop_config(),
            "webcam": SmartConfig._get_webcam_config(),
            "video_file": SmartConfig._get_video_config(),
            "auto": SmartConfig._get_balanced_config()
        }

        return configs.get(camera_type, configs["auto"])

    @staticmethod
    def _get_rtsp_config() -> Dict:
        """Optimized untuk RTSP CCTV cameras - jarak jauh, quality rendah"""
        return {
            # YOLO settings - lower confidence untuk distant detection
            'yolo_model': 'yolov8n.pt',      # Nano untuk speed pada CCTV
            'yolo_confidence': 0.35,         # Lower untuk catch distant vehicles
            'yolo_iou_threshold': 0.4,       # Lower IoU untuk overlapping objects
            'yolo_max_detections': 12,       # More detections untuk crowded scenes

            # Plate extraction - smaller plates dari jarak jauh
            'plate_min_area': 200,           # Much smaller untuk distant plates
            'plate_max_area': 15000,         # Smaller max area
            'plate_min_aspect_ratio': 1.5,   # More tolerant
            'plate_max_aspect_ratio': 6.0,   # More tolerant untuk perspective
            'plate_min_width': 25,           # Smaller minimum
            'plate_max_width': 300,
            'plate_min_height': 10,          # Much smaller
            'plate_max_height': 100,

            # OCR - aggressive upscaling untuk small plates
            'ocr_min_confidence': 45,        # Lower threshold untuk CCTV
            'ocr_psm_modes': [7, 8, 6, 13],  # More PSM modes
            'ocr_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'ocr_upscale_factor': 4,         # Higher upscaling
            'ocr_noise_reduction': True,     # Enable noise reduction

            # Performance optimization
            'max_candidates_to_process': 20, # More candidates untuk CCTV
            'early_termination_confidence': 75,  # Lower untuk CCTV quality

            # Indonesian validation - more lenient untuk CCTV
            'indonesian_patterns': [
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',
                r'^\d{1,4}\s*[A-Z]{2,4}$',
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',
                r'^[A-Z]\d{1,4}[A-Z]{1,3}$'  # No spaces pattern
            ],
            'regional_codes': ['B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
            'enable_strict_validation': False,  # More lenient

            # Stability - more frames untuk CCTV stability
            'stability_required_frames': 5,     # More frames untuk confirmation
            'stability_confidence_boost': 15.0, # Higher boost
            'stability_max_drift': 80,          # Allow more movement
            'stability_timeout': 8.0,           # Longer timeout

            # Performance
            'enable_parallel_processing': True,
            'enable_frame_skipping': True,      # Skip frames untuk performance
            'skip_every_n_frames': 2,           # Process every 2nd frame
            'target_fps': 8,                    # Lower FPS untuk stability

            # Scenario info
            'scenario': 'rtsp_cctv',
            'description': 'Optimized untuk RTSP CCTV cameras - distant detection'
        }

    @staticmethod
    def _get_laptop_config() -> Dict:
        """Optimized untuk laptop built-in cameras - close range, good quality"""
        return {
            # YOLO settings - higher confidence untuk close range
            'yolo_model': 'yolov8s.pt',      # Small model untuk balanced performance
            'yolo_confidence': 0.7,          # Higher untuk close-range accuracy
            'yolo_iou_threshold': 0.5,       # Standard IoU
            'yolo_max_detections': 6,        # Fewer objects di close range

            # Plate extraction - larger plates dari close camera
            'plate_min_area': 800,           # Larger plates expected
            'plate_max_area': 25000,
            'plate_min_aspect_ratio': 2.0,   # More strict
            'plate_max_aspect_ratio': 4.5,   # More strict
            'plate_min_width': 60,           # Larger minimum
            'plate_max_width': 400,
            'plate_min_height': 20,          # Larger minimum
            'plate_max_height': 120,

            # OCR - less aggressive processing untuk good quality
            'ocr_min_confidence': 65,        # Higher threshold
            'ocr_psm_modes': [7, 8],         # Fewer PSM modes
            'ocr_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'ocr_upscale_factor': 2,         # Less upscaling needed
            'ocr_noise_reduction': False,    # Good quality, no noise reduction

            # Indonesian validation - strict untuk good quality
            'indonesian_patterns': [
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',
                r'^\d{1,4}\s*[A-Z]{2,4}$',
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
            ],
            'regional_codes': ['B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
            'enable_strict_validation': True,   # Strict validation

            # Stability - fewer frames untuk responsiveness
            'stability_required_frames': 3,     # Standard frames
            'stability_confidence_boost': 10.0,
            'stability_max_drift': 40,          # Less movement expected
            'stability_timeout': 5.0,

            # Performance
            'enable_parallel_processing': True,
            'enable_frame_skipping': False,     # Process all frames
            'skip_every_n_frames': 1,
            'target_fps': 15,                   # Higher FPS

            # Scenario info
            'scenario': 'laptop_camera',
            'description': 'Optimized untuk laptop built-in cameras - close range detection'
        }

    @staticmethod
    def _get_webcam_config() -> Dict:
        """Optimized untuk USB webcams - medium range, variable quality"""
        return {
            # YOLO settings - balanced
            'yolo_model': 'yolov8n.pt',      # Nano untuk performance
            'yolo_confidence': 0.6,          # Balanced confidence
            'yolo_iou_threshold': 0.45,      # Standard IoU
            'yolo_max_detections': 8,        # Balanced detections

            # Plate extraction - medium settings
            'plate_min_area': 600,
            'plate_max_area': 20000,
            'plate_min_aspect_ratio': 1.8,
            'plate_max_aspect_ratio': 5.0,
            'plate_min_width': 50,
            'plate_max_width': 350,
            'plate_min_height': 15,
            'plate_max_height': 110,

            # OCR - balanced settings
            'ocr_min_confidence': 60,
            'ocr_psm_modes': [7, 8, 13],
            'ocr_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'ocr_upscale_factor': 3,         # Moderate upscaling
            'ocr_noise_reduction': True,     # Some noise reduction

            # Performance optimization
            'max_candidates_to_process': 15,  # Standard amount
            'early_termination_confidence': 80,  # Standard confidence

            # Indonesian validation - standard
            'indonesian_patterns': [
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',
                r'^\d{1,4}\s*[A-Z]{2,4}$',
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
            ],
            'regional_codes': ['B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
            'enable_strict_validation': True,

            # Stability - standard settings
            'stability_required_frames': 3,
            'stability_confidence_boost': 10.0,
            'stability_max_drift': 50,
            'stability_timeout': 6.0,

            # Performance
            'enable_parallel_processing': True,
            'enable_frame_skipping': False,
            'skip_every_n_frames': 1,
            'target_fps': 12,

            # Scenario info
            'scenario': 'webcam',
            'description': 'Optimized untuk USB webcams - balanced detection'
        }

    @staticmethod
    def _get_video_config() -> Dict:
        """Optimized untuk video file processing - accuracy over speed"""
        return {
            # YOLO settings - higher accuracy
            'yolo_model': 'yolov8s.pt',      # Small model untuk accuracy
            'yolo_confidence': 0.65,         # Higher confidence
            'yolo_iou_threshold': 0.45,
            'yolo_max_detections': 10,

            # Plate extraction - comprehensive settings
            'plate_min_area': 400,
            'plate_max_area': 30000,         # Larger range
            'plate_min_aspect_ratio': 1.5,
            'plate_max_aspect_ratio': 6.0,
            'plate_min_width': 30,
            'plate_max_width': 400,
            'plate_min_height': 12,
            'plate_max_height': 150,

            # OCR - comprehensive processing
            'ocr_min_confidence': 55,        # Lower untuk more detections
            'ocr_psm_modes': [6, 7, 8, 13],  # All PSM modes
            'ocr_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'ocr_upscale_factor': 3,
            'ocr_noise_reduction': True,

            # Indonesian validation - comprehensive
            'indonesian_patterns': [
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',
                r'^\d{1,4}\s*[A-Z]{2,4}$',
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',
                r'^[A-Z]\d{1,4}[A-Z]{1,3}$'
            ],
            'regional_codes': ['B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
            'enable_strict_validation': False,  # More lenient untuk video

            # Stability - thorough analysis
            'stability_required_frames': 2,     # Fewer frames untuk video
            'stability_confidence_boost': 5.0,  # Less boost
            'stability_max_drift': 100,         # More movement allowed
            'stability_timeout': 10.0,

            # Performance - accuracy over speed
            'enable_parallel_processing': True,
            'enable_frame_skipping': False,     # Process all frames
            'skip_every_n_frames': 1,
            'target_fps': 5,                    # Lower FPS untuk thorough analysis

            # Scenario info
            'scenario': 'video_file',
            'description': 'Optimized untuk video file processing - accuracy focused'
        }

    @staticmethod
    def _get_balanced_config() -> Dict:
        """Balanced config untuk auto/unknown scenarios"""
        return {
            # YOLO settings - balanced untuk semua scenarios
            'yolo_model': 'yolov8n.pt',
            'yolo_confidence': 0.55,         # Balanced confidence
            'yolo_iou_threshold': 0.45,
            'yolo_max_detections': 8,

            # Plate extraction - balanced settings
            'plate_min_area': 500,
            'plate_max_area': 20000,
            'plate_min_aspect_ratio': 1.8,
            'plate_max_aspect_ratio': 5.0,
            'plate_min_width': 40,
            'plate_max_width': 350,
            'plate_min_height': 15,
            'plate_max_height': 120,

            # OCR - balanced untuk detection dan speed
            'ocr_min_confidence': 35,
            'ocr_psm_modes': [6, 7, 8],
            'ocr_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'ocr_upscale_factor': 3,
            'ocr_noise_reduction': True,

            # Performance optimization
            'max_candidates_to_process': 15,  # Balanced amount
            'early_termination_confidence': 80,  # Standard confidence

            # Indonesian validation - standard
            'indonesian_patterns': [
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',
                r'^\d{1,4}\s*[A-Z]{2,4}$',
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
            ],
            'regional_codes': ['B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],
            'enable_strict_validation': True,

            # Stability - balanced
            'stability_required_frames': 3,
            'stability_confidence_boost': 10.0,
            'stability_max_drift': 50,
            'stability_timeout': 6.0,

            # Performance - balanced
            'enable_parallel_processing': True,
            'enable_frame_skipping': False,
            'skip_every_n_frames': 1,
            'target_fps': 10,

            # Scenario info
            'scenario': 'balanced',
            'description': 'Balanced config untuk general use'
        }

    @staticmethod
    def detect_camera_type(source) -> str:
        """
        Auto-detect camera type berdasarkan source

        Args:
            source: Video source (URL, file path, atau integer)

        Returns:
            str: Detected camera type
        """
        if isinstance(source, str):
            source_lower = source.lower()

            # RTSP camera detection
            if source_lower.startswith('rtsp://') or source_lower.startswith('http://'):
                return 'rtsp_cctv'

            # Video file detection
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
            if any(source_lower.endswith(ext) for ext in video_extensions):
                return 'video_file'

            # IP camera detection
            if ('192.168.' in source_lower or '10.' in source_lower or
                '172.' in source_lower or 'camera' in source_lower):
                return 'rtsp_cctv'

        elif isinstance(source, int):
            # Webcam or laptop camera
            if source == 0:
                return 'laptop_camera'  # Usually built-in camera
            else:
                return 'webcam'         # External USB camera

        # Default fallback
        return 'auto'

    @staticmethod
    def get_optimized_config(source) -> Dict:
        """
        Get optimized config dengan auto-detection

        Args:
            source: Video source

        Returns:
            Dict: Optimized configuration
        """
        detected_type = SmartConfig.detect_camera_type(source)
        config = SmartConfig.get_config_for_scenario(detected_type)

        # Add source info to config
        config['detected_source_type'] = detected_type
        config['original_source'] = str(source)

        return config

    @staticmethod
    def validate_config(config: Dict) -> bool:
        """
        Validate config completeness

        Args:
            config: Configuration dict

        Returns:
            bool: True if valid, False otherwise
        """
        required_keys = [
            'yolo_model', 'yolo_confidence', 'yolo_iou_threshold',
            'plate_min_area', 'plate_max_area',
            'ocr_min_confidence', 'ocr_psm_modes',
            'indonesian_patterns', 'regional_codes',
            'stability_required_frames'
        ]

        return all(key in config for key in required_keys)

    @staticmethod
    def print_config_summary(config: Dict):
        """Print configuration summary"""
        print(f"\n📋 Configuration Summary")
        print(f"   Scenario: {config.get('scenario', 'unknown')}")
        print(f"   Description: {config.get('description', 'No description')}")
        print(f"   YOLO Model: {config.get('yolo_model', 'unknown')}")
        print(f"   YOLO Confidence: {config.get('yolo_confidence', 0)}")
        print(f"   OCR Min Confidence: {config.get('ocr_min_confidence', 0)}")
        print(f"   Stability Frames: {config.get('stability_required_frames', 0)}")
        print(f"   Target FPS: {config.get('target_fps', 0)}")

        if 'detected_source_type' in config:
            print(f"   Detected Source: {config['detected_source_type']}")


# Factory functions untuk common scenarios
def get_rtsp_config() -> Dict:
    """Get RTSP CCTV optimized config"""
    return SmartConfig.get_config_for_scenario("rtsp_cctv")


def get_laptop_config() -> Dict:
    """Get laptop camera optimized config"""
    return SmartConfig.get_config_for_scenario("laptop_camera")


def get_webcam_config() -> Dict:
    """Get USB webcam optimized config"""
    return SmartConfig.get_config_for_scenario("webcam")


def get_video_config() -> Dict:
    """Get video file processing optimized config"""
    return SmartConfig.get_config_for_scenario("video_file")


def get_auto_config(source) -> Dict:
    """Get auto-detected optimized config"""
    return SmartConfig.get_optimized_config(source)


# Test function
def test_smart_config():
    """Test SmartConfig functionality"""
    print("🧪 Testing SmartConfig...")

    try:
        # Test different camera types
        rtsp_config = get_rtsp_config()
        laptop_config = get_laptop_config()
        webcam_config = get_webcam_config()

        print("✅ All config types generated successfully")

        # Test validation
        valid = SmartConfig.validate_config(rtsp_config)
        print(f"✅ Config validation: {'PASS' if valid else 'FAIL'}")

        # Test auto-detection
        test_sources = [
            "rtsp://192.168.1.100/stream",
            0,
            1,
            "video.mp4"
        ]

        for source in test_sources:
            detected = SmartConfig.detect_camera_type(source)
            print(f"✅ Source '{source}' detected as: {detected}")

        return True

    except Exception as e:
        print(f"❌ SmartConfig test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test when executed directly
    test_smart_config()