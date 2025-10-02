#!/usr/bin/env python3
"""
Dual Stream License Plate Detection System
Multi-camera system dengan IP camera dan webcam/local camera
"""

import cv2
import argparse
import logging
import signal
import sys
import time
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import custom modules
from config import *
from database import PlateDatabase
from display_manager import create_display_manager
from utils.video_stream import VideoStream, RTSPStream, WebcamStream
from utils.plate_detector import LicensePlateDetector
from utils.frame_processor import FrameProcessor
from utils.motorcycle_plate_detector import MotorcyclePlateDetector
from utils.detection_manager import detection_manager
from utils.camera_selector import CameraSelector, get_laptop_camera

class CameraStream:
    """Individual camera stream handler"""

    def __init__(self, camera_id: str, source: str, name: str = None):
        """
        Initialize camera stream

        Args:
            camera_id: Unique identifier untuk camera
            source: Video source (RTSP URL, webcam index, etc.)
            name: Display name untuk camera
        """
        self.camera_id = camera_id
        self.source = source
        self.name = name or f"Camera {camera_id}"

        # Stream components
        self.video_stream = None
        self.frame_processor = None
        self.running = False

        # Detection components
        self.plate_detector = None
        self.motorcycle_detector = None

        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        self.last_frame = None
        self.last_frame_time = 0

        # Setup logging
        self.logger = logging.getLogger(f"Camera_{camera_id}")

    def initialize(self) -> bool:
        """Initialize camera stream components"""
        try:
            # Initialize detectors
            self.plate_detector = LicensePlateDetector()

            try:
                self.motorcycle_detector = MotorcyclePlateDetector()
                if not self.motorcycle_detector.is_enabled():
                    self.motorcycle_detector = None
            except Exception as e:
                self.logger.warning(f"Motorcycle detector unavailable: {str(e)}")
                self.motorcycle_detector = None

            # Initialize video stream
            self.video_stream = self._create_video_stream(self.source)

            # Initialize frame processor
            self.frame_processor = FrameProcessor(
                self.plate_detector,
                max_threads=2,  # Reduced threads untuk multi-camera
                motorcycle_detector=self.motorcycle_detector
            )

            self.logger.info(f"Camera {self.camera_id} initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing camera {self.camera_id}: {str(e)}")
            return False

    def _create_video_stream(self, source):
        """Create appropriate video stream based on source type"""
        if isinstance(source, str) and source.startswith(('rtsp://', 'http://')):
            # RTSP or HTTP stream
            return RTSPStream(source, buffer_size=CCTVConfig.BUFFER_SIZE)
        elif isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            # Webcam
            camera_index = int(source)
            return WebcamStream(
                camera_index,
                resolution=(CCTVConfig.FRAME_WIDTH, CCTVConfig.FRAME_HEIGHT),
                buffer_size=CCTVConfig.BUFFER_SIZE
            )
        else:
            # Video file
            return VideoStream(source, buffer_size=CCTVConfig.BUFFER_SIZE)

    def start(self) -> bool:
        """Start camera stream processing"""
        try:
            # Start video stream
            if not self.video_stream.start():
                self.logger.error(f"Failed to start video stream for camera {self.camera_id}")
                return False

            # Start frame processor
            self.frame_processor.start()

            self.running = True
            self.logger.info(f"Camera {self.camera_id} started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting camera {self.camera_id}: {str(e)}")
            return False

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get latest frame from camera stream"""
        if not self.running or not self.video_stream:
            return False, None

        ret, frame = self.video_stream.get_latest_frame()
        if ret and frame is not None:
            self.last_frame = frame.copy()
            self.last_frame_time = time.time()
            self.total_frames += 1

            # Add frame untuk processing
            self.frame_processor.add_frame(frame)

        return ret, frame

    def stop(self):
        """Stop camera stream"""
        self.logger.info(f"Stopping camera {self.camera_id}...")

        self.running = False

        if self.frame_processor:
            self.frame_processor.stop()

        if self.video_stream:
            self.video_stream.stop()

        self.logger.info(f"Camera {self.camera_id} stopped")

    def get_statistics(self) -> Dict:
        """Get camera statistics"""
        stats = {
            'camera_id': self.camera_id,
            'name': self.name,
            'source': str(self.source),
            'total_frames': self.total_frames,
            'running': self.running,
            'has_recent_frame': (time.time() - self.last_frame_time) < 5.0 if self.last_frame_time > 0 else False
        }

        if self.frame_processor:
            proc_stats = self.frame_processor.get_statistics()
            stats.update(proc_stats)

        return stats

class DualStreamDetectionSystem:
    """
    Dual stream detection system untuk multiple cameras
    """

    def __init__(self, primary_source: str, secondary_source: str):
        """
        Initialize dual stream system

        Args:
            primary_source: Primary camera source
            secondary_source: Secondary camera source (IP camera)
        """
        self.primary_source = primary_source
        self.secondary_source = secondary_source
        self.running = False

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize cameras
        self.cameras = {
            'primary': CameraStream('primary', primary_source, 'Primary Camera'),
            'secondary': CameraStream('secondary', secondary_source, 'IP Camera')
        }

        # Shared components
        self.database = None
        self.display_manager = None

        # Multi-camera statistics
        self.global_detection_count = 0
        self.cross_camera_duplicates = 0
        self.start_time = time.time()

        # Ensure folders exist
        ensure_folders_exist()

        self.logger.info("Dual Stream Detection System initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        if not os.path.exists(SystemConfig.LOG_FOLDER):
            os.makedirs(SystemConfig.LOG_FOLDER)

        log_filename = os.path.join(
            SystemConfig.LOG_FOLDER,
            f"dual_stream_{datetime.now().strftime('%Y%m%d')}.log"
        )

        logging.basicConfig(
            level=getattr(logging, SystemConfig.LOG_LEVEL),
            format=SystemConfig.LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def initialize_components(self) -> bool:
        """Initialize semua komponen sistem"""
        try:
            # Initialize database
            self.logger.info("Initializing database...")
            self.database = PlateDatabase()

            # Initialize cameras
            for camera_id, camera in self.cameras.items():
                self.logger.info(f"Initializing {camera.name}...")
                if not camera.initialize():
                    self.logger.error(f"Failed to initialize {camera.name}")
                    return False

                # Setup detection callbacks
                camera.frame_processor.set_detection_callback(
                    lambda detections, cam_id=camera_id: self.on_detection(detections, cam_id)
                )
                camera.frame_processor.set_frame_callback(
                    lambda result, cam_id=camera_id: self.on_frame_processed(result, cam_id)
                )

            # Initialize display manager jika preview enabled
            if SystemConfig.SHOW_PREVIEW:
                self.logger.info("Initializing display manager...")
                try:
                    self.display_manager = create_display_manager(
                        window_name="Dual Stream Plate Detection",
                        force_fallback=False
                    )
                    self.logger.info(f"Display manager initialized: {type(self.display_manager).__name__}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize display manager: {str(e)}")
                    SystemConfig.SHOW_PREVIEW = False

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            return False

    def on_detection(self, detections: List, camera_id: str):
        """
        Handle detections from specific camera

        Args:
            detections: List of PlateDetection objects
            camera_id: Camera yang mendeteksi
        """
        for detection in detections:
            try:
                # Add camera info ke detection
                detection.camera_id = camera_id
                detection.camera_name = self.cameras[camera_id].name

                # Save ke database dengan camera info
                record_id = self.database.save_detection(
                    detection,
                    source_info=f"{self.cameras[camera_id].name} ({self.cameras[camera_id].source})",
                    save_image=True
                )

                self.global_detection_count += 1

                # Log detection dengan camera info
                self.logger.info(
                    f"🚗 DETECTED [{camera_id.upper()}]: {detection.text} "
                    f"(confidence: {detection.confidence:.1f}%) "
                    f"[ID: {record_id}]"
                )

            except Exception as e:
                self.logger.error(f"Error handling detection from {camera_id}: {str(e)}")

    def on_frame_processed(self, result, camera_id: str):
        """
        Handle processed frame from specific camera

        Args:
            result: ProcessingResult object
            camera_id: Camera yang memproses frame
        """
        # Show preview jika enabled
        if SystemConfig.SHOW_PREVIEW and self.display_manager:
            self.show_dual_preview()

    def show_dual_preview(self):
        """Show dual camera preview"""
        try:
            if not self.display_manager or not self.display_manager.is_running():
                return

            # Get latest frames dari both cameras
            frames = {}
            camera_info = {}

            for camera_id, camera in self.cameras.items():
                if camera.last_frame is not None:
                    # Create annotated frame
                    annotated_frame = camera.last_frame.copy()

                    # Add camera label
                    cv2.putText(annotated_frame, camera.name, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # Add camera stats
                    stats = camera.get_statistics()
                    detection_stats = stats.get('detection_stats', {})

                    info_text = f"Detections: {detection_stats.get('total_detections', 0)} | FPS: {stats.get('processing_fps', 0):.1f}"
                    cv2.putText(annotated_frame, info_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    frames[camera_id] = annotated_frame
                    camera_info[camera_id] = {
                        'name': camera.name,
                        'detections': detection_stats.get('total_detections', 0)
                    }

            # Create combined display
            if len(frames) >= 2:
                combined_frame = self._create_combined_frame(frames)

                # Add global statistics
                elapsed_time = time.time() - self.start_time
                global_fps = self.global_detection_count / elapsed_time if elapsed_time > 0 else 0

                global_info = f"Global Detections: {self.global_detection_count} | Runtime: {elapsed_time:.0f}s"
                cv2.putText(combined_frame, global_info, (10, combined_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Define keyboard callback
                def keyboard_callback(key):
                    if key == ord('q'):
                        self.logger.info("User pressed 'q', stopping system...")
                        self.stop()
                    elif key == ord('s'):
                        # Save screenshot
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        screenshot_path = f"dual_stream_screenshot_{timestamp}.jpg"
                        cv2.imwrite(screenshot_path, combined_frame)
                        self.logger.info(f"Screenshot saved: {screenshot_path}")
                    elif key == ord('p'):
                        # Print statistics
                        self.print_statistics()

                # Show combined frame
                self.display_manager.show_frame(combined_frame, keyboard_callback)

        except Exception as e:
            self.logger.error(f"Error showing dual preview: {str(e)}")

    def _create_combined_frame(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create combined frame dari multiple camera streams

        Args:
            frames: Dictionary of camera frames

        Returns:
            Combined frame
        """
        try:
            frame_list = list(frames.values())

            if len(frame_list) == 1:
                return frame_list[0]

            # Resize frames untuk consistent size
            target_width = 640
            target_height = 480

            resized_frames = []
            for frame in frame_list:
                resized = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(resized)

            # Create side-by-side layout
            if len(resized_frames) == 2:
                combined = np.hstack(resized_frames)
            else:
                # For more than 2 cameras, create grid layout
                rows = []
                for i in range(0, len(resized_frames), 2):
                    if i + 1 < len(resized_frames):
                        row = np.hstack([resized_frames[i], resized_frames[i + 1]])
                    else:
                        # Pad dengan black frame jika odd number
                        black_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        row = np.hstack([resized_frames[i], black_frame])
                    rows.append(row)

                combined = np.vstack(rows)

            return combined

        except Exception as e:
            self.logger.error(f"Error creating combined frame: {str(e)}")
            # Return black frame as fallback
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def start(self) -> bool:
        """Start dual stream detection system"""
        try:
            # Initialize components
            if not self.initialize_components():
                return False

            # Start all cameras
            for camera_id, camera in self.cameras.items():
                self.logger.info(f"Starting {camera.name}...")
                if not camera.start():
                    self.logger.error(f"Failed to start {camera.name}")
                    return False

            # Start display manager jika ada
            if self.display_manager:
                self.logger.info("Starting display manager...")
                display_started = self.display_manager.start()
                if not display_started:
                    self.logger.warning("Display manager failed to start, preview will be disabled")
                    SystemConfig.SHOW_PREVIEW = False

            self.running = True
            self.logger.info("🚀 Dual stream detection system started successfully!")

            print("\n" + "="*80)
            print("🎥 DUAL STREAM PLATE DETECTION SYSTEM")
            print("="*80)
            print(f"📹 Primary Camera: {self.primary_source}")
            print(f"📹 Secondary Camera (IP): {self.secondary_source}")
            print("⌨️  Controls:")
            print("   'q' - Quit")
            print("   's' - Save screenshot")
            print("   'p' - Print statistics")
            print("="*80 + "\n")

            # Main processing loop
            self.main_loop()

            return True

        except Exception as e:
            self.logger.error(f"Error starting dual stream system: {str(e)}")
            return False

    def main_loop(self):
        """Main processing loop untuk dual streams"""
        while self.running:
            try:
                all_cameras_running = True

                # Process frames dari semua cameras
                for camera_id, camera in self.cameras.items():
                    if camera.running:
                        ret, frame = camera.get_latest_frame()
                        if not ret:
                            # Camera might be disconnected
                            if time.time() - camera.last_frame_time > 10.0:  # 10 seconds timeout
                                self.logger.warning(f"{camera.name} appears disconnected")
                                all_cameras_running = False
                    else:
                        all_cameras_running = False

                # Check if we should continue
                if not all_cameras_running:
                    time.sleep(0.1)  # Brief pause if cameras have issues
                else:
                    time.sleep(0.01)  # Short sleep untuk main loop

            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(0.1)

    def stop(self):
        """Stop dual stream detection system"""
        self.logger.info("Stopping dual stream detection system...")

        self.running = False

        # Stop all cameras
        for camera_id, camera in self.cameras.items():
            camera.stop()

        # Stop display manager
        if self.display_manager:
            self.display_manager.stop()
        else:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        # Print final statistics
        self.print_statistics()

        # Print database info
        if self.database:
            db_info = self.database.get_database_info()
            print(f"\n💾 Database: {db_info.get('total_records', 0)} records saved")

        self.logger.info("Dual stream detection system stopped")

    def print_statistics(self):
        """Print comprehensive system statistics"""
        elapsed_time = time.time() - self.start_time

        print("\n" + "="*80)
        print("📊 DUAL STREAM SYSTEM STATISTICS")
        print("="*80)
        print(f"⏱️  Runtime: {elapsed_time:.1f}s")
        print(f"🚗 Global Detections: {self.global_detection_count}")
        print(f"🔄 Cross-Camera Duplicates: {self.cross_camera_duplicates}")

        # Camera-specific statistics
        for camera_id, camera in self.cameras.items():
            stats = camera.get_statistics()
            print(f"\n📹 {camera.name.upper()}:")
            print(f"   Total Frames: {stats['total_frames']}")
            print(f"   Processing FPS: {stats.get('processing_fps', 0):.1f}")
            print(f"   Status: {'🟢 Running' if stats['running'] else '🔴 Stopped'}")

            detection_stats = stats.get('detection_stats', {})
            if detection_stats:
                print(f"   Detections: {detection_stats.get('total_detections', 0)}")
                print(f"   Success Rate: {detection_stats.get('success_rate', 0):.1f}%")

        # Detection Manager Statistics
        dm_stats = detection_manager.get_statistics()
        print(f"\n🎯 DETECTION MANAGER:")
        print(f"   Total Valid Detections: {dm_stats['valid_detections']}")
        print(f"   Filtered Non-plates: {dm_stats['filtered_non_plates']}")
        print(f"   Duplicate Filter Rate: {dm_stats['duplicate_filter_rate']:.1f}%")

        print("="*80 + "\n")

def signal_handler(signum, frame):
    """Handle system signals (Ctrl+C)"""
    print("\n🛑 Received interrupt signal, shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dual Stream License Plate Detection System')
    parser.add_argument('--primary', '-p', type=str, default='0',
                        help='Primary camera source: RTSP URL, webcam index (0), or video file path')
    parser.add_argument('--secondary', '-s', type=str, default='rtsp://admin:password@192.168.1.195:554/cam/realmonitor?channel=1&subtype=0',
                        help='Secondary camera source (IP camera)')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable preview window')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--laptop-camera', action='store_true',
                        help='Use laptop built-in camera as primary')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List all available cameras and exit')

    args = parser.parse_args()

    # Handle camera listing
    if args.list_cameras:
        print("🔍 Scanning for available cameras...")
        selector = CameraSelector()
        cameras = selector.discover_cameras()
        selector.camera_manager.print_camera_summary()
        sys.exit(0)

    # Determine primary source
    primary_source = args.primary
    if args.laptop_camera:
        print("📹 Using laptop camera as primary...")
        laptop_cam_index = get_laptop_camera()
        if laptop_cam_index is not None:
            primary_source = laptop_cam_index
        else:
            print("❌ Laptop camera not available")
            sys.exit(1)

    # Secondary source (IP camera)
    secondary_source = args.secondary

    # Override config jika diperlukan
    if args.no_preview:
        SystemConfig.SHOW_PREVIEW = False

    if args.log_level:
        SystemConfig.LOG_LEVEL = args.log_level

    # Create and start dual stream system
    try:
        print(f"🚀 Starting Dual Stream Detection System...")
        print(f"📹 Primary: {primary_source}")
        print(f"📹 Secondary: {secondary_source}")

        detection_system = DualStreamDetectionSystem(primary_source, secondary_source)
        success = detection_system.start()

        if not success:
            print("❌ Failed to start dual stream detection system")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()