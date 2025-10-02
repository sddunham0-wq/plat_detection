#!/usr/bin/env python3
"""
Live CCTV License Plate Detection System
Main program untuk deteksi plat nomor real-time

Usage:
    python main.py --source rtsp://user:pass@ip:port/stream1  # RTSP Camera
    python main.py --source 0                                # Webcam index
    python main.py --source video.mp4                        # Video file
    python main.py --laptop-camera                           # Laptop built-in camera
    python main.py --camera-select                           # Interactive camera selection
    python main.py --auto-camera                             # Auto-select best camera
    python main.py --list-cameras                            # List available cameras
    python main.py --help                                    # Show help
"""

import cv2
import argparse
import logging
import signal
import sys
import time
import threading
from datetime import datetime

# Import custom modules
from config import *
from database import PlateDatabase
from display_manager import create_display_manager
from utils.video_stream import VideoStream, RTSPStream, WebcamStream
from utils.plate_detector import LicensePlateDetector
from utils.frame_processor import FrameProcessor
from utils.motorcycle_plate_detector import MotorcyclePlateDetector
from utils.camera_selector import CameraSelector, get_laptop_camera, interactive_camera_selection, auto_select_camera

class LivePlateDetectionSystem:
    """
    Main class untuk live plate detection system
    """
    
    def __init__(self, source, config_override=None, use_enhanced=False):
        """
        Initialize detection system
        
        Args:
            source: Video source (RTSP URL, webcam index, atau file path)
            config_override: Override konfigurasi default
            use_enhanced: Enable enhanced detection for distant/unclear plates
        """
        self.source = source
        self.running = False
        self.start_time = time.time()
        self.motorcycle_mode = True  # Enable motorcycle detection by default
        self.motorcycle_confidence = 0.5
        self.use_enhanced = use_enhanced
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.video_stream = None
        self.plate_detector = None
        self.motorcycle_plate_detector = None
        self.frame_processor = None
        self.database = None
        self.display_manager = None
        
        # Statistics
        self.total_frames = 0
        self.last_stats_time = time.time()
        
        # Ensure folders exist
        ensure_folders_exist()
        
        self.logger.info("Live Plate Detection System initialized")
    
    def set_motorcycle_mode(self, enabled: bool, confidence: float = 0.5):
        """Set motorcycle detection mode"""
        self.motorcycle_mode = enabled
        self.motorcycle_confidence = confidence
        self.logger.info(f"Motorcycle mode: {'enabled' if enabled else 'disabled'} (confidence: {confidence})")
    
    def setup_logging(self):
        """Setup logging configuration"""
        # Create log directory
        if not os.path.exists(SystemConfig.LOG_FOLDER):
            os.makedirs(SystemConfig.LOG_FOLDER)
        
        # Configure logging
        log_filename = os.path.join(
            SystemConfig.LOG_FOLDER, 
            f"plate_detection_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        logging.basicConfig(
            level=getattr(logging, SystemConfig.LOG_LEVEL),
            format=SystemConfig.LOG_FORMAT,
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_components(self):
        """Initialize semua komponen sistem"""
        try:
            # Initialize database
            self.logger.info("Initializing database...")
            self.database = PlateDatabase()
            
            # Initialize plate detector
            self.logger.info("Initializing plate detector...")
            self.plate_detector = LicensePlateDetector(use_enhanced=self.use_enhanced)
            
            if self.use_enhanced:
                self.logger.info("✅ Enhanced detection mode enabled for distant/unclear plates")
            
            # Initialize motorcycle plate detector jika motorcycle mode enabled
            if self.motorcycle_mode:
                self.logger.info("Initializing motorcycle plate detector...")
                try:
                    self.motorcycle_plate_detector = MotorcyclePlateDetector(confidence=self.motorcycle_confidence)
                    if self.motorcycle_plate_detector.is_enabled():
                        self.logger.info("✅ Motorcycle plate detector enabled")
                    else:
                        self.logger.warning("⚠️ Motorcycle plate detector disabled (YOLO not available)")
                        self.motorcycle_plate_detector = None
                except Exception as e:
                    self.logger.warning(f"Failed to initialize motorcycle detector: {str(e)}")
                    self.motorcycle_plate_detector = None
            else:
                self.motorcycle_plate_detector = None
            
            # Initialize video stream
            self.logger.info(f"Initializing video stream: {self.source}")
            self.video_stream = self.create_video_stream(self.source)
            
            # Initialize frame processor with both detectors
            self.logger.info("Initializing frame processor...")
            self.frame_processor = FrameProcessor(
                self.plate_detector,
                max_threads=SystemConfig.MAX_THREADS,
                motorcycle_detector=self.motorcycle_plate_detector
            )
            
            # Setup callbacks
            self.frame_processor.set_detection_callback(self.on_detection)
            self.frame_processor.set_frame_callback(self.on_frame_processed)
            
            # Initialize display manager jika preview enabled
            if SystemConfig.SHOW_PREVIEW:
                self.logger.info("Initializing display manager...")
                try:
                    self.display_manager = create_display_manager(
                        window_name="Live Plate Detection",
                        force_fallback=False
                    )
                    self.logger.info(f"Display manager initialized: {type(self.display_manager).__name__}")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize display manager: {str(e)}")
                    self.logger.info("Display preview will be disabled")
                    SystemConfig.SHOW_PREVIEW = False
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def create_video_stream(self, source):
        """
        Create appropriate video stream based on source type
        
        Args:
            source: Video source
            
        Returns:
            VideoStream: Appropriate stream object
        """
        if isinstance(source, str) and source.startswith(('rtsp://', 'http://')):
            # RTSP or HTTP stream
            return RTSPStream(source, buffer_size=CCTVConfig.BUFFER_SIZE)
        elif isinstance(source, int) or source.isdigit():
            # Webcam
            camera_index = int(source)
            return WebcamStream(
                camera_index,
                resolution=(CCTVConfig.FRAME_WIDTH, CCTVConfig.FRAME_HEIGHT),
                buffer_size=CCTVConfig.BUFFER_SIZE
            )
        elif isinstance(source, str) and source == "laptop":
            # Special case untuk laptop camera
            laptop_cam_index = get_laptop_camera()
            if laptop_cam_index is not None:
                return WebcamStream(
                    laptop_cam_index,
                    resolution=(CCTVConfig.FRAME_WIDTH, CCTVConfig.FRAME_HEIGHT),
                    buffer_size=CCTVConfig.BUFFER_SIZE
                )
            else:
                raise ValueError("Laptop camera not available")
        else:
            # Video file
            return VideoStream(source, buffer_size=CCTVConfig.BUFFER_SIZE)
    
    def on_detection(self, detections):
        """
        Callback untuk handle hasil deteksi
        
        Args:
            detections: List of PlateDetection objects
        """
        for detection in detections:
            try:
                # Save ke database
                record_id = self.database.save_detection(
                    detection,
                    source_info=str(self.source),
                    save_image=True
                )
                
                # Log detection
                self.logger.info(
                    f"🚗 DETECTED: {detection.text} "
                    f"(confidence: {detection.confidence:.1f}%) "
                    f"[ID: {record_id}]"
                )
                
                # Check alerts
                self.check_alerts(detection)
                
            except Exception as e:
                self.logger.error(f"Error handling detection: {str(e)}")
    
    def on_frame_processed(self, result):
        """
        Callback untuk handle processed frame
        
        Args:
            result: ProcessingResult object
        """
        self.total_frames += 1
        
        # Show preview jika enabled
        if SystemConfig.SHOW_PREVIEW:
            self.show_preview(result)
        
        # Print statistics setiap 30 detik
        current_time = time.time()
        if current_time - self.last_stats_time > 30:
            self.print_statistics()
            self.last_stats_time = current_time
    
    def check_alerts(self, detection):
        """
        Check apakah plat nomor ada di watchlist/blacklist
        
        Args:
            detection: PlateDetection object
        """
        plate_text = detection.text.upper()
        
        if AlertConfig.ENABLE_ALERTS:
            # Check watchlist
            if plate_text in [p.upper() for p in AlertConfig.WATCHLIST_PLATES]:
                alert_msg = f"🚨 WATCHLIST ALERT: {plate_text} detected!"
                
                if AlertConfig.ALERT_CONSOLE:
                    print(f"\n{alert_msg}\n")
                
                if AlertConfig.ALERT_LOG_FILE:
                    self.logger.warning(alert_msg)
            
            # Check blacklist
            if plate_text in [p.upper() for p in AlertConfig.BLACKLIST_PLATES]:
                alert_msg = f"⚠️ BLACKLIST ALERT: {plate_text} detected!"
                
                if AlertConfig.ALERT_CONSOLE:
                    print(f"\n{alert_msg}\n")
                
                if AlertConfig.ALERT_LOG_FILE:
                    self.logger.error(alert_msg)
    
    def show_preview(self, result):
        """
        Tampilkan preview window dengan deteksi menggunakan ThreadSafeDisplayManager
        
        Args:
            result: ProcessingResult object
        """
        try:
            # Hanya jika display manager tersedia dan running
            if not self.display_manager or not self.display_manager.is_running():
                return
                
            # Create annotated frame
            annotated_frame = self.plate_detector.draw_detections(
                result.frame, 
                result.detections,
                show_roi=SystemConfig.SHOW_ROI
            )
            
            # Resize untuk preview
            height, width = annotated_frame.shape[:2]
            preview_width, preview_height = SystemConfig.PREVIEW_WINDOW_SIZE
            
            if width > preview_width or height > preview_height:
                scale = min(preview_width/width, preview_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                annotated_frame = cv2.resize(annotated_frame, (new_width, new_height))
            
            # Add info overlay
            # Get real-time detection stats
            detection_stats = self.frame_processor.get_statistics().get('detection_stats', {})
            total_detections = detection_stats.get('total_detections', 0)
            success_rate = detection_stats.get('success_rate', 0)

            info_text = f"Frame: {result.frame_id} | Current: {len(result.detections)} | Total: {total_detections} | Success: {success_rate:.1f}% | FPS: {self.get_fps():.1f}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Define keyboard callback
            def keyboard_callback(key):
                if key == ord('q'):
                    self.logger.info("User pressed 'q', stopping system...")
                    self.stop()
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    screenshot_path = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    self.logger.info(f"Screenshot saved: {screenshot_path}")
                elif key == ord('p'):
                    # Print statistics
                    self.print_statistics()
            
            # Show frame through thread-safe display manager
            self.display_manager.show_frame(annotated_frame, keyboard_callback)
                
        except Exception as e:
            self.logger.error(f"Error showing preview: {str(e)}")
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        elapsed_time = time.time() - self.start_time
        return self.total_frames / elapsed_time if elapsed_time > 0 else 0
    
    def print_statistics(self):
        """Print system statistics"""
        elapsed_time = time.time() - self.start_time
        fps = self.get_fps()
        
        # Get processor statistics
        if self.frame_processor:
            proc_stats = self.frame_processor.get_statistics()
        else:
            proc_stats = {}
        
        # Get detector statistics
        if self.plate_detector:
            det_stats = self.plate_detector.get_statistics()
        else:
            det_stats = {}
        
        print("\n" + "="*60)
        print("📊 SYSTEM STATISTICS")
        print("="*60)
        print(f"⏱️  Runtime: {elapsed_time:.1f}s")
        print(f"🎬 Total Frames: {self.total_frames}")
        print(f"📈 FPS: {fps:.1f}")
        # Get detection stats dari DetectionManager
        if self.frame_processor:
            proc_stats = self.frame_processor.get_statistics()
            detection_stats = proc_stats.get('detection_stats', {})
            total_detections = detection_stats.get('total_detections', 0)
        else:
            total_detections = 0

        print(f"🚗 Total Detections: {total_detections}")
        
        if proc_stats:
            print(f"⚙️  Processing FPS: {proc_stats.get('processing_fps', 0):.1f}")
            print(f"⏳ Avg Processing Time: {proc_stats.get('avg_processing_time', 0):.3f}s")
            print(f"📦 Queue Sizes: {proc_stats.get('queue_sizes', {})}")
        
        if det_stats:
            print(f"🎯 OCR Success Rate: {det_stats.get('success_rate', 0):.1f}%")
        
        print("="*60 + "\n")
    
    def start(self):
        """Start live detection system"""
        try:
            # Initialize components
            if not self.initialize_components():
                return False
            
            # Start video stream
            self.logger.info("Starting video stream...")
            if not self.video_stream.start():
                self.logger.error("Failed to start video stream")
                return False
            
            # Start frame processor
            self.logger.info("Starting frame processor...")
            self.frame_processor.start()
            
            # Start display manager jika ada
            if self.display_manager:
                self.logger.info("Starting display manager...")
                display_started = self.display_manager.start()
                if not display_started:
                    self.logger.warning("Display manager failed to start, preview will be disabled")
                    SystemConfig.SHOW_PREVIEW = False
            
            self.running = True
            self.logger.info("🚀 Live detection system started successfully!")
            print("\n" + "="*60)
            print("🎥 LIVE PLATE DETECTION SYSTEM")
            print("="*60)
            print(f"📹 Source: {self.source}")
            print("⌨️  Controls:")
            print("   'q' - Quit")
            print("   's' - Save screenshot") 
            print("   'p' - Print statistics")
            print("="*60 + "\n")
            
            # Main processing loop
            self.main_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {str(e)}")
            return False
    
    def main_loop(self):
        """Main processing loop"""
        while self.running and self.video_stream.is_running():
            try:
                # Get frame dari video stream
                ret, frame = self.video_stream.get_latest_frame()
                
                if ret and frame is not None:
                    # Add frame untuk processing
                    self.frame_processor.add_frame(frame)
                else:
                    time.sleep(0.01)  # Short sleep jika tidak ada frame
                    
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                time.sleep(0.1)
    
    def stop(self):
        """Stop detection system"""
        self.logger.info("Stopping live detection system...")
        
        self.running = False
        
        # Stop components
        if self.frame_processor:
            self.frame_processor.stop()
        
        if self.video_stream:
            self.video_stream.stop()
        
        # Stop display manager dengan proper cleanup
        if self.display_manager:
            self.display_manager.stop()
        else:
            # Fallback cleanup untuk cv2 windows
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
        
        self.logger.info("Live detection system stopped")

def signal_handler(signum, frame):
    """Handle system signals (Ctrl+C)"""
    print("\n🛑 Received interrupt signal, shutting down...")
    sys.exit(0)

def main():
    """Main function"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Live CCTV License Plate Detection System')
    parser.add_argument('--source', '-s', type=str, default=CCTVConfig.DEFAULT_RTSP_URL,
                        help='Video source: RTSP URL, webcam index (0), or video file path')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable preview window')
    parser.add_argument('--config', '-c', type=str,
                        help='Custom config file path')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--motorcycle-mode', action='store_true',
                        help='Enable enhanced motorcycle plate detection')
    parser.add_argument('--motorcycle-confidence', type=float, default=0.5,
                        help='Confidence threshold for motorcycle detection (default: 0.5)')
    parser.add_argument('--enhanced', action='store_true',
                        help='Enable enhanced detection for distant/unclear plates')
    parser.add_argument('--super-resolution', action='store_true',
                        help='Enable super-resolution for very small plates')
    parser.add_argument('--laptop-camera', action='store_true',
                        help='Use laptop built-in camera (auto-detect)')
    parser.add_argument('--camera-select', action='store_true',
                        help='Show interactive camera selection menu')
    parser.add_argument('--camera-info', type=int, metavar='INDEX',
                        help='Show detailed information for camera at INDEX')
    parser.add_argument('--auto-camera', action='store_true',
                        help='Auto-select best available camera')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List all available cameras and exit')
    
    args = parser.parse_args()
    
    # Handle camera-specific arguments first
    if args.list_cameras:
        print("🔍 Scanning for available cameras...")
        selector = CameraSelector()
        cameras = selector.discover_cameras()
        selector.camera_manager.print_camera_summary()
        sys.exit(0)
    
    if args.camera_info is not None:
        selector = CameraSelector()
        selector.discover_cameras()
        selector.show_camera_info(args.camera_info)
        sys.exit(0)
    
    # Determine video source based on camera arguments
    video_source = args.source
    
    if args.laptop_camera:
        print("📹 Using laptop camera...")
        laptop_cam_index = get_laptop_camera()
        if laptop_cam_index is not None:
            video_source = laptop_cam_index
        else:
            print("❌ Laptop camera not available")
            sys.exit(1)
    
    elif args.camera_select:
        print("📹 Interactive camera selection...")
        selected_cam = interactive_camera_selection()
        if selected_cam is not None:
            video_source = selected_cam
        else:
            print("❌ No camera selected")
            sys.exit(1)
    
    elif args.auto_camera:
        print("🤖 Auto-selecting best camera...")
        auto_cam = auto_select_camera()
        if auto_cam is not None:
            video_source = auto_cam
        else:
            print("❌ No suitable camera found")
            sys.exit(1)
    
    # Override config jika diperlukan
    if args.no_preview:
        SystemConfig.SHOW_PREVIEW = False
    
    if args.log_level:
        SystemConfig.LOG_LEVEL = args.log_level
    
    # Override enhanced detection config if requested
    if args.enhanced or args.super_resolution:
        from config import EnhancedDetectionConfig
        if args.super_resolution:
            EnhancedDetectionConfig.USE_SUPER_RESOLUTION = True
    
    # Create and start detection system
    try:
        detection_system = LivePlateDetectionSystem(video_source, use_enhanced=args.enhanced)
        
        # Set motorcycle mode if requested
        if args.motorcycle_mode:
            detection_system.set_motorcycle_mode(True, args.motorcycle_confidence)
        success = detection_system.start()
        
        if not success:
            print("❌ Failed to start detection system")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()