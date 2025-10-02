#!/usr/bin/env python3
"""
Fast Streaming Mode untuk Live CCTV License Plate Detection
Optimized untuk streaming performance dengan deteksi real-time
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
from utils.camera_selector import CameraSelector, get_laptop_camera, interactive_camera_selection, auto_select_camera

class FastStreamingSystem:
    """
    Fast streaming system dengan optimasi untuk performance
    """
    
    def __init__(self, source, fast_mode=True):
        """
        Initialize fast streaming system
        
        Args:
            source: Video source (RTSP URL, webcam index, atau file path)
            fast_mode: Enable fast mode optimizations
        """
        self.source = source
        self.fast_mode = fast_mode
        self.running = False
        self.frame_skip_counter = 0
        
        # Override config untuk fast mode
        if fast_mode:
            # Temporary override untuk performance
            CCTVConfig.FPS_LIMIT = 30
            CCTVConfig.BUFFER_SIZE = 3
            SystemConfig.MAX_THREADS = 1
            DetectionConfig.MIN_CONTOUR_AREA = 800  # Larger to skip small objects
            TesseractConfig.MIN_CONFIDENCE = 35
            EnhancedDetectionConfig.ENABLE_ENHANCED_DETECTION = False
            TrackingConfig.ENABLE_TRACKING = False
            
        # Initialize components
        self.logger = self._setup_logging()
        self.database = PlateDatabase()
        self.detector = LicensePlateDetector()
        self.video_stream = None
        self.frame_processor = None
        self.display_manager = None
        
        # Statistics
        self.total_frames = 0
        self.processed_frames = 0
        self.skipped_frames = 0
        self.total_detections = 0
        self.start_time = time.time()
        
        self.logger.info("🚀 Fast Streaming System initialized")
        if fast_mode:
            self.logger.info("⚡ Fast mode optimizations enabled")
    
    def _setup_logging(self):
        """Setup logging untuk fast streaming"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'logs/fast_streaming_{datetime.now().strftime("%Y%m%d")}.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def start(self):
        """Start fast streaming system"""
        try:
            # Initialize video stream dengan optimasi
            self.video_stream = self._create_video_stream(self.source)
            if not self.video_stream.start():
                self.logger.error("Failed to start video stream")
                return False
            
            # Initialize frame processor dengan reduced threads
            self.frame_processor = FrameProcessor(
                plate_detector=self.detector,
                max_threads=1,  # Single thread untuk stability
                queue_size=5    # Small queue untuk low latency
            )
            
            # Set callbacks
            self.frame_processor.set_detection_callback(self.on_detection)
            self.frame_processor.set_frame_callback(self.on_frame_processed)
            
            # Start frame processor
            if not self.frame_processor.start():
                self.logger.error("Failed to start frame processor")
                return False
            
            # Initialize display manager
            self.display_manager = create_display_manager(
                show_preview=SystemConfig.SHOW_PREVIEW,
                window_size=SystemConfig.PREVIEW_WINDOW_SIZE
            )
            
            self.logger.info("🎬 Fast streaming started successfully")
            self.running = True
            
            # Main processing loop
            self._main_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in fast streaming: {str(e)}")
        finally:
            self.stop()
        
        return True
    
    def _create_video_stream(self, source):
        """Create optimized video stream"""
        if isinstance(source, str) and source.startswith('rtsp://'):
            return RTSPStream(
                source,
                timeout=CCTVConfig.RTSP_TIMEOUT,
                buffer_size=CCTVConfig.BUFFER_SIZE
            )
        elif isinstance(source, int) or source.isdigit():
            return WebcamStream(
                int(source),
                resolution=(CCTVConfig.FRAME_WIDTH, CCTVConfig.FRAME_HEIGHT),
                buffer_size=CCTVConfig.BUFFER_SIZE
            )
        else:
            return VideoStream(source, buffer_size=CCTVConfig.BUFFER_SIZE)
    
    def _main_loop(self):
        """Main processing loop dengan frame skipping"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running:
            # Read frame dari video stream
            ret, frame = self.video_stream.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            self.total_frames += 1
            
            # Frame skipping logic untuk performance
            if self.fast_mode and CCTVConfig.ENABLE_FRAME_SKIPPING:
                self.frame_skip_counter += 1
                if self.frame_skip_counter % CCTVConfig.PROCESS_EVERY_N_FRAMES != 0:
                    self.skipped_frames += 1
                    # Still show frame tapi skip processing
                    if self.display_manager:
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"SKIPPED FRAME", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        self.display_manager.update_frame(display_frame)
                    continue
            
            # Process frame
            self.processed_frames += 1
            self.frame_processor.add_frame(frame)
            
            # FPS calculation
            fps_counter += 1
            if fps_counter >= 30:  # Calculate FPS every 30 frames
                elapsed = time.time() - fps_start_time
                current_fps = fps_counter / elapsed
                self.logger.info(f"📊 Streaming FPS: {current_fps:.1f}, "
                               f"Processed: {self.processed_frames}, "
                               f"Skipped: {self.skipped_frames}, "
                               f"Detections: {self.total_detections}")
                fps_counter = 0
                fps_start_time = time.time()
            
            # Small delay untuk prevent CPU overload
            time.sleep(0.001)
    
    def on_detection(self, detections):
        """Handle deteksi plat nomor"""
        for detection in detections:
            try:
                # Save ke database
                record_id = self.database.save_detection(
                    detection,
                    source_info=str(self.source),
                    save_image=True
                )
                
                self.total_detections += 1
                
                # Log detection
                self.logger.info(
                    f"🚗 DETECTED: {detection.text} "
                    f"(conf: {detection.confidence:.1f}%) "
                    f"[ID: {record_id}]"
                )
                
            except Exception as e:
                self.logger.error(f"Error handling detection: {str(e)}")
    
    def on_frame_processed(self, result):
        """Handle processed frame"""
        if self.display_manager and result.frame is not None:
            # Add performance info to frame
            display_frame = result.frame.copy()
            
            # Add streaming info
            total_time = time.time() - self.start_time
            avg_fps = self.processed_frames / total_time if total_time > 0 else 0
            
            info_text = [
                f"FAST STREAMING MODE",
                f"FPS: {avg_fps:.1f}",
                f"Processed: {self.processed_frames}",
                f"Skipped: {self.skipped_frames}",
                f"Detections: {self.total_detections}",
                f"Process Time: {result.processing_time:.1f}ms"
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)  # Green for title
                cv2.putText(display_frame, text, (10, y_offset + i * 25),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            self.display_manager.update_frame(display_frame)
    
    def stop(self):
        """Stop fast streaming system"""
        self.logger.info("🛑 Stopping fast streaming system...")
        self.running = False
        
        if self.frame_processor:
            self.frame_processor.stop()
        
        if self.video_stream:
            self.video_stream.stop()
        
        if self.display_manager:
            self.display_manager.cleanup()
        
        # Print final statistics
        total_time = time.time() - self.start_time
        avg_fps = self.processed_frames / total_time if total_time > 0 else 0
        
        self.logger.info("📊 Final Statistics:")
        self.logger.info(f"   Total Runtime: {total_time:.1f} seconds")
        self.logger.info(f"   Average FPS: {avg_fps:.1f}")
        self.logger.info(f"   Total Frames: {self.total_frames}")
        self.logger.info(f"   Processed Frames: {self.processed_frames}")
        self.logger.info(f"   Skipped Frames: {self.skipped_frames}")
        self.logger.info(f"   Total Detections: {self.total_detections}")
        self.logger.info("✅ Fast streaming stopped")

def signal_handler(sig, frame):
    """Handle CTRL+C signal"""
    print("\n⚠️  Received interrupt signal. Stopping...")
    sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fast Streaming License Plate Detection')
    parser.add_argument('--source', type=str, help='Video source (RTSP URL, webcam index, atau file path)')
    parser.add_argument('--laptop-camera', action='store_true', help='Use laptop built-in camera')
    parser.add_argument('--camera-select', action='store_true', help='Interactive camera selection')
    parser.add_argument('--auto-camera', action='store_true', help='Auto-select best camera')
    parser.add_argument('--list-cameras', action='store_true', help='List available cameras')
    parser.add_argument('--normal-mode', action='store_true', help='Disable fast mode optimizations')
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Ensure required folders exist
    ensure_folders_exist()
    
    try:
        # Determine video source
        source = None
        
        if args.list_cameras:
            print("📹 Available cameras:")
            camera_selector = CameraSelector()
            cameras = camera_selector.list_available_cameras()
            for i, camera in enumerate(cameras):
                print(f"  {i}: {camera}")
            return
        
        if args.laptop_camera:
            source = get_laptop_camera()
            if source is None:
                print("❌ Laptop camera not available")
                return
            print(f"📱 Using laptop camera: {source}")
        
        elif args.camera_select:
            source = interactive_camera_selection()
            if source is None:
                print("❌ No camera selected")
                return
        
        elif args.auto_camera:
            source = auto_select_camera()
            if source is None:
                print("❌ No suitable camera found")
                return
            print(f"🎯 Auto-selected camera: {source}")
        
        elif args.source:
            source = args.source
        else:
            # Default ke laptop camera
            source = get_laptop_camera()
            if source is None:
                source = CCTVConfig.DEFAULT_RTSP_URL
            print(f"🎬 Using default source: {source}")
        
        # Initialize dan start fast streaming system
        fast_mode = not args.normal_mode
        system = FastStreamingSystem(source, fast_mode=fast_mode)
        
        print(f"🚀 Starting fast streaming detection...")
        print(f"📹 Source: {source}")
        print(f"⚡ Fast mode: {'Enabled' if fast_mode else 'Disabled'}")
        print(f"⏹️  Press CTRL+C to stop")
        
        success = system.start()
        
        if not success:
            print("❌ Failed to start fast streaming system")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)