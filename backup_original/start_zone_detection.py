#!/usr/bin/env python3
"""
Zone-Based Motorcycle Detection Launcher
Optimized untuk parking lot dengan banyak motor seperti screenshot
"""

import cv2
import argparse
import logging
import sys
import time
import numpy as np
from datetime import datetime

# Import modules
from config import *
from utils.zone_based_detection import ZoneBasedMotorcycleDetector
from utils.video_stream import VideoStream, RTSPStream, WebcamStream

def setup_logging():
    """Setup logging untuk zone detection"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_video_stream(source):
    """Create video stream berdasarkan source"""
    if isinstance(source, str) and source.startswith(('rtsp://', 'http://')):
        return RTSPStream(source, buffer_size=30)
    elif isinstance(source, int) or source.isdigit():
        camera_index = int(source)
        return WebcamStream(camera_index, resolution=(1280, 720), buffer_size=30)
    else:
        return VideoStream(source, buffer_size=30)

def run_zone_detection(source, mode="sequential", duration=None, show_zones=True):
    """
    Run zone-based motorcycle detection
    
    Args:
        source: Video source
        mode: Detection mode (sequential/parallel/adaptive)
        duration: Run duration in seconds (None = unlimited)
        show_zones: Show zone overlay
    """
    logger = logging.getLogger(__name__)
    
    # Initialize zone detector
    logger.info("Initializing zone-based motorcycle detector...")
    detector = ZoneBasedMotorcycleDetector(frame_width=1280, frame_height=720)
    detector.set_detection_mode(mode)
    
    # Initialize video stream
    logger.info(f"Initializing video stream: {source}")
    video_stream = create_video_stream(source)
    
    if not video_stream.start():
        logger.error("❌ Failed to start video stream!")
        return False
    
    logger.info("🚀 Starting zone-based motorcycle detection...")
    logger.info(f"🎯 Detection mode: {mode}")
    logger.info(f"📐 Zones: 3x3 grid (9 zones)")
    logger.info("⌨️  Press 'q' to quit, 's' to save, 'z' to toggle zones, 'm' to change mode")
    
    start_time = time.time()
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                logger.info(f"⏰ Duration {duration}s completed")
                break
            
            ret, frame = video_stream.get_latest_frame()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Detect motorcycles in zones
            zone_results = detector.detect_in_zones(frame)
            
            # Count detections
            current_detections = sum(len(results) for results in zone_results.values())
            total_detections += current_detections
            
            # Log significant detections
            if current_detections > 0:
                for zone_id, results in zone_results.items():
                    zone_name = next(z.zone_name for z in detector.zones if z.zone_id == zone_id)
                    logger.info(f"🏍️ Zone {zone_id} ({zone_name}): {len(results)} motorcycles")
                    
                    for result in results:
                        plates_text = ", ".join([p.text for p in result.plate_detections if p.text])
                        if plates_text:
                            logger.info(f"  📋 Plates: {plates_text}")
            
            # Draw visualizations
            annotated_frame = frame.copy()
            
            # Draw zone overlay
            if show_zones:
                annotated_frame = detector.draw_zone_overlay(annotated_frame, show_stats=True)
            
            # Draw detections
            for zone_id, results in zone_results.items():
                for result in results:
                    # Draw vehicle box
                    vx, vy, vw, vh = result.vehicle_detection.bbox
                    cv2.rectangle(annotated_frame, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 2)
                    
                    # Vehicle label dengan zone info
                    zone_name = next(z.zone_name for z in detector.zones if z.zone_id == zone_id)
                    label = f"🏍️ {zone_name} ({result.vehicle_detection.confidence:.1f}%)"
                    cv2.putText(annotated_frame, label, (vx, vy - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw plates
                    for plate in result.plate_detections:
                        px, py, pw, ph = plate.bbox
                        cv2.rectangle(annotated_frame, (px, py), (px + pw, py + ph), (0, 255, 255), 3)
                        
                        plate_text = f"📋 {plate.text} ({plate.confidence:.1f}%)"
                        cv2.putText(annotated_frame, plate_text, (px, py - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
            
            # Add info overlay
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            info_lines = [
                f"Zone Detection: {mode.upper()} mode",
                f"FPS: {fps:.1f} | Frames: {frame_count}",
                f"Total Detections: {total_detections}",
                f"Detection Rate: {total_detections}/{frame_count} frames"
            ]
            
            if duration:
                remaining = max(0, duration - elapsed_time)
                info_lines.insert(0, f"Time: {remaining:.1f}s remaining")
            
            y_offset = 60  # Start below mode indicator
            for line in info_lines:
                cv2.putText(annotated_frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Show frame
            cv2.imshow("Zone-Based Motorcycle Detection", annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User pressed 'q', stopping...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"zone_detection_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                logger.info(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('z'):
                # Toggle zone display
                show_zones = not show_zones
                logger.info(f"Zone display: {'ON' if show_zones else 'OFF'}")
            elif key == ord('m'):
                # Cycle through modes
                modes = ["sequential", "parallel", "adaptive"]
                current_idx = modes.index(detector.mode)
                new_mode = modes[(current_idx + 1) % len(modes)]
                detector.set_detection_mode(new_mode)
                logger.info(f"Switched to {new_mode} mode")
    
    except KeyboardInterrupt:
        logger.info("Detection interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
    
    finally:
        # Cleanup
        video_stream.stop()
        cv2.destroyAllWindows()
        
        # Print final statistics
        elapsed_time = time.time() - start_time
        stats = detector.get_zone_statistics()
        
        print("\n" + "="*70)
        print("📊 ZONE-BASED DETECTION RESULTS")
        print("="*70)
        print(f"⏱️  Duration: {elapsed_time:.1f}s")
        print(f"🎬 Total Frames: {frame_count}")
        print(f"📈 Average FPS: {frame_count/elapsed_time:.1f}")
        print(f"🏍️ Total Motorcycles: {stats['total_motorcycles']}")
        print(f"📋 Total Plates Read: {stats['total_plates_read']}")
        print(f"📊 Plate Success Rate: {stats['plate_success_rate']:.1f}%")
        print(f"🎯 Detection Mode: {stats['detection_mode']}")
        print(f"📐 Active Zones: {stats['active_zones']}")
        
        print("\n📋 Zone Breakdown:")
        for zone_id, zone_stats in stats['zone_breakdown'].items():
            zone_name = next(z.zone_name for z in detector.zones if z.zone_id == zone_id)
            print(f"  Zone {zone_id} ({zone_name}): "
                  f"🏍️{zone_stats['motorcycles']} motorcycles, "
                  f"📋{zone_stats['plates_read']} plates")
        
        print("="*70)
        
        if stats['total_motorcycles'] > 0:
            print("✅ SUCCESS: Zone-based detection working!")
            if stats['total_plates_read'] > 0:
                print("🎯 EXCELLENT: Plates were successfully read!")
            else:
                print("⚠️  NOTE: Motorcycles detected but plates hard to read (expected for distance)")
        else:
            print("❌ NO DETECTIONS: Try adjusting confidence or check camera angle")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Zone-Based Motorcycle Detection')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Video source: RTSP URL, webcam index (0), or video file')
    parser.add_argument('--mode', '-m', choices=['sequential', 'parallel', 'adaptive'],
                        default='sequential', help='Detection mode (default: sequential)')
    parser.add_argument('--duration', '-d', type=int,
                        help='Test duration in seconds (default: unlimited)')
    parser.add_argument('--no-zones', action='store_true',
                        help='Hide zone overlay')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger = logging.getLogger(__name__)
    logger.info("🎯 Zone-Based Motorcycle Detection")
    logger.info(f"📹 Source: {args.source}")
    logger.info(f"🎮 Mode: {args.mode}")
    
    if args.duration:
        logger.info(f"⏱️  Duration: {args.duration}s")
    
    print("\n💡 Zone Detection Benefits:")
    print("   ✅ Better performance untuk area luas")
    print("   ✅ Fokus detection per area")
    print("   ✅ Reduce false positives")
    print("   ✅ Optimal untuk parking lot layout")
    print()
    
    # Run detection
    try:
        success = run_zone_detection(
            args.source, 
            args.mode, 
            args.duration,
            show_zones=not args.no_zones
        )
        if not success:
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Detection failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()