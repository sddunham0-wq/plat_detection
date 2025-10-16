#!/usr/bin/env python3
"""
Debug script untuk test YOLO detection issue
"""
import cv2
import time
import logging
from config import CCTVConfig
from stream_manager import HeadlessStreamManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yolo_detection():
    """Test YOLO detection dengan debugging"""
    logger.info("🧪 Testing YOLO detection...")

    # Initialize stream manager dengan YOLO enabled
    manager = HeadlessStreamManager(
        source=CCTVConfig.DEFAULT_RTSP_URL,
        enable_yolo=True,
        enable_tracking=False  # Disable tracking untuk simplicity
    )

    frame_count = 0
    detection_count = 0

    def frame_callback(stream_frame):
        nonlocal frame_count, detection_count
        frame_count += 1

        if stream_frame.detections:
            detection_count += 1
            logger.info(f"✅ Frame {frame_count}: {len(stream_frame.detections)} plate detections")
            for det in stream_frame.detections:
                logger.info(f"   📋 {det['text']} (confidence: {det['confidence']:.1f}%)")

        if stream_frame.object_detections:
            logger.info(f"🚗 Frame {frame_count}: {len(stream_frame.object_detections)} vehicle detections")

        if frame_count % 30 == 0:  # Every 30 frames
            logger.info(f"📊 Progress: Frame {frame_count}, Detections: {detection_count}, FPS: {stream_frame.fps:.1f}")

    manager.add_frame_callback(frame_callback)

    try:
        if manager.start():
            logger.info("✅ Stream started successfully")
            logger.info("🔍 Running for 30 seconds...")

            start_time = time.time()
            while time.time() - start_time < 30:
                time.sleep(1)

                # Log statistics every 10 seconds
                if int(time.time() - start_time) % 10 == 0:
                    stats = manager.get_statistics()
                    logger.info(f"📈 Stats: {stats.get('total_frames', 0)} frames, "
                              f"{stats.get('total_detections', 0)} total detections, "
                              f"FPS: {stats.get('fps', 0):.1f}")

            logger.info(f"🏁 Test completed - Total frames: {frame_count}, Total detections: {detection_count}")

        else:
            logger.error("❌ Failed to start stream")

    except KeyboardInterrupt:
        logger.info("⏹️ Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        manager.stop()
        logger.info("🔚 Stream stopped")

if __name__ == "__main__":
    test_yolo_detection()