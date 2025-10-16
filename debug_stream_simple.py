#!/usr/bin/env python3
"""
Simple debug untuk test video stream
"""
import time
import logging
from config import CCTVConfig
from utils.video_stream import RTSPStream

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_rtsp_stream():
    """Test RTSP stream langsung"""
    logger.info("🧪 Testing RTSP stream directly...")

    stream = RTSPStream(CCTVConfig.DEFAULT_RTSP_URL, buffer_size=5)

    if stream.start():
        logger.info("✅ Stream started")

        # Test untuk 15 detik
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 15:
            ret, frame = stream.get_latest_frame()

            if ret and frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # Every 30 frames
                    logger.info(f"📊 Frame {frame_count}: {frame.shape}, Running: {stream.is_running()}")
            else:
                logger.debug("No frame received")

            time.sleep(0.1)  # Check every 100ms

        logger.info(f"🏁 Test completed: {frame_count} frames received")
        stream.stop()
    else:
        logger.error("❌ Failed to start stream")

if __name__ == "__main__":
    test_rtsp_stream()