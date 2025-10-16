#!/usr/bin/env python3
"""
Minimal debug untuk test hanging issue
"""
import cv2
import time
import logging
from config import CCTVConfig
from utils.video_stream import RTSPStream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_minimal_processing():
    """Test minimal frame processing"""
    logger.info("🧪 Testing minimal frame processing...")

    stream = RTSPStream(CCTVConfig.DEFAULT_RTSP_URL, buffer_size=5)

    if stream.start():
        logger.info("✅ Stream started")

        # Test processing untuk 15 detik
        start_time = time.time()
        frame_count = 0
        processing_times = []

        while time.time() - start_time < 15:
            ret, frame = stream.get_latest_frame()

            if ret and frame is not None:
                process_start = time.time()

                # Minimal processing - just convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Simple contour detection
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                process_time = time.time() - process_start
                processing_times.append(process_time)

                frame_count += 1

                if frame_count % 30 == 0:  # Every 30 frames
                    avg_time = sum(processing_times[-30:]) / min(30, len(processing_times))
                    logger.info(f"📊 Frame {frame_count}: {len(contours)} contours, Avg time: {avg_time:.3f}s")

            time.sleep(0.1)  # Check every 100ms

        logger.info(f"🏁 Test completed: {frame_count} frames processed")
        stream.stop()
    else:
        logger.error("❌ Failed to start stream")

if __name__ == "__main__":
    test_minimal_processing()