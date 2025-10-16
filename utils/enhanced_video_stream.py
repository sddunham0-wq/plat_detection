#!/usr/bin/env python3
"""
Enhanced Video Stream dengan Smart Connection Management
Dapat mencoba berbagai URL format dan fallback secara otomatis
"""

import cv2
import time
import logging
from typing import List, Optional, Tuple
from config import CCTVConfig

class EnhancedVideoStream:
    """
    Enhanced VideoStream dengan smart URL detection dan auto-fallback
    """

    def __init__(self, primary_url: str = None, timeout: int = 10):
        """
        Initialize enhanced video stream

        Args:
            primary_url: Primary URL to try first
            timeout: Connection timeout per URL
        """
        self.primary_url = primary_url or CCTVConfig.DEFAULT_RTSP_URL
        self.timeout = timeout
        self.current_url = None
        self.cap = None
        self.logger = logging.getLogger(__name__)

        # Build comprehensive URL list
        self.url_candidates = self._build_url_candidates()

    def _build_url_candidates(self) -> List[str]:
        """
        Build comprehensive list of URL candidates untuk testing
        """
        candidates = [self.primary_url]

        # Add fallback URLs dari config
        candidates.extend(CCTVConfig.FALLBACK_RTSP_URLS)

        # Generate additional candidates berdasarkan IP detection
        base_urls = self._generate_base_url_variants(self.primary_url)
        candidates.extend(base_urls)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for url in candidates:
            if url not in seen:
                seen.add(url)
                unique_candidates.append(url)

        return unique_candidates

    def _generate_base_url_variants(self, url: str) -> List[str]:
        """
        Generate URL variants berdasarkan detected IP dan pattern
        """
        variants = []

        try:
            # Extract IP dari URL
            if 'rtsp://' in url:
                # Extract IP dan credentials
                import re
                match = re.search(r'rtsp://(.+?)@(\d+\.\d+\.\d+\.\d+):(\d+)', url)
                if match:
                    credentials, ip, port = match.groups()

                    # Generate common RTSP paths untuk IP cameras
                    common_paths = [
                        f'rtsp://{credentials}@{ip}:{port}/',
                        f'rtsp://{credentials}@{ip}:{port}/stream',
                        f'rtsp://{credentials}@{ip}:{port}/live',
                        f'rtsp://{credentials}@{ip}:{port}/cam/realmonitor?channel=1&subtype=0',
                        f'rtsp://{credentials}@{ip}:{port}/cam/realmonitor?channel=1&subtype=1',
                        f'rtsp://{credentials}@{ip}:{port}/h264',
                        f'rtsp://{credentials}@{ip}:{port}/video1',
                        f'rtsp://{credentials}@{ip}:554/cam/realmonitor?channel=1&subtype=0',  # Standard port fallback
                    ]

                    variants.extend(common_paths)

        except Exception as e:
            self.logger.debug(f"Error generating URL variants: {e}")

        return variants

    def connect(self) -> bool:
        """
        Smart connection dengan auto-fallback ke working URL

        Returns:
            bool: True if connection successful
        """
        self.logger.info("🔍 Starting smart video connection...")

        for i, url in enumerate(self.url_candidates, 1):
            self.logger.info(f"Trying URL {i}/{len(self.url_candidates)}: {url[:50]}...")

            if self._test_single_url(url):
                self.current_url = url
                self.logger.info(f"✅ Successfully connected to: {url}")
                return True

        self.logger.error("❌ Failed to connect to any URL")
        return False

    def _test_single_url(self, url: str) -> bool:
        """
        Test single URL connection

        Args:
            url: URL to test

        Returns:
            bool: True if successful
        """
        try:
            # Use enhanced OpenCV settings
            if url.startswith('rtsp://'):
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(url)

            # Apply optimized settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 25)

            if not cap.isOpened():
                cap.release()
                return False

            # Try to read frame dengan timeout
            start_time = time.time()
            success = False

            for attempt in range(3):  # Try 3 times
                ret, frame = cap.read()

                if ret and frame is not None and frame.size > 0:
                    # Verify frame quality
                    height, width = frame.shape[:2]
                    if width > 100 and height > 100:  # Reasonable size
                        success = True
                        break

                # Check timeout
                if time.time() - start_time > self.timeout:
                    break

                time.sleep(0.3)

            # Store successful connection
            if success:
                self.cap = cap
                return True
            else:
                cap.release()
                return False

        except Exception as e:
            self.logger.debug(f"Connection error: {e}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """
        Read frame dari active connection

        Returns:
            Tuple of (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None

        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return False, None

    def get_stream_info(self) -> dict:
        """
        Get information about current stream

        Returns:
            dict: Stream information
        """
        if not self.cap:
            return {'connected': False}

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            return {
                'connected': True,
                'url': self.current_url,
                'width': width,
                'height': height,
                'fps': fps,
                'backend': 'FFMPEG' if self.current_url.startswith('rtsp://') else 'DEFAULT'
            }
        except Exception as e:
            return {'connected': False, 'error': str(e)}

    def reconnect(self) -> bool:
        """
        Reconnect using current URL atau try alternatives

        Returns:
            bool: True if reconnection successful
        """
        self.logger.info("🔄 Attempting to reconnect...")

        # Close current connection
        if self.cap:
            self.cap.release()
            self.cap = None

        # Try current URL first
        if self.current_url and self._test_single_url(self.current_url):
            self.logger.info("✅ Reconnected to same URL")
            return True

        # Try alternatives
        return self.connect()

    def close(self):
        """Close video stream connection"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_url = None
        self.logger.info("📡 Video stream closed")

def test_enhanced_stream():
    """Test function untuk enhanced stream"""
    print("🧪 Testing Enhanced Video Stream...")

    # Test dengan primary URL
    stream = EnhancedVideoStream()

    if stream.connect():
        info = stream.get_stream_info()
        print(f"✅ Connected!")
        print(f"   URL: {info.get('url', 'Unknown')}")
        print(f"   Resolution: {info.get('width')}x{info.get('height')}")
        print(f"   FPS: {info.get('fps', 0):.1f}")
        print(f"   Backend: {info.get('backend', 'Unknown')}")

        # Test frame reading
        print("\\n📸 Testing frame reading...")
        for i in range(5):
            ret, frame = stream.read_frame()
            if ret and frame is not None:
                print(f"   Frame {i+1}: {frame.shape}")
            else:
                print(f"   Frame {i+1}: Failed")
            time.sleep(0.2)

        stream.close()
    else:
        print("❌ Failed to connect")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_enhanced_stream()