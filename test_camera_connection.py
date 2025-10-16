#!/usr/bin/env python3
"""
Test Camera Connection Script
Mencoba koneksi ke berbagai URL kamera dan menampilkan hasil
"""

import cv2
import time
from config import CCTVConfig

def test_camera_connection(url: str, timeout: int = 10) -> bool:
    """
    Test koneksi ke kamera dengan timeout

    Args:
        url: RTSP/HTTP URL
        timeout: Timeout dalam detik

    Returns:
        bool: True jika berhasil connect dan baca frame
    """
    print(f"Testing: {url}")
    try:
        cap = cv2.VideoCapture(url)

        # Set timeout
        start_time = time.time()

        if cap.isOpened():
            ret, frame = cap.read()
            elapsed = time.time() - start_time

            if ret and frame is not None:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  ✅ SUCCESS! ({elapsed:.2f}s)")
                print(f"     Frame: {width}x{height}")
                print(f"     FPS: {fps:.1f}")
                cap.release()
                return True
            else:
                print(f"  ❌ No frame received ({elapsed:.2f}s)")
        else:
            print(f"  ❌ Could not open stream")

        cap.release()
        return False

    except Exception as e:
        print(f"  ❌ Error: {str(e)[:60]}...")
        return False

def find_working_camera():
    """
    Cari kamera yang bekerja dari daftar URL

    Returns:
        str atau None: URL kamera yang bekerja
    """
    print("🎥 Testing Camera Connections...")
    print("=" * 50)

    # Primary URL (baru)
    primary_url = CCTVConfig.DEFAULT_RTSP_URL
    if test_camera_connection(primary_url):
        print(f"\n🎯 PRIMARY CAMERA WORKING: {primary_url}")
        return primary_url

    print(f"\n⚠️  Primary camera failed, trying fallbacks...")

    # Fallback URLs
    for i, url in enumerate(CCTVConfig.FALLBACK_RTSP_URLS, 1):
        print(f"\nFallback {i}:")
        if test_camera_connection(url):
            print(f"\n✅ FALLBACK CAMERA WORKING: {url}")
            return url

    print("\n❌ NO WORKING CAMERAS FOUND!")
    return None

def main():
    """Main testing function"""
    working_url = find_working_camera()

    if working_url:
        print(f"\n🎯 RECOMMENDED URL: {working_url}")

        # Tanya apakah ingin update config
        print("\nDo you want to start stream with this camera? (y/n): ", end="")
        try:
            response = input().lower()
            if response == 'y':
                print(f"\nStarting stream with: {working_url}")

                # Start basic stream preview
                cap = cv2.VideoCapture(working_url)
                print("Press 'q' to quit the preview...")

                frame_count = 0
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame!")
                        break

                    frame_count += 1

                    # Resize for display
                    if frame.shape[1] > 800:
                        height, width = frame.shape[:2]
                        new_width = 800
                        new_height = int(height * (new_width / width))
                        frame = cv2.resize(frame, (new_width, new_height))

                    # Add info overlay
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Press 'q' to quit", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow('Camera Test', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
                print("Stream stopped.")
        except KeyboardInterrupt:
            print("\nTest cancelled.")
    else:
        print("\n❌ No working cameras found. Please check:")
        print("   1. Camera is powered on")
        print("   2. Network connection is working")
        print("   3. IP address and credentials are correct")
        print("   4. RTSP service is running on camera")

if __name__ == "__main__":
    main()