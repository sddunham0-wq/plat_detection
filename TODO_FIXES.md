# TODO: Fix Web Streaming Issues

## ✅ Completed Fixes
- [x] Improved error handling in `/api/start_stream` endpoint
- [x] Added `/api/test_source` endpoint for testing video sources
- [x] Added test button in web interface
- [x] Better error messages and logging

## 🔄 In Progress
- [ ] Test the web interface with current fixes
- [ ] Verify RTSP connection issues
- [ ] Add fallback to webcam if RTSP fails

## 📋 Remaining Tasks
- [ ] Fix RTSP connection (URL: rtsp://admin:H4nd4l9165!@192.168.1.195:554/85)
- [ ] Add webcam fallback option (camera index 0)
- [ ] Test with different video sources
- [ ] Improve error recovery mechanisms
- [ ] Add connection retry logic

## 🧪 Testing Steps
1. Start the headless server: `python headless_stream.py`
2. Open browser to http://localhost:5010
3. Test different video sources:
   - RTSP URL: rtsp://admin:H4nd4l9165!@192.168.1.195:554/85
   - Webcam: 0
   - Test video file if available
4. Verify error messages are clear
5. Check if stream starts successfully

## 🔍 Investigation Notes
- RTSP connection fails with current URL
- Need to verify camera IP and credentials
- Consider adding multiple RTSP URL options
- Web interface should show clear error when source unavailable
