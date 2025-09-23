# 📷 Multi-Camera Setup Guide

Panduan lengkap untuk menggunakan multiple cameras (termasuk laptop camera) dengan Live CCTV Plate Detection System.

## ✅ Prerequisites

### 1. Camera Permissions (macOS)
Sistem memerlukan camera permission untuk mengakses laptop camera:

```bash
# Reset camera permissions (if needed)
tccutil reset Camera

# Or grant permission melalui System Preferences:
# System Preferences > Security & Privacy > Camera > Allow Terminal/Python
```

### 2. Required Dependencies
```bash
# Install OpenCV dengan video support
pip install opencv-python

# Install additional dependencies jika belum ada
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Test Camera Discovery
```bash
# Test apakah cameras terdeteksi
python3 test_multi_camera.py
```

### 2. Start Web Server dengan Multi-Camera
```bash
# Start web server
python3 headless_stream.py

# Open browser
open http://localhost:5000
```

### 3. Enable Multi-Camera Mode
1. Di web interface, toggle **"Multi Camera"** switch
2. Click **"Discover Cameras"** button
3. Select cameras yang ingin digunakan
4. Click **"Start Multi"**

## 📋 Features Overview

### ✨ What's New in Multi-Camera
- **🔍 Auto Camera Discovery**: Otomatis detect semua available cameras
- **📷 Laptop Camera Support**: Built-in camera laptop bisa digunakan  
- **🎯 Multi-Source Detection**: Parallel plate detection dari multiple cameras
- **🌐 Web Interface**: User-friendly controls untuk manage cameras
- **📊 Real-time Monitoring**: Live preview dari semua active cameras
- **⚡ Performance Optimized**: Multi-threading untuk smooth operation

### 🎨 Web Interface Components
- **Camera Mode Toggle**: Switch antara single/multi camera mode
- **Camera Discovery**: Auto-detect available cameras di system
- **Camera Selection**: Pilih cameras yang ingin digunakan (max 4)
- **Grid View**: 2x2 grid layout untuk multiple camera feeds
- **Real-time Stats**: Individual stats per camera

## 🔧 Configuration

### Camera Settings (config.py)
```python
class MultiCameraConfig:
    # Multi-camera settings
    ENABLE_MULTI_CAMERA = True           
    MAX_CAMERAS = 4                      
    AUTO_DISCOVER_CAMERAS = True         
    
    # Default laptop camera config
    DEFAULT_LAPTOP_CONFIG = {
        'resolution': (640, 480),
        'fps_limit': 10,
        'auto_exposure': True,
        'buffer_size': 30
    }
    
    # Performance settings
    PARALLEL_DETECTION = True           
    CROSS_CAMERA_DEDUPLICATION = True  
```

## 📖 Usage Examples

### 1. CLI Mode - Multiple Cameras
```bash
# Laptop camera + RTSP camera
python3 main.py --source 0  # Laptop camera (single mode)

# Untuk multi-camera, gunakan web interface
```

### 2. Web Mode - Recommended
```bash
# Start server
python3 headless_stream.py

# Navigate to http://localhost:5000
# Toggle "Multi Camera" mode
# Click "Discover Cameras"
# Select desired cameras
# Click "Start Multi"
```

### 3. Programmatic Usage
```python
from utils.multi_camera_stream import MultiCameraStream
from utils.camera_manager import CameraManager

# Discover cameras
camera_mgr = CameraManager()
cameras = camera_mgr.enumerate_cameras()

# Create multi-stream
multi_stream = MultiCameraStream()

# Add laptop camera
multi_stream.add_laptop_camera("laptop")

# Add RTSP camera
multi_stream.add_rtsp_camera("rtsp_cam", 
    "rtsp://admin:pass@168.1.195:554")

# Add callbacks
def on_detection(camera_id, detections):
    for det in detections:
        print(f"Detection from {camera_id}: {det.text}")

multi_stream.add_detection_callback(on_detection)

# Start streaming
multi_stream.start()
```

## 🛠️ Troubleshooting

### Camera Permission Issues (macOS)
```bash
# Problem: "OpenCV: camera access has been denied"
# Solution 1: Reset camera permissions
tccutil reset Camera

# Solution 2: Grant permission via System Preferences
# Go to: System Preferences > Security & Privacy > Camera
# Check the box next to Terminal or your Python app
```

### Camera Not Detected
```bash
# Check available cameras manually
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"
```

### Performance Issues
```python
# Reduce resolution in config.py
DEFAULT_LAPTOP_CONFIG = {
    'resolution': (320, 240),  # Lower resolution
    'fps_limit': 5,            # Lower FPS
}

# Or disable some features
PARALLEL_DETECTION = False
CROSS_CAMERA_DEDUPLICATION = False
```

### Web Interface Not Loading
```bash
# Check if server is running
curl http://localhost:5000/api/stats

# Check firewall settings
# Ensure port 5000 is not blocked
```

## 📊 API Reference

### Multi-Camera Endpoints

#### Discover Cameras
```bash
POST /api/cameras/discover
Response: {
  "success": true,
  "cameras": [
    {
      "index": 0,
      "name": "FaceTime HD Camera",
      "resolution": [1280, 720],
      "fps": 30.0,
      "backend": "AVFoundation",
      "available": true
    }
  ]
}
```

#### Test Camera
```bash
POST /api/cameras/test/{camera_index}
Response: {
  "success": true,
  "message": "Camera test passed"
}
```

#### Start Multi-Stream
```bash
POST /api/multi_stream/start
Body: {
  "cameras": [
    {
      "id": "laptop",
      "name": "Laptop Camera", 
      "source": 0,
      "resolution": [640, 480],
      "fps_limit": 10
    }
  ]
}
```

#### Get Camera Frame
```bash
GET /api/multi_stream/camera/{camera_id}/latest
Response: {
  "success": true,
  "camera_id": "laptop",
  "frame_base64": "...",
  "detections": [...]
}
```

## 🎯 Best Practices

### 1. Camera Setup
- **Laptop Camera**: Best for close-range detection (documents, testing)
- **RTSP Camera**: Best for live monitoring (parking, entrance)
- **USB Camera**: Good for specific angles/positions

### 2. Performance Optimization
- Use appropriate resolution (640x480 for most cases)
- Limit FPS to 5-10 for laptop cameras
- Enable ROI (Region of Interest) untuk focus area
- Use parallel detection untuk multiple cameras

### 3. Detection Strategy
- Laptop camera: Good untuk testing dan demo
- RTSP camera: Primary untuk production monitoring  
- Cross-camera deduplication: Avoid duplicate alerts

### 4. Resource Management
- Monitor CPU usage dengan multiple cameras
- Adjust FPS based on system performance
- Use lower resolution untuk resource-constrained systems

## 📈 Performance Metrics

### Expected Performance
- **Single Laptop Camera**: 5-10 FPS, 10-30% CPU
- **Dual Camera (Laptop + RTSP)**: 3-8 FPS, 20-50% CPU  
- **Quad Camera**: 2-5 FPS, 40-80% CPU

### Resource Requirements
- **Memory**: ~100-200MB per active camera
- **CPU**: ~10-25% per camera (depending on resolution)
- **Network**: ~1-5 Mbps per RTSP camera

## 🔗 Integration Examples

### With Existing Single-Camera Code
```python
# Existing single camera code works as-is
python3 main.py --source 0

# New multi-camera mode via web interface
python3 headless_stream.py
# Then use web UI for multi-camera
```

### Custom Detection Handlers
```python
def handle_detection(camera_id, detections):
    for detection in detections:
        print(f"📱 [{camera_id}] Plate: {detection.text}")
        
        # Custom logic per camera
        if camera_id == "laptop":
            # Handle laptop camera detections
            save_to_local_db(detection)
        elif camera_id == "rtsp_main":
            # Handle RTSP camera detections  
            send_security_alert(detection)

multi_stream.add_detection_callback(handle_detection)
```

## ✅ Success Checklist

- [ ] Camera permissions granted
- [ ] Dependencies installed
- [ ] Camera discovery working
- [ ] Web interface accessible
- [ ] Multi-camera mode enabled
- [ ] Cameras detected and selected
- [ ] Live streaming functional
- [ ] Plate detection working
- [ ] Performance acceptable

## 💡 Tips & Tricks

1. **Testing**: Use `test_multi_camera.py` untuk verify setup
2. **Debugging**: Check browser console untuk detailed errors
3. **Performance**: Start dengan single camera, then add more
4. **Quality**: Ensure good lighting untuk better detection
5. **Positioning**: Angle cameras untuk optimal plate visibility

---

**🎉 Congratulations!** You now have multi-camera support integrated with your Live CCTV Plate Detection system. The laptop camera can work alongside RTSP cameras for comprehensive monitoring.