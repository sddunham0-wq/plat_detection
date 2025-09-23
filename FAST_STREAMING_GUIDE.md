# 🚀 Fast Streaming Mode Guide

Guide untuk menggunakan mode streaming yang dioptimasi untuk performance tinggi dan latency rendah.

## 🎯 Overview

Fast Streaming Mode adalah versi optimasi dari sistem deteksi plat nomor yang dirancang khusus untuk:
- **Streaming real-time** dengan latency minimal
- **Performance tinggi** pada hardware terbatas  
- **Smooth video display** tanpa lag
- **Efficient resource usage**

## ⚡ Optimasi yang Diterapkan

### 1. **Video Processing Optimizations**
```python
# Resolution optimized untuk speed
FRAME_WIDTH = 480          # Reduced from 640
FRAME_HEIGHT = 360         # Reduced from 480
FPS_LIMIT = 25             # Increased from 10
BUFFER_SIZE = 5            # Reduced from 30
```

### 2. **Frame Skipping**
```python
ENABLE_FRAME_SKIPPING = True
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
```

### 3. **Disabled Heavy Features**
- ❌ Enhanced Detection (super-resolution, ensemble OCR)
- ❌ Object Tracking 
- ❌ Multiple PSM modes
- ❌ Adaptive enhancement

### 4. **Optimized Thresholds**
```python
MIN_CONFIDENCE = 40         # Lowered from 60
MIN_CONTOUR_AREA = 800     # Increased to skip small objects
MAX_THREADS = 2            # Reduced for stability
```

## 🚀 Usage

### Quick Start
```bash
# Default fast streaming (laptop camera)
python start_fast_streaming.py

# RTSP camera
python start_fast_streaming.py --source "rtsp://user:pass@ip:port/stream"

# USB camera
python start_fast_streaming.py --source 0

# Interactive camera selection
python start_fast_streaming.py --camera-select

# Auto-select best camera
python start_fast_streaming.py --auto-camera
```

### Command Options
```bash
--source            # Video source (RTSP URL, camera index, file)
--laptop-camera     # Use laptop built-in camera
--camera-select     # Interactive camera selection
--auto-camera       # Auto-select best available camera
--list-cameras      # List all available cameras
--normal-mode       # Disable fast mode optimizations
```

## 📊 Performance Comparison

| Mode | FPS | Latency | CPU Usage | Detection Quality |
|------|-----|---------|-----------|-------------------|
| **Normal** | 8-12 | 2-3s | High | Excellent |
| **Fast Streaming** | 20-25 | <1s | Medium | Good |

## 🎛️ Configuration Tweaks

### For Even Higher Performance
Edit `config.py`:
```python
# Extreme performance mode
CCTVConfig.PROCESS_EVERY_N_FRAMES = 3    # Skip more frames
CCTVConfig.BUFFER_SIZE = 3               # Minimal buffer
DetectionConfig.MIN_CONTOUR_AREA = 1000  # Skip smaller objects
```

### For Better Quality
```python
# Quality vs performance balance
CCTVConfig.PROCESS_EVERY_N_FRAMES = 1    # Process all frames
TesseractConfig.MIN_CONFIDENCE = 50      # Higher confidence
```

## 🔧 Troubleshooting

### Stream Still Slow?
1. **Reduce resolution further**:
   ```python
   FRAME_WIDTH = 320
   FRAME_HEIGHT = 240
   ```

2. **Increase frame skipping**:
   ```python
   PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
   ```

3. **Disable preview window**:
   ```python
   SystemConfig.SHOW_PREVIEW = False
   ```

### High CPU Usage?
1. **Use single thread**:
   ```python
   SystemConfig.MAX_THREADS = 1
   ```

2. **Increase minimum contour area**:
   ```python
   DetectionConfig.MIN_CONTOUR_AREA = 1500
   ```

### Missing Detections?
1. **Lower confidence threshold**:
   ```python
   TesseractConfig.MIN_CONFIDENCE = 30
   ```

2. **Reduce frame skipping**:
   ```python
   PROCESS_EVERY_N_FRAMES = 1
   ```

## 📈 Monitoring Performance

Fast Streaming Mode provides real-time statistics:
- **Current FPS**
- **Processed vs Skipped frames**
- **Processing time per frame**
- **Total detections**

Example output:
```
📊 Streaming FPS: 24.3, Processed: 1250, Skipped: 625, Detections: 45
```

## ⚠️ Trade-offs

### What You Gain:
- ✅ **Faster streaming** (2-3x FPS improvement)
- ✅ **Lower latency** (<1 second delay)
- ✅ **Smoother display** (no stuttering)
- ✅ **Better resource efficiency**

### What You Lose:
- ⚠️ **Slightly lower detection accuracy** (90% vs 95%)
- ⚠️ **No tracking capabilities**
- ⚠️ **Reduced enhancement features**
- ⚠️ **May miss very small/distant plates**

## 🔄 Switching Between Modes

### Enable Fast Mode (Default)
```bash
python start_fast_streaming.py
```

### Disable Fast Mode
```bash
python start_fast_streaming.py --normal-mode
```

### Original System
```bash
python main.py  # Uses original configuration
```

## 🎯 Best Use Cases

**Use Fast Streaming Mode for:**
- Real-time monitoring dashboards
- Live security systems
- Performance-critical applications
- Hardware-limited environments

**Use Normal Mode for:**
- Maximum detection accuracy
- Archival/analysis purposes
- When processing time is not critical
- Complex tracking requirements

## 📝 Notes

1. **Frame skipping** means some frames are displayed but not processed for detection
2. **Lower confidence thresholds** may increase false positives but improve detection rate
3. **Reduced resolution** affects detection of very small plates
4. All optimizations can be **easily reversed** by switching back to normal mode

## 🚀 Quick Performance Test

Test current setup:
```bash
# Test with laptop camera
python start_fast_streaming.py --laptop-camera

# Test with RTSP
python start_fast_streaming.py --source "your_rtsp_url"

# Compare with normal mode
python start_fast_streaming.py --normal-mode
```

Monitor the console output for FPS and performance metrics.