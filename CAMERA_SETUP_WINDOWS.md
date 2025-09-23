# 📹 Panduan Setup Kamera di Windows

## 🎯 Setup Kamera Laptop untuk Windows

### 🔧 **LANGKAH-LANGKAH MENGAKTIFKAN KAMERA DI WINDOWS:**

#### **1. Check Camera Permission di Windows Settings:**
1. **Buka Windows Settings** (tekan `Win + I`)
2. **Pilih Privacy & Security** 
3. **Klik 'Camera' di panel kiri**
4. **Pastikan:**
   - "Camera access for this device" = **ON**
   - "Let apps access your camera" = **ON**
   - "Let desktop apps access your camera" = **ON**

#### **2. Check Camera di Device Manager:**
1. **Klik kanan 'This PC'** > Properties > Device Manager
2. **Expand 'Cameras' atau 'Imaging devices'**
3. **Pastikan kamera terlihat dan tidak ada tanda warning**
4. **Jika ada masalah:** Klik kanan > Update driver

#### **3. Test Kamera dengan Windows Camera App:**
1. **Buka 'Camera' app dari Start Menu**
2. **Pastikan kamera berfungsi normal**
3. **Jika tidak berfungsi, fix dulu di sini**

### 🐍 **Setup Python Environment di Windows:**

#### **Install Dependencies:**
```bash
# Install OpenCV untuk Windows
pip install opencv-python

# Install requirements project
pip install -r requirements.txt
```

#### **Test Kamera Python:**
```bash
# Test sederhana
python test_kamera.py

# Atau manual test:
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK!' if cap.isOpened() else 'Camera Failed')"
```

### 🚀 **Command untuk Windows:**

```bash
# 1. Test kamera dulu
python test_kamera.py

# 2. List kamera tersedia
python main.py --list-cameras

# 3. Gunakan laptop camera
python main.py --laptop-camera

# 4. Interactive selection
python main.py --camera-select

# 5. Auto-select terbaik
python main.py --auto-camera
```

### 🔧 **Troubleshooting Windows:**

#### **Problem: "Camera access denied"**
**Solution:**
1. Check Windows Privacy Settings (langkah 1 di atas)
2. Restart aplikasi yang menggunakan kamera
3. Update Windows dan driver kamera

#### **Problem: "No camera detected"**
**Solution:**
1. Check Device Manager
2. Test dengan Windows Camera app
3. Reconnect USB camera (jika external)

#### **Problem: "OpenCV error"**
**Solution:**
```bash
# Reinstall OpenCV
pip uninstall opencv-python
pip install opencv-python

# Atau specific version
pip install opencv-python==4.8.0.76
```

#### **Problem: "Python not found"**
**Solution:**
1. Install Python dari python.org
2. Atau gunakan: `py` instead of `python`
```bash
py test_kamera.py
py main.py --laptop-camera
```

### 🎥 **Windows-Specific Camera Settings:**

#### **Optimal Settings untuk Windows:**
- **Resolution**: 640x480 (untuk speed) atau 1280x720 (untuk quality)
- **FPS**: 15-30 FPS
- **Backend**: DirectShow (default Windows)

#### **Manual Camera Configuration:**
```python
# Di config.py, sesuaikan untuk Windows:
class LaptopCameraConfig:
    PREFERRED_RESOLUTION = (640, 480)  # Windows optimal
    PREFERRED_FPS = 20                 # Windows smooth FPS
    WINDOWS_DIRECTSHOW = True          # Use DirectShow backend
```

### 🏃 **Quick Start Windows:**

1. **Setup Permission:**
   - Windows Settings > Privacy > Camera > Enable all

2. **Test:**
   ```bash
   python test_kamera.py
   ```

3. **Start Detection:**
   ```bash
   python main.py --laptop-camera
   ```

### 📱 **Multiple Camera Support:**

```bash
# Check semua kamera
python main.py --list-cameras

# Test kamera specific
python main.py --camera-info 0  # Built-in camera
python main.py --camera-info 1  # External camera

# Auto-select terbaik
python main.py --auto-camera
```

### 🎯 **Expected Output di Windows:**

```
📹 Found laptop camera: Integrated Webcam
🔍 Testing laptop camera...
✅ Laptop camera ready: Integrated Webcam
🚀 Live detection system started successfully!

🎥 LIVE PLATE DETECTION SYSTEM
============================================================
📹 Source: 0
⌨️  Controls:
   'q' - Quit
   's' - Save screenshot
   'p' - Print statistics
============================================================
```

### ⚡ **Performance Tips Windows:**

1. **Close other camera apps** (Skype, Teams, dll)
2. **Use dedicated USB 3.0 port** untuk external camera
3. **Update graphics drivers** untuk better performance
4. **Adjust Windows power settings** ke High Performance

### 🔗 **Integration dengan Project:**

Project ini sudah support cross-platform, jadi semua fitur yang sama tersedia di Windows:

- ✅ Auto-detection kamera
- ✅ Interactive selection  
- ✅ Quality optimization
- ✅ Performance tuning
- ✅ Multiple camera support

Tinggal jalankan command yang sama seperti di dokumentasi utama!