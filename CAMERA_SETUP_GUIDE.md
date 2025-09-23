# 📹 Panduan Setup Kamera Laptop

## 🎯 Fitur Kamera yang Tersedia

Project ini sekarang sudah dilengkapi dengan fitur kamera laptop yang canggih:

### ✨ Fitur Utama
- **Auto-detection** kamera laptop built-in
- **Interactive selection** untuk memilih kamera
- **Auto-optimization** settings untuk detection
- **Quality testing** dan benchmarking
- **Cross-platform support** (macOS, Windows, Linux)

### 🚀 Command yang Tersedia

```bash
# 1. Gunakan laptop camera langsung
python3 main.py --laptop-camera

# 2. Menu interaktif pilih kamera
python3 main.py --camera-select

# 3. Auto-select kamera terbaik
python3 main.py --auto-camera

# 4. List semua kamera tersedia
python3 main.py --list-cameras

# 5. Info detail kamera tertentu
python3 main.py --camera-info 0

# 6. Test semua fitur camera
python3 test_camera_integration.py

# 7. Test kamera sederhana
python3 test_kamera.py
```

## 🔧 Setup Permission di macOS

### Masalah Umum
Jika Anda mendapat error:
```
OpenCV: camera access has been denied
```

### Solusi 1: System Preferences (Recommended)

1. **Buka System Preferences**
   - Klik Apple menu > System Preferences
   - Atau: System Settings (di macOS 13+)

2. **Ke Privacy & Security**
   - Pilih "Security & Privacy" 
   - Atau: "Privacy & Security" (macOS 13+)

3. **Aktifkan Camera Permission**
   - Klik tab "Privacy"
   - Pilih "Camera" dari daftar kiri
   - Centang kotak untuk **Terminal** atau **Python**

4. **Restart Terminal**
   - Tutup semua terminal
   - Buka terminal baru
   - Test lagi: `python3 test_kamera.py`

### Solusi 2: Command Line

```bash
# Reset camera permission
tccutil reset Camera

# Restart terminal lalu test
python3 test_kamera.py
```

### Solusi 3: Manual Permission Prompt

```bash
# Jalankan command ini untuk trigger permission prompt
python3 -c "import cv2; cv2.VideoCapture(0).read()"
```

## 🎥 Cara Menggunakan Kamera

### 1. Test Kamera Dulu
```bash
python3 test_kamera.py
```

### 2. Mulai Detection dengan Laptop Camera
```bash
# Basic detection
python3 main.py --laptop-camera

# Dengan enhanced detection
python3 main.py --laptop-camera --enhanced

# Tanpa preview window (lebih cepat)
python3 main.py --laptop-camera --no-preview

# Dengan motorcycle detection
python3 main.py --laptop-camera --motorcycle-mode
```

### 3. Interactive Camera Selection
```bash
python3 main.py --camera-select
```
Menu akan muncul untuk memilih kamera:
```
📷 CAMERA SELECTION MENU
============================================================
1. 🔴 Built-in Camera (Laptop Camera)
   Index: 0 | Resolution: 1280x720 | FPS: 30.0
   Backend: AVFoundation

2. 📹 External USB Camera (External Camera) 
   Index: 1 | Resolution: 640x480 | FPS: 15.0
   Backend: AVFoundation

0. Cancel
============================================================
Select camera (enter number): 1
```

### 4. List Semua Kamera
```bash
python3 main.py --list-cameras
```
Output:
```
📷 CAMERA DETECTION SUMMARY
============================================================
✅ Camera 0: Built-in Camera (FaceTime HD)
   Resolution: 1280x720
   FPS: 30.0
   Backend: AVFoundation

✅ Camera 1: USB Camera 1
   Resolution: 640x480
   FPS: 15.0
   Backend: AVFoundation

📊 Summary: 2/2 cameras available
============================================================
```

## ⚙️ Settings Optimization

### Preset Settings Available:
- **Default**: Balance antara quality dan performance
- **Quality**: High resolution, slower FPS
- **Performance**: Lower resolution, faster FPS  
- **Detection**: Optimized untuk plate detection

### Manual Configuration:
```python
from config import LaptopCameraConfig

# Ubah settings di config.py
LaptopCameraConfig.PREFERRED_RESOLUTION = (1280, 720)  # HD
LaptopCameraConfig.PREFERRED_FPS = 20  # Higher FPS
```

## 🐛 Troubleshooting

### Problem: "No cameras detected"
**Solution:**
1. Check camera permission di System Preferences
2. Restart terminal setelah enable permission
3. Test dengan aplikasi lain (Photo Booth, FaceTime)

### Problem: "Camera test failed"
**Solution:**
1. Tutup aplikasi lain yang pakai kamera
2. Restart Mac
3. Check connection kamera external

### Problem: "Low FPS atau lag"
**Solution:**
1. Gunakan performance mode: `--performance`
2. Reduce resolution di config
3. Close aplikasi berat lainnya

### Problem: "Dark/blurry image"
**Solution:**
1. Auto optimization sudah aktif untuk laptop camera
2. Manual adjustment brightness/contrast di code
3. Ensure good lighting

## 🎯 Contoh Usage

### Detection Plat Nomor dengan Laptop Camera
```bash
# Start detection
python3 main.py --laptop-camera

# Controls di preview window:
# 'q' - Quit
# 's' - Save screenshot  
# 'p' - Print statistics
```

### Batch Testing Multiple Cameras
```bash
# List cameras
python3 main.py --list-cameras

# Test setiap camera
python3 main.py --camera-info 0
python3 main.py --camera-info 1

# Auto-select terbaik
python3 main.py --auto-camera
```

## 📊 Performance Tips

1. **Optimal Resolution**: 640x480 untuk detection speed
2. **Quality Mode**: 1280x720 untuk accuracy
3. **Buffer Size**: Default 30 frames (bisa dikurangi)
4. **FPS Limit**: 10-15 FPS optimal untuk detection

## 🎉 Success Indicators

Jika setup berhasil, Anda akan melihat:
```
✅ Camera opened successfully!
📹 Camera Info:
   Resolution: 1280x720
   FPS: 30.0
✅ Frame capture successful!
🎉 KAMERA LAPTOP BERHASIL!
```

Kemudian bisa langsung pakai:
```bash
python3 main.py --laptop-camera
```

Dan Anda akan melihat:
```
📹 Found laptop camera: Built-in Camera (FaceTime HD)
🔍 Testing laptop camera...
✅ Laptop camera ready: Built-in Camera (FaceTime HD)
🚀 Live detection system started successfully!
```