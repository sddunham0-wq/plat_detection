# 🚗 Live CCTV License Plate Detection System

Sistem deteksi plat nomor kendaraan secara **real-time** menggunakan OpenCV dan Tesseract OCR. Mendukung CCTV IP camera, webcam, dan file video.

## ✨ Fitur Utama

- 🎥 **Multi-sumber video**: RTSP IP camera, USB webcam, file video
- 🔍 **Real-time detection**: Deteksi plat nomor live dengan multi-threading
- 🤖 **YOLOv8 Object Detection**: Deteksi kendaraan dan objek lain secara real-time
- 💾 **Database otomatis**: Simpan hasil deteksi dengan timestamp
- 🖼️ **Preview window**: Live preview dengan bounding boxes
- 📊 **Statistik real-time**: FPS, jumlah deteksi, success rate
- 🚨 **Alert system**: Watchlist dan blacklist plat nomor
- 📁 **Auto-save gambar**: Simpan gambar plat yang terdeteksi
- ⚙️ **ROI (Region of Interest)**: Fokus deteksi pada area tertentu
- 🌐 **Web Interface**: Control via browser dengan live streaming
- 🔧 **Konfigurasi lengkap**: Setting untuk berbagai kebutuhan

## 🛠️ Instalasi

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Tesseract OCR
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt install tesseract-ocr

# Windows
# Download dari: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Verifikasi Setup

```bash
python3 test_setup.py
```

Output yang diharapkan:
```
🎉 ALL TESTS PASSED!
System is ready for live plate detection!
```

## 🚀 Cara Penggunaan

### Web Interface Mode (Recommended)

#### 🚀 Quick Start dengan YOLOv8 Object Detection
```bash
# Super fast startup dengan auto-detection
./start_yolo.sh

# Atau manual:
python3 start_with_yolo.py

# Kemudian buka browser: http://localhost:5000
```

#### 🎛️ Manual Control
```bash
# Start web server (YOLOv8 background loading)
python3 headless_stream.py --host 0.0.0.0 --port 5000

# Start tanpa YOLOv8 (fastest)
python3 headless_stream.py --no-yolo

# Quick start original
./start_headless.sh
```

### CLI Mode (Traditional)

```bash
# Webcam (default)
python3 main.py --source 0

# RTSP IP Camera
python3 main.py --source "rtsp://username:password@192.168.1.100:554/stream1"

# Video file
python3 main.py --source "test_video.mp4"

# Tanpa preview window (untuk server)
python3 main.py --source 0 --no-preview
```

### Advanced Usage

```bash
# Web server dengan custom config
python3 headless_stream.py --source "test_video.mp4" --host 0.0.0.0 --port 8080

# CLI dengan log level
python3 main.py --source 0 --log-level DEBUG

# Help
python3 headless_stream.py --help
python3 main.py --help
```

### Controls (Live Preview)

- **'q'** - Quit/keluar program
- **'s'** - Save screenshot
- **'p'** - Print statistics

## ⚙️ Konfigurasi

### File `config.py`

Semua pengaturan ada di `config.py`:

#### CCTV Settings
```python
class CCTVConfig:
    FRAME_WIDTH = 640          # Resolusi processing
    FRAME_HEIGHT = 480
    FPS_LIMIT = 10             # Maksimal FPS
    BUFFER_SIZE = 30           # Buffer frame
```

#### Tesseract Settings
```python
class TesseractConfig:
    # Auto-detect path, atau set manual:
    TESSERACT_PATH = '/opt/homebrew/bin/tesseract'
    
    # Config untuk plat Indonesia
    OCR_CONFIG = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    MIN_CONFIDENCE = 60
```

#### Detection Settings
```python
class DetectionConfig:
    MIN_ASPECT_RATIO = 2.0     # Ratio plat nomor
    MAX_ASPECT_RATIO = 6.0
    ROI_AREA = (0.1, 0.3, 0.8, 0.4)  # Focus area (x%, y%, w%, h%)
    DUPLICATE_THRESHOLD = 5    # Detik anti-duplicate
```

#### Alert Settings
```python
class AlertConfig:
    ENABLE_ALERTS = True
    
    # Plat yang perlu dimonitor
    WATCHLIST_PLATES = [
        "B1234ABC",
        "D5678XYZ"
    ]
    
    # Plat yang diblacklist
    BLACKLIST_PLATES = [
        "B9999XXX"
    ]
```

## 📁 Struktur Project

```
project-plat-detection-dude/
├── main.py                    # Program utama (CLI mode)
├── headless_stream.py         # Web server mode
├── config.py                  # Konfigurasi sistem
├── database.py                # Handler database
├── display_manager.py         # Thread-safe display manager
├── stream_manager.py          # Headless video stream manager
├── requirements.txt           # Python dependencies
├── start_headless.sh          # Quick start script
├── test_video.mp4            # Test video file
├── README.md                 # Dokumentasi ini
├── SCREENSHOT_FEATURE.md     # Dokumentasi fitur screenshot
├── templates/
│   └── stream.html           # Web interface template
├── utils/
│   ├── video_stream.py       # Handler video streams
│   ├── plate_detector.py     # Deteksi plat nomor
│   └── frame_processor.py    # Multi-threading processor
├── detected_plates/          # Gambar hasil deteksi (auto-created)
├── logs/                     # Log files (auto-created)
└── detected_plates.db        # SQLite database (auto-created)
```

## 💾 Database

System otomatis menyimpan hasil deteksi ke SQLite database dengan schema:

```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    plate_text TEXT,
    confidence REAL,
    image_path TEXT,
    source_info TEXT,
    processed_time REAL
);
```

### Query Database

```python
from database import PlateDatabase

db = PlateDatabase()

# Get deteksi terbaru
recent = db.get_recent_detections(100)

# Search plat tertentu
results = db.search_plates(plate_text=\"B1234\")

# Statistik
stats = db.get_statistics(days=7)

# Export ke CSV
db.export_to_csv(\"results.csv\")
```

## 🎯 Optimasi Performance

### 1. ROI (Region of Interest)
Set area fokus deteksi di `config.py`:
```python
ROI_AREA = (0.1, 0.3, 0.8, 0.4)  # x%, y%, width%, height%
```

### 2. Threading Configuration
```python
class SystemConfig:
    MAX_THREADS = 4              # Processing threads
    MEMORY_LIMIT_MB = 512        # Memory limit
```

### 3. Video Resolution
```python
class CCTVConfig:
    FRAME_WIDTH = 640            # Lower = faster
    FRAME_HEIGHT = 480
    FPS_LIMIT = 10               # Lower = less CPU
```

### 4. OCR Settings
```python
class DetectionConfig:
    MIN_CONTOUR_AREA = 1000      # Skip small objects
    DUPLICATE_THRESHOLD = 5       # Avoid duplicate detections
```

## 📊 Output Contoh

### Console Output
```
🎥 LIVE PLATE DETECTION SYSTEM
============================================================
📹 Source: rtsp://192.168.1.100/stream1
⌨️  Controls:
   'q' - Quit
   's' - Save screenshot
   'p' - Print statistics
============================================================

🚗 DETECTED: B1234ABC (confidence: 89.2%) [ID: 1]
🚗 DETECTED: D5678XYZ (confidence: 92.1%) [ID: 2]

📊 SYSTEM STATISTICS
============================================================
⏱️  Runtime: 120.5s
🎬 Total Frames: 1205
📈 FPS: 10.0
🚗 Total Detections: 15
⚙️  Processing FPS: 8.2
⏳ Avg Processing Time: 0.122s
🎯 OCR Success Rate: 87.3%
============================================================
```

### File Output
- **detected_plates.db** - SQLite database dengan semua deteksi
- **detected_plates/** - Folder gambar plat yang terdeteksi
- **logs/** - Log files sistem
- **screenshot_YYYYMMDD_HHMMSS.jpg** - Screenshot manual

## 🔧 Troubleshooting

### Tesseract Error
```bash
# Check Tesseract installation
tesseract --version

# Update path di config.py
TESSERACT_PATH = '/usr/local/bin/tesseract'  # Linux
TESSERACT_PATH = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Windows
```

### RTSP Connection Failed
```python
# Di config.py, adjust timeout
class CCTVConfig:
    RTSP_TIMEOUT = 20          # Increase timeout
    MAX_RECONNECT_ATTEMPTS = 5  # More retry attempts
```

### Low Detection Rate
1. **Adjust ROI** - Set area fokus yang tepat
2. **Check lighting** - Pastikan pencahayaan cukup
3. **Lower confidence** - Set `MIN_CONFIDENCE = 50`
4. **Adjust aspect ratio** - Sesuaikan dengan jenis plat

### Performance Issues
1. **Reduce resolution** - Lower `FRAME_WIDTH/HEIGHT`
2. **Limit FPS** - Set `FPS_LIMIT = 5`
3. **Reduce threads** - Set `MAX_THREADS = 2`
4. **Disable preview** - Gunakan `--no-preview`

## 📈 Development

### Extend Detection
```python
# Custom detector di utils/plate_detector.py
class CustomPlateDetector(LicensePlateDetector):
    def custom_preprocessing(self, image):
        # Your custom image processing
        return processed_image
```

### Custom Alerts
```python
# Custom callback di main.py
def custom_detection_callback(detections):
    for det in detections:
        if det.confidence > 95:
            send_webhook_alert(det.text)
```

### Database Extensions
```python
# Custom database operations
class ExtendedDatabase(PlateDatabase):
    def add_location_info(self, detection, latitude, longitude):
        # Add GPS coordinates
        pass
```

## 📝 License

MIT License - Bebas digunakan untuk project apapun.

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📞 Support

Jika ada masalah atau pertanyaan:
1. Check troubleshooting section
2. Run `python3 test_setup.py` untuk diagnosis
3. Check log files di folder `logs/`
4. Create issue di repository

---

**Happy Detecting! 🚗✨**