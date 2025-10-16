# 🚗 Vehicle Access Control System
## Sistem Deteksi Plat Nomor dengan Web Interface

Sistem deteksi plat nomor Indonesia dengan akurasi tinggi, menggunakan computer vision dan OCR untuk kontrol akses kendaraan otomatis.

---

## ✅ Fitur Utama

- 🎯 **Deteksi Plat Nomor Akurat** - Algoritma preprocessing optimal untuk deteksi plat Indonesia
- 📖 **OCR Multi-Fallback** - Multiple OCR strategies dengan auto-correction
- 🎨 **Web Interface** - Dashboard modern dengan real-time monitoring
- 💾 **Database MySQL** - Storage terstruktur dengan Laragon
- 🚦 **Access Control** - Auto gate control berdasarkan database kendaraan terdaftar
- 📊 **Analytics** - Log lengkap dan statistik akses
- 🔄 **Auto-Reconnect** - Exponential backoff untuk camera resilience
- 🛡️ **Validasi Format** - Indonesian plate format validation

---

## 📋 Prerequisites

### 1. **Software yang Dibutuhkan:**
- ✅ Python 3.8+
- ✅ Laragon (MySQL server)
- ✅ Tesseract OCR

### 2. **Install Tesseract OCR:**

**macOS:**
```bash
brew install tesseract
```

**Windows:**
1. Download dari: https://github.com/UB-Mannheim/tesseract/wiki
2. Install dan tambahkan ke PATH

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

---

## 🚀 Instalasi

### **Step 1: Clone Project**
```bash
git clone <repository-url>
cd project-plat-detection-dude
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Dependencies include:**
- Flask (Web framework)
- mysql-connector-python (MySQL driver)
- opencv-python (Computer vision)
- pytesseract (OCR)
- python-dotenv (Environment variables)

### **Step 3: Setup Database MySQL**

#### A. Start Laragon
1. Buka **Laragon**
2. Klik **Start All** (Apache + MySQL)
3. MySQL akan running di port **3306**

#### B. Create Database
**Option 1: Via HeidiSQL/phpMyAdmin**
```sql
SOURCE database_setup.sql;
```

**Option 2: Manual SQL**
```sql
CREATE DATABASE sistem_parkir_smk CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sistem_parkir_smk;
-- Lalu run isi dari database_setup.sql
```

#### C. Migrate Data (Optional)
Jika punya data dari SQLite sebelumnya:
```bash
python3 migrate_sqlite_to_mysql.py
```

### **Step 4: Konfigurasi Environment**

Copy `.env.example` jadi `.env`:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Flask Secret Key (REQUIRED - Generate random!)
# Generate dengan: python3 -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your_random_secret_key_here

# Kamera CCTV
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=your_camera_password_here
CAMERA_CHANNEL=1
CAMERA_SUBTYPE=0

# Database MySQL (Laragon)
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=sistem_parkir_smk

# System
SAVE_FOLDER=gambarplat
LOG_LEVEL=INFO
MAX_RETRY_CAMERA=5
MIN_CONFIDENCE=0.6
```

**⚠️ IMPORTANT:**
- Generate random `SECRET_KEY` untuk security
- Ganti `CAMERA_PASSWORD` dengan password CCTV Anda
- `DB_PASSWORD` kosong untuk Laragon default

---

## 🎬 Cara Menggunakan

### **1. Start Laragon MySQL**
Pastikan MySQL di Laragon sudah running (indicator hijau)

### **2. Run Web Application**
```bash
python3 app.py
```

### **3. Buka Browser**
```
http://localhost:5001
```

### **4. Features Available:**
- 📹 **Live Camera Feed** - Real-time video dari CCTV
- 🚗 **Vehicle Management** - CRUD kendaraan terdaftar
- 📊 **Access Logs** - History akses dengan filter
- 🖼️ **Detected Plates Gallery** - Galeri plat terdeteksi
- 🚦 **Manual Override** - Buka gate secara manual

---

## 📊 System Flow

```
Camera → Plate Detection → OCR → Validation → Database Check → Gate Control
   ↓            ↓              ↓         ↓            ↓              ↓
CCTV     PlateDetector   OCRProcessor  Validator   MySQL      Access Log
```

**Flow Detail:**
1. **Camera** menangkap frame real-time
2. **PlateDetector** mencari area plat dengan preprocessing
3. **OCRProcessor** membaca teks dengan multiple fallback
4. **Validator** validasi format plat Indonesia
5. **MySQL** cek apakah plat terdaftar
6. **Gate Control** buka/tutup palang otomatis
7. **Access Log** catat semua aktivitas

---

## 📁 Project Structure

```
project-plat-detection-dude/
├── app.py                          # ⭐ Main Flask application
├── config.py                       # ⚙️ Configuration management
├── database_setup.sql              # 🗄️ MySQL schema + sample data
├── migrate_sqlite_to_mysql.py      # 🔄 Data migration script
├── requirements.txt                # 📦 Python dependencies
├── .env.example                    # 📝 Environment template
├── .env                           # 🔐 Actual config (gitignored)
│
├── utils/                          # 🛠️ Helper modules
│   ├── __init__.py
│   ├── plate_detector.py          # 🔍 Plate region detection
│   ├── ocr_processor.py           # 📖 OCR with fallback
│   ├── plate_validator.py         # ✅ Indonesian format validation
│   └── vehicle_analyzer.py        # 🚗 Color & type detection
│
├── templates/                      # 🎨 HTML templates
│   ├── index.html                 # Dashboard
│   ├── vehicles.html              # Vehicle management
│   ├── access_logs.html           # Access history
│   └── detected_plates.html       # Plate gallery
│
├── static/                         # 📸 Static files
│   └── screenshots/               # Camera screenshots
│
├── gambarplat/                     # 🖼️ Detected plate images
│   └── SUCCESS_*.jpg              # Successfully read plates
│
├── logs/                          # 📋 System logs
│   └── plate_detection.log
│
└── archive/                        # 📦 Old files (deprecated)
    ├── deteksi_plat.py            # Old SQLite version
    ├── deteksi_plat_enhanced.py   # Old enhanced version
    └── README.md                  # Archive explanation
```

---

## 🔧 Troubleshooting

### **❌ Database Connection Error**
```
MySQLError: Can't connect to MySQL server
```
**Fix:**
1. Pastikan Laragon **running** (MySQL indicator hijau)
2. Cek `.env` - `DB_HOST=localhost`, `DB_USER=root`, `DB_PASSWORD=` (kosong)
3. Test connection: `mysql -u root -p` di terminal

### **❌ Camera Connection Failed**
```
❌ Gagal koneksi ke kamera
```
**Fix:**
1. Cek CCTV nyala dan di network yang sama
2. Test RTSP URL dengan VLC: `rtsp://user:pass@ip:port/...`
3. Ganti `CAMERA_PASSWORD` di `.env`
4. Fallback: Sistem auto-switch ke webcam laptop

### **❌ Tesseract Not Found**
```
TesseractNotFoundError
```
**Fix:**
1. Install Tesseract OCR (lihat Prerequisites)
2. Add ke PATH system
3. Restart terminal

### **❌ SECRET_KEY Warning**
```
⚠️ SECRET_KEY masih menggunakan default!
```
**Fix:**
```bash
# Generate random key
python3 -c "import secrets; print(secrets.token_hex(32))"

# Copy output ke .env
SECRET_KEY=<generated_key_here>
```

### **❌ Low Detection Accuracy**
**Fix:**
1. Improve lighting kondisi
2. Adjust camera angle (15-30° downward)
3. Turunkan `MIN_CONFIDENCE` di `.env` (default: 0.6)
4. Check camera focus/resolution

---

## ⚙️ Configuration

### **Database Settings** (config.py)
```python
DB_HOST = 'localhost'      # MySQL host
DB_PORT = 3306             # MySQL port
DB_USER = 'root'           # MySQL user
DB_PASSWORD = ''           # Empty for Laragon default
DB_NAME = 'sistem_parkir_smk'
```

### **Detection Parameters** (utils/plate_detector.py)
```python
MIN_PLATE_WIDTH = 50       # Min width plat (px)
MIN_PLATE_HEIGHT = 15      # Min height plat (px)
MIN_ASPECT_RATIO = 2.5     # Min aspect ratio (width/height)
MAX_ASPECT_RATIO = 4.5     # Max aspect ratio
```

### **OCR Settings** (utils/ocr_processor.py)
```python
psm_modes = [
    ('PSM 7 - Single Line', '--psm 7 --oem 3'),
    ('PSM 8 - Single Word', '--psm 8 --oem 3'),
    # Multiple fallback modes
]
```

---

## 📈 Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Detection Accuracy** | ~80% | With optimal lighting |
| **OCR Success Rate** | ~75% | Indonesian plates |
| **False Positive Rate** | <15% | With validation |
| **Processing Speed** | ~10 FPS | On standard laptop |
| **Database Query** | <50ms | MySQL with indexes |

---

## 🔒 Security Features

- ✅ **Environment Variables** - Credentials tidak hardcoded
- ✅ **Secret Key** - Random Flask session key
- ✅ **SQL Injection Prevention** - Parameterized queries
- ✅ **Input Validation** - Plate format validation
- ✅ **Access Logs** - Complete audit trail

---

## 🚦 Gate Control Logic

```python
if plate_detected and plate_valid:
    if plate_registered and status_active:
        ✅ GRANT ACCESS
        🟢 Open gate for 5 seconds
        📝 Log: "boleh_masuk"
    else:
        ❌ DENY ACCESS
        🔴 Keep gate closed
        📝 Log: "ditolak"
```

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard homepage |
| `/video_feed` | GET | Camera live stream |
| `/api/latest_detection` | GET | Latest plate detection |
| `/api/screenshot` | GET | Capture camera frame |
| `/api/manual_override` | GET | Manual gate open |
| `/api/system_status` | GET | System health check |
| `/vehicles` | GET | Vehicle management page |
| `/add_vehicle` | POST | Add new vehicle |
| `/edit_vehicle/<id>` | POST | Edit vehicle data |
| `/delete_vehicle/<id>` | GET | Delete vehicle |
| `/access_logs` | GET | Access history |
| `/detected_plates` | GET | Plate gallery |

---

## 🎯 Future Improvements

- [ ] Multi-camera support
- [ ] Mobile app integration
- [ ] WhatsApp notifications
- [ ] License plate recognition AI model
- [ ] Cloud storage integration
- [ ] Advanced analytics dashboard

---

## 📞 Support & Contributing

**Issues?** Check logs: `logs/plate_detection.log`

**Want to contribute?**
1. Fork repository
2. Create feature branch
3. Commit changes
4. Push and create Pull Request

---

## 📄 License

MIT License - Feel free to use for educational and commercial purposes.

---

🎉 **Selamat menggunakan Vehicle Access Control System!**

**Made with ❤️ for SMK Projects**
