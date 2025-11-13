# 🛠️ PANDUAN INSTALASI DAN TOOLS

## 📋 Daftar Isi
1. [Tools yang Dibutuhkan](#tools-yang-dibutuhkan)
2. [Instalasi Step by Step](#instalasi-step-by-step)
3. [Konfigurasi Sistem](#konfigurasi-sistem)
4. [Troubleshooting](#troubleshooting)

---

## 🎯 Tools yang Dibutuhkan

### 1. **Python 3.8 atau Lebih Baru**
**Fungsi:** Bahasa pemrograman utama untuk menjalankan aplikasi

**Cara Install:**
- **macOS:**
  ```bash
  brew install python3
  ```
- **Windows:**
  Download dari [python.org](https://www.python.org/downloads/)
- **Cek versi:**
  ```bash
  python3 --version
  ```

---

### 2. **MySQL Database Server**
**Fungsi:** Menyimpan data kendaraan terdaftar dan log akses

**Cara Install:**
- **macOS (dengan Homebrew):**
  ```bash
  brew install mysql
  brew services start mysql
  ```
- **Windows:**
  Download [MySQL Installer](https://dev.mysql.com/downloads/installer/)
- **Cek status:**
  ```bash
  mysql --version
  ```

**Setup Database:**
```bash
mysql -u root -p < database_setup.sql
```

---

### 3. **Tesseract OCR**
**Fungsi:** Membaca text plat nomor dari gambar

**Cara Install:**
- **macOS:**
  ```bash
  brew install tesseract
  ```
- **Windows:**
  Download dari [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Cek instalasi:**
  ```bash
  tesseract --version
  ```

---

### 4. **Webcam atau CCTV**
**Fungsi:** Menangkap gambar kendaraan untuk deteksi

**Spesifikasi Minimal:**
- Resolusi: 720p (1280x720)
- FPS: 30 fps
- Koneksi: USB atau RTSP

**Test Webcam:**
```bash
python3 testing_webcam.py
```

---

### 5. **Git (Opsional)**
**Fungsi:** Version control untuk project

**Cara Install:**
- **macOS:**
  ```bash
  brew install git
  ```
- **Windows:**
  Download dari [git-scm.com](https://git-scm.com/)

---

## 📦 Python Libraries yang Dibutuhkan

### Computer Vision & Image Processing
| Library | Versi | Fungsi |
|---------|-------|--------|
| **opencv-python** | 4.8.1.78 | Deteksi objek, bounding box, video processing |
| **pytesseract** | 0.3.10 | OCR untuk baca text plat nomor |
| **numpy** | 1.24.3 | Operasi matematis untuk image processing |
| **Pillow** | 10.0.1 | Image manipulation dan enhancement |
| **scipy** | 1.11.3 | Scientific computing untuk preprocessing |
| **scikit-image** | 0.21.0 | Advanced image processing |

### Web Framework
| Library | Versi | Fungsi |
|---------|-------|--------|
| **Flask** | 2.3.3 | Web server untuk dashboard dan API |

### Database
| Library | Versi | Fungsi |
|---------|-------|--------|
| **PyMySQL** | 1.1.0 | Koneksi ke MySQL database |
| **DBUtils** | 3.0.3 | Connection pooling untuk performa |

### Utilities
| Library | Versi | Fungsi |
|---------|-------|--------|
| **python-dotenv** | 1.0.0 | Manage environment variables (.env file) |
| **colorlog** | 6.7.0 | Colored logging untuk debugging |
| **typing-extensions** | 4.8.0 | Type hints untuk Python |

---

## 🚀 Instalasi Step by Step

### Langkah 1: Clone atau Download Project
```bash
cd ~/Documents/DWI
# Jika sudah ada project, skip langkah ini
```

### Langkah 2: Install Python Dependencies
```bash
cd project-plat-detection-dude
pip3 install -r requirements.txt
```

**Troubleshooting Install:**
- Kalau error permission: `pip3 install --user -r requirements.txt`
- Kalau pip3 tidak ada: `python3 -m pip install -r requirements.txt`

### Langkah 3: Install System Dependencies

**macOS:**
```bash
# Install Tesseract OCR
brew install tesseract

# Install MySQL
brew install mysql
brew services start mysql
```

**Windows:**
1. Install Tesseract dari [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install MySQL dari [MySQL Installer](https://dev.mysql.com/downloads/installer/)
3. Tambahkan Tesseract ke PATH

### Langkah 4: Setup Database
```bash
# Login ke MySQL
mysql -u root -p

# Jalankan setup script
source database_setup.sql

# Atau dari terminal langsung
mysql -u root -p < database_setup.sql
```

### Langkah 5: Konfigurasi Environment
```bash
# Copy file .env.example jadi .env
cp .env.example .env

# Edit .env dengan editor
nano .env
```

**Isi file .env:**
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=sistem_parkir_smk

# Camera Configuration
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=your_camera_password

# Flask Configuration
FLASK_PORT=8080
SECRET_KEY=generate_random_key_here

# Logging
LOG_LEVEL=INFO
```

**Generate SECRET_KEY:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### Langkah 6: Test Instalasi

**Test Database:**
```bash
mysql -u root -p -e "SELECT COUNT(*) FROM sistem_parkir_smk.kendaraan_terdaftar;"
```

**Test Webcam:**
```bash
python3 testing_webcam.py
```

**Test OCR:**
```bash
tesseract --version
```

### Langkah 7: Jalankan Aplikasi
```bash
python3 app.py
```

**Buka browser:**
```
http://localhost:8080
```

---

## ⚙️ Konfigurasi Sistem

### Konfigurasi Camera

**Untuk Webcam Laptop:**
Edit `app.py` baris 176-213, comment kode CCTV, pakai webcam index 0.

**Untuk CCTV RTSP:**
Edit `.env`:
```env
CAMERA_HOST=192.168.1.100
CAMERA_PORT=554
CAMERA_USER=admin
CAMERA_PASSWORD=password123
```

### Konfigurasi Database

**Ganti Password MySQL:**
```bash
mysql -u root -p
ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
FLUSH PRIVILEGES;
```

Update `.env`:
```env
DB_PASSWORD=new_password
```

### Konfigurasi OCR

Edit `config.py`:
```python
# OCR Configuration
MIN_OCR_CONFIDENCE = 0.01  # Threshold confidence (0.01 = 1%)
OCR_CONFIG = '--psm 8'     # Page segmentation mode
```

### Konfigurasi Detection

Edit `config.py`:
```python
# Ukuran minimum plat
MIN_PLATE_WIDTH = 70
MIN_PLATE_HEIGHT = 20

# Confidence threshold
MIN_CONFIDENCE = 0.6
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"
**Solusi:**
```bash
pip3 install -r requirements.txt --force-reinstall
```

### Problem: "Can't connect to MySQL"
**Solusi:**
```bash
# Start MySQL service
brew services start mysql  # macOS
# atau
sudo systemctl start mysql  # Linux
```

### Problem: "Camera not authorized"
**Solusi:**
1. Buka System Preferences → Security & Privacy → Privacy → Camera
2. Centang Terminal atau Python
3. Restart Terminal

### Problem: "Tesseract not found"
**Solusi:**
```bash
# macOS
brew install tesseract

# Windows - tambahkan ke PATH:
C:\Program Files\Tesseract-OCR
```

### Problem: "Port 8080 already in use"
**Solusi:**
Edit `.env`:
```env
FLASK_PORT=8081
```

### Problem: "Database too many connections"
**Solusi:**
```bash
# Restart MySQL
brew services restart mysql
```

---

## 📊 Struktur Project

```
project-plat-detection-dude/
├── app.py                      # Main application
├── config.py                   # Configuration
├── requirements.txt            # Python dependencies
├── database_setup.sql          # Database schema
├── .env                        # Environment variables (BUAT SENDIRI)
├── models/
│   └── best.pt                 # YOLO model untuk deteksi plat
├── utils/
│   ├── yolo_plate_detector.py  # YOLO detector
│   ├── ocr_processor.py        # OCR engine
│   └── vehicle_analyzer.py     # Analisis kendaraan
├── templates/
│   ├── index.html              # Dashboard
│   ├── vehicles.html           # Daftar kendaraan
│   └── access_logs.html        # Log akses
├── static/
│   ├── css/                    # Styling
│   ├── js/                     # JavaScript
│   └── gambarplat/             # Foto plat terdeteksi
└── TUTORIAL_*.md               # Tutorial files
```

---

## 🎓 Penjelasan Tools (Bahasa Sederhana)

### Python
**Analogi:** Seperti "mesin" yang menjalankan aplikasi
**Fungsi:** Bahasa pemrograman untuk logika sistem

### MySQL
**Analogi:** Seperti "lemari arsip digital"
**Fungsi:** Tempat simpan data kendaraan dan log

### OpenCV
**Analogi:** Seperti "mata" sistem
**Fungsi:** Deteksi objek, kotak hijau/biru, video processing

### Tesseract OCR
**Analogi:** Seperti "pembaca text otomatis"
**Fungsi:** Baca huruf dan angka di plat nomor

### Flask
**Analogi:** Seperti "pelayan" yang kasih website
**Fungsi:** Web server untuk dashboard

### YOLO Model
**Analogi:** Seperti "otak AI" untuk deteksi
**Fungsi:** AI model yang tau mana plat nomor

---

## 📝 Checklist Instalasi

- [] Python 3.8+ terinstall
- [] MySQL terinstall dan running
- [] Tesseract OCR terinstall
- [] `pip3 install -r requirements.txt` berhasil
- [] Database `sistem_parkir_smk` sudah dibuat
- [] File `.env` sudah dikonfigurasi
- [] Webcam/CCTV sudah terdeteksi
- [] `python3 app.py` jalan tanpa error
- [] Browser bisa buka `http://localhost:8080`

---

## 🆘 Butuh Bantuan?

1. Cek file `TUTORIAL_GANTI_WEBCAM.md` untuk setup webcam
2. Cek file `SOLUSI_FIX_LABEL_TEXT.md` untuk masalah label
3. Jalankan `python3 testing_webcam.py` untuk test camera
4. Lihat log di terminal untuk error message

---

**Selamat! Sistem siap digunakan! 🎉**
