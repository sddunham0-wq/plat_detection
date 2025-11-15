# 🏠 TUTORIAL INSTALASI OFFLINE / LOCAL
## Panduan Lengkap Setup Sistem Tanpa Internet

## 🎯 Tujuan
Menjalankan sistem deteksi plat nomor **100% offline** di komputer lokal, tanpa internet dan tanpa server external.

**Cocok untuk:**
- 🏫 Sekolah/kampus dengan internet terbatas
- 🏢 Kantor dengan network terisolasi
- 🏘️ Perumahan/komplek tanpa WiFi
- 🔒 Sistem yang butuh keamanan tinggi (no cloud)

---

## 📋 Daftar Isi
1. [Persiapan & Planning](#persiapan--planning)
2. [Download Semua File (Saat Ada Internet)](#download-semua-file-saat-ada-internet)
3. [Instalasi Tools Offline](#instalasi-tools-offline)
4. [Setup Database Local](#setup-database-local)
5. [Konfigurasi Aplikasi](#konfigurasi-aplikasi)
6. [Cara Menjalankan](#cara-menjalankan)
7. [Akses dari Komputer Lain (LAN)](#akses-dari-komputer-lain-lan)
8. [Testing & Validasi](#testing--validasi)
9. [Backup & Maintenance](#backup--maintenance)
10. [Troubleshooting Detail](#troubleshooting-detail)
11. [FAQ](#faq)
12. [Appendix](#appendix)

---

## 🎯 Persiapan

### Yang Dibutuhkan:
- ✅ 1 Komputer/Laptop (untuk jadi server local)
- ✅ Webcam atau CCTV (terhubung ke komputer)
- ✅ Installer offline (download dulu saat ada internet)
- ✅ Project folder lengkap

### Spesifikasi Minimal:
- **OS:** Windows 10/11, macOS 10.15+, atau Linux
- **Processor:** Intel i5 atau setara
- **RAM:** 8GB (minimal 4GB)
- **Storage:** 5GB free space
- **Network:** LAN (opsional, kalau mau akses dari komputer lain)

---

## 📦 STEP 1: Download Installer Offline

### Saat Masih Ada Internet, Download:

#### 1. **Python 3.11 (Standalone Installer)**
**macOS:**
```
https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg
```

**Windows:**
```
https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
```

**Centang:** "Add Python to PATH" saat install

---

#### 2. **MySQL Community Server (Standalone)**
**macOS:**
```
https://dev.mysql.com/downloads/mysql/
Download: macOS DMG Archive (x86, 64-bit)
```

**Windows:**
```
https://dev.mysql.com/downloads/mysql/
Download: Windows (x86, 64-bit), MSI Installer
```

---

#### 3. **Tesseract OCR (Standalone)**
**macOS:**
```
brew install tesseract
# Atau download dari Homebrew offline package
```

**Windows:**
```
https://github.com/UB-Mannheim/tesseract/releases
Download: tesseract-ocr-w64-setup-5.3.3.20231005.exe
```

---

#### 4. **Python Libraries (Download Wheels)**

**Cara 1: Download semua sekaligus**
```bash
# Di komputer dengan internet
pip3 download -r requirements.txt -d ./python_packages

# Hasilnya folder python_packages berisi semua .whl files
```

**Cara 2: Download manual satu-satu**
Buka: https://pypi.org/ dan download `.whl` untuk:
- opencv-python-4.8.1.78
- pytesseract-0.3.10
- Flask-2.3.3
- PyMySQL-1.1.0
- numpy-1.24.3
- (dan library lainnya dari requirements.txt)

**PENTING:** Download untuk Python 3.11 dan OS yang sesuai!

---

## 🔧 STEP 2: Instalasi Offline di Komputer Local

### A. Install Python (Offline)

**macOS:**
```bash
# Double click file .pkg yang sudah didownload
sudo installer -pkg python-3.11.7-macos11.pkg -target /
```

**Windows:**
```
1. Double click python-3.11.7-amd64.exe
2. Centang "Add Python to PATH"
3. Click "Install Now"
```

**Test instalasi:**
```bash
python3 --version
# Output: Python 3.11.7
```

---

### B. Install MySQL (Offline)

**macOS:**
```bash
# Double click file .dmg
# Drag MySQL ke Applications
# Start MySQL dari System Preferences
```

**Windows:**
```
1. Double click mysql-installer-community.exe
2. Pilih "Developer Default"
3. Set root password (catat baik-baik!)
4. Finish installation
```

**Test MySQL:**
```bash
mysql --version
# Output: mysql Ver 8.0.x
```

**Start MySQL Service:**

**macOS:**
```bash
# Via System Preferences → MySQL → Start
# Atau via terminal:
sudo /usr/local/mysql/support-files/mysql.server start
```

**Windows:**
```
# Via Services.msc → MySQL80 → Start
# Atau via Command Prompt (Admin):
net start MySQL80
```

---

### C. Install Tesseract OCR (Offline)

**macOS:**
```bash
# Install dari .pkg atau via Homebrew
# Jika sudah download via brew:
brew install tesseract
```

**Windows:**
```
1. Double click tesseract-ocr-setup.exe
2. Install ke C:\Program Files\Tesseract-OCR
3. Tambahkan ke PATH:
   - Control Panel → System → Advanced → Environment Variables
   - Edit "Path" → Add: C:\Program Files\Tesseract-OCR
4. Restart Command Prompt
```

**Test Tesseract:**
```bash
tesseract --version
# Output: tesseract 5.3.3
```

---

### D. Install Python Libraries (Offline)

**Cara 1: Install dari folder packages**
```bash
cd /path/to/project
pip3 install --no-index --find-links=./python_packages -r requirements.txt
```

**Cara 2: Install manual satu-satu**
```bash
cd /path/to/python_packages
pip3 install opencv_python-4.8.1.78-*.whl
pip3 install pytesseract-0.3.10-*.whl
pip3 install Flask-2.3.3-*.whl
pip3 install PyMySQL-1.1.0-*.whl
# ... dan seterusnya
```

**Verify instalasi:**
```bash
pip3 list | grep opencv
pip3 list | grep Flask
```

---

## 💾 STEP 3: Setup Database Local (Offline)

### A. Login ke MySQL

```bash
mysql -u root -p
# Masukkan password yang dibuat saat instalasi
```

### B. Import Database Schema

**Dari terminal:**
```bash
mysql -u root -p < database_setup.sql
# Masukkan password
```

**Atau dari MySQL shell:**
```sql
mysql> source /path/to/database_setup.sql;
```

### C. Verifikasi Database

```bash
mysql -u root -p
```

```sql
-- Cek database
SHOW DATABASES;

-- Gunakan database
USE sistem_parkir_smk;

-- Cek tabel
SHOW TABLES;

-- Cek data dummy
SELECT COUNT(*) FROM kendaraan_terdaftar;
-- Output: 16 (jumlah data dummy)

-- Exit
EXIT;
```

---

## ⚙️ STEP 4: Konfigurasi Aplikasi untuk Local

### A. Buat File .env

```bash
cd /path/to/project-plat-detection-dude
touch .env  # macOS/Linux
# atau
echo. > .env  # Windows
```

### B. Edit File .env untuk Local

**Buka dengan text editor, isi:**

```env
# ==========================================
# KONFIGURASI LOCAL (OFFLINE)
# ==========================================

# Database Configuration (LOCAL)
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_root_password
DB_NAME=sistem_parkir_smk

# Camera Configuration (WEBCAM LAPTOP)
# Jika pakai webcam laptop, tidak perlu RTSP
CAMERA_HOST=0
CAMERA_PORT=0
CAMERA_USER=
CAMERA_PASSWORD=

# Flask Configuration (LOCAL)
FLASK_PORT=8080
FLASK_HOST=0.0.0.0
SECRET_KEY=local-dev-secret-key-ganti-nanti

# System Configuration
SAVE_FOLDER=static/gambarplat
LOG_LEVEL=INFO
MAX_RETRY_CAMERA=5
```

**PENTING:** Ganti `your_mysql_root_password` dengan password MySQL Anda!

---

### C. Edit app.py untuk Local/Webcam

**Buka `app.py`, cari baris 176-213 (fungsi `initialize_camera`):**

**Untuk pakai WEBCAM (tidak pakai CCTV), comment baris CCTV:**

```python
def initialize_camera():
    """
    Penjelasan SMK: Seperti 'nyalakan kamera' dan pastikan bisa dipakai
    Coba connect ke CCTV dulu, kalau gagal pakai webcam laptop
    """
    global camera, system_status

    # ★ DISABLE CCTV - Comment semua baris CCTV
    # try:
    #     from config import config
    #     camera_url = config.CAMERA_URL
    #     logger.info(f"🎥 Mencoba koneksi ke CCTV: {camera_url}")
    #     camera = cv2.VideoCapture(camera_url)
    #
    #     if camera.isOpened():
    #         ret, frame = camera.read()
    #         if ret and frame is not None:
    #             system_status['camera_connected'] = True
    #             logger.info("✅ CCTV berhasil terhubung")
    #             return True
    # except Exception as e:
    #     logger.warning(f"⚠️ CCTV tidak tersedia: {e}")

    # ★ LANGSUNG PAKAI WEBCAM LAPTOP
    try:
        logger.info("🎥 Mencoba webcam laptop...")
        camera = cv2.VideoCapture(0)  # Index 0 = webcam default
        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ Webcam laptop berhasil terhubung")
                return True
    except Exception as e:
        logger.error(f"❌ Webcam juga gagal: {e}")

    system_status['camera_connected'] = False
    logger.error("❌ Tidak ada kamera yang tersedia")
    return False
```

**Save file!**

---

## 🚀 STEP 5: Menjalankan Aplikasi (Offline)

### A. Test Webcam

```bash
cd /path/to/project-plat-detection-dude
python3 testing_webcam.py
```

**Output yang diharapkan:**
```
Testing webcam...

Trying index 0...
✅ Webcam index 0 WORKS!
   Resolution: 1280x720

Test selesai!
```

---

### B. Jalankan Aplikasi

```bash
python3 app.py
```

**Output yang diharapkan:**
```
🚀 Starting Vehicle Access Control System...
📁 Folder created/verified: static/screenshots
📁 Folder created/verified: static/gambarplat
📁 Folder created/verified: logs
🎥 Initializing camera...
🎥 Mencoba webcam laptop...
✅ Webcam laptop berhasil terhubung
✅ Camera initialized successfully
🌐 Starting web server on http://localhost:8080
 * Running on http://0.0.0.0:8080
```

---

### C. Akses Dashboard

**Buka browser (Chrome/Firefox/Safari):**
```
http://localhost:8080
```

**Atau dari IP local:**
```
http://192.168.1.100:8080
# (ganti dengan IP komputer Anda)
```

**Cek IP komputer:**

**macOS/Linux:**
```bash
ifconfig | grep "inet "
```

**Windows:**
```bash
ipconfig
```

---

## 🌐 STEP 6: Akses dari Komputer Lain di LAN (Opsional)

### Skenario:
Komputer A (server) menjalankan aplikasi, Komputer B/C/D (client) ingin akses dashboard.

### A. Setup di Komputer Server (A)

**1. Cek IP Address komputer server:**

**macOS/Linux:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
# Output contoh: inet 192.168.1.100
```

**Windows:**
```bash
ipconfig
# Cari IPv4 Address: 192.168.1.100
```

**2. Pastikan Flask listening di 0.0.0.0:**

File `.env`:
```env
FLASK_HOST=0.0.0.0  # Accept connection dari semua IP
FLASK_PORT=8080
```

**3. Allow Firewall (kalau ada):**

**macOS:**
```bash
# System Preferences → Security & Privacy → Firewall
# Allow incoming connections untuk Python
```

**Windows:**
```bash
# Control Panel → Windows Defender Firewall → Allow an app
# Add Python.exe
```

---

### B. Akses dari Komputer Client (B/C/D)

**Di komputer client, buka browser:**
```
http://192.168.1.100:8080
```

**Ganti `192.168.1.100` dengan IP server yang tadi dicek!**

---

### C. Test Koneksi

**Dari komputer client, test ping dulu:**

```bash
ping 192.168.1.100
# Harusnya reply
```

**Kalau tidak bisa ping:**
- Cek firewall di server
- Pastikan di LAN yang sama (WiFi/Ethernet)
- Cek router settings

---

## 📊 STEP 7: Struktur Folder untuk Offline

```
project-plat-detection-dude/
├── app.py                          # Main application
├── config.py                       # Configuration
├── requirements.txt                # Dependencies list
├── database_setup.sql              # Database schema
├── .env                            # ★ BUAT MANUAL (local config)
│
├── installers/                     # ★ Folder installer offline
│   ├── python-3.11.7-macos11.pkg
│   ├── mysql-8.0.35-macos.dmg
│   ├── tesseract-ocr-setup.exe
│   └── ...
│
├── python_packages/                # ★ Folder Python wheels offline
│   ├── opencv_python-4.8.1.78.whl
│   ├── Flask-2.3.3.whl
│   ├── PyMySQL-1.1.0.whl
│   └── ...
│
├── models/
│   └── best.pt                     # YOLO model (sudah ada)
│
├── utils/
│   ├── yolo_plate_detector.py
│   ├── ocr_processor.py
│   └── vehicle_analyzer.py
│
├── templates/
│   ├── index.html
│   ├── vehicles.html
│   └── access_logs.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── gambarplat/                 # Foto plat tersimpan di sini
│
└── TUTORIAL_*.md                   # Tutorial files
```

---

## 🐛 Troubleshooting Offline

### Problem 1: "ModuleNotFoundError: No module named 'cv2'"

**Penyebab:** Python libraries belum diinstall

**Solusi:**
```bash
# Install dari folder packages offline
pip3 install --no-index --find-links=./python_packages opencv-python
```

---

### Problem 2: "Can't connect to MySQL server on 'localhost'"

**Penyebab:** MySQL service belum jalan

**Solusi:**

**macOS:**
```bash
sudo /usr/local/mysql/support-files/mysql.server start
```

**Windows:**
```bash
net start MySQL80
```

---

### Problem 3: "Camera not authorized (status 0)"

**Penyebab:** macOS Privacy Settings belum allow Python akses camera

**Solusi:**
1. System Preferences → Security & Privacy → Privacy → Camera
2. Centang Terminal atau Python
3. Restart Terminal

---

### Problem 4: "Tesseract not found"

**Penyebab:** Tesseract belum di-install atau belum di PATH

**Solusi:**

**macOS:**
```bash
brew install tesseract
```

**Windows:**
```bash
# Install dari installer
# Tambahkan ke PATH: C:\Program Files\Tesseract-OCR
```

---

### Problem 5: "Address already in use (Port 8080)"

**Penyebab:** Port 8080 sudah dipakai aplikasi lain

**Solusi:**
Edit `.env`:
```env
FLASK_PORT=8081  # Ganti port
```

---

### Problem 6: Tidak bisa akses dari komputer lain

**Penyebab:** Firewall block atau FLASK_HOST salah

**Solusi:**
1. Edit `.env`:
   ```env
   FLASK_HOST=0.0.0.0
   ```
2. Allow firewall untuk Python
3. Pastikan di LAN yang sama

---

## ✅ Checklist Instalasi Offline

### Persiapan (Saat Ada Internet):
- [ ] Download Python installer
- [ ] Download MySQL installer
- [ ] Download Tesseract installer
- [ ] Download Python packages (pip download)
- [ ] Copy project folder lengkap

### Instalasi (Offline):
- [ ] Install Python 3.11
- [ ] Install MySQL 8.0
- [ ] Install Tesseract OCR
- [ ] Install Python libraries dari folder packages
- [ ] Start MySQL service

### Konfigurasi:
- [ ] Import database_setup.sql
- [ ] Buat file .env
- [ ] Edit app.py untuk webcam (comment CCTV)
- [ ] Set MySQL password di .env
- [ ] Set FLASK_HOST=0.0.0.0 untuk LAN access

### Testing:
- [ ] `python3 --version` works
- [ ] `mysql --version` works
- [ ] `tesseract --version` works
- [ ] `pip3 list` show all libraries
- [ ] `python3 testing_webcam.py` works
- [ ] `python3 app.py` jalan tanpa error
- [ ] Browser bisa buka `http://localhost:8080`

### LAN Access (Opsional):
- [ ] Cek IP address server
- [ ] Allow firewall
- [ ] Test ping dari client
- [ ] Browser client bisa akses `http://IP:8080`

---

## 💡 Tips Offline Mode

### 1. **Backup Database Rutin**
```bash
# Export database
mysqldump -u root -p sistem_parkir_smk > backup_$(date +%Y%m%d).sql

# Import saat restore
mysql -u root -p sistem_parkir_smk < backup_20250113.sql
```

### 2. **Cleanup Foto Lama**
```bash
# Hapus foto >30 hari
find static/gambarplat -name "*.jpg" -mtime +30 -delete
```

### 3. **Monitor Resource**
```bash
# Cek CPU & Memory usage
top | grep python3
```

### 4. **Auto-start (Opsional)**

**macOS (LaunchAgent):**
```bash
# Buat file ~/Library/LaunchAgents/com.alpr.plist
```

**Windows (Task Scheduler):**
```
Task Scheduler → Create Task → Start python3 app.py at login
```

---

## 📚 Dokumentasi Terkait

- `INSTALASI_DAN_TOOLS.md` - Panduan instalasi lengkap
- `TUTORIAL_GANTI_WEBCAM.md` - Setup webcam
- `MATERI_PRESENTASI.md` - Materi presentasi
- `README.md` - Overview project

---

## 🆘 Butuh Bantuan?

Kalau stuck atau error:
1. Cek log di terminal
2. Cek file log di folder `logs/`
3. Restart aplikasi
4. Restart MySQL service
5. Restart komputer

---

**Selamat! Sistem siap jalan 100% offline! 🎉**
