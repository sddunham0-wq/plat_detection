# 🏠 PANDUAN SUPER LENGKAP - INSTALASI OFFLINE/LOCAL
## Sistem Deteksi Plat Nomor - 100% Tanpa Internet

---

## 📖 PENGANTAR

### Tentang Panduan Ini
Dokumen ini adalah **panduan paling lengkap** untuk menginstall dan menjalankan sistem deteksi plat nomor secara **100% offline** (tanpa internet) di komputer lokal.

### Siapa yang Membutuhkan Panduan Ini?
- 🏫 **Sekolah/Kampus** dengan internet terbatas atau tidak stabil
- 🏢 **Kantor/Perusahaan** dengan network terisolasi (no internet access)
- 🏘️ **Perumahan/Komplek** tanpa koneksi WiFi
- 🔒 **Organisasi** yang butuh keamanan tinggi (data tidak ke cloud)
- 🚧 **Lokasi Remote** (tambang, pabrik, dll) tanpa infrastruktur internet

### Apa yang Akan Anda Pelajari?
1. ✅ Download semua file yang dibutuhkan (saat masih ada internet)
2. ✅ Install tools tanpa internet
3. ✅ Setup database lokal
4. ✅ Konfigurasi aplikasi untuk mode offline
5. ✅ Jalankan sistem di localhost
6. ✅ Akses dari komputer lain via LAN
7. ✅ Backup dan maintenance
8. ✅ Troubleshooting lengkap

### Estimasi Waktu
- **Persiapan (download):** 2-4 jam (saat ada internet)
- **Instalasi offline:** 1-2 jam
- **Konfigurasi:** 30 menit
- **Testing:** 1 jam
- **Total:** 5-8 jam (untuk pertama kali)

---

## 📋 DAFTAR ISI LENGKAP

### BAGIAN 1: PERSIAPAN
1.1 [Perencanaan](#11-perencanaan)
1.2 [Spesifikasi Hardware](#12-spesifikasi-hardware)
1.3 [Checklist Persiapan](#13-checklist-persiapan)

### BAGIAN 2: DOWNLOAD FILE (SAAT ADA INTERNET)
2.1 [Download Python](#21-download-python)
2.2 [Download MySQL](#22-download-mysql)
2.3 [Download Tesseract OCR](#23-download-tesseract-ocr)
2.4 [Download Python Libraries](#24-download-python-libraries)
2.5 [Download Project Files](#25-download-project-files)
2.6 [Verifikasi Download](#26-verifikasi-download)

### BAGIAN 3: TRANSFER KE KOMPUTER OFFLINE
3.1 [Persiapan Media Transfer](#31-persiapan-media-transfer)
3.2 [Struktur Folder](#32-struktur-folder)
3.3 [Copy Files](#33-copy-files)

### BAGIAN 4: INSTALASI OFFLINE
4.1 [Install Python](#41-install-python)
4.2 [Install MySQL](#42-install-mysql)
4.3 [Install Tesseract OCR](#43-install-tesseract-ocr)
4.4 [Install Python Libraries](#44-install-python-libraries)
4.5 [Verifikasi Instalasi](#45-verifikasi-instalasi)

### BAGIAN 5: SETUP DATABASE
5.1 [Start MySQL Service](#51-start-mysql-service)
5.2 [Login MySQL](#52-login-mysql)
5.3 [Import Schema](#53-import-schema)
5.4 [Verifikasi Database](#54-verifikasi-database)
5.5 [Setup User & Permissions](#55-setup-user--permissions)

### BAGIAN 6: KONFIGURASI APLIKASI
6.1 [Setup Environment Variables](#61-setup-environment-variables)
6.2 [Konfigurasi Camera](#62-konfigurasi-camera)
6.3 [Konfigurasi Detection](#63-konfigurasi-detection)
6.4 [Konfigurasi Network](#64-konfigurasi-network)

### BAGIAN 7: MENJALANKAN SISTEM
7.1 [Test Camera](#71-test-camera)
7.2 [Start Application](#72-start-application)
7.3 [Akses Dashboard](#73-akses-dashboard)
7.4 [Test Detection](#74-test-detection)

### BAGIAN 8: SETUP LAN ACCESS
8.1 [Konfigurasi Server](#81-konfigurasi-server)
8.2 [Setup Firewall](#82-setup-firewall)
8.3 [Akses dari Client](#83-akses-dari-client)
8.4 [Test Koneksi](#84-test-koneksi)

### BAGIAN 9: TESTING & VALIDASI
9.1 [Test Fungsional](#91-test-fungsional)
9.2 [Test Performa](#92-test-performa)
9.3 [Test Security](#93-test-security)
9.4 [User Acceptance Test](#94-user-acceptance-test)

### BAGIAN 10: BACKUP & MAINTENANCE
10.1 [Backup Database](#101-backup-database)
10.2 [Backup Files](#102-backup-files)
10.3 [Maintenance Rutin](#103-maintenance-rutin)
10.4 [Update System](#104-update-system)

### BAGIAN 11: TROUBLESHOOTING
11.1 [Masalah Instalasi](#111-masalah-instalasi)
11.2 [Masalah Database](#112-masalah-database)
11.3 [Masalah Camera](#113-masalah-camera)
11.4 [Masalah Network](#114-masalah-network)
11.5 [Masalah Performa](#115-masalah-performa)

### BAGIAN 12: FAQ & TIPS
12.1 [FAQ](#121-faq)
12.2 [Tips Optimasi](#122-tips-optimasi)
12.3 [Best Practices](#123-best-practices)

### BAGIAN 13: APPENDIX
13.1 [Command Reference](#131-command-reference)
13.2 [File Locations](#132-file-locations)
13.3 [Port Reference](#133-port-reference)
13.4 [Error Code Reference](#134-error-code-reference)

---

# BAGIAN 1: PERSIAPAN

## 1.1 Perencanaan

### A. Skenario Deployment

**Skenario 1: Single Computer (Standalone)**
```
[Computer + Webcam] → Localhost → Browser di komputer yang sama
```
- Paling simple
- Cocok untuk: pos satpam kecil, testing
- Tidak perlu network

**Skenario 2: Server + Multiple Clients (LAN)**
```
[Server Computer + Camera] → LAN → [Client 1, 2, 3...] → Browser
```
- Lebih flexible
- Cocok untuk: sekolah, kantor, perumahan
- Butuh network switch/router

**Skenario 3: Multiple Locations**
```
[Location A Server + Camera] → VPN → [Location B Client]
```
- Advanced
- Cocok untuk: multi-cabang
- Butuh VPN setup

**Pilih skenario mana?** Untuk offline, **Skenario 1** atau **2** paling cocok.

---

### B. Timeline Planning

| Phase | Durasi | Kapan | Keterangan |
|-------|--------|-------|------------|
| **1. Download** | 2-4 jam | Hari 1 | Saat ada internet |
| **2. Transfer** | 30 menit | Hari 1 | USB/External drive |
| **3. Install Tools** | 1-2 jam | Hari 2 | Offline |
| **4. Setup Database** | 30 menit | Hari 2 | Offline |
| **5. Config App** | 30 menit | Hari 2 | Offline |
| **6. Testing** | 1-2 jam | Hari 2-3 | Offline |
| **7. Go-Live** | - | Hari 3 | Production |

**Total:** 2-3 hari untuk first-time setup

---

### C. Checklist Tim & Resources

**Tim yang Dibutuhkan:**
- [ ] 1 IT Person (install & config)
- [ ] 1 Security/Admin (manage database)
- [ ] 1 User/Satpam (testing & feedback)

**Resources:**
- [ ] 1 Computer/Laptop (specs di bawah)
- [ ] 1 Webcam atau CCTV
- [ ] 1 USB Flash Drive (16GB+) atau External HDD
- [ ] 1 LAN cable (jika setup multi-client)
- [ ] 1 Network Switch/Router (jika setup multi-client)
- [ ] Akses internet (untuk download files)
- [ ] Listrik stabil (UPS recommended)

---

## 1.2 Spesifikasi Hardware

### A. Komputer Server (Minimal)

| Komponen | Minimal | Recommended | Optimal |
|----------|---------|-------------|---------|
| **CPU** | Intel i3/AMD Ryzen 3 | Intel i5/AMD Ryzen 5 | Intel i7/AMD Ryzen 7 |
| **RAM** | 4GB | 8GB | 16GB |
| **Storage** | 128GB SSD/HDD | 256GB SSD | 512GB SSD |
| **OS** | Windows 10/macOS 10.15 | Windows 11/macOS 12+ | Windows 11/macOS 13+ |
| **Network** | - | Ethernet 100Mbps | Ethernet 1Gbps |
| **USB** | USB 2.0 | USB 3.0 | USB 3.1 |

**Catatan:**
- SSD **lebih cepat** dari HDD (recommended!)
- RAM 8GB cukup untuk 1-2 camera
- CPU i5 bisa handle 2-3 camera sekaligus

---

### B. Camera Specs

| Jenis | Resolusi | FPS | Koneksi | Harga | Cocok Untuk |
|-------|----------|-----|---------|-------|-------------|
| **Webcam USB** | 720p | 30fps | USB 2.0 | Rp 200-500rb | Testing, indoor |
| **Webcam HD** | 1080p | 30fps | USB 3.0 | Rp 500rb-1jt | Production, indoor |
| **IP Camera** | 1080p | 30fps | LAN/WiFi | Rp 1-3jt | Outdoor, jarak jauh |
| **CCTV Analog** | 720p | 25fps | BNC | Rp 500rb-2jt | Existing CCTV |

**Rekomendasi:**
- **Budget:** Webcam Logitech C270 (Rp 300rb)
- **Mid:** Webcam Logitech C920 (Rp 1jt)
- **Pro:** IP Camera Hikvision/Dahua (Rp 2-3jt)

---

### C. Network Equipment (untuk LAN setup)

| Item | Specs | Harga | Keterangan |
|------|-------|-------|------------|
| **Router** | 4-8 port, Gigabit | Rp 200-500rb | Bagi koneksi LAN |
| **Switch** | 8-24 port, Gigabit | Rp 300-1jt | Tambah port |
| **LAN Cable** | Cat6, 5-20 meter | Rp 50-200rb | Koneksi antar device |
| **UPS** | 600VA-1000VA | Rp 500rb-2jt | Backup power |

**Catatan:**
- Router/switch **TP-Link** atau **D-Link** sudah cukup
- UPS **penting** untuk hindari data loss saat mati lampu

---

## 1.3 Checklist Persiapan

### Sebelum Mulai, Pastikan Anda Punya:

**Hardware:**
- [ ] Computer/laptop sesuai specs (minimal i3, 4GB RAM)
- [ ] Webcam atau CCTV yang berfungsi
- [ ] USB Flash Drive 16GB+ (untuk transfer files)
- [ ] Mouse, keyboard, monitor (kalau pakai desktop)
- [ ] LAN cable (kalau setup multi-client)

**Software:**
- [ ] OS sudah terinstall (Windows/macOS/Linux)
- [ ] Browser (Chrome/Firefox) sudah terinstall
- [ ] Text editor (Notepad++/VSCode) sudah terinstall
- [ ] Unzip tool (7-Zip/WinRAR) sudah terinstall

**Access:**
- [ ] Admin access ke komputer
- [ ] Admin access ke router/firewall (kalau setup LAN)
- [ ] Password WiFi/LAN (kalau ada)

**Knowledge:**
- [ ] Bisa buka command line/terminal
- [ ] Bisa edit file text (.txt, .env)
- [ ] Bisa copy-paste command
- [ ] Bisa restart komputer 😄

**Internet (untuk download saja):**
- [ ] Koneksi internet stabil (minimal 5Mbps)
- [ ] Kuota cukup (~2-3GB untuk download)

---

# BAGIAN 2: DOWNLOAD FILE (SAAT ADA INTERNET)

**⚠️ PENTING:** Bagian ini dilakukan **SAAT MASIH ADA INTERNET**. Download semua file ini ke USB/External drive, lalu transfer ke komputer offline.

---

## 2.1 Download Python

### A. Pilih Versi yang Sesuai

**macOS:**
```
URL: https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg
File: python-3.11.7-macos11.pkg
Size: ~43MB
```

**Windows 64-bit:**
```
URL: https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
File: python-3.11.7-amd64.exe
Size: ~26MB
```

**Windows 32-bit:**
```
URL: https://www.python.org/ftp/python/3.11.7/python-3.11.7.exe
File: python-3.11.7.exe
Size: ~25MB
```

**Cara Cek OS 32/64-bit:**
- **Windows:** Settings → System → About → System type
- **macOS:** Semua Mac modern = 64-bit

---

### B. Download via Browser

1. Buka browser (Chrome/Firefox)
2. Copy-paste URL di atas ke address bar
3. File akan otomatis download
4. Simpan di folder `Downloads/installers/`

**Alternatif via Command Line:**

**macOS/Linux:**
```bash
mkdir -p ~/Downloads/installers
cd ~/Downloads/installers
curl -O https://www.python.org/ftp/python/3.11.7/python-3.11.7-macos11.pkg
```

**Windows (PowerShell):**
```powershell
New-Item -Path "$env:USERPROFILE\Downloads\installers" -ItemType Directory -Force
cd $env:USERPROFILE\Downloads\installers
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe" -OutFile "python-3.11.7-amd64.exe"
```

---

### C. Verifikasi Download

**Cek file size:**
```bash
# macOS/Linux
ls -lh python-3.11.7-macos11.pkg
# Harusnya ~43MB

# Windows
dir python-3.11.7-amd64.exe
# Harusnya ~26MB
```

**Cek MD5 Checksum (opsional):**
```bash
# macOS/Linux
md5 python-3.11.7-macos11.pkg

# Windows
certutil -hashfile python-3.11.7-amd64.exe MD5
```

Compare dengan checksum official di python.org

---

## 2.2 Download MySQL

### A. Pilih Versi yang Sesuai

**macOS (Intel):**
```
URL: https://dev.mysql.com/get/Downloads/MySQL-8.0/mysql-8.0.35-macos13-x86_64.dmg
File: mysql-8.0.35-macos13-x86_64.dmg
Size: ~400MB
```

**macOS (Apple Silicon M1/M2):**
```
URL: https://dev.mysql.com/get/Downloads/MySQL-8.0/mysql-8.0.35-macos13-arm64.dmg
File: mysql-8.0.35-macos13-arm64.dmg
Size: ~350MB
```

**Windows 64-bit:**
```
URL: https://dev.mysql.com/get/Downloads/MySQLInstaller/mysql-installer-community-8.0.35.0.msi
File: mysql-installer-community-8.0.35.0.msi
Size: ~400MB
```

**Cara Cek Mac Processor:**
```bash
uname -m
# x86_64 = Intel
# arm64 = Apple Silicon
```

---

### B. Download dari MySQL Official

**Opsi 1: Via Browser**
1. Buka: https://dev.mysql.com/downloads/mysql/
2. Pilih OS yang sesuai
3. Click "Download"
4. **Klik** "No thanks, just start my download" (tidak perlu login!)
5. Save ke `Downloads/installers/`

**Opsi 2: Via Command Line**

**macOS (Intel):**
```bash
cd ~/Downloads/installers
curl -L -O https://dev.mysql.com/get/Downloads/MySQL-8.0/mysql-8.0.35-macos13-x86_64.dmg
```

**Windows:**
```powershell
cd $env:USERPROFILE\Downloads\installers
Invoke-WebRequest -Uri "https://dev.mysql.com/get/Downloads/MySQLInstaller/mysql-installer-community-8.0.35.0.msi" -OutFile "mysql-installer.msi"
```

---

### C. Alternative: MariaDB (Open Source)

Kalau MySQL tidak bisa didownload, pakai MariaDB (compatible):

**macOS:**
```
URL: https://downloads.mariadb.org/mariadb/10.11.6/bintar-macos-x86_64/
```

**Windows:**
```
URL: https://downloads.mariadb.org/mariadb/10.11.6/winx64-packages/
```

---

## 2.3 Download Tesseract OCR

### A. macOS

**Via Homebrew (butuh install Homebrew dulu):**
```bash
# Install Homebrew (sekali saja)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract
brew install tesseract

# Tesseract akan diinstall di: /usr/local/bin/tesseract
```

**Manual Download:**
```
URL: https://github.com/tesseract-ocr/tesseract/releases
Cari: tesseract-5.3.3.pkg (untuk macOS)
```

---

### B. Windows

**Download Installer:**
```
URL: https://github.com/UB-Mannheim/tesseract/releases
File: tesseract-ocr-w64-setup-5.3.3.20231005.exe
Size: ~60MB
```

**Via Command:**
```powershell
cd $env:USERPROFILE\Downloads\installers
Invoke-WebRequest -Uri "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.3/tesseract-ocr-w64-setup-5.3.3.20231005.exe" -OutFile "tesseract-setup.exe"
```

---

### C. Linux (Ubuntu/Debian)

**Download Package:**
```bash
# Untuk Ubuntu 22.04
wget http://archive.ubuntu.com/ubuntu/pool/universe/t/tesseract/tesseract-ocr_5.3.0-2_amd64.deb

# Download dependencies juga
apt-cache depends tesseract-ocr | grep Depends | cut -d: -f2 | xargs -I {} apt-get download {}
```

---

## 2.4 Download Python Libraries

### A. Cara Download Semua Library Sekaligus

**Step 1: Pastikan pip update**
```bash
python3 -m pip install --upgrade pip
```

**Step 2: Download semua dari requirements.txt**
```bash
# Buat folder untuk simpan packages
mkdir -p ~/Downloads/python_packages

# Download SEMUA library + dependencies
pip3 download -r requirements.txt -d ~/Downloads/python_packages

# Ini akan download ~200MB file .whl
```

**Penjelasan:**
- `-r requirements.txt` = baca list dari file
- `-d ~/Downloads/python_packages` = save ke folder ini
- Akan download **semua dependencies** otomatis!

---

### B. Verifikasi Download

```bash
cd ~/Downloads/python_packages
ls -lh

# Harusnya ada 50-100 files .whl
# Total size ~150-250MB
```

**List files:**
```bash
ls -1 *.whl | head -20

# Output contoh:
# opencv_python-4.8.1.78-cp311-cp311-macosx_11_0_x86_64.whl
# Flask-2.3.3-py3-none-any.whl
# PyMySQL-1.1.0-py3-none-any.whl
# numpy-1.24.3-cp311-cp311-macosx_11_0_x86_64.whl
# ...
```

---

### C. Download Manual (jika pip download gagal)

**Visit PyPI untuk setiap library:**

1. **OpenCV:**
```
https://pypi.org/project/opencv-python/4.8.1.78/#files
Download: opencv_python-4.8.1.78-cp311-cp311-[your_platform].whl
```

2. **Flask:**
```
https://pypi.org/project/Flask/2.3.3/#files
Download: Flask-2.3.3-py3-none-any.whl
```

3. **PyMySQL:**
```
https://pypi.org/project/PyMySQL/1.1.0/#files
Download: PyMySQL-1.1.0-py3-none-any.whl
```

... (ulangi untuk semua library di requirements.txt)

**Catatan Platform:**
- `cp311` = Python 3.11
- `macosx` = macOS
- `win_amd64` = Windows 64-bit
- `any` = semua platform

---

### D. List Lengkap Library yang Dibutuhkan

| Library | Version | Size | Critical? |
|---------|---------|------|-----------|
| opencv-python | 4.8.1.78 | ~95MB | ✅ Yes |
| pytesseract | 0.3.10 | ~20KB | ✅ Yes |
| numpy | 1.24.3 | ~15MB | ✅ Yes |
| Pillow | 10.0.1 | ~3MB | ✅ Yes |
| Flask | 2.3.3 | ~700KB | ✅ Yes |
| PyMySQL | 1.1.0 | ~50KB | ✅ Yes |
| DBUtils | 3.0.3 | ~30KB | ✅ Yes |
| python-dotenv | 1.0.0 | ~30KB | ✅ Yes |
| scipy | 1.11.3 | ~35MB | ⚠️ Optional |
| scikit-image | 0.21.0 | ~30MB | ⚠️ Optional |
| colorlog | 6.7.0 | ~20KB | ⚠️ Optional |

**Total Size:** ~180MB (minimal) to ~250MB (full)

---

## 2.5 Download Project Files

### A. Download dari GitHub (jika public)

```bash
cd ~/Downloads
git clone https://github.com/username/project-plat-detection.git
```

**Atau download ZIP:**
1. Buka GitHub page
2. Click "Code" → "Download ZIP"
3. Extract ZIP ke folder

---

### B. Copy Project Files

Jika sudah punya project di komputer lain:

```bash
# macOS/Linux
rsync -av /path/to/project ~/Downloads/project-plat-detection-dude

# Windows
xcopy /E /I C:\path\to\project C:\Downloads\project-plat-detection-dude
```

---

### C. Struktur Project yang Harus Ada

```
project-plat-detection-dude/
├── app.py                  ✅ Must have
├── config.py               ✅ Must have
├── requirements.txt        ✅ Must have
├── database_setup.sql      ✅ Must have
├── .env.example            ✅ Must have
├── models/
│   └── best.pt             ✅ Must have (YOLO model)
├── utils/
│   ├── __init__.py
│   ├── yolo_plate_detector.py
│   ├── ocr_processor.py
│   └── vehicle_analyzer.py
├── templates/
│   ├── index.html
│   ├── vehicles.html
│   └── access_logs.html
├── static/
│   ├── css/
│   ├── js/
│   └── images/
└── README.md
```

**Check completeness:**
```bash
cd project-plat-detection-dude

# Check critical files
ls app.py config.py requirements.txt database_setup.sql
ls models/best.pt
ls utils/*.py

# Jika ada missing, download/copy yang kurang
```

---

## 2.6 Verifikasi Download

### A. Checklist Files

Buat file `checklist.txt` untuk tracking:

```bash
cd ~/Downloads

# List semua files
find . -name "*.pkg" -o -name "*.exe" -o -name "*.dmg" -o -name "*.whl" > checklist.txt

# Count
wc -l checklist.txt
```

**Yang harus ada (minimal):**
```
installers/
├── python-3.11.7-macos11.pkg       (~43MB)
├── mysql-8.0.35-macos13-x86_64.dmg (~400MB)
└── tesseract-ocr-setup.exe         (~60MB)

python_packages/
├── opencv_python-*.whl              (~95MB)
├── Flask-*.whl                      (~700KB)
├── PyMySQL-*.whl                    (~50KB)
└── ... (50-100 files)               (~180-250MB)

project-plat-detection-dude/
├── app.py
├── config.py
├── requirements.txt
├── database_setup.sql
└── models/best.pt                   (~6MB)
```

---

### B. Calculate Total Size

```bash
cd ~/Downloads

# Total size installers
du -sh installers/
# Expected: 500MB - 1GB

# Total size python packages
du -sh python_packages/
# Expected: 180MB - 250MB

# Total size project
du -sh project-plat-detection-dude/
# Expected: 20MB - 50MB

# Grand total
du -sh .
# Expected: 700MB - 1.3GB
```

**Siapkan USB/External Drive minimal 2GB!**

---

### C. Create Download Summary

Buat file `download-summary.txt`:

```bash
cat > ~/Downloads/download-summary.txt << EOF
===================================
DOWNLOAD SUMMARY
Date: $(date)
===================================

INSTALLERS:
- Python 3.11.7
- MySQL 8.0.35
- Tesseract OCR 5.3.3

PYTHON PACKAGES:
- Total files: $(ls python_packages/*.whl | wc -l)
- Total size: $(du -sh python_packages/ | awk '{print $1}')

PROJECT FILES:
- app.py: $(ls -lh project-plat-detection-dude/app.py | awk '{print $5}')
- YOLO model: $(ls -lh project-plat-detection-dude/models/best.pt | awk '{print $5}')

TOTAL SIZE: $(du -sh . | awk '{print $1}')

Next Step: Copy to USB/External Drive
EOF

cat ~/Downloads/download-summary.txt
```

---

# BAGIAN 3: TRANSFER KE KOMPUTER OFFLINE

## 3.1 Persiapan Media Transfer

### A. Pilih Media Transfer

**Opsi 1: USB Flash Drive (Recommended)**
- Kapasitas: 16GB minimum, 32GB recommended
- Brand: SanDisk, Kingston, dll
- Speed: USB 3.0 (lebih cepat)

**Opsi 2: External HDD/SSD**
- Kapasitas: Bebas (min 10GB free)
- Lebih cepat untuk file besar
- Lebih aman (tidak gampang hilang)

**Opsi 3: Network Transfer (jika ada LAN)**
- Copy via LAN cable
- Lebih cepat dari USB 2.0
- Setup: Share folder di source computer

---

### B. Format USB (Opsional)

**macOS:**
```bash
# List USB devices
diskutil list

# Format (ganti disk2 dengan disk Anda!)
diskutil eraseDisk exFAT INSTALLER disk2
```

**Windows:**
```
1. Insert USB
2. Right-click USB di "This PC"
3. Format → File system: exFAT
4. Start
```

**Pilih exFAT** karena compatible Windows + macOS!

---

## 3.2 Struktur Folder

Buat struktur folder yang rapi di USB:

```
USB:/ALPR-Offline/
├── 0-README.txt                 # Instruksi
├── 1-Installers/
│   ├── python-3.11.7.pkg
│   ├── mysql-8.0.35.dmg
│   └── tesseract-setup.exe
├── 2-Python-Packages/
│   └── *.whl (50-100 files)
├── 3-Project/
│   └── project-plat-detection-dude/
├── 4-Documentation/
│   ├── PANDUAN_LENGKAP_INSTALASI_OFFLINE.md
│   ├── INSTALASI_DAN_TOOLS.md
│   └── MATERI_PRESENTASI.md
└── download-summary.txt
```

---

## 3.3 Copy Files

### A. Copy ke USB

**macOS:**
```bash
# Mount USB (biasanya auto-mount)
# Check di /Volumes/

# Create structure
USB_PATH="/Volumes/INSTALLER"
mkdir -p "$USB_PATH/ALPR-Offline"/{1-Installers,2-Python-Packages,3-Project,4-Documentation}

# Copy installers
cp ~/Downloads/installers/* "$USB_PATH/ALPR-Offline/1-Installers/"

# Copy Python packages
cp ~/Downloads/python_packages/* "$USB_PATH/ALPR-Offline/2-Python-Packages/"

# Copy project
cp -r ~/Downloads/project-plat-detection-dude "$USB_PATH/ALPR-Offline/3-Project/"

# Copy documentation
cp ~/Documents/DWI/project-plat-detection-dude/*.md "$USB_PATH/ALPR-Offline/4-Documentation/"

# Create README
cat > "$USB_PATH/ALPR-Offline/0-README.txt" << EOF
ALPR SYSTEM - OFFLINE INSTALLER
================================

CONTENTS:
1. Installers/ - Python, MySQL, Tesseract
2. Python-Packages/ - All Python libraries (.whl files)
3. Project/ - Application source code
4. Documentation/ - Installation guides

NEXT STEP:
1. Copy folder "ALPR-Offline" to target computer
2. Follow: 4-Documentation/PANDUAN_LENGKAP_INSTALASI_OFFLINE.md
3. Start with section "BAGIAN 4: INSTALASI OFFLINE"

Date created: $(date)
EOF

# Sync (wait until complete)
sync

echo "✅ Copy complete! Safe to eject USB."
```

**Windows (PowerShell):**
```powershell
# Check USB drive letter (e.g., E:)
Get-Volume

# Set USB path
$USB = "E:\ALPR-Offline"

# Create structure
New-Item -Path "$USB\1-Installers" -ItemType Directory -Force
New-Item -Path "$USB\2-Python-Packages" -ItemType Directory -Force
New-Item -Path "$USB\3-Project" -ItemType Directory -Force
New-Item -Path "$USB\4-Documentation" -ItemType Directory -Force

# Copy files
Copy-Item "$env:USERPROFILE\Downloads\installers\*" -Destination "$USB\1-Installers\" -Recurse
Copy-Item "$env:USERPROFILE\Downloads\python_packages\*" -Destination "$USB\2-Python-Packages\" -Recurse
Copy-Item "$env:USERPROFILE\Downloads\project-plat-detection-dude" -Destination "$USB\3-Project\" -Recurse

# Create README
@"
ALPR SYSTEM - OFFLINE INSTALLER
================================

CONTENTS:
1. Installers/ - Python, MySQL, Tesseract
2. Python-Packages/ - All Python libraries (.whl files)
3. Project/ - Application source code
4. Documentation/ - Installation guides

NEXT STEP:
1. Copy folder "ALPR-Offline" to target computer
2. Follow: 4-Documentation/PANDUAN_LENGKAP_INSTALASI_OFFLINE.md
3. Start with section "BAGIAN 4: INSTALASI OFFLINE"

Date created: $(Get-Date)
"@ | Out-File -FilePath "$USB\0-README.txt" -Encoding UTF8

Write-Host "✅ Copy complete! Safe to eject USB."
```

---

### B. Verify Copy

**Check file count:**
```bash
# Count files in each folder
find /Volumes/INSTALLER/ALPR-Offline -type f | wc -l

# Should be 100-150 files
```

**Check total size:**
```bash
du -sh /Volumes/INSTALLER/ALPR-Offline

# Should be 700MB - 1.3GB
```

**Spot check critical files:**
```bash
USB_PATH="/Volumes/INSTALLER/ALPR-Offline"

# Check installers
ls -lh "$USB_PATH/1-Installers/"
# Should see: python*.pkg, mysql*.dmg, tesseract*.exe

# Check Python packages
ls "$USB_PATH/2-Python-Packages/"opencv*.whl
# Should see opencv wheel file

# Check project
ls "$USB_PATH/3-Project/project-plat-detection-dude/app.py"
# Should exist

# Check YOLO model
ls -lh "$USB_PATH/3-Project/project-plat-detection-dude/models/best.pt"
# Should be ~6MB
```

---

### C. Eject USB Safely

**macOS:**
```bash
diskutil eject /Volumes/INSTALLER
```

**Windows:**
```
Right-click USB → Eject
```

**Atau:**
```
Taskbar → Safely Remove Hardware → Eject USB
```

---

# BAGIAN 4: INSTALASI OFFLINE

**⚠️ SEKARANG ANDA DI KOMPUTER OFFLINE (TANPA INTERNET)**

## 4.1 Install Python

### A. macOS

**Step 1: Copy USB files ke komputer**
```bash
# Insert USB
# Copy ke home directory
cp -r /Volumes/INSTALLER/ALPR-Offline ~/

cd ~/ALPR-Offline/1-Installers
```

**Step 2: Install Python**
```bash
# Double click .pkg file
# Atau via terminal:
sudo installer -pkg python-3.11.7-macos11.pkg -target /
```

**Step 3: Follow installer wizard**
- Continue → Continue → Agree
- Install for all users
- Enter admin password
- Wait ~2 minutes
- Close

**Step 4: Verify**
```bash
python3 --version
# Output: Python 3.11.7

which python3
# Output: /usr/local/bin/python3
```

**Troubleshooting:**
- Jika command not found: Restart Terminal
- Jika permission denied: Check admin password

---

### B. Windows

**Step 1: Copy USB files**
```
1. Insert USB
2. Copy "ALPR-Offline" folder to C:\
3. Path sekarang: C:\ALPR-Offline\
```

**Step 2: Run installer**
```
1. Open C:\ALPR-Offline\1-Installers\
2. Double-click python-3.11.7-amd64.exe
3. ⚠️ PENTING: Centang "Add Python 3.11 to PATH"
4. Click "Install Now"
5. Wait ~3 minutes
6. Click "Close"
```

**Step 3: Verify**
```cmd
python --version
REM Output: Python 3.11.7

where python
REM Output: C:\Users\...\AppData\Local\Programs\Python\Python311\python.exe
```

**Troubleshooting:**
- Command not found → Restart Command Prompt
- Permission denied → Right-click installer → Run as Administrator
- PATH not set → Add manual:
  ```
  Control Panel → System → Advanced → Environment Variables
  Path → Edit → Add: C:\Users\[Username]\AppData\Local\Programs\Python\Python311\
  ```

---

## 4.2 Install MySQL

### A. macOS

**Step 1: Install DMG**
```bash
cd ~/ALPR-Offline/1-Installers

# Double-click mysql-8.0.35-macos13-x86_64.dmg
# Atau:
open mysql-8.0.35-macos13-x86_64.dmg
```

**Step 2: Follow installer**
```
1. Double-click .pkg in mounted volume
2. Continue → Continue → Agree
3. Install type: Default
4. ⚠️ IMPORTANT: Set root password! (Catat password ini!)
5. Legacy Password Encryption: ✓ (untuk compatibility)
6. Start MySQL server: ✓
7. Finish
```

**Step 3: Start MySQL**
```bash
# Via System Preferences
System Preferences → MySQL → Start MySQL Server

# Atau via terminal:
sudo /usr/local/mysql/support-files/mysql.server start
```

**Step 4: Add MySQL to PATH**
```bash
# Edit ~/.zshrc atau ~/.bash_profile
echo 'export PATH="/usr/local/mysql/bin:$PATH"' >> ~/.zshrc

# Reload
source ~/.zshrc
```

**Step 5: Verify**
```bash
mysql --version
# Output: mysql Ver 8.0.35 for macos13 on x86_64

# Test login
mysql -u root -p
# Enter password
# mysql> prompt muncul = SUCCESS!
```

---

### B. Windows

**Step 1: Run MSI installer**
```
1. Double-click mysql-installer-community-8.0.35.0.msi
2. Setup type: "Developer Default" (recommended)
3. Click "Next" → "Execute" (download skipped karena offline)
4. Wait ~10 minutes
```

**Step 2: Configuration**
```
1. Type and Networking:
   ✓ Standalone MySQL Server
   Port: 3306 (default)

2. Authentication Method:
   ● Use Legacy Authentication (untuk compatibility)

3. Accounts and Roles:
   Root Password: [SET PASSWORD KUAT!] (CATAT!)
   Add User (optional): admin / password

4. Windows Service:
   ✓ Configure MySQL Server as Windows Service
   ✓ Start at System Startup
   Service Name: MySQL80

5. Apply Configuration → Execute → Finish
```

**Step 3: Add to PATH**
```
Control Panel → System → Advanced system settings
→ Environment Variables
→ System variables → Path → Edit → New
→ Add: C:\Program Files\MySQL\MySQL Server 8.0\bin
→ OK → OK → OK
```

**Step 4: Verify**
```cmd
mysql --version
REM Output: mysql Ver 8.0.35 for Win64 on x86_64

REM Test login
mysql -u root -p
REM Enter password
REM mysql> prompt = SUCCESS!
```

**Troubleshooting:**
- Service not starting:
  ```cmd
  net start MySQL80
  ```
- Port 3306 already in use:
  ```cmd
  netstat -ano | findstr :3306
  taskkill /PID [PID] /F
  ```

---

## 4.3 Install Tesseract OCR

### A. macOS

**Via Homebrew (jika sudah ada Homebrew):**
```bash
brew install tesseract
```

**Manual dari .pkg (jika ada):**
```bash
sudo installer -pkg tesseract-5.3.3.pkg -target /
```

**Verify:**
```bash
tesseract --version
# Output: tesseract 5.3.3

which tesseract
# Output: /usr/local/bin/tesseract
```

**Download language data (optional, jika ada internet):**
```bash
# English
curl -L -o /usr/local/share/tessdata/eng.traineddata \
https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
```

---

### B. Windows

**Step 1: Run installer**
```
1. Double-click tesseract-ocr-w64-setup-5.3.3.20231005.exe
2. Language: English
3. Install location: C:\Program Files\Tesseract-OCR (default)
4. Components: ✓ English language data
5. Install
6. Finish
```

**Step 2: Add to PATH**
```
Control Panel → System → Advanced → Environment Variables
→ System variables → Path → Edit → New
→ Add: C:\Program Files\Tesseract-OCR
→ OK
```

**Step 3: Verify**
```cmd
tesseract --version
REM Output: tesseract 5.3.3

where tesseract
REM Output: C:\Program Files\Tesseract-OCR\tesseract.exe
```

**Troubleshooting:**
- Command not found: Restart Command Prompt
- PATH not set: Add manually (lihat Step 2)

---

## 4.4 Install Python Libraries

### A. Install dari Folder Offline

**macOS/Linux:**
```bash
cd ~/ALPR-Offline/2-Python-Packages

# Install SEMUA libraries sekaligus
pip3 install --no-index --find-links=. opencv-python Flask PyMySQL numpy Pillow pytesseract DBUtils python-dotenv scipy scikit-image colorlog typing-extensions

# Atau install satu per satu:
pip3 install --no-index --find-links=. opencv_python-4.8.1.78*.whl
pip3 install --no-index --find-links=. Flask-2.3.3*.whl
# ... dst
```

**Windows:**
```cmd
cd C:\ALPR-Offline\2-Python-Packages

REM Install semua
pip install --no-index --find-links=. opencv-python Flask PyMySQL numpy Pillow pytesseract DBUtils python-dotenv scipy scikit-image colorlog typing-extensions
```

**Penjelasan:**
- `--no-index` = Tidak cari ke internet
- `--find-links=.` = Cari .whl di folder current

---

### B. Verify Installation

```bash
# List installed packages
pip3 list

# Should see:
# opencv-python  4.8.1.78
# Flask          2.3.3
# PyMySQL        1.1.0
# numpy          1.24.3
# ... dst
```

**Test import (Python REPL):**
```bash
python3

>>> import cv2
>>> print(cv2.__version__)
4.8.1.78

>>> import flask
>>> print(flask.__version__)
2.3.3

>>> import pytesseract
>>> # No error = SUCCESS!

>>> exit()
```

---

## 4.5 Verifikasi Instalasi

### Checklist Verifikasi

```bash
# Check Python
python3 --version
# ✅ Python 3.11.7

# Check MySQL
mysql --version
# ✅ mysql Ver 8.0.35

# Check Tesseract
tesseract --version
# ✅ tesseract 5.3.3

# Check pip packages
pip3 list | grep -E "opencv|Flask|PyMySQL|numpy"
# ✅ Semua ada

# Check MySQL service
# macOS:
ps aux | grep mysql | grep -v grep
# Windows:
sc query MySQL80
# ✅ Running

echo "🎉 SEMUA TOOLS TERINSTALL!"
```

---

**[LANJUTAN DI COMMENT BERIKUTNYA - File terlalu panjang untuk satu response]**

**Progress so far:**
- ✅ Bagian 1-4 Complete (Persiapan sampai Instalasi)
- 🔄 Bagian 5-13 (Setup Database, Config, Running, Testing, Troubleshooting)

**Mau saya lanjutkan semua bagian sisanya (5-13) sekarang?** 🚀