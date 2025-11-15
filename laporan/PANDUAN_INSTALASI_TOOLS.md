# Panduan Lengkap Instalasi Tools - Project Deteksi Plat Nomor

## Daftar Isi
1. [Penjelasan Umum](#penjelasan-umum)
2. [Software Wajib](#software-wajib)
3. [Python Libraries](#python-libraries)
4. [Database Setup](#database-setup)
5. [Model YOLO](#model-yolo)
6. [Tesseract OCR](#tesseract-ocr)
7. [Panduan Download Tools di Windows](#panduan-download-tools-di-windows)
8. [Setup VS Code](#setup-vs-code)
9. [Langkah Instalasi](#langkah-instalasi)
10. [Verifikasi Instalasi](#verifikasi-instalasi)

---

## Penjelasan Umum

Project ini adalah **Sistem Deteksi Plat Nomor Kendaraan** yang menggunakan kamera untuk mendeteksi plat mobil/motor secara otomatis. Ibarat seperti security guard digital yang bisa membaca plat nomor dan memutuskan apakah kendaraan boleh masuk atau tidak.

**Cara Kerja Sederhana:**
1. Kamera ambil gambar kendaraan
2. Sistem cari plat nomor di gambar (pakai AI YOLO)
3. Baca tulisan di plat (pakai OCR/Tesseract)
4. Cek database: apakah plat terdaftar?
5. Buka/tutup palang pintu otomatis

---

## Software Wajib

### 1. Python 3.8 - 3.11
**Apa itu?** Bahasa pemrograman yang dipakai project ini

**Kenapa perlu?** Semua kode program ditulis dalam Python

**Download:**
- **macOS**: Sudah terinstall, atau download dari [python.org](https://www.python.org/downloads/)
- **Windows**: Download installer dari [python.org](https://www.python.org/downloads/)
- **Linux**: Biasanya sudah ada, atau `sudo apt install python3`

**Cara install di macOS:**
```bash
# Cek versi Python yang sudah ada
python3 --version

# Jika belum ada atau versi lama, install via Homebrew
brew install python@3.11
```

**Cara install di Windows:**
1. Download installer dari python.org
2. **PENTING:** Centang "Add Python to PATH" saat install
3. Klik Install Now
4. Buka Command Prompt, ketik `python --version` untuk cek

---

### 2. MySQL Database Server
**Apa itu?** Tempat menyimpan data (seperti Excel tapi lebih canggih)

**Kenapa perlu?** Untuk menyimpan:
- Daftar kendaraan yang terdaftar (plat nomor, pemilik, dll)
- Log akses (siapa masuk kapan)
- Riwayat deteksi

**Download & Install:**

**macOS:**
```bash
# Install MySQL via Homebrew
brew install mysql

# Start MySQL service
brew services start mysql

# Setup password (ikuti petunjuk)
mysql_secure_installation
```

**Windows (Pakai Laragon - RECOMMENDED):**
1. Download Laragon Full dari [laragon.org](https://laragon.org/download/)
2. Install Laragon (sudah include MySQL + PHP + Apache)
3. Jalankan Laragon
4. MySQL otomatis jalan di background

**Catatan:** Project ini sudah dikonfigurasi untuk MySQL di localhost:3306 (default Laragon)

---

### 3. Git (Opsional)
**Apa itu?** Tools untuk download dan kelola kode

**Kenapa perlu?** Untuk clone/download project dari GitHub

**Install:**
```bash
# macOS
brew install git

# Windows - download dari git-scm.com
# Atau sudah include di Laragon
```

---

## Python Libraries

Berikut adalah **library Python** yang perlu diinstall. Library itu seperti "tools tambahan" yang dipakai program.

### Libraries & Fungsinya:

| Library | Versi | Fungsi | Penjelasan Sederhana |
|---------|-------|--------|----------------------|
| **opencv-python** | 4.8.1.78 | Computer Vision | Untuk olah gambar/video dari kamera |
| **pytesseract** | 0.3.10 | OCR Engine | Untuk baca tulisan di plat nomor |
| **numpy** | 1.24.3 | Math Operations | Untuk hitung-hitungan matematika |
| **Pillow** | 10.0.1 | Image Processing | Untuk edit/olah gambar |
| **Flask** | 2.3.3 | Web Framework | Untuk buat website/dashboard |
| **PyMySQL** | 1.1.0 | MySQL Driver | Untuk koneksi ke database MySQL |
| **DBUtils** | 3.0.3 | Database Pool | Untuk manage koneksi database |
| **python-dotenv** | 1.0.0 | Config Manager | Untuk baca file .env (password dll) |
| **scipy** | 1.11.3 | Scientific Computing | Untuk operasi matematika tingkat lanjut |
| **scikit-image** | 0.21.0 | Image Enhancement | Untuk perbaiki kualitas gambar |
| **colorlog** | 6.7.0 | Logging | Untuk tampilkan log warna-warni |
| **typing-extensions** | 4.8.0 | Type Hints | Untuk code quality |
| **ultralytics** | - | YOLO Framework | Untuk AI deteksi plat (YOLO v8) |

### Cara Install Semua Libraries:

Project sudah menyediakan file `requirements.txt` yang berisi semua library.

```bash
# Masuk ke folder project
cd /path/to/project-plat-detection-dude

# Install semua library sekaligus
pip3 install -r requirements.txt

# Atau di Windows
pip install -r requirements.txt
```

**Waktu Install:** Sekitar 5-15 menit tergantung internet

**Ukuran Total:** Sekitar 1-2 GB (karena ada OpenCV, YOLO, dll)

---

## Database Setup

### 1. Buat Database MySQL

```sql
-- Buka MySQL console
mysql -u root -p

-- Buat database baru
CREATE DATABASE sistem_parkir_smk;

-- Gunakan database
USE sistem_parkir_smk;
```

### 2. Buat Tabel-tabel

**Tabel 1: kendaraan_terdaftar** (Daftar kendaraan yang boleh masuk)
```sql
CREATE TABLE kendaraan_terdaftar (
    id_kendaraan INT AUTO_INCREMENT PRIMARY KEY,
    nomor_plat VARCHAR(20) UNIQUE NOT NULL,
    nama_pemilik VARCHAR(100) NOT NULL,
    jenis_kendaraan ENUM('mobil', 'motor', 'truk') NOT NULL,
    nomor_hp VARCHAR(20),
    status ENUM('aktif', 'nonaktif') DEFAULT 'aktif',
    tanggal_daftar TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Tabel 2: log_akses_masuk** (Catatan siapa masuk kapan)
```sql
CREATE TABLE log_akses_masuk (
    id_log INT AUTO_INCREMENT PRIMARY KEY,
    plat_terdeteksi VARCHAR(20) NOT NULL,
    tingkat_yakin FLOAT,
    status_akses ENUM('boleh_masuk', 'ditolak', 'manual_override', 'error') NOT NULL,
    aksi_palang ENUM('opened', 'closed', 'manual') NOT NULL,
    waktu_deteksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    path_foto VARCHAR(255),
    catatan TEXT
);
```

### 3. Test Data (Opsional)

Tambahkan beberapa kendaraan untuk testing:

```sql
INSERT INTO kendaraan_terdaftar (nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp) VALUES
('B1234ABC', 'Budi Santoso', 'mobil', '081234567890'),
('B5678XYZ', 'Siti Rahayu', 'motor', '081298765432'),
('D9999AAA', 'Ahmad Wijaya', 'mobil', '081312345678');
```

---

## Model YOLO

### Apa itu Model YOLO?

YOLO (You Only Look Once) adalah **AI untuk deteksi objek**. Seperti mata manusia yang bisa langsung tahu "ini mobil, ini plat nomor" hanya dengan sekali lihat.

Project ini pakai **2 model YOLO:**

### 1. YOLOv8n.pt (Model Umum)
**Fungsi:** Deteksi mobil/motor/kendaraan (kotak besar)

**Ukuran:** ~6 MB

**Auto-download:** Model ini akan otomatis didownload saat pertama kali dijalankan

**Lokasi:** Otomatis tersimpan di cache Ultralytics

### 2. best.pt (Model Custom untuk Plat)
**Fungsi:** Deteksi plat nomor spesifik (kotak kecil di area plat)

**Ukuran:** ~6 MB

**Lokasi:** `models/best.pt`

**Status:** Sudah ada di project (tidak perlu download)

**Cara verifikasi:**
```bash
ls -lh models/best.pt

# Output seharusnya:
# -rw-r--r--  1 user staff 6.2M Oct 20 12:33 models/best.pt
```

---

## Tesseract OCR

### Apa itu Tesseract?

Tesseract adalah **software untuk baca tulisan di gambar** (OCR = Optical Character Recognition).

**Analogi:** Seperti manusia yang bisa baca tulisan di foto, Tesseract bisa baca tulisan "B 1234 ABC" dari gambar plat nomor.

### Install Tesseract:

**macOS:**
```bash
# Install via Homebrew
brew install tesseract

# Install bahasa Indonesia (opsional)
brew install tesseract-lang
```

**Windows:**
1. Download installer dari [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install ke `C:\Program Files\Tesseract-OCR`
3. **PENTING:** Tambahkan ke PATH:
   - Buka System Properties > Environment Variables
   - Edit PATH
   - Tambahkan `C:\Program Files\Tesseract-OCR`

**Linux:**
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

### Verifikasi Install:
```bash
tesseract --version

# Output seharusnya:
# tesseract 5.x.x
```

---

## Panduan Download Tools di Windows

### 🪟 Panduan Lengkap untuk Windows (Step-by-Step dengan Screenshot)

Berikut adalah panduan **download dan install** semua tools yang diperlukan di Windows. Cocok untuk pemula!

---

### 1. Download dan Install Python di Windows

**Langkah 1: Download Python**
1. Buka browser (Chrome/Firefox)
2. Kunjungi [https://www.python.org/downloads/](https://www.python.org/downloads/)
3. Klik tombol kuning **"Download Python 3.11.x"** (versi terbaru yang stabil)
4. File akan terdownload (nama: `python-3.11.x-amd64.exe`, ukuran ~30 MB)

**Langkah 2: Install Python**
1. Double-click file `python-3.11.x-amd64.exe`
2. **⚠️ PENTING:** Centang "Add Python to PATH" (paling bawah)
   - Tanpa ini, Python tidak bisa diakses dari Command Prompt!
3. Pilih **"Install Now"**
4. Tunggu proses instalasi (~2 menit)
5. Klik **"Close"** setelah selesai

**Langkah 3: Verifikasi**
1. Buka **Command Prompt** (tekan Windows + R, ketik `cmd`, Enter)
2. Ketik: `python --version`
3. Harusnya muncul: `Python 3.11.x`
4. Ketik: `pip --version`
5. Harusnya muncul: `pip 23.x.x from ...`

✅ **Python berhasil terinstall!**

---

### 2. Download dan Install Laragon (MySQL, Apache, PHP)

**Apa itu Laragon?**
- Paket lengkap server lokal (MySQL + Apache + PHP + NodeJS)
- Lebih mudah dari install MySQL manual
- Cocok untuk development

**Langkah 1: Download Laragon Full**
1. Buka [https://laragon.org/download/](https://laragon.org/download/)
2. Klik **"Laragon - Full (64-bit)"**
   - File: `laragon-wamp.exe`
   - Ukuran: ~200 MB
3. Tunggu download selesai

**Langkah 2: Install Laragon**
1. Double-click file `laragon-wamp.exe`
2. Pilih bahasa **English** > OK
3. **Destination Folder:** Biarkan default `C:\laragon` > Next
4. **Auto Virtual Hosts:** Centang (biarkan default) > Next
5. Klik **Install**
6. Tunggu proses (~5 menit)
7. Centang **"Run Laragon"** > Finish

**Langkah 3: Jalankan Laragon**
1. Buka Laragon (icon di Desktop atau Start Menu)
2. Klik **"Start All"** (pojok kanan bawah)
3. Status akan berubah jadi:
   - ✅ Apache: Started
   - ✅ MySQL: Started
4. MySQL sekarang jalan di background!

**Langkah 4: Verifikasi MySQL**
1. Di Laragon, klik **"Menu"** > **"MySQL"** > **"MySQL Console"**
2. Password: **kosong** (tekan Enter aja)
3. Ketik: `SHOW DATABASES;`
4. Harusnya muncul list database
5. Ketik: `exit` untuk keluar

✅ **MySQL berhasil terinstall!**

**Catatan:**
- MySQL berjalan di `localhost:3306`
- Username: `root`
- Password: kosong (default Laragon)

---

### 3. Download dan Install Tesseract OCR

**Langkah 1: Download Tesseract**
1. Buka [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Scroll ke bagian **"Tesseract at UB Mannheim"**
3. Klik link **"tesseract-ocr-w64-setup-5.x.x.exe"** (versi 64-bit terbaru)
4. File akan terdownload (ukuran ~60 MB)

**Langkah 2: Install Tesseract**
1. Double-click file installer
2. Klik **Next** beberapa kali
3. **Destination Folder:** Biarkan default `C:\Program Files\Tesseract-OCR`
4. **Select Components:**
   - ✅ Centang "English" (sudah default)
   - ✅ Centang "Additional languages" > pilih "Indonesian" (opsional)
5. Klik **Install**
6. Tunggu proses (~1 menit)
7. Klik **Finish**

**Langkah 3: Tambahkan Tesseract ke PATH**

Ini **sangat penting** agar Python bisa akses Tesseract!

**Cara Mudah (Windows 10/11):**
1. Tekan **Windows + R**
2. Ketik: `sysdm.cpl` > Enter
3. Klik tab **"Advanced"**
4. Klik **"Environment Variables"**
5. Di bagian **"System variables"**, cari **"Path"** > klik **"Edit"**
6. Klik **"New"**
7. Ketik: `C:\Program Files\Tesseract-OCR`
8. Klik **OK** > **OK** > **OK**

**Langkah 4: Verifikasi**
1. Buka **Command Prompt BARU** (penting: buka yang baru!)
2. Ketik: `tesseract --version`
3. Harusnya muncul: `tesseract 5.x.x`

✅ **Tesseract berhasil terinstall!**

---

### 4. Download dan Install Git (Opsional)

**Langkah 1: Download Git**
1. Buka [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Download akan otomatis mulai (64-bit)
3. File: `Git-2.x.x-64-bit.exe` (ukuran ~50 MB)

**Langkah 2: Install Git**
1. Double-click installer
2. Klik **Next** beberapa kali (pakai default settings)
3. **Adjusting your PATH:** Pilih **"Git from the command line and also from 3rd-party software"**
4. Klik **Next** > **Install**
5. Tunggu proses (~2 menit)
6. Klik **Finish**

**Langkah 3: Verifikasi**
```cmd
git --version
# Output: git version 2.x.x
```

✅ **Git berhasil terinstall!**

---

### 5. Download Project dari GitHub

**Cara 1: Pakai Git (Recommended)**
```cmd
# Buka Command Prompt
# Pindah ke folder Documents
cd %USERPROFILE%\Documents

# Clone project
git clone https://github.com/username/project-plat-detection-dude.git

# Masuk folder project
cd project-plat-detection-dude
```

**Cara 2: Download ZIP (Tanpa Git)**
1. Buka link GitHub project di browser
2. Klik tombol hijau **"Code"**
3. Klik **"Download ZIP"**
4. Extract file ZIP ke folder `Documents\project-plat-detection-dude`
5. Buka **Command Prompt**
6. Ketik:
   ```cmd
   cd %USERPROFILE%\Documents\project-plat-detection-dude
   ```

✅ **Project berhasil didownload!**

---

### 6. Install VS Code (Opsional tapi Recommended)

Akan dijelaskan lengkap di bagian [Setup VS Code](#setup-vs-code)

---

## Setup VS Code

### 💻 Cara Setup dan Jalankan Project di Visual Studio Code

VS Code adalah **text editor** yang paling populer untuk coding. Seperti Microsoft Word tapi untuk programmer!

---

### 1. Download dan Install VS Code

**Langkah 1: Download**
1. Buka [https://code.visualstudio.com/](https://code.visualstudio.com/)
2. Klik **"Download for Windows"**
3. File: `VSCodeUserSetup-x64-1.x.x.exe` (ukuran ~90 MB)

**Langkah 2: Install**
1. Double-click installer
2. Klik **Next**
3. **Select Additional Tasks:**
   - ✅ Centang "Add to PATH"
   - ✅ Centang "Create a desktop icon"
   - ✅ Centang "Register Code as an editor for supported file types"
4. Klik **Next** > **Install**
5. Tunggu proses (~2 menit)
6. Klik **Finish**

✅ **VS Code berhasil terinstall!**

---

### 2. Install Extension Python di VS Code

Extension itu seperti **"plugin"** yang menambah fitur VS Code.

**Langkah 1: Buka VS Code**
1. Buka VS Code (icon di Desktop)
2. Akan muncul Welcome screen

**Langkah 2: Install Extension Python**
1. Klik icon **Extensions** di sidebar kiri (icon kotak-kotak)
   - Atau tekan **Ctrl + Shift + X**
2. Di kotak pencarian, ketik: `Python`
3. Cari extension **"Python"** by Microsoft (yang paling atas)
4. Klik **"Install"**
5. Tunggu proses (~1 menit)

**Langkah 3: Install Extension Tambahan (Opsional)**

Extension yang berguna untuk project ini:

| Extension | Fungsi | Cara Install |
|-----------|--------|--------------|
| **Pylance** | Autocomplete Python lebih pintar | Search "Pylance" > Install |
| **MySQL** | Lihat database MySQL dari VS Code | Search "MySQL" > Install |
| **Git Graph** | Visualisasi Git history | Search "Git Graph" > Install |
| **Python Indent** | Auto indent Python | Search "Python Indent" > Install |
| **Better Comments** | Warna-warni di comment | Search "Better Comments" > Install |

✅ **Extension berhasil terinstall!**

---

### 3. Buka Project di VS Code

**Langkah 1: Open Folder**
1. Di VS Code, klik **"File"** > **"Open Folder"**
2. Browse ke folder project:
   ```
   C:\Users\[NamaKamu]\Documents\project-plat-detection-dude
   ```
3. Klik **"Select Folder"**
4. VS Code akan buka semua file project di sidebar

**Langkah 2: Trust Workspace (jika diminta)**
1. Akan muncul popup "Do you trust the authors..."
2. Klik **"Yes, I trust the authors"**

✅ **Project terbuka di VS Code!**

---

### 4. Setup Python Interpreter

**Langkah 1: Pilih Python Interpreter**
1. Tekan **Ctrl + Shift + P** (Command Palette)
2. Ketik: `Python: Select Interpreter`
3. Pilih Python versi yang terinstall (contoh: `Python 3.11.x 64-bit`)
4. Akan muncul di status bar bawah

**Langkah 2: Buat Virtual Environment di VS Code**

**Apa itu Virtual Environment?**
- Seperti "kotak isolasi" untuk Python libraries
- Agar tidak bentrok dengan project lain

**Cara Buat:**
1. Tekan **Ctrl + Shift + `** (backtick) untuk buka Terminal
2. Di Terminal VS Code, ketik:
   ```cmd
   python -m venv venv
   ```
3. Tunggu proses (~30 detik)
4. Folder `venv` akan muncul di sidebar

**Langkah 3: Aktifkan Virtual Environment**

VS Code akan otomatis deteksi virtual environment dan tanya:
- "We noticed a new virtual environment has been created. Do you want to select it?"
- Klik **"Yes"**

Atau aktifkan manual:
```cmd
# Di Terminal VS Code
venv\Scripts\activate

# Prompt akan berubah jadi:
# (venv) PS C:\Users\...\project-plat-detection-dude>
```

✅ **Virtual Environment aktif!** (ada tulisan `(venv)` di terminal)

---

### 5. Install Dependencies di VS Code

**Langkah 1: Pastikan Virtual Environment Aktif**
- Cek ada `(venv)` di terminal
- Jika belum, ketik: `venv\Scripts\activate`

**Langkah 2: Install Requirements**
```cmd
pip install -r requirements.txt
```

**Proses:**
- VS Code akan download dan install semua library
- Waktu: ~5-15 menit
- Ukuran: ~1.5 GB

**Progress di Terminal:**
```
Collecting opencv-python==4.8.1.78
  Downloading opencv_python-4.8.1.78-cp311-cp311-win_amd64.whl (38.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.1/38.1 MB 5.2 MB/s eta 0:00:00
...
Successfully installed opencv-python-4.8.1.78 pytesseract-0.3.10 ...
```

✅ **Semua library terinstall!**

---

### 6. Setup Database di VS Code

**Cara 1: Pakai Laragon MySQL Console**
1. Buka Laragon
2. Klik **"Start All"**
3. Klik **"Menu"** > **"MySQL"** > **"MySQL Console"**
4. Copy-paste SQL dari bagian [Database Setup](#database-setup)

**Cara 2: Pakai VS Code Extension MySQL**
1. Install extension **"MySQL"** by cweijan
2. Klik icon **MySQL** di sidebar kiri
3. Klik **"+"** untuk add connection
4. Isi:
   - **Host:** localhost
   - **Port:** 3306
   - **Username:** root
   - **Password:** (kosong)
5. Klik **"Connect"**
6. Klik kanan connection > **"New Query"**
7. Copy-paste SQL dari bagian [Database Setup](#database-setup)
8. Tekan **Ctrl + Enter** untuk run

✅ **Database ready!**

---

### 7. Setup File .env di VS Code

**Langkah 1: Buat File .env**
1. Di Explorer (sidebar kiri), klik kanan > **"New File"**
2. Nama file: `.env`
3. Copy isi dari `.env.example` (jika ada)

**Langkah 2: Edit .env**
```env
# Secret Key - Generate pakai Python
SECRET_KEY=your-secret-key-here

# Database MySQL
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=sistem_parkir_smk

# Kamera (jika pakai CCTV, jika tidak biarkan default)
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=

# Flask Server
FLASK_PORT=8080
```

**Generate SECRET_KEY:**
```cmd
# Di Terminal VS Code
python -c "import secrets; print(secrets.token_hex(32))"

# Copy hasil output (contoh: a1b2c3d4e5f6...)
# Paste ke SECRET_KEY di .env
```

**Langkah 3: Save File**
- Tekan **Ctrl + S**

✅ **File .env ready!**

---

### 8. Run Project di VS Code

**Cara 1: Via Terminal VS Code**
```cmd
# Pastikan virtual environment aktif (ada tulisan venv)
python app.py
```

**Output di Terminal:**
```
🚀 Starting Vehicle Access Control System...
📁 Folder created/verified: static/screenshots
📁 Folder created/verified: static/gambarplat
📁 Folder created/verified: logs
🎥 Initializing camera...
✅ Webcam laptop berhasil terhubung
✅ Camera initialized successfully
🌐 Starting web server on http://localhost:8080
 * Running on http://0.0.0.0:8080
```

**Cara 2: Via VS Code Debugger (Recommended)**

**Langkah 1: Buat Launch Configuration**
1. Klik icon **"Run and Debug"** di sidebar kiri (icon play)
2. Klik **"create a launch.json file"**
3. Pilih **"Python File"**
4. File `.vscode/launch.json` akan terbuka

**Langkah 2: Edit launch.json**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Flask App",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app.py",
            "console": "integratedTerminal",
            "env": {
                "FLASK_ENV": "development",
                "FLASK_DEBUG": "1"
            },
            "justMyCode": true
        }
    ]
}
```

**Langkah 3: Run dengan Debugger**
1. Buka file `app.py`
2. Tekan **F5** atau klik **"Start Debugging"**
3. Program akan jalan dengan debugger aktif

**Keuntungan pakai Debugger:**
- ✅ Bisa pause program pakai **breakpoint**
- ✅ Bisa inspect variable
- ✅ Auto-reload saat edit code
- ✅ Lebih mudah debug error

---

### 9. Akses Website di Browser

**Langkah 1: Buka Browser**
1. Buka Chrome/Firefox
2. Ketik di address bar: `http://localhost:8080`
3. Tekan Enter

**Langkah 2: Halaman Utama**
- Seharusnya muncul halaman **"Sistem Deteksi Plat Nomor"**
- Ada video feed dari kamera
- Ada tabel log akses
- Ada statistik

✅ **Project berhasil jalan!**

---

### 10. Tips VS Code untuk Development

**Keyboard Shortcuts yang Berguna:**

| Shortcut | Fungsi |
|----------|--------|
| **Ctrl + `** | Toggle Terminal |
| **Ctrl + Shift + P** | Command Palette |
| **Ctrl + P** | Quick Open File |
| **F5** | Start Debugging |
| **Ctrl + C** | Stop Running Program (di terminal) |
| **Ctrl + /** | Comment/Uncomment Line |
| **Alt + ↑/↓** | Move Line Up/Down |
| **Ctrl + D** | Select Next Occurrence |
| **Ctrl + F** | Find in File |
| **Ctrl + Shift + F** | Find in All Files |

**VS Code Features:**

1. **IntelliSense (Autocomplete):**
   - Ketik huruf pertama function/variable
   - VS Code akan suggest kode yang relevan
   - Tekan **Tab** untuk accept

2. **Go to Definition:**
   - Klik kanan function/class
   - Pilih **"Go to Definition"** (atau **F12**)
   - Langsung jump ke source code

3. **Problems Panel:**
   - Tekan **Ctrl + Shift + M**
   - Lihat semua error/warning di code

4. **Integrated Git:**
   - Klik icon **Source Control** di sidebar
   - Lihat file yang berubah
   - Commit langsung dari VS Code

---

### 11. Troubleshooting VS Code

**Problem: Terminal tidak recognize 'python'**
```cmd
# Solusi: Gunakan python3 atau py
python3 app.py
# atau
py app.py
```

**Problem: Virtual environment tidak aktif**
```cmd
# Aktifkan manual
venv\Scripts\activate

# Jika error "cannot be loaded because running scripts is disabled"
# Buka PowerShell as Administrator, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Problem: Module not found setelah install**
```cmd
# Pastikan install di virtual environment yang benar
# 1. Cek venv aktif (ada tulisan (venv))
# 2. Reinstall
pip install -r requirements.txt
```

**Problem: Port 8080 already in use**
```cmd
# Edit .env, ganti port
FLASK_PORT=8081
```

---

## Langkah Instalasi

### Step-by-Step untuk Pemula:

#### 1. Download Project
```bash
# Clone dari GitHub (jika ada)
git clone https://github.com/username/project-plat-detection-dude.git

# Atau download ZIP dan extract
```

#### 2. Masuk Folder Project
```bash
cd project-plat-detection-dude
```

#### 3. Buat Virtual Environment (RECOMMENDED)
```bash
# Buat virtual environment
python3 -m venv venv

# Aktifkan (macOS/Linux)
source venv/bin/activate

# Aktifkan (Windows)
venv\Scripts\activate

# Setelah aktif, prompt akan berubah jadi (venv)
```

**Kenapa pakai venv?** Agar library Python tidak bentrok dengan project lain.

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Setup Database
```bash
# Jalankan MySQL
# macOS:
brew services start mysql

# Windows (Laragon):
# Buka Laragon > Start All

# Buat database (jalankan SQL di atas)
mysql -u root -p < database/setup.sql
```

#### 6. Setup Environment Variables
```bash
# Copy file .env.example jadi .env
cp .env.example .env

# Edit file .env (pakai text editor)
nano .env
```

**Isi .env:**
```env
# Secret Key (generate random)
SECRET_KEY=your-secret-key-here

# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=sistem_parkir_smk

# Kamera (jika pakai CCTV)
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=your-camera-password

# Flask
FLASK_PORT=8080
```

#### 7. Test Run
```bash
# Jalankan program
python3 app.py

# Buka browser
http://localhost:8080
```

---

## Verifikasi Instalasi

### Checklist Tools:

Jalankan command berikut untuk verifikasi:

```bash
# 1. Python
python3 --version
# ✅ Harus: Python 3.8.x - 3.11.x

# 2. pip
pip3 --version
# ✅ Harus: pip 20.x atau lebih baru

# 3. MySQL
mysql --version
# ✅ Harus: mysql Ver 8.x atau 5.7

# 4. Tesseract
tesseract --version
# ✅ Harus: tesseract 5.x atau 4.x

# 5. Git (opsional)
git --version
# ✅ Harus: git version 2.x
```

### Test Python Libraries:

Buat file `test_imports.py`:
```python
# Test semua library penting
try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except:
    print("❌ OpenCV tidak terinstall")

try:
    import pytesseract
    print("✅ Pytesseract: OK")
except:
    print("❌ Pytesseract tidak terinstall")

try:
    from ultralytics import YOLO
    print("✅ YOLO/Ultralytics: OK")
except:
    print("❌ Ultralytics tidak terinstall")

try:
    import flask
    print("✅ Flask:", flask.__version__)
except:
    print("❌ Flask tidak terinstall")

try:
    import pymysql
    print("✅ PyMySQL:", pymysql.__version__)
except:
    print("❌ PyMySQL tidak terinstall")

print("\n✅ Semua library OK!" if all else "\n⚠️ Ada library yang kurang")
```

Jalankan:
```bash
python3 test_imports.py
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'cv2'"
**Solusi:**
```bash
pip3 install opencv-python
```

### Error: "pytesseract.pytesseract.TesseractNotFoundError"
**Solusi:** Tesseract belum terinstall atau tidak ada di PATH
```bash
# macOS
brew install tesseract

# Windows - tambahkan C:\Program Files\Tesseract-OCR ke PATH
```

### Error: "Can't connect to MySQL server"
**Solusi:** MySQL belum jalan
```bash
# macOS
brew services start mysql

# Windows (Laragon)
# Buka Laragon > Start All
```

### Error: "Access denied for user 'root'@'localhost'"
**Solusi:** Password MySQL salah, edit file `.env`

---

## Ringkasan Ukuran Download

| Item | Ukuran | Waktu Download (10 Mbps) |
|------|--------|--------------------------|
| Python 3.11 | ~30 MB | 30 detik |
| MySQL (Laragon Full) | ~200 MB | 3 menit |
| Tesseract OCR | ~60 MB | 1 menit |
| Python Libraries | ~1.5 GB | 20 menit |
| Model YOLO | ~12 MB | 10 detik |
| **TOTAL** | **~1.8 GB** | **~25 menit** |

---

## Kontak & Support

Jika ada masalah saat instalasi:
1. Cek error message dengan teliti
2. Google error message tersebut
3. Baca dokumentasi official (Python, MySQL, dll)
4. Tanya ke forum (Stack Overflow, Reddit)

---

**Dibuat:** 15 November 2025
**Versi:** 1.0
**Project:** Sistem Deteksi Plat Nomor Kendaraan
