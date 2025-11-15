# Tutorial: Cara Ganti Kamera dari CCTV ke Webcam

## 📹 Penjelasan Umum

Project ini bisa menggunakan **2 jenis kamera**:
1. **CCTV IP Camera** - Kamera CCTV yang terhubung via jaringan (WiFi/LAN)
2. **Webcam Laptop/USB** - Kamera laptop built-in atau webcam USB external

**Kapan Pakai Webcam?**
- Testing/development tanpa CCTV
- CCTV tidak tersedia atau rusak
- Demo project di laptop
- Biaya lebih murah untuk belajar

---

## 🔧 Cara Kerja Sistem Kamera

### Kode di `app.py`:

```python
def initialize_camera():
    """Coba connect ke CCTV dulu, kalau gagal pakai webcam"""
    global camera, system_status

    try:
        # STEP 1: Coba CCTV dari config.py
        camera_url = config.CAMERA_URL
        logger.info(f"🎥 Mencoba koneksi ke CCTV: {camera_url}")
        camera = cv2.VideoCapture(camera_url)

        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ CCTV berhasil terhubung")
                return True
    except Exception as e:
        logger.warning(f"⚠️ CCTV tidak tersedia: {e}")

    # STEP 2: Fallback ke Webcam Laptop
    try:
        logger.info("🎥 Mencoba webcam laptop...")
        camera = cv2.VideoCapture(0)  # 0 = webcam default
        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ Webcam laptop berhasil terhubung")
                return True
    except Exception as e:
        logger.error(f"❌ Webcam juga gagal: {e}")

    return False
```

**Penjelasan Sederhana:**
- Program coba CCTV dulu
- Kalau CCTV gagal → otomatis pakai webcam
- Kalau webcam juga gagal → error

**GOOD NEWS:** Sistem sudah **auto-fallback** ke webcam! 🎉

---

## ✅ Metode 1: Auto-Fallback (RECOMMENDED - Paling Mudah)

Cara paling mudah adalah **biarkan CCTV gagal** dan sistem otomatis pakai webcam.

### Langkah 1: Edit File `.env`

Buka file `.env` di project folder, edit bagian kamera:

```env
# Kamera CCTV - SET KE IP YANG SALAH AGAR GAGAL CONNECT
CAMERA_HOST=192.168.999.999  # IP invalid → CCTV pasti gagal
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=wrong-password  # Password salah juga bisa

# Atau bisa dikosongkan
CAMERA_HOST=
CAMERA_PASSWORD=
```

### Langkah 2: Jalankan Program

```bash
# Di Terminal/Command Prompt
python app.py
```

### Output yang Muncul:

```
🎥 Mencoba koneksi ke CCTV: rtsp://admin:wrong-password@192.168.999.999:5503/...
⚠️ CCTV tidak tersedia: [WinError 10060] Connection timed out
🎥 Mencoba webcam laptop...
✅ Webcam laptop berhasil terhubung
✅ Camera initialized successfully
🌐 Starting web server on http://localhost:8080
```

✅ **Selesai!** Sistem otomatis pakai webcam tanpa edit kode!

---

## ⚙️ Metode 2: Force Webcam via Config (Lebih Teknis)

Kalau mau **paksa sistem pakai webcam langsung** tanpa coba CCTV dulu.

### Langkah 1: Edit File `config.py`

Buka file `config.py`, cari bagian ini:

```python
@property
def CAMERA_URL(self):
    """Generate RTSP URL dari konfigurasi"""
    return f"rtsp://{self.CAMERA_USER}:{self.CAMERA_PASSWORD}@{self.CAMERA_HOST}:{self.CAMERA_PORT}/cam/realmonitor?channel={self.CAMERA_CHANNEL}&subtype={self.CAMERA_SUBTYPE}"
```

**Ganti jadi:**

```python
@property
def CAMERA_URL(self):
    """Force webcam dengan return integer 0"""
    return 0  # 0 = webcam default, 1 = webcam USB external
```

### Langkah 2: Jalankan Program

```bash
python app.py
```

### Output yang Muncul:

```
🎥 Mencoba koneksi ke CCTV: 0
✅ Webcam laptop berhasil terhubung
✅ Camera initialized successfully
```

✅ **Selesai!** Sistem langsung pakai webcam tanpa coba CCTV.

---

## 🔄 Metode 3: Edit Kode `app.py` (Paling Fleksibel)

Kalau mau kontrol penuh, edit langsung kode di `app.py`.

### Opsi A: Hapus Bagian CCTV (Webcam Only)

Buka file `app.py`, cari fungsi `initialize_camera()`:

**BEFORE (Original Code):**
```python
def initialize_camera():
    global camera, system_status
    try:
        # Coba connect ke CCTV dari config
        from config import config
        camera_url = config.CAMERA_URL
        logger.info(f"🎥 Mencoba koneksi ke CCTV: {camera_url}")
        camera = cv2.VideoCapture(camera_url)

        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ CCTV berhasil terhubung")
                return True
    except Exception as e:
        logger.warning(f"⚠️ CCTV tidak tersedia: {e}")

    # Fallback ke webcam laptop
    try:
        logger.info("🎥 Mencoba webcam laptop...")
        camera = cv2.VideoCapture(0)
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

**AFTER (Webcam Only - Hapus bagian CCTV):**
```python
def initialize_camera():
    """Initialize webcam only (CCTV disabled)"""
    global camera, system_status

    try:
        logger.info("🎥 Menghubungkan ke webcam...")
        camera = cv2.VideoCapture(0)  # 0 = webcam default

        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ Webcam berhasil terhubung")
                return True

        logger.error("❌ Webcam tidak bisa dibuka")
        return False

    except Exception as e:
        logger.error(f"❌ Error webcam: {e}")
        system_status['camera_connected'] = False
        return False
```

### Opsi B: Tambah Parameter Pilihan

Tambahkan parameter untuk switch kamera via command line:

**Edit `app.py` (tambahkan di bagian bawah):**

```python
if __name__ == '__main__':
    import sys

    # Check command line arguments
    use_webcam = '--webcam' in sys.argv or '-w' in sys.argv

    try:
        logger.info("🚀 Starting Vehicle Access Control System...")
        create_required_folders()

        # Initialize camera dengan pilihan
        logger.info("🎥 Initializing camera...")

        if use_webcam:
            logger.info("📷 FORCE WEBCAM MODE (via --webcam flag)")
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                system_status['camera_connected'] = True
                logger.info("✅ Webcam initialized successfully")
            else:
                logger.warning("⚠️ Webcam initialization failed")
        else:
            # Normal flow: try CCTV first, then fallback to webcam
            if initialize_camera():
                logger.info("✅ Camera initialized successfully")
            else:
                logger.warning("⚠️ Camera initialization failed - running without camera")

        # Start Flask app
        logger.info(f"🌐 Starting web server on http://localhost:{config.FLASK_PORT}")
        app.run(debug=True, host='0.0.0.0', port=config.FLASK_PORT, threaded=True)

    except Exception as e:
        logger.error(f"❌ Fatal error starting system: {e}")
    finally:
        if camera:
            camera.release()
        logger.info("👋 System shutdown complete")
```

**Cara Pakai:**

```bash
# Normal mode (coba CCTV dulu)
python app.py

# Force webcam mode
python app.py --webcam
# atau
python app.py -w
```

---

## 🎥 Webcam Index (Multiple Webcam)

Kalau punya **lebih dari 1 webcam** (laptop built-in + USB external):

```python
camera = cv2.VideoCapture(0)  # Webcam laptop built-in
camera = cv2.VideoCapture(1)  # Webcam USB external pertama
camera = cv2.VideoCapture(2)  # Webcam USB external kedua
```

### Cara Test Webcam Mana yang Aktif:

Buat file `test_webcam.py`:

```python
import cv2

print("🔍 Testing available webcams...")

for i in range(5):  # Test webcam index 0-4
    print(f"\n📹 Testing webcam index {i}...")

    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Webcam {i} AVAILABLE")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")

            # Tampilkan preview 2 detik
            cv2.imshow(f'Webcam {i}', frame)
            cv2.waitKey(2000)  # 2 detik
            cv2.destroyAllWindows()
        else:
            print(f"❌ Webcam {i} opened but can't read frame")

        cap.release()
    else:
        print(f"❌ Webcam {i} NOT AVAILABLE")

print("\n✅ Test selesai!")
```

Jalankan:
```bash
python test_webcam.py
```

Output contoh:
```
📹 Testing webcam index 0...
✅ Webcam 0 AVAILABLE
   Resolution: 1280x720

📹 Testing webcam index 1...
✅ Webcam 1 AVAILABLE
   Resolution: 640x480

📹 Testing webcam index 2...
❌ Webcam 2 NOT AVAILABLE
```

---

## 🔧 Troubleshooting Webcam

### Problem 1: Webcam Tidak Terdeteksi

**Error Message:**
```
❌ Webcam juga gagal: [Error -1]
❌ Tidak ada kamera yang tersedia
```

**Solusi:**

1. **Cek Permission Camera (Windows 10/11):**
   - Settings > Privacy > Camera
   - Toggle ON untuk "Allow apps to access your camera"
   - Scroll ke bawah, cari Python atau Terminal, toggle ON

2. **Cek Webcam di Device Manager:**
   - Windows + X > Device Manager
   - Expand "Cameras" atau "Imaging devices"
   - Cek ada warning/error icon?
   - Klik kanan > Update driver

3. **Test Webcam dengan Windows Camera App:**
   - Buka app "Camera" dari Start Menu
   - Kalau Camera app juga gagal → masalah hardware/driver
   - Kalau Camera app jalan → masalah Python/OpenCV

4. **Reinstall OpenCV:**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python==4.8.1.78
   ```

---

### Problem 2: Webcam Terbuka tapi Gambar Hitam

**Solusi:**

1. **Tutup aplikasi lain yang pakai webcam:**
   - Zoom, Skype, Teams, Discord
   - Browser dengan camera permission
   - Aplikasi streaming (OBS, Streamlabs)

2. **Restart webcam service:**
   ```bash
   # Di Python, tambahkan delay setelah open
   camera = cv2.VideoCapture(0)
   time.sleep(1)  # Tunggu 1 detik
   ret, frame = camera.read()
   ```

---

### Problem 3: Resolution Terlalu Kecil/Besar

**Solusi - Set Resolution Manual:**

Edit `app.py`, tambahkan setelah `cv2.VideoCapture(0)`:

```python
camera = cv2.VideoCapture(0)

# Set resolution (optional)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Width 1280px
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Height 720px
camera.set(cv2.CAP_PROP_FPS, 30)             # 30 FPS

logger.info(f"📹 Webcam resolution: {camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
```

**Resolution Recommended:**
- **720p (HD)**: 1280x720 - Balance antara kualitas dan performa
- **480p (SD)**: 640x480 - Lebih smooth, deteksi tetap bagus
- **1080p (Full HD)**: 1920x1080 - Paling tajam tapi berat

---

### Problem 4: FPS Rendah / Lag

**Solusi:**

1. **Reduce Resolution** (lihat di atas)

2. **Increase Frame Skip** di `generate_video_frames()`:

```python
# Edit app.py, cari bagian ini
DETECTION_INTERVAL = 2  # Process detection every 2 frames

# Ganti jadi
DETECTION_INTERVAL = 5  # Skip 4 frames, process 1 frame (lebih smooth)
```

3. **Lower JPEG Quality:**

```python
# Cari bagian ini di generate_video_frames()
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

# Ganti jadi
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Lower quality = faster
```

---

## 📊 Perbandingan CCTV vs Webcam

| Aspek | CCTV IP Camera | Webcam Laptop/USB |
|-------|----------------|-------------------|
| **Harga** | Rp 500K - 5 juta | Rp 0 (built-in) - 500K |
| **Kualitas Gambar** | Bagus (1080p-4K) | Cukup (480p-1080p) |
| **Jarak Deteksi** | Jauh (5-20 meter) | Dekat (1-3 meter) |
| **Setup** | Kompleks (WiFi, RTSP) | Mudah (plug & play) |
| **Cocok untuk** | Production, outdoor | Development, testing |
| **Reliability** | Tinggi | Medium |
| **Power** | POE/adapter | USB power |

**Kesimpulan:**
- **Development/Learning:** Pakai **Webcam** ✅
- **Production/Deployment:** Pakai **CCTV** ✅

---

## 🔄 Cara Switch Balik ke CCTV

Kalau sudah punya CCTV dan mau balik pakai CCTV:

### Langkah 1: Edit `.env`

```env
# Isi dengan IP CCTV yang benar
CAMERA_HOST=192.168.1.203     # IP CCTV di jaringan
CAMERA_PORT=5503               # Port RTSP
CAMERA_USER=admin              # Username CCTV
CAMERA_PASSWORD=password123    # Password CCTV (ganti!)
```

### Langkah 2: Test Koneksi CCTV

```bash
# Test ping ke IP CCTV
ping 192.168.1.203

# Harusnya reply
Reply from 192.168.1.203: bytes=32 time=2ms TTL=64
```

### Langkah 3: Jalankan Program

```bash
python app.py
```

**Output yang Diharapkan:**
```
🎥 Mencoba koneksi ke CCTV: rtsp://admin:***@192.168.1.203:5503/...
✅ CCTV berhasil terhubung
```

### Langkah 4: Troubleshoot CCTV

**Jika CCTV Gagal Connect:**

1. **Cek IP dengan nmap/Advanced IP Scanner:**
   - Download Advanced IP Scanner (Windows)
   - Scan jaringan 192.168.1.0/24
   - Cari device dengan port 554 atau 5503

2. **Cek RTSP URL di browser:**
   - Buka VLC Media Player
   - Media > Open Network Stream
   - Paste URL: `rtsp://admin:password@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0`
   - Kalau VLC bisa play → URL benar
   - Kalau VLC error → URL salah atau CCTV offline

3. **Cek Firewall:**
   - Windows Firewall mungkin block port RTSP
   - Temporarily disable untuk test

---

## 📝 Checklist Setup Webcam

- [ ] Python 3.8+ terinstall
- [ ] OpenCV terinstall (`pip install opencv-python`)
- [ ] Webcam permission enabled (Windows Privacy Settings)
- [ ] Tidak ada app lain yang pakai webcam
- [ ] Test webcam dengan `test_webcam.py`
- [ ] Edit `.env` atau `config.py` untuk force webcam
- [ ] Jalankan `python app.py`
- [ ] Buka browser `http://localhost:8080`
- [ ] Video feed muncul dari webcam ✅

---

## 🎯 Quick Reference

### Force Webcam - Cepat (Edit .env)
```env
CAMERA_HOST=invalid-ip
```

### Force Webcam - Via Flag
```bash
python app.py --webcam
```

### Test Webcam
```bash
python test_webcam.py
```

### Set Resolution
```python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Multiple Webcam
```python
camera = cv2.VideoCapture(0)  # Built-in
camera = cv2.VideoCapture(1)  # USB
```

---

**Dibuat:** 15 November 2025
**Versi:** 1.0
**Project:** Sistem Deteksi Plat Nomor Kendaraan
