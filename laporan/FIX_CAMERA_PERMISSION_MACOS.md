# Fix: Camera Permission Denied di macOS

## ❌ Error yang Muncul

```
OpenCV: not authorized to capture video (status 0), requesting...
OpenCV: camera failed to properly initialize!
ERROR:__main__:❌ Tidak ada kamera yang tersedia
```

## 🔍 Penyebab

macOS **memblokir akses kamera** untuk aplikasi Python/Terminal karena alasan privasi dan keamanan. Ini adalah fitur keamanan macOS yang harus diizinkan secara manual.

---

## ✅ Solusi 1: Berikan Permission via System Settings (RECOMMENDED)

### Langkah 1: Buka System Settings

**macOS Ventura (13.0+) / Sonoma (14.0+):**
1. Klik **Apple Menu** () di pojok kiri atas
2. Pilih **System Settings...**
3. Di sidebar kiri, klik **Privacy & Security**
4. Scroll ke bawah, cari dan klik **Camera**

**macOS Monterey (12.0) dan lebih lama:**
1. Klik **Apple Menu** () di pojok kiri atas
2. Pilih **System Preferences...**
3. Klik **Security & Privacy**
4. Klik tab **Privacy**
5. Di sidebar kiri, pilih **Camera**

### Langkah 2: Tambahkan Terminal/Python

Di halaman Camera, Anda akan melihat daftar aplikasi yang boleh akses kamera.

**Jika menggunakan Terminal/iTerm2:**
1. Cari **Terminal** atau **iTerm2** di list
2. Centang checkbox di sebelah Terminal/iTerm2
3. Jika tidak ada, klik tombol **+** (plus)
4. Browse ke `/Applications/Utilities/Terminal.app`
5. Klik **Open**
6. Centang checkbox Terminal

**Jika menggunakan VS Code:**
1. Cari **Visual Studio Code** atau **Code** di list
2. Centang checkbox di sebelahnya
3. Jika tidak ada, klik **+**
4. Browse ke `/Applications/Visual Studio Code.app`
5. Klik **Open**
6. Centang checkbox VS Code

**Jika menggunakan Python langsung:**
1. Klik tombol **+**
2. Tekan **Cmd + Shift + G** (Go to Folder)
3. Ketik salah satu path berikut:
   ```
   /usr/local/bin/python3
   ```
   atau
   ```
   /Library/Frameworks/Python.framework/Versions/3.11/bin/python3
   ```
   atau cek path Python Anda dengan:
   ```bash
   which python3
   ```
4. Klik **Go**
5. Pilih file `python3`
6. Klik **Open**
7. Centang checkbox Python

### Langkah 3: Restart Terminal/VS Code

**PENTING:** Setelah memberikan permission, Anda **HARUS restart** aplikasi!

```bash
# Tutup Terminal/VS Code
# Buka lagi Terminal/VS Code
# Jalankan ulang program
python3 app.py
```

---

## ✅ Solusi 2: Reset Camera Permission Database (Jika Solusi 1 Gagal)

Kadang macOS "stuck" dan perlu reset database permission.

### Langkah 1: Tutup Semua Aplikasi

Tutup Terminal, VS Code, Python, dll.

### Langkah 2: Reset TCC Database

Buka Terminal BARU, jalankan:

```bash
# Reset permission database untuk Camera
tccutil reset Camera
```

**Output:**
```
Successfully reset Camera approval status for all clients
```

### Langkah 3: Restart Mac (Opsional tapi Recommended)

```bash
sudo shutdown -r now
```

Atau restart manual via Apple Menu > Restart.

### Langkah 4: Ulangi Solusi 1

Setelah restart, berikan permission lagi via System Settings (lihat Solusi 1).

---

## ✅ Solusi 3: Jalankan dengan Admin Permission (Temporary)

**WARNING:** Ini solusi temporary, bukan solusi permanen!

```bash
# Jalankan dengan sudo (admin)
sudo python3 app.py
```

**Output yang Diharapkan:**
```
Password: [ketik password Mac Anda]
🎥 Initializing camera...
✅ Webcam laptop berhasil terhubung
```

**Kekurangan:**
- Harus pakai `sudo` setiap kali run
- Security risk (running as root)
- Tidak recommended untuk development

---

## ✅ Solusi 4: Code-Signing Python Binary (Advanced)

Kalau permission tetap tidak bisa, coba code-sign Python binary Anda.

### Langkah 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

### Langkah 2: Create Self-Signed Certificate

1. Buka **Keychain Access** (Applications > Utilities)
2. Klik **Keychain Access** menu > **Certificate Assistant** > **Create a Certificate**
3. Isi form:
   - **Name:** PythonCameraCert
   - **Identity Type:** Self-Signed Root
   - **Certificate Type:** Code Signing
4. Klik **Create**
5. Klik **Continue**
6. Klik **Done**

### Langkah 3: Sign Python Binary

```bash
# Cek path Python Anda
which python3
# Output: /usr/local/bin/python3 (atau path lain)

# Code-sign Python
codesign -s "PythonCameraCert" -f /usr/local/bin/python3

# Verifikasi
codesign -dv /usr/local/bin/python3
```

### Langkah 4: Ulangi Solusi 1

Berikan permission lagi via System Settings.

---

## ✅ Solusi 5: Pakai Python Virtual Environment (RECOMMENDED untuk Development)

Virtual environment kadang lebih mudah dapat permission.

### Langkah 1: Buat Virtual Environment

```bash
cd /Users/andra/Documents/DWI/project-plat-detection-dude

# Buat venv
python3 -m venv venv

# Aktifkan
source venv/bin/activate
```

### Langkah 2: Install Requirements di venv

```bash
# Install semua library
pip install -r requirements.txt
```

### Langkah 3: Jalankan dari venv

```bash
# Pastikan venv aktif (ada tulisan (venv) di prompt)
python app.py
```

### Langkah 4: Berikan Permission untuk Python di venv

Jika tetap diminta permission:
1. Buka System Settings > Privacy & Security > Camera
2. Klik **+**
3. Browse ke path venv Python:
   ```
   /Users/andra/Documents/DWI/project-plat-detection-dude/venv/bin/python
   ```
4. Centang checkbox

---

## 🧪 Test Camera Permission

Buat file test sederhana `test_camera_permission.py`:

```python
import cv2
import sys

print("🔍 Testing camera permission...")

try:
    # Try to open camera
    camera = cv2.VideoCapture(0)

    if camera.isOpened():
        ret, frame = camera.read()

        if ret and frame is not None:
            print("✅ SUCCESS! Camera accessible")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")

            # Show frame for 3 seconds
            cv2.imshow('Camera Test', frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print("❌ FAILED: Camera opened but can't read frame")
            print("   Possible causes:")
            print("   - Camera is being used by another app")
            print("   - Camera hardware issue")
    else:
        print("❌ FAILED: Camera not accessible")
        print("   Possible causes:")
        print("   - Permission denied")
        print("   - No camera available")
        print("   - Camera disabled in System Settings")

    camera.release()

except Exception as e:
    print(f"❌ ERROR: {e}")
    print("   This usually means permission denied")

print("\n📝 Next steps:")
print("   1. Open System Settings > Privacy & Security > Camera")
print("   2. Add Terminal/VS Code/Python to allowed apps")
print("   3. Restart Terminal/VS Code")
print("   4. Run this test again")
```

Jalankan:
```bash
python3 test_camera_permission.py
```

---

## 🔧 Troubleshooting Lanjutan

### Problem: "Operation not permitted" saat reset TCC

```bash
# Solusi: Disable SIP temporarily (ADVANCED - RISKY!)

# 1. Restart Mac dalam Recovery Mode
#    (Hold Cmd+R saat booting)
# 2. Utilities > Terminal
# 3. Jalankan:
csrutil disable

# 4. Restart Mac
# 5. Reset TCC
tccutil reset Camera

# 6. Enable SIP lagi
#    Restart ke Recovery Mode
csrutil enable
```

**WARNING:** Disable SIP mengurangi keamanan Mac! Hanya lakukan jika sangat perlu.

---

### Problem: Camera terdeteksi tapi gambar hitam

**Penyebab:**
- App lain sedang pakai camera (Zoom, FaceTime, Photo Booth)
- Camera physically covered (sticker, tape)

**Solusi:**
```bash
# 1. Cek app yang pakai camera
lsof | grep "Camera"

# 2. Tutup semua app yang pakai camera
# 3. Test lagi
python3 test_camera_permission.py
```

---

### Problem: "Camera not found" di MacBook dengan external webcam

```python
# Edit app.py, coba berbagai camera index
camera = cv2.VideoCapture(0)  # Built-in camera
# atau
camera = cv2.VideoCapture(1)  # External USB webcam
```

---

## 📊 Checklist Fix Camera Permission

- [ ] Buka System Settings > Privacy & Security > Camera
- [ ] Tambahkan Terminal/VS Code/Python ke allowed apps
- [ ] Centang checkbox aplikasi tersebut
- [ ] **Restart** Terminal/VS Code/aplikasi
- [ ] Test dengan `test_camera_permission.py`
- [ ] Jika gagal, coba `tccutil reset Camera`
- [ ] Restart Mac
- [ ] Ulangi langkah 1-5
- [ ] Jika masih gagal, coba virtual environment
- [ ] Jika tetap gagal, coba code-signing (advanced)

---

## ✅ Solusi Tercepat (TL;DR)

```bash
# 1. Buka System Settings
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"

# 2. Tambahkan Terminal ke daftar
#    (manual - klik + > pilih Terminal.app)

# 3. Restart Terminal
# Cmd+Q (quit) lalu buka lagi

# 4. Test
cd /Users/andra/Documents/DWI/project-plat-detection-dude
python3 app.py

# Seharusnya berhasil! ✅
```

---

## 📝 Catatan Penting

1. **macOS 13+ lebih ketat permission** - Harus manual approve setiap app
2. **Terminal vs Python** - Kadang perlu approve keduanya
3. **VS Code vs Terminal** - Permission berbeda untuk setiap app
4. **Virtual Environment** - venv Python dianggap app terpisah
5. **Restart Wajib** - Permission baru aktif setelah restart app

---

## 🔗 Link Referensi

- [Apple Developer - Camera Privacy](https://developer.apple.com/documentation/avfoundation/cameras_and_media_capture/requesting_authorization_for_media_capture_on_macos)
- [OpenCV macOS Permission Issues](https://github.com/opencv/opencv/issues/16352)
- [tccutil Documentation](https://github.com/jacobsalmela/tccutil)

---

**Dibuat:** 15 November 2025
**Versi:** 1.0
**Platform:** macOS Monterey, Ventura, Sonoma
**Project:** Sistem Deteksi Plat Nomor Kendaraan
