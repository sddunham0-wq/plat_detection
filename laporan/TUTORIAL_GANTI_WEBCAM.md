# 📹 TUTORIAL: Cara Ganti CCTV ke Webcam Laptop

## 🎯 Masalah yang Terjadi

Sistem mencoba koneksi ke CCTV dulu (dari `config.py`), kalau CCTV tidak tersedia maka **hang/stuck** sebelum fallback ke webcam. Anda ingin **langsung pakai webcam laptop**.

---

## ✅ SOLUSI 1: Edit File `app.py` (MUDAH - RECOMMENDED)

### Langkah 1: Buka file `app.py`

Cari fungsi `initialize_camera()` di baris **176-213**

### Langkah 2: Comment CCTV Code

Tambahkan tanda `#` di depan baris 183-196 untuk **disable CCTV**:

```python
def initialize_camera():
    """
    Penjelasan SMK: Seperti 'nyalakan kamera' dan pastikan bisa dipakai
    Coba connect ke CCTV dulu, kalau gagal pakai webcam laptop
    """
    global camera, system_status

    # ★ DISABLE CCTV - LANGSUNG PAKAI WEBCAM ★
    # try:
    #     # Coba connect ke CCTV dari config
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

    # Fallback ke webcam laptop
    try:
        logger.info("🎥 Mencoba webcam laptop...")
        camera = cv2.VideoCapture(0)  # ← INDEX 0 untuk webcam
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

### Langkah 3: Save & Restart

```bash
python app.py
```

Sekarang sistem akan **langsung pakai webcam**, tidak coba CCTV lagi.

---

## ✅ SOLUSI 2: Ganti Index Webcam (Kalau Index 0 Gagal)

Beberapa laptop punya **multiple webcam** atau webcam di index **1** atau **2**.

### Coba Index Berbeda:

Ganti baris **201** dari:
```python
camera = cv2.VideoCapture(0)  # Index 0
```

Jadi:
```python
camera = cv2.VideoCapture(1)  # Coba index 1
```

Atau:
```python
camera = cv2.VideoCapture(2)  # Coba index 2
```

---

## ✅ SOLUSI 3: Test Webcam Dulu (Sebelum Run App)

Buat file test sederhana `test_webcam.py`:

```python
import cv2

print("Testing webcam...")

for i in range(3):
    print(f"\nTrying index {i}...")
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Webcam index {i} WORKS!")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            break
        else:
            print(f"❌ Index {i} opened but can't read frame")
    else:
        print(f"❌ Index {i} not available")

    cap.release()

print("\nTest selesai!")
```

Run:
```bash
python test_webcam.py
```

Output akan kasih tau **index mana yang work**.

---

## 🔧 SOLUSI 4: Troubleshooting Kalau Webcam Tetap Gagal

### 1. **Check Aplikasi Lain**
Tutup aplikasi yang pakai webcam:
- Zoom
- Microsoft Teams
- FaceTime
- Skype
- Google Meet di browser

### 2. **Check macOS Privacy Settings**

```
System Preferences → Security & Privacy → Privacy → Camera
```

Pastikan **Terminal** atau **Python** atau **VS Code** ada di list dan **dicentang**.

### 3. **Restart Webcam Service** (macOS)

```bash
sudo killall VDCAssistant
sudo killall AppleCameraAssistant
```

Lalu restart aplikasi.

### 4. **Check dengan System Report**

```
Apple Menu → About This Mac → System Report → Camera
```

Pastikan webcam terdeteksi di sistem.

---

## 📊 Ringkasan

| Solusi | Kesulitan | Waktu | Cocok Untuk |
|--------|-----------|-------|-------------|
| **Solusi 1** (Comment CCTV code) | ⭐ Mudah | 2 menit | **RECOMMENDED** |
| **Solusi 2** (Ganti index) | ⭐ Mudah | 1 menit | Kalau index 0 gagal |
| **Solusi 3** (Test script) | ⭐⭐ Sedang | 5 menit | Debugging |
| **Solusi 4** (Troubleshooting) | ⭐⭐⭐ Sulit | 10 menit | Kalau semua gagal |

---

## 🎥 Bonus: Cara Kembali ke CCTV

Kalau nanti mau pakai CCTV lagi, **hapus tanda `#`** di baris yang di-comment tadi.

---

## ❓ FAQ (Frequently Asked Questions)

### Q: Kenapa webcam index 0 tidak work?
**A:** Beberapa laptop punya external webcam atau multi-camera. Coba index 1 atau 2.

### Q: Apakah bisa pakai external USB webcam?
**A:** Bisa! External webcam biasanya di index 1 atau 2. Gunakan SOLUSI 3 untuk detect index-nya.

### Q: Webcam sudah work tapi gambar gelap?
**A:** Itu normal untuk deteksi plat. Sistem akan otomatis adjust brightness saat OCR.

### Q: Bisa pakai IP camera selain CCTV?
**A:** Bisa! Ganti `camera_url` di config.py dengan URL IP camera Anda (format: `http://IP:PORT/stream`)

---

## 📞 Masih Gagal?

Coba jalankan app dan **copy paste output error** ke sini. Saya akan bantu debug lebih lanjut.

**Yang perlu dicek:**
1. Error message di terminal
2. Output dari `test_webcam.py`
3. macOS Privacy Settings screenshot

---

## 📝 Catatan Penting

- Tutorial ini khusus untuk **macOS**
- Untuk Windows/Linux, prinsipnya sama tapi troubleshooting berbeda
- Pastikan OpenCV sudah terinstall: `pip install opencv-python`
- Kalau masih error, pastikan Python version >= 3.8

---

**Good luck! 🚀**

import cv2

print("Testing webcam...")

for i in range(3):
    print(f"\nTrying index {i}...")
    cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Webcam index {i} WORKS!")
            print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        else:
            print(f"❌ Index {i} opened but can't read frame")
    else:
        print(f"❌ Index {i} not available")
    
    cap.release()

print("\nTest selesai!")