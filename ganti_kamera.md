# Cara Ganti Kamera dari CCTV ke Webcam Laptop

## 🎯 Apa yang akan kita lakukan?
Kita akan mengubah sistem deteksi plat nomor yang tadinya pakai **kamera CCTV** jadi pakai **webcam laptop** (kamera bawaan laptop).

**Kenapa perlu ganti?**
- Kalau CCTV rusak atau tidak ada
- Mau test di laptop sendiri
- Lebih mudah untuk development

---

## 🚀 Cara Termudah (Pakai File .env)

### Langkah 1: Buka File `.env`

**Di Windows:**
```bash
notepad .env
```

**Di Mac/Linux:**
```bash
nano .env
# atau kalau pakai VS Code
code .env
```

### Langkah 2: Matikan Setting CCTV

Cari baris yang ada tulisan `CAMERA_HOST`, `CAMERA_PORT`, dll.

**SEBELUM (Pakai CCTV):**
```env
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=password123
CAMERA_CHANNEL=1
CAMERA_SUBTYPE=0
```

**SESUDAH (Pakai Webcam):**
Tambahkan tanda `#` di depan setiap baris:
```env
#CAMERA_HOST=192.168.1.203
#CAMERA_PORT=5503
#CAMERA_USER=admin
#CAMERA_PASSWORD=password123
#CAMERA_CHANNEL=1
#CAMERA_SUBTYPE=0
```

**Penjelasan:**
Tanda `#` artinya "jangan dipakai". Jadi settingan CCTV jadi tidak aktif.

### Langkah 3: Save dan Tutup File

**Di Notepad:** Klik File → Save → Tutup
**Di VS Code:** Tekan Ctrl+S (Windows) atau Cmd+S (Mac)
**Di Nano:** Tekan Ctrl+X, lalu Y, lalu Enter

### Langkah 4: Restart Aplikasi

**Kalau aplikasi masih jalan:**
1. Tekan Ctrl+C untuk stop
2. Jalankan ulang: `python3 app.py`

**Kalau aplikasi belum jalan:**
```bash
python3 app.py
```

### ✅ Cek Hasilnya

Lihat log di terminal, harus muncul tulisan:
```
🎥 Mencoba koneksi ke CCTV: rtsp://...
⚠️ CCTV tidak tersedia
🎥 Mencoba webcam laptop...
✅ Webcam laptop berhasil terhubung
```

Buka browser, ketik: `http://localhost:8080`

Kalau kamera laptop keluar di browser → **BERHASIL!** 🎉

---

## 🔄 Cara Balik ke CCTV Lagi

Kalau nanti mau balik pakai CCTV:

### Langkah 1: Buka `.env` lagi
```bash
notepad .env
# atau
code .env
```

### Langkah 2: Hapus Tanda `#`

**DARI:**
```env
#CAMERA_HOST=192.168.1.203
#CAMERA_PORT=5503
```

**JADI:**
```env
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
```

### Langkah 3: Save & Restart

Save file, lalu restart aplikasi.

---

## 🐛 Kalau Ada Masalah

### Problem 1: "Webcam not found" / Webcam tidak ketemu

**Penyebab:**
- Webcam sedang dipakai aplikasi lain (Zoom, Teams, Skype, dll)
- Webcam belum nyala
- Driver webcam belum terinstall

**Solusi:**
1. Tutup semua aplikasi yang pakai webcam (Zoom, Teams, dll)
2. Restart laptop
3. Coba lagi

### Problem 2: Video tidak keluar / Layar hitam

**Solusi:**
1. Pastikan webcam tidak ditutup (ada laptop yang punya penutup kamera)
2. Cek permission webcam:
   - **Windows:** Settings → Privacy → Camera → Allow apps
   - **Mac:** System Preferences → Security & Privacy → Camera
3. Test webcam pakai aplikasi lain dulu (Photo Booth di Mac, Camera di Windows)

### Problem 3: "Permission denied"

**Solusi (Linux):**
```bash
# Kasih permission ke user
sudo usermod -a -G video $USER

# Logout dan login lagi
```

### Problem 4: Error "cv2.VideoCapture failed"

**Solusi:**
```bash
# Cek apakah webcam terdeteksi
# Di Linux:
ls /dev/video*

# Di Mac:
system_profiler SPCameraDataType

# Di Windows:
# Buka Device Manager, cek "Cameras"
```

---

## 💡 Tips dan Trik

### Tip 1: Cari Index Webcam yang Benar

Kalau laptop punya **lebih dari 1 webcam** (contoh: webcam built-in + webcam USB), mungkin perlu ganti index.

**Test webcam mana yang aktif:**
```bash
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'✅ Webcam {i}: ADA')
        cap.release()
    else:
        print(f'❌ Webcam {i}: TIDAK ADA')
"
```

**Hasil contoh:**
```
✅ Webcam 0: ADA          ← Built-in webcam
✅ Webcam 1: ADA          ← USB webcam
❌ Webcam 2: TIDAK ADA
❌ Webcam 3: TIDAK ADA
❌ Webcam 4: TIDAK ADA
```

**Kalau mau pakai webcam USB (index 1):**
Edit file `app.py`, cari baris `cv2.VideoCapture(0)`, ganti jadi `cv2.VideoCapture(1)`.

### Tip 2: Cek Kualitas Webcam

**Jalankan test ini:**
```bash
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
print(f'Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}')
print(f'Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')
cap.release()
"
```

**Hasil bagus:**
```
FPS: 30.0
Width: 1280
Height: 720
```

### Tip 3: Jarak Optimal untuk Webcam Laptop

| Jarak | Hasil Deteksi |
|-------|---------------|
| 0-1 meter | ❌ Terlalu dekat, gambar blur |
| 1-2 meter | ✅ OPTIMAL - plat terlihat jelas |
| 2-3 meter | ⚠️ Bisa, tapi kurang jelas |
| >3 meter | ❌ Terlalu jauh, plat kecil |

**Rekomendasi:**
- Posisi mobil **1.5-2 meter** dari laptop
- Cahaya harus **terang** (outdoor siang hari atau ruangan terang)
- Plat harus **menghadap langsung** ke kamera

---

## 📊 Perbandingan CCTV vs Webcam Laptop

| Aspek | CCTV | Webcam Laptop |
|-------|------|---------------|
| **Jarak** | 5-10 meter ✅ | 1-2 meter ⚠️ |
| **Kualitas** | HD/Full HD ✅ | HD (tergantung laptop) |
| **FPS** | 10-15 fps | 30 fps ✅ |
| **Cahaya** | Bisa malam hari ✅ | Perlu cahaya terang ⚠️ |
| **Harga** | Mahal 💰💰 | Gratis (built-in) ✅ |
| **Setup** | Ribet ⚠️ | Mudah ✅ |

---

## ✅ Checklist Sebelum Mulai

Sebelum test dengan webcam laptop, pastikan:

- [ ] Webcam tidak ditutup/diblokir
- [ ] Tidak ada aplikasi lain yang pakai webcam (Zoom, Teams, dll)
- [ ] Cahaya ruangan terang
- [ ] Jarak mobil 1.5-2 meter dari laptop
- [ ] Plat nomor menghadap langsung ke kamera
- [ ] File `.env` sudah di-edit (CAMERA_* di-comment)
- [ ] Aplikasi sudah di-restart

---

## 🎓 Penjelasan Teknis (Opsional)

**Kenapa cukup comment di `.env` saja?**

Sistem deteksi plat punya logika fallback:
1. Coba connect ke CCTV dulu (pakai setting di `.env`)
2. Kalau gagal → otomatis pakai webcam laptop
3. Kalau webcam juga gagal → error

Jadi kalau `.env` di-comment, sistem tidak bisa baca setting CCTV → langsung fallback ke webcam.

**Kode di app.py (baris 183-209):**
```python
try:
    # Coba CCTV dulu
    camera_url = config.CAMERA_URL  # ← Baca dari .env
    camera = cv2.VideoCapture(camera_url)
    if camera.isOpened():
        return True
except:
    pass  # Gagal → lanjut ke webcam

# Fallback: Pakai webcam laptop
camera = cv2.VideoCapture(0)  # ← 0 = webcam default
if camera.isOpened():
    return True
```

---

## 📞 Bantuan

Kalau masih bingung atau ada error:

1. **Screenshot error** yang muncul di terminal
2. **Screenshot hasil** yang keluar di browser
3. **Cek log file** di folder `logs/` (kalau ada)

---

## 🎯 Ringkasan

**Untuk Ganti ke Webcam:**
1. Edit `.env` → tambah `#` di depan semua `CAMERA_*`
2. Save file
3. Restart aplikasi: `python3 app.py`
4. Buka browser: `http://localhost:8080`

**Untuk Balik ke CCTV:**
1. Edit `.env` → hapus `#` di depan semua `CAMERA_*`
2. Save file
3. Restart aplikasi

**Selesai!** 🚀

---

**Dibuat oleh:** Claude Code
**Terakhir diupdate:** 2025-10-29
**Kesulitan:** ⭐ Mudah (Pemula)
