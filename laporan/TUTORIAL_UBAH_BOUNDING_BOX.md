# 📐 TUTORIAL: Cara Mengubah Ketebalan Bounding Box

## 🎯 Apa itu Bounding Box?

**Bounding Box** adalah kotak hijau/biru yang muncul di video untuk menandai:
- **Kotak BIRU** = Kendaraan (mobil/motor)
- **Kotak HIJAU** = Plat nomor

---

## 📍 Lokasi Kode Bounding Box

File: `app.py`
Fungsi: `draw_detection_info(frame)` di **baris 935-1056**

---

## ✅ CARA 1: Ubah Ketebalan Kotak KENDARAAN (Biru)

### Lokasi: Baris 996

**Kode asli:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 3)
```

**Parameter:**
- `BLUE` = Warna biru (255, 0, 0)
- `3` = **Ketebalan 3 pixel** ← INI YANG DIUBAH

### Contoh Perubahan:

**Lebih Tipis:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 1)  # Tipis (1 pixel)
```

**Lebih Tebal:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 5)  # Tebal (5 pixel)
```

**Sangat Tebal:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 8)  # Sangat tebal (8 pixel)
```

---

## ✅ CARA 2: Ubah Ketebalan Kotak PLAT NOMOR (Hijau)

### Lokasi: Baris 1015

**Kode asli:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)
```

**Parameter:**
- `GREEN` = Warna hijau (0, 255, 0)
- `2` = **Ketebalan 2 pixel** ← INI YANG DIUBAH

### Contoh Perubahan:

**Lebih Tipis:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 1)  # Tipis (1 pixel)
```

**Lebih Tebal:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 4)  # Tebal (4 pixel)
```

**Sangat Tebal:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 6)  # Sangat tebal (6 pixel)
```

---

## ✅ CARA 3: Ubah Ketebalan Text Label

### Lokasi: Baris 1000 (Label Kendaraan)

**Kode asli:**
```python
cv2.putText(frame, vehicle_label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLUE, 2)
```

**Parameter:**
- `0.9` = Ukuran font
- `2` = **Ketebalan text 2 pixel** ← INI YANG DIUBAH

### Contoh:
```python
cv2.putText(frame, vehicle_label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLUE, 3)  # Text lebih tebal
```

---

### Lokasi: Baris 1047 (Label Plat Nomor)

**Kode asli:**
```python
cv2.putText(frame, label_text, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
```

**Parameter:**
- `0.6` = Ukuran font
- `2` = **Ketebalan text 2 pixel** ← INI YANG DIUBAH

### Contoh:
```python
cv2.putText(frame, label_text, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 3)  # Text lebih tebal
```

---

## 🎨 BONUS: Ubah Warna Bounding Box

### Warna dalam OpenCV (Format BGR)

```python
# Definisi warna (B, G, R)
BLUE = (255, 0, 0)      # Biru
GREEN = (0, 255, 0)     # Hijau
RED = (0, 0, 255)       # Merah
YELLOW = (0, 255, 255)  # Kuning
PURPLE = (255, 0, 255)  # Ungu
CYAN = (255, 255, 0)    # Cyan
WHITE = (255, 255, 255) # Putih
BLACK = (0, 0, 0)       # Hitam
```

### Contoh: Ganti Kotak Kendaraan Jadi Merah

**Lokasi: Baris 971-972**

```python
# Ganti ini:
BLUE = (255, 0, 0)  # BGR format - Blue untuk mobil

# Jadi ini:
RED = (0, 0, 255)   # BGR format - Red untuk mobil
```

**Lalu di baris 996, ganti:**
```python
cv2.rectangle(frame, (x, y), (x+w, y+h), RED, 3)
```

---

## 📊 Rekomendasi Ketebalan

| Jenis | Ketebalan Default | Rekomendasi | Keterangan |
|-------|-------------------|-------------|------------|
| **Kotak Kendaraan** | 3 pixel | 2-5 pixel | Lebih tebal = lebih jelas |
| **Kotak Plat** | 2 pixel | 1-4 pixel | Jangan terlalu tebal |
| **Text Label** | 2 pixel | 2-3 pixel | Terlalu tebal = sulit dibaca |

---

## 🛠️ Contoh Lengkap

### Skenario: Buat bounding box lebih tebal dan jelas

**Edit di `app.py` baris 996:**
```python
# SEBELUM:
cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 3)

# SESUDAH:
cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 5)  # Tebal 5px
```

**Edit di `app.py` baris 1015:**
```python
# SEBELUM:
cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

# SESUDAH:
cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 4)  # Tebal 4px
```

**Save file dan restart aplikasi:**
```bash
python3 app.py
```

---

## 🔍 Visual Perbandingan

```
Ketebalan 1:  ┌─────┐  (tipis, sulit dilihat)
Ketebalan 2:  ┏━━━━━┓  (default, cukup jelas)
Ketebalan 3:  ┏━━━━━┓  (tebal, sangat jelas)
Ketebalan 5:  ┏━━━━━┓  (sangat tebal, bold)
Ketebalan 8:  ┏━━━━━┓  (terlalu tebal, menghalangi view)
```

---

## ⚠️ Catatan Penting

1. **Jangan terlalu tebal** (>8 pixel) - akan menghalangi view video
2. **Test dulu** dengan ketebalan 3-5 pixel
3. **Restart aplikasi** setiap kali ganti nilai
4. **Backup file** sebelum edit (copy `app.py` jadi `app_backup.py`)

---

## 📝 Quick Reference

| Ingin Ubah | Baris | Parameter | Nilai Default |
|------------|-------|-----------|---------------|
| Kotak Kendaraan | 996 | thickness | 3 |
| Kotak Plat | 1015 | thickness | 2 |
| Label Kendaraan | 1000 | thickness | 2 |
| Label Plat | 1047 | thickness | 2 |

---

## ❓ FAQ

### Q: Kenapa setelah edit masih tidak berubah?
**A:** Pastikan restart aplikasi dengan `python3 app.py`

### Q: Bisa beda ketebalan untuk mobil dan motor?
**A:** Bisa! Gunakan if-else berdasarkan `vehicle_label` (MOBIL/MOTOR)

### Q: Bagaimana buat bounding box berkedip-kedip?
**A:** Tidak disarankan - akan mengganggu dan menyebabkan lag

### Q: Bisa hapus bounding box sama sekali?
**A:** Bisa, comment semua baris `cv2.rectangle()` dengan tanda `#`

---

## 🎓 Penjelasan Parameter cv2.rectangle()

```python
cv2.rectangle(image, pt1, pt2, color, thickness)
```

- `image` = Frame video
- `pt1` = Titik kiri atas (x, y)
- `pt2` = Titik kanan bawah (x+w, y+h)
- `color` = Warna dalam format BGR
- `thickness` = Ketebalan garis dalam pixel
  - Nilai positif (1-10) = garis
  - Nilai -1 = kotak terisi penuh

---

**Good luck! 🎨**
