

# Kodingan lama
# Cari matching plate text dari all_detected_plates
                        label_text = f"PLAT {idx}"  # Default label

                        # Try to find matching detected plate text
                        global all_detected_plates
                        if all_detected_plates:
                            for plate_info in all_detected_plates:
                                plate_bbox = plate_info.get('bbox', [])
                                if len(plate_bbox) >= 4:
                                    px, py, pw, ph = plate_bbox[:4]
                                    # Check if bbox matches (with tolerance)
                                    if abs(x - px) < 10 and abs(y - py) < 10:
                                        plate_text = plate_info.get('text', '')
                                        confidence = plate_info.get('confidence', 0)
                                        if plate_text:
                                            label_text = f"{plate_text} ({confidence:.0%})"
                                        break
                                        
# 🔧 SOLUSI: Fix Label Text yang Tidak Muncul

## 🐛 Masalah

Label di bounding box hijau muncul:
- ❌ "wait...." terus
- ❌ "❌ UNREADABLE" terus
- ❌ Tidak muncul text plat nomor

## 🎯 Penyebab

Ada **bug di struktur if-else** di baris 1028-1040 di file `app.py`.

Kode saat ini salah strukturnya:
```python
if abs(x - px) < 10 and abs(y - py) < 10:
    plate_text = plate_info.get('text', '')
    confidence = plate_info.get('confidence', 0)
    if confidence >= 0.8:
        label_text = f"✅ {plate_text}"
    elif confidence >= 0.5:
        label_text = f"⚠️ {plate_text}"
    else:
        label_text = f"❓ {plate_text}"
else:
    label_text = "❌ UNREADABLE"  # ← SALAH! Ini jalan terus
break
```

**Problem:** Blok `else` dijalankan kalau koordinat tidak match, dan langsung `break`, sehingga loop berhenti dan label jadi "UNREADABLE".

---

## ✅ SOLUSI 1: Fix dengan Struktur yang Benar (RECOMMENDED)

### Langkah 1: Buka file `app.py`

### Langkah 2: Cari baris 1028-1040

### Langkah 3: **HAPUS semua baris 1028-1040**

### Langkah 4: **GANTI dengan kode ini:**

```python
                                    # Check if bbox matches (with tolerance)
                                    if abs(x - px) < 10 and abs(y - py) < 10:
                                        plate_text = plate_info.get('text', '')
                                        confidence = plate_info.get('confidence', 0)

                                        # Cek apakah plate_text ada isinya
                                        if plate_text:
                                            # Tampilkan text plat (tanpa emoji untuk simple)
                                            label_text = f"{plate_text}"
                                        else:
                                            label_text = "NO TEXT"

                                        # Break setelah ketemu match
                                        break
```

### Langkah 5: Save (Cmd + S) dan Restart

```bash
python3 app.py
```

**Hasil:** Label akan tampil text plat nomor seperti **"F 1234 HF"**.

---

## ✅ SOLUSI 2: Dengan Emoji Status (Opsi Lain)

Kalau mau pakai emoji status seperti sebelumnya:

```python
                                    # Check if bbox matches (with tolerance)
                                    if abs(x - px) < 10 and abs(y - py) < 10:
                                        plate_text = plate_info.get('text', '')
                                        confidence = plate_info.get('confidence', 0)

                                        if plate_text:
                                            # Status berdasarkan confidence
                                            if confidence >= 0.8:
                                                label_text = f"✅ {plate_text}"
                                            elif confidence >= 0.5:
                                                label_text = f"⚠️ {plate_text}"
                                            else:
                                                label_text = f"❓ {plate_text}"
                                        else:
                                            label_text = "❌ NO TEXT"

                                        break
```

**Catatan:** Emoji mungkin tidak muncul di OpenCV, jadi lebih baik pakai **SOLUSI 1** (tanpa emoji).

---

## ✅ SOLUSI 3: Paling Simple (Tanpa Status)

Kalau mau paling simple, cuma text plat saja:

```python
                                    # Check if bbox matches (with tolerance)
                                    if abs(x - px) < 10 and abs(y - py) < 10:
                                        plate_text = plate_info.get('text', '')

                                        if plate_text:
                                            label_text = plate_text  # Langsung text plat

                                        break
```

**Hasil:** Label cuma tampil **"F 1234 HF"** tanpa embel-embel.

---

## 📋 Copy-Paste Ready Code

Untuk memudahkan, ini **kode lengkap baris 1017-1054** yang sudah fix:

```python
                        # ★ NEW: Draw plate label di atas bbox
                        label_text = "DETECTING..."  # Default label

                        # Try to find matching detected plate text
                        global all_detected_plates
                        if all_detected_plates:
                            for plate_info in all_detected_plates:
                                plate_bbox = plate_info.get('bbox', [])
                                if len(plate_bbox) >= 4:
                                    px, py, pw, ph = plate_bbox[:4]
                                    # Check if bbox matches (with tolerance)
                                    if abs(x - px) < 10 and abs(y - py) < 10:
                                        plate_text = plate_info.get('text', '')

                                        if plate_text:
                                            label_text = f"{plate_text}"  # ← TEXT PLAT MUNCUL DI SINI

                                        break  # ← PENTING: break di sini

                        # Draw label background (semi-transparent black box)
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        label_w, label_h = label_size

                        # Background rectangle
                        cv2.rectangle(frame,
                                    (x, y - label_h - 10),
                                    (x + label_w + 10, y),
                                    (0, 0, 0), -1)  # Black filled

                        # Text label
                        cv2.putText(frame, label_text, (x + 5, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
```

---

## 🔍 Penjelasan Bug

**Yang Salah:**
```python
if kondisi:
    # code
else:
    label_text = "UNREADABLE"  # ← Ini jalan kalau kondisi false
break  # ← Loop langsung stop
```

**Yang Benar:**
```python
if kondisi:
    # code
    if plate_text:
        label_text = plate_text  # ← Set label kalau ada text
    break  # ← Break di dalam if, setelah ketemu match
```

**Key Point:**
- `break` harus **di dalam** blok `if` yang match
- Tidak pakai `else` untuk case yang tidak match (skip aja, lanjut loop berikutnya)

---

## 🐛 Troubleshooting Setelah Fix

### Problem: Masih muncul "DETECTING..."

**Penyebab:** OCR belum berhasil baca plat

**Solusi:**
1. **Dekatkan plat ke kamera** (20-50 cm)
2. **Pastikan lighting cukup**
3. **Cek log terminal** untuk `✅ OCR SUCCESS`
4. **Turunkan threshold** di baris 856:
   ```python
   MIN_OCR_CONFIDENCE = 0.001
   ```

---

### Problem: Text muncul tapi cepat hilang

**Penyebab:** Smoothing terlalu ketat

**Solusi:** Di baris 647, turunkan threshold:
```python
smoothed_plates = smooth_bounding_boxes(bboxes, plate_tracking_history, iou_threshold=0.1)
# 0.15 → 0.1
```

---

### Problem: Label terpotong di tepi

**Solusi:** Batasi posisi X di baris 1053:
```python
x_safe = max(5, min(x + 5, frame.shape[1] - 150))
cv2.putText(frame, label_text, (x_safe, y - 5), ...)
```

---

## ✅ Checklist Verifikasi

Setelah apply fix, pastikan:

- [ ] File `app.py` sudah di-save (Cmd + S)
- [ ] Aplikasi sudah di-restart (Ctrl + C, lalu `python3 app.py`)
- [ ] Webcam sudah terhubung (cek log `✅ Webcam berhasil terhubung`)
- [ ] Plat berada 20-50 cm dari kamera
- [ ] Lighting cukup (tidak terlalu gelap/terang)
- [ ] Plat frontal (tidak miring)

---

## 📊 Perbandingan Before & After

### Before (Bug):
```
[Kotak Hijau]
   ❌ UNREADABLE  ← Selalu ini atau "wait...."
```

### After (Fixed):
```
[Kotak Hijau]
   F 1234 HF  ← Text plat muncul!
```

---

## 💡 Tips Pro

1. **Test dengan plat asli** di depan kamera
2. **Cek terminal log** untuk debug OCR
3. **Turunkan confidence threshold** kalau text jarang muncul
4. **Gunakan lighting yang baik** (natural light atau lampu putih)
5. **Jarak ideal:** 20-50 cm dari kamera

---

## 📞 Masih Bermasalah?

Kalau masih error setelah ikuti solusi di atas:

1. **Screenshot error** di terminal
2. **Screenshot tampilan** video dengan label
3. **Copy paste baris 1017-1054** dari file Anda
4. Kirim ke sini untuk debug lebih lanjut

---

**Good luck! 🔧**
