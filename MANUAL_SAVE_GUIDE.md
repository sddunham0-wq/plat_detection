# 💾 MANUAL SAVE FEATURE - Panduan Lengkap

## 🎯 FITUR BARU: Manual Save Button

Sistem sekarang memiliki **kontrol manual** untuk save plat nomor yang terdeteksi. Tidak lagi auto-save semua deteksi!

---

## ✅ APA YANG BERUBAH?

### **Sebelum:**
- ❌ **Auto-save semua deteksi** → Folder penuh sampah
- ❌ Ratusan foto plat yang tidak diperlukan
- ❌ Tidak ada kontrol untuk filter hasil

### **Sekarang:**
- ✅ **Deteksi real-time tetap jalan** (OCR otomatis)
- ✅ **Tampil di web interface** tanpa save otomatis
- ✅ **Button "Save"** di setiap deteksi
- ✅ **Manual control** - hanya save plat yang penting
- ✅ **Folder detected_plates** lebih bersih dan terorganisir

---

## 🚀 CARA MENGGUNAKAN

### 1. **Start Server**
```bash
python3 headless_stream.py
```

### 2. **Buka Web Interface**
```
http://localhost:5000
```

### 3. **Workflow Baru**

#### **a) Real-time Detection (Otomatis)**
- Camera stream berjalan
- Deteksi plat otomatis
- OCR otomatis (Tesseract ind+eng)
- **Hasil muncul di sidebar** "Live Detections"

#### **b) Manual Save (Hanya Yang Penting)**
1. **Lihat deteksi** muncul di sidebar kanan
2. **Review hasil** OCR dan confidence
3. **Click button "Save"** jika plat bagus
4. **Foto crop otomatis tersimpan** ke `detected_plates/`
5. **Data masuk database** untuk tracking

---

## 🎨 WEB INTERFACE - LIVE DETECTIONS PANEL

```
┌─────────────────────────────────────┐
│  Live Detections              [5]   │
├─────────────────────────────────────┤
│  B1234ABC                           │
│  [95.5%]              [💾 Save]     │
│  14:32:15                           │
├─────────────────────────────────────┤
│  D5678EFG                           │
│  [88.2%]              [💾 Save]     │
│  14:31:45                           │
├─────────────────────────────────────┤
│  F9012HIJ                           │
│  [92.1%]              [✅ Saved]    │
│  14:30:20                           │
└─────────────────────────────────────┘
```

### **Button States:**
- **[💾 Save]** → Hijau, ready to save
- **[⏳ Loading]** → Spinner saat saving
- **[✅ Saved]** → Abu-abu, sudah tersimpan (disabled)

---

## 📁 HASIL SAVED PLATES

### **Lokasi:**
```
detected_plates/
├── B1234ABC_20251009_143215.jpg  ← Crop plat
├── D5678EFG_20251009_143145.jpg  ← Crop plat
└── F9012HIJ_20251009_143020.jpg  ← Crop plat
```

### **Format Filename:**
```
[PLATE_TEXT]_[YYYYMMDD]_[HHMMSS].jpg
```

### **Database:**
```sql
-- Semua saved plates masuk ke database
SELECT * FROM plate_detections
WHERE plate_number = 'B1234ABC';
```

---

## ⚙️ KONFIGURASI

### **Toggle Auto-Save (Optional)**

Edit `config.py`:
```python
class SystemConfig:
    # Manual save mode (default)
    AUTO_SAVE_PLATES = False

    # Auto-save mode (old behavior)
    AUTO_SAVE_PLATES = True
```

**Recommended:** Keep `False` untuk kontrol manual

---

## 🔧 API ENDPOINT

### **Manual Save Plate**
```http
POST /api/save_plate
Content-Type: application/json

{
  "plate": "B1234ABC",
  "confidence": 95.5
}
```

**Response (Success):**
```json
{
  "success": true,
  "record_id": 123,
  "plate": "B1234ABC",
  "message": "Plate B1234ABC saved successfully"
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Detection not found in current frame"
}
```

---

## 🎯 USE CASES

### **1. Monitoring Parkir**
- Deteksi semua kendaraan real-time
- Save hanya plat yang **masuk** atau **keluar**
- Folder tetap bersih

### **2. Toll Gate / Access Control**
- Tampilkan semua plat yang lewat
- Save hanya yang **authorized** atau **perlu dicatat**
- Reduce storage usage

### **3. Security / Surveillance**
- Monitor semua kendaraan
- Save hanya plat **suspicious** atau **blacklist**
- Fokus pada yang penting

### **4. Traffic Analysis**
- Deteksi semua plat untuk counting
- Save sample untuk **quality check**
- Data analysis tanpa clutter

---

## 📊 PERBANDINGAN

| Feature | Auto-Save (Old) | Manual Save (New) |
|---------|-----------------|-------------------|
| **Detection** | ✅ Real-time | ✅ Real-time |
| **OCR** | ✅ Automatic | ✅ Automatic |
| **Display** | ✅ Web Interface | ✅ Web Interface |
| **Auto-Save** | ✅ All detections | ❌ None (manual only) |
| **Button Control** | ❌ No control | ✅ Manual save button |
| **Storage** | ❌ 100+ files/min | ✅ Only important plates |
| **Folder Cleanup** | ❌ Required often | ✅ Minimal cleanup |

---

## 🛠️ TROUBLESHOOTING

### **Problem: Button tidak muncul**
**Solution:**
- Refresh browser (Ctrl+F5)
- Clear cache
- Check console untuk errors

### **Problem: Save button tidak bekerja**
**Solution:**
1. Check console log untuk error messages
2. Pastikan stream sedang running
3. Cek network tab (F12) untuk API response

### **Problem: Saved image tidak ada/kosher**
**Solution:**
- Check folder permissions: `detected_plates/`
- Pastikan bbox detection valid
- Lihat log untuk error messages

### **Problem: Ingin kembali ke auto-save**
**Solution:**
Edit `config.py`:
```python
AUTO_SAVE_PLATES = True
```
Restart server.

---

## 📝 TIPS & BEST PRACTICES

### ✅ **DO:**
- Review confidence score sebelum save
- Save plat dengan confidence >70% untuk quality
- Monitor "Live Detections" panel secara berkala
- Gunakan screenshot button untuk full frame jika perlu

### ❌ **DON'T:**
- Jangan save plat dengan confidence <50% (likely false positive)
- Jangan spam click button (ada loading state)
- Jangan lupa cleanup folder detected_plates secara berkala

---

## 🎉 KEUNTUNGAN FITUR INI

1. **🚀 Storage Efficiency**
   - Reduce storage usage 90%+
   - Hanya save yang penting

2. **🎯 Better Quality Control**
   - Manual review sebelum save
   - Filter false positives

3. **⚡ Faster Workflow**
   - Real-time detection tetap cepat
   - No overhead auto-save semua frame

4. **📊 Clean Database**
   - Database hanya berisi data berkualitas
   - Analytics lebih akurat

5. **🧹 Easy Maintenance**
   - Folder detected_plates lebih bersih
   - Minimal cleanup required

---

## 🆘 SUPPORT

Jika ada masalah atau pertanyaan:
1. Check log output di terminal
2. Buka browser console (F12) untuk errors
3. Review dokumentasi ini kembali

**Happy plate detecting! 🚗📷**
