# 📷 Screenshot Feature Documentation

Fitur screenshot telah berhasil diimplementasikan untuk web interface Live CCTV Plate Detection System.

## ✨ Fitur yang Ditambahkan

### Backend (headless_stream.py)
- **Endpoint baru**: `POST /api/screenshot`
- **Fungsi**: Mengambil screenshot dari frame video saat ini
- **Output**: Menyimpan gambar ke folder `detected_plates/`

### Frontend (templates/stream.html)
- **Button screenshot**: Sudah ada di control panel
- **JavaScript handler**: Mengirim request ke backend
- **Loading state**: Menampilkan spinner saat proses
- **Notifikasi**: Alert sukses/gagal dengan nama file

## 🔧 Cara Kerja

### 1. Backend Implementation
```python
@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    """Take screenshot dari current frame dan simpan ke detected_plates"""
    # 1. Validasi stream sedang berjalan
    # 2. Ambil current frame dari stream manager
    # 3. Decode base64 image ke format OpenCV
    # 4. Generate filename unik dengan timestamp
    # 5. Simpan ke folder detected_plates/
    # 6. Return info file yang tersimpan
```

### 2. Frontend Integration
```javascript
screenshotBtn.addEventListener('click', function() {
    // 1. Validasi stream status
    // 2. Show loading spinner
    // 3. Send POST request ke /api/screenshot
    // 4. Display success/error notification
    // 5. Restore button state
});
```

## 📁 File Struktur

Screenshot akan disimpan di:
```
detected_plates/
├── screenshot_20250913_143052.jpg    # Normal screenshot
├── screenshot_20250913_143052_1.jpg  # Jika ada duplicate
├── screenshot_20250913_143052_2.jpg  # Auto increment
└── ...
```

## 🚀 Cara Menggunakan

### 1. Start Web Server
```bash
python3 headless_stream.py --host 0.0.0.0 --port 5000
```

### 2. Buka Browser
```
http://localhost:5000
```

### 3. Start Video Stream
- Masukkan RTSP URL atau webcam index
- Klik tombol "Start"
- Tunggu sampai stream berjalan

### 4. Ambil Screenshot
- Klik button camera icon (📷) di control panel
- Screenshot akan otomatis tersimpan ke folder `detected_plates/`
- Notifikasi sukses akan muncul dengan nama file

## 🧪 Testing

### Manual Test via Browser
1. Start server
2. Buka http://localhost:5000
3. Start stream
4. Klik screenshot button
5. Check folder `detected_plates/`

### Automated Test Script
```bash
python3 test_screenshot.py
```

Test script akan:
- ✅ Test screenshot tanpa stream (should fail)
- ✅ Start stream
- ✅ Take screenshot (should succeed)
- ✅ Take multiple screenshots (different filenames)
- ✅ Stop stream

## 📊 Response Format

### Success Response
```json
{
    "success": true,
    "message": "Screenshot saved successfully",
    "filename": "screenshot_20250913_143052.jpg",
    "path": "/path/to/detected_plates/screenshot_20250913_143052.jpg",
    "frame_id": 1234,
    "timestamp": 1694608252.123
}
```

### Error Response
```json
{
    "success": false,
    "error": "Stream not running"
}
```

## 🛡️ Error Handling

### Frontend Validations
- ⚠️ Stream harus running sebelum screenshot
- 🔄 Loading state saat proses screenshot
- 📱 User-friendly error messages

### Backend Validations
- ✅ Stream status check
- ✅ Frame availability check
- ✅ File system permissions
- ✅ Duplicate filename handling
- ✅ Image decode validation

## 🔧 Konfigurasi

Screenshot menggunakan konfigurasi dari `config.py`:
```python
class SystemConfig:
    OUTPUT_FOLDER = "detected_plates"  # Folder output screenshot
```

## 🎯 Use Cases

1. **Manual Evidence Collection**: Capture frame tertentu untuk dokumentasi
2. **Quality Control**: Ambil screenshot untuk review kualitas deteksi
3. **Debugging**: Capture frame dengan masalah untuk analisis
4. **Reporting**: Collect visual evidence untuk laporan

## 🔄 Integration dengan Sistem Existing

### Folder Structure
Screenshot tersimpan di folder yang sama dengan detected plates:
```
detected_plates/
├── TEST123_20250912_101415.jpg     # Automatic detection
├── screenshot_20250913_143052.jpg  # Manual screenshot
└── ...
```

### Database Integration
Screenshot saat ini **tidak** tersimpan ke database, hanya file gambar. Jika diperlukan integrasi database:

```python
# Optional: Save screenshot info to database
screenshot_record = {
    'type': 'manual_screenshot',
    'filename': filename,
    'timestamp': current_frame.timestamp,
    'frame_id': current_frame.frame_id
}
database.save_screenshot_record(screenshot_record)
```

## ✅ Status Implementation

- [x] ✅ Backend endpoint `/api/screenshot`
- [x] ✅ Frontend button & JavaScript handler
- [x] ✅ Image decoding dan saving
- [x] ✅ Error handling & validation
- [x] ✅ Loading states & notifications
- [x] ✅ Unique filename generation
- [x] ✅ Test script
- [x] ✅ Documentation

## 🎉 Kesimpulan

Fitur screenshot telah **berhasil diimplementasikan** dan siap digunakan! 

**Key Features:**
- 📷 One-click screenshot dari web interface
- 💾 Auto-save ke folder `detected_plates/`
- 🔄 Unique filename dengan timestamp
- ⚡ Real-time feedback & error handling
- 🧪 Comprehensive testing

**Next Steps:**
1. Start server: `python3 headless_stream.py`
2. Test di browser: `http://localhost:5000`
3. Enjoy taking screenshots! 📸