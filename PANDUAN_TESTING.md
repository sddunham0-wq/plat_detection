# 🚀 PANDUAN TESTING UNIFIED PLATE DETECTOR

## 📋 PERSIAPAN TESTING

### **1. Pastikan Sistem Ready:**
```bash
# Check semua komponen
python3 demo_local.py status
```

**Harus muncul:**
```
✅ YOLO Vehicle Detection: Available
✅ OCR Engine: Available
✅ Indonesian Validator: Available
✅ Stability Tracker: Available
```

---

## 🧪 TESTING METHODS

### **METHOD 1: Quick Test (Tercepat)**
```bash
python3 demo_local.py single
```

**Apa yang terjadi:**
- Ambil 1 gambar dari folder `contoh/`
- Proses detection
- Tampilkan hasil text

**Contoh output:**
```
🔍 Processing: plat.png
🎯 Results: 1
   1. '0NE' (confidence: 67.0%)
```

---

### **METHOD 2: Visual Test (Lihat Hasil)**
```bash
python3 demo_visual.py
```

**Apa yang terjadi:**
- Proses semua gambar di folder `contoh/`
- Simpan hasil visual sebagai `detection_result_*.jpg`
- Bisa lihat boxes dan annotations

**Cara lihat hasil:**
```bash
# macOS
open detection_result_*.jpg

# Atau copy ke desktop
cp detection_result_*.jpg ~/Desktop/
```

---

### **METHOD 3: Test dengan Gambar Sendiri**

#### **Langkah 1: Tambah gambar Anda**
```bash
# Copy gambar ke folder contoh
cp /path/to/your/image.jpg ./contoh/

# Atau drag & drop ke folder contoh/
```

#### **Langkah 2: Test gambar baru**
```bash
python3 demo_visual.py
```

#### **Langkah 3: Lihat hasil**
```bash
open detection_result_your_image.jpg
```

---

### **METHOD 4: Manual Testing (Advanced)**

Buat file test sendiri:

```python
# File: my_test.py
from unified_plate_detector import create_unified_detector
import cv2

# Initialize detector
detector = create_unified_detector()

# Load your image
frame = cv2.imread("path/to/your/image.jpg")

# Run detection
results = detector.detect(frame)

# Print results
print(f"Detected {len(results)} plates:")
for result in results:
    print(f"- {result.text} (confidence: {result.confidence:.1f}%)")
```

```bash
python3 my_test.py
```

---

## 📷 TESTING DENGAN GAMBAR SENDIRI

### **Format Gambar yang Didukung:**
- ✅ JPG/JPEG
- ✅ PNG
- ✅ Resolusi apapun (auto-resize)

### **Tips Gambar yang Bagus:**
- 🎯 **Plat nomor terlihat jelas**
- 🎯 **Tidak terlalu blur**
- 🎯 **Pencahayaan cukup**
- 🎯 **Kontras yang baik**

### **Contoh Scenario Testing:**
```bash
# Test 1: Foto CCTV
cp cctv_capture.jpg ./contoh/
python3 demo_visual.py

# Test 2: Foto smartphone
cp phone_photo.jpg ./contoh/
python3 demo_visual.py

# Test 3: Screenshot
cp screenshot.png ./contoh/
python3 demo_visual.py
```

---

## 🔧 DEBUGGING JIKA ADA MASALAH

### **Problem 1: No plates detected**
```bash
# Cek apakah sistem jalan
python3 demo_local.py status

# Cek log detail
python3 -c "
from unified_plate_detector import create_unified_detector
import cv2
detector = create_unified_detector()
frame = cv2.imread('./contoh/your_image.jpg')
vehicles = detector._detect_vehicles(frame)
candidates = detector._extract_plate_regions(frame, vehicles)
print(f'Vehicles: {len(vehicles)}, Candidates: {len(candidates)}')
"
```

### **Problem 2: Error saat import**
```bash
# Install dependencies
pip install opencv-python ultralytics pytesseract

# Atau
pip install -r requirements.txt
```

### **Problem 3: Slow processing**
```bash
# Test dengan gambar kecil dulu
# Resize gambar: 800x600 atau lebih kecil
```

---

## 📊 EVALUASI HASIL

### **Cara Baca Hasil:**

#### **Text Output:**
```
🎯 Results: 1
   1. 'B1234ABC' (confidence: 85.0%)
```
- **Results: 1** = Jumlah plat terdeteksi
- **'B1234ABC'** = Text yang terbaca
- **confidence: 85.0%** = Tingkat keyakinan sistem

#### **Visual Output:**
- 🟩 **Green Box** = Kendaraan terdeteksi
- 🟦 **Blue Box** = Kandidat area plat
- 🟥 **Red Box** = Plat nomor terdeteksi

### **Kriteria Sukses:**
- ✅ **Confidence > 60%** = Hasil bagus
- ✅ **Text readable** = OCR berhasil
- ✅ **No crash** = Sistem stabil

---

## 🎯 TESTING SCENARIOS

### **Scenario 1: Basic Test**
```bash
# Gunakan gambar yang sudah ada
python3 demo_local.py single
```

### **Scenario 2: Multiple Images**
```bash
# Copy beberapa gambar ke contoh/
cp image1.jpg image2.jpg image3.jpg ./contoh/
python3 demo_visual.py
```

### **Scenario 3: Real-world Test**
```bash
# Test dengan foto real dari CCTV/smartphone
cp real_photo.jpg ./contoh/
python3 demo_visual.py
open detection_result_real_photo.jpg
```

### **Scenario 4: Performance Test**
```bash
# Test dengan gambar besar
python3 -c "
import time
from unified_plate_detector import create_unified_detector
import cv2

detector = create_unified_detector()
frame = cv2.imread('./contoh/large_image.jpg')

start = time.time()
results = detector.detect(frame)
end = time.time()

print(f'Processing time: {end-start:.2f}s')
print(f'Results: {len(results)}')
"
```

---

## 🚀 QUICK START CHECKLIST

- [ ] 1. Run `python3 demo_local.py status` - semua ✅
- [ ] 2. Run `python3 demo_local.py single` - dapat hasil
- [ ] 3. Run `python3 demo_visual.py` - dapat file gambar
- [ ] 4. Open `detection_result_*.jpg` - lihat visual
- [ ] 5. Copy gambar sendiri ke `contoh/` folder
- [ ] 6. Test ulang dengan gambar sendiri

**Jika semua checklist ✅ = SISTEM WORKING PERFECT!** 🎉

---

## 💡 TIPS TESTING

### **Untuk Hasil Terbaik:**
1. 📸 **Gunakan gambar berkualitas baik**
2. 🔍 **Pastikan plat nomor visible**
3. ⚡ **Test dengan gambar ukuran sedang (800x600)**
4. 🎯 **Cek confidence score (>60% = bagus)**

### **Troubleshooting:**
1. 🔧 **Jika lambat**: Resize gambar lebih kecil
2. 🔧 **Jika no detection**: Cek lighting dan clarity
3. 🔧 **Jika error**: Cek dependencies

**Happy Testing!** 🚀