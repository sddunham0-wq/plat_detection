# 🚀 CARA PAKAI UNIFIED PLATE DETECTOR

## ⚡ QUICK COMMANDS (COPY & PASTE)

### **1. Test File Gambar Spesifik** ⭐ (PALING MUDAH)
```bash
python3 demo_visual.py gambar.png
```

**Contoh:**
```bash
python3 demo_visual.py ./contoh/plat.png
python3 demo_visual.py ~/Downloads/foto_mobil.jpg
python3 demo_visual.py /path/ke/gambar/anda.png
```

**Hasil:** File `detection_result_nama_gambar.jpg` + info di terminal

---

### **2. Test Mudah Text Only**
```bash
python3 test_mudah.py
```
**Hasil:** Text detection di terminal saja

---

### **3. Test Semua Gambar di Folder**
```bash
python3 demo_visual.py
```
**Hasil:** Proses semua gambar di folder `contoh/`

---

### **4. Cek Status Sistem**
```bash
python3 demo_local.py status
```
**Hasil:** Info sistem working atau tidak

---

## 📱 USAGE PATTERNS

### **Pattern 1: Test Gambar Langsung** (RECOMMENDED)
```bash
# Drag & drop gambar ke terminal atau copy path
python3 demo_visual.py "/Users/nama/Desktop/foto_plat.jpg"

# Relative path
python3 demo_visual.py ./gambar_test.png

# Absolute path
python3 demo_visual.py /full/path/to/image.jpg
```

### **Pattern 2: Test Multiple Files**
```bash
# Test file satu per satu
python3 demo_visual.py image1.jpg
python3 demo_visual.py image2.png
python3 demo_visual.py image3.jpeg
```

### **Pattern 3: Workflow Testing**
```bash
# Step 1: Cek sistem
python3 demo_local.py status

# Step 2: Test gambar
python3 demo_visual.py your_image.jpg

# Step 3: Lihat hasil
open detection_result_your_image.jpg
```

---

## 🎯 OUTPUT YANG DIHASILKAN

### **Terminal Output:**
```
🎯 VISUAL DEMO - SINGLE FILE
==================================================
📱 Initializing detector...
✅ Detector ready!
📸 Processing: plat.png
✅ Loaded: (711, 1273, 3)
⏱️  Detection time: 114.73s
🚗 Vehicles: 2
📋 Candidates: 91
🎯 Plates detected: 1

✅ DETECTED PLATES:
   1. '0NE' (confidence: 67.0%)

💾 Visual result saved: detection_result_plat.jpg
🔍 Open with: open detection_result_plat.jpg
```

### **File Output:**
- **File hasil:** `detection_result_nama_gambar.jpg`
- **Isi:** Gambar dengan boxes dan annotations
- **Info panel:** Statistics di bagian atas

---

## 🔍 CARA BACA HASIL

### **Terminal Info:**
- `Vehicles: 2` = Jumlah kendaraan/region detected
- `Candidates: 91` = Jumlah area yang dianalisis
- `Plates detected: 1` = Jumlah plat nomor ketemu
- `'0NE' (confidence: 67.0%)` = Text + tingkat keyakinan

### **Visual Boxes:**
- 🟥 **Red Box** = Plat nomor terdeteksi ✅
- 🟦 **Blue Box** = Area kandidat yang dianalisis
- 🟩 **Green Box** = Kendaraan/search region

### **Info Panel (atas gambar):**
- File name, statistics, processing time, hasil detection

---

## 📸 FORMAT GAMBAR YANG DIDUKUNG

### **Format File:**
- ✅ JPG, JPEG
- ✅ PNG
- ✅ Semua resolusi (auto-resize)

### **Tips Gambar Bagus:**
- 🎯 Plat nomor terlihat jelas
- 🎯 Kontras yang baik (tidak terlalu gelap/terang)
- 🎯 Tidak blur atau shaky
- 🎯 Ukuran plat minimal 50x20 pixel

---

## 🚀 CONTOH PENGGUNAAN REAL

### **Scenario 1: Test Foto Smartphone**
```bash
# Copy foto dari smartphone ke komputer
python3 demo_visual.py ~/Downloads/foto_dari_hp.jpg
open detection_result_foto_dari_hp.jpg
```

### **Scenario 2: Test Screenshot CCTV**
```bash
# Test screenshot dari CCTV
python3 demo_visual.py screenshot_cctv.png
open detection_result_screenshot_cctv.jpg
```

### **Scenario 3: Test Multiple Photos**
```bash
# Test beberapa foto sekaligus
python3 demo_visual.py foto1.jpg
python3 demo_visual.py foto2.jpg
python3 demo_visual.py foto3.jpg

# Lihat semua hasil
open detection_result_*.jpg
```

### **Scenario 4: Drag & Drop (macOS)**
```bash
# Drag file ke terminal setelah command
python3 demo_visual.py
# [drag file gambar ke terminal]
# [enter]
```

---

## 🔧 TROUBLESHOOTING

### **Error: File tidak ditemukan**
```bash
❌ File tidak ditemukan: gambar.png
💡 Pastikan path file benar dan file exists
```
**Solusi:** Check path file, gunakan absolute path atau relative path yang benar

### **Error: Gagal load gambar**
```bash
❌ Gagal load gambar: file.xyz
💡 Pastikan format gambar didukung (jpg, png)
```
**Solusi:** Convert ke JPG atau PNG

### **No plates detected**
```bash
⚠️  No plates detected
```
**Solusi:**
- Coba gambar dengan plat nomor yang lebih jelas
- Check pencahayaan dan kontras
- Pastikan ukuran plat cukup besar

---

## 💡 TIPS & TRICKS

### **Performance Tips:**
```bash
# Resize gambar besar untuk speed
# Ukuran optimal: 800x600 atau 1280x720
```

### **Quality Tips:**
```bash
# Untuk hasil terbaik:
# - Plat nomor facing camera (tidak miring)
# - Pencahayaan yang cukup
# - Kontras yang baik antara plat dan background
```

### **Batch Processing:**
```bash
# Test multiple files dengan loop
for file in *.jpg; do
    python3 demo_visual.py "$file"
done
```

---

## 🎉 SUMMARY COMMANDS

```bash
# Test 1 gambar spesifik (PALING SERING DIPAKAI)
python3 demo_visual.py gambar.png

# Test cepat text only
python3 test_mudah.py

# Test semua gambar di folder contoh
python3 demo_visual.py

# Cek status sistem
python3 demo_local.py status

# Lihat hasil visual
open detection_result_*.jpg
```

**🚀 Command paling berguna:** `python3 demo_visual.py nama_gambar.jpg`