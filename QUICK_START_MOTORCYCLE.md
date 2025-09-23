# 🏍️ Quick Start: Motorcycle Plate Detection

## 🚀 Cara Cepat Menjalankan Deteksi Plat Motor

### 1. **Test dengan Webcam (Rekomendasi)**
```bash
python3 start_motorcycle_detection.py --test-mode
```

### 2. **Test dengan RTSP Camera** 
```bash
python3 start_motorcycle_detection.py --source "rtsp://admin:H4nd4l9165!@168.1.195:554" --test-mode
```

### 3. **Mode Produksi (Live Detection)**
```bash
python3 start_motorcycle_detection.py --source 0 --confidence 0.4
```

## 🎯 Parameter Penting

- `--confidence 0.3`: Lebih sensitif (deteksi lebih banyak motor)
- `--confidence 0.6`: Lebih selektif (deteksi hanya motor yang jelas)
- `--test-mode`: Mode test 30 detik untuk evaluasi
- `--source 0`: Webcam default
- `--source "rtsp://..."`: RTSP camera

## 📋 Yang Akan Anda Lihat

- **🟡 Kotak Kuning**: Plat motor terdeteksi
- **🟢 Kotak Hijau**: Plat kendaraan lain  
- **🏍️ Label**: Jenis kendaraan motor
- **📋 Teks**: Nomor plat dan confidence %

## ⌨️ Kontrol

- **'q'**: Keluar dari aplikasi
- **'s'**: Save screenshot
- **'p'**: Tampilkan statistik

## 🔧 Jika Ada Masalah

### YOLOv8 tidak tersedia:
```bash
pip install ultralytics
```

### Deteksi rendah:
```bash
python3 start_motorcycle_detection.py --confidence 0.3 --test-mode
```

### Terlalu banyak false positive:
```bash
python3 start_motorcycle_detection.py --confidence 0.6 --test-mode
```

## 📊 Hasil Test yang Baik

- **Motorcycles Detected**: >0 (ada motor yang terdeteksi)
- **Plates Detected**: >0 (ada plat yang terbaca)
- **Detection Rate**: >50% (deteksi di lebih dari 50% frame)

---

**Tip**: Mulai dengan `--test-mode` untuk melihat apakah sistem bekerja dengan baik di lingkungan Anda!