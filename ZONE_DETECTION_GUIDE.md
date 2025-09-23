# 🎯 Zone-Based Motorcycle Detection Guide

Optimized untuk parking lot dengan banyak motor seperti screenshot Anda!

## 🚀 Quick Start Zone Detection

### **Mode Sequential (Rekomendasi untuk kondisi Anda)**
```bash
# Test mode (60 detik)
python3 start_zone_detection.py --source "rtsp://admin:H4nd4l9165!@168.1.195:554" --mode sequential --duration 60

# Production mode
python3 start_zone_detection.py --source "rtsp://admin:H4nd4l9165!@168.1.195:554" --mode sequential
```

### **Mode Parallel (untuk performa tinggi)**
```bash
python3 start_zone_detection.py --source "rtsp://admin:H4nd4l9165!@168.1.195:554" --mode parallel
```

### **Mode Adaptive (otomatis menyesuaikan)**
```bash
python3 start_zone_detection.py --source "rtsp://admin:H4nd4l9165!@168.1.195:554" --mode adaptive
```

## 📐 Zone Layout

Sistem membagi frame menjadi **3x3 grid (9 zones)** yang optimal untuk parking lot:

```
┌─────────┬─────────┬─────────┐
│ Zone 0  │ Zone 1  │ Zone 2  │
│Top-Left │Top-Cntr │Top-Rght │
├─────────┼─────────┼─────────┤
│ Zone 3  │ Zone 4  │ Zone 5  │
│Mid-Left │Mid-Cntr │Mid-Rght │  ← Priority Zones
├─────────┼─────────┼─────────┤
│ Zone 6  │ Zone 7  │ Zone 8  │
│Bot-Left │Bot-Cntr │Bot-Rght │
└─────────┴─────────┴─────────┘
```

### **Zone Confidence Settings**
- **Zone 6-8** (Bottom): 0.15 (paling dekat, confidence rendah)
- **Zone 3-5** (Middle): 0.20 (area utama)
- **Zone 0-2** (Top): 0.25-0.30 (terjauh, confidence lebih tinggi)

## 🎮 Detection Modes

### 1. **Sequential Mode** ⭐ (Rekomendasi)
- **Cara kerja**: Scan satu zone per waktu secara berurutan
- **Kelebihan**: 
  - Performance optimal untuk jarak jauh
  - Resource efficient
  - Cocok untuk motor yang parkir/diam
- **Ideal untuk**: Parking lot monitoring seperti screenshot Anda

### 2. **Parallel Mode**
- **Cara kerja**: Scan semua zones sekaligus
- **Kelebihan**: 
  - Detection comprehensive
  - Cocok untuk traffic monitoring
- **Perlu**: Hardware yang lebih powerful

### 3. **Adaptive Mode**
- **Cara kerja**: Fokus pada zones dengan aktivitas tinggi
- **Kelebihan**: 
  - Otomatis menyesuaikan prioritas
  - Optimal untuk mixed conditions

## ⌨️ Interactive Controls

### **Keyboard Shortcuts:**
- **'q'**: Quit aplikasi
- **'s'**: Save screenshot dengan zone overlay
- **'z'**: Toggle zone display ON/OFF
- **'m'**: Switch detection mode (sequential → parallel → adaptive)
- **'p'**: Print real-time statistics

## 🎨 Visual Indicators

### **Zone Colors:**
- **🟡 Yellow**: Active zone (sequential mode)
- **🟢 Green**: Enabled zones
- **⚫ Gray**: Disabled zones

### **Detection Colors:**
- **🟢 Green Box**: Motorcycle detected
- **🟡 Yellow Box**: License plate detected
- **📋 Text**: Zone name + confidence

### **Zone Statistics:**
- **M:X**: X motorcycles detected in zone
- **P:Y**: Y plates read in zone

## 📊 Expected Performance untuk Screenshot Anda

### **Realistic Expectations:**
- **Motorcycle Detection**: 70-85% (YOLO bagus untuk detect motor)
- **Zone Coverage**: 90%+ (semua area ter-cover)
- **Plate Reading**: 10-25% (jarak jauh, resolusi rendah)
- **Performance**: 15-20 FPS dengan sequential mode

### **Optimal Results:**
- **Movement Detection**: Excellent (motor masuk/keluar)
- **Counting**: Very Good (jumlah motor per zone)
- **Area Monitoring**: Excellent (aktivitas per area)
- **Text Recognition**: Limited (tapi motor detection bagus)

## 🔧 Customization untuk Kondisi Anda

### **Adjust Zone Configuration:**
```python
# Edit start_zone_detection.py atau buat custom script
detector.configure_zone(4, confidence=0.15)  # Zone center lebih sensitif
detector.configure_zone(0, confidence=0.35)  # Zone jauh kurang sensitif
```

### **Optimize untuk Parking Lot:**
```bash
# Confidence lebih rendah untuk motor parkir
python3 start_zone_detection.py --source "rtsp://..." --mode sequential
```

## 📈 Monitoring & Statistics

### **Real-time Stats yang Ditampilkan:**
- FPS dan frame count
- Total detections across all zones
- Detection rate per frame
- Zone breakdown (motorcycles and plates per zone)

### **End-of-Session Report:**
- Comprehensive zone statistics
- Plate success rate
- Performance metrics
- Per-zone breakdown

## 💡 Tips untuk Kondisi Screenshot Anda

### **Optimal Settings:**
1. **Mode**: Sequential (performance terbaik)
2. **Zone Focus**: Middle zones (3,4,5) paling produktif
3. **Expectation**: Focus pada counting dan movement, bukan plate reading

### **Performance Tuning:**
```bash
# Untuk hardware terbatas
python3 start_zone_detection.py --mode sequential --duration 120

# Untuk monitoring 24/7
python3 start_zone_detection.py --mode adaptive --no-zones
```

### **Best Practices:**
- Start dengan sequential mode
- Monitor zone statistics untuk optimize
- Use adaptive mode untuk kondisi mixed
- Save screenshots untuk analysis

## 🎯 Use Cases Optimal

### 1. **Parking Management**
- Count motor per area
- Monitor occupancy
- Detect movement patterns

### 2. **Security Monitoring**
- Detect unauthorized access
- Monitor suspicious activity
- Zone-based alerting

### 3. **Traffic Analysis**
- Entry/exit monitoring
- Peak hour analysis
- Flow pattern detection

## 🔮 Advanced Features

### **Zone Priority System:**
- High priority zones checked lebih sering
- Adaptive learning berdasarkan historical data
- Custom confidence per zone

### **Statistics Tracking:**
- Per-zone detection history
- Performance metrics
- Activity heatmaps

---

## 📞 Quick Troubleshooting

### **Low Detection Rate:**
```bash
# Lower confidence untuk semua zones
python3 start_zone_detection.py --mode sequential
# Lalu press 'm' untuk coba mode lain
```

### **Performance Issues:**
```bash
# Gunakan sequential mode
python3 start_zone_detection.py --mode sequential --no-zones
```

### **No Motorcycles Detected:**
- Check camera angle dan lighting
- Try parallel mode untuk coverage maksimal
- Verify RTSP stream working

**Zone mode adalah solusi optimal untuk kondisi parking lot seperti screenshot Anda! 🎯🏍️**