# Person Detection Feature - Panduan Lengkap

## 📋 Overview

Person detection telah ditambahkan sebagai **isolated optional feature** yang tidak mengganggu sistem plate detection yang sudah stabil.

### ✅ Key Features

- **Isolated System**: Person detection berjalan terpisah dari plate detection
- **Error-Safe**: Jika person detection error, plate detection tetap berjalan normal
- **Optional**: Default disabled, dapat diaktifkan/disabled kapan saja
- **Real-time Toggle**: Dapat diaktifkan/disabled saat streaming berjalan
- **Blue Bounding Boxes**: Person menggunakan warna BIRU, berbeda dari plate (hijau/merah)

---

## 🚀 Quick Start

### 1. Aktifkan Person Detection (via Config)

Edit `config.py`:

```python
class PersonDetectionConfig:
    ENABLE_PERSON_DETECTION = True  # ← Ubah jadi True
```

### 2. Start Streaming

```bash
python headless_stream.py
```

### 3. Toggle via API (Runtime)

**Enable person detection:**
```bash
curl -X POST http://localhost:5000/api/toggle_person_detection \
  -H "Content-Type: application/json" \
  -d '{"enable": true}'
```

**Disable person detection:**
```bash
curl -X POST http://localhost:5000/api/toggle_person_detection \
  -H "Content-Type: application/json" \
  -d '{"enable": false}'
```

---

## 🔧 Configuration

Semua pengaturan ada di `config.py`:

```python
class PersonDetectionConfig:
    # Enable/disable (default: disabled untuk backward compatibility)
    ENABLE_PERSON_DETECTION = False

    # Detection thresholds
    PERSON_CONFIDENCE = 0.5              # 50% confidence threshold
    PERSON_MAX_DETECTIONS = 20           # Max 20 persons per frame

    # Visual styling
    PERSON_BBOX_COLOR = (255, 0, 0)      # Blue (BGR)
    PERSON_BBOX_THICKNESS = 2
    PERSON_SHOW_CONFIDENCE = True        # Show confidence scores

    # Model
    PERSON_YOLO_MODEL = 'yolov8n.pt'     # YOLOv8 nano model

    # Performance
    PERSON_DETECTION_PARALLEL = True     # Run parallel dengan plate detection
    PERSON_FRAME_SKIP = 1                # Process every frame (1 = no skip)
```

---

## 📊 Statistics API

Person detection menambahkan statistics baru:

```bash
curl http://localhost:5000/api/stats
```

**Response includes:**
```json
{
  "total_persons_detected": 150,
  "person_detection_enabled": true,
  "person_enabled": true,
  "person_total_detections": 150,
  "person_avg_detection_time": 0.045,
  "person_detection_fps": 22.2,
  "person_confidence_threshold": 0.5,
  ...
}
```

---

## 🧪 Testing Guide

### Test 1: Baseline (Person Detection OFF)

1. **Ensure person detection disabled:**
   ```python
   # config.py
   ENABLE_PERSON_DETECTION = False
   ```

2. **Start streaming:**
   ```bash
   python headless_stream.py
   ```

3. **Verify plate detection works:**
   - Open browser: `http://localhost:5000`
   - Check plat nomor terdeteksi dengan bounding box hijau/merah
   - No person bounding boxes (blue) muncul

4. **Check statistics:**
   ```bash
   curl http://localhost:5000/api/stats | grep person_detection_enabled
   # Should show: "person_detection_enabled": false
   ```

**✅ Expected:** Plate detection berfungsi sempurna tanpa person detection

---

### Test 2: Person Detection Enabled

1. **Enable person detection:**
   ```python
   # config.py
   ENABLE_PERSON_DETECTION = True
   ```

2. **Start streaming:**
   ```bash
   python headless_stream.py
   ```

3. **Verify both detections:**
   - Plat nomor: Bounding box hijau/merah (plate detection)
   - Orang: Bounding box BIRU dengan label "Person: 0.85" (person detection)

4. **Check statistics:**
   ```bash
   curl http://localhost:5000/api/stats | grep -E "(person_detection|total_persons)"
   ```

   Should show:
   - `"person_detection_enabled": true`
   - `"total_persons_detected": 50` (or higher)

**✅ Expected:** Both plate AND person detection berfungsi bersamaan

---

### Test 3: Runtime Toggle

1. **Start with person detection OFF:**
   ```bash
   python headless_stream.py
   # (dengan ENABLE_PERSON_DETECTION = False)
   ```

2. **Enable via API saat streaming:**
   ```bash
   curl -X POST http://localhost:5000/api/toggle_person_detection \
     -H "Content-Type: application/json" \
     -d '{"enable": true}'
   ```

3. **Verify person detection activated:**
   - Blue bounding boxes mulai muncul di stream
   - Statistics menunjukkan `total_persons_detected` naik

4. **Disable via API:**
   ```bash
   curl -X POST http://localhost:5000/api/toggle_person_detection \
     -H "Content-Type: application/json" \
     -d '{"enable": false}'
   ```

5. **Verify person detection stopped:**
   - Blue bounding boxes hilang
   - Plate detection masih berfungsi

**✅ Expected:** Toggle berfungsi tanpa restart, plate detection tidak terpengaruh

---

### Test 4: Error Isolation

1. **Simulate person detection error:**
   - Edit `utils/person_detector.py`:
   ```python
   def detect_persons(self, frame):
       raise Exception("Test error")  # Simulate error
   ```

2. **Start streaming dengan person detection enabled**

3. **Verify error isolation:**
   - Error log muncul: `"❌ Person detection error (isolated): Test error"`
   - Plate detection **tetap berfungsi normal**
   - Stream tidak crash

4. **Remove error simulation dan restart**

**✅ Expected:** Sistem tetap stable meskipun person detection error

---

## 🎨 Visual Indicators

| Detection Type | Bounding Box Color | Label Format |
|----------------|-------------------|--------------|
| License Plate  | 🟢 Green / 🔴 Red | "B1234ABC (85.5%)" |
| Person         | 🔵 Blue           | "Person: 0.85" |

---

## 🔍 Troubleshooting

### Person detection tidak berfungsi?

1. **Check YOLO model:**
   ```bash
   ls -la yolov8n.pt
   # Model harus ada di project root
   ```

2. **Check ultralytics installed:**
   ```bash
   pip list | grep ultralytics
   # Harus installed
   ```

3. **Check logs:**
   ```bash
   # Saat start streaming, harus ada log:
   # "✅ Person Detector initialized successfully"
   ```

4. **Check config:**
   ```python
   # Pastikan enabled di config.py:
   ENABLE_PERSON_DETECTION = True
   ```

### Person detection too sensitive?

Ubah confidence threshold:
```python
PERSON_CONFIDENCE = 0.7  # Increase dari 0.5 ke 0.7
```

### Performance issues?

1. **Skip frames:**
   ```python
   PERSON_FRAME_SKIP = 2  # Process every 2nd frame
   ```

2. **Reduce max detections:**
   ```python
   PERSON_MAX_DETECTIONS = 10  # Reduce dari 20
   ```

3. **Disable person detection sementara:**
   ```bash
   curl -X POST http://localhost:5000/api/toggle_person_detection \
     -H "Content-Type: application/json" \
     -d '{"enable": false}'
   ```

---

## 📝 Implementation Details

### Architecture

```
┌─────────────────────────────────────┐
│      HeadlessStreamManager          │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Plate        │  │ Person      │ │
│  │ Detector     │  │ Detector    │ │
│  │ (Primary)    │  │ (Optional)  │ │
│  └──────────────┘  └─────────────┘ │
│         │                 │         │
│         └────────┬────────┘         │
│                  │                  │
│         ┌────────▼─────────┐        │
│         │  Combined Frame  │        │
│         │  (Both Bboxes)   │        │
│         └──────────────────┘        │
└─────────────────────────────────────┘
```

### Key Components

1. **`utils/person_detector.py`**
   - Isolated person detection class
   - YOLOv8-based detection
   - Error-safe implementation

2. **`stream_manager.py`**
   - Integrated person detector (optional)
   - Parallel detection processing
   - Runtime toggle support

3. **`config.py`**
   - `PersonDetectionConfig` class
   - All person detection settings

4. **`headless_stream.py`**
   - API endpoint: `/api/toggle_person_detection`
   - Statistics include person detection

### Safety Features

- ✅ **Error Isolation**: Person detection errors tidak crash plate detection
- ✅ **Try-Catch Blocks**: Semua person detection code wrapped dalam exception handlers
- ✅ **Default Disabled**: Backward compatible - tidak mempengaruhi existing users
- ✅ **Independent Threading**: Person detection tidak block plate detection
- ✅ **Graceful Degradation**: Jika YOLO unavailable, sistem tetap jalan tanpa person detection

---

## 🎯 Use Cases

1. **Security Monitoring**: Deteksi orang + vehicle plate bersamaan
2. **People Counting**: Hitung jumlah orang di area CCTV
3. **Parking Monitoring**: Track vehicles AND people
4. **Traffic Analysis**: Analyze pedestrian + vehicle patterns

---

## 📞 Support

Jika ada issue:
1. Check logs di console
2. Verify configuration di `config.py`
3. Test dengan person detection OFF dulu (baseline)
4. Enable person detection dan test
5. Check API response untuk error messages

**Note:** Person detection adalah **optional feature** - sistem plate detection akan tetap berfungsi sempurna dengan atau tanpa person detection enabled.
