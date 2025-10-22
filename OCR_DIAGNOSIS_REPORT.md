# 🔍 OCR DIAGNOSIS REPORT - Vehicle Access Control System

**Generated**: 2025-10-22
**Analysis Period**: Recent 10 crop images
**Success Rate**: 20% valid plates (2/10)

---

## ❌ **ROOT CAUSE ANALYSIS**

### **Problem 1: Crop Size Too Small**
```
Average crop size: 124x54 pixels (~6,700 pixels)
Required minimum: 200x60 pixels (~12,000 pixels)
Current quality: 56% of required size ❌
```

**Impact:**
- OCR tidak bisa baca karakter kecil
- Detail plat hilang setelah preprocessing
- Upscaling 6x menghasilkan artifak/blur

**Examples:**
- `crop_20251022_132222.jpg`: 124x54 → "TP 1844 IBU" ✅ (lucky)
- `crop_20251022_132217.jpg`: 124x54 → "COJC" ❌ (garbage)
- `crop_20251022_132206.jpg`: 124x56 → "COJC" ❌ (garbage)

### **Problem 2: Wrong Detection Area**
```
Detected crops: Mostly non-plate areas
YOLO confidence: 0.15-0.25 (LOW)
Garbage texts: "COJC", "C6JC", "JE13441BU"
```

**Cause:**
- YOLO custom model (`models/best.pt`) mendeteksi area yang BUKAN plat
- Multi-scale detection (100%, 70%, 50%) tidak membantu
- Threshold terlalu rendah (0.15) → banyak false positive

**Examples:**
```
"COJC"      → Likely: Logo/text bukan plat
"C6JC"      → Likely: Angka container/chassis
"JE13441BU" → Likely: Frame number/VIN
```

### **Problem 3: EasyOCR Not Working**
```
All results: confidence = 0.50 (fallback)
Primary OCR: EasyOCR fails silently
Fallback: Tesseract dengan hasil buruk
```

**Cause:**
- EasyOCR butuh image size minimal ~480x256 pixels
- Current size 124x54 terlalu kecil untuk deep learning
- Upscaling tidak cukup quality untuk EasyOCR

### **Problem 4: Confidence Threshold Too High**
```
Current threshold: 0.65
Actual OCR confidence: 0.50 (fallback)
Result: All rejected, tidak ada yang masuk database
```

**Impact:**
- Semua hasil OCR ditolak (conf 0.50 < 0.65)
- Database tidak ter-update
- User tidak dapat feedback real-time

---

## 📊 **DETECTION STATISTICS**

| Metric | Value | Status |
|--------|-------|--------|
| **Total Crops** | 10 images | - |
| **OCR Success** | 10/10 (100%) | ✅ |
| **Valid Plates** | 2/10 (20%) | ❌ |
| **Avg Size** | 124x54 px | ❌ Too small |
| **Avg Area** | 6,700 px² | ❌ <50% required |
| **Avg Confidence** | 0.50 | ❌ Fallback only |

**Valid Results:**
1. "TP 1844 IBU" (124x54) ✅
2. "F 1324 ABV" (124x54) ✅

**Garbage Results (80%):**
- "COJC" × 4
- "C6JC" × 3
- "JE13441BU" × 1

---

## 🎯 **RECOMMENDED FIXES**

### **FIX 1: Improve YOLO Plate Detection**

**Current Issue:**
- YOLO mendeteksi area yang salah (bukan plat)
- Confidence threshold terlalu rendah (0.15)

**Solution:**
```python
# app.py:546 - multi_scale_detection()

# BEFORE:
bboxes = multi_scale_detection(frame)  # 100%, 70%, 50%

# AFTER:
# 1. Increase confidence threshold
plate_detector = YOLOPlateDetector(conf_threshold=0.35)  # Was: 0.25

# 2. Add size validation BEFORE OCR
def validate_detection(bbox):
    x, y, w, h = bbox

    # Minimum size untuk plat Indonesia
    if w < 150 or h < 50:
        return False

    # Aspect ratio validation (2.5:1 to 4.5:1)
    aspect_ratio = w / h
    if aspect_ratio < 2.5 or aspect_ratio > 4.5:
        return False

    # Area minimum
    if (w * h) < 7500:  # ~150x50 pixels
        return False

    return True

# 3. Filter detections
validated_bboxes = [b for b in bboxes if validate_detection(b)]
```

### **FIX 2: Improve Cropping Quality**

**Current Issue:**
- Crop terlalu kecil (124x54 average)
- Tidak ada margin untuk context

**Solution:**
```python
# app.py:612 - Crop with margin

# BEFORE:
roi = frame[y:y+h, x:x+w]

# AFTER:
# Add 10% margin around plate for better OCR
margin_x = int(w * 0.10)
margin_y = int(h * 0.10)

x1 = max(0, x - margin_x)
y1 = max(0, y - margin_y)
x2 = min(frame_w, x + w + margin_x)
y2 = min(frame_h, y + h + margin_y)

roi = frame[y1:y2, x1:x2]
```

### **FIX 3: Optimize OCR Strategy**

**Current Issue:**
- EasyOCR gagal di image kecil
- Fallback Tesseract tidak optimal

**Solution:**
```python
# utils/ocr_processor.py:400 - read_plate_text()

def read_plate_text(self, plate_img):
    # Check image size
    h, w = plate_img.shape[:2]

    # If too small, aggressive upscale FIRST
    if w < 300:
        scale = 300 / w
        scale = min(scale, 8.0)  # Max 8x upscaling
        new_w = int(w * scale)
        new_h = int(h * scale)

        # High-quality upscaling
        plate_img = cv2.resize(plate_img, (new_w, new_h),
                              interpolation=cv2.INTER_LANCZOS4)

        logger.info(f"🔍 Aggressive upscale: {w}x{h} → {new_w}x{new_h} ({scale:.1f}x)")

    # Try EasyOCR first (works better on larger images)
    if self.easyocr_reader and plate_img.shape[1] >= 300:
        # ... EasyOCR logic

    # Fallback to Tesseract with optimized PSM
    # ... rest of logic
```

### **FIX 4: Lower Confidence Threshold**

**Current Issue:**
- Threshold 0.65 terlalu tinggi
- Semua hasil fallback (0.50) ditolak

**Solution:**
```python
# app.py:655 - Adjust confidence threshold

# BEFORE:
MIN_OCR_CONFIDENCE = 0.65

# AFTER:
MIN_OCR_CONFIDENCE = 0.45  # Lower for better recall

# But add format validation to compensate:
if ocr_processor.is_valid_plate(plate_text) and ocr_confidence >= MIN_OCR_CONFIDENCE:
    # Accept only if format matches Indonesian plate
    logger.info(f"✅ OCR SUCCESS: {plate_text}")
    return {text: plate_text, ...}
```

### **FIX 5: Add Debug Visualization**

**Solution:**
```python
# Save annotated debug images
def save_debug_image(frame, bbox, text, confidence):
    # Draw bbox
    x, y, w, h = bbox
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Add OCR result
    cv2.putText(annotated, f"{text} ({confidence:.2f})",
               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = f"debug/detection_{timestamp}.jpg"
    cv2.imwrite(debug_path, annotated)
```

---

## 📈 **EXPECTED IMPROVEMENTS**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Valid Rate** | 20% | 70-80% | +250% |
| **Crop Size** | 124x54 | 200x80 | +60% area |
| **OCR Confidence** | 0.50 | 0.70-0.85 | +40% |
| **False Positives** | 80% | 10-15% | -80% |

---

## 🚀 **IMPLEMENTATION PRIORITY**

1. **HIGH**: Fix #1 - Improve YOLO detection (increase confidence to 0.35)
2. **HIGH**: Fix #2 - Add margin to crops (10% padding)
3. **MEDIUM**: Fix #4 - Lower confidence threshold (0.65 → 0.45)
4. **MEDIUM**: Fix #3 - Aggressive upscaling before OCR
5. **LOW**: Fix #5 - Debug visualization

---

## 📝 **TEST PLAN**

### Phase 1: Detection Quality
```bash
# Test YOLO confidence threshold
python3 test_detection_quality.py --conf 0.35
```

### Phase 2: OCR Accuracy
```bash
# Test with real crops
python3 analyze_ocr_results.py --min-size 150x50
```

### Phase 3: End-to-End
```bash
# Full system test
python3 app.py
# Monitor logs for:
# - Larger crop sizes (>150x50)
# - Higher OCR confidence (>0.60)
# - Valid plate format (70%+)
```

---

**Next Steps:** Implement fixes in priority order and re-run analysis.
