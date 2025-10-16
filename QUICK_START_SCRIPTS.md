# Quick Start Scripts - Person Detection

## 🚀 Available Scripts

### 1. **`restart_stream.sh`** - Auto-kill & Restart
```bash
./restart_stream.sh
```
- Auto-kill process yang sedang berjalan di port 5000
- Start `headless_stream.py`
- **Use case:** Quick restart tanpa manual kill

---

### 2. **`start_with_person_detection.sh`** - Start with Person Detection Info
```bash
./start_with_person_detection.sh
```
- Auto-kill existing process
- Start server dengan info person detection
- Show helpful commands untuk toggle

---

### 3. **`toggle_person_detection.sh`** - Quick Toggle Person Detection
```bash
# Enable person detection
./toggle_person_detection.sh on

# Disable person detection
./toggle_person_detection.sh off

# Check status
./toggle_person_detection.sh status
```

**Aliases:**
- `on` = `enable` = `1`
- `off` = `disable` = `0`
- `status` = `check`

---

## 📋 Typical Workflow

### **Step 1: Start Server**
```bash
./restart_stream.sh
```

### **Step 2: Open Browser**
```
http://localhost:5000
```

### **Step 3: Enable Person Detection (Optional)**
```bash
# Di terminal baru:
./toggle_person_detection.sh on
```

### **Step 4: Check Status**
```bash
./toggle_person_detection.sh status
```

**Output:**
```
Person Detection: ✅ ENABLED
Total Persons Detected: 25
```

### **Step 5: Disable Jika Perlu**
```bash
./toggle_person_detection.sh off
```

---

## 🔧 Manual Commands

### Kill Process Manual
```bash
# Kill by port
kill -9 $(lsof -ti:5000)

# Verify killed
lsof -ti:5000  # Should be empty
```

### Start Manual
```bash
python headless_stream.py
```

### Toggle via curl
```bash
# Enable
curl -X POST http://localhost:5000/api/toggle_person_detection \
  -H "Content-Type: application/json" \
  -d '{"enable": true}'

# Disable
curl -X POST http://localhost:5000/api/toggle_person_detection \
  -H "Content-Type: application/json" \
  -d '{"enable": false}'

# Check stats
curl http://localhost:5000/api/stats | grep person_detection_enabled
```

---

## 🎯 Testing Person Detection

### Test 1: Baseline (OFF)
```bash
# Start server
./restart_stream.sh

# Verify person detection OFF
./toggle_person_detection.sh status
# Should show: ⏸️ DISABLED

# Open browser: http://localhost:5000
# Should see: Only plate bounding boxes (no blue boxes)
```

### Test 2: Enable Person Detection
```bash
# Enable
./toggle_person_detection.sh on

# Check status
./toggle_person_detection.sh status
# Should show: ✅ ENABLED

# Browser: Blue bounding boxes muncul untuk person
```

### Test 3: Toggle Runtime
```bash
# Disable
./toggle_person_detection.sh off
# Blue boxes hilang

# Enable again
./toggle_person_detection.sh on
# Blue boxes muncul lagi
```

---

## 📊 Monitor Statistics

```bash
# Watch person detection stats real-time
watch -n 2 './toggle_person_detection.sh status'

# Or use curl with jq (if installed)
curl -s http://localhost:5000/api/stats | jq '.person_detection_enabled, .total_persons_detected'
```

---

## 🛠️ Troubleshooting

### Port 5000 still in use?
```bash
# Force kill
kill -9 $(lsof -ti:5000)

# Restart
./restart_stream.sh
```

### Person detection not working?
```bash
# Check status
./toggle_person_detection.sh status

# If disabled, enable it
./toggle_person_detection.sh on

# Check config
grep ENABLE_PERSON_DETECTION config.py
```

### Scripts not executable?
```bash
chmod +x *.sh
```

---

## 📝 Notes

- ✅ Semua scripts **auto-kill** existing process sebelum start
- ✅ Person detection **default OFF** (backward compatible)
- ✅ Toggle **tidak perlu restart** - real-time switch
- ✅ Plate detection **selalu ON** - tidak terpengaruh person detection

---

**Read full documentation:** `PERSON_DETECTION_GUIDE.md`
