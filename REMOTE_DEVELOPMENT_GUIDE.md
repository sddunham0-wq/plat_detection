# 🌐 Remote Development Guide

## Situasi: Windows Local ↔️ macOS Remote

### 🎯 **Current Setup:**
- **Local**: Windows (Development)
- **Remote**: macOS (Execution)
- **Issue**: Camera permission di remote macOS

## 🔧 **Solutions:**

### **Option 1: Fix Remote macOS Camera (Recommended)**

#### **Step 1: Remote GUI Access**
```bash
# Enable Screen Sharing di remote macOS:
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.screensharing.plist

# Atau gunakan VNC/TeamViewer/AnyDesk
```

#### **Step 2: Fix Permission via GUI**
1. Remote desktop ke macOS
2. **Apple Menu > System Preferences**
3. **Security & Privacy > Privacy**
4. **Camera** (kiri) → **✅ Enable Terminal/Python**
5. **Restart terminal session**

#### **Step 3: Test Remote**
```bash
# Di remote macOS terminal:
python3 quick_camera_test.py
```

---

### **Option 2: Local Windows Development**

Setup project di Windows lokal dengan kamera Windows:

#### **Step 1: Copy Project ke Windows**
```bash
# Copy project folder ke Windows local
scp -r user@macos-remote:/path/to/project ./local-project
```

#### **Step 2: Setup Windows Environment**
```bash
# Di Windows Command Prompt:
cd local-project
pip install -r requirements.txt
pip install opencv-python

# Test kamera Windows:
python test_kamera.py
```

#### **Step 3: Run Detection di Windows**
```bash
# Windows camera detection:
python main.py --laptop-camera
python main.py --camera-select
```

---

### **Option 3: Hybrid Development**

**Develop di Windows, Execute di macOS:**

#### **Code di Windows:**
- Edit code di Windows dengan VS Code/PyCharm
- Use remote SSH extension
- Real-time sync dengan remote

#### **Execute di macOS:**
```bash
# Sync code ke remote:
rsync -av ./local-project/ user@remote:/path/to/project/

# Execute di remote (setelah fix camera permission):
ssh user@remote "cd /path/to/project && python3 main.py --laptop-camera"
```

---

### **Option 4: Mock Camera (Development Only)**

Untuk development tanpa real camera:

```python
# Di config.py tambah:
class MockCameraConfig:
    USE_MOCK_CAMERA = True
    MOCK_VIDEO_FILE = "test_video.mp4"  # Video file untuk testing
    MOCK_RESOLUTION = (640, 480)
```

```bash
# Test dengan mock:
python main.py --source test_video.mp4
```

---

## 🛠️ **Recommended Workflow:**

### **For Active Development:**
```
Windows (Local) ←→ macOS (Remote)
     ↓                    ↓
  Code Edit         Execute & Test
  Git Commit        Camera Access
```

### **Commands:**
```bash
# Local Windows (development):
git add .
git commit -m "camera feature"
git push origin main

# Remote macOS (execution):
git pull origin main
python3 main.py --laptop-camera
```

---

## 🎥 **Camera Access Status Check:**

### **Remote macOS Check:**
```bash
# Check camera permission status:
tccutil reset Camera  # Reset first
python3 quick_camera_test.py  # Test access
```

### **Expected Success:**
```
🎥 Quick Camera Test
==============================
Opening camera...
✅ Camera opened!
✅ Frame captured: 1280x720
🎉 CAMERA WORKS!

🚀 Now you can use:
   python3 main.py --laptop-camera
```

---

## 📋 **Next Steps:**

1. **Choose your preferred option** (Remote fix vs Local Windows)
2. **Fix camera permission** if using remote macOS
3. **Test camera access** dengan quick_camera_test.py
4. **Start detection** dengan main.py --laptop-camera

The camera integration is fully ready - just need to resolve the permission issue! 🎉