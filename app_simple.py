#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTEM DETEKSI PLAT NOMOR - VERSI SEDERHANA & STABIL
Sistem akses kendaraan otomatis untuk SMK

Fitur:
- Streaming kamera stabil
- Deteksi plat nomor
- OCR baca plat
- Cek database
- Control palang otomatis
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import mysql.connector
import logging
from datetime import datetime
import threading
import time
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from config import config

# Try import YOLO detector dengan fallback ke simple detector
try:
    from utils.yolo_plate_detector import YOLOPlateDetector
    USE_YOLO = True
    logger.info("✅ YOLO detector module available")
except ImportError as e:
    from utils.plate_detector_simple import SimplePlateDetector
    USE_YOLO = False
    logger.warning(f"⚠️ YOLO not available: {e}")
    logger.info("ℹ️  Will use contour-based detector")

from utils.ocr_processor import OCRProcessor
from utils.plate_validator import IndonesianPlateValidator

# ========================================
# FLASK APP INIT
# ========================================

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ========================================
# GLOBAL VARIABLES
# ========================================

camera = None
camera_lock = threading.Lock()
last_detection = None
gate_status = "closed"
is_detecting = False

# Initialize detector with YOLO atau fallback ke Simple
# Penjelasan SMK: Bikin "mesin pencari plat" yang paling akurat tersedia
# Priority: YOLO (akurat) → Contour (fallback)
if USE_YOLO:
    try:
        plate_detector = YOLOPlateDetector(
            model_path='models/best.pt',  # Try custom model first
            conf_threshold=0.25
        )
        logger.info("✅ YOLO Plate Detector initialized")
        logger.info(f"   Model: {plate_detector.model_path}")
        logger.info(f"   Type: {plate_detector.model_type}")
    except Exception as e:
        logger.error(f"❌ YOLO initialization failed: {e}")
        logger.info("ℹ️  Falling back to Simple Detector...")
        plate_detector = SimplePlateDetector()
        USE_YOLO = False
else:
    # Fallback: Simple contour-based detector
    plate_detector = SimplePlateDetector()
    logger.info("✅ Simple Plate Detector initialized (contour-based)")

ocr_processor = OCRProcessor()
plate_validator = IndonesianPlateValidator()

# ========================================
# DATABASE FUNCTIONS
# ========================================

def get_db():
    """Connect ke MySQL database"""
    try:
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            autocommit=True
        )
        return conn
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None

def check_vehicle(plate_text):
    """
    Cek kendaraan di database

    Returns:
        dict: {'allowed': bool, 'owner': str, 'type': str}
    """
    conn = get_db()
    if not conn:
        return {'allowed': False, 'owner': 'DB Error', 'type': 'unknown'}

    try:
        cursor = conn.cursor(dictionary=True)

        # Remove spaces untuk match database
        plate_clean = plate_text.replace(' ', '')

        # Query database
        cursor.execute("""
            SELECT nama_pemilik, jenis_kendaraan
            FROM kendaraan_terdaftar
            WHERE nomor_plat = %s AND status = 'aktif'
        """, (plate_clean,))

        vehicle = cursor.fetchone()
        cursor.close()
        conn.close()

        if vehicle:
            return {
                'allowed': True,
                'owner': vehicle['nama_pemilik'],
                'type': vehicle['jenis_kendaraan']
            }
        else:
            return {
                'allowed': False,
                'owner': 'Tidak Terdaftar',
                'type': 'unknown'
            }

    except Exception as e:
        logger.error(f"Check vehicle error: {e}")
        return {'allowed': False, 'owner': 'Error', 'type': 'unknown'}

def save_plate_image(roi, plate_text, vehicle_type="unknown"):
    """
    Save cropped plate image ke folder gambarplat/

    Args:
        roi: Cropped plate image
        plate_text: Plate text (e.g., "F 1818 HG")
        vehicle_type: "mobil" atau "motor"

    Returns:
        path: Relative path ke saved image (e.g., "gambarplat/F1818HG_20250116_143025.jpg")
    """
    try:
        # Create folder jika belum ada
        os.makedirs('gambarplat', exist_ok=True)

        # Generate filename: PLATNO_DATE_TIME.jpg
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plate_clean = plate_text.replace(' ', '').replace('-', '')  # F1818HG
        filename = f"{plate_clean}_{vehicle_type}_{timestamp}.jpg"
        filepath = os.path.join('gambarplat', filename)

        # Save image
        cv2.imwrite(filepath, roi)

        logger.info(f"💾 Plate image saved: {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Save plate image error: {e}")
        return None

def log_access(plate_text, confidence, allowed, image_path=None):
    """Catat akses ke database dengan foto plat"""
    conn = get_db()
    if not conn:
        return

    try:
        cursor = conn.cursor()

        status = 'boleh_masuk' if allowed else 'ditolak'
        action = 'opened' if allowed else 'closed'

        cursor.execute("""
            INSERT INTO log_akses_masuk
            (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, path_foto)
            VALUES (%s, %s, %s, %s, %s)
        """, (plate_text, confidence, status, action, image_path))

        cursor.close()
        conn.close()

        logger.info(f"Access logged: {plate_text} - {status} - {image_path}")

    except Exception as e:
        logger.error(f"Log access error: {e}")

# ========================================
# CAMERA FUNCTIONS
# ========================================

class StableCamera:
    """Camera dengan auto-reconnect dan frame validation"""

    def __init__(self, url):
        self.url = url
        self.cap = None
        self.is_opened = False
        self.reconnect_delay = 2  # seconds
        self.last_frame = None
        self.frame_count = 0
        self.bad_frame_count = 0
        self.max_bad_frames = 5

    def connect(self):
        """Connect ke kamera dengan optimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.url)

            # Optimal OpenCV settings untuk stability
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer - fresh frames
            self.cap.set(cv2.CAP_PROP_FPS, 25)  # Target 25 FPS
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG codec

            if self.cap.isOpened():
                # Test read untuk verify
                test_ret, test_frame = self.cap.read()
                if test_ret and self.validate_frame(test_frame):
                    self.is_opened = True
                    self.last_frame = test_frame
                    self.bad_frame_count = 0
                    logger.info("✅ Camera connected and validated")
                    return True
                else:
                    logger.error("❌ Camera opened but frame invalid")
                    self.release()
                    return False
            else:
                logger.error("❌ Camera failed to open")
                return False

        except Exception as e:
            logger.error(f"Camera connect error: {e}")
            return False

    def validate_frame(self, frame):
        """Validate frame quality"""
        if frame is None:
            return False

        # Check frame shape
        if len(frame.shape) != 3:
            return False

        # Check frame size (minimum 320x240)
        h, w = frame.shape[:2]
        if w < 320 or h < 240:
            return False

        # Check if frame is not empty (all black/white)
        mean_val = frame.mean()
        if mean_val < 10 or mean_val > 245:
            return False

        # Check if frame has variance (not uniform)
        std_val = frame.std()
        if std_val < 5:
            return False

        return True

    def read(self):
        """Read frame dengan validation dan retry"""
        if not self.is_opened or self.cap is None:
            if not self.connect():
                return False, self.last_frame

        try:
            ret, frame = self.cap.read()

            if ret and self.validate_frame(frame):
                # Good frame
                self.last_frame = frame.copy()  # Deep copy
                self.frame_count += 1
                self.bad_frame_count = 0
                return True, frame

            else:
                # Bad frame
                self.bad_frame_count += 1
                logger.debug(f"Bad frame detected ({self.bad_frame_count}/{self.max_bad_frames})")

                # Too many bad frames - reconnect
                if self.bad_frame_count >= self.max_bad_frames:
                    logger.warning("Too many bad frames, reconnecting...")
                    self.release()
                    time.sleep(self.reconnect_delay)
                    self.connect()

                # Return last good frame
                return False, self.last_frame

        except Exception as e:
            logger.error(f"Frame read exception: {e}")
            self.bad_frame_count += 1
            return False, self.last_frame

    def release(self):
        """Release camera safely"""
        try:
            if self.cap:
                self.cap.release()
        except:
            pass
        self.is_opened = False
        self.bad_frame_count = 0

# ========================================
# DETECTION FUNCTION
# ========================================

def process_frame(frame):
    """
    Process frame untuk deteksi plat dengan stable OpenCV

    Returns:
        annotated_frame: Frame dengan bounding box hijau + label
    """
    global last_detection, gate_status, is_detecting

    # Skip jika sedang proses
    if is_detecting:
        return frame

    is_detecting = True

    try:
        # Validate frame dulu
        if frame is None or frame.size == 0:
            is_detecting = False
            return frame

        # 1. Detect plate dengan error handling
        # Gunakan YOLO atau Simple detector (auto-detected saat init)
        try:
            boxes = plate_detector.detect(frame)

            # Debug logging
            if boxes:
                logger.debug(f"✅ Detection: {len(boxes)} box(es) found")
                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    ratio = w / h if h > 0 else 0
                    logger.debug(f"  Box #{i+1}: pos=({x},{y}) size={w}x{h} ratio={ratio:.2f}")
            else:
                logger.debug("⚠️  Detection: No boxes found")

        except Exception as e:
            logger.error(f"Detection error: {e}")
            is_detecting = False
            return frame

        if not boxes:
            is_detecting = False
            return frame

        # 2. Prepare annotated frame
        annotated = frame.copy()
        GREEN = (0, 255, 0)  # Warna hijau konsisten

        # 3. DRAW BOUNDING BOX SEGERA (sebelum OCR!)
        # Gambar semua boxes yang terdeteksi
        for i, box in enumerate(boxes):
            bx, by, bw, bh = box
            # Draw rectangle hijau
            cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), GREEN, 2)

            # Label sementara
            label = "DETECTING..." if i == 0 else f"PLATE #{i+1}"
            cv2.putText(annotated, label, (bx, by-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        # 4. Process PERTAMA plat saja untuk OCR
        x, y, w, h = boxes[0]

        # Validate ROI bounds
        frame_h, frame_w = frame.shape[:2]
        if x < 0 or y < 0 or x+w > frame_w or y+h > frame_h:
            logger.warning("ROI out of bounds, skipping OCR")
            is_detecting = False
            return annotated  # Return dengan box sudah tergambar

        # Extract ROI dengan safe bounds
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            is_detecting = False
            return annotated  # Return dengan box sudah tergambar

        # 5. OCR dengan error handling (optional - box sudah tampil)
        try:
            plate_text, confidence = ocr_processor.read_plate_with_confidence(roi)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            plate_text = None
            confidence = 0.0

        # 6. Update label jika OCR berhasil
        if plate_text and confidence > 0.3:
            logger.info(f"📋 Detected: {plate_text} ({confidence:.2f})")

            # Validate format
            try:
                is_valid = plate_validator.is_valid_format(plate_text)
            except:
                is_valid = False

            if is_valid:
                # Check database
                vehicle = check_vehicle(plate_text)

                # Save cropped plate image
                try:
                    image_path = save_plate_image(roi, plate_text, vehicle['type'])
                except Exception as e:
                    logger.error(f"Save image error: {e}")
                    image_path = None

                # Update status
                last_detection = {
                    'plate': plate_text,
                    'confidence': confidence,
                    'allowed': vehicle['allowed'],
                    'owner': vehicle['owner'],
                    'type': vehicle['type'],
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'image': image_path
                }

                # Control gate
                if vehicle['allowed']:
                    gate_status = "opened"
                    logger.info(f"✅ ALLOWED: {vehicle['owner']}")
                else:
                    gate_status = "closed"
                    logger.warning(f"❌ DENIED: {plate_text}")

                # Log to database dengan foto
                try:
                    log_access(plate_text, confidence, vehicle['allowed'], image_path)
                except Exception as e:
                    logger.error(f"Log access error: {e}")

                # 7. UPDATE LABEL dengan info lengkap (re-draw)
                # Clear old label area dengan rectangle hitam
                cv2.rectangle(annotated, (x, y-90), (x+w+100, y), (0, 0, 0), -1)

                # Label JENIS KENDARAAN
                vehicle_type = vehicle['type'].upper()  # "MOBIL" atau "MOTOR"
                cv2.putText(annotated, vehicle_type, (x, y-60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)

                # Status
                status_text = "BOLEH MASUK" if vehicle['allowed'] else "DITOLAK"
                cv2.putText(annotated, status_text, (x, y-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

                # Nama pemilik
                if y > 90:
                    cv2.putText(annotated, vehicle['owner'][:30], (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)

        is_detecting = False
        return annotated

    except Exception as e:
        logger.error(f"Process frame error: {e}")
        is_detecting = False
        return frame

# ========================================
# VIDEO STREAMING
# ========================================

def generate_frames():
    """Generate frames untuk streaming dengan smooth motion"""
    global camera

    # Init camera
    if camera is None:
        camera = StableCamera(config.CAMERA_URL)

    frame_skip = 0
    last_processed_frame = None

    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success or frame is None:
            # Use blank frame atau last frame
            if last_processed_frame is not None:
                frame = last_processed_frame
            else:
                frame = cv2.imread('static/no_camera.jpg')
                if frame is None:
                    frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            time.sleep(0.05)
        else:
            # Process every 2nd frame untuk smooth motion
            frame_skip += 1
            if frame_skip % 2 == 0:
                processed = process_frame(frame)
                if processed is not None:
                    last_processed_frame = processed
                    frame = processed
            # Else: gunakan frame asli (tanpa detection)

        # Encode dengan quality optimal untuk smooth streaming
        ret, buffer = cv2.imencode('.jpg', frame, [
            cv2.IMWRITE_JPEG_QUALITY, 90,  # Naik dari 85
            cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimize encoding
        ])

        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.02)  # ~50 FPS untuk smooth motion

# ========================================
# ROUTES
# ========================================

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/status')
def api_status():
    """API status sistem"""
    return jsonify({
        'gate': gate_status,
        'last_detection': last_detection,
        'camera': 'connected' if (camera and camera.is_opened) else 'disconnected'
    })

@app.route('/api/gate/<action>')
def api_gate(action):
    """Manual control palang"""
    global gate_status

    if action in ['open', 'close']:
        gate_status = 'opened' if action == 'open' else 'closed'
        logger.info(f"Manual gate: {gate_status}")
        return jsonify({'success': True, 'gate': gate_status})

    return jsonify({'success': False, 'error': 'Invalid action'})

@app.route('/api/stats')
def api_stats():
    """Statistik hari ini"""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database error'})

    try:
        cursor = conn.cursor(dictionary=True)

        # Count today
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status_akses = 'boleh_masuk' THEN 1 ELSE 0 END) as allowed,
                SUM(CASE WHEN status_akses = 'ditolak' THEN 1 ELSE 0 END) as denied
            FROM log_akses_masuk
            WHERE DATE(waktu_deteksi) = CURDATE()
        """)

        stats = cursor.fetchone()
        cursor.close()
        conn.close()

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/log_akses')
def log_akses_page():
    """Halaman log akses dengan foto plat"""
    return render_template('log_akses.html')

@app.route('/api/log_akses')
def api_log_akses():
    """API untuk get log akses terbaru"""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database error'})

    try:
        cursor = conn.cursor(dictionary=True)

        # Get logs (limit 50 terbaru)
        cursor.execute("""
            SELECT
                id_log,
                plat_terdeteksi,
                tingkat_yakin,
                status_akses,
                aksi_palang,
                path_foto,
                waktu_deteksi,
                catatan
            FROM log_akses_masuk
            ORDER BY waktu_deteksi DESC
            LIMIT 50
        """)

        logs = cursor.fetchall()

        # Format datetime untuk JSON
        for log in logs:
            if log['waktu_deteksi']:
                log['waktu_deteksi'] = log['waktu_deteksi'].strftime('%Y-%m-%d %H:%M:%S')

        cursor.close()
        conn.close()

        return jsonify({'success': True, 'logs': logs})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/gambarplat/<filename>')
def serve_plate_image(filename):
    """Serve cropped plate images dari folder gambarplat/"""
    try:
        # Security: Prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': 'Invalid filename'}), 400

        # Serve file dari folder gambarplat/
        from flask import send_from_directory
        return send_from_directory('gambarplat', filename)

    except FileNotFoundError:
        # Return placeholder image jika file tidak ditemukan
        return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': str(e)}), 500

# ========================================
# MAIN
# ========================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 STARTING SISTEM DETEKSI PLAT NOMOR")
    logger.info("=" * 60)

    # Create folders
    os.makedirs('gambarplat', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    logger.info(f"📹 Camera: {config.CAMERA_HOST}")
    logger.info(f"💾 Database: {config.DB_NAME}")
    logger.info(f"🌐 Server: http://localhost:{config.FLASK_PORT}")
    logger.info("=" * 60)

    # Run Flask
    app.run(
        host='0.0.0.0',
        port=config.FLASK_PORT,
        debug=False,  # False untuk stabilitas
        threaded=True
    )
