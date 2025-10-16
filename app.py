from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import mysql.connector
from mysql.connector import Error as MySQLError
from datetime import datetime
import cv2
import os
import json
import threading
import time
import logging

# Setup logging untuk development (HARUS DI ATAS!)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import plate detector, OCR, dan vehicle analyzer
try:
    from utils.yolo_plate_detector import YOLOPlateDetector
    USE_YOLO = True
    logger.info("✅ YOLO detector available")
except ImportError as e:
    from utils.plate_detector import PlateDetector
    USE_YOLO = False
    logger.warning(f"⚠️ YOLO not available: {e}")
    logger.info("ℹ️  Using contour-based detector as fallback")

from utils.ocr_processor import OCRProcessor
from utils.vehicle_analyzer import vehicle_analyzer
from config import config

app = Flask(__name__)
app.secret_key = config.SECRET_KEY  # Secret key dari environment variable

# Global variables untuk camera dan detection
camera = None
latest_detection = None
all_detected_bboxes = []  # Simpan SEMUA bounding boxes untuk ditampilkan
detection_lock = threading.Lock()
system_status = {
    'camera_connected': False,
    'detection_active': False,
    'gate_status': 'closed',
    'last_detection_time': None
}

# Initialize plate detector (YOLO atau Contour-based dengan fallback)
# Penjelasan SMK: Bikin "mesin pencari plat" yang siap dipakai
# Prioritas: YOLO (lebih akurat) → Contour (fallback)
if USE_YOLO:
    try:
        plate_detector = YOLOPlateDetector(
            model_path='models/best.pt',
            conf_threshold=0.25
        )
        logger.info("✅ YOLO Plate Detector initialized successfully")
    except Exception as e:
        logger.error(f"❌ YOLO initialization failed: {e}")
        logger.info("ℹ️  Falling back to Contour-based detector...")
        from utils.plate_detector import PlateDetector
        plate_detector = PlateDetector(method='contour', max_detections=3)
        USE_YOLO = False
else:
    # Fallback contour-based detector
    plate_detector = PlateDetector(method='contour', max_detections=3)
    logger.info("✅ Contour-based Plate Detector initialized")

# Initialize OCR processor
# Penjelasan SMK: Bikin "mesin baca huruf" untuk baca teks plat
ocr_processor = OCRProcessor()

# Fungsi koneksi DB - MySQL (Laragon)
def get_db_connection():
    """
    Penjelasan SMK: Seperti 'buka pintu' untuk bicara dengan database
    MySQL = database server (Laragon), perlu server yang running!
    """
    try:
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            autocommit=False  # Manual commit untuk transaction control
        )
        return conn
    except MySQLError as e:
        logger.error(f"❌ MySQL connection error: {e}")
        logger.error(f"   Check: Laragon running? Database '{config.DB_NAME}' exists?")
        return None

# =====================================================
# CAMERA & DETECTION FUNCTIONS
# =====================================================

def initialize_camera():
    """
    Penjelasan SMK: Seperti 'nyalakan kamera' dan pastikan bisa dipakai
    Coba connect ke CCTV dulu, kalau gagal pakai webcam laptop
    """
    global camera, system_status
    try:
        # Coba connect ke CCTV dari config
        from config import config
        camera_url = config.CAMERA_URL
        logger.info(f"🎥 Mencoba koneksi ke CCTV: {camera_url}")
        camera = cv2.VideoCapture(camera_url)

        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ CCTV berhasil terhubung")
                return True
    except Exception as e:
        logger.warning(f"⚠️ CCTV tidak tersedia: {e}")

    # Fallback ke webcam laptop
    try:
        logger.info("🎥 Mencoba webcam laptop...")
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            ret, frame = camera.read()
            if ret and frame is not None:
                system_status['camera_connected'] = True
                logger.info("✅ Webcam laptop berhasil terhubung")
                return True
    except Exception as e:
        logger.error(f"❌ Webcam juga gagal: {e}")

    system_status['camera_connected'] = False
    logger.error("❌ Tidak ada kamera yang tersedia")
    return False

def generate_video_frames():
    """
    Penjelasan SMK: Seperti 'streaming video' di YouTube
    Ambil frame dari kamera, proses deteksi, kirim ke browser
    """
    global camera, latest_detection, system_status

    last_detection_time = 0
    DETECTION_COOLDOWN = 5  # 5 detik cooldown (santai, tidak spam)

    # Exponential backoff untuk camera reconnection
    retry_count = 0
    MAX_RETRY = 5
    BASE_DELAY = 2  # Start dengan 2 detik

    while True:
        if not camera or not camera.isOpened():
            # Coba reconnect camera dengan exponential backoff
            if not initialize_camera():
                retry_count += 1
                if retry_count >= MAX_RETRY:
                    logger.error(f"❌ Max retry reached ({MAX_RETRY}), using max delay")
                    retry_count = MAX_RETRY  # Cap at max

                # Exponential backoff: 2, 4, 8, 16, 32 detik (max)
                delay = min(BASE_DELAY * (2 ** (retry_count - 1)), 32)
                logger.warning(f"⏳ Camera reconnect failed, retry in {delay}s (attempt {retry_count}/{MAX_RETRY})")
                time.sleep(delay)
                continue
            else:
                # Reset retry count kalau berhasil
                retry_count = 0

        ret, frame = camera.read()
        if not ret:
            logger.warning("⚠️ Gagal baca frame, reconnecting...")
            camera = None
            continue

        # TIDAK resize frame - pakai resolution PENUH untuk deteksi tajam!
        # Penjelasan SMK: Semakin besar resolution, semakin detail deteksinya
        # Frame asli = lebih banyak pixel = plat lebih jelas meski jauh

        # Detection dengan cooldown (tidak spam)
        current_time = time.time()
        if current_time - last_detection_time > DETECTION_COOLDOWN:
            try:
                # Real deteksi plat dari kamera (BUKAN simulasi lagi!)
                detected_plate = real_plate_detection(frame)

                if detected_plate:
                    with detection_lock:
                        latest_detection = {
                            'plate_text': detected_plate['text'],
                            'confidence': detected_plate['confidence'],
                            'timestamp': datetime.now().isoformat(),
                            'bbox': detected_plate.get('bbox', [0, 0, 100, 50])
                        }

                    # Process access control
                    process_vehicle_access(detected_plate['text'], detected_plate['confidence'])
                    last_detection_time = current_time
                    system_status['last_detection_time'] = datetime.now()

            except Exception as e:
                logger.error(f"❌ Error detection: {e}")

        # Draw detection info pada frame
        draw_detection_info(frame)

        # Encode frame untuk streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.1)  # Small delay untuk CPU

def real_plate_detection(frame):
    """
    Penjelasan SMK: Deteksi plat ASLI + BACA TEKS dengan OCR
    Mendukung YOLO dan Contour-based detection
    """
    global all_detected_bboxes

    try:
        # Step 1: Deteksi SEMUA area plat
        if USE_YOLO:
            # YOLO detection - returns list of (x,y,w,h) tuples
            bboxes = plate_detector.detect(frame)
        else:
            # Contour-based detection
            bboxes = plate_detector.detect_plate_region(frame)

        # Simpan ke global variable untuk ditampilkan
        all_detected_bboxes = bboxes if bboxes else []

        if bboxes and len(bboxes) > 0:
            # Ambil plat TERBAIK (yang pertama, sudah sorted by confidence)
            # OPTIMIZED: Plat sudah dirank berdasarkan quality score
            best_bbox = bboxes[0]
            x, y, w, h = best_bbox

            # Validasi tambahan: skip plat yang terlalu kecil (mungkin noise)
            MIN_AREA_FOR_OCR = 1400  # 70px x 20px minimum (1400 pixels)
            plate_area = w * h

            if plate_area < MIN_AREA_FOR_OCR:
                logger.warning(f"⚠️ Plate too small for OCR: {w}x{h} ({plate_area} pixels)")
                return None

            # Step 2: Crop area plat
            roi = frame[y:y+h, x:x+w]

            # Save cropped plate
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('gambarplat', exist_ok=True)

            # Step 3: Analisis kendaraan (warna + tipe)
            vehicle_info = vehicle_analyzer.analyze_vehicle(roi, frame, best_bbox)
            vehicle_color = vehicle_info['color']
            vehicle_type = vehicle_info['type']

            logger.info(f"🚗 Vehicle detected: {vehicle_type}, Color: {vehicle_color}, Size: {w}x{h}")

            # Save original crop
            debug_path = f"gambarplat/crop_{timestamp}.jpg"
            cv2.imwrite(debug_path, roi)
            logger.info(f"💾 Cropped plate saved: {debug_path} (size: {w}x{h})")

            # Step 4: OCR - BACA TEKS dari plat!
            plate_text, ocr_confidence = ocr_processor.read_plate_with_confidence(roi)

            # Log hasil OCR
            if plate_text:
                logger.info(f"✅ OCR SUCCESS: {plate_text} (confidence: {ocr_confidence:.2f})")

                # Save SUCCESS result dengan annotasi + metadata
                try:
                    annotated = roi.copy()
                    # Tambah text hasil OCR di gambar
                    cv2.putText(annotated, plate_text, (5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Filename dengan metadata: SUCCESS_PLAT_TYPE_COLOR_TIMESTAMP.jpg
                    success_path = f"gambarplat/SUCCESS_{plate_text}_{vehicle_type}_{vehicle_color}_{timestamp}.jpg"
                    cv2.imwrite(success_path, annotated)
                    logger.info(f"💾 Success saved: {success_path}")
                except Exception as e:
                    logger.error(f"Error saving success image: {e}")

                return {
                    'text': plate_text,
                    'confidence': ocr_confidence,
                    'bbox': [x, y, w, h]
                }
            else:
                logger.warning(f"❌ OCR FAILED - Size: {w}x{h}, Confidence: {ocr_confidence:.2f}")

                # Coba OCR simple tanpa validasi ketat
                try:
                    simple_text = ocr_processor.read_plate_text(roi)
                    if simple_text:
                        logger.info(f"⚠️ Simple OCR: {simple_text}")
                        return {
                            'text': simple_text,
                            'confidence': 0.5,
                            'bbox': [x, y, w, h]
                        }
                except:
                    pass

                return {
                    'text': 'UNKNOWN',
                    'confidence': 0.3,
                    'bbox': [x, y, w, h]
                }
        else:
            return None

    except Exception as e:
        logger.error(f"❌ Error in real plate detection: {e}")
        all_detected_bboxes = []
        return None

def draw_detection_info(frame):
    """
    Penjelasan SMK: Gambar info di video seperti subtitle
    Tampilkan SEMUA plat yang terdeteksi dengan kotak warna berbeda
    """
    global latest_detection, system_status, all_detected_bboxes

    # Status kamera
    status_color = (0, 255, 0) if system_status['camera_connected'] else (0, 0, 255)
    cv2.putText(frame, "CAMERA: CONNECTED" if system_status['camera_connected'] else "CAMERA: DISCONNECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Gate status
    gate_color = (0, 255, 0) if system_status['gate_status'] == 'opened' else (0, 0, 255)
    gate_text = f"GATE: {system_status['gate_status'].upper()}"
    cv2.putText(frame, gate_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gate_color, 2)

    # Latest detection (plat yang sedang diproses)
    if latest_detection:
        detection_text = f"PROCESSING: {latest_detection['plate_text']} ({latest_detection['confidence']:.0%})"
        cv2.putText(frame, detection_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Draw SEMUA bounding boxes yang terdeteksi
    # Penjelasan SMK: Pakai fungsi draw_detections dari PlateDetector
    # Otomatis gambar semua kotak dengan warna berbeda!
    if all_detected_bboxes:
        if USE_YOLO:
            # YOLO drawer - simple green boxes
            plate_detector.draw(frame, all_detected_bboxes, "PLAT")
        else:
            # Contour-based drawer - multi-color boxes
            plate_detector.draw_detections(frame, all_detected_bboxes)

def process_vehicle_access(plate_text, confidence):
    """
    Penjelasan SMK: Seperti 'security guard digital'
    Cek database: boleh masuk atau tidak?
    """
    global system_status

    try:
        conn = get_db_connection()
        if not conn:
            return 'error', "Database connection failed"

        cursor = conn.cursor(dictionary=True)  # MySQL: dictionary=True

        # Cek apakah kendaraan terdaftar
        query = "SELECT * FROM kendaraan_terdaftar WHERE nomor_plat = %s AND status = 'aktif'"  # MySQL: %s instead of ?
        cursor.execute(query, (plate_text.replace(' ', ''),))
        vehicle = cursor.fetchone()

        if vehicle:
            # AUTHORIZED - Boleh masuk
            access_status = 'boleh_masuk'
            gate_action = 'opened'
            system_status['gate_status'] = 'opened'

            message = f"🟢 AUTHORIZED - Welcome {vehicle['nama_pemilik']}!"
            logger.info(f"✅ ACCESS GRANTED: {plate_text} - {vehicle['nama_pemilik']}")

            # Simulasi gate buka (print ke console)
            print("\n" + "=" * 50)
            print("🟢 PALANG PINTU TERBUKA")
            print(f"Plat: {plate_text}")
            print(f"Pemilik: {vehicle['nama_pemilik']}")
            print(f"Jenis: {vehicle['jenis_kendaraan']}")
            print(f"Confidence: {confidence:.1%}")
            print("=" * 50 + "\n")

            # Auto close gate setelah 5 detik
            def close_gate():
                system_status['gate_status'] = 'closed'
                logger.info("🔴 Gate automatically closed after 5 seconds")

            threading.Timer(5.0, close_gate).start()

        else:
            # DENIED - Tidak boleh masuk
            access_status = 'ditolak'
            gate_action = 'closed'
            system_status['gate_status'] = 'closed'

            message = f"🔴 ACCESS DENIED - Plate {plate_text} not registered!"
            logger.warning(f"❌ ACCESS DENIED: {plate_text}")

            print("\n" + "=" * 50)
            print("🔴 AKSES DITOLAK")
            print(f"Plat: {plate_text}")
            print("Alasan: Tidak terdaftar dalam database")
            print(f"Confidence: {confidence:.1%}")
            print("=" * 50 + "\n")

        # Simpan ke access_logs
        save_access_log(plate_text, confidence, access_status, gate_action, message)

        cursor.close()
        conn.close()

        return access_status, message

    except Exception as e:
        logger.error(f"❌ Error processing access: {e}")
        return 'error', f"System error: {e}"

def save_access_log(plate_text, confidence, access_status, gate_action, notes):
    """
    Penjelasan SMK: Seperti 'tulis di buku tamu'
    Catat semua aktivitas ke database
    """
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("❌ Cannot save log: Database connection failed")
            return

        cursor = conn.cursor()

        # Simpan foto (simulasi path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = f"access_photos/{timestamp}_{plate_text}.jpg"

        query = """
        INSERT INTO log_akses_masuk (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, path_foto, catatan)
        VALUES (%s, %s, %s, %s, %s, %s)
        """  # MySQL: %s instead of ?
        cursor.execute(query, (plate_text, confidence, access_status, gate_action, photo_path, notes))
        conn.commit()

        logger.info(f"📝 Log saved: {plate_text} - {access_status}")

        cursor.close()
        conn.close()

    except MySQLError as e:
        logger.error(f"❌ MySQL Error saving log: {e}")
    except Exception as e:
        logger.error(f"❌ Error saving log: {e}")

# =====================================================
# FLASK ROUTES
# =====================================================

@app.route('/')
def index():
    """
    Penjelasan SMK: Halaman utama website
    Seperti 'homepage' yang user pertama kali lihat
    """
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor(dictionary=True)  # MySQL: pakai dictionary=True

        # Get recent access logs
        cursor.execute("""
            SELECT al.*, v.nama_pemilik, v.jenis_kendaraan
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            ORDER BY al.waktu_deteksi DESC
            LIMIT 10
        """)
        recent_logs = cursor.fetchall()

        # Get statistics
        cursor.execute("SELECT COUNT(*) as total FROM kendaraan_terdaftar WHERE status = 'aktif'")
        vehicle_count = cursor.fetchone()['total']

        cursor.execute("""
            SELECT COUNT(*) as today_access
            FROM log_akses_masuk
            WHERE DATE(waktu_deteksi) = CURDATE()
        """)  # MySQL: CURDATE() instead of DATE('now')
        today_access = cursor.fetchone()['today_access']

        cursor.close()
        conn.close()

        return render_template('index.html',
                             recent_logs=recent_logs,
                             vehicle_count=vehicle_count,
                             today_access=today_access,
                             system_status=system_status)
    except Exception as e:
        logger.error(f"❌ Error loading index: {e}")
        return render_template('index.html',
                             recent_logs=[],
                             vehicle_count=0,
                             today_access=0,
                             system_status=system_status)

# =====================================================
# API ROUTES (untuk komunikasi JavaScript)
# =====================================================

@app.route('/video_feed')
def video_feed():
    """
    Penjelasan SMK: Endpoint untuk streaming video
    Seperti 'saluran TV' yang terus kirim gambar ke browser
    """
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/latest_detection')
def api_latest_detection():
    """
    Penjelasan SMK: API untuk ambil data deteksi terbaru
    Seperti 'WhatsApp status' yang bisa dicek kapan saja
    """
    global latest_detection
    with detection_lock:
        if latest_detection:
            return jsonify({
                'status': 'detected',
                'plate_text': latest_detection['plate_text'],
                'confidence': latest_detection['confidence'],
                'timestamp': latest_detection['timestamp'],
                'system_status': system_status
            })
        else:
            return jsonify({
                'status': 'no_detection',
                'system_status': system_status
            })

@app.route('/api/screenshot')
def api_screenshot():
    """
    Penjelasan SMK: API untuk ambil screenshot kamera
    Seperti tombol 'Print Screen' di keyboard
    """
    global camera

    if not camera or not camera.isOpened():
        return jsonify({'status': 'error', 'message': 'Camera not available'})

    try:
        ret, frame = camera.read()
        if ret:
            # Buat folder jika belum ada
            os.makedirs('static/screenshots', exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            filepath = os.path.join('static/screenshots', filename)

            cv2.imwrite(filepath, frame)
            logger.info(f"📸 Screenshot saved: {filename}")

            return jsonify({
                'status': 'success',
                'filename': filename,
                'path': filepath,
                'timestamp': timestamp
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to capture frame'})

    except Exception as e:
        logger.error(f"❌ Screenshot error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/manual_override')
def api_manual_override():
    """
    Penjelasan SMK: API untuk buka gate secara manual
    Seperti tombol 'Emergency' untuk security guard
    """
    global system_status

    try:
        system_status['gate_status'] = 'opened'

        message = "🟡 MANUAL OVERRIDE - Gate opened by security"
        logger.info("🟡 Manual override activated")

        # Print ke console
        print("\n" + "=" * 50)
        print("🟡 MANUAL OVERRIDE")
        print("Security guard membuka gate secara manual")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 50 + "\n")

        # Save log
        save_access_log('MANUAL', 1.0, 'manual_override', 'manual', message)

        # Auto close setelah 10 detik
        def close_gate_manual():
            system_status['gate_status'] = 'closed'
            logger.info("🔴 Gate automatically closed after 10 seconds (manual override)")

        threading.Timer(10.0, close_gate_manual).start()

        return jsonify({
            'status': 'success',
            'message': message,
            'gate_status': 'opened'
        })

    except Exception as e:
        logger.error(f"❌ Manual override error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/system_status')
def api_system_status():
    """
    Penjelasan SMK: API untuk cek status sistem
    Seperti 'dashboard mobil' yang tunjukkan semua indikator
    """
    return jsonify({
        'camera_connected': system_status['camera_connected'],
        'gate_status': system_status['gate_status'],
        'detection_active': system_status['detection_active'],
        'last_detection_time': system_status['last_detection_time'].isoformat() if system_status['last_detection_time'] else None,
        'latest_detection': latest_detection
    })

@app.route('/api/detected_plates')
def api_detected_plates():
    """
    Penjelasan SMK: API untuk list gambar plat yang berhasil dideteksi
    Return: list gambar dengan detail (plat, tipe, warna, waktu)
    """
    try:
        import glob
        from pathlib import Path

        # Cari semua file SUCCESS
        success_files = glob.glob('gambarplat/SUCCESS_*.jpg')
        success_files.sort(key=os.path.getmtime, reverse=True)  # Terbaru dulu

        detected_plates = []

        for filepath in success_files[:50]:  # Max 50 terbaru
            filename = os.path.basename(filepath)

            # Parse filename: SUCCESS_PLAT_TYPE_COLOR_YYYYMMDD_HHMMSS.jpg
            # Example: SUCCESS_B1234ABC_Mobil_Hitam_20250108_153045.jpg
            parts = filename.replace('SUCCESS_', '').replace('.jpg', '').split('_')

            # Default values
            plate_text = "Unknown"
            vehicle_type = "Unknown"
            vehicle_color = "Unknown"
            waktu_deteksi = "Unknown"

            try:
                if len(parts) >= 5:
                    # New format with vehicle info
                    plate_text = parts[0]
                    vehicle_type = parts[1]
                    vehicle_color = parts[2]
                    date_str = parts[3]
                    time_str = parts[4]

                    # Format waktu
                    timestamp_str = f"{date_str}_{time_str}"
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    waktu_deteksi = dt.strftime("%d/%m/%Y %H:%M:%S")

                elif len(parts) >= 3:
                    # Old format without vehicle info
                    plate_text = parts[0]
                    date_str = parts[1]
                    time_str = parts[2]

                    # Format waktu
                    timestamp_str = f"{date_str}_{time_str}"
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    waktu_deteksi = dt.strftime("%d/%m/%Y %H:%M:%S")

                    # Cek dari database untuk backward compatibility
                    try:
                        conn = get_db_connection()
                        if conn:
                            cursor = conn.cursor(dictionary=True)  # MySQL: dictionary=True
                            cursor.execute("SELECT jenis_kendaraan FROM kendaraan_terdaftar WHERE nomor_plat = %s", (plate_text.replace(' ', ''),))  # MySQL: %s, remove spaces
                            vehicle = cursor.fetchone()
                            conn.close()

                            if vehicle:
                                vehicle_type = vehicle['jenis_kendaraan'].capitalize()
                    except:
                        pass

            except Exception as parse_error:
                logger.warning(f"Error parsing filename {filename}: {parse_error}")

            # Cek pemilik dari database
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor(dictionary=True)  # MySQL: dictionary=True
                    cursor.execute("SELECT nama_pemilik FROM kendaraan_terdaftar WHERE nomor_plat = %s", (plate_text.replace(' ', ''),))  # MySQL: %s, remove spaces
                    vehicle = cursor.fetchone()
                    conn.close()

                    nama_pemilik = vehicle['nama_pemilik'] if vehicle else "Tidak Terdaftar"
                else:
                    nama_pemilik = "Unknown"
            except:
                nama_pemilik = "Unknown"

            # File size
            file_size = os.path.getsize(filepath)

            detected_plates.append({
                'image_path': filepath,
                'plate_text': plate_text,
                'vehicle_type': vehicle_type,
                'vehicle_color': vehicle_color,
                'waktu_deteksi': waktu_deteksi,
                'nama_pemilik': nama_pemilik,
                'file_size': file_size,
                'filename': filename
            })

        return jsonify({
            'status': 'success',
            'total': len(detected_plates),
            'plates': detected_plates
        })

    except Exception as e:
        logger.error(f"Error getting detected plates: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'plates': []
        })

# =====================================================
# WEB PAGES ROUTES
# =====================================================

@app.route('/vehicles')
def vehicles():
    """
    Penjelasan SMK: Halaman daftar kendaraan
    Seperti 'phonebook' yang tunjukkan semua kontak
    """
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor(dictionary=True)  # MySQL: dictionary=True
        cursor.execute("SELECT * FROM kendaraan_terdaftar ORDER BY tanggal_daftar DESC")
        vehicles_list = cursor.fetchall()
        cursor.close()
        conn.close()

        return render_template('vehicles.html', vehicles=vehicles_list)
    except Exception as e:
        logger.error(f"❌ Error loading vehicles: {e}")
        flash(f'Error loading vehicles: {e}')
        return redirect(url_for('index'))

@app.route('/access_logs')
def access_logs():
    """
    Penjelasan SMK: Halaman log akses
    Seperti 'history browser' yang tunjukkan semua aktivitas
    """
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor(dictionary=True)  # MySQL: dictionary=True

        # Get filter dari URL parameter
        date_filter = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        status_filter = request.args.get('status', 'all')

        # Build query dengan filter
        base_query = """
            SELECT al.*, v.nama_pemilik, v.jenis_kendaraan
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            WHERE DATE(al.waktu_deteksi) = %s
        """  # MySQL: %s instead of ?
        params = [date_filter]

        if status_filter != 'all':
            base_query += " AND al.status_akses = %s"  # MySQL: %s
            params.append(status_filter)

        base_query += " ORDER BY al.waktu_deteksi DESC LIMIT 100"

        cursor.execute(base_query, params)
        logs = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template('access_logs.html',
                             logs=logs,
                             date_filter=date_filter,
                             status_filter=status_filter)
    except Exception as e:
        logger.error(f"❌ Error loading access logs: {e}")
        flash(f'Error loading access logs: {e}')
        return redirect(url_for('index'))

@app.route('/export_access_logs_csv')
def export_access_logs_csv():
    """
    Penjelasan SMK: Export log akses ke file CSV
    Download data dalam format Excel-friendly
    """
    try:
        import csv
        from io import StringIO
        from flask import make_response

        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor(dictionary=True)

        # Get filter dari URL parameter (sama seperti access_logs)
        date_filter = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        status_filter = request.args.get('status', 'all')

        # Build query dengan filter
        base_query = """
            SELECT
                al.id_log,
                al.plat_terdeteksi,
                al.tingkat_yakin,
                al.status_akses,
                al.aksi_palang,
                al.waktu_deteksi,
                al.catatan,
                v.nama_pemilik,
                v.jenis_kendaraan,
                v.nomor_hp
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            WHERE DATE(al.waktu_deteksi) = %s
        """
        params = [date_filter]

        if status_filter != 'all':
            base_query += " AND al.status_akses = %s"
            params.append(status_filter)

        base_query += " ORDER BY al.waktu_deteksi DESC"

        cursor.execute(base_query, params)
        logs = cursor.fetchall()

        cursor.close()
        conn.close()

        # Create CSV in memory
        si = StringIO()
        writer = csv.writer(si)

        # Header CSV
        writer.writerow([
            'ID', 'Plat Nomor', 'Nama Pemilik', 'Jenis Kendaraan',
            'Status Akses', 'Aksi Palang', 'Confidence', 'Waktu Deteksi',
            'Nomor HP', 'Catatan'
        ])

        # Data rows
        for log in logs:
            writer.writerow([
                log['id_log'],
                log['plat_terdeteksi'],
                log['nama_pemilik'] or 'Tidak Dikenal',
                log['jenis_kendaraan'] or '-',
                log['status_akses'],
                log['aksi_palang'],
                f"{log['tingkat_yakin']:.2f}" if log['tingkat_yakin'] else '0.00',
                log['waktu_deteksi'].strftime('%Y-%m-%d %H:%M:%S'),
                log['nomor_hp'] or '-',
                log['catatan'] or '-'
            ])

        # Create response
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = f"attachment; filename=log_akses_{date_filter}.csv"
        output.headers["Content-type"] = "text/csv"

        logger.info(f"✅ CSV exported: {len(logs)} records for {date_filter}")

        return output

    except Exception as e:
        logger.error(f"❌ Error exporting CSV: {e}")
        flash(f'Error exporting CSV: {e}')
        return redirect(url_for('access_logs'))

@app.route('/detected_plates')
def detected_plates():
    """
    Penjelasan SMK: Halaman gallery plat terdeteksi
    Tampilkan semua plat yang berhasil dibaca OCR dengan detail
    """
    return render_template('detected_plates.html')

# =====================================================
# CRUD OPERATIONS (Create, Read, Update, Delete)
# =====================================================

def normalize_phone(phone):
    """
    Penjelasan SMK: Normalisasi nomor HP ke format konsisten
    Input: '0812-3456-789', '+62-812-3456-789', '(0812) 3456 789'
    Output: '08123456789'
    """
    if not phone:
        return ''

    # Remove semua karakter non-digit
    phone = ''.join(filter(str.isdigit, phone))

    # Convert +62 ke 0
    if phone.startswith('62'):
        phone = '0' + phone[2:]

    # Pastikan dimulai dengan 08
    if phone and not phone.startswith('08'):
        return ''  # Invalid phone

    return phone

@app.route('/add_vehicle', methods=['POST'])
def add_vehicle():
    """
    Penjelasan SMK: Tambah kendaraan baru ke database
    Seperti 'Add Contact' di phonebook
    """
    try:
        nomor_plat = request.form['nomor_plat'].replace(' ', '').upper()
        nama_pemilik = request.form['nama_pemilik'].strip()
        jenis_kendaraan = request.form['jenis_kendaraan']
        nomor_hp = normalize_phone(request.form.get('nomor_hp', ''))

        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor()

        query = """
        INSERT INTO kendaraan_terdaftar (nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp, status)
        VALUES (%s, %s, %s, %s, 'aktif')
        """  # MySQL: %s instead of ?
        cursor.execute(query, (nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp))
        conn.commit()

        cursor.close()
        conn.close()

        flash(f'✅ Kendaraan {nomor_plat} berhasil ditambahkan!')
        logger.info(f"✅ Vehicle added: {nomor_plat} - {nama_pemilik}")

    except mysql.connector.IntegrityError as e:  # MySQL: IntegrityError
        # Check if duplicate key error
        if 'Duplicate entry' in str(e):
            flash(f'❌ Plat nomor {nomor_plat} sudah terdaftar!')
            logger.warning(f"⚠️  Duplicate plate attempt: {nomor_plat}")
        else:
            flash(f'❌ Error database: {e}')
            logger.error(f"❌ Database error: {e}")
    except Exception as e:
        flash(f'❌ Error menambah kendaraan: {e}')
        logger.error(f"❌ Error adding vehicle: {e}")

    return redirect(url_for('vehicles'))

@app.route('/edit_vehicle/<int:vehicle_id>', methods=['POST'])
def edit_vehicle(vehicle_id):
    """
    Penjelasan SMK: Edit data kendaraan yang sudah ada
    Seperti 'Edit Contact' di phonebook
    """
    try:
        nama_pemilik = request.form['nama_pemilik'].strip()
        jenis_kendaraan = request.form['jenis_kendaraan']
        nomor_hp = normalize_phone(request.form.get('nomor_hp', ''))
        status = request.form['status']

        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor()

        query = """
        UPDATE kendaraan_terdaftar
        SET nama_pemilik = %s, jenis_kendaraan = %s, nomor_hp = %s, status = %s
        WHERE id_kendaraan = %s
        """  # MySQL: %s instead of ?
        cursor.execute(query, (nama_pemilik, jenis_kendaraan, nomor_hp, status, vehicle_id))
        conn.commit()

        cursor.close()
        conn.close()

        flash('✅ Data kendaraan berhasil diupdate!')
        logger.info(f"✅ Vehicle updated: ID {vehicle_id}")

    except Exception as e:
        flash(f'❌ Error update kendaraan: {e}')
        logger.error(f"❌ Error updating vehicle: {e}")

    return redirect(url_for('vehicles'))

@app.route('/delete_vehicle/<int:vehicle_id>')
def delete_vehicle(vehicle_id):
    """
    Penjelasan SMK: Hapus kendaraan dari database
    Seperti 'Delete Contact' di phonebook
    """
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Database connection failed")

        cursor = conn.cursor(dictionary=True)  # MySQL: dictionary=True

        # Get vehicle info dulu untuk log
        cursor.execute("SELECT nomor_plat, nama_pemilik FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))  # MySQL: %s
        vehicle = cursor.fetchone()

        if vehicle:
            cursor.execute("DELETE FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))  # MySQL: %s
            conn.commit()

            flash(f'✅ Kendaraan {vehicle["nomor_plat"]} ({vehicle["nama_pemilik"]}) berhasil dihapus!')
            logger.info(f"✅ Vehicle deleted: {vehicle['nomor_plat']} - {vehicle['nama_pemilik']}")
        else:
            flash('❌ Kendaraan tidak ditemukan!')

        cursor.close()
        conn.close()

    except Exception as e:
        flash(f'❌ Error hapus kendaraan: {e}')
        logger.error(f"❌ Error deleting vehicle: {e}")

    return redirect(url_for('vehicles'))

# =====================================================
# SYSTEM INITIALIZATION & MAIN
# =====================================================

def create_required_folders():
    """
    Penjelasan SMK: Buat folder-folder yang dibutuhkan sistem
    Seperti 'setup workspace' sebelum mulai kerja
    """
    folders = [
        'static/screenshots',
        'access_photos',
        'logs'
    ]

    for folder in folders:
        try:
            os.makedirs(folder, exist_ok=True)
            logger.info(f"📁 Folder created/verified: {folder}")
        except Exception as e:
            logger.error(f"❌ Error creating folder {folder}: {e}")

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting Vehicle Access Control System...")

        # Create required folders
        create_required_folders()

        # Initialize camera
        logger.info("🎥 Initializing camera...")
        if initialize_camera():
            logger.info("✅ Camera initialized successfully")
        else:
            logger.warning("⚠️ Camera initialization failed - running without camera")

        # Start Flask app
        logger.info(f"🌐 Starting web server on http://localhost:{config.FLASK_PORT}")
        app.run(debug=True, host='0.0.0.0', port=config.FLASK_PORT, threaded=True)

    except Exception as e:
        logger.error(f"❌ Fatal error starting system: {e}")
    finally:
        # Cleanup
        if camera:
            camera.release()
        logger.info("👋 System shutdown complete")