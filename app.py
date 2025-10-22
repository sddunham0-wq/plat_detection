from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import pymysql
from pymysql import Error as MySQLError
from dbutils.pooled_db import PooledDB
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
all_vehicle_bboxes = []  # ★ NEW: Simpan bounding boxes MOBIL (kotak besar)
all_detected_plates = []  # ★ NEW: Store detected plates dengan text labels (bbox + text + confidence)
detection_lock = threading.Lock()
bboxes_lock = threading.Lock()  # ★ BUG FIX #1: Thread-safe protection untuk all_detected_bboxes

# ★ REAL-TIME NOTIFICATION SYSTEM
# Penjelasan SMK: Simpan notifikasi terbaru untuk ditampilkan di beranda
latest_notification = {
    'status': None,  # 'authorized' atau 'denied'
    'plate_text': None,
    'owner_name': None,
    'vehicle_type': None,
    'timestamp': None,
    'message': None
}
notification_lock = threading.Lock()

system_status = {
    'camera_connected': False,
    'detection_active': False,
    'gate_status': 'closed',
    'last_detection_time': None
}

# ★ BOUNDING BOX STABILIZATION SYSTEM
# Penjelasan SMK: Sistem untuk bikin kotak stabil, tidak kedip-kedip
# Tracking history untuk smooth bounding boxes
from collections import deque
vehicle_tracking_history = deque(maxlen=5)  # Simpan 5 frame terakhir untuk smoothing
plate_tracking_history = deque(maxlen=7)    # Plate tracking dengan smoothing lebih baik

# ★ 2-MODEL DUAL DETECTION SYSTEM
# Penjelasan SMK: Gunakan 2 model untuk deteksi lengkap
# Model 1: YOLOv8n → Detect MOBIL (kotak hijau besar)
# Model 2: best.pt custom → Detect PLAT (kotak hijau kecil)

# Initialize VEHICLE detector (YOLOv8n - general object detection)
vehicle_detector = None
if USE_YOLO:
    try:
        from ultralytics import YOLO
        vehicle_detector = YOLO('yolov8n.pt')  # General model untuk detect mobil
        logger.info("✅ Vehicle Detector (YOLOv8n) initialized - for VEHICLE detection")
    except Exception as e:
        logger.warning(f"⚠️  Vehicle detector init failed: {e}")
        vehicle_detector = None

# Initialize LICENSE PLATE detector (best.pt - custom trained model)
plate_detector = None
if USE_YOLO:
    try:
        plate_detector = YOLOPlateDetector(
            model_path='models/best.pt',
            conf_threshold=0.15  # OPTIMIZED: Turun ke 0.15 untuk lebih sensitif
        )
        logger.info("✅ Plate Detector (best.pt) initialized - for PLATE detection")
    except Exception as e:
        logger.error(f"❌ YOLO plate detector initialization failed: {e}")
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

# ★ DATABASE CONNECTION - SIMPLIFIED FOR "TOO MANY CONNECTIONS" FIX
# Penjelasan SMK: Pakai direct connection (bukan pool) untuk avoid MySQL limit
# Strategy: Buka-tutup connection setiap kali pakai (lebih lambat tapi aman)

def get_db_connection():
    """
    Get direct MySQL connection with retry mechanism

    Penjelasan SMK: Buka koneksi baru setiap kali, lalu TUTUP setelah pakai
    Ini lebih lambat tapi tidak akan kena "Too many connections"

    IMPORTANT: Setelah pakai connection, HARUS tutup dengan conn.close()!
    """
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            conn = pymysql.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False,
                connect_timeout=10,  # Timeout after 10 seconds
                read_timeout=30,     # Read timeout 30 seconds
                write_timeout=30     # Write timeout 30 seconds
            )

            if attempt > 0:
                logger.info(f"✅ Database connected (after {attempt + 1} attempts)")

            return conn

        except pymysql.err.OperationalError as e:
            error_code = e.args[0] if e.args else 0

            if error_code == 1040:  # Too many connections
                logger.warning(f"⚠️  Attempt {attempt + 1}/{max_retries}: Too many connections, retrying in {retry_delay}s...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("❌ Failed to connect after all retries: Too many connections")
                    logger.error("   💡 SOLUSI: Restart MySQL dengan 'brew services restart mysql'")
                    return None
            else:
                logger.error(f"❌ MySQL connection error: {e}")
                return None

        except Exception as e:
            logger.error(f"❌ Unexpected database error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None

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

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IOU) antara 2 bounding boxes

    Penjelasan SMK: Ngukur seberapa "overlap" 2 kotak
    IOU = 1.0 → kotak sama persis
    IOU = 0.0 → tidak overlap sama sekali

    Args:
        box1, box2: (x, y, w, h) format
    Returns:
        IOU score (0.0 - 1.0)
    """
    x1, y1, w1, h1 = box1[:4]  # Support both (x,y,w,h) and (x,y,w,h,class,conf)
    x2, y2, w2, h2 = box2[:4]

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def non_maximum_suppression(bboxes, iou_threshold=0.5):
    """
    Non-Maximum Suppression untuk filter overlapping bounding boxes

    Penjelasan SMK: Seperti "pilih satu kotak terbaik dari yang overlap"
    Kalau ada 3 kotak hijau menimpa di satu plat → ambil 1 yang terbesar/terbaik

    Algorithm:
    1. Sort boxes by area (largest first)
    2. Loop each box:
       - Keep box jika tidak overlap dengan box yang sudah dipilih
       - Skip box jika overlap tinggi (IOU > threshold)

    Args:
        bboxes: List of (x,y,w,h) or (x,y,w,h,class,conf)
        iou_threshold: Maximum IOU untuk consider sebagai overlap (0.5 = 50% overlap)

    Returns:
        Filtered list of non-overlapping bboxes
    """
    if not bboxes or len(bboxes) == 0:
        return []

    # If hanya 1 box, langsung return
    if len(bboxes) == 1:
        return bboxes

    # Calculate area untuk setiap box
    boxes_with_area = []
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        area = w * h
        boxes_with_area.append({
            'bbox': bbox,
            'area': area
        })

    # Sort by area (largest first) → prioritize larger detections
    boxes_with_area.sort(key=lambda b: b['area'], reverse=True)

    # NMS algorithm
    keep = []

    while boxes_with_area:
        # Ambil box dengan area terbesar (index 0)
        current = boxes_with_area.pop(0)
        keep.append(current['bbox'])

        # Filter remaining boxes: remove yang overlap tinggi dengan current
        remaining = []
        for other in boxes_with_area:
            iou = calculate_iou(current['bbox'], other['bbox'])

            # Keep box jika IOU rendah (tidak overlap banyak)
            if iou < iou_threshold:
                remaining.append(other)
            else:
                logger.debug(f"NMS: Suppressed box (IOU={iou:.2f} with kept box)")

        boxes_with_area = remaining

    logger.info(f"✨ NMS: {len(bboxes)} boxes → {len(keep)} non-overlapping boxes (threshold={iou_threshold})")

    return keep

def smooth_bounding_boxes(current_detections, history, iou_threshold=0.5):
    """
    Smooth bounding boxes using temporal filtering

    Penjelasan SMK: Bikin kotak stabil dengan rata-rata beberapa frame terakhir

    Args:
        current_detections: List of current frame detections
        history: deque of previous frames' detections
        iou_threshold: Minimum IOU untuk consider sebagai "same object"

    Returns:
        Smoothed bounding boxes
    """
    if not current_detections:
        return []

    # Add current to history
    history.append(current_detections)

    # Need at least 2 frames untuk smoothing
    if len(history) < 2:
        return current_detections

    smoothed = []

    for curr_det in current_detections:
        # Find matching detections in history (same object across frames)
        matching_history = []

        for hist_frame in list(history)[:-1]:  # Exclude current frame
            for hist_det in hist_frame:
                iou = calculate_iou(curr_det, hist_det)
                if iou > iou_threshold:
                    matching_history.append(hist_det)
                    break  # Only one match per frame

        # If found matches in history, average the positions with WEIGHTED AVERAGE
        if matching_history:
            all_boxes = matching_history + [curr_det]

            # ★ IMPROVED: Balanced Weighted Average
            # Frame terbaru dan history balance (0.5/0.5)
            # Lebih smooth dan stabil untuk tracking plat
            n_history = len(matching_history)

            # Weight untuk frame current (50% importance)
            current_weight = 0.5
            # Sisa weight (50%) dibagi ke semua history frames
            history_weight = 0.5 / n_history if n_history > 0 else 0

            # Weighted average x, y, w, h
            avg_x = int(curr_det[0] * current_weight + sum(b[0] * history_weight for b in matching_history))
            avg_y = int(curr_det[1] * current_weight + sum(b[1] * history_weight for b in matching_history))
            avg_w = int(curr_det[2] * current_weight + sum(b[2] * history_weight for b in matching_history))
            avg_h = int(curr_det[3] * current_weight + sum(b[3] * history_weight for b in matching_history))

            # Keep class and confidence from current detection if present
            if len(curr_det) > 4:
                smoothed.append((avg_x, avg_y, avg_w, avg_h, curr_det[4], curr_det[5]))
            else:
                smoothed.append((avg_x, avg_y, avg_w, avg_h))
        else:
            # New detection, use as is
            smoothed.append(curr_det)

    return smoothed

def multi_scale_detection(frame):
    """
    LEVEL 3 OPTIMIZATION: Multi-scale detection untuk plat jarak berbeda

    Penjelasan SMK: Seperti "zoom in zoom out" untuk cari plat.
    Try deteksi di 3 resolusi berbeda:
    - 100% (full res) → untuk plat dekat/besar
    - 70% (scaled down) → untuk medium range
    - 50% (scaled down) → untuk plat jauh

    Returns:
        Best bbox (x,y,w,h) atau None
    """
    global all_detected_bboxes
    all_detections = []

    scales = [
        (1.0, "Full Resolution"),
        (0.7, "70% Scale"),
        (0.5, "50% Scale")
    ]

    for scale, label in scales:
        try:
            if scale < 1.0:
                # Resize frame untuk scale ini
                h, w = frame.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                scaled_frame = frame

            # Deteksi di scale ini
            if USE_YOLO:
                bboxes = plate_detector.detect(scaled_frame)
            else:
                bboxes = plate_detector.detect_plate_region(scaled_frame)

            if bboxes:
                for bbox in bboxes:
                    x, y, w, h = bbox

                    # Scale back coordinates ke original size ★ BUG FIX!
                    # Kalau detect di 50% (scale=0.5), coordinate harus dikali 2!
                    if scale < 1.0:
                        scale_factor = 1.0 / scale  # 0.5 → 2.0x, 0.7 → 1.43x
                        x = int(x * scale_factor)
                        y = int(y * scale_factor)
                        w = int(w * scale_factor)
                        h = int(h * scale_factor)

                    # ★ BUG FIX #2: Validate scaled coordinates masih dalam bounds
                    frame_h, frame_w = frame.shape[:2]
                    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
                        logger.debug(f"Skipping OOB bbox at {label}: ({x},{y},{w},{h}) for frame {frame_w}x{frame_h}")
                        continue

                    # ★ BUG FIX #3: Validate dimensions tidak negative atau zero
                    if w <= 0 or h <= 0:
                        logger.debug(f"Skipping invalid dimensions at {label}: {w}x{h}")
                        continue

                    area = w * h
                    all_detections.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'scale': scale,
                        'label': label
                    })

                    logger.debug(f"Detection at {label}: bbox=({x},{y},{w},{h}), area={area}")

        except Exception as e:
            logger.debug(f"Error at scale {scale}: {e}")
            continue

    # Return ALL detections (bukan cuma best!) ★ BUG FIX!
    if all_detections:
        # Sort by area (largest first)
        all_detections.sort(key=lambda d: d['area'], reverse=True)

        # Return SEMUA deteksi (maksimal 5 untuk performa)
        top_detections = all_detections[:5]
        bboxes = [d['bbox'] for d in top_detections]

        logger.info(f"✅ Multi-scale: {len(bboxes)} plate(s) detected across scales")
        for i, d in enumerate(top_detections):
            logger.debug(f"  Plate #{i+1}: {d['label']}, area={d['area']}, bbox={d['bbox']}")

        return bboxes  # ★ Return SEMUA bboxes!

    return []  # Return empty list kalau tidak ada

def real_plate_detection(frame):
    """
    Penjelasan SMK: Deteksi plat ASLI + BACA TEKS dengan OCR

    ★ 2-STAGE DETECTION (seperti image9.png):
    Stage 1: Detect MOBIL (kotak hijau besar)
    Stage 2: Detect PLAT dalam area mobil (kotak hijau kecil)

    Analogi: Security guard lihat mobil dulu, baru zoom ke platnya!
    """
    global all_detected_bboxes, all_vehicle_bboxes

    try:
        # ★ STAGE 1: Detect MOBIL dengan YOLOv8n (kotak BESAR)
        # Penjelasan: YOLOv8n detect mobil/kendaraan/motor dengan kotak besar
        # Hasilnya: Kotak hijau besar kayak di image9.png
        vehicle_bboxes = []

        if USE_YOLO and vehicle_detector is not None:
            # YOLOv8n detect mobil (classes: car, motorcycle, bus, truck, etc.)
            # Classes yang diambil: 2=car, 3=motorcycle, 5=bus, 7=truck
            vehicle_results = vehicle_detector(frame, conf=0.3, verbose=False, device='cpu')

            for result in vehicle_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get class ID and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())

                        # Filter hanya kendaraan (car=2, motorcycle=3 only)
                        if cls in [2, 3]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x = int(x1)
                            y = int(y1)
                            w = int(x2 - x1)
                            h = int(y2 - y1)

                            # Validasi ukuran minimum untuk vehicle
                            if w > 100 and h > 100:
                                # ★ SAVE WITH CLASS ID untuk vehicle type identification
                                # Format: (x, y, w, h, class_id, confidence)
                                vehicle_bboxes.append((x, y, w, h, cls, conf))
                                logger.debug(f"🚗 Vehicle detected: class={cls}, conf={conf:.2f}, size={w}x{h} at ({x},{y})")

        # ★ APPLY SMOOTHING untuk stabilisasi bounding boxes
        # Penjelasan: Kotak tidak kedip-kedip, smooth antar frame
        # IOU threshold 0.3 = lebih toleran untuk object yang bergerak/berubah ukuran dikit
        smoothed_vehicles = smooth_bounding_boxes(vehicle_bboxes, vehicle_tracking_history, iou_threshold=0.3)

        # ★ NEW OPTIMIZATION: Apply NMS untuk vehicle bboxes juga
        # Penjelasan: Filter overlapping vehicle detections (mobil yang sama terdeteksi berkali-kali)
        if smoothed_vehicles and len(smoothed_vehicles) > 1:
            # Vehicle threshold 0.4 (sedikit lebih toleran karena mobil bisa bergerak)
            nms_vehicles = non_maximum_suppression(smoothed_vehicles, iou_threshold=0.4)
        else:
            nms_vehicles = smoothed_vehicles

        # Simpan vehicle bboxes untuk drawing nanti
        with bboxes_lock:
            all_vehicle_bboxes = nms_vehicles

        logger.info(f"✅ Stage 1: {len(nms_vehicles)} vehicle(s) detected (NMS from {len(vehicle_bboxes)} raw)")

        # ★ STAGE 2: Detect PLAT dalam area mobil (kotak KECIL)
        # Penjelasan: Dalam setiap mobil yang terdetect, cari platnya
        # Hasilnya: Kotak hijau kecil di area plat
        bboxes = multi_scale_detection(frame)

        # Fallback ke single-scale jika multi-scale gagal
        if not bboxes or len(bboxes) == 0:
            logger.debug("Multi-scale failed, trying single-scale...")
            if USE_YOLO:
                # YOLO detection - returns list of (x,y,w,h) tuples
                bboxes = plate_detector.detect(frame)
            else:
                # Contour-based detection
                bboxes = plate_detector.detect_plate_region(frame)

        # ★ APPLY SMOOTHING untuk plate detections juga
        # Penjelasan: Kotak plat juga stabil, tidak jitter
        # IOU threshold 0.40 = lebih ketat untuk stabilitas lebih baik
        if bboxes:
            smoothed_plates = smooth_bounding_boxes(bboxes, plate_tracking_history, iou_threshold=0.40)
        else:
            smoothed_plates = []

        # ★ NEW OPTIMIZATION: Apply Non-Maximum Suppression (NMS)
        # Penjelasan SMK: Filter overlapping boxes dari multi-scale detection
        # Hanya keep 1 box terbaik per plat → tidak ada kotak yang menimpa!
        if smoothed_plates and len(smoothed_plates) > 1:
            # NMS dengan IOU threshold 0.3 = 30% overlap (AGGRESSIVE filtering)
            # Kalau 2 box overlap >30% → buang yang lebih kecil
            # Threshold rendah untuk handle multi-scale detection yang overlap tinggi
            nms_plates = non_maximum_suppression(smoothed_plates, iou_threshold=0.3)
        else:
            nms_plates = smoothed_plates

        # ★ BUG FIX #1: Thread-safe update ke global variable
        with bboxes_lock:
            all_detected_bboxes = nms_plates if nms_plates else []

        # ★ NEW: Process ALL detected plates untuk text labels
        # Penjelasan: Loop semua plate yang terdeteksi, coba OCR setiap plat
        # Update all_detected_plates dengan hasil OCR (text + confidence + bbox)
        temp_plates = []
        if nms_plates:
            frame_h, frame_w = frame.shape[:2]
            for idx, bbox in enumerate(nms_plates[:5]):  # Max 5 plates untuk performa
                try:
                    x, y, w, h = bbox
                    # Quick validation
                    if w < 70 or h < 20 or w * h < 2400:
                        continue

                    # Crop dengan margin 15%
                    margin_x = int(w * 0.15)
                    margin_y = int(h * 0.15)
                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(frame_w, x + w + margin_x)
                    y2 = min(frame_h, y + h + margin_y)

                    if x1 >= x2 or y1 >= y2:
                        continue

                    roi = frame[y1:y2, x1:x2]

                    # ★ FIX: Use FULL PLATE for OCR (no cropping)
                    # Try OCR on full plate
                    plate_text, ocr_conf = ocr_processor.read_plate_with_confidence(roi)

                    if plate_text and ocr_processor.is_valid_plate(plate_text) and ocr_conf >= 0.50:
                        temp_plates.append({
                            'text': plate_text,
                            'confidence': ocr_conf,
                            'bbox': [x, y, w, h]
                        })
                        logger.debug(f"✅ Plate {idx+1} OCR: {plate_text} ({ocr_conf:.2f})")
                    else:
                        # Fallback: Add bbox without text untuk labeling "PLAT 1", "PLAT 2"
                        temp_plates.append({
                            'text': '',
                            'confidence': 0,
                            'bbox': [x, y, w, h]
                        })
                except Exception as e:
                    logger.debug(f"⚠️ Error processing plate {idx+1}: {e}")
                    continue

        # Update global all_detected_plates dengan thread safety
        with bboxes_lock:
            all_detected_plates = temp_plates

        if bboxes and len(bboxes) > 0:
            # Ambil plat TERBAIK (yang pertama, sudah sorted by confidence)
            # OPTIMIZED: Plat sudah dirank berdasarkan quality score
            best_bbox = bboxes[0]
            x, y, w, h = best_bbox

            # ★ BUG FIX #5: Improve OCR area threshold dan validation
            # Plat Indonesia standar minimal ~200x60 pixels untuk OCR yang reliable
            # Threshold ditingkatkan untuk OCR lebih akurat
            MIN_AREA_FOR_OCR = 2400  # ~80px x 30px minimum - IMPROVED for better OCR
            MIN_WIDTH = 70  # Minimum width untuk plat yang valid
            MIN_HEIGHT = 20  # Minimum height untuk plat yang valid
            MIN_ASPECT_RATIO = 1.8  # Plat Indonesia biasanya ~3:1 aspect ratio (lowered from 2.0 to allow 1.94)
            MAX_ASPECT_RATIO = 6.0  # Maximum untuk filter noise

            plate_area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Validate area
            if plate_area < MIN_AREA_FOR_OCR:
                logger.warning(f"⚠️ Plate too small for OCR: {w}x{h} ({plate_area} pixels, min={MIN_AREA_FOR_OCR})")
                return None

            # Validate dimensions
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                logger.warning(f"⚠️ Plate dimensions too small: {w}x{h} (min={MIN_WIDTH}x{MIN_HEIGHT})")
                return None

            # Validate aspect ratio
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                logger.warning(f"⚠️ Invalid aspect ratio: {aspect_ratio:.2f} (expected {MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO})")
                return None

            # Step 2: Crop area plat WITH MARGIN
            # ★ OPTIMIZATION: Add 15% margin around plate for better OCR context
            # Penjelasan SMK: Kasih "ruang nafas" di sekitar plat agar OCR lebih akurat
            # Margin membantu OCR detect edge characters dengan lebih baik
            frame_h, frame_w = frame.shape[:2]

            # Calculate margin (15% of bbox size)
            MARGIN_PERCENT = 0.15
            margin_x = int(w * MARGIN_PERCENT)
            margin_y = int(h * MARGIN_PERCENT)

            # Apply margin dengan bounds checking
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame_w, x + w + margin_x)
            y2 = min(frame_h, y + h + margin_y)

            # Validate final coordinates
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"⚠️ Invalid crop coordinates after margin: ({x1},{y1}) to ({x2},{y2})")
                return None

            # Crop with margin
            roi = frame[y1:y2, x1:x2]

            logger.debug(f"🔍 Crop: original=({x},{y},{w},{h}), with_margin=({x1},{y1},{x2-x1},{y2-y1})")

            # Save cropped plate
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('gambarplat', exist_ok=True)

            # Step 3: Analisis kendaraan (warna + tipe)
            vehicle_info = vehicle_analyzer.analyze_vehicle(roi, frame, best_bbox)
            vehicle_color = vehicle_info['color']
            vehicle_type = vehicle_info['type']

            logger.info(f"🚗 Vehicle detected: {vehicle_type}, Color: {vehicle_color}, Size: {w}x{h}")

            # Save original FULL crop (dengan BARIS UTAMA + BARIS TAHUN PAJAK)
            debug_path = f"gambarplat/crop_{timestamp}.jpg"
            cv2.imwrite(debug_path, roi)
            logger.info(f"💾 Full plate saved: {debug_path} (size: {w}x{h})")

            # ★ FIX: DISABLE cropping 65% - gunakan FULL PLATE untuk OCR
            # Alasan: Cropping malah merusak text, full plate lebih reliable
            # OCR sudah cukup pintar untuk ignore baris bawah (tahun pajak)
            # Plus: EasyOCR dengan preprocessing bagus bisa handle full plate

            logger.debug(f"🔍 OCR input: FULL PLATE ({roi.shape[1]}x{roi.shape[0]})")

            # ★ VALIDATION: Check brightness & contrast sebelum OCR
            import numpy as np
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            # Skip jika terlalu gelap atau terlalu terang
            if mean_brightness < 40 or mean_brightness > 220:
                logger.warning(f"⚠️ Poor lighting condition: brightness={mean_brightness:.1f} (valid: 40-220)")
                return None

            # Step 4: OCR - BACA TEKS dari plat (FULL PLATE, bukan crop!)
            plate_text, ocr_confidence = ocr_processor.read_plate_with_confidence(roi)

            # ★ CONFIDENCE THRESHOLD: Filter hasil OCR berdasarkan confidence
            MIN_OCR_CONFIDENCE = 0.50  # BALANCED 0.50 - reject garbage (<0.3) but allow valid plates (0.5-0.9)

            # Log hasil OCR
            if plate_text:
                # ★ BUG FIX #6: Validate plate text sebelum return
                # Pastikan hasil OCR adalah plat yang valid (bukan garbage)
                if ocr_processor.is_valid_plate(plate_text) and ocr_confidence >= MIN_OCR_CONFIDENCE:
                    logger.info(f"✅ OCR SUCCESS: {plate_text} (confidence: {ocr_confidence:.2f})")

                    # Save SUCCESS result dengan annotasi + metadata
                    # ★ IMPORTANT: Simpan FULL PLAT (roi), bukan roi_upper_only!
                    # Untuk dokumentasi lengkap dengan BARIS UTAMA + BARIS TAHUN PAJAK
                    try:
                        annotated = roi.copy()  # ← FULL PLAT (2 baris)
                        # Tambah text hasil OCR di gambar
                        cv2.putText(annotated, plate_text, (5, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Filename dengan metadata: SUCCESS_PLAT_TYPE_COLOR_TIMESTAMP.jpg
                        success_path = f"gambarplat/SUCCESS_{plate_text}_{vehicle_type}_{vehicle_color}_{timestamp}.jpg"
                        cv2.imwrite(success_path, annotated)  # ← FULL PLAT disimpan
                        logger.info(f"💾 Success saved with full plate: {success_path}")
                    except Exception as e:
                        logger.error(f"Error saving success image: {e}")

                    return {
                        'text': plate_text,
                        'confidence': ocr_confidence,
                        'bbox': [x, y, w, h]
                    }
                elif ocr_confidence < MIN_OCR_CONFIDENCE:
                    logger.warning(f"⚠️ OCR confidence too low: {ocr_confidence:.2f} < {MIN_OCR_CONFIDENCE}")
                    # Continue to fallback
                else:
                    logger.warning(f"⚠️ OCR returned invalid plate format: {plate_text}")
                    # Continue to fallback

            # ★ BUG FIX #6: Improved fallback logic
            # Jika OCR utama gagal atau return invalid format, coba fallback
            logger.warning(f"❌ Primary OCR FAILED - Size: {w}x{h}, Confidence: {ocr_confidence:.2f}")

            # Fallback: Try simple OCR tanpa strict validation
            # ★ FIX: Gunakan roi (full plate) untuk fallback juga
            try:
                simple_text = ocr_processor.read_plate_text(roi)

                if simple_text and len(simple_text) >= 3:
                    # Get real format confidence instead of hardcoded 0.5
                    _, format_confidence = ocr_processor.format_indonesian_plate(simple_text)

                    logger.info(f"⚠️ Fallback OCR: {simple_text} (format_conf: {format_confidence:.2f})")

                    # Only return if format confidence is reasonable
                    if format_confidence >= 0.3:
                        return {
                            'text': simple_text,
                            'confidence': format_confidence,
                            'bbox': [x, y, w, h]
                        }
                    else:
                        logger.warning(f"⚠️ Fallback text has low format confidence: {format_confidence:.2f}")

            except Exception as e:
                logger.warning(f"Simple OCR fallback error: {e}")

            # ★ BUG FIX: Return None to prevent database spam with garbage
            logger.warning(f"❌ Complete OCR failure - returning None to skip logging")
            return None
        else:
            return None

    except Exception as e:
        logger.error(f"❌ Error in real plate detection: {e}")
        # ★ BUG FIX #1: Thread-safe clear bboxes on error
        with bboxes_lock:
            all_detected_bboxes = []
        return None

def draw_detection_info(frame):
    """
    Penjelasan SMK: Gambar info di video seperti subtitle

    ★ 2-STAGE DRAWING (seperti image9.png):
    1. Gambar kotak BIRU untuk MOBIL (kotak besar)
    2. Gambar kotak HIJAU untuk PLAT (kotak kecil)

    Analogi: Seperti gambar border di foto - mobil = frame luar, plat = frame dalam
    """
    global latest_detection, system_status, all_detected_bboxes, all_vehicle_bboxes

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

    # ★ DRAW 2-STAGE BOUNDING BOXES (seperti image9.png)
    # Penjelasan: Gambar 2 jenis kotak - MOBIL (biru) dan PLAT (hijau)
    # ★ BUG FIX #8: Thread-safe access + error handling untuk drawing
    try:
        with bboxes_lock:
            frame_h, frame_w = frame.shape[:2]

            # ★ STEP 1: Draw VEHICLE bboxes (kotak BIRU besar untuk mobil)
            if all_vehicle_bboxes:
                BLUE = (255, 0, 0)  # BGR format - Blue untuk mobil

                # ★ VEHICLE TYPE MAPPING
                # Penjelasan: Terjemahan class ID ke bahasa Indonesia
                vehicle_types = {
                    2: "MOBIL",      # car
                    3: "MOTOR",      # motorcycle
                    5: "BUS",        # bus
                    7: "TRUK"        # truck
                }

                for i, bbox in enumerate(all_vehicle_bboxes):
                    # Extract x, y, w, h, dan class_id jika ada
                    if len(bbox) >= 6:
                        x, y, w, h, cls, conf = bbox
                        vehicle_label = vehicle_types.get(cls, "KENDARAAN")
                    elif len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        vehicle_label = "KENDARAAN"
                    else:
                        continue

                    # Validate bbox
                    if x >= 0 and y >= 0 and x + w <= frame_w and y + h <= frame_h and w > 0 and h > 0:
                        # Draw rectangle BIRU (thick untuk mobil)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), BLUE, 3)

                        # ★ Label dengan JENIS KENDARAAN (MOBIL / MOTOR / BUS / TRUK)
                        cv2.putText(frame, vehicle_label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLUE, 2)
                        logger.debug(f"🚗 Drew {vehicle_label} bbox: ({x},{y},{w},{h})")

            # ★ STEP 2: Draw PLATE bboxes WITH TEXT LABELS (kotak HIJAU dengan label)
            # Penjelasan SMK: Gambar kotak hijau + text plat di atasnya
            if all_detected_bboxes:
                GREEN = (0, 255, 0)  # BGR format - Green untuk plat
                YELLOW = (0, 255, 255)  # BGR format - Yellow untuk label text

                for idx, bbox in enumerate(all_detected_bboxes, 1):
                    x, y, w, h = bbox[:4]  # Support both (x,y,w,h) and (x,y,w,h,extras)

                    # Validate bbox masih dalam frame bounds
                    if x >= 0 and y >= 0 and x + w <= frame_w and y + h <= frame_h and w > 0 and h > 0:
                        # Draw green rectangle
                        cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

                        # ★ NEW: Draw plate label di atas bbox
                        # Cari matching plate text dari all_detected_plates
                        label_text = f"PLAT {idx}"  # Default label

                        # Try to find matching detected plate text
                        global all_detected_plates
                        if all_detected_plates:
                            for plate_info in all_detected_plates:
                                plate_bbox = plate_info.get('bbox', [])
                                if len(plate_bbox) >= 4:
                                    px, py, pw, ph = plate_bbox[:4]
                                    # Check if bbox matches (with tolerance)
                                    if abs(x - px) < 10 and abs(y - py) < 10:
                                        plate_text = plate_info.get('text', '')
                                        confidence = plate_info.get('confidence', 0)
                                        if plate_text:
                                            label_text = f"{plate_text} ({confidence:.0%})"
                                        break

                        # Draw label background (semi-transparent black box)
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        label_w, label_h = label_size

                        # Background rectangle
                        cv2.rectangle(frame,
                                    (x, y - label_h - 10),
                                    (x + label_w + 10, y),
                                    (0, 0, 0), -1)  # Black filled

                        # Text label
                        cv2.putText(frame, label_text, (x + 5, y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

                        logger.debug(f"✅ Drew plate bbox {idx}: {label_text} at ({x},{y},{w},{h})")
                    else:
                        logger.debug(f"Skipping invalid plate bbox: ({x},{y},{w},{h})")

    except Exception as e:
        logger.error(f"Error drawing bboxes: {e}")

def process_vehicle_access(plate_text, confidence):
    """
    Penjelasan SMK: Seperti 'security guard digital'
    Cek database: boleh masuk atau tidak?
    """
    global system_status, latest_notification

    try:
        conn = get_db_connection()
        if not conn:
            return 'error', "Database connection failed"

        cursor = conn.cursor()  # DictCursor already set in pool

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

            # ★ UPDATE NOTIFICATION untuk real-time display di beranda
            with notification_lock:
                latest_notification = {
                    'status': 'authorized',
                    'plate_text': plate_text,
                    'owner_name': vehicle['nama_pemilik'],
                    'vehicle_type': vehicle['jenis_kendaraan'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': f"Welcome {vehicle['nama_pemilik']}!"
                }

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

            # ★ UPDATE NOTIFICATION untuk real-time display di beranda
            with notification_lock:
                latest_notification = {
                    'status': 'denied',
                    'plate_text': plate_text,
                    'owner_name': 'Unknown',
                    'vehicle_type': 'Unknown',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': 'Kendaraan tidak terdaftar!'
                }

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
    ★ UPDATE: Sekarang return SEMUA plat yang terdeteksi (all_plates)
    """
    global latest_detection, all_detected_plates
    with detection_lock:
        # ★ NEW: Include all detected plates untuk status display
        response_data = {
            'system_status': system_status
        }

        # Add latest single detection (backwards compatibility)
        if latest_detection:
            response_data['status'] = 'detected'
            response_data['plate_text'] = latest_detection['plate_text']
            response_data['confidence'] = latest_detection['confidence']
            response_data['timestamp'] = latest_detection['timestamp']
        else:
            response_data['status'] = 'no_detection'

        # ★ NEW: Add all detected plates dengan format yang siap ditampilkan
        # Format: [{"text": "B 1234 ABC", "confidence": 0.75, "bbox": [x,y,w,h]}, ...]
        with bboxes_lock:
            response_data['all_plates'] = []
            for idx, plate_info in enumerate(all_detected_plates, 1):
                plate_data = {
                    'index': idx,
                    'text': plate_info.get('text', ''),
                    'confidence': plate_info.get('confidence', 0),
                    'bbox': plate_info.get('bbox', [0, 0, 0, 0]),
                    'label': f"PLAT {idx}"  # Default label
                }
                # Override label jika ada text
                if plate_data['text']:
                    plate_data['label'] = f"{plate_data['text']} ({plate_data['confidence']:.0%})"

                response_data['all_plates'].append(plate_data)

        return jsonify(response_data)

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
                            cursor = conn.cursor()  # DictCursor already set in pool
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
                    cursor = conn.cursor()  # DictCursor already set in pool
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

@app.route('/api/latest_notification')
def api_latest_notification():
    """
    API untuk get notifikasi akses terbaru (Authorized/Denied)

    Penjelasan SMK: Frontend polling endpoint ini setiap 2 detik
    untuk update notifikasi real-time di beranda

    Returns:
        JSON dengan status, plat, owner, message, timestamp
    """
    try:
        with notification_lock:
            notification = latest_notification.copy()

        if notification['status'] is None:
            # Belum ada detection
            return jsonify({
                'status': 'no_data',
                'message': 'Waiting for vehicle detection...'
            })

        return jsonify({
            'status': 'success',
            'notification': notification
        })

    except Exception as e:
        logger.error(f"Error getting latest notification: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
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

        cursor = conn.cursor()  # DictCursor already set in pool
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

        cursor = conn.cursor()  # DictCursor already set in pool

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

@app.route('/gambarplat/<path:filename>')
def serve_plate_image(filename):
    """
    Penjelasan SMK: Serve gambar plat dari folder gambarplat/
    Seperti static file server untuk gambar
    """
    from flask import send_from_directory
    return send_from_directory('gambarplat', filename)

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

        cursor = conn.cursor()  # DictCursor already set in pool

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