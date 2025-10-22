"""
Headless Stream Server dengan Flask
Live streaming CCTV + detection results ke browser
"""

import os
import time
import logging
import json
import cv2
import csv
from io import StringIO
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash, make_response
from flask_socketio import SocketIO, emit
import threading
from stream_manager import HeadlessStreamManager
from database import PlateDatabase
from config import CCTVConfig, MySQLConfig
from utils.yolo_detector import check_and_install_yolo

# MySQL Integration (optional)
try:
    from mysql_database import MySQLPlateDatabase
    from access_controller import AccessController
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logging.warning("MySQL modules not available. CRUD features will be limited.")

# Setup Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'live_cctv_detection_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
stream_manager = None
database = PlateDatabase()

# MySQL database instance (singleton)
mysql_db = None
if MYSQL_AVAILABLE and MySQLConfig.USE_MYSQL_DATABASE:
    try:
        mysql_db = MySQLPlateDatabase.get_instance()
        logging.info("✅ MySQL database initialized for CRUD operations")
    except Exception as e:
        logging.warning(f"⚠️ MySQL initialization failed: {str(e)}")
        MYSQL_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main page dengan live stream"""
    return render_template('stream.html')

@app.route('/api/stats')
def get_stats():
    """API endpoint untuk statistics"""
    if stream_manager and stream_manager.is_running():
        stats = stream_manager.get_statistics()
        return jsonify(stats)
    else:
        return jsonify({'error': 'Stream not running'})

@app.route('/api/recent_detections')
def get_recent_detections():
    """API endpoint untuk recent detections"""
    try:
        detections = database.get_recent_detections(limit=10)
        return jsonify(detections)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    """Start streaming"""
    global stream_manager
    
    try:
        data = request.get_json()
        source = data.get('source', CCTVConfig.DEFAULT_RTSP_URL)
        
        if stream_manager and stream_manager.is_running():
            return jsonify({'error': 'Stream already running'})
        
        stream_manager = HeadlessStreamManager(source, database, enable_tracking=True)
        
        # Add callbacks
        stream_manager.add_frame_callback(on_new_frame)
        stream_manager.add_detection_callback(on_new_detection)
        
        if stream_manager.start():
            return jsonify({'success': True, 'message': 'Stream started'})
        else:
            return jsonify({'error': 'Failed to start stream'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    """Stop streaming"""
    global stream_manager

    try:
        if stream_manager:
            stream_manager.stop()
            stream_manager = None

        return jsonify({'success': True, 'message': 'Stream stopped'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/toggle_person_detection', methods=['POST'])
def toggle_person_detection():
    """Toggle person detection on/off"""
    global stream_manager

    try:
        if not stream_manager or not stream_manager.is_running():
            return jsonify({'error': 'Stream not running'})

        data = request.get_json()
        enable = data.get('enable', True)

        # Toggle person detection
        success = stream_manager.toggle_person_detection(enable)

        if success:
            status = "enabled" if enable else "disabled"
            return jsonify({
                'success': True,
                'message': f'Person detection {status}',
                'person_detection_enabled': enable
            })
        else:
            return jsonify({'error': 'Failed to toggle person detection'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/toggle_vehicle_detection', methods=['POST'])
def toggle_vehicle_detection():
    """Toggle vehicle detection on/off"""
    global stream_manager

    try:
        if not stream_manager or not stream_manager.is_running():
            return jsonify({'error': 'Stream not running'})

        data = request.get_json()
        enable = data.get('enable', True)

        # Toggle vehicle detection
        success = stream_manager.toggle_vehicle_detection(enable)

        if success:
            status = "enabled" if enable else "disabled"
            return jsonify({
                'success': True,
                'message': f'Vehicle detection {status}',
                'vehicle_detection_enabled': enable
            })
        else:
            return jsonify({'error': 'Failed to toggle vehicle detection'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/screenshot', methods=['POST'])
def take_screenshot():
    """Take screenshot dari current frame dan simpan ke detected_plates"""
    global stream_manager
    
    try:
        if not stream_manager or not stream_manager.is_running():
            return jsonify({'error': 'Stream not running'})
        
        # Get current frame
        current_frame = stream_manager.get_current_frame()
        
        if not current_frame or not current_frame.image_base64:
            return jsonify({'error': 'No frame available'})
        
        # Decode base64 to image
        import base64
        import cv2
        import numpy as np
        from datetime import datetime
        from config import SystemConfig
        
        # Decode base64
        frame_bytes = base64.b64decode(current_frame.image_base64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode frame'})
        
        # Ensure output folder exists
        import os
        if not os.path.exists(SystemConfig.OUTPUT_FOLDER):
            os.makedirs(SystemConfig.OUTPUT_FOLDER)
        
        # Generate filename dengan timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"screenshot_{timestamp}.jpg"
        
        # Handle duplicate filenames
        counter = 1
        base_filename = filename
        full_path = os.path.join(SystemConfig.OUTPUT_FOLDER, filename)
        while os.path.exists(full_path):
            name, ext = os.path.splitext(base_filename)
            filename = f"{name}_{counter}{ext}"
            full_path = os.path.join(SystemConfig.OUTPUT_FOLDER, filename)
            counter += 1
        
        # Save screenshot
        success = cv2.imwrite(full_path, frame)
        
        if success:
            # Log screenshot
            logger.info(f"📷 Screenshot saved: {full_path}")
            
            return jsonify({
                'success': True, 
                'message': 'Screenshot saved successfully',
                'filename': filename,
                'path': full_path,
                'frame_id': current_frame.frame_id,
                'timestamp': current_frame.timestamp
            })
        else:
            return jsonify({'error': 'Failed to save screenshot'})
            
    except Exception as e:
        logger.error(f"Screenshot error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/toggle_yolo', methods=['POST'])
def toggle_yolo():
    """Toggle YOLOv8 object detection"""
    global stream_manager
    
    try:
        if not stream_manager or not stream_manager.is_running():
            return jsonify({'error': 'Stream not running'})
        
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        if enabled:
            success = stream_manager.enable_yolo()
        else:
            stream_manager.disable_yolo()
            success = True
        
        if success:
            status = 'enabled' if enabled else 'disabled'
            return jsonify({
                'success': True, 
                'message': f'Object detection {status}',
                'enabled': enabled
            })
        else:
            return jsonify({'error': 'Failed to enable object detection - YOLO not available'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/toggle_sequential', methods=['POST'])
def toggle_sequential():
    """Toggle sequential detection mode"""
    global stream_manager
    
    try:
        if not stream_manager or not stream_manager.is_running():
            return jsonify({'error': 'Stream not running'})
        
        data = request.get_json()
        enabled = data.get('enabled', False)
        grid_zones = data.get('grid_zones', [3, 3])
        cycle_time = data.get('cycle_time', 2.0)
        
        if enabled:
            success = stream_manager.enable_sequential_detection(
                grid_zones=tuple(grid_zones), 
                cycle_time=cycle_time
            )
        else:
            success = stream_manager.disable_sequential_detection()
        
        if success:
            status = 'enabled' if enabled else 'disabled'
            return jsonify({
                'success': True, 
                'message': f'Sequential detection {status}',
                'enabled': enabled,
                'grid_zones': grid_zones,
                'cycle_time': cycle_time
            })
        else:
            return jsonify({'error': 'Failed to toggle sequential detection'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/sequential_info')
def get_sequential_info():
    """Get sequential detection information"""
    if stream_manager and stream_manager.is_running():
        info = stream_manager.get_sequential_info()
        return jsonify(info)
    else:
        return jsonify({'sequential_mode': False})

def generate_frames():
    """Generator untuk video frames"""
    while True:
        try:
            if stream_manager and stream_manager.is_running():
                current_frame = stream_manager.get_current_frame()
                
                if current_frame and current_frame.image_base64:
                    # Decode base64 to bytes
                    import base64
                    frame_bytes = base64.b64decode(current_frame.image_base64)
                    
                    # Create proper MJPEG frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.1)
            else:
                # Send placeholder frame when not streaming
                placeholder_bytes = create_placeholder_frame_bytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error in generate_frames: {str(e)}")
            # Send error frame
            error_bytes = create_error_frame_bytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_bytes + b'\r\n')
            time.sleep(1)

def create_placeholder_frame():
    """Create placeholder frame saat tidak streaming"""
    import cv2
    import numpy as np
    import base64
    
    # Create simple placeholder
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    cv2.putText(frame, "No Stream", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(frame, "Click Start to begin", (150, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return frame_base64

def create_placeholder_frame_bytes():
    """Create placeholder frame as bytes"""
    import cv2
    import numpy as np
    
    # Create animated placeholder
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 30
    
    # Add gradient background
    for i in range(480):
        frame[i, :] = [30 + i//10, 40 + i//15, 50 + i//20]
    
    # Add text
    cv2.putText(frame, "CCTV PLATE DETECTION", (120, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, "No Stream Active", (180, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.putText(frame, "Click 'Start Stream' to begin", (120, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Add timestamp
    import time
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (200, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Encode to JPEG bytes
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()

def create_error_frame_bytes():
    """Create error frame as bytes"""
    import cv2
    import numpy as np
    
    # Create error frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 20
    frame[:, :, 2] = 80  # Red tint
    
    cv2.putText(frame, "CONNECTION ERROR", (150, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(frame, "Check video source", (160, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
    cv2.putText(frame, "and try again", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), 2)
    
    # Encode to JPEG bytes
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def on_new_frame(stream_frame):
    """Callback untuk new frame - emit ke WebSocket"""
    try:
        # Emit frame data via WebSocket
        socketio.emit('new_frame', {
            'frame_id': stream_frame.frame_id,
            'timestamp': stream_frame.timestamp,
            'fps': stream_frame.fps,
            'processing_time': stream_frame.processing_time,
            'detections': stream_frame.detections
        })
    except Exception as e:
        logger.error(f"Error emitting frame: {str(e)}")

def on_new_detection(detections):
    """Callback untuk new detection - emit ke WebSocket"""
    try:
        detection_data = []
        access_results = []

        for det in detections:
            det_info = {
                'text': det.text,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'timestamp': time.time()
            }

            # Check if detection has access control result (MySQL integration)
            if hasattr(det, 'access_result') and det.access_result:
                det_info['access_result'] = det.access_result
                access_results.append(det.access_result)

            detection_data.append(det_info)

        # Emit detection via WebSocket
        socketio.emit('new_detection', {
            'detections': detection_data,
            'count': len(detection_data)
        })

        # Emit access control results separately (if any)
        if access_results:
            for access_result in access_results:
                socketio.emit('access_control_result', access_result)
                logger.info(f"📤 Sent access control result: {access_result['access']} - {access_result['plate_number']}")

    except Exception as e:
        logger.error(f"Error emitting detection: {str(e)}")

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    
    # Send current status
    if stream_manager and stream_manager.is_running():
        emit('stream_status', {'running': True})
        
        # Send current stats
        stats = stream_manager.get_statistics()
        emit('stats_update', stats)
    else:
        emit('stream_status', {'running': False})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

@socketio.on('get_stats')
def handle_get_stats():
    """Handle stats request"""
    if stream_manager and stream_manager.is_running():
        stats = stream_manager.get_statistics()
        emit('stats_update', stats)

# ================== MULTI-CAMERA ENDPOINTS ==================

@app.route('/api/cameras/discover', methods=['POST'])
def discover_cameras():
    """Discover available cameras di system"""
    try:
        from utils.camera_manager import CameraManager
        
        camera_mgr = CameraManager()
        cameras = camera_mgr.enumerate_cameras()
        available = camera_mgr.get_available_cameras()
        
        camera_list = []
        for camera in available:
            camera_list.append({
                'index': camera.index,
                'name': camera.name,
                'resolution': camera.resolution,
                'fps': camera.fps,
                'backend': camera.backend,
                'available': camera.available
            })
        
        return jsonify({
            'success': True,
            'cameras': camera_list,
            'total_discovered': len(cameras),
            'total_available': len(available)
        })
        
    except Exception as e:
        logger.error(f"Camera discovery error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/cameras/test/<int:camera_index>', methods=['POST'])
def test_camera(camera_index):
    """Test specific camera"""
    try:
        from utils.camera_manager import CameraManager, CameraInfo
        
        camera_mgr = CameraManager()
        
        # Create temp camera info
        temp_camera = CameraInfo(
            index=camera_index,
            name=f"Camera {camera_index}",
            resolution=(640, 480),
            fps=30.0,
            backend="auto",
            available=True
        )
        
        # Test camera
        success = camera_mgr.test_camera_capture(temp_camera, duration=2.0)
        
        return jsonify({
            'success': success,
            'camera_index': camera_index,
            'message': 'Camera test passed' if success else 'Camera test failed'
        })
        
    except Exception as e:
        logger.error(f"Camera test error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/multi_stream/start', methods=['POST'])
def start_multi_stream():
    """Start multi-camera streaming"""
    global stream_manager
    
    try:
        data = request.get_json()
        cameras = data.get('cameras', [])
        
        if not cameras:
            return jsonify({'error': 'No cameras specified'})
        
        # Import multi-camera stream
        from utils.multi_camera_stream import MultiCameraStream
        
        # Stop existing single stream if running
        if stream_manager and stream_manager.is_running():
            stream_manager.stop()
            stream_manager = None
        
        # Create multi-camera stream manager
        multi_stream = MultiCameraStream()
        
        # Add cameras
        added_cameras = []
        for camera_config in cameras:
            camera_id = camera_config.get('id', f"camera_{camera_config.get('index', 0)}")
            camera_source = camera_config.get('source')
            config = {
                'name': camera_config.get('name', camera_id),
                'resolution': tuple(camera_config.get('resolution', [640, 480])),
                'fps_limit': camera_config.get('fps_limit', 10)
            }
            
            if multi_stream.add_camera_source(camera_id, camera_source, config):
                added_cameras.append(camera_id)
        
        if not added_cameras:
            return jsonify({'error': 'Failed to add any cameras'})
        
        # Add callbacks
        multi_stream.add_frame_callback(on_multi_camera_frame)
        multi_stream.add_detection_callback(on_multi_camera_detection)
        
        # Start multi-stream
        if multi_stream.start():
            # Store reference (reuse global stream_manager)
            stream_manager = multi_stream
            
            return jsonify({
                'success': True,
                'message': f'Multi-camera stream started with {len(added_cameras)} cameras',
                'cameras': added_cameras
            })
        else:
            return jsonify({'error': 'Failed to start multi-camera stream'})
            
    except Exception as e:
        logger.error(f"Multi-stream start error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/multi_stream/status', methods=['GET'])
def get_multi_stream_status():
    """Get multi-camera stream status"""
    try:
        from utils.multi_camera_stream import MultiCameraStream
        
        if isinstance(stream_manager, MultiCameraStream) and stream_manager.running:
            cameras = stream_manager.get_camera_list()
            stats = stream_manager.get_statistics()
            
            return jsonify({
                'success': True,
                'running': True,
                'type': 'multi_camera',
                'cameras': cameras,
                'statistics': stats
            })
        else:
            return jsonify({
                'success': True,
                'running': False,
                'type': 'single_camera' if stream_manager else 'none'
            })
            
    except Exception as e:
        logger.error(f"Multi-stream status error: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/multi_stream/camera/<camera_id>/latest', methods=['GET'])
def get_camera_latest_frame(camera_id):
    """Get latest frame dari specific camera"""
    try:
        from utils.multi_camera_stream import MultiCameraStream
        
        if not isinstance(stream_manager, MultiCameraStream) or not stream_manager.running:
            return jsonify({'error': 'Multi-camera stream not running'})
        
        results = stream_manager.get_latest_results()
        
        if camera_id not in results or results[camera_id] is None:
            return jsonify({'error': f'No frame available for camera {camera_id}'})
        
        result = results[camera_id]
        
        # Encode frame ke base64
        import base64
        _, buffer = cv2.imencode('.jpg', result.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'camera_name': result.camera_name,
            'frame_id': result.frame_id,
            'timestamp': result.timestamp,
            'detections': [
                {
                    'text': det.text,
                    'confidence': det.confidence,
                    'bbox': det.bbox
                }
                for det in result.detections
            ],
            'frame_base64': frame_base64
        })
        
    except Exception as e:
        logger.error(f"Camera frame error: {str(e)}")
        return jsonify({'error': str(e)})

def on_multi_camera_frame(result):
    """Callback untuk multi-camera frame"""
    try:
        # Emit frame data untuk specific camera
        socketio.emit('multi_camera_frame', {
            'camera_id': result.camera_id,
            'camera_name': result.camera_name,
            'frame_id': result.frame_id,
            'timestamp': result.timestamp,
            'detections': len(result.detections)
        })
    except Exception as e:
        logger.error(f"Error emitting multi-camera frame: {str(e)}")

def on_multi_camera_detection(camera_id, detections):
    """Callback untuk multi-camera detection"""
    try:
        detection_data = []
        for det in detections:
            detection_data.append({
                'text': det.text,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'timestamp': time.time()
            })
        
        # Emit detection untuk specific camera
        socketio.emit('multi_camera_detection', {
            'camera_id': camera_id,
            'detections': detection_data,
            'count': len(detection_data)
        })
        
    except Exception as e:
        logger.error(f"Error emitting multi-camera detection: {str(e)}")

# ================== END MULTI-CAMERA ENDPOINTS ==================

# ================== CRUD ROUTES FOR VEHICLES & ACCESS LOG ==================

@app.route('/vehicles')
def vehicles():
    """Vehicles management page"""
    if not MYSQL_AVAILABLE or not mysql_db:
        flash('MySQL database not available. Please check configuration.', 'danger')
        return redirect(url_for('index'))

    try:
        # Get all vehicles from MySQL
        vehicles_list = mysql_db.get_all_vehicles()

        # Get statistics
        stats = mysql_db.get_statistics()

        # Count by status
        present_count = len([v for v in vehicles_list if v.get('status') == 'Terdaftar'])
        absent_count = len([v for v in vehicles_list if v.get('status') == 'Belum'])

        return render_template('vehicles.html',
                             vehicles=vehicles_list,
                             total_vehicles=len(vehicles_list),
                             present_count=present_count,
                             absent_count=absent_count,
                             today_access=stats.get('access_today', 0))
    except Exception as e:
        logger.error(f"Error loading vehicles: {str(e)}")
        flash(f'Error loading vehicles: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/vehicles/add', methods=['GET', 'POST'])
def add_vehicle():
    """Add new vehicle"""
    if not MYSQL_AVAILABLE or not mysql_db:
        flash('MySQL database not available', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            plate_number = request.form.get('plate_number', '').strip().upper()
            owner_name = request.form.get('owner_name', '').strip()
            vehicle_type = request.form.get('vehicle_type', '').strip()
            contact_info = request.form.get('contact_info', '').strip()

            # Validation
            if not plate_number or not owner_name or not vehicle_type:
                flash('Plate number, owner name, and vehicle type are required', 'warning')
                return render_template('vehicle_form.html', vehicle=None)

            # Register vehicle
            vehicle_id = mysql_db.register_vehicle(
                plate_number=plate_number,
                owner_name=owner_name,
                vehicle_type=vehicle_type,
                contact_info=contact_info,
                status='Belum'
            )

            if vehicle_id:
                flash(f'Vehicle {plate_number} added successfully!', 'success')
                return redirect(url_for('vehicles'))
            else:
                flash(f'Failed to add vehicle. Plate number {plate_number} may already exist.', 'danger')
                return render_template('vehicle_form.html', vehicle=None)

        except Exception as e:
            logger.error(f"Error adding vehicle: {str(e)}")
            flash(f'Error adding vehicle: {str(e)}', 'danger')
            return render_template('vehicle_form.html', vehicle=None)

    # GET request - show form
    return render_template('vehicle_form.html', vehicle=None)

@app.route('/vehicles/edit/<int:vehicle_id>', methods=['GET', 'POST'])
def edit_vehicle(vehicle_id):
    """Edit existing vehicle"""
    if not MYSQL_AVAILABLE or not mysql_db:
        flash('MySQL database not available', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            owner_name = request.form.get('owner_name', '').strip()
            vehicle_type = request.form.get('vehicle_type', '').strip()
            contact_info = request.form.get('contact_info', '').strip()
            status = request.form.get('status', 'Belum').strip()

            # Validation
            if not owner_name or not vehicle_type:
                flash('Owner name and vehicle type are required', 'warning')
                return redirect(url_for('edit_vehicle', vehicle_id=vehicle_id))

            # Update vehicle
            with mysql_db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE vehicles
                        SET owner_name = %s, vehicle_type = %s, contact_info = %s,
                            status = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (owner_name, vehicle_type, contact_info, status, vehicle_id))

                    if cursor.rowcount > 0:
                        flash('Vehicle updated successfully!', 'success')
                    else:
                        flash('Vehicle not found', 'warning')

            return redirect(url_for('vehicles'))

        except Exception as e:
            logger.error(f"Error updating vehicle: {str(e)}")
            flash(f'Error updating vehicle: {str(e)}', 'danger')
            return redirect(url_for('edit_vehicle', vehicle_id=vehicle_id))

    # GET request - show edit form
    try:
        with mysql_db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM vehicles WHERE id = %s", (vehicle_id,))
                vehicle = cursor.fetchone()

                if not vehicle:
                    flash('Vehicle not found', 'warning')
                    return redirect(url_for('vehicles'))

                return render_template('vehicle_form.html', vehicle=vehicle)
    except Exception as e:
        logger.error(f"Error loading vehicle: {str(e)}")
        flash(f'Error loading vehicle: {str(e)}', 'danger')
        return redirect(url_for('vehicles'))

@app.route('/vehicles/delete/<int:vehicle_id>', methods=['POST'])
def delete_vehicle(vehicle_id):
    """Delete vehicle (with PIN protection)"""
    if not MYSQL_AVAILABLE or not mysql_db:
        return jsonify({'success': False, 'message': 'MySQL database not available'})

    try:
        # Check PIN (simple protection)
        pin = request.form.get('pin', '')
        if pin != '1234':  # Simple PIN check
            return jsonify({'success': False, 'message': 'Invalid PIN'})

        # Delete vehicle
        with mysql_db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get vehicle info first
                cursor.execute("SELECT plate_number FROM vehicles WHERE id = %s", (vehicle_id,))
                vehicle = cursor.fetchone()

                if not vehicle:
                    return jsonify({'success': False, 'message': 'Vehicle not found'})

                # Delete vehicle
                cursor.execute("DELETE FROM vehicles WHERE id = %s", (vehicle_id,))

                if cursor.rowcount > 0:
                    logger.info(f"Vehicle deleted: {vehicle['plate_number']} (ID: {vehicle_id})")
                    return jsonify({
                        'success': True,
                        'message': f"Vehicle {vehicle['plate_number']} deleted successfully"
                    })
                else:
                    return jsonify({'success': False, 'message': 'Failed to delete vehicle'})

    except Exception as e:
        logger.error(f"Error deleting vehicle: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/access-log')
def access_log():
    """Access log history page with filters"""
    if not MYSQL_AVAILABLE or not mysql_db:
        flash('MySQL database not available', 'danger')
        return redirect(url_for('index'))

    try:
        # Get filter parameters
        date_range = request.args.get('date_range', 'last7days')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status_filter = request.args.get('status', '')
        plate_search = request.args.get('plate', '')

        # Build query with filters
        query = """
            SELECT al.*, v.owner_name, v.vehicle_type
            FROM access_log al
            LEFT JOIN vehicles v ON al.vehicle_id = v.id
            WHERE 1=1
        """
        params = []

        # Date filter
        if date_range == 'today':
            query += " AND DATE(al.acces_time) = CURDATE()"
        elif date_range == 'yesterday':
            query += " AND DATE(al.acces_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)"
        elif date_range == 'last7days':
            query += " AND al.acces_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
        elif date_range == 'last30days':
            query += " AND al.acces_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
        elif date_range == 'custom' and start_date and end_date:
            query += " AND DATE(al.acces_time) BETWEEN %s AND %s"
            params.extend([start_date, end_date])

        # Status filter
        if status_filter:
            query += " AND al.status = %s"
            params.append(status_filter)

        # Plate search
        if plate_search:
            query += " AND al.plate_number LIKE %s"
            params.append(f"%{plate_search}%")

        query += " ORDER BY al.acces_time DESC LIMIT 1000"

        # Execute query
        with mysql_db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                access_logs_list = cursor.fetchall()

        # Calculate statistics
        total_records = len(access_logs_list)
        entry_count = len([log for log in access_logs_list if log.get('status') == 'masuk'])
        exit_count = len([log for log in access_logs_list if log.get('status') == 'keluar'])
        denied_count = len([log for log in access_logs_list if log.get('status') == 'ditolak'])

        return render_template('access_log.html',
                             access_logs=access_logs_list,
                             total_records=total_records,
                             entry_count=entry_count,
                             exit_count=exit_count,
                             denied_count=denied_count)
    except Exception as e:
        logger.error(f"Error loading access log: {str(e)}")
        flash(f'Error loading access log: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/access-log/export')
def export_access_log_csv():
    """Export access log to CSV"""
    if not MYSQL_AVAILABLE or not mysql_db:
        flash('MySQL database not available', 'danger')
        return redirect(url_for('index'))

    try:
        # Get filter parameters (same as access_log route)
        date_range = request.args.get('date_range', 'last7days')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        status_filter = request.args.get('status', '')
        plate_search = request.args.get('plate', '')

        # Build query (same as access_log route)
        query = """
            SELECT al.acces_time, al.plate_number, v.owner_name, v.vehicle_type, al.status, al.image_url
            FROM access_log al
            LEFT JOIN vehicles v ON al.vehicle_id = v.id
            WHERE 1=1
        """
        params = []

        # Apply same filters
        if date_range == 'today':
            query += " AND DATE(al.acces_time) = CURDATE()"
        elif date_range == 'yesterday':
            query += " AND DATE(al.acces_time) = DATE_SUB(CURDATE(), INTERVAL 1 DAY)"
        elif date_range == 'last7days':
            query += " AND al.acces_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
        elif date_range == 'last30days':
            query += " AND al.acces_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
        elif date_range == 'custom' and start_date and end_date:
            query += " AND DATE(al.acces_time) BETWEEN %s AND %s"
            params.extend([start_date, end_date])

        if status_filter:
            query += " AND al.status = %s"
            params.append(status_filter)

        if plate_search:
            query += " AND al.plate_number LIKE %s"
            params.append(f"%{plate_search}%")

        query += " ORDER BY al.acces_time DESC"

        # Execute query
        with mysql_db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                access_logs = cursor.fetchall()

        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(['Timestamp', 'Plate Number', 'Owner Name', 'Vehicle Type', 'Status', 'Image Path'])

        # Write data
        for log in access_logs:
            writer.writerow([
                log.get('acces_time', ''),
                log.get('plate_number', ''),
                log.get('owner_name', 'Unknown'),
                log.get('vehicle_type', 'Unknown'),
                log.get('status', ''),
                log.get('image_url', '')
            ])

        # Create response
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=access_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        response.headers['Content-Type'] = 'text/csv; charset=utf-8-sig'  # UTF-8 BOM for Excel

        return response

    except Exception as e:
        logger.error(f"Error exporting CSV: {str(e)}")
        flash(f'Error exporting CSV: {str(e)}', 'danger')
        return redirect(url_for('access_log'))

@app.route('/access-log/bulk-delete', methods=['POST'])
def bulk_delete_access_log():
    """Bulk delete access log records with PIN protection"""
    if not MYSQL_AVAILABLE or not mysql_db:
        return jsonify({'success': False, 'error': 'MySQL database not available'}), 503

    try:
        data = request.get_json()
        ids = data.get('ids', [])
        pin = data.get('pin', '')

        # Validate PIN
        if pin != '1234':
            return jsonify({'success': False, 'error': 'Invalid PIN'}), 403

        # Validate IDs
        if not ids or not isinstance(ids, list):
            return jsonify({'success': False, 'error': 'No records selected'}), 400

        # Convert IDs to integers and validate
        try:
            ids = [int(id_val) for id_val in ids]
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid ID format'}), 400

        # Delete records
        with mysql_db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Use parameterized query to prevent SQL injection
                placeholders = ','.join(['%s'] * len(ids))
                query = f"DELETE FROM access_log WHERE id IN ({placeholders})"
                cursor.execute(query, ids)
                deleted_count = cursor.rowcount

        logger.info(f"Bulk deleted {deleted_count} access log records")
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Successfully deleted {deleted_count} record(s)'
        })

    except Exception as e:
        logger.error(f"Error bulk deleting access logs: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vehicles/stats')
def api_vehicles_stats():
    """API endpoint for vehicles statistics"""
    if not MYSQL_AVAILABLE or not mysql_db:
        return jsonify({'error': 'MySQL not available'})

    try:
        vehicles_list = mysql_db.get_all_vehicles()
        stats = mysql_db.get_statistics()

        present_count = len([v for v in vehicles_list if v.get('status') == 'Hadir'])
        absent_count = len([v for v in vehicles_list if v.get('status') == 'Belum'])

        return jsonify({
            'total': len(vehicles_list),
            'present': present_count,
            'absent': absent_count,
            'today_access': stats.get('access_today', 0)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/access-log/stats')
def api_access_log_stats():
    """API endpoint for access log statistics"""
    if not MYSQL_AVAILABLE or not mysql_db:
        return jsonify({'error': 'MySQL not available'})

    try:
        # Get filter parameters
        date_range = request.args.get('date_range', 'last7days')
        status_filter = request.args.get('status', '')
        plate_search = request.args.get('plate', '')

        # Build query
        query = "SELECT status FROM access_log WHERE 1=1"
        params = []

        # Apply filters
        if date_range == 'today':
            query += " AND DATE(acces_time) = CURDATE()"
        elif date_range == 'last7days':
            query += " AND acces_time >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
        elif date_range == 'last30days':
            query += " AND acces_time >= DATE_SUB(NOW(), INTERVAL 30 DAY)"

        if status_filter:
            query += " AND status = %s"
            params.append(status_filter)

        if plate_search:
            query += " AND plate_number LIKE %s"
            params.append(f"%{plate_search}%")

        # Execute query
        with mysql_db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                logs = cursor.fetchall()

        # Count by status
        entry_count = len([log for log in logs if log.get('status') == 'masuk'])
        exit_count = len([log for log in logs if log.get('status') == 'keluar'])
        denied_count = len([log for log in logs if log.get('status') == 'ditolak'])

        return jsonify({
            'total': len(logs),
            'entry': entry_count,
            'exit': exit_count,
            'denied': denied_count
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ================== END CRUD ROUTES ==================

def stats_broadcaster():
    """Background thread untuk broadcast stats"""
    while True:
        try:
            if stream_manager and stream_manager.is_running():
                stats = stream_manager.get_statistics()
                socketio.emit('stats_update', stats)
            
            time.sleep(2)  # Broadcast every 2 seconds
            
        except Exception as e:
            logger.error(f"Error broadcasting stats: {str(e)}")
            time.sleep(5)

def main(source=None, host='0.0.0.0', port=5010, debug=False, no_yolo=False):
    """
    Main function untuk run headless stream server
    
    Args:
        source: Video source (default: dari config)
        host: Server host (default: 0.0.0.0)
        port: Server port (default: 5010)
        debug: Debug mode
    """
    global stream_manager
    
    print("🌐 HEADLESS CCTV STREAMING SERVER")
    print("=" * 60)
    print(f"🚀 Starting server at: http://{host}:{port}")
    print(f"📹 Video source: {source or CCTVConfig.DEFAULT_RTSP_URL}")
    print(f"💾 Database: {database.db_path}")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    
    # Check and install YOLOv8 if needed (skip if no_yolo)
    if not no_yolo:
        print("🔍 Checking YOLOv8 dependencies...")
        yolo_available = check_and_install_yolo()
        if yolo_available:
            print("✅ YOLOv8 (ultralytics) is available")
        else:
            print("⚠️  YOLOv8 not available - object detection will be disabled")
    else:
        print("⚡ YOLOv8 disabled for faster startup")
    print("=" * 60)
    
    # Start stats broadcaster thread
    stats_thread = threading.Thread(target=stats_broadcaster, daemon=True)
    stats_thread.start()
    
    # Auto-start stream jika source diberikan
    if source:
        try:
            stream_manager = HeadlessStreamManager(source, database, enable_yolo=not no_yolo, enable_tracking=True)
            stream_manager.add_frame_callback(on_new_frame)
            stream_manager.add_detection_callback(on_new_detection)
            
            if stream_manager.start():
                logger.info("Auto-started stream successfully")
            else:
                logger.error("Failed to auto-start stream")
        except Exception as e:
            logger.error(f"Error auto-starting stream: {str(e)}")
    
    try:
        # Run Flask-SocketIO server
        socketio.run(app, 
                    host=host, 
                    port=port, 
                    debug=debug,
                    use_reloader=False,
                    allow_unsafe_werkzeug=True)  # Allow for development
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        
        if stream_manager:
            stream_manager.stop()
        
        print("✅ Server stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Headless CCTV Streaming Server')
    parser.add_argument('--source', '-s', type=str, 
                        help='Video source (RTSP URL, webcam index, file)')
    parser.add_argument('--host', default='0.0.0.0', 
                        help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5010, 
                        help='Server port (default: 5010)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode')
    parser.add_argument('--no-yolo', action='store_true',
                        help='Disable YOLOv8 for faster startup')
    
    args = parser.parse_args()
    
    main(
        source=args.source,
        host=args.host,
        port=args.port,
        debug=args.debug,
        no_yolo=args.no_yolo
    )