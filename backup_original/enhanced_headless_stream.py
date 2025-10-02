"""
Enhanced Headless Stream Server dengan Flask + Enhanced Hybrid YOLO
Live streaming CCTV + Enhanced detection results ke browser
Integrasi dengan License Plate YOLO + Ultra-Stable Counter
"""

import os
import time
import logging
import json
import cv2
import base64
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import threading
from utils.enhanced_hybrid_stream_manager import create_enhanced_stream_manager
from database import PlateDatabase
from config import CCTVConfig

# Setup Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced_hybrid_detection_2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
stream_manager = None
database = PlateDatabase()
current_preset = 'cctv_monitoring'  # Default preset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main page dengan enhanced live stream"""
    return render_template('enhanced_stream.html')

@app.route('/api/enhanced_stats')
def get_enhanced_stats():
    """API endpoint untuk enhanced statistics"""
    if stream_manager and stream_manager.is_system_ready():
        try:
            # Get comprehensive stats dari enhanced system
            stats = stream_manager.get_comprehensive_statistics()

            # Add system status
            system_status = stream_manager.get_system_status()
            stats['system_status'] = system_status

            # Add unique plate metrics (main feature)
            stats['unique_metrics'] = {
                'total_unique_plates': stream_manager.get_unique_plate_count(),
                'current_visible_plates': stream_manager.get_current_visible_plates()
            }

            # Recent detection history
            stats['recent_detections'] = stream_manager.get_detection_history(limit=20)

            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return jsonify({'error': str(e)})
    else:
        return jsonify({
            'error': 'Enhanced stream not running',
            'system_ready': False
        })

@app.route('/api/detection_methods')
def get_detection_methods():
    """API endpoint untuk detection method breakdown"""
    if stream_manager and stream_manager.is_system_ready():
        stats = stream_manager.get_comprehensive_statistics()

        # Extract detection method stats
        detection_methods = {
            'plate_yolo': stats.get('enhanced_stream_manager', {}).get('detection_methods', {}).get('plate_yolo', 0),
            'ocr_fallback': stats.get('enhanced_stream_manager', {}).get('detection_methods', {}).get('ocr_fallback', 0),
            'hybrid_validated': stats.get('enhanced_stream_manager', {}).get('detection_methods', {}).get('hybrid_validated', 0),
            'total': stats.get('enhanced_stream_manager', {}).get('detection_methods', {}).get('total_detections', 0)
        }

        return jsonify(detection_methods)
    else:
        return jsonify({'error': 'Stream not running'})

@app.route('/api/presets')
def get_available_presets():
    """Get available configuration presets"""
    presets = {
        'laptop_camera': {
            'name': 'Laptop Camera',
            'description': 'Optimized untuk laptop testing',
            'vehicle_yolo': False,
            'license_plate_yolo': True,
            'frame_skipping': True
        },
        'cctv_monitoring': {
            'name': 'CCTV Monitoring',
            'description': 'Production CCTV dengan dual YOLO',
            'vehicle_yolo': True,
            'license_plate_yolo': True,
            'frame_skipping': True
        },
        'high_accuracy': {
            'name': 'High Accuracy',
            'description': 'Maximum accuracy mode',
            'vehicle_yolo': True,
            'license_plate_yolo': True,
            'frame_skipping': False
        },
        'performance_optimized': {
            'name': 'Performance Mode',
            'description': 'Speed optimized',
            'vehicle_yolo': False,
            'license_plate_yolo': True,
            'frame_skipping': True
        }
    }
    return jsonify(presets)

@app.route('/api/start_enhanced_stream', methods=['POST'])
def start_enhanced_stream():
    """Start enhanced streaming dengan YOLO"""
    global stream_manager, current_preset

    try:
        data = request.get_json()
        source = data.get('source', CCTVConfig.DEFAULT_RTSP_URL)
        preset = data.get('preset', 'cctv_monitoring')
        current_preset = preset

        if stream_manager and stream_manager.is_system_ready():
            return jsonify({'error': 'Enhanced stream already running'})

        logger.info(f"🚀 Starting Enhanced Hybrid Stream with preset: {preset}")
        logger.info(f"📹 Source: {source}")

        # Create enhanced stream manager
        stream_manager = create_enhanced_stream_manager(source, preset)

        if stream_manager.start_stream():
            # Start background thread untuk socketio updates
            threading.Thread(target=emit_real_time_updates, daemon=True).start()

            logger.info("✅ Enhanced stream started successfully")
            return jsonify({
                'success': True,
                'message': 'Enhanced stream started',
                'preset': preset,
                'system_status': stream_manager.get_system_status()
            })
        else:
            logger.error("❌ Failed to start enhanced stream")
            return jsonify({'error': 'Failed to start enhanced stream'})

    except Exception as e:
        logger.error(f"Error starting enhanced stream: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/stop_enhanced_stream', methods=['POST'])
def stop_enhanced_stream():
    """Stop enhanced streaming"""
    global stream_manager

    try:
        if stream_manager:
            stream_manager.stop_stream()
            stream_manager = None

        logger.info("🛑 Enhanced stream stopped")
        return jsonify({'success': True, 'message': 'Enhanced stream stopped'})

    except Exception as e:
        logger.error(f"Error stopping enhanced stream: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/change_preset', methods=['POST'])
def change_preset():
    """Change detection preset"""
    global stream_manager, current_preset

    try:
        data = request.get_json()
        new_preset = data.get('preset', 'cctv_monitoring')

        if stream_manager and stream_manager.is_system_ready():
            # Stop current stream
            source = stream_manager.source
            stream_manager.stop_stream()

            # Create dengan preset baru
            stream_manager = create_enhanced_stream_manager(source, new_preset)

            if stream_manager.start_stream():
                current_preset = new_preset
                logger.info(f"✅ Preset changed to: {new_preset}")
                return jsonify({
                    'success': True,
                    'message': f'Preset changed to {new_preset}',
                    'preset': new_preset
                })
            else:
                return jsonify({'error': 'Failed to restart with new preset'})
        else:
            current_preset = new_preset
            return jsonify({
                'success': True,
                'message': f'Preset will be used on next start: {new_preset}',
                'preset': new_preset
            })

    except Exception as e:
        logger.error(f"Error changing preset: {e}")
        return jsonify({'error': str(e)})

def generate_enhanced_frames():
    """Generate enhanced frames dengan annotations"""
    global stream_manager

    while True:
        if stream_manager and stream_manager.is_system_ready():
            try:
                # Get annotated frame
                frame = stream_manager.get_frame_with_annotations()

                if frame is not None:
                    # Encode frame untuk streaming
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error generating frame: {e}")
                time.sleep(0.1)
        else:
            # Show placeholder when stream not ready
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Enhanced Hybrid Stream", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Start stream to see detection", (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.5)

@app.route('/enhanced_video_feed')
def enhanced_video_feed():
    """Enhanced video stream endpoint"""
    return Response(generate_enhanced_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def emit_real_time_updates():
    """Emit real-time updates via SocketIO"""
    while True:
        try:
            if stream_manager and stream_manager.is_system_ready():
                # Get current statistics
                stats = stream_manager.get_comprehensive_statistics()

                # Emit enhanced statistics
                socketio.emit('enhanced_stats_update', {
                    'unique_plates': stream_manager.get_unique_plate_count(),
                    'current_visible': stream_manager.get_current_visible_plates(),
                    'detection_methods': stats.get('enhanced_stream_manager', {}).get('detection_methods', {}),
                    'performance': stats.get('enhanced_stream_manager', {}).get('performance', {}),
                    'system_status': stream_manager.get_system_status(),
                    'timestamp': time.time()
                })

                # Emit recent detections
                recent = stream_manager.get_detection_history(limit=5)
                if recent:
                    socketio.emit('new_detections', recent)

            time.sleep(2)  # Update every 2 seconds

        except Exception as e:
            logger.error(f"Error in real-time updates: {e}")
            time.sleep(5)

@app.route('/api/reset_statistics', methods=['POST'])
def reset_statistics():
    """Reset all statistics"""
    if stream_manager:
        stream_manager.reset_statistics()
        logger.info("📊 Statistics reset")
        return jsonify({'success': True, 'message': 'Statistics reset'})
    else:
        return jsonify({'error': 'Stream not running'})

@app.route('/api/system_health')
def get_system_health():
    """Get system health check"""
    if stream_manager:
        status = stream_manager.get_system_status()
        health = {
            'overall': 'healthy' if status['system_ready'] else 'warning',
            'components': status['components_status'],
            'stream_active': status['stream_active'],
            'detection_ready': status['hybrid_detector_ready'],
            'counter_ready': status['stable_counter_ready'],
            'preset': current_preset
        }
        return jsonify(health)
    else:
        return jsonify({
            'overall': 'stopped',
            'message': 'Enhanced stream not running'
        })

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to enhanced stream")
    emit('connection_status', {'status': 'connected', 'type': 'enhanced_hybrid'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from enhanced stream")

@socketio.on('request_stats')
def handle_stats_request():
    """Handle manual stats request"""
    if stream_manager and stream_manager.is_system_ready():
        stats = stream_manager.get_comprehensive_statistics()
        emit('enhanced_stats_update', stats)

if __name__ == '__main__':
    print("🚀 ENHANCED HYBRID HEADLESS STREAM SERVER")
    print("=" * 50)
    print("Features:")
    print("✅ License Plate YOLO Integration")
    print("✅ Enhanced Hybrid Detection")
    print("✅ Ultra-Stable Unique Counting")
    print("✅ Multiple Configuration Presets")
    print("✅ Real-time Performance Monitoring")
    print("✅ Detection Method Breakdown")

    print(f"\n📋 Available Presets:")
    print("• laptop_camera - Laptop testing")
    print("• cctv_monitoring - Production CCTV")
    print("• high_accuracy - Maximum accuracy")
    print("• performance_optimized - Speed optimized")

    print(f"\n🌐 Server starting on http://localhost:5000")
    print("📊 Enhanced dashboard with YOLO statistics")

    # Check if license_plate_yolo.pt exists
    if os.path.exists('license_plate_yolo.pt'):
        print("✅ license_plate_yolo.pt found - YOLO detection enabled")
    else:
        print("⚠️  license_plate_yolo.pt not found - will use OCR fallback")
        print("   Run: python3 download_license_plate_model.py")

    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print(f"\n🛑 Server stopped by user")
        if stream_manager:
            stream_manager.stop_stream()
    except Exception as e:
        print(f"❌ Server error: {e}")
        if stream_manager:
            stream_manager.stop_stream()