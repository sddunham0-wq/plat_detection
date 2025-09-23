#!/usr/bin/env python3
"""
Enhanced License Plate Detection App
Aplikasi utama dengan enhanced detector untuk akurasi tinggi
"""
import cv2
import sys
import time
import os
import base64
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import urllib.parse
import sqlite3
from enhanced_plate_detector import EnhancedPlateDetector

class EnhancedDetectionServer(BaseHTTPRequestHandler):
    def __init__(self, *args, detector=None, **kwargs):
        self.detector = detector
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_main_page()
        elif self.path == '/api/status':
            self.serve_status()
        elif self.path == '/api/stats':
            self.serve_stats()
        elif self.path == '/api/history':
            self.serve_history()
        elif self.path.startswith('/api/'):
            self.serve_api_error(404, "Endpoint not found")
        else:
            self.serve_404()

    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/api/detect':
            self.handle_detection()
        elif self.path == '/api/upload':
            self.handle_upload()
        elif self.path == '/api/camera':
            self.handle_camera()
        else:
            self.serve_api_error(404, "Endpoint not found")

    def serve_main_page(self):
        """Serve main HTML page"""
        html_content = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Plate Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .main-content { padding: 40px; }
        .detection-area {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .input-section, .output-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
        }
        .input-section h3, .output-section h3 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3em;
        }
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover { border-color: #2196F3; background: #f0f8ff; }
        .upload-area.dragover { border-color: #21CBF3; background: #e3f2fd; }
        .btn {
            background: linear-gradient(45deg, #2196F3, #21CBF3);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            margin: 5px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(33,150,243,0.3); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .result-area {
            background: white;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #eee;
            min-height: 300px;
        }
        .detection-result {
            margin: 15px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #2196F3;
        }
        .stats-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-card h4 { font-size: 2em; margin-bottom: 10px; }
        .stat-card p { opacity: 0.9; }
        .loading { display: none; text-align: center; padding: 20px; }
        .loading.show { display: block; }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #eee;
            border-radius: 3px;
            margin: 10px 0;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2196F3, #21CBF3);
            width: 0%;
            transition: width 0.3s ease;
        }
        .camera-section { margin-top: 20px; }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            aspect-ratio: 16/9;
        }
        #videoStream { width: 100%; height: 100%; object-fit: cover; }
        .error { background: #ffebee; color: #c62828; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .success { background: #e8f5e8; color: #2e7d32; padding: 15px; border-radius: 8px; margin: 10px 0; }
        @media (max-width: 768px) {
            .detection-area { grid-template-columns: 1fr; }
            .stats-section { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Plate Detection</h1>
            <p>Sistem Deteksi Plat Nomor dengan Akurasi Tinggi YOLO v8</p>
        </div>

        <div class="main-content">
            <div class="detection-area">
                <div class="input-section">
                    <h3>📤 Input Media</h3>
                    <div class="upload-area" id="uploadArea">
                        <p>🖼️ Drag & Drop gambar atau video di sini</p>
                        <p>atau</p>
                        <button class="btn" onclick="document.getElementById('fileInput').click()">
                            📁 Pilih File
                        </button>
                        <input type="file" id="fileInput" accept="image/*,video/*" style="display: none;">
                    </div>

                    <div class="camera-section">
                        <button class="btn" id="cameraBtn" onclick="toggleCamera()">
                            📹 Start Camera
                        </button>
                        <button class="btn" id="detectBtn" onclick="detectFromCamera()" disabled>
                            🎯 Detect dari Camera
                        </button>
                    </div>

                    <div class="video-container" id="videoContainer" style="display: none;">
                        <video id="videoStream" autoplay muted></video>
                        <canvas id="captureCanvas" style="display: none;"></canvas>
                    </div>
                </div>

                <div class="output-section">
                    <h3>📊 Hasil Deteksi</h3>
                    <div class="result-area" id="resultArea">
                        <p style="text-align: center; color: #666; margin-top: 100px;">
                            Belum ada deteksi. Upload gambar atau gunakan camera untuk memulai.
                        </p>
                    </div>

                    <div class="loading" id="loadingArea">
                        <p>🔄 Memproses deteksi...</p>
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill"></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="stats-section" id="statsSection">
                <div class="stat-card">
                    <h4 id="totalDetections">0</h4>
                    <p>Total Deteksi</p>
                </div>
                <div class="stat-card">
                    <h4 id="avgProcessingTime">0.0s</h4>
                    <p>Avg Processing Time</p>
                </div>
                <div class="stat-card">
                    <h4 id="detectionRate">0%</h4>
                    <p>Detection Rate</p>
                </div>
                <div class="stat-card">
                    <h4 id="systemStatus">Ready</h4>
                    <p>System Status</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let camera = null;
        let cameraActive = false;

        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        function handleFileUpload(file) {
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            showLoading(true);

            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                displayResults(data);
            })
            .catch(error => {
                showLoading(false);
                showError('Error uploading file: ' + error.message);
            });
        }

        async function toggleCamera() {
            const btn = document.getElementById('cameraBtn');
            const container = document.getElementById('videoContainer');
            const video = document.getElementById('videoStream');
            const detectBtn = document.getElementById('detectBtn');

            if (!cameraActive) {
                try {
                    camera = await navigator.mediaDevices.getUserMedia({
                        video: { width: 640, height: 480 }
                    });
                    video.srcObject = camera;
                    container.style.display = 'block';
                    btn.textContent = '⏹️ Stop Camera';
                    detectBtn.disabled = false;
                    cameraActive = true;
                } catch (error) {
                    showError('Camera access denied: ' + error.message);
                }
            } else {
                if (camera) {
                    camera.getTracks().forEach(track => track.stop());
                }
                container.style.display = 'none';
                btn.textContent = '📹 Start Camera';
                detectBtn.disabled = true;
                cameraActive = false;
            }
        }

        function detectFromCamera() {
            if (!cameraActive) return;

            const video = document.getElementById('videoStream');
            const canvas = document.getElementById('captureCanvas');
            const ctx = canvas.getContext('2d');

            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);

            canvas.toBlob((blob) => {
                const formData = new FormData();
                formData.append('file', blob, 'camera_capture.jpg');

                showLoading(true);

                fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    showLoading(false);
                    displayResults(data);
                })
                .catch(error => {
                    showLoading(false);
                    showError('Error detecting from camera: ' + error.message);
                });
            }, 'image/jpeg', 0.8);
        }

        function showLoading(show) {
            const loading = document.getElementById('loadingArea');
            const progressFill = document.getElementById('progressFill');

            if (show) {
                loading.classList.add('show');
                // Simulate progress
                let progress = 0;
                const interval = setInterval(() => {
                    progress += Math.random() * 20;
                    if (progress > 90) progress = 90;
                    progressFill.style.width = progress + '%';
                }, 200);

                setTimeout(() => {
                    clearInterval(interval);
                    progressFill.style.width = '100%';
                }, 3000);
            } else {
                loading.classList.remove('show');
                progressFill.style.width = '0%';
            }
        }

        function displayResults(data) {
            const resultArea = document.getElementById('resultArea');

            if (data.error) {
                resultArea.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                return;
            }

            if (!data.results || data.results.length === 0) {
                resultArea.innerHTML = '<div class="error">🔍 Tidak ada plat nomor terdeteksi</div>';
                return;
            }

            let html = '<div class="success">✅ Deteksi berhasil!</div>';

            data.results.forEach((result, index) => {
                html += `
                    <div class="detection-result">
                        <h4>🚗 Kendaraan ${index + 1}: ${result.vehicle_type}</h4>
                        <p><strong>📋 Plat Nomor:</strong> <span style="font-size: 1.2em; color: #2196F3; font-weight: bold;">${result.plate_text}</span></p>
                        <p><strong>🎯 Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>⚡ Method:</strong> ${result.detection_method}</p>
                        <p><strong>🔧 OCR Config:</strong> ${result.ocr_config}</p>
                        <p><strong>⏱️ Processing Time:</strong> ${result.processing_time.toFixed(3)}s</p>
                    </div>
                `;
            });

            resultArea.innerHTML = html;
            updateStats();
        }

        function showError(message) {
            const resultArea = document.getElementById('resultArea');
            resultArea.innerHTML = `<div class="error">❌ ${message}</div>`;
        }

        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalDetections').textContent = data.total_detections || 0;
                    document.getElementById('avgProcessingTime').textContent = (data.avg_processing_time || 0).toFixed(3) + 's';
                    document.getElementById('detectionRate').textContent = (data.detection_rate || 0).toFixed(1) + '%';
                    document.getElementById('systemStatus').textContent = data.system_status || 'Ready';
                })
                .catch(error => console.error('Stats update failed:', error));
        }

        // Auto-update stats every 10 seconds
        setInterval(updateStats, 10000);

        // Initial stats load
        updateStats();
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def handle_detection(self):
        """Handle detection API request"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # Parse request data
            data = json.loads(post_data.decode())

            # Process detection (implementation depends on data format)
            result = {'status': 'success', 'message': 'Detection endpoint ready'}

            self.send_json_response(result)

        except Exception as e:
            self.serve_api_error(500, f"Detection error: {str(e)}")

    def handle_upload(self):
        """Handle file upload and detection"""
        try:
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                self.serve_api_error(400, "Invalid content type")
                return

            # Parse multipart data (simplified)
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # Extract boundary
            boundary = content_type.split('boundary=')[1].encode()

            # Split parts
            parts = post_data.split(b'--' + boundary)

            image_data = None
            for part in parts:
                if b'Content-Disposition: form-data; name="file"' in part:
                    # Extract image data
                    start = part.find(b'\r\n\r\n') + 4
                    if start > 3:
                        image_data = part[start:]
                        break

            if image_data is None:
                self.serve_api_error(400, "No image data found")
                return

            # Save temporary file
            temp_path = f"temp_upload_{int(time.time())}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(image_data)

            try:
                # Load image dengan OpenCV
                frame = cv2.imread(temp_path)
                if frame is None:
                    self.serve_api_error(400, "Invalid image format")
                    return

                # Process dengan enhanced detector
                start_time = time.time()
                results = self.detector.process_frame_enhanced(frame)
                processing_time = time.time() - start_time

                # Prepare response
                response_data = {
                    'status': 'success',
                    'results': results,
                    'processing_time': processing_time,
                    'image_size': f"{frame.shape[1]}x{frame.shape[0]}",
                    'detections_count': len(results)
                }

                self.send_json_response(response_data)

            finally:
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            self.serve_api_error(500, f"Upload processing error: {str(e)}")

    def handle_camera(self):
        """Handle camera detection request"""
        try:
            # This would integrate with camera feed
            result = {
                'status': 'success',
                'message': 'Camera detection endpoint ready',
                'camera_available': True
            }

            self.send_json_response(result)

        except Exception as e:
            self.serve_api_error(500, f"Camera error: {str(e)}")

    def serve_status(self):
        """Serve system status"""
        try:
            status = {
                'status': 'active',
                'detector_loaded': self.detector is not None,
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time(),
                'version': '2.0.0-enhanced'
            }

            self.send_json_response(status)

        except Exception as e:
            self.serve_api_error(500, f"Status error: {str(e)}")

    def serve_stats(self):
        """Serve performance statistics"""
        try:
            if self.detector:
                stats = self.detector.get_performance_stats()

                # Add database stats
                db_stats = self.get_database_stats()
                stats.update(db_stats)

                stats['system_status'] = 'Active'
            else:
                stats = {
                    'system_status': 'Detector not loaded',
                    'total_detections': 0,
                    'avg_processing_time': 0,
                    'detection_rate': 0
                }

            self.send_json_response(stats)

        except Exception as e:
            self.serve_api_error(500, f"Stats error: {str(e)}")

    def serve_history(self):
        """Serve detection history"""
        try:
            history = self.get_detection_history()
            self.send_json_response({'history': history})

        except Exception as e:
            self.serve_api_error(500, f"History error: {str(e)}")

    def get_database_stats(self):
        """Get statistics from database"""
        try:
            # Use same database as detector
            db_file = 'detected_plates.db'
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            # Total detections
            cursor.execute("SELECT COUNT(*) FROM enhanced_detections")
            total_detections = cursor.fetchone()[0]

            # Today's detections
            cursor.execute("""
                SELECT COUNT(*) FROM enhanced_detections
                WHERE DATE(timestamp) = DATE('now')
            """)
            today_detections = cursor.fetchone()[0]

            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM enhanced_detections")
            avg_confidence = cursor.fetchone()[0] or 0

            conn.close()

            return {
                'total_detections': total_detections,
                'today_detections': today_detections,
                'avg_confidence': avg_confidence
            }

        except Exception:
            return {
                'total_detections': 0,
                'today_detections': 0,
                'avg_confidence': 0
            }

    def get_detection_history(self, limit=50):
        """Get recent detection history"""
        try:
            # Use same database as detector
            db_file = 'detected_plates.db'
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT timestamp, vehicle_type, plate_text, confidence, processing_time
                FROM enhanced_detections
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            history = []
            for row in cursor.fetchall():
                history.append({
                    'timestamp': row[0],
                    'vehicle_type': row[1],
                    'plate_text': row[2],
                    'confidence': row[3],
                    'processing_time': row[4]
                })

            conn.close()
            return history

        except Exception:
            return []

    def send_json_response(self, data):
        """Send JSON response"""
        response = json.dumps(data, indent=2)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(response.encode())

    def serve_api_error(self, code, message):
        """Serve API error response"""
        error_data = {
            'error': True,
            'code': code,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }

        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(error_data).encode())

    def serve_404(self):
        """Serve 404 page"""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<h1>404 - Page Not Found</h1>')

    def log_message(self, format, *args):
        """Override to reduce console spam"""
        return

def create_server_handler(detector):
    """Create server handler with detector instance"""
    def handler(*args, **kwargs):
        return EnhancedDetectionServer(*args, detector=detector, **kwargs)
    return handler

def run_enhanced_app():
    """Run enhanced detection web application"""
    print("🚀 Starting Enhanced License Plate Detection App")
    print("=" * 50)

    # Initialize enhanced detector
    try:
        print("📡 Initializing enhanced detector...")
        detector = EnhancedPlateDetector('enhanced_detection_config.ini')
        print("✅ Enhanced detector loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load detector: {e}")
        print("💡 Make sure enhanced_detection_config.ini exists and YOLO models are available")
        return False

    # Setup web server
    port = 8000
    server_address = ('', port)

    try:
        # Create server dengan detector instance
        handler = create_server_handler(detector)
        httpd = HTTPServer(server_address, handler)

        print(f"🌐 Enhanced detection server starting on port {port}")
        print(f"🔗 Access web interface: http://localhost:{port}")
        print("📊 Features available:")
        print("   • 📤 Image/Video upload detection")
        print("   • 📹 Real-time camera detection")
        print("   • 📈 Performance statistics")
        print("   • 📋 Detection history")
        print("   • 🎯 Multi-model YOLO detection")
        print("   • 🔧 Enhanced OCR with multiple configs")
        print("\n⏹️ Press Ctrl+C to stop the server")

        # Start server
        httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        httpd.server_close()
        return True

    except Exception as e:
        print(f"❌ Server error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced License Plate Detection Application")
    print("High Accuracy YOLO v8 + Optimized OCR System")
    print()

    success = run_enhanced_app()

    if success:
        print("\n✅ Application stopped successfully")
    else:
        print("\n❌ Application failed to start")
        sys.exit(1)