#!/usr/bin/env python3
"""
Enhanced Plate Detector - High Accuracy YOLO + Optimized OCR
Fokus pada akurasi tinggi untuk deteksi kendaraan dan plat nomor
"""
import cv2
import numpy as np
import time
from datetime import datetime
import configparser
import os
import sys
from ultralytics import YOLO
import pytesseract
from PIL import Image
import re
import sqlite3
import threading
from collections import deque
import logging

class EnhancedPlateDetector:
    def __init__(self, config_path='enhanced_detection_config.ini'):
        """
        Initialize Enhanced Plate Detector dengan konfigurasi optimized
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Load enhanced YOLO models
        self.load_enhanced_models()

        # Enhanced detection parameters
        self.setup_enhanced_parameters()

        # ROI configurations untuk berbagai jenis kendaraan
        self.setup_roi_configurations()

        # OCR configurations
        self.setup_ocr_configurations()

        # Detection history untuk stabilitas
        self.detection_history = deque(maxlen=10)

        # Database untuk logging (using existing project database)
        self.setup_database()

        # Setup logging
        self.setup_logging()

    def load_enhanced_models(self):
        """Load multiple YOLO models untuk enhanced accuracy"""
        try:
            # Primary model (speed-accuracy balance)
            primary_model = self.config.get('YOLO_ENHANCED', 'PRIMARY_MODEL', fallback='yolov8n.pt')
            self.primary_model = YOLO(primary_model)

            # Secondary model (higher accuracy)
            try:
                secondary_model = self.config.get('YOLO_ENHANCED', 'SECONDARY_MODEL', fallback='yolov8s.pt')
                self.secondary_model = YOLO(secondary_model)
                self.use_secondary = True
            except:
                self.use_secondary = False

            # Tertiary model (maximum accuracy)
            try:
                tertiary_model = self.config.get('YOLO_ENHANCED', 'TERTIARY_MODEL', fallback='yolov8m.pt')
                self.tertiary_model = YOLO(tertiary_model)
                self.use_tertiary = True
            except:
                self.use_tertiary = False

            print(f"✅ Enhanced YOLO models loaded successfully")
            print(f"   Primary: {primary_model}")
            if self.use_secondary:
                print(f"   Secondary: {secondary_model}")
            if self.use_tertiary:
                print(f"   Tertiary: {tertiary_model}")

        except Exception as e:
            print(f"❌ Error loading YOLO models: {e}")
            sys.exit(1)

    def setup_enhanced_parameters(self):
        """Setup enhanced detection parameters"""
        # Enhanced YOLO parameters
        self.enhanced_conf_threshold = float(self.config.get('YOLO_ENHANCED', 'ENHANCED_CONF_THRESHOLD', fallback='0.25'))
        self.enhanced_iou_threshold = float(self.config.get('YOLO_ENHANCED', 'ENHANCED_IOU_THRESHOLD', fallback='0.45'))
        self.max_detections = int(self.config.get('YOLO_ENHANCED', 'MAX_DETECTIONS', fallback='100'))

        # Multi-pass detection
        self.use_multi_pass = self.config.getboolean('YOLO_ENHANCED', 'USE_MULTI_PASS', fallback=True)
        self.confidence_boost = float(self.config.get('YOLO_ENHANCED', 'CONFIDENCE_BOOST', fallback='0.1'))

        # Enhanced filtering
        self.min_plate_area = int(self.config.get('DETECTION', 'MIN_PLATE_AREA', fallback='500'))
        self.max_plate_area = int(self.config.get('DETECTION', 'MAX_PLATE_AREA', fallback='50000'))

        # Vehicle classes (COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def setup_roi_configurations(self):
        """Setup ROI configurations untuk berbagai jenis kendaraan"""
        self.roi_configs = {
            'motorcycle': {
                'front_regions': [(0.2, 0.3, 0.8, 0.8)],  # (x1, y1, x2, y2) normalized
                'rear_regions': [(0.2, 0.4, 0.8, 0.9)],
                'aspect_ratio_range': (2.0, 5.0),
                'size_range': (0.02, 0.15)  # relative to frame size
            },
            'car': {
                'front_regions': [(0.15, 0.4, 0.85, 0.85)],
                'rear_regions': [(0.15, 0.5, 0.85, 0.95)],
                'aspect_ratio_range': (1.8, 4.5),
                'size_range': (0.03, 0.25)
            },
            'bus': {
                'front_regions': [(0.1, 0.3, 0.9, 0.8)],
                'rear_regions': [(0.1, 0.4, 0.9, 0.9)],
                'aspect_ratio_range': (1.5, 4.0),
                'size_range': (0.04, 0.3)
            },
            'truck': {
                'front_regions': [(0.1, 0.35, 0.9, 0.85)],
                'rear_regions': [(0.1, 0.45, 0.9, 0.95)],
                'aspect_ratio_range': (1.5, 4.0),
                'size_range': (0.04, 0.35)
            }
        }

    def setup_ocr_configurations(self):
        """Setup multiple OCR configurations untuk enhanced accuracy"""
        self.ocr_configs = [
            # Config 1: Standard Indonesian plate
            {
                'config': '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'lang': 'eng',
                'name': 'standard_plate'
            },
            # Config 2: Single text line
            {
                'config': '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'lang': 'eng',
                'name': 'single_line'
            },
            # Config 3: Single word
            {
                'config': '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'lang': 'eng',
                'name': 'single_word'
            },
            # Config 4: Raw line without word segmentation
            {
                'config': '--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'lang': 'eng',
                'name': 'raw_line'
            },
            # Config 5: Indonesian language support
            {
                'config': '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                'lang': 'ind+eng',
                'name': 'indonesian_support'
            }
        ]

    def setup_database(self):
        """Setup database untuk logging detection results"""
        try:
            # Use existing project database
            db_file = self.config.get('DATABASE', 'DATABASE_FILE', fallback='detected_plates.db')
            self.conn = sqlite3.connect(db_file, check_same_thread=False)
            self.cursor = self.conn.cursor()

            # Create enhanced table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    vehicle_type TEXT,
                    plate_text TEXT,
                    confidence REAL,
                    detection_method TEXT,
                    processing_time REAL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    ocr_config TEXT,
                    enhancement_applied TEXT
                )
            ''')
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ Database setup error: {e}")

    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def preprocess_image_enhanced(self, image, method='standard'):
        """
        Enhanced image preprocessing dengan multiple methods
        """
        if method == 'clahe':
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        elif method == 'bilateral':
            # Bilateral filtering untuk noise reduction
            return cv2.bilateralFilter(image, 9, 75, 75)

        elif method == 'morph':
            # Morphological operations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            return cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)

        elif method == 'gamma':
            # Gamma correction
            gamma = 1.2
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)

        elif method == 'adaptive':
            # Adaptive thresholding
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)

        else:
            # Standard preprocessing
            return image

    def detect_vehicles_enhanced(self, frame):
        """
        Enhanced vehicle detection dengan multiple models
        """
        detections = []

        # Primary model detection
        results_primary = self.primary_model(frame, conf=self.enhanced_conf_threshold,
                                           iou=self.enhanced_iou_threshold,
                                           max_det=self.max_detections)

        for result in results_primary:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id in self.vehicle_classes:
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        detection = {
                            'bbox': (x1, y1, x2-x1, y2-y1),
                            'confidence': confidence,
                            'class_id': class_id,
                            'method': 'primary',
                            'vehicle_type': self.get_vehicle_type(class_id)
                        }
                        detections.append(detection)

        # Secondary model detection (if available and needed)
        if self.use_secondary and len(detections) < 3:
            results_secondary = self.secondary_model(frame, conf=self.enhanced_conf_threshold-0.1,
                                                   iou=self.enhanced_iou_threshold)

            for result in results_secondary:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id in self.vehicle_classes:
                            confidence = float(box.conf[0]) + self.confidence_boost
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Check if this detection is not duplicate
                            bbox = (x1, y1, x2-x1, y2-y1)
                            if not self.is_duplicate_detection(bbox, detections):
                                detection = {
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'method': 'secondary',
                                    'vehicle_type': self.get_vehicle_type(class_id)
                                }
                                detections.append(detection)

        return detections

    def get_vehicle_type(self, class_id):
        """Get vehicle type from class ID"""
        vehicle_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return vehicle_map.get(class_id, 'unknown')

    def is_duplicate_detection(self, bbox, existing_detections, threshold=0.5):
        """Check if detection is duplicate using IoU"""
        x1, y1, w1, h1 = bbox

        for detection in existing_detections:
            x2, y2, w2, h2 = detection['bbox']

            # Calculate IoU
            intersection_x1 = max(x1, x2)
            intersection_y1 = max(y1, y2)
            intersection_x2 = min(x1 + w1, x2 + w2)
            intersection_y2 = min(y1 + h1, y2 + h2)

            if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                box1_area = w1 * h1
                box2_area = w2 * h2
                union_area = box1_area + box2_area - intersection_area

                iou = intersection_area / union_area if union_area > 0 else 0

                if iou > threshold:
                    return True

        return False

    def extract_plate_regions_enhanced(self, frame, vehicle_detection):
        """
        Enhanced plate region extraction dengan ROI optimization
        """
        x, y, w, h = vehicle_detection['bbox']
        vehicle_type = vehicle_detection['vehicle_type']
        vehicle_roi = frame[y:y+h, x:x+w]

        plate_regions = []

        # Get ROI configuration untuk vehicle type
        roi_config = self.roi_configs.get(vehicle_type, self.roi_configs['car'])

        # Extract front regions
        for region in roi_config['front_regions']:
            x1, y1, x2, y2 = region
            roi_x1 = int(x1 * w)
            roi_y1 = int(y1 * h)
            roi_x2 = int(x2 * w)
            roi_y2 = int(y2 * h)

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = vehicle_roi[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi.size > 0:
                    plate_regions.append({
                        'region': roi,
                        'position': (x + roi_x1, y + roi_y1),
                        'size': (roi_x2 - roi_x1, roi_y2 - roi_y1),
                        'type': 'front'
                    })

        # Extract rear regions
        for region in roi_config['rear_regions']:
            x1, y1, x2, y2 = region
            roi_x1 = int(x1 * w)
            roi_y1 = int(y1 * h)
            roi_x2 = int(x2 * w)
            roi_y2 = int(y2 * h)

            if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                roi = vehicle_roi[roi_y1:roi_y2, roi_x1:roi_x2]
                if roi.size > 0:
                    plate_regions.append({
                        'region': roi,
                        'position': (x + roi_x1, y + roi_y1),
                        'size': (roi_x2 - roi_x1, roi_y2 - roi_y1),
                        'type': 'rear'
                    })

        return plate_regions

    def detect_plates_in_region(self, region_data, vehicle_type):
        """
        Detect plates dalam region dengan enhanced filtering
        """
        region = region_data['region']
        position = region_data['position']

        # Multiple preprocessing methods
        preprocessing_methods = ['standard', 'clahe', 'bilateral', 'morph', 'adaptive']
        best_plates = []

        for method in preprocessing_methods:
            processed_region = self.preprocess_image_enhanced(region, method)

            # Convert to grayscale untuk contour detection
            gray = cv2.cvtColor(processed_region, cv2.COLOR_BGR2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get ROI configuration
            roi_config = self.roi_configs.get(vehicle_type, self.roi_configs['car'])
            aspect_ratio_range = roi_config['aspect_ratio_range']
            size_range = roi_config['size_range']

            region_area = region.shape[0] * region.shape[1]

            for contour in contours:
                area = cv2.contourArea(contour)

                # Size filtering
                relative_area = area / region_area
                if relative_area < size_range[0] or relative_area > size_range[1]:
                    continue

                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Aspect ratio filtering
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                    continue

                # Extract plate candidate
                plate_img = processed_region[y:y+h, x:x+w]

                if plate_img.size > 0:
                    plate_data = {
                        'image': plate_img,
                        'bbox': (position[0] + x, position[1] + y, w, h),
                        'confidence': self.calculate_plate_confidence(plate_img, aspect_ratio, relative_area),
                        'preprocessing': method
                    }
                    best_plates.append(plate_data)

        # Sort by confidence and return top candidates
        best_plates.sort(key=lambda x: x['confidence'], reverse=True)
        return best_plates[:3]  # Return top 3 candidates

    def calculate_plate_confidence(self, plate_img, aspect_ratio, relative_area):
        """
        Calculate confidence score untuk plate candidate
        """
        confidence = 0.0

        # Aspect ratio score (ideal untuk Indonesian plates: 3.5-4.5)
        if 3.0 <= aspect_ratio <= 5.0:
            confidence += 0.3

        # Size score
        if 0.02 <= relative_area <= 0.2:
            confidence += 0.2

        # Texture analysis
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        if 0.1 <= edge_density <= 0.4:
            confidence += 0.2

        # Contrast
        contrast = gray.std()
        if contrast > 30:
            confidence += 0.15

        # Horizontal lines (typical untuk plate text)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / (gray.shape[0] * gray.shape[1])
        if horizontal_score > 0.05:
            confidence += 0.15

        return confidence

    def perform_ocr_enhanced(self, plate_img):
        """
        Enhanced OCR dengan multiple configurations
        """
        best_result = None
        best_confidence = 0.0

        # Resize image jika terlalu kecil
        height, width = plate_img.shape[:2]
        if height < 40 or width < 120:
            scale_factor = max(40 / height, 120 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            plate_img = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Try each OCR configuration
        for config in self.ocr_configs:
            try:
                # Convert to PIL Image
                pil_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))

                # Perform OCR
                text = pytesseract.image_to_string(
                    pil_img,
                    lang=config['lang'],
                    config=config['config']
                ).strip()

                # Clean text
                cleaned_text = self.clean_plate_text(text)

                if cleaned_text:
                    # Calculate confidence based on text pattern
                    confidence = self.calculate_text_confidence(cleaned_text)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            'text': cleaned_text,
                            'confidence': confidence,
                            'config': config['name']
                        }

            except Exception as e:
                self.logger.warning(f"OCR error with config {config['name']}: {e}")
                continue

        return best_result

    def clean_plate_text(self, text):
        """Clean dan validate plate text"""
        if not text:
            return ""

        # Remove unwanted characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Indonesian plate patterns
        patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',  # Standard format
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',        # No spaces
            r'^\d{1,4}[A-Z]{1,3}$',                   # Numbers + letters
            r'^[A-Z]{1,2}\d{1,4}$',                   # Letters + numbers
        ]

        # Check patterns
        for pattern in patterns:
            if re.match(pattern, cleaned):
                return cleaned

        # Return if reasonable length
        if 3 <= len(cleaned) <= 10:
            return cleaned

        return ""

    def calculate_text_confidence(self, text):
        """Calculate confidence score untuk OCR text"""
        if not text:
            return 0.0

        confidence = 0.0

        # Length score
        if 5 <= len(text) <= 8:
            confidence += 0.3
        elif 3 <= len(text) <= 10:
            confidence += 0.2

        # Pattern score untuk Indonesian plates
        if re.match(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$', text):
            confidence += 0.4
        elif re.match(r'^\d{1,4}[A-Z]{1,3}$', text):
            confidence += 0.3

        # Character variety
        has_letters = bool(re.search(r'[A-Z]', text))
        has_numbers = bool(re.search(r'\d', text))
        if has_letters and has_numbers:
            confidence += 0.2

        # No repeated characters
        if len(set(text)) > len(text) * 0.6:
            confidence += 0.1

        return min(confidence, 1.0)

    def save_detection_result(self, result):
        """Save detection result ke database"""
        try:
            self.cursor.execute('''
                INSERT INTO enhanced_detections
                (timestamp, vehicle_type, plate_text, confidence, detection_method,
                 processing_time, bbox_x, bbox_y, bbox_w, bbox_h, ocr_config, enhancement_applied)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['timestamp'],
                result['vehicle_type'],
                result['plate_text'],
                result['confidence'],
                result['detection_method'],
                result['processing_time'],
                result['bbox'][0],
                result['bbox'][1],
                result['bbox'][2],
                result['bbox'][3],
                result.get('ocr_config', ''),
                result.get('enhancement_applied', '')
            ))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database save error: {e}")

    def process_frame_enhanced(self, frame):
        """
        Main method untuk processing frame dengan enhanced detection
        """
        start_time = time.time()
        results = []

        try:
            # Detect vehicles
            vehicle_detections = self.detect_vehicles_enhanced(frame)

            for vehicle_detection in vehicle_detections:
                # Extract plate regions
                plate_regions = self.extract_plate_regions_enhanced(frame, vehicle_detection)

                # Detect plates dalam each region
                for region_data in plate_regions:
                    plate_candidates = self.detect_plates_in_region(region_data, vehicle_detection['vehicle_type'])

                    for plate_data in plate_candidates:
                        # Perform OCR
                        ocr_result = self.perform_ocr_enhanced(plate_data['image'])

                        if ocr_result and ocr_result['text']:
                            # Create result
                            result = {
                                'timestamp': datetime.now().isoformat(),
                                'vehicle_type': vehicle_detection['vehicle_type'],
                                'vehicle_bbox': vehicle_detection['bbox'],
                                'plate_text': ocr_result['text'],
                                'plate_bbox': plate_data['bbox'],
                                'confidence': (plate_data['confidence'] + ocr_result['confidence']) / 2,
                                'detection_method': vehicle_detection['method'],
                                'ocr_config': ocr_result['config'],
                                'enhancement_applied': plate_data['preprocessing'],
                                'processing_time': time.time() - start_time
                            }

                            results.append(result)

                            # Save to database
                            self.save_detection_result({
                                'timestamp': result['timestamp'],
                                'vehicle_type': result['vehicle_type'],
                                'plate_text': result['plate_text'],
                                'confidence': result['confidence'],
                                'detection_method': result['detection_method'],
                                'processing_time': result['processing_time'],
                                'bbox': result['plate_bbox'],
                                'ocr_config': result['ocr_config'],
                                'enhancement_applied': result['enhancement_applied']
                            })

            # Update detection history
            self.detection_history.append({
                'timestamp': time.time(),
                'vehicle_count': len(vehicle_detections),
                'plate_count': len(results),
                'processing_time': time.time() - start_time
            })

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")

        return results

    def draw_enhanced_results(self, frame, results):
        """Draw enhanced detection results pada frame"""
        output_frame = frame.copy()

        for result in results:
            # Draw vehicle bbox
            vx, vy, vw, vh = result['vehicle_bbox']
            cv2.rectangle(output_frame, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 2)

            # Draw plate bbox
            px, py, pw, ph = result['plate_bbox']
            cv2.rectangle(output_frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)

            # Add text
            label = f"{result['vehicle_type']}: {result['plate_text']}"
            confidence_text = f"Conf: {result['confidence']:.2f}"

            cv2.putText(output_frame, label, (px, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(output_frame, confidence_text, (px, py + ph + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Add performance info
        if self.detection_history:
            latest = self.detection_history[-1]
            perf_text = f"Processing: {latest['processing_time']:.2f}s | Vehicles: {latest['vehicle_count']} | Plates: {latest['plate_count']}"
            cv2.putText(output_frame, perf_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output_frame

    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.detection_history:
            return {}

        processing_times = [h['processing_time'] for h in self.detection_history]
        vehicle_counts = [h['vehicle_count'] for h in self.detection_history]
        plate_counts = [h['plate_count'] for h in self.detection_history]

        return {
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'min_processing_time': np.min(processing_times),
            'avg_vehicle_count': np.mean(vehicle_counts),
            'avg_plate_count': np.mean(plate_counts),
            'detection_rate': np.mean(plate_counts) / max(np.mean(vehicle_counts), 1) * 100,
            'total_frames': len(self.detection_history)
        }

    def get_statistics(self):
        """Get statistics (alias for get_performance_stats for compatibility)"""
        return self.get_performance_stats()

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except:
            pass

if __name__ == "__main__":
    # Test enhanced detector
    detector = EnhancedPlateDetector()

    # Test dengan webcam
    cap = cv2.VideoCapture(0)

    print("🚀 Enhanced Plate Detector Started")
    print("📊 Press 's' to show statistics")
    print("❌ Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = detector.process_frame_enhanced(frame)

        # Draw results
        output_frame = detector.draw_enhanced_results(frame, results)

        # Show frame
        cv2.imshow('Enhanced Plate Detection', output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            stats = detector.get_performance_stats()
            print(f"\n📊 Performance Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Enhanced detector stopped")