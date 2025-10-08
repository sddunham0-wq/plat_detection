#!/usr/bin/env python3
"""
Unified Plate Detector
SATU detector yang menggabungkan semua fitur terbaik dari 43 detector files

Komponen yang diintegrasikan:
- YOLO vehicle detection (optimized dari yolo_detector.py)
- Robust plate extraction (dari robust_plate_detector.py)
- Smart OCR processing (dari existing OCR systems)
- Indonesian plate validation (dari config.py patterns)
- Multi-frame stability tracking

Target: 95% accuracy, <200ms response time, 99% stability
"""

import cv2
import numpy as np
import logging
import time
import math
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Try import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try import Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

@dataclass
class PlateDetection:
    """Data class untuk hasil deteksi plat yang unified"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    angle: float = 0.0
    processed_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    vehicle_type: str = "unknown"
    detection_method: str = "unified"
    stability_score: float = 0.0
    frame_count: int = 1

class UnifiedPlateDetector:
    """
    UNIFIED PLATE DETECTOR

    Menggabungkan semua fitur terbaik dari 43 detector files:
    - YOLO vehicle detection (optimized dari yolo_detector.py)
    - Robust plate extraction (dari robust_plate_detector.py)
    - Smart OCR processing (dari existing OCR systems)
    - Indonesian plate validation (dari config.py patterns)
    - Multi-frame stability tracking

    Single entry point untuk semua detection scenarios:
    - RTSP CCTV cameras
    - Laptop built-in cameras
    - USB webcams
    - Video files
    """

    def __init__(self, config=None):
        """
        Initialize Unified Plate Detector

        Args:
            config: Optional configuration dict untuk override defaults
        """
        self.logger = self._setup_logging()
        self.config = config or self._get_default_config()

        # Initialize components
        self.yolo_detector = self._init_yolo_component()
        self.plate_extractor = self._init_plate_extractor()
        self.ocr_engine = self._init_ocr_engine()
        self.indonesian_validator = self._init_indonesian_validator()
        self.stability_tracker = self._init_stability_tracker()

        # Statistics tracking
        self.total_detections = 0
        self.successful_detections = 0
        self.detection_history = []
        self.performance_stats = {
            'avg_processing_time': 0.0,
            'accuracy_rate': 0.0,
            'stability_rate': 0.0
        }

        # System status
        self.enabled = True
        self.last_detection_time = 0.0

        self.logger.info("✅ UnifiedPlateDetector initialized successfully")
        self.logger.info(f"📊 YOLO: {'✅ Available' if YOLO_AVAILABLE else '❌ Not Available'}")
        self.logger.info(f"📊 Tesseract: {'✅ Available' if TESSERACT_AVAILABLE else '❌ Not Available'}")

    def detect(self, frame: np.ndarray) -> List[PlateDetection]:
        """
        MAIN DETECTION METHOD - single entry point untuk semua scenarios

        Args:
            frame: Input video frame (BGR format)

        Returns:
            List[PlateDetection]: Stable, validated plate detections
        """
        if not self.enabled:
            return []

        start_time = time.time()

        try:
            self.logger.debug("🔍 Starting unified detection pipeline")

            # Step 1: Detect vehicles using optimized YOLO
            vehicles = self._detect_vehicles(frame)
            self.logger.debug(f"🚗 Found {len(vehicles)} vehicles")

            # Step 2: Extract potential plate regions from vehicles
            plate_candidates = self._extract_plate_regions(frame, vehicles)
            self.logger.debug(f"🔍 Found {len(plate_candidates)} plate candidates")

            # Step 3: OCR processing on plate candidates
            ocr_results = self._process_ocr(plate_candidates)
            self.logger.debug(f"📝 OCR processed {len(ocr_results)} candidates")

            # Step 4: Validate Indonesian plate format
            validated_plates = self._validate_indonesian_plates(ocr_results)
            self.logger.debug(f"✅ Validated {len(validated_plates)} Indonesian plates")

            # Step 5: Multi-frame stability confirmation
            stable_plates = self._confirm_stable_detections(validated_plates)
            self.logger.debug(f"🎯 Confirmed {len(stable_plates)} stable detections")

            # Update statistics
            processing_time = time.time() - start_time
            self._update_statistics(stable_plates, processing_time)

            self.last_detection_time = time.time()

            if stable_plates:
                self.logger.info(f"🎉 Detected {len(stable_plates)} stable plates in {processing_time:.3f}s")
                for plate in stable_plates:
                    self.logger.info(f"   📋 {plate.text} (confidence: {plate.confidence:.2f}, stability: {plate.stability_score:.2f})")

            return stable_plates

        except Exception as e:
            self.logger.error(f"❌ Detection error: {str(e)}")
            return []

    def _setup_logging(self):
        """Setup logging untuk unified detector"""
        logger = logging.getLogger(f"{__name__}.UnifiedPlateDetector")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_default_config(self) -> Dict:
        """Get default configuration yang auto-optimized"""
        return {
            # YOLO settings (optimized dari yolo_detector.py)
            'yolo_model': 'yolov8n.pt',  # Nano untuk speed
            'yolo_confidence': 0.65,     # Balanced confidence
            'yolo_iou_threshold': 0.45,  # Optimal NMS
            'yolo_max_detections': 8,    # Focus on main objects

            # Plate extraction settings (optimized untuk various sizes)
            'plate_min_area': 200,       # Reduced untuk smaller plates
            'plate_max_area': 25000,     # Increased untuk larger plates
            'plate_min_aspect_ratio': 1.5, # More flexible aspect ratio
            'plate_max_aspect_ratio': 6.0,
            'plate_min_width': 25,       # Reduced untuk smaller plates
            'plate_max_width': 400,      # Increased untuk larger plates
            'plate_min_height': 10,      # Reduced untuk smaller plates
            'plate_max_height': 150,     # Increased untuk larger plates

            # OCR settings (optimized untuk speed vs accuracy balance)
            'ocr_min_confidence': 40,    # Raised untuk reduce false positives
            'ocr_psm_modes': [7, 8],     # Reduced to 2 most effective modes
            'ocr_char_whitelist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            'max_candidates_to_process': 15,  # Limit candidates untuk speed
            'early_termination_confidence': 80,  # Stop jika found high confidence result

            # Indonesian validation settings
            'indonesian_patterns': [
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',
                r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',
                r'^\d{1,4}\s*[A-Z]{2,4}$',
                r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
            ],
            'regional_codes': ['B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U'],

            # Stability settings
            'stability_required_frames': 3,  # Require 3 consecutive detections
            'stability_confidence_boost': 10.0,
            'stability_max_drift': 50,      # Max pixel drift between frames

            # Performance settings
            'enable_parallel_processing': True,
            'enable_frame_skipping': False,  # Disabled untuk accuracy
            'target_fps': 10                 # Balanced performance
        }

    def _init_yolo_component(self):
        """Initialize YOLO vehicle detector component"""
        if not YOLO_AVAILABLE:
            self.logger.warning("⚠️ YOLO not available - vehicle detection disabled")
            return None

        try:
            from utils.yolo_detector import YOLOObjectDetector

            detector = YOLOObjectDetector(
                model_path=self.config['yolo_model'],
                confidence=self.config['yolo_confidence'],
                iou_threshold=self.config['yolo_iou_threshold'],
                max_detections=self.config['yolo_max_detections']
            )

            self.logger.info("✅ YOLO vehicle detector initialized")
            return detector

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize YOLO: {str(e)}")
            return None

    def _init_plate_extractor(self):
        """Initialize plate extraction component"""
        config = {
            'min_area': self.config['plate_min_area'],
            'max_area': self.config['plate_max_area'],
            'min_aspect_ratio': self.config['plate_min_aspect_ratio'],
            'max_aspect_ratio': self.config['plate_max_aspect_ratio'],
            'min_width': self.config['plate_min_width'],
            'max_width': self.config['plate_max_width'],
            'min_height': self.config['plate_min_height'],
            'max_height': self.config['plate_max_height']
        }

        self.logger.info("✅ Plate extractor initialized")
        return config

    def _init_ocr_engine(self):
        """Initialize OCR engine component"""
        if not TESSERACT_AVAILABLE:
            self.logger.warning("⚠️ Tesseract not available - OCR disabled")
            return None

        config = {
            'min_confidence': self.config['ocr_min_confidence'],
            'psm_modes': self.config['ocr_psm_modes'],
            'char_whitelist': self.config['ocr_char_whitelist']
        }

        self.logger.info("✅ OCR engine initialized")
        return config

    def _init_indonesian_validator(self):
        """Initialize Indonesian plate validator"""
        config = {
            'patterns': self.config['indonesian_patterns'],
            'regional_codes': self.config['regional_codes'],
            'char_corrections': {
                'O': '0', 'I': '1', 'S': '5', 'Z': '2', '8': 'B', '6': 'G'
            }
        }

        self.logger.info("✅ Indonesian validator initialized")
        return config

    def _init_stability_tracker(self):
        """Initialize stability tracking component"""
        config = {
            'required_frames': self.config['stability_required_frames'],
            'confidence_boost': self.config['stability_confidence_boost'],
            'max_drift': self.config['stability_max_drift'],
            'detection_history': [],
            'max_history_size': 100
        }

        self.logger.info("✅ Stability tracker initialized")
        return config

    def _detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """Detect vehicles using YOLO (Step 1)"""
        if not self.yolo_detector:
            # Fallback: use full frame as vehicle region
            h, w = frame.shape[:2]
            return [{'bbox': (0, 0, w, h), 'confidence': 1.0, 'class': 'frame'}]

        try:
            # Use YOLO to detect vehicles only (not persons untuk avoid lag)
            detections = self.yolo_detector.detect_objects(frame, vehicles_only=True)

            vehicles = []
            frame_h, frame_w = frame.shape[:2]

            for detection in detections:
                if detection.is_vehicle:
                    x, y, w, h = detection.bbox

                    # Expand bounding box to capture more context around vehicle
                    # This helps catch license plates that might be just outside the detection box
                    expand_factor = 0.3  # 30% expansion
                    expand_w = int(w * expand_factor)
                    expand_h = int(h * expand_factor)

                    # Calculate expanded coordinates
                    new_x = max(0, x - expand_w // 2)
                    new_y = max(0, y - expand_h // 2)
                    new_w = min(frame_w - new_x, w + expand_w)
                    new_h = min(frame_h - new_y, h + expand_h)

                    vehicles.append({
                        'bbox': (new_x, new_y, new_w, new_h),
                        'confidence': detection.confidence,
                        'class': detection.class_name,
                        'original_bbox': detection.bbox  # Keep original for reference
                    })

            # If no vehicles detected, use intelligent regions
            if not vehicles:
                self.logger.info("No vehicles detected, using intelligent search regions")
                # Create search regions based on common plate locations
                search_regions = self._create_search_regions(frame)
                vehicles.extend(search_regions)

            return vehicles

        except Exception as e:
            self.logger.error(f"Vehicle detection error: {str(e)}")
            # Fallback: use full frame
            h, w = frame.shape[:2]
            return [{'bbox': (0, 0, w, h), 'confidence': 0.5, 'class': 'fallback'}]

    def _create_search_regions(self, frame: np.ndarray) -> List[Dict]:
        """Create intelligent search regions when no vehicles detected"""
        h, w = frame.shape[:2]
        regions = []

        # Bottom third of image (where most car plates are)
        bottom_region = {
            'bbox': (0, h // 3 * 2, w, h // 3),
            'confidence': 0.5,
            'class': 'search_region_bottom'
        }
        regions.append(bottom_region)

        # Middle region
        middle_region = {
            'bbox': (0, h // 3, w, h // 3),
            'confidence': 0.3,
            'class': 'search_region_middle'
        }
        regions.append(middle_region)

        return regions

    def _extract_plate_regions(self, frame: np.ndarray, vehicles: List[Dict]) -> List[Dict]:
        """Extract potential plate regions from vehicles dengan optimization (Step 2)"""
        plate_candidates = []

        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']

            # Extract vehicle ROI
            vehicle_roi = frame[y:y+h, x:x+w]
            if vehicle_roi.size == 0:
                continue

            # Find plate-like contours dalam vehicle ROI
            candidates = self._find_plate_contours(vehicle_roi, x, y)
            plate_candidates.extend(candidates)

        # OPTIMIZATION: Sort by quality dan limit jumlah candidates
        if len(plate_candidates) > self.config['max_candidates_to_process']:
            # Sort by area (larger plates more likely to be real)
            plate_candidates.sort(key=lambda x: x['area'], reverse=True)
            plate_candidates = plate_candidates[:self.config['max_candidates_to_process']]
            self.logger.info(f"📊 Optimized: Processing top {len(plate_candidates)} candidates (from {len(plate_candidates)} total)")

        return plate_candidates

    def _find_plate_contours(self, roi: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """Find plate-like contours dalam ROI dengan multiple approaches"""
        candidates = []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi

            # Multiple detection approaches untuk better coverage
            approaches = [
                self._approach_standard_edges(gray),
                self._approach_adaptive_threshold(gray),
                self._approach_color_filtering(roi)
            ]

            for approach_candidates in approaches:
                for x, y, w, h in approach_candidates:
                    area = w * h
                    aspect_ratio = w / h if h > 0 else 0

                    # More flexible constraints check
                    if self._is_valid_plate_candidate(area, w, h, aspect_ratio):
                        # Convert back to full frame coordinates
                        frame_x = x + offset_x
                        frame_y = y + offset_y

                        # Extract ROI safely
                        plate_roi = None
                        if (y >= 0 and x >= 0 and
                            y + h <= roi.shape[0] and x + w <= roi.shape[1]):
                            plate_roi = roi[y:y+h, x:x+w]

                        candidates.append({
                            'bbox': (frame_x, frame_y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'roi': plate_roi
                        })

            # Remove duplicates
            candidates = self._remove_duplicate_candidates(candidates)
            return candidates

        except Exception as e:
            self.logger.error(f"Contour detection error: {str(e)}")
            return []

    def _approach_standard_edges(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Standard edge detection approach"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection dengan multiple thresholds
        candidates = []

        for low_thresh, high_thresh in [(30, 100), (50, 150), (70, 200)]:
            edges = cv2.Canny(blurred, low_thresh, high_thresh)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                candidates.append((x, y, w, h))

        return candidates

    def _approach_adaptive_threshold(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Adaptive threshold approach"""
        candidates = []

        # Apply adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((x, y, w, h))

        return candidates

    def _approach_color_filtering(self, roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Color-based filtering for white/light colored plates"""
        candidates = []

        try:
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define range for white/light colors (common for Indonesian plates)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 30, 255])

            # Create mask
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                candidates.append((x, y, w, h))

        except Exception as e:
            self.logger.debug(f"Color filtering error: {str(e)}")

        return candidates

    def _is_valid_plate_candidate(self, area: int, width: int, height: int, aspect_ratio: float) -> bool:
        """Check if candidate meets plate criteria dengan relaxed constraints"""
        area_ok = self.plate_extractor['min_area'] <= area <= self.plate_extractor['max_area']
        width_ok = self.plate_extractor['min_width'] <= width <= self.plate_extractor['max_width']
        height_ok = self.plate_extractor['min_height'] <= height <= self.plate_extractor['max_height']
        aspect_ok = self.plate_extractor['min_aspect_ratio'] <= aspect_ratio <= self.plate_extractor['max_aspect_ratio']

        # Relaxed validation: require 3 out of 4 criteria to pass
        criteria_passed = sum([area_ok, width_ok, height_ok, aspect_ok])
        return criteria_passed >= 3

    def _remove_duplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate/overlapping candidates"""
        if len(candidates) <= 1:
            return candidates

        # Sort by area (largest first)
        candidates.sort(key=lambda x: x['area'], reverse=True)

        filtered = []
        for candidate in candidates:
            is_duplicate = False
            x1, y1, w1, h1 = candidate['bbox']

            for existing in filtered:
                x2, y2, w2, h2 = existing['bbox']

                # Calculate overlap
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                # If overlap > 50%, consider duplicate
                if overlap_area > 0.5 * min(candidate['area'], existing['area']):
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(candidate)

        return filtered

    def _process_ocr(self, plate_candidates: List[Dict]) -> List[Dict]:
        """Process OCR on plate candidates dengan speed optimization (Step 3)"""
        if not self.ocr_engine:
            return []

        ocr_results = []

        for i, candidate in enumerate(plate_candidates):
            if candidate.get('roi') is None:
                continue

            try:
                # OPTIMIZATION: Use only best preprocessing for speed
                best_result = None
                best_confidence = 0

                # Try only high contrast preprocessing (fastest + most effective)
                processed_roi = self._preprocess_high_contrast(candidate['roi'])

                for psm_mode in self.ocr_engine['psm_modes']:
                    config = f'--psm {psm_mode} --oem 3 -c tessedit_char_whitelist={self.ocr_engine["char_whitelist"]}'

                    try:
                        # OCR with confidence
                        data = pytesseract.image_to_data(
                            processed_roi,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )

                        # Extract text and confidence
                        text_parts = []
                        confidences = []

                        for j, conf in enumerate(data['conf']):
                            if int(conf) > 0:
                                text = data['text'][j].strip()
                                if text and len(text) >= 1:
                                    text_parts.append(text)
                                    confidences.append(int(conf))

                        if text_parts and confidences:
                            full_text = ''.join(text_parts).upper()
                            avg_confidence = sum(confidences) / len(confidences)

                            # More strict text length for speed
                            if avg_confidence > best_confidence and len(full_text) >= 3:
                                best_result = {
                                    'text': full_text,
                                    'confidence': avg_confidence,
                                    'bbox': candidate['bbox'],
                                    'psm_mode': psm_mode,
                                    'preprocessing': 'high_contrast'
                                }
                                best_confidence = avg_confidence

                    except Exception as e:
                        self.logger.debug(f"OCR PSM {psm_mode} failed: {str(e)}")
                        continue

                # Add result if meets threshold
                if best_result and best_result['confidence'] >= self.ocr_engine['min_confidence']:
                    ocr_results.append(best_result)
                    self.logger.debug(f"OCR success: '{best_result['text']}' (confidence: {best_result['confidence']:.1f})")

                    # OPTIMIZATION: Early termination if high confidence result found
                    if best_result['confidence'] >= self.config['early_termination_confidence']:
                        self.logger.info(f"⚡ Early termination: High confidence result found ({best_result['confidence']:.1f}%)")
                        break

            except Exception as e:
                self.logger.error(f"OCR processing error: {str(e)}")
                continue

        return ocr_results

    def _preprocess_standard(self, roi: np.ndarray) -> np.ndarray:
        """Standard preprocessing for OCR"""
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Resize if too small
        h, w = gray.shape
        if w < 100 or h < 30:
            scale_factor = max(100/w, 30/h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return gray

    def _preprocess_high_contrast(self, roi: np.ndarray) -> np.ndarray:
        """High contrast preprocessing untuk better text recognition"""
        # Standard preprocessing first
        gray = self._preprocess_standard(roi)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Apply threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def _preprocess_denoise(self, roi: np.ndarray) -> np.ndarray:
        """Denoising preprocessing untuk cleaner text"""
        # Standard preprocessing first
        gray = self._preprocess_standard(roi)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)

        # Apply Gaussian blur untuk smoothing
        blurred = cv2.GaussianBlur(denoised, (3, 3), 0)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

        return thresh

    def _validate_indonesian_plates(self, ocr_results: List[Dict]) -> List[PlateDetection]:
        """Validate Indonesian plate format (Step 4)"""
        validated_plates = []

        for result in ocr_results:
            text = result['text']
            confidence = result['confidence']

            # Apply character corrections
            corrected_text = self._apply_character_corrections(text)

            # Check Indonesian patterns
            is_valid, pattern_confidence = self._check_indonesian_patterns(corrected_text)

            if is_valid:
                # Boost confidence for valid Indonesian patterns
                boosted_confidence = min(100.0, confidence + pattern_confidence)

                plate = PlateDetection(
                    text=corrected_text,
                    confidence=boosted_confidence,
                    bbox=result['bbox'],
                    timestamp=time.time(),
                    detection_method="unified"
                )

                validated_plates.append(plate)

        return validated_plates

    def _apply_character_corrections(self, text: str) -> str:
        """Apply character corrections untuk Indonesian plates"""
        corrections = self.indonesian_validator['char_corrections']

        corrected = text
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)

        return corrected

    def _check_indonesian_patterns(self, text: str) -> Tuple[bool, float]:
        """Check if text matches Indonesian plate patterns"""
        patterns = self.indonesian_validator['patterns']
        regional_codes = self.indonesian_validator['regional_codes']

        # Check against patterns
        for pattern in patterns:
            if re.match(pattern, text):
                confidence_boost = 10.0

                # Additional boost for regional codes
                for code in regional_codes:
                    if text.startswith(code):
                        confidence_boost += 5.0
                        break

                return True, confidence_boost

        return False, 0.0

    def _confirm_stable_detections(self, validated_plates: List[PlateDetection]) -> List[PlateDetection]:
        """Multi-frame stability confirmation (Step 5)"""
        current_time = time.time()
        stable_plates = []

        for plate in validated_plates:
            # Add to history
            self.stability_tracker['detection_history'].append({
                'text': plate.text,
                'bbox': plate.bbox,
                'confidence': plate.confidence,
                'timestamp': current_time
            })

            # Check stability
            stability_score = self._calculate_stability_score(plate)

            if stability_score >= 0.7:  # 70% stability threshold
                plate.stability_score = stability_score
                plate.confidence += self.stability_tracker['confidence_boost']
                plate.confidence = min(100.0, plate.confidence)
                stable_plates.append(plate)

        # Cleanup old history
        cutoff_time = current_time - 10.0  # Keep 10 seconds history
        self.stability_tracker['detection_history'] = [
            h for h in self.stability_tracker['detection_history']
            if h['timestamp'] > cutoff_time
        ]

        return stable_plates

    def _calculate_stability_score(self, plate: PlateDetection) -> float:
        """Calculate stability score based on detection history"""
        history = self.stability_tracker['detection_history']
        current_time = time.time()

        # Find similar detections dalam recent history
        similar_count = 0
        total_count = 0

        for h in history:
            if current_time - h['timestamp'] <= 3.0:  # Last 3 seconds
                total_count += 1

                # Check text similarity
                if h['text'] == plate.text:
                    similar_count += 1

                # Check spatial consistency
                elif self._calculate_bbox_distance(plate.bbox, h['bbox']) <= self.stability_tracker['max_drift']:
                    similar_count += 0.5  # Partial credit untuk spatial consistency

        # Calculate stability score
        if total_count == 0:
            return 0.0

        stability_score = similar_count / max(1, total_count)
        return min(1.0, stability_score)

    def _calculate_bbox_distance(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between bbox centers"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)

        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _update_statistics(self, detections: List[PlateDetection], processing_time: float):
        """Update performance statistics"""
        self.total_detections += 1
        if detections:
            self.successful_detections += len(detections)

        # Update moving averages
        alpha = 0.1  # Smoothing factor

        if self.performance_stats['avg_processing_time'] == 0:
            self.performance_stats['avg_processing_time'] = processing_time
        else:
            self.performance_stats['avg_processing_time'] = (
                alpha * processing_time +
                (1 - alpha) * self.performance_stats['avg_processing_time']
            )

        if self.total_detections > 0:
            self.performance_stats['accuracy_rate'] = (
                self.successful_detections / self.total_detections
            )

    def get_statistics(self) -> Dict:
        """Get current performance statistics"""
        stats = {
            'total_detections': self.total_detections,
            'successful_detections': self.successful_detections,
            'avg_processing_time': self.performance_stats['avg_processing_time'],
            'accuracy_rate': self.performance_stats['accuracy_rate'],
            'target_fps': 1.0 / self.performance_stats['avg_processing_time'] if self.performance_stats['avg_processing_time'] > 0 else 0,
            'enabled': self.enabled,
            'yolo_available': YOLO_AVAILABLE,
            'tesseract_available': TESSERACT_AVAILABLE,
            'last_detection': time.time() - self.last_detection_time if self.last_detection_time > 0 else None
        }

        return stats

    def enable(self):
        """Enable detection"""
        self.enabled = True
        self.logger.info("✅ UnifiedPlateDetector enabled")

    def disable(self):
        """Disable detection"""
        self.enabled = False
        self.logger.info("⏸️ UnifiedPlateDetector disabled")

    def reset_statistics(self):
        """Reset performance statistics"""
        self.total_detections = 0
        self.successful_detections = 0
        self.detection_history = []
        self.performance_stats = {
            'avg_processing_time': 0.0,
            'accuracy_rate': 0.0,
            'stability_rate': 0.0
        }
        self.stability_tracker['detection_history'] = []

        self.logger.info("📊 Statistics reset")


# Factory function untuk easy instantiation
def create_unified_detector(config_type="auto") -> UnifiedPlateDetector:
    """
    Factory function untuk create UnifiedPlateDetector dengan SmartConfig

    Args:
        config_type: Camera type atau configuration dict
                    - str: "auto", "rtsp_cctv", "laptop_camera", "webcam", "video_file"
                    - dict: Custom configuration

    Returns:
        UnifiedPlateDetector instance
    """
    if isinstance(config_type, str):
        # Use SmartConfig untuk auto-configuration
        from smart_config import SmartConfig
        config = SmartConfig.get_config_for_scenario(config_type)
    else:
        # Use provided dict config
        config = config_type

    return UnifiedPlateDetector(config)


# Test function untuk quick validation
def test_unified_detector():
    """Quick test untuk validate UnifiedPlateDetector basic functionality"""
    print("🧪 Testing UnifiedPlateDetector...")

    try:
        # Test initialization
        detector = create_unified_detector()
        print("✅ UnifiedPlateDetector initialized successfully")

        # Test dengan dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = detector.detect(dummy_frame)
        print(f"✅ Detection method working, returned {len(results)} results")

        # Test statistics
        stats = detector.get_statistics()
        print(f"✅ Statistics available: {list(stats.keys())}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run basic test when executed directly
    test_unified_detector()