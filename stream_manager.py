"""
Stream Manager untuk Headless Video Processing
Menghandle video stream dan detection untuk web browser
"""

import cv2
import base64
import threading
import time
import logging
import numpy as np
import queue
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from utils.video_stream import VideoStream, RTSPStream, WebcamStream
from utils.plate_detector import LicensePlateDetector, PlateDetection
from utils.robust_plate_detector import RobustPlateDetector
from utils.yolo_plate_detector import YOLOPlateDetector
from utils.tracking_manager import TrackingManager
from utils.person_detector import PersonDetector, PersonDetection  # NEW: Person detection
from utils.yolo_detector import YOLOObjectDetector, ObjectDetection  # NEW: Vehicle detection
from utils.plate_validator import PlateValidator  # NEW: Indonesian plate validation
from database import PlateDatabase
from config import TrackingConfig, PersonDetectionConfig, MySQLConfig  # NEW: Person detection config + MySQL config

# MySQL Integration (optional)
try:
    from mysql_database import MySQLPlateDatabase
    from access_controller import AccessController
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

@dataclass
class StreamFrame:
    """Frame data untuk web streaming"""
    image_base64: str
    timestamp: float
    frame_id: int
    detections: list
    object_detections: list
    person_detections: list  # NEW: Person detections
    fps: float
    processing_time: float

class HeadlessStreamManager:
    """
    Manager untuk headless video streaming ke browser
    """
    
    def __init__(self, source: str, database: PlateDatabase = None, enable_yolo: bool = False, enable_tracking: bool = True, enable_person_detection: bool = None, use_mysql: bool = None, enable_access_control: bool = None):
        """
        Initialize stream manager

        Args:
            source: Video source (RTSP URL, webcam index, file)
            database: Database instance untuk save results (SQLite)
            enable_yolo: Deprecated - no longer used (kept for compatibility)
            enable_tracking: Enable object tracking system
            enable_person_detection: Enable person detection (None = use config default)
            use_mysql: Enable MySQL access control (None = use config default)
            enable_access_control: Enable access control system (None = use config default)
        """
        self.source = source
        self.database = database or PlateDatabase()

        # Setup logging FIRST (before any component initialization)
        self.logger = logging.getLogger(__name__)

        # MySQL Integration (Hybrid Mode: SQLite + MySQL)
        self.use_mysql = use_mysql if use_mysql is not None else (MYSQL_AVAILABLE and MySQLConfig.USE_MYSQL_DATABASE)
        self.enable_access_control = enable_access_control if enable_access_control is not None else (self.use_mysql and MySQLConfig.ENABLE_ACCESS_CONTROL)
        self.mysql_db = None
        self.access_controller = None

        # Components
        self.video_stream = None
        self.plate_detector = YOLOPlateDetector(
            confidence=0.55,  # ✅ CRITICAL FIX: Raised from 0.2 to 0.55 (55%) - detect ONLY high-quality plates
            streaming_mode=True,
            enable_bbox_refinement=True  # ✅ ENABLED: Bbox refinement for more precise plate crop (better OCR)
        )  # ✅ YOLO PLATE DETECTION (Deep Learning, 55% confidence - quality over quantity)
        self.tracking_manager = None
        self.tracking_enabled = enable_tracking and TrackingConfig.ENABLE_TRACKING

        # NEW: Person detection (ISOLATED dari plate detection)
        self.person_detector = None
        self.person_detection_enabled = enable_person_detection if enable_person_detection is not None else PersonDetectionConfig.ENABLE_PERSON_DETECTION
        if self.person_detection_enabled:
            try:
                self.person_detector = PersonDetector(
                    model_path=PersonDetectionConfig.PERSON_YOLO_MODEL,
                    confidence=PersonDetectionConfig.PERSON_CONFIDENCE,
                    max_detections=PersonDetectionConfig.PERSON_MAX_DETECTIONS
                )
                if not self.person_detector.is_enabled():
                    self.logger.warning("⚠️ Person detector created but not enabled")
                    self.person_detection_enabled = False
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize person detector: {e}")
                self.person_detector = None
                self.person_detection_enabled = False

        # NEW: Vehicle detection (car, motorcycle, bus, truck)
        self.vehicle_detector = None
        self.vehicle_detection_enabled = False  # ✅ DISABLED - Focus on plate detection only (2x faster)
        try:
            self.vehicle_detector = YOLOObjectDetector(
                model_path='yolov8n.pt',
                confidence=0.005,  # 0.5% - match actual detection levels (was 0.4 = 40%)
                max_detections=10
            )
            self.logger.info("✅ Vehicle detector initialized (disabled by default)")
        except Exception as e:
            self.logger.warning(f"⚠️ Vehicle detector initialization failed: {e}")
            self.vehicle_detector = None

        # Threading
        self.running = False
        self.stream_thread = None
        self.lock = threading.Lock()

        # Background database save queue (non-blocking saves)
        self.db_save_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.db_save_thread = None
        self.db_worker_running = False

        # Current frame data
        self.current_frame = None
        self.frame_callbacks = []
        self.detection_callbacks = []

        # Cache last detections for frame skipping (persist bbox across frames)
        self.last_plate_detections = []
        self.last_vehicle_detections = []
        self.last_person_detections = []

        # NEW: Time-based duplicate filtering (5 second window)
        self.recent_detections = {}  # {plate_text: last_timestamp}
        self.duplicate_window = 5.0  # 5 seconds

        # NEW: Text stability system for consistent bounding box labels
        self.stable_plate_texts = {}  # {bbox_key: locked_text}
        self.plate_text_votes = {}    # {bbox_key: {text: vote_count}}
        self.bbox_vote_threshold = 3  # Lock text after 3 consistent votes
        self.newly_locked_plates = set()  # Track plates that just got locked (for frontend notification)

        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,  # Total plat nomor yang berhasil dibaca (akumulatif)
            'total_detection_events': 0,  # Total detection events (akumulatif)
            'total_persons_detected': 0,  # NEW: Total person detections (akumulatif)
            'fps': 0.0,
            'avg_processing_time': 0.0,
            'last_detection_time': None,
            'start_time': time.time(),
            'tracking_enabled': False,
            'person_detection_enabled': self.person_detection_enabled  # NEW
        }

        # Initialize MySQL (using singleton pattern)
        if self.use_mysql and MYSQL_AVAILABLE:
            try:
                self.mysql_db = MySQLPlateDatabase.get_instance()
                if self.enable_access_control:
                    self.access_controller = AccessController(self.mysql_db)
                    self.logger.info("✅ MySQL Access Control enabled (singleton instance)")
                else:
                    self.logger.info("✅ MySQL database enabled (logging only, singleton instance)")
            except Exception as e:
                self.logger.warning(f"⚠️ MySQL initialization failed: {str(e)}")
                self.logger.info("   Falling back to SQLite only")
                self.use_mysql = False
                self.enable_access_control = False
        elif self.use_mysql and not MYSQL_AVAILABLE:
            self.logger.warning("⚠️ MySQL requested but not available. Install: pip install pymysql python-dotenv")
            self.use_mysql = False
            self.enable_access_control = False

        # Initialize tracking manager
        if self.tracking_enabled:
            self.logger.info("Initializing tracking system...")
            tracking_config = {
                'max_disappeared': TrackingConfig.MAX_DISAPPEARED_FRAMES,
                'max_distance': TrackingConfig.MAX_TRACKING_DISTANCE,
                'min_hits': TrackingConfig.MIN_HITS_FOR_CONFIRMATION,
                'iou_threshold': TrackingConfig.IOU_THRESHOLD
            }
            
            self.tracking_manager = TrackingManager(
                tracking_config=tracking_config,
                plate_confirmation_threshold=TrackingConfig.PLATE_CONFIRMATION_THRESHOLD,
                max_plate_age=TrackingConfig.MAX_PLATE_AGE
            )
            self.stats['tracking_enabled'] = True
            self.logger.info("✅ Tracking system initialized")

        # Initialize plate validator
        self.plate_validator = PlateValidator()
        self.logger.info("✅ Plate validator initialized")

        self.logger.info("HeadlessStreamManager initialized")
    
    def add_frame_callback(self, callback: Callable[[StreamFrame], None]):
        """Add callback untuk new frames"""
        self.frame_callbacks.append(callback)

    def add_detection_callback(self, callback: Callable[[list], None]):
        """Add callback untuk detections"""
        self.detection_callbacks.append(callback)

    def _database_save_worker(self):
        """Background worker untuk save detections ke database (non-blocking)"""
        self.logger.info("Database save worker started")

        while self.db_worker_running:
            try:
                # Get detection from queue (block with timeout)
                try:
                    detection_data = self.db_save_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Unpack detection data
                detection = detection_data['detection']
                source_info = detection_data['source_info']
                full_frame = detection_data.get('full_frame')  # NEW: Get full frame

                # Save to SQLite (always)
                try:
                    self.database.save_detection(
                        detection,
                        source_info=source_info,
                        save_image=True
                    )
                except Exception as e:
                    self.logger.error(f"SQLite save error: {e}")

                # Save vehicle image (full frame) if available
                vehicle_image_path = None
                if full_frame is not None:
                    try:
                        vehicle_image_path = self.database.save_vehicle_image(
                            full_frame=full_frame,
                            plate_number=detection.text,
                            timestamp=detection.timestamp
                        )
                    except Exception as e:
                        self.logger.error(f"Vehicle image save error: {e}")

                # Process through Access Controller if enabled (MySQL)
                if self.enable_access_control and self.access_controller:
                    try:
                        access_result = self.access_controller.process_detection(
                            detection,
                            image_path=vehicle_image_path  # NEW: Pass vehicle image path instead of plate image
                        )

                        # Attach access_result to detection for frontend
                        detection.access_result = access_result

                        # Log access result (ONLY if not filtered spam)
                        if access_result['access'] == 'Authorized':
                            self.logger.info(f"✅ ACCESS AUTHORIZED: {detection.text} - {access_result['vehicle']['owner_name']}")
                        elif access_result['access'] == 'Denied':
                            self.logger.warning(f"❌ ACCESS DENIED: {detection.text} - Not registered")
                        elif access_result['access'] == 'Filtered':
                            # Spam filtered - don't log (keep terminal clean)
                            pass
                    except Exception as e:
                        self.logger.error(f"MySQL access control error: {e}")

                # Mark task as done
                self.db_save_queue.task_done()

            except Exception as e:
                self.logger.error(f"Database worker error: {e}")
                time.sleep(0.1)

        self.logger.info("Database save worker stopped")

    def is_duplicate(self, plate_text: str) -> bool:
        """
        Check if plate was detected within duplicate_window (5 seconds)

        Args:
            plate_text: Plate text to check

        Returns:
            bool: True if duplicate, False if unique
        """
        current_time = time.time()

        # Check if plate exists in recent detections
        if plate_text in self.recent_detections:
            time_since_last = current_time - self.recent_detections[plate_text]

            if time_since_last < self.duplicate_window:
                # Still within duplicate window
                return True
            else:
                # Outside window, update timestamp
                self.recent_detections[plate_text] = current_time
                return False
        else:
            # New plate, add to recent detections
            self.recent_detections[plate_text] = current_time
            return False

    def _generate_plate_variants(self, plate_text: str) -> list:
        """
        Generate plate text variants to handle common OCR errors

        Variants include:
        - Space variations (with/without spaces)
        - Common prefix additions (if starts with digit)
        - Digit substitutions (3↔8, 0↔O, 1↔I)

        Args:
            plate_text: Original plate text from OCR

        Returns:
            list: List of plate text variants to try
        """
        variants = [plate_text]

        # Remove spaces version
        no_space = plate_text.replace(' ', '')
        if no_space != plate_text:
            variants.append(no_space)

        # Add spaces in standard positions: [XX] [NNNN] [XXX]
        # Example: B1263EZU → B 1263 EZU
        import re
        match = re.match(r'^([A-Z]{1,2})(\d{3,4})([A-Z]{2,3})$', no_space)
        if match:
            prefix, digits, suffix = match.groups()
            spaced = f"{prefix} {digits} {suffix}"
            if spaced not in variants:
                variants.append(spaced)

        # ✅ If starts with digit, try adding common prefixes
        if no_space and no_space[0].isdigit():
            common_prefixes = ['B', 'D', 'F', 'N', 'T', 'L']
            for prefix in common_prefixes:
                variant_with_prefix = prefix + no_space
                if variant_with_prefix not in variants:
                    variants.append(variant_with_prefix)
                # Also try with spaces
                match = re.match(r'^([A-Z]{1,2})(\d{3,4})([A-Z]{2,3})$', variant_with_prefix)
                if match:
                    p, d, s = match.groups()
                    spaced_variant = f"{p} {d} {s}"
                    if spaced_variant not in variants:
                        variants.append(spaced_variant)

        # ✅ Digit substitutions (3↔8 most common OCR error)
        # Example: 1268 → try 1263, 1238, 1208
        if '8' in no_space:
            # Try replacing 8 with 3
            variant_3 = no_space.replace('8', '3', 1)  # Replace first occurrence
            if variant_3 not in variants:
                variants.append(variant_3)
                # Also try with spaces
                match = re.match(r'^([A-Z]{1,2})(\d{3,4})([A-Z]{2,3})$', variant_3)
                if match:
                    p, d, s = match.groups()
                    variants.append(f"{p} {d} {s}")

        if '3' in no_space:
            # Try replacing 3 with 8
            variant_8 = no_space.replace('3', '8', 1)
            if variant_8 not in variants:
                variants.append(variant_8)
                match = re.match(r'^([A-Z]{1,2})(\d{3,4})([A-Z]{2,3})$', variant_8)
                if match:
                    p, d, s = match.groups()
                    variants.append(f"{p} {d} {s}")

        self.logger.debug(f"Generated {len(variants)} variants for '{plate_text}': {variants[:5]}...")
        return variants

    def _check_plate_in_database(self, plate_text: str) -> bool:
        """
        Check if plate exists in vehicles database (MySQL) with fuzzy matching
        Enhanced with variant generation for OCR errors

        Args:
            plate_text: Plate number to check

        Returns:
            bool: True if plate exists in database, False otherwise
        """
        try:
            if self.use_mysql and self.mysql_db:
                # Try exact match first
                vehicle = self.mysql_db.check_vehicle_registered(plate_text)
                if vehicle:
                    self.logger.info(f"✅ Plate '{plate_text}' FOUND in database (exact match): {vehicle.get('owner_name', 'Unknown')}")
                    return True

                # ✅ ACCURACY FIX: Fuzzy matching when exact match fails
                self.logger.info(f"⚠️ Exact match failed for '{plate_text}', trying fuzzy matching...")

                # Generate plate variants to try
                variants = self._generate_plate_variants(plate_text)

                for variant in variants:
                    if variant == plate_text:
                        continue  # Skip original, already tried

                    vehicle = self.mysql_db.check_vehicle_registered(variant)
                    if vehicle:
                        self.logger.info(f"✅ Plate '{plate_text}' FOUND via fuzzy match '{variant}': {vehicle.get('owner_name', 'Unknown')}")
                        return True

                self.logger.info(f"❌ Plate '{plate_text}' NOT FOUND in database (tried {len(variants)} variants)")
                return False
            else:
                # No database connection, allow all plates (fallback)
                self.logger.warning(f"⚠️ No database connection, allowing plate: '{plate_text}'")
                return True
        except Exception as e:
            self.logger.error(f"Database check error for '{plate_text}': {e}")
            return False  # On error, don't lock

    def _get_bbox_key(self, bbox: tuple) -> str:
        """
        Generate unique key for bounding box location with high tolerance
        Ensures same physical plate gets same key across frames

        Args:
            bbox: (x, y, w, h) bounding box

        Returns:
            str: Unique key based on center position and size
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        # Round to nearest 50 pixels untuk toleransi gerakan lebih besar
        # Increased from 20px to 50px to handle camera shake and plate movement
        center_x = (center_x // 50) * 50
        center_y = (center_y // 50) * 50

        # Add size component untuk distinguish plates yang berdekatan
        # Round size to nearest 30 pixels
        size_key = ((w + h) // 60) * 30

        return f"{center_x}_{center_y}_{size_key}"

    def get_stable_text(self, detection: PlateDetection) -> str:
        """
        Get stable text for detection using voting system
        Only votes for VALID Indonesian plate formats
        Prefers longer, more complete text

        Args:
            detection: PlateDetection object

        Returns:
            str: Most stable/consistent text for this location
        """
        bbox_key = self._get_bbox_key(detection.bbox)

        # Check if text already locked
        if bbox_key in self.stable_plate_texts:
            return self.stable_plate_texts[bbox_key]

        # Initialize vote dict for this bbox
        if bbox_key not in self.plate_text_votes:
            self.plate_text_votes[bbox_key] = {}

        # IMPORTANT: Only add vote if text is VALID Indonesian format AND length >= 4
        text = detection.text
        if len(text) >= 4 and self.plate_validator.validate(text):
            # Give extra votes for longer text (more complete reads)
            vote_weight = 1
            if len(text) >= 6:  # Full plate text (e.g., "F1346" or "B1234ABC")
                vote_weight = 2  # Double votes for complete reads

            self.plate_text_votes[bbox_key][text] = self.plate_text_votes[bbox_key].get(text, 0) + vote_weight
            self.logger.debug(f"✅ Vote +{vote_weight} for '{text}' (len={len(text)})")
        else:
            # Invalid format or too short, don't vote for it
            self.logger.debug(f"⏭️  Skip voting: '{text}' (len={len(text)}, valid={self.plate_validator.validate(text)})")

        # Get text with most votes (if any valid votes exist)
        votes = self.plate_text_votes[bbox_key]
        if not votes:
            # No valid votes yet, return empty string
            return ""

        # Prefer longest text if votes are close (within 1 vote)
        sorted_by_votes = sorted(votes.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        best_text = sorted_by_votes[0][0]
        best_vote_count = sorted_by_votes[0][1]

        # Lock text if reached threshold
        if best_vote_count >= self.bbox_vote_threshold:
            # Check if this is a NEW lock (not already locked)
            if bbox_key not in self.stable_plate_texts:
                self.stable_plate_texts[bbox_key] = best_text
                self.newly_locked_plates.add(best_text)  # Track as newly locked!
                self.logger.info(f"🔒 NEW LOCK: '{best_text}' (key={bbox_key}, len={len(best_text)}) after {best_vote_count} votes")
                self.logger.info(f"🔍 newly_locked_plates now contains: {self.newly_locked_plates}")
            else:
                self.logger.debug(f"🔄 Already locked: '{best_text}' (key={bbox_key})")

        return best_text

    
    def start(self) -> bool:
        """Start streaming"""
        try:
            # Initialize video stream
            if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                self.video_stream = RTSPStream(self.source, buffer_size=10)
            elif isinstance(self.source, int) or self.source.isdigit():
                self.video_stream = WebcamStream(int(self.source), buffer_size=10)
            else:
                self.video_stream = VideoStream(self.source, buffer_size=10)

            # Start video stream
            if not self.video_stream.start():
                self.logger.error("Failed to start video stream")
                return False

            # WARMUP LAYER 3: Stabilization delay
            # Allow all components to fully initialize before processing
            self.logger.info("⏳ Stabilizing system (allowing all components to initialize)...")
            time.sleep(3.0)  # 3-second stabilization delay
            self.logger.info("✅ System stabilization complete")

            # Start background database save worker (non-blocking saves)
            self.db_worker_running = True
            self.db_save_thread = threading.Thread(target=self._database_save_worker, daemon=True)
            self.db_save_thread.start()
            self.logger.info("✅ Background database worker started")

            # Start processing thread
            self.running = True
            self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.stream_thread.start()

            self.logger.info("✅ HeadlessStreamManager started successfully")
            self.logger.info("🎯 System ready for accurate detection on first frame!")
            return True

        except Exception as e:
            self.logger.error(f"Error starting stream manager: {str(e)}")
            return False
    
    def stop(self):
        """Stop streaming"""
        self.logger.info("Stopping HeadlessStreamManager...")

        self.running = False

        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=2.0)

        # Stop database worker
        self.db_worker_running = False
        if self.db_save_thread and self.db_save_thread.is_alive():
            self.db_save_thread.join(timeout=2.0)
        self.logger.info("✅ Background database worker stopped")

        if self.video_stream:
            self.video_stream.stop()

        self.logger.info("HeadlessStreamManager stopped")
    
    def _stream_worker(self):
        """Main streaming worker thread"""
        frame_count = 0
        start_time = time.time()
        processing_times = []
        frame_skip_counter = 0  # Frame skipping optimization

        # WARMUP LAYER 2: Frame buffer warmup
        # Skip first 20 frames from RTSP stream (usually corrupted/low quality)
        WARMUP_FRAMES = 20
        warmup_frame_count = 0
        warmup_complete = False

        self.logger.info(f"⏳ Frame buffer warmup: skipping first {WARMUP_FRAMES} frames (RTSP stabilization)...")

        while self.running and self.video_stream.is_running():
            try:
                # Get latest frame
                ret, frame = self.video_stream.get_latest_frame()

                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                # WARMUP LAYER 2: Skip unstable initial frames
                if not warmup_complete:
                    warmup_frame_count += 1
                    if warmup_frame_count <= WARMUP_FRAMES:
                        # Show warmup progress every 5 frames
                        if warmup_frame_count % 5 == 0:
                            self.logger.info(f"⏳ Warming up frame buffer... ({warmup_frame_count}/{WARMUP_FRAMES})")
                        continue
                    else:
                        # Warmup complete
                        warmup_complete = True
                        self.logger.info(f"✅ Frame buffer warmup complete ({warmup_frame_count} frames skipped)")
                        self.logger.info("🎯 Starting detection with stable frames!")

                frame_count += 1

                # PERFORMANCE OPTIMIZATION: Safe frame skipping with bbox persistence
                from config import CCTVConfig
                if CCTVConfig.ENABLE_FRAME_SKIPPING:
                    frame_skip_counter += 1
                    if frame_skip_counter % CCTVConfig.PROCESS_EVERY_N_FRAMES != 0:
                        # Skip heavy processing (OCR), but SHOW cached detections
                        # This keeps bbox ALWAYS VISIBLE even on skipped frames

                        # Draw last cached detections on current frame
                        annotated_frame = frame.copy()

                        # Draw plate bboxes (cached)
                        if self.last_plate_detections:
                            annotated_frame = self.plate_detector.draw_detections(
                                annotated_frame, self.last_plate_detections, show_roi=False
                            )

                        # Draw vehicle bboxes (cached)
                        if self.last_vehicle_detections and self.vehicle_detector:
                            try:
                                annotated_frame = self.vehicle_detector.draw_detections(
                                    annotated_frame, self.last_vehicle_detections, show_confidence=True
                                )
                            except:
                                pass

                        # Draw person bboxes (cached)
                        if self.last_person_detections and self.person_detector:
                            try:
                                annotated_frame = self.person_detector.draw_detections(
                                    annotated_frame, self.last_person_detections,
                                    show_confidence=PersonDetectionConfig.PERSON_SHOW_CONFIDENCE
                                )
                            except:
                                pass

                        # Convert to base64 with cached bboxes drawn
                        frame_base64 = self._frame_to_base64(annotated_frame)

                        # Create stream frame with cached detection data
                        stream_frame = StreamFrame(
                            image_base64=frame_base64,
                            timestamp=time.time(),
                            frame_id=frame_count,
                            detections=[{
                                'text': det.text,
                                'confidence': det.confidence,
                                'bbox': det.bbox
                            } for det in self.last_plate_detections],
                            object_detections=[{
                                'class_name': det.class_name,
                                'confidence': det.confidence,
                                'bbox': det.bbox
                            } for det in self.last_vehicle_detections],
                            person_detections=[{
                                'confidence': det.confidence,
                                'bbox': det.bbox
                            } for det in self.last_person_detections],
                            fps=frame_count / (time.time() - start_time),
                            processing_time=0.0  # Skipped frame, no processing
                        )

                        # Update current frame
                        with self.lock:
                            self.current_frame = stream_frame

                        # Skip to next frame (no heavy processing)
                        continue

                process_start = time.time()
                
                # Detect plates using YOLO (deep learning plate detection)
                plate_detections = self.plate_detector.detect_plates(frame)

                # NEW: Detect persons (ISOLATED - tidak mempengaruhi plate detection)
                person_detections = []
                if self.person_detection_enabled and self.person_detector:
                    try:
                        person_detections = self.person_detector.detect_persons(frame)
                    except Exception as e:
                        # ERROR ISOLATION: Person detection error tidak crash sistem
                        self.logger.error(f"❌ Person detection error (isolated): {e}")
                        person_detections = []

                # NEW: Detect vehicles (car, motorcycle, bus, truck)
                vehicle_detections = []
                if self.vehicle_detection_enabled and self.vehicle_detector:
                    try:
                        vehicle_detections = self.vehicle_detector.detect_objects(frame, vehicles_only=True)
                        self.logger.debug(f"🚗 Detected {len(vehicle_detections)} vehicles")
                    except Exception as e:
                        # ERROR ISOLATION: Vehicle detection error tidak crash sistem
                        self.logger.error(f"❌ Vehicle detection error (isolated): {e}")
                        vehicle_detections = []

                # Process dengan tracking system jika enabled
                tracked_plates = []

                if self.tracking_enabled and self.tracking_manager:
                    # Tracking manager now only processes plates
                    _, tracked_plates = self.tracking_manager.process_frame(
                        [], plate_detections  # No object detections, only plates
                    )

                # Prepare frame dengan plate bounding boxes
                annotated_frame = frame.copy()

                # NEW: Filter plate detections before display
                filtered_plate_detections = []
                for detection in plate_detections:
                    # Filter 1: Minimum confidence threshold (EMERGENCY: Accept all detections to show bboxes)
                    if detection.confidence < 0:
                        continue

                    # Filter 2: Get stable text FIRST (voting system)
                    stable_text = self.get_stable_text(detection)

                    # EMERGENCY FIX: Disable length filter to show ALL detections
                    # if len(stable_text) < 4:
                    #     self.logger.debug(f"❌ Text too short: '{stable_text}' (len={len(stable_text)})")
                    #     continue

                    # EMERGENCY FIX: Disable validation filter to show ALL detections
                    # if not self.plate_validator.validate(stable_text):
                    #     self.logger.debug(f"❌ Invalid plate format: '{stable_text}'")
                    #     continue

                    # Use stable text if available, otherwise keep original OCR result
                    if stable_text:  # Only override if stable text exists
                        detection.text = stable_text
                    # Else: keep detection.text from OCR (may be wrong but visible)

                    filtered_plate_detections.append(detection)

                # Draw only filtered, valid plate detections
                if filtered_plate_detections:
                    annotated_frame = self.plate_detector.draw_detections(
                        annotated_frame, filtered_plate_detections, show_roi=False
                    )

                # NEW: Draw person detections (BLUE bounding boxes)
                if person_detections and self.person_detector:
                    try:
                        annotated_frame = self.person_detector.draw_detections(
                            annotated_frame, person_detections,
                            show_confidence=PersonDetectionConfig.PERSON_SHOW_CONFIDENCE
                        )
                    except Exception as e:
                        # ERROR ISOLATION: Drawing error tidak crash sistem
                        self.logger.error(f"❌ Person drawing error (isolated): {e}")

                # NEW: Draw vehicle detections (YELLOW bounding boxes - car, motorcycle, bus, truck)
                if vehicle_detections and self.vehicle_detector:
                    try:
                        annotated_frame = self.vehicle_detector.draw_detections(
                            annotated_frame,
                            vehicle_detections,
                            show_confidence=True
                        )
                    except Exception as e:
                        # ERROR ISOLATION: Drawing error tidak crash sistem
                        self.logger.error(f"❌ Vehicle drawing error (isolated): {e}")

                # Calculate processing time
                processing_time = time.time() - process_start
                processing_times.append(processing_time)
                
                # Keep only last 30 processing times for average
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Update statistics
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                avg_processing_time = sum(processing_times) / len(processing_times)
                
                with self.lock:
                    # Update detection events counter (akumulatif untuk historical tracking)
                    if len(plate_detections) > 0:
                        self.stats['total_detection_events'] += 1

                    # Use tracking results untuk statistics jika available
                    confirmed_plates = len([plate for plate in tracked_plates if plate.confirmed]) if tracked_plates else 0

                    # NEW: Update person detection stats
                    if len(person_detections) > 0:
                        self.stats['total_persons_detected'] += len(person_detections)

                    self.stats.update({
                        'total_frames': frame_count,
                        'total_detections': len(self.stable_plate_texts),  # NEW: Unique locked plates count
                        'confirmed_plates': confirmed_plates,
                        'fps': round(current_fps, 1),
                        'avg_processing_time': round(avg_processing_time, 3),
                        'last_detection_time': time.time() if plate_detections else self.stats['last_detection_time']
                    })
                
                # Convert frame to base64
                frame_base64 = self._frame_to_base64(annotated_frame)
                
                # Create stream frame
                stream_frame = StreamFrame(
                    image_base64=frame_base64,
                    timestamp=time.time(),
                    frame_id=frame_count,
                    detections=[{
                        'text': det.text,
                        'confidence': det.confidence,
                        'bbox': det.bbox
                    } for det in plate_detections],
                    object_detections=[{  # NEW: Vehicle detections (car, motorcycle, bus, truck)
                        'class_name': det.class_name,
                        'confidence': det.confidence,
                        'bbox': det.bbox
                    } for det in vehicle_detections],
                    person_detections=[{  # NEW: Person detections
                        'confidence': det.confidence,
                        'bbox': det.bbox
                    } for det in person_detections],
                    fps=current_fps,
                    processing_time=processing_time
                )
                
                # Update current frame
                with self.lock:
                    self.current_frame = stream_frame

                    # Cache detections for frame skipping (persist bboxes across frames)
                    self.last_plate_detections = filtered_plate_detections
                    self.last_vehicle_detections = vehicle_detections
                    self.last_person_detections = person_detections

                # Call frame callbacks
                for callback in self.frame_callbacks:
                    try:
                        callback(stream_frame)
                    except Exception as e:
                        self.logger.error(f"Frame callback error: {str(e)}")
                
                # Handle detections - filter by confidence, length, pattern, and duplicates
                final_detections = []

                # Process all plate detections with comprehensive filtering
                for detection in plate_detections:
                    # Filter 1: Minimum confidence threshold (EMERGENCY: Accept all detections to show bboxes)
                    if detection.confidence < 0:
                        continue

                    # Filter 2: Get stable text for this detection
                    stable_text = self.get_stable_text(detection)

                    # EMERGENCY FIX: Disable length filter to save ALL detections
                    # if len(stable_text) < 4:
                    #     continue

                    # EMERGENCY FIX: Disable validation filter to save ALL detections
                    # if not self.plate_validator.validate(stable_text):
                    #     continue

                    # Filter 5: Check for duplicates (5 second window) using stable text
                    if self.is_duplicate(stable_text):
                        continue

                    # Use stable, validated text for database save
                    detection.text = stable_text

                    # Passed all filters, add to final detections
                    final_detections.append(detection)
                
                if final_detections:
                    # PERFORMANCE OPTIMIZATION: Queue database saves (non-blocking)
                    for detection in final_detections:
                        try:
                            # Add to background save queue instead of blocking save
                            detection_data = {
                                'detection': detection,
                                'source_info': str(self.source),
                                'full_frame': frame.copy()  # NEW: Pass full frame for vehicle image
                            }

                            # Try to add to queue (non-blocking)
                            try:
                                self.db_save_queue.put_nowait(detection_data)
                            except queue.Full:
                                self.logger.warning(f"⚠️ Database save queue full, dropping detection: {detection.text}")

                        except Exception as e:
                            self.logger.error(f"Queue error: {str(e)}")

                    # IMPORTANT: Only send NEWLY LOCKED plates to frontend (not every frame!)
                    # This prevents detection counter from incrementing for same plate

                    # DEBUG: Log current state
                    self.logger.debug(f"🔍 final_detections count: {len(final_detections)}")
                    self.logger.debug(f"🔍 newly_locked_plates: {self.newly_locked_plates}")

                    newly_locked_detections = [
                        det for det in final_detections
                        if det.text in self.newly_locked_plates
                    ]

                    self.logger.debug(f"🔍 newly_locked_detections count: {len(newly_locked_detections)}")

                    if newly_locked_detections:
                        # Call detection callbacks with ONLY newly locked plates
                        for callback in self.detection_callbacks:
                            try:
                                callback(newly_locked_detections)
                                self.logger.info(f"📤 Sent {len(newly_locked_detections)} newly locked plate(s) to frontend")
                            except Exception as e:
                                self.logger.error(f"Detection callback error: {str(e)}")

                        # Clear newly locked set after sending
                        self.newly_locked_plates.clear()
                    else:
                        self.logger.debug(f"⏭️  No newly locked plates to send (already sent or not locked yet)")
                    
                    # Log detection dengan tracking info
                    for det in final_detections:
                        if self.tracking_enabled and hasattr(det, 'vehicle_id'):
                            vehicle_info = f" -> Vehicle {det.vehicle_id}" if det.vehicle_id else ""
                            self.logger.info(f"🚗 TRACKED PLATE: {det.text} (confidence: {det.confidence:.1f}%){vehicle_info}")
                        else:
                            self.logger.info(f"🚗 DETECTED: {det.text} (confidence: {det.confidence:.1f}%)")
                
            except Exception as e:
                self.logger.error(f"Error in stream worker: {str(e)}")
                time.sleep(0.1)
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert OpenCV frame ke base64 string"""
        try:
            # Resize frame untuk web (optional)
            height, width = frame.shape[:2]
            if width > 1280:  # Max width untuk web
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Encode ke JPEG dengan REDUCED quality untuk streaming performance (85% → 70%)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            # Convert ke base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            return frame_base64

        except Exception as e:
            self.logger.error(f"Error converting frame to base64: {str(e)}")
            return ""
    
    def get_current_frame(self) -> Optional[StreamFrame]:
        """Get current frame data"""
        with self.lock:
            return self.current_frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        with self.lock:
            stats = self.stats.copy()

        # Add detector statistics
        detector_stats = self.plate_detector.get_statistics()
        stats.update(detector_stats)

        # Add tracking statistics jika available
        if self.tracking_manager:
            tracking_stats = self.tracking_manager.get_statistics()
            for key, value in tracking_stats.items():
                stats[f'tracking_{key}'] = value

        # NEW: Add person detection statistics
        if self.person_detector:
            person_stats = self.person_detector.get_statistics()
            for key, value in person_stats.items():
                stats[f'person_{key}'] = value

        # Add uptime
        stats['uptime'] = round(time.time() - stats['start_time'], 1)

        return stats

    def toggle_person_detection(self, enable: bool) -> bool:
        """
        Toggle person detection on/off runtime

        Args:
            enable: True untuk enable, False untuk disable

        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if enable:
                # Enable person detection
                if self.person_detector is None:
                    # Create new detector jika belum ada
                    self.person_detector = PersonDetector(
                        model_path=PersonDetectionConfig.PERSON_YOLO_MODEL,
                        confidence=PersonDetectionConfig.PERSON_CONFIDENCE,
                        max_detections=PersonDetectionConfig.PERSON_MAX_DETECTIONS
                    )
                    if not self.person_detector.is_enabled():
                        self.logger.error("❌ Failed to enable person detector")
                        return False

                self.person_detector.enable()
                self.person_detection_enabled = True
                self.stats['person_detection_enabled'] = True
                self.logger.info("✅ Person detection ENABLED")
                return True

            else:
                # Disable person detection
                if self.person_detector:
                    self.person_detector.disable()
                self.person_detection_enabled = False
                self.stats['person_detection_enabled'] = False
                self.logger.info("⏸️ Person detection DISABLED")
                return True

        except Exception as e:
            self.logger.error(f"❌ Error toggling person detection: {e}")
            return False

    def toggle_vehicle_detection(self, enable: bool) -> bool:
        """
        Toggle vehicle detection on/off runtime

        Args:
            enable: True untuk enable, False untuk disable

        Returns:
            bool: True jika berhasil, False jika gagal
        """
        try:
            if enable:
                # Enable vehicle detection
                if self.vehicle_detector is None:
                    # Create new detector jika belum ada
                    self.vehicle_detector = YOLOObjectDetector(
                        model_path='yolov8n.pt',
                        confidence=0.005,  # 0.5% - match actual detection levels
                        max_detections=10
                    )
                    if not self.vehicle_detector.enabled:
                        self.logger.error("❌ Failed to enable vehicle detector")
                        return False

                self.vehicle_detection_enabled = True
                self.logger.info("✅ Vehicle detection ENABLED")
                return True

            else:
                # Disable vehicle detection
                self.vehicle_detection_enabled = False
                self.logger.info("⏸️ Vehicle detection DISABLED")
                return True

        except Exception as e:
            self.logger.error(f"❌ Error toggling vehicle detection: {e}")
            return False

    def is_running(self) -> bool:
        """Check if streaming is running"""
        return self.running and self.video_stream and self.video_stream.is_running()

def test_stream_manager():
    """Test function untuk stream manager"""
    print("Testing HeadlessStreamManager...")
    
    # Test dengan webcam atau RTSP
    source = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"  # NEW working camera URL ✅
    
    manager = HeadlessStreamManager(source)
    
    def frame_callback(stream_frame):
        print(f"Frame {stream_frame.frame_id}: {len(stream_frame.detections)} detections, FPS: {stream_frame.fps:.1f}")
    
    def detection_callback(detections):
        for det in detections:
            print(f"  - Detected: {det.text} ({det.confidence:.1f}%)")
    
    manager.add_frame_callback(frame_callback)
    manager.add_detection_callback(detection_callback)
    
    if manager.start():
        print("Stream manager started, running for 10 seconds...")
        time.sleep(10)
        
        stats = manager.get_statistics()
        print(f"Final statistics: {stats}")
        
        manager.stop()
        print("Test completed")
    else:
        print("Failed to start stream manager")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_stream_manager()