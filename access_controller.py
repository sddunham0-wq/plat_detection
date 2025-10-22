"""
Access Controller untuk Access Control System
Handles business logic untuk check whitelist, log access, dan update status
"""

import logging
import time
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
from mysql_database import MySQLPlateDatabase
from utils.plate_detector import PlateDetection
from config import MySQLConfig

class AccessController:
    """
    Controller untuk handle access control logic
    Bridge antara plate detection dan MySQL database

    Features:
    - Anti-spam detection dengan cooldown system
    - Bbox overlap detection untuk skip duplicate detections
    - Smart logging (only log once per detection session)
    """

    def __init__(self, mysql_db: MySQLPlateDatabase = None):
        """
        Initialize access controller

        Args:
            mysql_db: MySQL database instance (optional, will use singleton if None)
        """
        self.mysql_db = mysql_db or MySQLPlateDatabase.get_instance()
        self.logger = logging.getLogger(__name__)
        self.config = MySQLConfig

        # Statistics
        self.stats = {
            'total_processed': 0,
            'access_granted': 0,
            'access_denied': 0,
            'errors': 0,
            'spam_filtered': 0  # New: track filtered spam
        }

        # Anti-spam detection tracking
        # Format: {plate_number: {'timestamp': float, 'bbox': tuple, 'cooldown_until': float}}
        self.processed_detections = {}

        # Bbox location-based tracking (independent of OCR result)
        # Format: {bbox_grid_key: {'timestamp': float, 'cooldown_until': float}}
        self.processed_locations = {}

        # Cooldown settings (from config or defaults)
        self.cooldown_time = 30  # 30 seconds default cooldown
        self.bbox_overlap_threshold = 0.5  # 50% IoU threshold (lowered for better tolerance)
        self.debounce_time = 2  # 2 seconds debounce for same location

        self.logger.info("AccessController initialized with ENHANCED anti-spam system")

    def _normalize_plate_text(self, text: str) -> str:
        """
        Normalize plate text for consistent comparison
        Removes spaces, special chars, converts to uppercase

        Args:
            text: Raw plate text

        Returns:
            str: Normalized text
        """
        if not text:
            return ""

        # Remove spaces, special chars, uppercase
        normalized = ''.join(c for c in text.upper() if c.isalnum())
        return normalized

    def _get_bbox_grid_key(self, bbox: Tuple[int, int, int, int]) -> str:
        """
        Generate grid-based location key for bbox
        Groups nearby bboxes (±50px tolerance)

        Args:
            bbox: (x, y, width, height)

        Returns:
            str: Grid key for location
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        # Grid size: 50 pixels (high tolerance for camera shake)
        grid_x = (center_x // 50) * 50
        grid_y = (center_y // 50) * 50

        # Include size for distinguishing nearby plates
        size_key = ((w + h) // 60) * 30

        return f"{grid_x}_{grid_y}_{size_key}"

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes

        Args:
            bbox1: (x, y, width, height)
            bbox2: (x, y, width, height)

        Returns:
            float: IoU score (0.0 to 1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _is_spam_detection(self, plate_number: str, bbox: Tuple[int, int, int, int],
                          confidence: float) -> bool:
        """
        ENHANCED 4-Layer Spam Detection System

        Layer 1: Confidence check (>=70%)
        Layer 2: Text normalization + cooldown
        Layer 3: Bbox location-based cooldown (independent of OCR)
        Layer 4: IoU overlap check with debounce

        Args:
            plate_number: Detected plate number
            bbox: Bounding box (x, y, width, height)
            confidence: OCR confidence

        Returns:
            bool: True if spam (should skip), False if valid
        """
        current_time = time.time()

        # Cleanup expired cooldowns
        expired_plates = [p for p, d in self.processed_detections.items()
                         if current_time > d['cooldown_until']]
        expired_locations = [k for k, d in self.processed_locations.items()
                            if current_time > d['cooldown_until']]

        for plate in expired_plates:
            del self.processed_detections[plate]

        for loc in expired_locations:
            del self.processed_locations[loc]

        # LAYER 1: Confidence check (don't process garbage OCR)
        MIN_CONFIDENCE_TO_LOG = 70
        if confidence < MIN_CONFIDENCE_TO_LOG:
            self.logger.debug(f"❌ Layer 1 BLOCK: Low confidence {plate_number} ({confidence:.1f}%)")
            return True

        # LAYER 2: Text-based cooldown with normalization
        normalized_text = self._normalize_plate_text(plate_number)

        if normalized_text in self.processed_detections:
            cooldown_data = self.processed_detections[normalized_text]
            if current_time < cooldown_data['cooldown_until']:
                remaining = cooldown_data['cooldown_until'] - current_time
                self.logger.debug(f"❌ Layer 2 BLOCK: Text cooldown {normalized_text} ({remaining:.1f}s left)")
                return True

        # LAYER 3: Bbox location-based cooldown (independent of OCR result)
        bbox_key = self._get_bbox_grid_key(bbox)

        if bbox_key in self.processed_locations:
            location_data = self.processed_locations[bbox_key]

            # Short debounce (2 seconds) for same location
            debounce_until = location_data['timestamp'] + self.debounce_time
            if current_time < debounce_until:
                remaining = debounce_until - current_time
                self.logger.debug(f"❌ Layer 3 BLOCK: Location debounce {bbox_key} ({remaining:.1f}s)")
                return True

            # Long cooldown (30 seconds) for same location
            if current_time < location_data['cooldown_until']:
                remaining = location_data['cooldown_until'] - current_time
                self.logger.debug(f"❌ Layer 3 BLOCK: Location cooldown {bbox_key} ({remaining:.1f}s)")
                return True

        # LAYER 4: IoU overlap check with ALL active detections
        for plate, data in self.processed_detections.items():
            if current_time < data['cooldown_until']:
                processed_bbox = data['bbox']
                iou = self._calculate_iou(bbox, processed_bbox)

                if iou > self.bbox_overlap_threshold:
                    self.logger.debug(f"❌ Layer 4 BLOCK: Bbox overlap {iou:.2f} with {plate}")
                    return True

        # ✅ PASSED ALL LAYERS - Valid detection
        self.logger.debug(f"✅ VALID detection: {normalized_text} ({confidence:.1f}%)")
        return False

    def _add_to_cooldown(self, plate_number: str, bbox: Tuple[int, int, int, int]):
        """
        Add detection to DUAL cooldown tracking (text + location)

        Args:
            plate_number: Plate number to track
            bbox: Bounding box (x, y, width, height)
        """
        current_time = time.time()
        normalized_text = self._normalize_plate_text(plate_number)
        bbox_key = self._get_bbox_grid_key(bbox)

        # Track by normalized text
        self.processed_detections[normalized_text] = {
            'timestamp': current_time,
            'bbox': bbox,
            'cooldown_until': current_time + self.cooldown_time
        }

        # Track by bbox location (independent of OCR)
        self.processed_locations[bbox_key] = {
            'timestamp': current_time,
            'cooldown_until': current_time + self.cooldown_time
        }

        self.logger.debug(f"✅ Cooldown added: text={normalized_text}, location={bbox_key} ({self.cooldown_time}s)")

    def process_detection(self, detection: PlateDetection,
                         image_path: str = None) -> Dict[str, Any]:
        """
        Process plate detection untuk access control dengan anti-spam

        Args:
            detection: PlateDetection object dari detector
            image_path: Path ke gambar plat yang tersimpan (optional)

        Returns:
            Dict: Access control result dengan struktur:
                {
                    'access': 'Authorized' atau 'Denied' atau 'Filtered',
                    'plate_number': str,
                    'confidence': float,
                    'vehicle': Dict (jika Authorized),
                    'reason': str (jika Denied/Filtered),
                    'timestamp': str,
                    'access_log_id': int (jika logged)
                }
        """
        try:
            plate_number = detection.text
            confidence = detection.confidence
            bbox = detection.bbox if hasattr(detection, 'bbox') else (0, 0, 0, 0)

            # ANTI-SPAM CHECK
            if self._is_spam_detection(plate_number, bbox, confidence):
                self.stats['spam_filtered'] += 1
                return {
                    'access': 'Filtered',
                    'plate_number': plate_number,
                    'confidence': confidence,
                    'reason': 'Spam filtered (cooldown active or low confidence)',
                    'timestamp': datetime.now().isoformat(),
                    'spam_filtered': True
                }

            # Valid detection - proceed with processing
            self.stats['total_processed'] += 1
            self.logger.info(f"Processing detection: {plate_number} ({confidence:.1f}%)")

            # Check apakah kendaraan terdaftar di whitelist
            vehicle = self.mysql_db.check_vehicle_registered(plate_number)

            # Add to cooldown tracking (prevent spam)
            self._add_to_cooldown(plate_number, bbox)

            if vehicle:
                # ACCESS AUTHORIZED - Kendaraan terdaftar
                return self._handle_access_granted(vehicle, plate_number, confidence, image_path)
            else:
                # ACCESS DENIED - Kendaraan tidak terdaftar
                return self._handle_access_denied(plate_number, confidence, image_path)

        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Error processing detection: {str(e)}")
            return {
                'access': 'error',
                'plate_number': detection.text,
                'confidence': detection.confidence,
                'reason': f'System error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }

    def _handle_access_granted(self, vehicle: Dict, plate_number: str,
                               confidence: float, image_path: str = None) -> Dict[str, Any]:
        """
        Handle kendaraan terdaftar - AUTHORIZED

        Args:
            vehicle: Vehicle info dari database
            plate_number: Nomor plat
            confidence: Detection confidence
            image_path: Path gambar (optional)

        Returns:
            Dict: Access authorized response
        """
        self.stats['access_granted'] += 1

        vehicle_id = vehicle['id']
        access_log_id = None

        # Log akses ke access_log table
        access_log_id = self.mysql_db.log_access(
            vehicle_id=vehicle_id,
            plate_number=plate_number,
            status='masuk',
            image_url=image_path
        )

        # Update vehicle status jika enabled
        if self.config.AUTO_UPDATE_VEHICLE_STATUS:
            self.mysql_db.update_vehicle_status(plate_number, 'Hadir')

        self.logger.info(f"✅ ACCESS AUTHORIZED: {plate_number} - {vehicle['owner_name']} ({vehicle['vehicle_type']})")

        return {
            'access': 'Authorized',
            'plate_number': plate_number,
            'confidence': confidence,
            'vehicle': {
                'id': vehicle['id'],
                'plate_number': vehicle['plate_number'],
                'owner_name': vehicle['owner_name'],
                'vehicle_type': vehicle['vehicle_type'],
                'contact_info': vehicle['contact_info'],
                'status': 'Hadir' if self.config.AUTO_UPDATE_VEHICLE_STATUS else vehicle['status']
            },
            'access_log_id': access_log_id,
            'timestamp': datetime.now().isoformat(),
            'message': f"Selamat datang, {vehicle['owner_name']}!"
        }

    def _handle_access_denied(self, plate_number: str, confidence: float,
                             image_path: str = None) -> Dict[str, Any]:
        """
        Handle kendaraan tidak terdaftar - DENIED

        Args:
            plate_number: Nomor plat
            confidence: Detection confidence
            image_path: Path gambar (optional)

        Returns:
            Dict: Access denied response
        """
        self.stats['access_denied'] += 1

        access_log_id = None

        # Log denied access jika enabled
        if self.config.LOG_DENIED_ACCESS:
            # FIXED: Use NULL (None) instead of 0 untuk unregistered vehicles
            # Foreign key constraint allows NULL but not invalid IDs
            access_log_id = self.mysql_db.log_access(
                vehicle_id=None,  # NULL untuk kendaraan tidak terdaftar
                plate_number=plate_number,
                status='ditolak',
                image_url=image_path
            )

        self.logger.warning(f"❌ ACCESS DENIED: {plate_number} - Not registered")

        return {
            'access': 'Denied',
            'plate_number': plate_number,
            'confidence': confidence,
            'reason': 'Vehicle not registered',
            'access_log_id': access_log_id,
            'timestamp': datetime.now().isoformat(),
            'message': f"Akses ditolak. Kendaraan {plate_number} tidak terdaftar."
        }

    def check_vehicle_status(self, plate_number: str) -> Optional[Dict]:
        """
        Check status kendaraan terdaftar

        Args:
            plate_number: Nomor plat kendaraan

        Returns:
            Dict: Vehicle info atau None
        """
        try:
            return self.mysql_db.get_vehicle_info(plate_number)
        except Exception as e:
            self.logger.error(f"Error checking vehicle status: {str(e)}")
            return None

    def get_access_history(self, plate_number: str = None,
                          limit: int = 50) -> list:
        """
        Get history akses kendaraan

        Args:
            plate_number: Filter by plate (optional)
            limit: Maksimal records

        Returns:
            list: Access history records
        """
        try:
            return self.mysql_db.get_access_history(plate_number, limit)
        except Exception as e:
            self.logger.error(f"Error getting access history: {str(e)}")
            return []

    def get_statistics(self) -> Dict:
        """
        Get statistik access controller

        Returns:
            Dict: Statistics data
        """
        db_stats = self.mysql_db.get_statistics()

        return {
            'controller': {
                'total_processed': self.stats['total_processed'],
                'access_granted': self.stats['access_granted'],
                'access_denied': self.stats['access_denied'],
                'errors': self.stats['errors'],
                'grant_rate': (self.stats['access_granted'] / self.stats['total_processed'] * 100)
                              if self.stats['total_processed'] > 0 else 0
            },
            'database': db_stats
        }

    def register_vehicle(self, plate_number: str, owner_name: str,
                        vehicle_type: str, contact_info: str = "") -> Dict:
        """
        Register kendaraan baru ke whitelist

        Args:
            plate_number: Nomor plat
            owner_name: Nama pemilik
            vehicle_type: Jenis kendaraan
            contact_info: Kontak info

        Returns:
            Dict: Result dengan status dan vehicle_id
        """
        try:
            vehicle_id = self.mysql_db.register_vehicle(
                plate_number=plate_number,
                owner_name=owner_name,
                vehicle_type=vehicle_type,
                contact_info=contact_info
            )

            if vehicle_id:
                self.logger.info(f"Vehicle registered: {plate_number} - {owner_name}")
                return {
                    'success': True,
                    'vehicle_id': vehicle_id,
                    'message': f'Vehicle {plate_number} registered successfully'
                }
            else:
                return {
                    'success': False,
                    'message': f'Failed to register vehicle {plate_number} (may already exist)'
                }
        except Exception as e:
            self.logger.error(f"Error registering vehicle: {str(e)}")
            return {
                'success': False,
                'message': f'Error: {str(e)}'
            }

    def get_all_vehicles(self, status: str = None) -> list:
        """
        Get list semua kendaraan terdaftar

        Args:
            status: Filter by status (optional)

        Returns:
            list: List of vehicles
        """
        try:
            return self.mysql_db.get_all_vehicles(status)
        except Exception as e:
            self.logger.error(f"Error getting vehicles: {str(e)}")
            return []

    def test_connection(self) -> bool:
        """
        Test MySQL connection

        Returns:
            bool: True if connection OK
        """
        try:
            return self.mysql_db.test_connection()
        except Exception as e:
            self.logger.error(f"Connection test failed: {str(e)}")
            return False

# Helper function untuk testing
def test_access_controller():
    """Quick test access controller"""
    logging.basicConfig(level=logging.INFO)

    print("Testing Access Controller...")
    controller = AccessController()

    # Test connection
    print("\n1. Testing MySQL connection...")
    if controller.test_connection():
        print("   ✅ Connection OK")
    else:
        print("   ❌ Connection failed")
        return

    # Get statistics
    print("\n2. Getting statistics...")
    stats = controller.get_statistics()
    print(f"   Total Vehicles: {stats['database'].get('total_vehicles', 0)}")
    print(f"   Access Today: {stats['database'].get('access_today', 0)}")

    # Check registered vehicle (from SQL: F1818HG)
    print("\n3. Checking registered vehicle (F1818HG)...")
    vehicle = controller.check_vehicle_status('F1818HG')
    if vehicle:
        print(f"   ✅ Found: {vehicle['owner_name']} ({vehicle['vehicle_type']})")
    else:
        print("   ⚠️ Vehicle not found (may need to import SQL)")

    # Test unknown vehicle
    print("\n4. Checking unknown vehicle (TEST9999)...")
    vehicle = controller.check_vehicle_status('TEST9999')
    if vehicle:
        print(f"   Found: {vehicle['owner_name']}")
    else:
        print("   ❌ Not registered (expected)")

    print("\n✅ Access Controller test completed!")

if __name__ == "__main__":
    test_access_controller()
