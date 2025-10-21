"""
Access Controller untuk Access Control System
Handles business logic untuk check whitelist, log access, dan update status
"""

import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime
from mysql_database import MySQLPlateDatabase
from utils.plate_detector import PlateDetection
from config import MySQLConfig

class AccessController:
    """
    Controller untuk handle access control logic
    Bridge antara plate detection dan MySQL database
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
            'errors': 0
        }

        self.logger.info("AccessController initialized")

    def process_detection(self, detection: PlateDetection,
                         image_path: str = None) -> Dict[str, Any]:
        """
        Process plate detection untuk access control

        Args:
            detection: PlateDetection object dari detector
            image_path: Path ke gambar plat yang tersimpan (optional)

        Returns:
            Dict: Access control result dengan struktur:
                {
                    'access': 'Authorized' atau 'Denied',
                    'plate_number': str,
                    'confidence': float,
                    'vehicle': Dict (jika Authorized),
                    'reason': str (jika Denied),
                    'timestamp': str,
                    'access_log_id': int (jika logged)
                }
        """
        try:
            self.stats['total_processed'] += 1
            plate_number = detection.text
            confidence = detection.confidence

            self.logger.info(f"Processing detection: {plate_number} ({confidence:.1f}%)")

            # Check apakah kendaraan terdaftar di whitelist
            vehicle = self.mysql_db.check_vehicle_registered(plate_number)

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
