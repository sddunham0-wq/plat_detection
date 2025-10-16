"""
MySQL Database Handler untuk Access Control System
Handles connection ke MySQL dan operasi CRUD untuk vehicles dan access_log
"""

import pymysql
import logging
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import contextmanager
from config import MySQLConfig

class MySQLPlateDatabase:
    """
    MySQL database handler untuk access control system
    Manages vehicles (whitelist) dan access_log tables
    """

    def __init__(self):
        """Initialize MySQL connection handler"""
        self.config = MySQLConfig
        self.logger = logging.getLogger(__name__)
        self._connection_pool = []
        self._pool_size = 0
        self._max_pool_size = MySQLConfig.MYSQL_POOL_SIZE

        self.logger.info(f"MySQL Database Handler initialized for {MySQLConfig.MYSQL_HOST}:{MySQLConfig.MYSQL_PORT}")

    @contextmanager
    def get_connection(self):
        """
        Context manager untuk get MySQL connection
        Auto-commit dan cleanup
        """
        connection = None
        try:
            connection = self._get_connection()
            yield connection
            connection.commit()
        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            if connection:
                self._release_connection(connection)

    def _get_connection(self):
        """
        Get connection dari pool atau create new

        Returns:
            pymysql.Connection: Database connection
        """
        # Try to get from pool
        if self._connection_pool:
            return self._connection_pool.pop()

        # Create new connection
        try:
            connection = pymysql.connect(
                host=self.config.MYSQL_HOST,
                port=self.config.MYSQL_PORT,
                user=self.config.MYSQL_USER,
                password=self.config.MYSQL_PASSWORD,
                database=self.config.MYSQL_DATABASE,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=self.config.MYSQL_CONNECT_TIMEOUT,
                read_timeout=self.config.MYSQL_READ_TIMEOUT,
                write_timeout=self.config.MYSQL_WRITE_TIMEOUT,
                autocommit=False
            )
            self._pool_size += 1
            self.logger.debug(f"Created new MySQL connection (pool size: {self._pool_size})")
            return connection
        except Exception as e:
            self.logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise

    def _release_connection(self, connection):
        """
        Release connection back to pool

        Args:
            connection: Connection to release
        """
        try:
            if connection and connection.open:
                if len(self._connection_pool) < self._max_pool_size:
                    self._connection_pool.append(connection)
                else:
                    connection.close()
                    self._pool_size -= 1
        except Exception as e:
            self.logger.warning(f"Error releasing connection: {str(e)}")

    def test_connection(self) -> bool:
        """
        Test MySQL connection

        Returns:
            bool: True if connection successful
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    self.logger.info("✅ MySQL connection test successful")
                    return True
        except Exception as e:
            self.logger.error(f"❌ MySQL connection test failed: {str(e)}")
            return False

    def check_vehicle_registered(self, plate_number: str) -> Optional[Dict]:
        """
        Check apakah kendaraan terdaftar di whitelist

        Args:
            plate_number: Nomor plat kendaraan

        Returns:
            Dict: Vehicle info jika terdaftar, None jika tidak
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT id, plate_number, owner_name, vehicle_type,
                               contact_info, status, created_at, updated_at
                        FROM vehicles
                        WHERE plate_number = %s
                    """, (plate_number,))

                    result = cursor.fetchone()

                    if result:
                        self.logger.debug(f"Vehicle found: {plate_number} - {result['owner_name']}")
                    else:
                        self.logger.debug(f"Vehicle not registered: {plate_number}")

                    return result
        except Exception as e:
            self.logger.error(f"Error checking vehicle registration: {str(e)}")
            return None

    def log_access(self, vehicle_id: int, plate_number: str, status: str,
                   image_url: str = "") -> Optional[int]:
        """
        Log akses kendaraan ke access_log table

        Args:
            vehicle_id: ID kendaraan dari vehicles table
            plate_number: Nomor plat kendaraan
            status: Status akses (masuk, keluar, ditolak)
            image_url: Path/URL gambar plat (optional)

        Returns:
            int: ID record yang tersimpan, None jika error
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO access_log
                        (vehicle_id, plate_number, acces_time, status, image_url)
                        VALUES (%s, %s, NOW(), %s, %s)
                    """, (vehicle_id, plate_number, status, image_url))

                    access_id = cursor.lastrowid
                    self.logger.info(f"Access logged: {plate_number} - {status} (ID: {access_id})")
                    return access_id
        except Exception as e:
            self.logger.error(f"Error logging access: {str(e)}")
            return None

    def update_vehicle_status(self, plate_number: str, status: str) -> bool:
        """
        Update status kendaraan di vehicles table

        Args:
            plate_number: Nomor plat kendaraan
            status: Status baru (Hadir, Tidak Hadir, dll)

        Returns:
            bool: True jika berhasil update
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE vehicles
                        SET status = %s, updated_at = NOW()
                        WHERE plate_number = %s
                    """, (status, plate_number))

                    rows_affected = cursor.rowcount

                    if rows_affected > 0:
                        self.logger.info(f"Vehicle status updated: {plate_number} -> {status}")
                        return True
                    else:
                        self.logger.warning(f"No vehicle found to update: {plate_number}")
                        return False
        except Exception as e:
            self.logger.error(f"Error updating vehicle status: {str(e)}")
            return False

    def get_vehicle_info(self, plate_number: str) -> Optional[Dict]:
        """
        Get complete vehicle info dengan history akses terakhir

        Args:
            plate_number: Nomor plat kendaraan

        Returns:
            Dict: Complete vehicle info dengan last_access data
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get vehicle info
                    cursor.execute("""
                        SELECT v.*,
                               (SELECT COUNT(*) FROM access_log WHERE vehicle_id = v.id) as total_access,
                               (SELECT MAX(acces_time) FROM access_log WHERE vehicle_id = v.id) as last_access_time
                        FROM vehicles v
                        WHERE v.plate_number = %s
                    """, (plate_number,))

                    result = cursor.fetchone()
                    return result
        except Exception as e:
            self.logger.error(f"Error getting vehicle info: {str(e)}")
            return None

    def get_access_history(self, plate_number: str = None,
                          limit: int = 100) -> List[Dict]:
        """
        Get history akses kendaraan

        Args:
            plate_number: Filter by plate number (optional)
            limit: Maksimal records

        Returns:
            List[Dict]: List of access records
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if plate_number:
                        cursor.execute("""
                            SELECT al.*, v.owner_name, v.vehicle_type
                            FROM access_log al
                            LEFT JOIN vehicles v ON al.vehicle_id = v.id
                            WHERE al.plate_number = %s
                            ORDER BY al.acces_time DESC
                            LIMIT %s
                        """, (plate_number, limit))
                    else:
                        cursor.execute("""
                            SELECT al.*, v.owner_name, v.vehicle_type
                            FROM access_log al
                            LEFT JOIN vehicles v ON al.vehicle_id = v.id
                            ORDER BY al.acces_time DESC
                            LIMIT %s
                        """, (limit,))

                    results = cursor.fetchall()
                    return results
        except Exception as e:
            self.logger.error(f"Error getting access history: {str(e)}")
            return []

    def register_vehicle(self, plate_number: str, owner_name: str,
                        vehicle_type: str, contact_info: str = "",
                        status: str = "Belum") -> Optional[int]:
        """
        Register kendaraan baru ke whitelist

        Args:
            plate_number: Nomor plat kendaraan
            owner_name: Nama pemilik
            vehicle_type: Jenis kendaraan (karyawan, tamu, dll)
            contact_info: Kontak info (optional)
            status: Status awal (default: Tidak Hadir)

        Returns:
            int: Vehicle ID, None jika error
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO vehicles
                        (plate_number, owner_name, vehicle_type, contact_info, status, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                    """, (plate_number, owner_name, vehicle_type, contact_info, status))

                    vehicle_id = cursor.lastrowid
                    self.logger.info(f"Vehicle registered: {plate_number} - {owner_name} (ID: {vehicle_id})")
                    return vehicle_id
        except pymysql.IntegrityError as e:
            self.logger.warning(f"Vehicle already exists: {plate_number}")
            return None
        except Exception as e:
            self.logger.error(f"Error registering vehicle: {str(e)}")
            return None

    def get_all_vehicles(self, status: str = None) -> List[Dict]:
        """
        Get list semua kendaraan terdaftar

        Args:
            status: Filter by status (optional)

        Returns:
            List[Dict]: List of vehicles
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if status:
                        cursor.execute("""
                            SELECT * FROM vehicles
                            WHERE status = %s
                            ORDER BY plate_number
                        """, (status,))
                    else:
                        cursor.execute("""
                            SELECT * FROM vehicles
                            ORDER BY plate_number
                        """)

                    results = cursor.fetchall()
                    return results
        except Exception as e:
            self.logger.error(f"Error getting vehicles list: {str(e)}")
            return []

    def get_statistics(self) -> Dict:
        """
        Get statistik database

        Returns:
            Dict: Database statistics
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    stats = {}

                    # Total vehicles
                    cursor.execute("SELECT COUNT(*) as total FROM vehicles")
                    stats['total_vehicles'] = cursor.fetchone()['total']

                    # Vehicles by status
                    cursor.execute("""
                        SELECT status, COUNT(*) as count
                        FROM vehicles
                        GROUP BY status
                    """)
                    stats['by_status'] = cursor.fetchall()

                    # Total access logs
                    cursor.execute("SELECT COUNT(*) as total FROM access_log")
                    stats['total_access_logs'] = cursor.fetchone()['total']

                    # Access today
                    cursor.execute("""
                        SELECT COUNT(*) as total
                        FROM access_log
                        WHERE DATE(acces_time) = CURDATE()
                    """)
                    stats['access_today'] = cursor.fetchone()['total']

                    return stats
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}

    def close_all_connections(self):
        """Close all connections in pool"""
        try:
            for conn in self._connection_pool:
                if conn and conn.open:
                    conn.close()
            self._connection_pool.clear()
            self._pool_size = 0
            self.logger.info("All MySQL connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")

# Helper function untuk quick testing
def test_mysql_connection():
    """Quick test MySQL connection"""
    logging.basicConfig(level=logging.INFO)
    db = MySQLPlateDatabase()

    print("Testing MySQL connection...")
    if db.test_connection():
        print("✅ Connection successful!")

        # Get statistics
        stats = db.get_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Total Vehicles: {stats.get('total_vehicles', 0)}")
        print(f"  Total Access Logs: {stats.get('total_access_logs', 0)}")
        print(f"  Access Today: {stats.get('access_today', 0)}")
    else:
        print("❌ Connection failed!")

    db.close_all_connections()

if __name__ == "__main__":
    test_mysql_connection()
