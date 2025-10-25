"""
MySQL Database Handler untuk Access Control System
Handles connection ke MySQL dan operasi CRUD untuk vehicles dan access_log
"""

import pymysql
import logging
import time
import threading
import atexit
from typing import Optional, Dict, List, Any
from datetime import datetime
from contextlib import contextmanager
from config import MySQLConfig

class MySQLPlateDatabase:
    """
    MySQL database handler untuk access control system
    Manages vehicles (whitelist) dan access_log tables

    SINGLETON PATTERN: Only one instance per process to prevent connection pool exhaustion
    Use MySQLPlateDatabase.get_instance() for singleton access
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MySQLPlateDatabase, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """
        Get singleton instance of MySQLPlateDatabase

        Returns:
            MySQLPlateDatabase: Singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize MySQL connection handler (called once due to singleton)"""
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = MySQLConfig
        self.logger = logging.getLogger(__name__)
        self._connection_pool = []
        self._pool_size = 0
        self._max_pool_size = MySQLConfig.MYSQL_POOL_SIZE

        # Connection tracking for health check and cleanup
        self._connection_last_used = {}  # Track last usage time
        self._cleanup_thread = None
        self._cleanup_running = False

        # Mark as initialized
        self._initialized = True

        self.logger.info(f"MySQL Database Handler initialized for {MySQLConfig.MYSQL_HOST}:{MySQLConfig.MYSQL_PORT}")
        self.logger.info(f"Connection pool size: {self._max_pool_size} (optimized for multi-developer)")

        # Start auto-cleanup thread
        self._start_cleanup_thread()

        # Register cleanup on exit
        atexit.register(self.close_all_connections)

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

    def _is_connection_healthy(self, connection) -> bool:
        """
        Check if connection is still alive and healthy

        Args:
            connection: Connection to check

        Returns:
            bool: True if connection is healthy
        """
        try:
            if not connection or not connection.open:
                return False

            # Ping to check connection
            connection.ping(reconnect=False)

            # Check if connection has been idle too long
            conn_id = id(connection)
            if conn_id in self._connection_last_used:
                idle_time = time.time() - self._connection_last_used[conn_id]
                if idle_time > MySQLConfig.MYSQL_MAX_IDLE_TIME:
                    self.logger.debug(f"Connection {conn_id} idle for {idle_time:.0f}s, marking as unhealthy")
                    return False

            return True
        except Exception as e:
            self.logger.debug(f"Connection health check failed: {str(e)}")
            return False

    def _get_connection(self):
        """
        Get connection dari pool atau create new
        With health check to prevent using stale connections

        Returns:
            pymysql.Connection: Database connection
        """
        # Try to get from pool with health check
        while self._connection_pool:
            connection = self._connection_pool.pop()

            # Health check before reuse
            if self._is_connection_healthy(connection):
                # Update last used time
                self._connection_last_used[id(connection)] = time.time()
                self.logger.debug(f"Reusing healthy connection from pool (pool size: {len(self._connection_pool)})")
                return connection
            else:
                # Close stale connection
                try:
                    connection.close()
                    self._pool_size -= 1
                    conn_id = id(connection)
                    if conn_id in self._connection_last_used:
                        del self._connection_last_used[conn_id]
                    self.logger.debug(f"Closed stale connection (pool size: {self._pool_size})")
                except Exception as e:
                    self.logger.warning(f"Error closing stale connection: {str(e)}")

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
            self._connection_last_used[id(connection)] = time.time()
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

    def log_access(self, vehicle_id: Optional[int], plate_number: str, status: str,
                   image_url: str = "") -> Optional[int]:
        """
        Log akses kendaraan ke access_log table

        Args:
            vehicle_id: ID kendaraan dari vehicles table (None untuk unregistered vehicles)
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
                    vehicle_info = f"vehicle_id={vehicle_id}" if vehicle_id else "unregistered"
                    self.logger.info(f"Access logged: {plate_number} - {status} ({vehicle_info}, ID: {access_id})")
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

    def _start_cleanup_thread(self):
        """
        Start background thread for automatic connection cleanup
        Runs every MYSQL_HEALTH_CHECK_INTERVAL seconds
        """
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return

        self._cleanup_running = True

        def cleanup_loop():
            """Background thread function for cleanup"""
            while self._cleanup_running:
                try:
                    time.sleep(MySQLConfig.MYSQL_HEALTH_CHECK_INTERVAL)
                    if self._cleanup_running:  # Check again after sleep
                        self._cleanup_stale_connections()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {str(e)}")

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            name="MySQL-Cleanup-Thread",
            daemon=True
        )
        self._cleanup_thread.start()
        self.logger.info(f"Auto-cleanup thread started (interval: {MySQLConfig.MYSQL_HEALTH_CHECK_INTERVAL}s)")

    def _cleanup_stale_connections(self):
        """
        Clean up stale connections from pool
        Called periodically by cleanup thread
        """
        if not self._connection_pool:
            return

        cleaned_count = 0
        healthy_connections = []

        # Check all connections in pool
        while self._connection_pool:
            connection = self._connection_pool.pop()

            if self._is_connection_healthy(connection):
                healthy_connections.append(connection)
            else:
                # Close stale connection
                try:
                    connection.close()
                    self._pool_size -= 1
                    conn_id = id(connection)
                    if conn_id in self._connection_last_used:
                        del self._connection_last_used[conn_id]
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Error closing stale connection during cleanup: {str(e)}")

        # Put healthy connections back
        self._connection_pool.extend(healthy_connections)

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} stale connections (active pool: {len(self._connection_pool)})")

    def close_all_connections(self):
        """
        Close all connections in pool
        Gracefully shutdown with cleanup thread termination
        """
        try:
            # Stop cleanup thread first
            self._cleanup_running = False
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=2.0)

            # Close all connections
            for conn in self._connection_pool:
                try:
                    if conn and conn.open:
                        conn.close()
                except Exception as e:
                    self.logger.warning(f"Error closing connection: {str(e)}")

            self._connection_pool.clear()
            self._connection_last_used.clear()
            self._pool_size = 0
            self.logger.info("All MySQL connections closed gracefully")
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")

    # ==================== MANUAL OVERRIDE METHODS ====================

    def log_manual_override(
        self,
        detection_id: Optional[int],
        original_plate: str,
        corrected_plate: Optional[str],
        original_decision: str,
        override_decision: str,
        reason: str,
        operator_pin: str,
        operator_name: str = 'operator',
        duration: str = 'one-time'
    ) -> Optional[int]:
        """
        Log manual override action

        Args:
            detection_id: ID of access_log entry (if exists)
            original_plate: Original OCR result
            corrected_plate: Corrected plate number
            original_decision: Original decision (granted/denied/pending)
            override_decision: Override decision (approved/rejected)
            reason: Reason for override
            operator_pin: Operator PIN
            operator_name: Operator name
            duration: Access duration

        Returns:
            Override record ID atau None jika gagal
        """
        try:
            from config import OverrideConfig
            from datetime import timedelta

            # Calculate expiry time
            expire_at = None
            if duration in OverrideConfig.DURATIONS and OverrideConfig.DURATIONS[duration]:
                expire_at = datetime.now() + timedelta(seconds=OverrideConfig.DURATIONS[duration])

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO manual_overrides
                        (detection_id, original_plate, corrected_plate, original_decision,
                         override_decision, reason, operator_pin, operator_name, duration, expire_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (detection_id, original_plate, corrected_plate, original_decision,
                          override_decision, reason, operator_pin, operator_name, duration, expire_at))

                    override_id = cursor.lastrowid
                    self.logger.info(f"✅ Manual override logged: {corrected_plate or original_plate} - {override_decision}")
                    return override_id

        except Exception as e:
            self.logger.error(f"❌ Error logging manual override: {str(e)}")
            return None

    def grant_temporary_access(
        self,
        plate_number: str,
        granted_by: str,
        reason: str,
        duration: str = 'one-time'
    ) -> bool:
        """
        Grant temporary access untuk plate number

        Args:
            plate_number: License plate number
            granted_by: Who granted access
            reason: Reason for temporary access
            duration: Access duration (one-time, 1-hour, 1-day, permanent)

        Returns:
            True if successful
        """
        try:
            from config import OverrideConfig
            from datetime import timedelta

            # Calculate expiry time
            expire_at = None
            if duration in OverrideConfig.DURATIONS and OverrideConfig.DURATIONS[duration]:
                expire_at = datetime.now() + timedelta(seconds=OverrideConfig.DURATIONS[duration])

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert or update temporary access
                    cursor.execute("""
                        INSERT INTO temporary_access
                        (plate_number, granted_by, reason, duration, expire_at, is_active)
                        VALUES (%s, %s, %s, %s, %s, TRUE)
                        ON DUPLICATE KEY UPDATE
                            granted_by = VALUES(granted_by),
                            reason = VALUES(reason),
                            duration = VALUES(duration),
                            expire_at = VALUES(expire_at),
                            granted_at = CURRENT_TIMESTAMP,
                            is_active = TRUE,
                            access_count = 0
                    """, (plate_number, granted_by, reason, duration, expire_at))

                    self.logger.info(f"✅ Temporary access granted: {plate_number} ({duration})")
                    return True

        except Exception as e:
            self.logger.error(f"❌ Error granting temporary access: {str(e)}")
            return False

    def check_temporary_access(self, plate_number: str) -> tuple[bool, Optional[str]]:
        """
        Check if plate has valid temporary access

        Args:
            plate_number: License plate number

        Returns:
            (has_access, reason)
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT expire_at, duration, access_count, reason
                        FROM temporary_access
                        WHERE plate_number = %s AND is_active = TRUE
                        LIMIT 1
                    """, (plate_number,))

                    result = cursor.fetchone()

                    if not result:
                        return False, None

                    expire_at = result.get('expire_at')
                    duration = result.get('duration')
                    access_count = result.get('access_count', 0)
                    reason = result.get('reason')

                    # Check validity
                    now = datetime.now()

                    # Permanent access
                    if duration == 'permanent':
                        # Update usage
                        cursor.execute("""
                            UPDATE temporary_access
                            SET access_count = access_count + 1,
                                last_access = NOW()
                            WHERE plate_number = %s
                        """, (plate_number,))
                        return True, reason

                    # One-time access
                    if duration == 'one-time':
                        if access_count == 0:
                            # Grant access and mark as used
                            cursor.execute("""
                                UPDATE temporary_access
                                SET access_count = 1,
                                    last_access = NOW(),
                                    is_active = FALSE
                                WHERE plate_number = %s
                            """, (plate_number,))
                            return True, reason
                        else:
                            return False, None

                    # Time-based access
                    if expire_at and now < expire_at:
                        # Update usage
                        cursor.execute("""
                            UPDATE temporary_access
                            SET access_count = access_count + 1,
                                last_access = NOW()
                            WHERE plate_number = %s
                        """, (plate_number,))
                        return True, reason
                    else:
                        # Expired - deactivate
                        cursor.execute("""
                            UPDATE temporary_access
                            SET is_active = FALSE
                            WHERE plate_number = %s
                        """, (plate_number,))
                        return False, None

        except Exception as e:
            self.logger.error(f"❌ Error checking temporary access: {str(e)}")
            return False, None

    def get_pending_reviews(self, limit: int = 50) -> List[Dict]:
        """
        Get detections yang membutuhkan manual review

        Args:
            limit: Maximum number of results

        Returns:
            List of pending review detections
        """
        try:
            from config import OverrideConfig

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT
                            al.id,
                            al.plate_number,
                            al.ocr_confidence,
                            al.status,
                            al.acces_time,
                            al.image_url,
                            v.owner_name,
                            v.vehicle_type,
                            CASE
                                WHEN al.ocr_confidence < %s THEN 'CRITICAL'
                                WHEN al.ocr_confidence < %s THEN 'HIGH'
                                ELSE 'MEDIUM'
                            END as priority
                        FROM access_log al
                        LEFT JOIN vehicles v ON al.vehicle_id = v.id
                        WHERE al.manual_override = FALSE
                            AND al.ocr_confidence < %s
                            AND al.acces_time >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
                        ORDER BY
                            CASE
                                WHEN al.ocr_confidence < %s THEN 1
                                WHEN al.ocr_confidence < %s THEN 2
                                ELSE 3
                            END,
                            al.acces_time DESC
                        LIMIT %s
                    """, (
                        OverrideConfig.QUEUE_CONFIDENCE_THRESHOLD,
                        OverrideConfig.OCR_CONFIDENCE_THRESHOLD,
                        OverrideConfig.OCR_CONFIDENCE_THRESHOLD,
                        OverrideConfig.QUEUE_CONFIDENCE_THRESHOLD,
                        OverrideConfig.OCR_CONFIDENCE_THRESHOLD,
                        limit
                    ))

                    return cursor.fetchall()

        except Exception as e:
            self.logger.error(f"❌ Error getting pending reviews: {str(e)}")
            return []

    def get_alert_settings(self, user_id: str = 'default') -> Dict:
        """
        Get alert settings untuk user

        Args:
            user_id: User ID

        Returns:
            Alert settings dict
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM alert_settings
                        WHERE user_id = %s
                        LIMIT 1
                    """, (user_id,))

                    result = cursor.fetchone()

                    if result:
                        return dict(result)
                    else:
                        # Return default settings
                        from config import OverrideConfig
                        return {
                            'enable_audio': OverrideConfig.ENABLE_AUDIO_ALERTS,
                            'audio_volume': OverrideConfig.ALERT_VOLUME,
                            'sound_denied': True,
                            'sound_granted_auto': False,
                            'sound_granted_manual': True,
                            'sound_manual_required': True,
                            'auto_dismiss_seconds': OverrideConfig.AUTO_DISMISS_ALERTS,
                            'max_visible_alerts': OverrideConfig.MAX_VISIBLE_ALERTS,
                            'enable_grouping': True,
                            'show_critical': True,
                            'show_high': True,
                            'show_medium': False,
                            'show_low': False,
                            'enable_quiet_hours': OverrideConfig.ENABLE_QUIET_HOURS,
                            'quiet_start_time': OverrideConfig.QUIET_START_TIME,
                            'quiet_end_time': OverrideConfig.QUIET_END_TIME,
                            'enable_dnd': False
                        }

        except Exception as e:
            self.logger.error(f"❌ Error getting alert settings: {str(e)}")
            return {}

    def save_alert_settings(self, user_id: str, settings: Dict) -> bool:
        """
        Save alert settings untuk user

        Args:
            user_id: User ID
            settings: Settings dict

        Returns:
            True if successful
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO alert_settings
                        (user_id, enable_audio, audio_volume, sound_denied, sound_granted_auto,
                         sound_granted_manual, sound_manual_required, auto_dismiss_seconds,
                         max_visible_alerts, enable_grouping, show_critical, show_high,
                         show_medium, show_low, enable_quiet_hours, quiet_start_time,
                         quiet_end_time, enable_dnd)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            enable_audio = VALUES(enable_audio),
                            audio_volume = VALUES(audio_volume),
                            sound_denied = VALUES(sound_denied),
                            sound_granted_auto = VALUES(sound_granted_auto),
                            sound_granted_manual = VALUES(sound_granted_manual),
                            sound_manual_required = VALUES(sound_manual_required),
                            auto_dismiss_seconds = VALUES(auto_dismiss_seconds),
                            max_visible_alerts = VALUES(max_visible_alerts),
                            enable_grouping = VALUES(enable_grouping),
                            show_critical = VALUES(show_critical),
                            show_high = VALUES(show_high),
                            show_medium = VALUES(show_medium),
                            show_low = VALUES(show_low),
                            enable_quiet_hours = VALUES(enable_quiet_hours),
                            quiet_start_time = VALUES(quiet_start_time),
                            quiet_end_time = VALUES(quiet_end_time),
                            enable_dnd = VALUES(enable_dnd)
                    """, (
                        user_id,
                        settings.get('enable_audio', True),
                        settings.get('audio_volume', 0.8),
                        settings.get('sound_denied', True),
                        settings.get('sound_granted_auto', False),
                        settings.get('sound_granted_manual', True),
                        settings.get('sound_manual_required', True),
                        settings.get('auto_dismiss_seconds', 5),
                        settings.get('max_visible_alerts', 3),
                        settings.get('enable_grouping', True),
                        settings.get('show_critical', True),
                        settings.get('show_high', True),
                        settings.get('show_medium', False),
                        settings.get('show_low', False),
                        settings.get('enable_quiet_hours', False),
                        settings.get('quiet_start_time', '22:00:00'),
                        settings.get('quiet_end_time', '06:00:00'),
                        settings.get('enable_dnd', False)
                    ))

                    self.logger.info(f"✅ Alert settings saved for user: {user_id}")
                    return True

        except Exception as e:
            self.logger.error(f"❌ Error saving alert settings: {str(e)}")
            return False

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
