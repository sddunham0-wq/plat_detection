"""
Database Handler untuk Live CCTV License Plate Detection
Menyimpan hasil deteksi ke SQLite database
"""

import sqlite3
import json
import os
import cv2
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from utils.plate_detector import PlateDetection
from config import DatabaseConfig, SystemConfig

class PlateDatabase:
    """
    Database handler untuk menyimpan hasil deteksi plat nomor
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path ke database file
        """
        self.db_path = db_path or DatabaseConfig.DATABASE_PATH
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Database initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(DatabaseConfig.CREATE_TABLE_SQL)
                
                # Create index untuk performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON detections(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_plate_text 
                    ON detections(plate_text)
                """)
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def save_detection(self, detection: PlateDetection, source_info: str = "unknown", 
                      save_image: bool = True) -> int:
        """
        Simpan hasil deteksi ke database
        
        Args:
            detection: PlateDetection object
            source_info: Info sumber video (RTSP URL, webcam, dll)
            save_image: Apakah simpan gambar plat
            
        Returns:
            int: ID record yang tersimpan
        """
        try:
            image_path = None
            
            # Save image jika diminta
            if save_image and detection.processed_image is not None:
                image_path = self._save_plate_image(detection)
            
            # Insert ke database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO detections 
                    (plate_text, confidence, image_path, source_info, processed_time, timestamp)
                    VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
                """, (
                    detection.text,
                    detection.confidence,
                    image_path,
                    source_info,
                    detection.timestamp
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                self.logger.debug(f"Detection saved: ID={record_id}, Plate={detection.text}")
                return record_id
                
        except Exception as e:
            self.logger.error(f"Error saving detection: {str(e)}")
            return -1
    
    def _save_plate_image(self, detection: PlateDetection) -> str:
        """
        Simpan gambar plat ke file
        
        Args:
            detection: PlateDetection object
            
        Returns:
            str: Path ke file gambar
        """
        try:
            # Ensure output folder exists
            if not os.path.exists(SystemConfig.OUTPUT_FOLDER):
                os.makedirs(SystemConfig.OUTPUT_FOLDER)
            
            # Generate filename dengan timestamp
            timestamp = datetime.fromtimestamp(detection.timestamp)
            filename = f"{detection.text}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            # Handle duplicate filenames
            counter = 1
            base_filename = filename
            while os.path.exists(os.path.join(SystemConfig.OUTPUT_FOLDER, filename)):
                name, ext = os.path.splitext(base_filename)
                filename = f"{name}_{counter}{ext}"
                counter += 1
            
            image_path = os.path.join(SystemConfig.OUTPUT_FOLDER, filename)
            
            # Save image
            cv2.imwrite(image_path, detection.processed_image)
            
            return image_path
            
        except Exception as e:
            self.logger.error(f"Error saving plate image: {str(e)}")
            return None
    
    def get_recent_detections(self, limit: int = 100) -> List[Dict]:
        """
        Get deteksi terbaru
        
        Args:
            limit: Maksimal jumlah record
            
        Returns:
            List[Dict]: List of detection records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM detections 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting recent detections: {str(e)}")
            return []
    
    def search_plates(self, plate_text: str = None, 
                     start_date: datetime = None, 
                     end_date: datetime = None,
                     source_info: str = None,
                     limit: int = 1000) -> List[Dict]:
        """
        Search deteksi berdasarkan kriteria
        
        Args:
            plate_text: Text plat nomor (support partial match)
            start_date: Tanggal mulai
            end_date: Tanggal selesai
            source_info: Info sumber
            limit: Maksimal hasil
            
        Returns:
            List[Dict]: Matching detections
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                query = "SELECT * FROM detections WHERE 1=1"
                params = []
                
                if plate_text:
                    query += " AND plate_text LIKE ?"
                    params.append(f"%{plate_text}%")
                
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
                
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))
                
                if source_info:
                    query += " AND source_info LIKE ?"
                    params.append(f"%{source_info}%")
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error searching plates: {str(e)}")
            return []
    
    def get_statistics(self, days: int = 7) -> Dict:
        """
        Get statistik deteksi
        
        Args:
            days: Jumlah hari ke belakang
            
        Returns:
            Dict: Statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Total detections
                cursor.execute("""
                    SELECT COUNT(*) as total_detections,
                           AVG(confidence) as avg_confidence,
                           MIN(timestamp) as first_detection,
                           MAX(timestamp) as last_detection
                    FROM detections 
                    WHERE timestamp >= ?
                """, (start_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                stats = dict(cursor.fetchone())
                
                # Top plates
                cursor.execute("""
                    SELECT plate_text, COUNT(*) as count
                    FROM detections 
                    WHERE timestamp >= ?
                    GROUP BY plate_text
                    ORDER BY count DESC
                    LIMIT 10
                """, (start_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                stats['top_plates'] = [dict(row) for row in cursor.fetchall()]
                
                # Daily counts
                cursor.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM detections 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (start_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                stats['daily_counts'] = [dict(row) for row in cursor.fetchall()]
                
                # Sources
                cursor.execute("""
                    SELECT source_info, COUNT(*) as count
                    FROM detections 
                    WHERE timestamp >= ?
                    GROUP BY source_info
                    ORDER BY count DESC
                """, (start_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                stats['sources'] = [dict(row) for row in cursor.fetchall()]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """
        Cleanup record lama untuk menghemat space
        
        Args:
            days_to_keep: Jumlah hari data yang disimpan
            
        Returns:
            int: Jumlah record yang dihapus
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get images yang akan dihapus
                cursor.execute("""
                    SELECT image_path FROM detections 
                    WHERE timestamp < ? AND image_path IS NOT NULL
                """, (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                image_paths = [row[0] for row in cursor.fetchall()]
                
                # Delete old records
                cursor.execute("""
                    DELETE FROM detections 
                    WHERE timestamp < ?
                """, (cutoff_date.strftime('%Y-%m-%d %H:%M:%S'),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                # Delete image files
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except Exception as e:
                            self.logger.warning(f"Failed to delete image {image_path}: {str(e)}")
                
                self.logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old records: {str(e)}")
            return 0
    
    def export_to_csv(self, output_file: str, start_date: datetime = None, 
                     end_date: datetime = None) -> bool:
        """
        Export data ke CSV file
        
        Args:
            output_file: Path output file
            start_date: Tanggal mulai (optional)
            end_date: Tanggal selesai (optional)
            
        Returns:
            bool: Success status
        """
        try:
            import csv
            
            # Search data
            detections = self.search_plates(
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                if detections:
                    fieldnames = detections[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for detection in detections:
                        writer.writerow(detection)
                
                self.logger.info(f"Exported {len(detections)} records to {output_file}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
            return False
    
    def get_database_info(self) -> Dict:
        """Get informasi database"""
        try:
            info = {
                'database_path': self.db_path,
                'database_size_mb': round(os.path.getsize(self.db_path) / 1024 / 1024, 2),
                'exists': os.path.exists(self.db_path)
            }
            
            # Get record count
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM detections")
                info['total_records'] = cursor.fetchone()[0]
                
                # Get date range
                cursor.execute("""
                    SELECT MIN(timestamp) as first, MAX(timestamp) as last 
                    FROM detections
                """)
                result = cursor.fetchone()
                info['first_detection'] = result[0]
                info['last_detection'] = result[1]
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {str(e)}")
            return {}

def test_database():
    """Test function untuk database"""
    print("Testing PlateDatabase...")
    
    # Create test database
    db = PlateDatabase("test_plates.db")
    
    # Create dummy detection
    import numpy as np
    from utils.plate_detector import PlateDetection
    
    test_detection = PlateDetection(
        text="B1234TEST",
        confidence=95.5,
        bbox=(100, 100, 200, 50),
        processed_image=np.ones((50, 200, 3), dtype=np.uint8) * 255,
        timestamp=time.time()
    )
    
    # Save detection
    record_id = db.save_detection(test_detection, "test_camera")
    print(f"Saved detection with ID: {record_id}")
    
    # Get recent detections
    recent = db.get_recent_detections(10)
    print(f"Recent detections: {len(recent)}")
    for det in recent:
        print(f"  - {det['plate_text']} ({det['confidence']}%) at {det['timestamp']}")
    
    # Get statistics
    stats = db.get_statistics(7)
    print(f"Statistics: {stats}")
    
    # Get database info
    info = db.get_database_info()
    print(f"Database info: {info}")
    
    print("Test completed")

if __name__ == "__main__":
    # Setup logging untuk testing
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_database()