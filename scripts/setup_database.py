#!/usr/bin/env python3
"""
MySQL Database Setup Script
Auto-creates database and tables untuk multi-developer environment
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get MySQL config from environment
MYSQL_HOST = os.getenv('MYSQL_HOST', '127.0.0.1')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3307))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'plat_detection')

# SQL schemas
CREATE_VEHICLES_TABLE = """
CREATE TABLE IF NOT EXISTS vehicles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(20) UNIQUE NOT NULL,
    owner_name VARCHAR(100) NOT NULL,
    vehicle_type VARCHAR(50),
    contact_info VARCHAR(100),
    status VARCHAR(20) DEFAULT 'Belum',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_plate_number (plate_number),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

CREATE_ACCESS_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS access_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    vehicle_id INT NULL,
    plate_number VARCHAR(20) NOT NULL,
    acces_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) NOT NULL,
    image_url VARCHAR(255),
    FOREIGN KEY (vehicle_id) REFERENCES vehicles(id) ON DELETE SET NULL,
    INDEX idx_plate_number (plate_number),
    INDEX idx_acces_time (acces_time),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

# Sample data
SAMPLE_VEHICLES = [
    ('B1234ABC', 'John Doe', 'Karyawan', '081234567890', 'Belum'),
    ('B5678XYZ', 'Jane Smith', 'Tamu', '081298765432', 'Belum'),
    ('D9999KKK', 'Bob Wilson', 'Karyawan', '081355556666', 'Belum'),
]

def create_database_if_not_exists():
    """Create database jika belum ada"""
    try:
        # Connect without database
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Check if database exists
            cursor.execute(f"SHOW DATABASES LIKE '{MYSQL_DATABASE}'")
            result = cursor.fetchone()
            
            if not result:
                logger.info(f"Creating database: {MYSQL_DATABASE}")
                cursor.execute(f"CREATE DATABASE {MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                logger.info(f"✅ Database '{MYSQL_DATABASE}' created successfully!")
            else:
                logger.info(f"ℹ️  Database '{MYSQL_DATABASE}' already exists")
        
        connection.commit()
        connection.close()
        return True
        
    except pymysql.Error as e:
        logger.error(f"❌ Error creating database: {e}")
        return False

def create_tables():
    """Create tables di database"""
    try:
        # Connect to database
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Create vehicles table
            logger.info("Creating 'vehicles' table...")
            cursor.execute(CREATE_VEHICLES_TABLE)
            logger.info("✅ Table 'vehicles' created/verified")
            
            # Create access_log table
            logger.info("Creating 'access_log' table...")
            cursor.execute(CREATE_ACCESS_LOG_TABLE)
            logger.info("✅ Table 'access_log' created/verified")
        
        connection.commit()
        connection.close()
        return True
        
    except pymysql.Error as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False

def insert_sample_data():
    """Insert sample data (optional)"""
    try:
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM vehicles")
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info(f"ℹ️  Sample data already exists ({count} vehicles)")
                connection.close()
                return True
            
            # Insert sample vehicles
            logger.info("Inserting sample vehicle data...")
            for vehicle in SAMPLE_VEHICLES:
                try:
                    cursor.execute("""
                        INSERT INTO vehicles 
                        (plate_number, owner_name, vehicle_type, contact_info, status)
                        VALUES (%s, %s, %s, %s, %s)
                    """, vehicle)
                except pymysql.IntegrityError:
                    # Skip if duplicate
                    pass
            
            connection.commit()
            logger.info(f"✅ Inserted {len(SAMPLE_VEHICLES)} sample vehicles")
        
        connection.close()
        return True
        
    except pymysql.Error as e:
        logger.error(f"❌ Error inserting sample data: {e}")
        return False

def verify_setup():
    """Verify database setup"""
    try:
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4'
        )
        
        with connection.cursor() as cursor:
            # Check tables
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            
            logger.info("\n" + "="*50)
            logger.info("DATABASE SETUP VERIFICATION")
            logger.info("="*50)
            logger.info(f"Database: {MYSQL_DATABASE}")
            logger.info(f"Tables: {', '.join(tables)}")
            
            # Check vehicles count
            cursor.execute("SELECT COUNT(*) FROM vehicles")
            vehicle_count = cursor.fetchone()[0]
            logger.info(f"Vehicles: {vehicle_count} records")
            
            # Check access_log count
            cursor.execute("SELECT COUNT(*) FROM access_log")
            log_count = cursor.fetchone()[0]
            logger.info(f"Access Logs: {log_count} records")
            logger.info("="*50 + "\n")
        
        connection.close()
        return True
        
    except pymysql.Error as e:
        logger.error(f"❌ Error verifying setup: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("\n" + "="*50)
    logger.info("MySQL Database Setup Script")
    logger.info("Multi-Developer Environment Configuration")
    logger.info("="*50 + "\n")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"  Host: {MYSQL_HOST}:{MYSQL_PORT}")
    logger.info(f"  User: {MYSQL_USER}")
    logger.info(f"  Database: {MYSQL_DATABASE}")
    logger.info("")
    
    # Validate configuration
    if MYSQL_DATABASE == 'plat_detection_YOUR_NAME_HERE':
        logger.error("❌ Error: Please update MYSQL_DATABASE in .env file!")
        logger.error("   Current: plat_detection_YOUR_NAME_HERE")
        logger.error("   Change to: plat_detection_[your_name]")
        logger.error("\n   Example: MYSQL_DATABASE=plat_detection_andra")
        return False
    
    # Step 1: Create database
    logger.info("Step 1: Creating database...")
    if not create_database_if_not_exists():
        return False
    
    # Step 2: Create tables
    logger.info("\nStep 2: Creating tables...")
    if not create_tables():
        return False
    
    # Step 3: Insert sample data (optional)
    logger.info("\nStep 3: Inserting sample data...")
    response = input("Insert sample vehicle data? (y/n): ").lower().strip()
    if response == 'y':
        insert_sample_data()
    else:
        logger.info("⏭️  Skipping sample data insertion")
    
    # Step 4: Verify setup
    logger.info("\nStep 4: Verifying setup...")
    verify_setup()
    
    # Success message
    logger.info("\n" + "="*50)
    logger.info("✅ DATABASE SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info("\nNext steps:")
    logger.info("1. Review database in phpMyAdmin:")
    logger.info(f"   http://localhost/phpmyadmin → {MYSQL_DATABASE}")
    logger.info("\n2. Start your application:")
    logger.info("   python headless_stream.py")
    logger.info("\n3. Check MYSQL_SETUP.md for troubleshooting")
    logger.info("="*50 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
