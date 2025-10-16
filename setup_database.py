"""
Setup Database Script
Auto-setup MySQL database dan import schema jika diperlukan
"""

import pymysql
import logging
import os
import sys
from config import MySQLConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SQL_FILE = "contoh/plat_detection (1).sql"  # Updated SQL file

def check_mysql_connection() -> bool:
    """
    Check apakah MySQL server bisa diakses

    Returns:
        bool: True jika bisa connect
    """
    try:
        conn = pymysql.connect(
            host=MySQLConfig.MYSQL_HOST,
            port=MySQLConfig.MYSQL_PORT,
            user=MySQLConfig.MYSQL_USER,
            password=MySQLConfig.MYSQL_PASSWORD,
            connect_timeout=5
        )
        conn.close()
        logger.info("✅ MySQL server accessible")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot connect to MySQL server: {str(e)}")
        return False

def check_database_exists() -> bool:
    """
    Check apakah database plat_detection sudah ada

    Returns:
        bool: True jika database exists
    """
    try:
        conn = pymysql.connect(
            host=MySQLConfig.MYSQL_HOST,
            port=MySQLConfig.MYSQL_PORT,
            user=MySQLConfig.MYSQL_USER,
            password=MySQLConfig.MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"SHOW DATABASES LIKE '{MySQLConfig.MYSQL_DATABASE}'")
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result:
            logger.info(f"✅ Database '{MySQLConfig.MYSQL_DATABASE}' exists")
            return True
        else:
            logger.warning(f"⚠️ Database '{MySQLConfig.MYSQL_DATABASE}' not found")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking database: {str(e)}")
        return False

def create_database() -> bool:
    """
    Create database plat_detection

    Returns:
        bool: True jika berhasil create
    """
    try:
        conn = pymysql.connect(
            host=MySQLConfig.MYSQL_HOST,
            port=MySQLConfig.MYSQL_PORT,
            user=MySQLConfig.MYSQL_USER,
            password=MySQLConfig.MYSQL_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MySQLConfig.MYSQL_DATABASE}")
        cursor.close()
        conn.close()

        logger.info(f"✅ Database '{MySQLConfig.MYSQL_DATABASE}' created")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating database: {str(e)}")
        return False

def import_sql_file(sql_file: str) -> bool:
    """
    Import SQL file ke database

    Args:
        sql_file: Path ke SQL file

    Returns:
        bool: True jika berhasil import
    """
    try:
        # Check file exists
        if not os.path.exists(sql_file):
            logger.error(f"❌ SQL file not found: {sql_file}")
            return False

        # Read SQL file
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Connect to database
        conn = pymysql.connect(
            host=MySQLConfig.MYSQL_HOST,
            port=MySQLConfig.MYSQL_PORT,
            user=MySQLConfig.MYSQL_USER,
            password=MySQLConfig.MYSQL_PASSWORD,
            database=MySQLConfig.MYSQL_DATABASE
        )
        cursor = conn.cursor()

        # Split by semicolon and execute each statement
        statements = sql_content.split(';')
        success_count = 0

        for statement in statements:
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    cursor.execute(statement)
                    success_count += 1
                except Exception as e:
                    # Ignore some errors (like duplicate table, etc)
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Warning executing statement: {str(e)}")

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✅ SQL file imported successfully ({success_count} statements)")
        return True

    except Exception as e:
        logger.error(f"❌ Error importing SQL file: {str(e)}")
        return False

def verify_tables() -> bool:
    """
    Verify bahwa tables vehicles dan access_log ada

    Returns:
        bool: True jika tables ada
    """
    try:
        conn = pymysql.connect(
            host=MySQLConfig.MYSQL_HOST,
            port=MySQLConfig.MYSQL_PORT,
            user=MySQLConfig.MYSQL_USER,
            password=MySQLConfig.MYSQL_PASSWORD,
            database=MySQLConfig.MYSQL_DATABASE
        )
        cursor = conn.cursor()

        # Check vehicles table
        cursor.execute("SHOW TABLES LIKE 'vehicles'")
        vehicles_exists = cursor.fetchone() is not None

        # Check access_log table
        cursor.execute("SHOW TABLES LIKE 'access_log'")
        access_log_exists = cursor.fetchone() is not None

        cursor.close()
        conn.close()

        if vehicles_exists and access_log_exists:
            logger.info("✅ All required tables exist")
            return True
        else:
            if not vehicles_exists:
                logger.warning("⚠️ Table 'vehicles' not found")
            if not access_log_exists:
                logger.warning("⚠️ Table 'access_log' not found")
            return False

    except Exception as e:
        logger.error(f"❌ Error verifying tables: {str(e)}")
        return False

def setup_database():
    """Main setup function"""
    print("\n" + "="*60)
    print("MySQL Database Setup Script")
    print("="*60)

    print(f"\n📋 Configuration:")
    print(f"   Host: {MySQLConfig.MYSQL_HOST}")
    print(f"   Port: {MySQLConfig.MYSQL_PORT}")
    print(f"   User: {MySQLConfig.MYSQL_USER}")
    print(f"   Database: {MySQLConfig.MYSQL_DATABASE}")

    # Step 1: Check MySQL connection
    print(f"\n🔍 Step 1: Checking MySQL server connection...")
    if not check_mysql_connection():
        print("\n❌ Cannot connect to MySQL server!")
        print("   Please check:")
        print("   1. MySQL server is running")
        print("   2. Host and port are correct in .env")
        print("   3. User and password are correct")
        return False

    # Step 2: Check/Create database
    print(f"\n🔍 Step 2: Checking database...")
    if not check_database_exists():
        print(f"   Creating database '{MySQLConfig.MYSQL_DATABASE}'...")
        if not create_database():
            print(f"\n❌ Failed to create database!")
            return False

    # Step 3: Check tables
    print(f"\n🔍 Step 3: Checking tables...")
    if not verify_tables():
        print(f"   Tables missing. Importing SQL file...")

        # Step 4: Import SQL file
        print(f"\n🔍 Step 4: Importing SQL file...")
        if os.path.exists(SQL_FILE):
            if import_sql_file(SQL_FILE):
                print(f"   ✅ SQL file imported successfully")

                # Verify again
                if verify_tables():
                    print(f"   ✅ Tables verified")
                else:
                    print(f"   ⚠️ Warning: Tables verification failed after import")
            else:
                print(f"   ❌ Failed to import SQL file")
                return False
        else:
            print(f"\n⚠️ SQL file not found: {SQL_FILE}")
            print(f"   Manual import required:")
            print(f"   mysql -u {MySQLConfig.MYSQL_USER} -P {MySQLConfig.MYSQL_PORT} -h {MySQLConfig.MYSQL_HOST} -p {MySQLConfig.MYSQL_DATABASE} < {SQL_FILE}")
            return False
    else:
        print(f"   ✅ All tables exist")

    # Final verification
    print(f"\n🔍 Final verification...")
    try:
        from mysql_database import MySQLPlateDatabase
        db = MySQLPlateDatabase()

        if db.test_connection():
            print(f"   ✅ Connection test passed")

            stats = db.get_statistics()
            print(f"\n📊 Database Status:")
            print(f"   Total Vehicles: {stats.get('total_vehicles', 0)}")
            print(f"   Total Access Logs: {stats.get('total_access_logs', 0)}")

            if stats.get('total_vehicles', 0) > 0:
                print(f"\n✅ Setup complete! Database is ready to use.")
            else:
                print(f"\n⚠️ Database setup complete but no vehicles registered yet.")
                print(f"   You can add vehicles via AccessController.register_vehicle()")

            return True
        else:
            print(f"   ❌ Connection test failed")
            return False

    except Exception as e:
        logger.error(f"❌ Final verification failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting database setup...\n")

    try:
        success = setup_database()

        if success:
            print("\n" + "="*60)
            print("✅ Setup Complete!")
            print("="*60)
            print("\nNext steps:")
            print("   1. Run: python test_mysql_connection.py")
            print("   2. Run: python test_mysql_integration.py")
            print("   3. Use MySQL with: python headless_stream.py")
            print("="*60 + "\n")
        else:
            print("\n" + "="*60)
            print("❌ Setup Failed!")
            print("="*60)
            print("\nPlease fix the errors above and try again.")
            print("="*60 + "\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.exception("Unexpected error in setup")
        sys.exit(1)
