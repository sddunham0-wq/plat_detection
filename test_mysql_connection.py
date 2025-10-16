"""
Test MySQL Connection
Simple script untuk test koneksi ke MySQL database
"""

import logging
import sys
from mysql_database import MySQLPlateDatabase
from access_controller import AccessController
from config import MySQLConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mysql_connection():
    """Test basic MySQL connection"""
    print("="*60)
    print("MySQL Connection Test")
    print("="*60)

    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Host: {MySQLConfig.MYSQL_HOST}")
    print(f"   Port: {MySQLConfig.MYSQL_PORT}")
    print(f"   User: {MySQLConfig.MYSQL_USER}")
    print(f"   Database: {MySQLConfig.MYSQL_DATABASE}")
    print(f"   Password: {'(empty)' if not MySQLConfig.MYSQL_PASSWORD else '(set)'}")

    # Test 1: Basic connection
    print(f"\n🔍 Test 1: Testing MySQL connection...")
    try:
        db = MySQLPlateDatabase()
        if db.test_connection():
            print("   ✅ Connection successful!")
        else:
            print("   ❌ Connection failed!")
            return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

    # Test 2: Check tables exist
    print(f"\n🔍 Test 2: Checking database tables...")
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Check vehicles table
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'vehicles'
                """, (MySQLConfig.MYSQL_DATABASE,))
                result = cursor.fetchone()

                if result['count'] > 0:
                    print("   ✅ Table 'vehicles' exists")
                else:
                    print("   ❌ Table 'vehicles' not found!")
                    print("   💡 You may need to import plat_detection.sql")
                    return False

                # Check access_log table
                cursor.execute("""
                    SELECT COUNT(*) as count
                    FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = 'access_log'
                """, (MySQLConfig.MYSQL_DATABASE,))
                result = cursor.fetchone()

                if result['count'] > 0:
                    print("   ✅ Table 'access_log' exists")
                else:
                    print("   ❌ Table 'access_log' not found!")
                    print("   💡 You may need to import plat_detection.sql")
                    return False
    except Exception as e:
        print(f"   ❌ Error checking tables: {str(e)}")
        return False

    # Test 3: Get statistics
    print(f"\n🔍 Test 3: Getting database statistics...")
    try:
        stats = db.get_statistics()
        print(f"   📊 Total Vehicles: {stats.get('total_vehicles', 0)}")
        print(f"   📊 Total Access Logs: {stats.get('total_access_logs', 0)}")
        print(f"   📊 Access Today: {stats.get('access_today', 0)}")

        if stats.get('by_status'):
            print(f"   📊 Vehicles by Status:")
            for status_info in stats['by_status']:
                print(f"      - {status_info['status']}: {status_info['count']}")
    except Exception as e:
        print(f"   ⚠️ Warning: {str(e)}")

    # Test 4: Check sample vehicle (F1818HG from SQL)
    print(f"\n🔍 Test 4: Checking sample vehicle (F1818HG)...")
    try:
        vehicle = db.check_vehicle_registered('F1818HG')
        if vehicle:
            print(f"   ✅ Vehicle found:")
            print(f"      - Plate: {vehicle['plate_number']}")
            print(f"      - Owner: {vehicle['owner_name']}")
            print(f"      - Type: {vehicle['vehicle_type']}")
            print(f"      - Status: {vehicle['status']}")
        else:
            print(f"   ⚠️ Sample vehicle not found")
            print(f"   💡 You may need to import sample data from plat_detection.sql")
    except Exception as e:
        print(f"   ⚠️ Error: {str(e)}")

    # Test 5: Test AccessController
    print(f"\n🔍 Test 5: Testing AccessController...")
    try:
        controller = AccessController(db)
        if controller.test_connection():
            print("   ✅ AccessController initialized successfully")

            # Get controller stats
            ctrl_stats = controller.get_statistics()
            print(f"   📊 Controller Stats:")
            print(f"      - Processed: {ctrl_stats['controller']['total_processed']}")
            print(f"      - Granted: {ctrl_stats['controller']['access_granted']}")
            print(f"      - Denied: {ctrl_stats['controller']['access_denied']}")
        else:
            print("   ❌ AccessController test failed")
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")

    # Cleanup
    db.close_all_connections()

    print(f"\n{'='*60}")
    print("✅ All tests passed!")
    print("✅ MySQL integration is ready to use!")
    print(f"{'='*60}\n")

    return True

def test_import_status():
    """Check if plat_detection.sql has been imported"""
    print("\n" + "="*60)
    print("Database Import Status Check")
    print("="*60)

    try:
        db = MySQLPlateDatabase()
        stats = db.get_statistics()

        if stats.get('total_vehicles', 0) == 0:
            print("\n⚠️ WARNING: No vehicles found in database!")
            print("\n📝 To import the database:")
            print("   1. Open MySQL/phpMyAdmin")
            print("   2. Select database 'plat_detection'")
            print("   3. Import file: contoh/plat_detection.sql")
            print("\n   OR via command line:")
            print(f"   mysql -u {MySQLConfig.MYSQL_USER} -P {MySQLConfig.MYSQL_PORT} -h {MySQLConfig.MYSQL_HOST} -p {MySQLConfig.MYSQL_DATABASE} < contoh/plat_detection.sql")
            return False
        else:
            print(f"\n✅ Database imported successfully!")
            print(f"   Found {stats['total_vehicles']} vehicle(s) registered")
            return True

    except Exception as e:
        print(f"\n❌ Error checking import status: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "🚀 Starting MySQL Connection Tests..." + "\n")

    try:
        # Run connection test
        success = test_mysql_connection()

        if success:
            # Check import status
            test_import_status()

            print("\n✅ Setup complete! You can now:")
            print("   1. Run headless_stream.py with MySQL support")
            print("   2. Test access control with registered vehicles")
            print("   3. View access logs in MySQL database\n")
        else:
            print("\n❌ Connection test failed!")
            print("   Please check your MySQL configuration in .env file\n")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
