#!/usr/bin/env python3
"""
Test MySQL Connection untuk Vehicle Access Control System
"""

import mysql.connector
from config import config

def test_connection():
    """Test MySQL connection dengan config yang sama seperti app.py"""
    try:
        print("🔗 Testing MySQL connection...")
        print(f"   Host: {config.DB_HOST}")
        print(f"   Port: {config.DB_PORT}")
        print(f"   Database: {config.DB_NAME}")
        print(f"   User: {config.DB_USER}")
        print("")

        # Attempt connection
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )

        print("✅ MySQL connection SUCCESSFUL!")
        print("")

        # Test query
        cursor = conn.cursor(dictionary=True)

        # Count vehicles
        cursor.execute("SELECT COUNT(*) as total FROM kendaraan_terdaftar WHERE status='aktif'")
        result = cursor.fetchone()
        print(f"📊 Total kendaraan aktif: {result['total']}")

        # Count access logs
        cursor.execute("SELECT COUNT(*) as total FROM log_akses_masuk")
        result = cursor.fetchone()
        print(f"📊 Total access logs: {result['total']}")

        # Show sample data
        cursor.execute("SELECT nomor_plat, nama_pemilik, jenis_kendaraan FROM kendaraan_terdaftar WHERE status='aktif' LIMIT 3")
        vehicles = cursor.fetchall()

        print("")
        print("📋 Sample kendaraan terdaftar:")
        for v in vehicles:
            print(f"   • {v['nomor_plat']} - {v['nama_pemilik']} ({v['jenis_kendaraan']})")

        cursor.close()
        conn.close()

        print("")
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("")
        print("🚀 MySQL siap digunakan oleh aplikasi!")
        print("   Jalankan: python3 app.py")
        print("")

        return True

    except mysql.connector.Error as err:
        print(f"❌ MySQL connection FAILED!")
        print(f"   Error: {err}")
        print("")
        print("💡 Troubleshooting:")
        print("   1. Check MySQL running: mysql.server status")
        print("   2. Check .env file configuration")
        print("   3. Verify database exists: mysql -u root -e 'SHOW DATABASES;'")
        print("")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_connection()
