#!/usr/bin/env python3
"""
Test Database Connection - No Password Version
Verify koneksi MySQL Laragon tanpa password
"""

import mysql.connector
from mysql.connector import Error as MySQLError
from config import config

def test_connection():
    """Test koneksi ke MySQL database"""
    print("=" * 60)
    print("🔍 TESTING DATABASE CONNECTION (NO PASSWORD)")
    print("=" * 60)
    print(f"\n📋 Database Configuration:")
    print(f"   Host: {config.DB_HOST}")
    print(f"   Port: {config.DB_PORT}")
    print(f"   User: {config.DB_USER}")
    print(f"   Password: {'(empty)' if not config.DB_PASSWORD else '(set)'}")
    print(f"   Database: {config.DB_NAME}")
    print()

    try:
        # Attempt connection
        print("🔌 Attempting connection...")
        conn = mysql.connector.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME,
            autocommit=False
        )

        if conn.is_connected():
            print("✅ CONNECTION SUCCESSFUL!")

            # Get MySQL version
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"\n📊 MySQL Version: {version[0]}")

            # Check tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"\n📋 Tables in database '{config.DB_NAME}':")
            if tables:
                for table in tables:
                    print(f"   ✓ {table[0]}")
            else:
                print("   ⚠️  No tables found - run database_setup.sql first!")

            # Count records in kendaraan_terdaftar
            try:
                cursor.execute("SELECT COUNT(*) FROM kendaraan_terdaftar")
                count = cursor.fetchone()[0]
                print(f"\n🚗 Total kendaraan terdaftar: {count}")

                cursor.execute("SELECT COUNT(*) FROM log_akses_masuk")
                count = cursor.fetchone()[0]
                print(f"📝 Total access logs: {count}")
            except MySQLError as e:
                print(f"\n⚠️  Tables not found: {e}")
                print("   Run database_setup.sql to create tables")

            cursor.close()
            conn.close()

            print("\n" + "=" * 60)
            print("✅ DATABASE TEST PASSED - Connection working perfectly!")
            print("=" * 60)
            return True

    except MySQLError as e:
        print(f"\n❌ CONNECTION FAILED!")
        print(f"\n🔴 Error Details:")
        print(f"   Error Code: {e.errno}")
        print(f"   Error Message: {e.msg}")

        print(f"\n📝 Troubleshooting Steps:")
        if e.errno == 1045:  # Access denied
            print("   1. Pastikan MySQL root user tidak pakai password")
            print("   2. Cek Laragon MySQL settings:")
            print("      - Buka Laragon > MySQL > root@localhost")
            print("      - Pastikan password = (blank)")
            print("   3. Atau set password di .env file:")
            print("      DB_PASSWORD=your_password")
        elif e.errno == 2003:  # Can't connect
            print("   1. Pastikan Laragon sudah running")
            print("   2. Start MySQL service di Laragon")
            print("   3. Check port 3306 tidak dipakai aplikasi lain")
        elif e.errno == 1049:  # Unknown database
            print(f"   1. Database '{config.DB_NAME}' belum dibuat")
            print("   2. Run database_setup.sql untuk create database")
            print("   3. Atau buat manual:")
            print(f"      CREATE DATABASE {config.DB_NAME};")
        else:
            print("   1. Restart Laragon")
            print("   2. Check MySQL service status")
            print("   3. Verify .env configuration")

        print("\n" + "=" * 60)
        return False

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        print("=" * 60)
        return False

if __name__ == '__main__':
    success = test_connection()
    exit(0 if success else 1)
