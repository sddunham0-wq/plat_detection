#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATABASE SETUP TOOL - Sistem Parkir SMK
Setup dan verifikasi database MySQL
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration from .env
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'sistem_parkir_smk')
}

def test_connection():
    """Test koneksi MySQL (tanpa database)"""
    print("=" * 60)
    print("🔍 TESTING MySQL CONNECTION...")
    print("=" * 60)

    try:
        # Connect without database
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )

        if conn.is_connected():
            print("✅ MySQL connection: SUCCESS")
            print(f"   Server: {DB_CONFIG['host']}")
            print(f"   User: {DB_CONFIG['user']}")

            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"   MySQL Version: {version[0]}")

            cursor.close()
            conn.close()
            return True

    except Error as e:
        print(f"❌ MySQL connection FAILED: {e}")
        return False

def database_exists():
    """Cek apakah database sudah ada"""
    print("\n" + "=" * 60)
    print("🔍 CHECKING DATABASE EXISTENCE...")
    print("=" * 60)

    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )

        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]

        if DB_CONFIG['database'] in databases:
            print(f"✅ Database '{DB_CONFIG['database']}' EXISTS")
            cursor.close()
            conn.close()
            return True
        else:
            print(f"❌ Database '{DB_CONFIG['database']}' NOT FOUND")
            print(f"   Available databases: {', '.join(databases)}")
            cursor.close()
            conn.close()
            return False

    except Error as e:
        print(f"❌ Error checking database: {e}")
        return False

def create_database():
    """Buat database baru"""
    print("\n" + "=" * 60)
    print("🔨 CREATING DATABASE...")
    print("=" * 60)

    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )

        cursor = conn.cursor()

        # Drop if exists (WARNING!)
        confirm = input(f"\n⚠️  This will DROP existing database '{DB_CONFIG['database']}' if exists!\n   Continue? (yes/no): ")

        if confirm.lower() != 'yes':
            print("❌ Database creation CANCELLED")
            return False

        cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG['database']}")
        print(f"   Dropped old database (if exists)")

        cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
        print(f"✅ Database '{DB_CONFIG['database']}' CREATED")

        cursor.close()
        conn.close()
        return True

    except Error as e:
        print(f"❌ Error creating database: {e}")
        return False

def setup_tables():
    """Setup tables dari SQL file"""
    print("\n" + "=" * 60)
    print("🔨 SETTING UP TABLES...")
    print("=" * 60)

    sql_file = 'database_setup.sql'

    if not os.path.exists(sql_file):
        print(f"❌ SQL file not found: {sql_file}")
        return False

    try:
        # Read SQL file
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Connect to database
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Split SQL statements (simple split by semicolon)
        # Note: This is basic splitting, might not handle all complex SQL
        statements = []
        current_statement = []
        in_delimiter_block = False

        for line in sql_content.split('\n'):
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith('--') or not line:
                continue

            # Handle DELIMITER
            if line.startswith('DELIMITER'):
                in_delimiter_block = not in_delimiter_block
                continue

            current_statement.append(line)

            # Check for statement end
            if in_delimiter_block:
                if line.endswith('//'):
                    statements.append(' '.join(current_statement))
                    current_statement = []
            else:
                if line.endswith(';'):
                    statements.append(' '.join(current_statement))
                    current_statement = []

        # Execute statements
        success_count = 0
        error_count = 0

        for i, statement in enumerate(statements):
            if not statement or statement.isspace():
                continue

            try:
                # Remove trailing semicolon or //
                statement = statement.rstrip(';').rstrip('//')

                if statement.strip():
                    cursor.execute(statement)
                    success_count += 1

            except Error as e:
                error_count += 1
                print(f"   ⚠️  Statement {i+1} error (might be ok): {str(e)[:80]}...")

        conn.commit()
        cursor.close()
        conn.close()

        print(f"✅ Tables setup completed")
        print(f"   Success: {success_count} statements")
        print(f"   Errors: {error_count} statements (might be warnings)")

        return True

    except Error as e:
        print(f"❌ Error setting up tables: {e}")
        return False

def verify_tables():
    """Verifikasi tabel sudah dibuat"""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING TABLES...")
    print("=" * 60)

    expected_tables = ['kendaraan_terdaftar', 'log_akses_masuk']

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Show all tables
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]

        print(f"\n📋 Tables in database:")
        for table in tables:
            if table in expected_tables:
                print(f"   ✅ {table}")
            else:
                print(f"   ℹ️  {table}")

        # Check each expected table
        all_exist = True
        for table in expected_tables:
            if table not in tables:
                print(f"\n   ❌ Missing table: {table}")
                all_exist = False
            else:
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"\n   ✅ {table}: {count} rows")

                # Show sample data (first 3 rows)
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    rows = cursor.fetchall()
                    print(f"      Sample data: {len(rows)} rows")

        cursor.close()
        conn.close()

        if all_exist:
            print(f"\n✅ All expected tables EXIST")
        else:
            print(f"\n❌ Some tables MISSING")

        return all_exist

    except Error as e:
        print(f"❌ Error verifying tables: {e}")
        return False

def show_statistics():
    """Tampilkan statistik database"""
    print("\n" + "=" * 60)
    print("📊 DATABASE STATISTICS")
    print("=" * 60)

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        # Vehicles by type
        cursor.execute("""
            SELECT jenis_kendaraan, status, COUNT(*) as jumlah
            FROM kendaraan_terdaftar
            GROUP BY jenis_kendaraan, status
            ORDER BY jenis_kendaraan, status
        """)

        print("\n🚗 Vehicles by Type & Status:")
        for row in cursor.fetchall():
            print(f"   {row['jenis_kendaraan'].capitalize():10} - {row['status']:8} : {row['jumlah']:3} vehicles")

        # Access logs by status
        cursor.execute("""
            SELECT status_akses, COUNT(*) as jumlah
            FROM log_akses_masuk
            GROUP BY status_akses
            ORDER BY jumlah DESC
        """)

        print("\n📝 Access Logs by Status:")
        for row in cursor.fetchall():
            print(f"   {row['status_akses']:20} : {row['jumlah']:3} logs")

        # Recent access
        cursor.execute("""
            SELECT plat_terdeteksi, status_akses, waktu_deteksi
            FROM log_akses_masuk
            ORDER BY waktu_deteksi DESC
            LIMIT 5
        """)

        print("\n🕐 Recent Access (Last 5):")
        for row in cursor.fetchall():
            print(f"   {row['plat_terdeteksi']:15} - {row['status_akses']:15} - {row['waktu_deteksi']}")

        cursor.close()
        conn.close()

        return True

    except Error as e:
        print(f"❌ Error getting statistics: {e}")
        return False

def main_menu():
    """Main menu"""
    print("\n" + "=" * 60)
    print("🏫 SISTEM PARKIR SMK - DATABASE SETUP TOOL")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Test MySQL Connection")
    print("  2. Check Database Exists")
    print("  3. Create Database (DESTRUCTIVE!)")
    print("  4. Setup Tables from SQL file")
    print("  5. Verify Tables")
    print("  6. Show Statistics")
    print("  7. Full Setup (Create DB + Tables)")
    print("  8. Quick Verify (Check Everything)")
    print("  0. Exit")

    choice = input("\nSelect option (0-8): ").strip()

    if choice == '1':
        test_connection()
    elif choice == '2':
        database_exists()
    elif choice == '3':
        create_database()
    elif choice == '4':
        setup_tables()
    elif choice == '5':
        verify_tables()
    elif choice == '6':
        show_statistics()
    elif choice == '7':
        # Full setup
        if test_connection():
            if create_database():
                if setup_tables():
                    verify_tables()
                    show_statistics()
    elif choice == '8':
        # Quick verify
        if test_connection():
            if database_exists():
                verify_tables()
                show_statistics()
            else:
                print("\n❌ Database not found! Run option 7 for full setup.")
    elif choice == '0':
        print("\n👋 Goodbye!")
        return False
    else:
        print("\n❌ Invalid option!")

    return True

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("DATABASE SETUP TOOL FOR SISTEM PARKIR SMK")
    print("🚀" * 30)

    # Show current config
    print(f"\n📝 Current Configuration:")
    print(f"   Host: {DB_CONFIG['host']}")
    print(f"   User: {DB_CONFIG['user']}")
    print(f"   Database: {DB_CONFIG['database']}")
    print(f"   Password: {'*' * len(DB_CONFIG['password']) if DB_CONFIG['password'] else '(empty)'}")

    # Main loop
    while main_menu():
        input("\nPress Enter to continue...")

    print("\n✅ Setup tool closed.\n")
