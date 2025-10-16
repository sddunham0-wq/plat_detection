#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIGRATION SCRIPT: SQLite → MySQL
Migrate data dari sistem_parkir_smk.db (SQLite) ke MySQL
"""

import sqlite3
import mysql.connector
import os
from datetime import datetime

# Configuration
SQLITE_DB = 'sistem_parkir_smk.db'
MYSQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'sistem_parkir_smk'
}

def check_sqlite_exists():
    """Check apakah SQLite database exist"""
    if not os.path.exists(SQLITE_DB):
        print(f"❌ Error: {SQLITE_DB} tidak ditemukan!")
        print(f"   Path: {os.path.abspath(SQLITE_DB)}")
        return False
    return True

def connect_sqlite():
    """Connect ke SQLite database"""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        conn.row_factory = sqlite3.Row
        print(f"✅ Connected to SQLite: {SQLITE_DB}")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to SQLite: {e}")
        return None

def connect_mysql():
    """Connect ke MySQL database"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        print(f"✅ Connected to MySQL: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
        return conn
    except mysql.connector.Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        print(f"   Make sure:")
        print(f"   1. Laragon/MySQL is running")
        print(f"   2. Database '{MYSQL_CONFIG['database']}' exists")
        print(f"   3. Run database_setup.sql first to create tables")
        return None

def migrate_kendaraan_terdaftar(sqlite_conn, mysql_conn):
    """Migrate table kendaraan_terdaftar"""
    print("\n📋 Migrating kendaraan_terdaftar...")

    try:
        # Read from SQLite
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT * FROM kendaraan_terdaftar")
        rows = sqlite_cursor.fetchall()

        if not rows:
            print("   ⚠️  No data in SQLite kendaraan_terdaftar")
            return 0

        # Insert into MySQL
        mysql_cursor = mysql_conn.cursor()

        # Clear existing data (optional - uncomment if needed)
        # mysql_cursor.execute("DELETE FROM kendaraan_terdaftar")

        migrated = 0
        for row in rows:
            try:
                query = """
                INSERT INTO kendaraan_terdaftar
                (nomor_plat, nama_pemilik, jenis_kendaraan, status, nomor_hp, tanggal_daftar)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                nama_pemilik = VALUES(nama_pemilik),
                jenis_kendaraan = VALUES(jenis_kendaraan),
                status = VALUES(status),
                nomor_hp = VALUES(nomor_hp)
                """

                values = (
                    row['nomor_plat'],
                    row['nama_pemilik'],
                    row['jenis_kendaraan'],
                    row['status'],
                    row.get('nomor_hp', None),
                    row.get('tanggal_daftar', datetime.now())
                )

                mysql_cursor.execute(query, values)
                migrated += 1

            except mysql.connector.Error as e:
                print(f"   ⚠️  Error migrating {row['nomor_plat']}: {e}")
                continue

        mysql_conn.commit()
        print(f"   ✅ Migrated {migrated}/{len(rows)} records")
        return migrated

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def migrate_log_akses_masuk(sqlite_conn, mysql_conn):
    """Migrate table log_akses_masuk"""
    print("\n📋 Migrating log_akses_masuk...")

    try:
        # Read from SQLite
        sqlite_cursor = sqlite_conn.cursor()
        sqlite_cursor.execute("SELECT * FROM log_akses_masuk")
        rows = sqlite_cursor.fetchall()

        if not rows:
            print("   ⚠️  No data in SQLite log_akses_masuk")
            return 0

        # Insert into MySQL
        mysql_cursor = mysql_conn.cursor()

        # Clear existing data (optional - uncomment if needed)
        # mysql_cursor.execute("DELETE FROM log_akses_masuk")

        migrated = 0
        for row in rows:
            try:
                query = """
                INSERT INTO log_akses_masuk
                (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, path_foto, catatan, waktu_deteksi)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """

                values = (
                    row['plat_terdeteksi'],
                    row.get('tingkat_yakin', 0.0),
                    row['status_akses'],
                    row['aksi_palang'],
                    row.get('path_foto', None),
                    row.get('catatan', None),
                    row.get('waktu_deteksi', datetime.now())
                )

                mysql_cursor.execute(query, values)
                migrated += 1

            except mysql.connector.Error as e:
                print(f"   ⚠️  Error migrating log #{row['id']}: {e}")
                continue

        mysql_conn.commit()
        print(f"   ✅ Migrated {migrated}/{len(rows)} records")
        return migrated

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def verify_migration(mysql_conn):
    """Verify migration hasil"""
    print("\n🔍 Verifying migration...")

    try:
        cursor = mysql_conn.cursor()

        # Count kendaraan_terdaftar
        cursor.execute("SELECT COUNT(*) FROM kendaraan_terdaftar")
        kendaraan_count = cursor.fetchone()[0]
        print(f"   📊 kendaraan_terdaftar: {kendaraan_count} records")

        # Count log_akses_masuk
        cursor.execute("SELECT COUNT(*) FROM log_akses_masuk")
        log_count = cursor.fetchone()[0]
        print(f"   📊 log_akses_masuk: {log_count} records")

        # Sample data
        cursor.execute("SELECT * FROM kendaraan_terdaftar LIMIT 3")
        samples = cursor.fetchall()
        if samples:
            print(f"\n   Sample data:")
            for sample in samples:
                print(f"      - {sample[1]} ({sample[2]})")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main migration function"""
    print("=" * 60)
    print("  SQLite → MySQL Migration Script")
    print("  Sistem Parkir SMK - Database Migration")
    print("=" * 60)

    # Step 1: Check SQLite exists
    if not check_sqlite_exists():
        return

    # Step 2: Connect to both databases
    sqlite_conn = connect_sqlite()
    if not sqlite_conn:
        return

    mysql_conn = connect_mysql()
    if not mysql_conn:
        sqlite_conn.close()
        return

    # Step 3: Migration
    print("\n🚀 Starting migration...")

    kendaraan_migrated = migrate_kendaraan_terdaftar(sqlite_conn, mysql_conn)
    log_migrated = migrate_log_akses_masuk(sqlite_conn, mysql_conn)

    # Step 4: Verify
    verify_migration(mysql_conn)

    # Step 5: Cleanup
    sqlite_conn.close()
    mysql_conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("  MIGRATION SUMMARY")
    print("=" * 60)
    print(f"  ✅ kendaraan_terdaftar: {kendaraan_migrated} records")
    print(f"  ✅ log_akses_masuk: {log_migrated} records")
    print("=" * 60)
    print("\n🎉 Migration complete!")
    print("\n📝 Next steps:")
    print("   1. Verify data di MySQL (HeidiSQL/phpMyAdmin)")
    print("   2. Run app.py untuk test connection")
    print("   3. Backup SQLite file (rename ke sistem_parkir_smk.db.backup)")

if __name__ == "__main__":
    main()
