#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST KONEKSI MYSQL - Script sederhana untuk cek koneksi database
Jalankan script ini sebelum menjalankan app.py untuk memastikan database sudah siap
"""

import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mysql_connection():
    """Test koneksi ke MySQL dan cek database"""

    print("=" * 60)
    print("  TEST KONEKSI MYSQL - Sistem Parkir SMK")
    print("=" * 60)

    # Konfigurasi dari .env
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '3306')),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'sistem_parkir_smk')
    }

    print(f"\n📋 Konfigurasi Database:")
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   User: {config['user']}")
    print(f"   Password: {'(kosong)' if not config['password'] else '***'}")
    print(f"   Database: {config['database']}")

    # Step 1: Test koneksi ke MySQL Server
    print(f"\n🔍 Step 1: Test koneksi ke MySQL Server...")
    try:
        conn = mysql.connector.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password']
        )

        if conn.is_connected():
            print("   ✅ Koneksi ke MySQL Server BERHASIL!")

            # Cek versi MySQL
            cursor = conn.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            print(f"   📊 MySQL Version: {version}")

            conn.close()
        else:
            print("   ❌ Gagal koneksi ke MySQL Server")
            return False

    except Error as e:
        print(f"   ❌ Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Pastikan Laragon sudah running (MySQL indicator hijau)")
        print(f"   2. Cek kredensial di file .env")
        print(f"   3. Default Laragon: user=root, password=(kosong)")
        return False

    # Step 2: Test koneksi ke Database
    print(f"\n🔍 Step 2: Test koneksi ke Database '{config['database']}'...")
    try:
        conn = mysql.connector.connect(**config)

        if conn.is_connected():
            print(f"   ✅ Koneksi ke Database '{config['database']}' BERHASIL!")

            cursor = conn.cursor()

            # Cek tabel yang ada
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

            if tables:
                print(f"\n   📊 Tabel yang ditemukan:")
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    count = cursor.fetchone()[0]
                    print(f"      - {table[0]}: {count} records")
            else:
                print(f"   ⚠️  Database kosong, belum ada tabel")
                print(f"   💡 Jalankan database_setup.sql di HeidiSQL")

            conn.close()

        else:
            print(f"   ❌ Gagal koneksi ke database")
            return False

    except Error as e:
        if e.errno == 1049:  # Database doesn't exist
            print(f"   ❌ Database '{config['database']}' tidak ditemukan!")
            print(f"\n💡 Cara membuat database:")
            print(f"   1. Buka HeidiSQL (dari Laragon)")
            print(f"   2. Klik kanan di sidebar → Create new → Database")
            print(f"   3. Nama: sistem_parkir_smk")
            print(f"   4. Character set: utf8mb4")
            print(f"   5. Collation: utf8mb4_unicode_ci")
            print(f"   6. Lalu run file: database_setup.sql")
        else:
            print(f"   ❌ Error: {e}")
        return False

    # Step 3: Test sample query
    print(f"\n🔍 Step 3: Test sample query...")
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor(dictionary=True)

        # Test query kendaraan_terdaftar
        cursor.execute("SELECT * FROM kendaraan_terdaftar LIMIT 1")
        sample = cursor.fetchone()

        if sample:
            print(f"   ✅ Query test BERHASIL!")
            print(f"   📄 Sample data:")
            print(f"      - Plat: {sample['nomor_plat']}")
            print(f"      - Pemilik: {sample['nama_pemilik']}")
            print(f"      - Status: {sample['status']}")
        else:
            print(f"   ⚠️  Tabel ada tapi belum ada data")
            print(f"   💡 Data sample sudah ada di database_setup.sql")

        conn.close()

    except Error as e:
        print(f"   ❌ Error saat query: {e}")
        return False

    # Summary
    print(f"\n" + "=" * 60)
    print(f"  ✅ SEMUA TEST BERHASIL!")
    print(f"=" * 60)
    print(f"\n🎉 Database siap digunakan!")
    print(f"📝 Next step:")
    print(f"   1. Jalankan: python3 app.py")
    print(f"   2. Buka browser: http://localhost:5001")
    print(f"   3. Test fitur deteksi plat nomor")
    print(f"\n")

    return True

if __name__ == "__main__":
    try:
        success = test_mysql_connection()

        if not success:
            print(f"\n❌ Test gagal. Perbaiki error di atas sebelum lanjut.")
            print(f"\n📚 Dokumentasi lengkap ada di README.md")

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test dibatalkan oleh user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
