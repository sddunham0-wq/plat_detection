#!/usr/bin/env python3
"""
Test Script: INSERT & SELECT untuk Plat F 1818 HG
Plat motor dari image2.png
"""

import mysql.connector
from datetime import datetime
import sys

# =====================================================
# DATABASE CONFIG
# =====================================================

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Adjust sesuai MySQL password Anda
    'database': 'sistem_parkir_smk'
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_db_connection():
    """Create database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"❌ Database connection error: {e}")
        sys.exit(1)

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_record(record, prefix="  "):
    """Print dictionary record"""
    for key, value in record.items():
        print(f"{prefix}{key}: {value}")

# =====================================================
# TEST FUNCTIONS
# =====================================================

def test_check_existing(cursor):
    """Check if F 1818 HG already exists"""
    print_section("1. CHECK EXISTING DATA")

    cursor.execute("""
        SELECT * FROM kendaraan_terdaftar
        WHERE nomor_plat = 'F 1818 HG'
    """)

    existing = cursor.fetchone()

    if existing:
        print("✅ Plat F 1818 HG sudah terdaftar:")
        print_record(existing)
        return True
    else:
        print("ℹ️  Plat F 1818 HG belum terdaftar")
        return False

def test_insert_vehicle(cursor, conn):
    """Insert F 1818 HG ke kendaraan_terdaftar"""
    print_section("2. INSERT VEHICLE DATA")

    try:
        cursor.execute("""
            INSERT INTO kendaraan_terdaftar
            (nomor_plat, nama_pemilik, jenis_kendaraan, status, nomor_hp)
            VALUES (%s, %s, %s, %s, %s)
        """, ('F 1818 HG', 'Pak Ahmad - Staff IT', 'motor', 'aktif', '08123456789'))

        conn.commit()

        print("✅ Berhasil insert plat F 1818 HG!")

        # Verify
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE nomor_plat = 'F 1818 HG'")
        result = cursor.fetchone()
        print("\nData yang diinsert:")
        print_record(result)

        return True

    except mysql.connector.Error as e:
        print(f"❌ Insert error: {e}")
        return False

def test_select_vehicle(cursor):
    """SELECT data F 1818 HG"""
    print_section("3. SELECT VEHICLE DATA")

    cursor.execute("""
        SELECT
            id_kendaraan,
            nomor_plat,
            nama_pemilik,
            jenis_kendaraan,
            status,
            nomor_hp,
            tanggal_daftar
        FROM kendaraan_terdaftar
        WHERE nomor_plat = 'F 1818 HG'
    """)

    result = cursor.fetchone()

    if result:
        print("✅ Data ditemukan:")
        print_record(result)
        return True
    else:
        print("❌ Data tidak ditemukan")
        return False

def test_insert_log(cursor, conn):
    """Insert test log akses untuk F 1818 HG"""
    print_section("4. INSERT LOG AKSES")

    try:
        cursor.execute("""
            INSERT INTO log_akses_masuk
            (plat_terdeteksi, tingkat_yakin, status_akses, aksi_palang, catatan)
            VALUES (%s, %s, %s, %s, %s)
        """, ('F 1818 HG', 0.92, 'boleh_masuk', 'opened', 'Test deteksi motor Bogor - Python script'))

        conn.commit()

        print("✅ Berhasil insert log akses!")

        # Get inserted log
        cursor.execute("""
            SELECT * FROM log_akses_masuk
            WHERE plat_terdeteksi = 'F 1818 HG'
            ORDER BY waktu_deteksi DESC
            LIMIT 1
        """)

        log = cursor.fetchone()
        print("\nLog yang diinsert:")
        print_record(log)

        return True

    except mysql.connector.Error as e:
        print(f"❌ Insert log error: {e}")
        return False

def test_select_logs(cursor):
    """SELECT all logs untuk F 1818 HG"""
    print_section("5. SELECT ALL LOGS")

    cursor.execute("""
        SELECT
            id_log,
            plat_terdeteksi,
            tingkat_yakin,
            status_akses,
            aksi_palang,
            path_foto,
            waktu_deteksi,
            catatan
        FROM log_akses_masuk
        WHERE plat_terdeteksi = 'F 1818 HG'
        ORDER BY waktu_deteksi DESC
    """)

    logs = cursor.fetchall()

    if logs:
        print(f"✅ Ditemukan {len(logs)} log akses:")
        for i, log in enumerate(logs, 1):
            print(f"\n--- Log #{i} ---")
            print_record(log)
    else:
        print("ℹ️  Belum ada log untuk F 1818 HG")

    return len(logs) > 0

def test_join_query(cursor):
    """Test JOIN query - gabung kendaraan dengan log"""
    print_section("6. JOIN QUERY - Kendaraan + Log")

    cursor.execute("""
        SELECT
            l.id_log,
            l.plat_terdeteksi,
            k.nama_pemilik,
            k.jenis_kendaraan,
            l.tingkat_yakin,
            l.status_akses,
            l.waktu_deteksi,
            l.catatan
        FROM log_akses_masuk l
        LEFT JOIN kendaraan_terdaftar k ON l.plat_terdeteksi = k.nomor_plat
        WHERE l.plat_terdeteksi = 'F 1818 HG'
        ORDER BY l.waktu_deteksi DESC
    """)

    results = cursor.fetchall()

    if results:
        print(f"✅ Ditemukan {len(results)} record:")
        for i, record in enumerate(results, 1):
            print(f"\n--- Record #{i} ---")
            print_record(record)
    else:
        print("ℹ️  Tidak ada data")

    return len(results) > 0

def test_statistics(cursor):
    """Show summary statistics"""
    print_section("7. STATISTICS SUMMARY")

    # Total kendaraan
    cursor.execute("SELECT COUNT(*) as total FROM kendaraan_terdaftar WHERE status = 'aktif'")
    total_vehicles = cursor.fetchone()['total']
    print(f"  📊 Total kendaraan aktif: {total_vehicles}")

    # Total motor
    cursor.execute("SELECT COUNT(*) as total FROM kendaraan_terdaftar WHERE jenis_kendaraan = 'motor' AND status = 'aktif'")
    total_motors = cursor.fetchone()['total']
    print(f"  🏍️  Total motor aktif: {total_motors}")

    # Total Bogor vehicles (F series)
    cursor.execute("SELECT COUNT(*) as total FROM kendaraan_terdaftar WHERE nomor_plat LIKE 'F%'")
    total_bogor = cursor.fetchone()['total']
    print(f"  🏙️  Total kendaraan Bogor (F): {total_bogor}")

    # Total logs for F 1818 HG
    cursor.execute("SELECT COUNT(*) as total FROM log_akses_masuk WHERE plat_terdeteksi = 'F 1818 HG'")
    total_logs = cursor.fetchone()['total']
    print(f"  📝 Total log F 1818 HG: {total_logs}")

    # Check path_foto column
    cursor.execute("""
        SELECT COUNT(*) as total
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'sistem_parkir_smk'
        AND TABLE_NAME = 'log_akses_masuk'
        AND COLUMN_NAME = 'path_foto'
    """)
    has_path_foto = cursor.fetchone()['total'] > 0
    status = "✅ Ada" if has_path_foto else "❌ Tidak ada"
    print(f"  📸 Kolom path_foto: {status}")

def test_cleanup(cursor, conn):
    """Optional: cleanup test data"""
    print_section("8. CLEANUP (Optional)")

    print("ℹ️  Cleanup dilewati (uncomment untuk cleanup)")
    print("    Untuk cleanup, uncomment code di function test_cleanup()")

    # Uncomment untuk cleanup test data:
    # cursor.execute("DELETE FROM log_akses_masuk WHERE plat_terdeteksi = 'F 1818 HG' AND catatan LIKE '%Test%'")
    # cursor.execute("DELETE FROM kendaraan_terdaftar WHERE nomor_plat = 'F 1818 HG'")
    # conn.commit()
    # print("✅ Test data cleaned up")

# =====================================================
# MAIN TEST RUNNER
# =====================================================

def main():
    """Main test function"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "TEST: INSERT & SELECT F 1818 HG" + " " * 17 + "║")
    print("║" + " " * 15 + "Plat Motor dari image2.png" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")

    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # Test 1: Check existing
        exists = test_check_existing(cursor)

        # Test 2: Insert if not exists
        if not exists:
            test_insert_vehicle(cursor, conn)

        # Test 3: Select vehicle
        test_select_vehicle(cursor)

        # Test 4: Insert log
        test_insert_log(cursor, conn)

        # Test 5: Select logs
        test_select_logs(cursor)

        # Test 6: Join query
        test_join_query(cursor)

        # Test 7: Statistics
        test_statistics(cursor)

        # Test 8: Cleanup (optional)
        test_cleanup(cursor, conn)

        # Final summary
        print_section("✅ ALL TESTS COMPLETED")
        print("\n  Plat F 1818 HG siap untuk ditest dengan deteksi!")
        print("  Jalankan aplikasi: python3 app_simple.py")
        print("  Arahkan kamera ke plat F 1818 HG")
        print("  Lihat hasil di: http://localhost:5000/log_akses\n")

    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    main()
