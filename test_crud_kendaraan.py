#!/usr/bin/env python3
"""
Test Script untuk CRUD Kendaraan
Pastikan MySQL running dan database sudah setup
"""

import mysql.connector
from datetime import datetime

print("\n" + "="*70)
print("🧪 TEST CRUD KENDARAAN")
print("="*70 + "\n")

# Config database
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'sistem_parkir_smk'
}

def get_connection():
    """Get database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        print("✅ Database connection successful\n")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}\n")
        return None

def test_create():
    """Test CREATE - Tambah kendaraan"""
    print("📝 TEST 1: CREATE (Tambah Kendaraan)")
    print("-" * 70)

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Test data
        test_plat = "ZTEST999"
        test_nama = "Test User - CRUD Test"
        test_jenis = "mobil"
        test_hp = "08123456789"

        # Insert
        query = """
        INSERT INTO kendaraan_terdaftar (nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp, status)
        VALUES (%s, %s, %s, %s, 'aktif')
        """
        cursor.execute(query, (test_plat, test_nama, test_jenis, test_hp))
        conn.commit()

        # Verify
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE nomor_plat = %s", (test_plat,))
        result = cursor.fetchone()

        if result:
            print(f"✅ CREATE SUCCESS")
            print(f"   Plat: {result[1]}")
            print(f"   Nama: {result[2]}")
            print(f"   Jenis: {result[3]}")
            print(f"   ID: {result[0]}")

            cursor.close()
            conn.close()
            return result[0]  # Return ID for next tests
        else:
            print("❌ CREATE FAILED - Data not found after insert")
            cursor.close()
            conn.close()
            return False

    except mysql.connector.IntegrityError:
        print(f"⚠️  Plat {test_plat} sudah ada, akan digunakan untuk test update/delete")
        cursor.execute("SELECT id_kendaraan FROM kendaraan_terdaftar WHERE nomor_plat = %s", (test_plat,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result[0] if result else False
    except Exception as e:
        print(f"❌ CREATE FAILED: {e}")
        conn.close()
        return False

def test_read(vehicle_id):
    """Test READ - Baca data kendaraan"""
    print("\n📖 TEST 2: READ (Baca Data)")
    print("-" * 70)

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor(dictionary=True)

        # Read by ID
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))
        vehicle = cursor.fetchone()

        if vehicle:
            print(f"✅ READ SUCCESS")
            print(f"   ID: {vehicle['id_kendaraan']}")
            print(f"   Plat: {vehicle['nomor_plat']}")
            print(f"   Nama: {vehicle['nama_pemilik']}")
            print(f"   Jenis: {vehicle['jenis_kendaraan']}")
            print(f"   Status: {vehicle['status']}")
            print(f"   HP: {vehicle['nomor_hp']}")

            cursor.close()
            conn.close()
            return True
        else:
            print(f"❌ READ FAILED - Vehicle ID {vehicle_id} not found")
            cursor.close()
            conn.close()
            return False

    except Exception as e:
        print(f"❌ READ FAILED: {e}")
        conn.close()
        return False

def test_update(vehicle_id):
    """Test UPDATE - Edit data kendaraan"""
    print("\n✏️  TEST 3: UPDATE (Edit Data)")
    print("-" * 70)

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Update data
        new_nama = "Test User Updated - " + datetime.now().strftime("%H:%M:%S")
        new_jenis = "motor"
        new_hp = "08999999999"
        new_status = "nonaktif"

        query = """
        UPDATE kendaraan_terdaftar
        SET nama_pemilik = %s, jenis_kendaraan = %s, nomor_hp = %s, status = %s
        WHERE id_kendaraan = %s
        """
        cursor.execute(query, (new_nama, new_jenis, new_hp, new_status, vehicle_id))
        conn.commit()

        # Verify
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))
        result = cursor.fetchone()

        if result:
            print(f"✅ UPDATE SUCCESS")
            print(f"   Nama baru: {result[2]}")
            print(f"   Jenis baru: {result[3]}")
            print(f"   Status baru: {result[4]}")
            print(f"   HP baru: {result[5]}")

            cursor.close()
            conn.close()
            return True
        else:
            print("❌ UPDATE FAILED - Data not found after update")
            cursor.close()
            conn.close()
            return False

    except Exception as e:
        print(f"❌ UPDATE FAILED: {e}")
        conn.close()
        return False

def test_delete(vehicle_id):
    """Test DELETE - Hapus kendaraan"""
    print("\n🗑️  TEST 4: DELETE (Hapus Data)")
    print("-" * 70)

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor(dictionary=True)

        # Get info dulu
        cursor.execute("SELECT nomor_plat, nama_pemilik FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))
        vehicle = cursor.fetchone()

        if not vehicle:
            print(f"❌ DELETE FAILED - Vehicle ID {vehicle_id} not found")
            cursor.close()
            conn.close()
            return False

        print(f"   Akan menghapus: {vehicle['nomor_plat']} - {vehicle['nama_pemilik']}")

        # Delete
        cursor.execute("DELETE FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))
        conn.commit()

        # Verify
        cursor.execute("SELECT * FROM kendaraan_terdaftar WHERE id_kendaraan = %s", (vehicle_id,))
        result = cursor.fetchone()

        if result is None:
            print(f"✅ DELETE SUCCESS - Data berhasil dihapus")
            cursor.close()
            conn.close()
            return True
        else:
            print("❌ DELETE FAILED - Data masih ada setelah delete")
            cursor.close()
            conn.close()
            return False

    except Exception as e:
        print(f"❌ DELETE FAILED: {e}")
        conn.close()
        return False

def test_statistics():
    """Test statistik database"""
    print("\n📊 TEST 5: STATISTICS")
    print("-" * 70)

    conn = get_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Total kendaraan
        cursor.execute("SELECT COUNT(*) FROM kendaraan_terdaftar")
        total = cursor.fetchone()[0]

        # Total aktif
        cursor.execute("SELECT COUNT(*) FROM kendaraan_terdaftar WHERE status = 'aktif'")
        aktif = cursor.fetchone()[0]

        # Total by jenis
        cursor.execute("SELECT jenis_kendaraan, COUNT(*) FROM kendaraan_terdaftar GROUP BY jenis_kendaraan")
        by_jenis = cursor.fetchall()

        print(f"✅ STATISTICS")
        print(f"   Total kendaraan: {total}")
        print(f"   Kendaraan aktif: {aktif}")
        print(f"   Kendaraan nonaktif: {total - aktif}")
        print(f"\n   By jenis:")
        for jenis, count in by_jenis:
            print(f"     - {jenis}: {count}")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ STATISTICS FAILED: {e}")
        conn.close()
        return False

# Run all tests
def run_all_tests():
    """Run all CRUD tests"""
    print("🚀 Starting CRUD tests...\n")

    results = {
        'create': False,
        'read': False,
        'update': False,
        'delete': False,
        'statistics': False
    }

    # Test 1: CREATE
    vehicle_id = test_create()
    if vehicle_id:
        results['create'] = True

        # Test 2: READ
        if test_read(vehicle_id):
            results['read'] = True

        # Test 3: UPDATE
        if test_update(vehicle_id):
            results['update'] = True

        # Test 4: DELETE
        if test_delete(vehicle_id):
            results['delete'] = True

    # Test 5: STATISTICS
    if test_statistics():
        results['statistics'] = True

    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test.upper()}")

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! CRUD is working correctly!")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please check the errors above.")

    print("="*70 + "\n")

if __name__ == "__main__":
    run_all_tests()
