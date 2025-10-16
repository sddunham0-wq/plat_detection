#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATABASE VIEWER & MONITOR TOOL
Remote MySQL Database Access dari Windows ke macOS

Usage:
    python view_database.py

Fitur:
- Connection test & troubleshooting
- View kendaraan terdaftar
- View log akses (real-time)
- Statistik akses
- Top vehicles
"""

import mysql.connector
from mysql.connector import Error as MySQLError
from datetime import datetime
import sys
import socket
import time

# =====================================================
# KONFIGURASI DATABASE (EDIT SESUAI SETUP ANDA!)
# =====================================================

# ⚠️ GANTI dengan IP macOS Server Anda!
MACOS_SERVER_IP = "192.168.1.50"  # Cek dengan: ifconfig | grep "inet " di macOS

# Database Credentials
DB_CONFIG = {
    'host': MACOS_SERVER_IP,
    'port': 3306,
    'user': 'remote_user',       # Atau 'root' kalau sudah enable remote
    'password': 'password_kuat_123',  # ⚠️ GANTI dengan password Anda
    'database': 'sistem_parkir_smk',
    'connect_timeout': 10,
    'charset': 'utf8mb4'
}

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def print_header(title):
    """Print header dengan border"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_section(title):
    """Print section separator"""
    print(f"\n{title}")
    print("-" * 70)

def test_network_connectivity():
    """Test koneksi network ke server macOS"""
    print_header("🌐 NETWORK CONNECTIVITY TEST")

    try:
        print(f"\n1. Testing network connection to {MACOS_SERVER_IP}...")

        # Test DNS/IP
        ip = socket.gethostbyname(MACOS_SERVER_IP)
        print(f"   ✅ IP resolved: {ip}")

        # Test ping (TCP connection ke port 3306)
        print(f"\n2. Testing MySQL port 3306...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((MACOS_SERVER_IP, 3306))
        sock.close()

        if result == 0:
            print(f"   ✅ Port 3306 is OPEN")
            return True
        else:
            print(f"   ❌ Port 3306 is CLOSED or FILTERED")
            print("\n🔧 Troubleshooting:")
            print("   - MySQL di macOS belum running?")
            print("   - Firewall macOS block port 3306?")
            print("   - bind-address di my.cnf bukan 0.0.0.0?")
            return False

    except socket.gaierror:
        print(f"   ❌ Cannot resolve hostname: {MACOS_SERVER_IP}")
        print("\n🔧 Troubleshooting:")
        print("   - IP address salah?")
        print("   - Windows dan macOS tidak di network yang sama?")
        return False

    except Exception as e:
        print(f"   ❌ Network error: {e}")
        return False

def test_mysql_connection():
    """Test koneksi MySQL"""
    print_header("🔌 MYSQL CONNECTION TEST")

    try:
        print(f"\nConnecting to MySQL server at {MACOS_SERVER_IP}:3306...")
        print(f"User: {DB_CONFIG['user']}")
        print(f"Database: {DB_CONFIG['database']}")

        conn = mysql.connector.connect(**DB_CONFIG)

        if conn.is_connected():
            db_info = conn.get_server_info()
            cursor = conn.cursor()
            cursor.execute("SELECT DATABASE();")
            db_name = cursor.fetchone()[0]

            print(f"\n✅ CONNECTION SUCCESS!")
            print(f"   MySQL Server version: {db_info}")
            print(f"   Connected to database: {db_name}")

            cursor.close()
            conn.close()
            return True

    except MySQLError as e:
        print(f"\n❌ MySQL Error: {e}")

        error_code = e.errno

        if error_code == 1045:
            print("\n🔧 Access Denied! Troubleshooting:")
            print("   1. Username atau password salah")
            print("   2. User belum dibuat di MySQL server")
            print("   3. Di macOS, jalankan:")
            print("      mysql -u root -p")
            print("      CREATE USER 'remote_user'@'%' IDENTIFIED BY 'password_kuat_123';")
            print("      GRANT ALL PRIVILEGES ON sistem_parkir_smk.* TO 'remote_user'@'%';")
            print("      FLUSH PRIVILEGES;")

        elif error_code == 2003:
            print("\n🔧 Cannot Connect! Troubleshooting:")
            print("   1. MySQL di macOS belum running")
            print("   2. Jalankan di macOS: brew services start mysql")
            print("   3. Cek firewall: sudo /usr/libexec/ApplicationFirewall/socketfilterfw --listapps")

        elif error_code == 1049:
            print("\n🔧 Database Not Found! Troubleshooting:")
            print("   1. Database 'sistem_parkir_smk' belum dibuat")
            print("   2. Jalankan file database_setup.sql di macOS")

        else:
            print(f"\n🔧 Error code: {error_code}")

        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def get_db_connection():
    """Get database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except MySQLError as e:
        print(f"❌ Connection failed: {e}")
        return None

# =====================================================
# DATABASE VIEWER FUNCTIONS
# =====================================================

def view_vehicles():
    """Lihat daftar kendaraan terdaftar"""
    print_header("🚗 DAFTAR KENDARAAN TERDAFTAR")

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Kendaraan aktif
        print_section("📋 Kendaraan Aktif:")
        cursor.execute("""
            SELECT nomor_plat, nama_pemilik, jenis_kendaraan, nomor_hp,
                   DATE(tanggal_daftar) as tgl_daftar
            FROM kendaraan_terdaftar
            WHERE status = 'aktif'
            ORDER BY tanggal_daftar DESC
        """)
        vehicles = cursor.fetchall()

        if vehicles:
            for v in vehicles:
                hp = v['nomor_hp'] or 'N/A'
                print(f"  Plat: {v['nomor_plat']:12} | {v['nama_pemilik']:30} | "
                      f"{v['jenis_kendaraan'].capitalize():8} | HP: {hp:15} | "
                      f"Daftar: {v['tgl_daftar']}")
            print(f"\n  Total: {len(vehicles)} kendaraan aktif")
        else:
            print("  Tidak ada kendaraan aktif")

        # Kendaraan nonaktif
        print_section("📋 Kendaraan Nonaktif:")
        cursor.execute("""
            SELECT nomor_plat, nama_pemilik, jenis_kendaraan
            FROM kendaraan_terdaftar
            WHERE status = 'nonaktif'
            ORDER BY tanggal_daftar DESC
        """)
        inactive = cursor.fetchall()

        if inactive:
            for v in inactive:
                print(f"  Plat: {v['nomor_plat']:12} | {v['nama_pemilik']:30} | "
                      f"{v['jenis_kendaraan'].capitalize():8}")
            print(f"\n  Total: {len(inactive)} kendaraan nonaktif")
        else:
            print("  Tidak ada kendaraan nonaktif")

        # Statistik per jenis
        print_section("📊 Statistik per Jenis Kendaraan:")
        cursor.execute("""
            SELECT jenis_kendaraan,
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'aktif' THEN 1 ELSE 0 END) as aktif
            FROM kendaraan_terdaftar
            GROUP BY jenis_kendaraan
        """)
        stats = cursor.fetchall()

        for s in stats:
            print(f"  {s['jenis_kendaraan'].capitalize():8} | "
                  f"Total: {s['total']:3} | Aktif: {s['aktif']:3}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()

def view_access_logs():
    """Lihat log akses"""
    print_header("📝 LOG AKSES MASUK")

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Log hari ini
        print_section("📅 Log Akses Hari Ini:")
        cursor.execute("""
            SELECT al.plat_terdeteksi, al.status_akses,
                   TIME(al.waktu_deteksi) as waktu,
                   ROUND(al.tingkat_yakin * 100, 1) as confidence,
                   v.nama_pemilik
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            WHERE DATE(al.waktu_deteksi) = CURDATE()
            ORDER BY al.waktu_deteksi DESC
            LIMIT 20
        """)
        logs_today = cursor.fetchall()

        if logs_today:
            for log in logs_today:
                status_icon = "✅" if log['status_akses'] == 'boleh_masuk' else "❌" if log['status_akses'] == 'ditolak' else "🟡"
                nama = log['nama_pemilik'] or 'Tidak Terdaftar'
                print(f"  {status_icon} {log['waktu']} | {log['plat_terdeteksi']:12} | "
                      f"{log['status_akses']:15} | {log['confidence']:5}% | {nama}")
            print(f"\n  Total: {len(logs_today)} akses hari ini")
        else:
            print("  Belum ada akses hari ini")

        # Log terbaru (semua)
        print_section("📅 Log Terbaru (10 terakhir):")
        cursor.execute("""
            SELECT al.plat_terdeteksi, al.status_akses,
                   DATE_FORMAT(al.waktu_deteksi, '%Y-%m-%d %H:%i:%s') as waktu,
                   ROUND(al.tingkat_yakin * 100, 1) as confidence,
                   v.nama_pemilik
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            ORDER BY al.waktu_deteksi DESC
            LIMIT 10
        """)
        logs_recent = cursor.fetchall()

        if logs_recent:
            for log in logs_recent:
                status_icon = "✅" if log['status_akses'] == 'boleh_masuk' else "❌" if log['status_akses'] == 'ditolak' else "🟡"
                nama = log['nama_pemilik'] or 'Tidak Terdaftar'
                print(f"  {status_icon} {log['waktu']} | {log['plat_terdeteksi']:12} | "
                      f"{log['status_akses']:15} | {nama}")
        else:
            print("  Tidak ada log")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()

def view_statistics():
    """Lihat statistik akses"""
    print_header("📊 STATISTIK AKSES")

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Statistik hari ini
        print_section("📅 Statistik Hari Ini:")
        cursor.execute("""
            SELECT
                COUNT(*) as total_akses,
                SUM(CASE WHEN status_akses = 'boleh_masuk' THEN 1 ELSE 0 END) as boleh_masuk,
                SUM(CASE WHEN status_akses = 'ditolak' THEN 1 ELSE 0 END) as ditolak,
                SUM(CASE WHEN status_akses = 'manual_override' THEN 1 ELSE 0 END) as manual_override,
                ROUND(AVG(tingkat_yakin) * 100, 1) as avg_confidence
            FROM log_akses_masuk
            WHERE DATE(waktu_deteksi) = CURDATE()
        """)
        today = cursor.fetchone()

        print(f"  Total Akses: {today['total_akses']}")
        print(f"  ✅ Boleh Masuk: {today['boleh_masuk']}")
        print(f"  ❌ Ditolak: {today['ditolak']}")
        print(f"  🟡 Manual Override: {today['manual_override']}")
        print(f"  📈 Avg Confidence: {today['avg_confidence']}%")

        # Statistik minggu ini
        print_section("📅 Statistik Minggu Ini:")
        cursor.execute("""
            SELECT
                COUNT(*) as total_akses,
                SUM(CASE WHEN status_akses = 'boleh_masuk' THEN 1 ELSE 0 END) as boleh_masuk,
                SUM(CASE WHEN status_akses = 'ditolak' THEN 1 ELSE 0 END) as ditolak
            FROM log_akses_masuk
            WHERE YEARWEEK(waktu_deteksi) = YEARWEEK(NOW())
        """)
        week = cursor.fetchone()

        print(f"  Total Akses: {week['total_akses']}")
        print(f"  ✅ Boleh Masuk: {week['boleh_masuk']}")
        print(f"  ❌ Ditolak: {week['ditolak']}")

        # Statistik per hari (7 hari terakhir)
        print_section("📅 Trend 7 Hari Terakhir:")
        cursor.execute("""
            SELECT
                DATE(waktu_deteksi) as tanggal,
                COUNT(*) as total,
                SUM(CASE WHEN status_akses = 'boleh_masuk' THEN 1 ELSE 0 END) as masuk,
                SUM(CASE WHEN status_akses = 'ditolak' THEN 1 ELSE 0 END) as tolak
            FROM log_akses_masuk
            WHERE waktu_deteksi >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            GROUP BY DATE(waktu_deteksi)
            ORDER BY tanggal DESC
        """)
        trend = cursor.fetchall()

        if trend:
            for t in trend:
                print(f"  {t['tanggal']} | Total: {t['total']:3} | "
                      f"✅ {t['masuk']:3} | ❌ {t['tolak']:3}")
        else:
            print("  Tidak ada data")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()

def view_top_vehicles():
    """Lihat kendaraan paling aktif"""
    print_header("🏆 TOP KENDARAAN PALING AKTIF")

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Top minggu ini
        print_section("📅 Top 10 Kendaraan Minggu Ini:")
        cursor.execute("""
            SELECT
                al.plat_terdeteksi,
                v.nama_pemilik,
                v.jenis_kendaraan,
                COUNT(*) as jumlah_akses,
                MAX(al.waktu_deteksi) as akses_terakhir
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            WHERE al.status_akses = 'boleh_masuk'
              AND YEARWEEK(al.waktu_deteksi) = YEARWEEK(NOW())
            GROUP BY al.plat_terdeteksi, v.nama_pemilik, v.jenis_kendaraan
            ORDER BY jumlah_akses DESC
            LIMIT 10
        """)
        top_week = cursor.fetchall()

        if top_week:
            for i, tv in enumerate(top_week, 1):
                nama = tv['nama_pemilik'] or 'Tidak Terdaftar'
                jenis = tv['jenis_kendaraan'].capitalize() if tv['jenis_kendaraan'] else 'N/A'
                last_access = tv['akses_terakhir'].strftime("%Y-%m-%d %H:%M")
                print(f"  #{i:2}. {tv['plat_terdeteksi']:12} | {nama:30} | "
                      f"{jenis:8} | {tv['jumlah_akses']:3}x | Last: {last_access}")
        else:
            print("  Tidak ada data minggu ini")

        # Top bulan ini
        print_section("📅 Top 10 Kendaraan Bulan Ini:")
        cursor.execute("""
            SELECT
                al.plat_terdeteksi,
                v.nama_pemilik,
                COUNT(*) as jumlah_akses
            FROM log_akses_masuk al
            LEFT JOIN kendaraan_terdaftar v ON al.plat_terdeteksi = v.nomor_plat
            WHERE al.status_akses = 'boleh_masuk'
              AND YEAR(al.waktu_deteksi) = YEAR(NOW())
              AND MONTH(al.waktu_deteksi) = MONTH(NOW())
            GROUP BY al.plat_terdeteksi, v.nama_pemilik
            ORDER BY jumlah_akses DESC
            LIMIT 10
        """)
        top_month = cursor.fetchall()

        if top_month:
            for i, tv in enumerate(top_month, 1):
                nama = tv['nama_pemilik'] or 'Tidak Terdaftar'
                print(f"  #{i:2}. {tv['plat_terdeteksi']:12} | {nama:30} | {tv['jumlah_akses']:3}x")
        else:
            print("  Tidak ada data bulan ini")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()

def search_vehicle():
    """Cari kendaraan tertentu"""
    print_header("🔍 CARI KENDARAAN")

    plat = input("\nMasukkan nomor plat (contoh: B1234ABC): ").strip().upper().replace(' ', '')

    if not plat:
        print("❌ Plat nomor tidak boleh kosong")
        return

    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor(dictionary=True)

        # Cari di database
        print_section(f"📋 Data Kendaraan: {plat}")
        cursor.execute("""
            SELECT * FROM kendaraan_terdaftar
            WHERE nomor_plat = %s
        """, (plat,))
        vehicle = cursor.fetchone()

        if vehicle:
            print(f"  Plat: {vehicle['nomor_plat']}")
            print(f"  Pemilik: {vehicle['nama_pemilik']}")
            print(f"  Jenis: {vehicle['jenis_kendaraan'].capitalize()}")
            print(f"  Status: {vehicle['status'].capitalize()}")
            print(f"  No HP: {vehicle['nomor_hp'] or 'N/A'}")
            print(f"  Tgl Daftar: {vehicle['tanggal_daftar']}")
        else:
            print(f"  ❌ Kendaraan dengan plat '{plat}' tidak terdaftar")

        # History akses
        print_section(f"📝 History Akses (10 terakhir):")
        cursor.execute("""
            SELECT status_akses, waktu_deteksi,
                   ROUND(tingkat_yakin * 100, 1) as confidence
            FROM log_akses_masuk
            WHERE plat_terdeteksi = %s
            ORDER BY waktu_deteksi DESC
            LIMIT 10
        """, (plat,))
        history = cursor.fetchall()

        if history:
            for h in history:
                status_icon = "✅" if h['status_akses'] == 'boleh_masuk' else "❌"
                waktu = h['waktu_deteksi'].strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {status_icon} {waktu} | {h['status_akses']:15} | {h['confidence']}%")

            # Statistik
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status_akses = 'boleh_masuk' THEN 1 ELSE 0 END) as masuk
                FROM log_akses_masuk
                WHERE plat_terdeteksi = %s
            """, (plat,))
            stats = cursor.fetchone()
            print(f"\n  Total Akses: {stats['total']} | Berhasil: {stats['masuk']}")
        else:
            print("  Belum ada history akses")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()

# =====================================================
# MENU SYSTEM
# =====================================================

def show_menu():
    """Tampilkan menu utama"""
    print("\n" + "=" * 70)
    print("  DATABASE VIEWER - Sistem Parkir SMK")
    print("=" * 70)
    print("\n  MENU:")
    print("  [1] 🚗 Lihat Kendaraan Terdaftar")
    print("  [2] 📝 Lihat Log Akses")
    print("  [3] 📊 Lihat Statistik")
    print("  [4] 🏆 Top Kendaraan Aktif")
    print("  [5] 🔍 Cari Kendaraan")
    print("  [6] 🔧 Test Connection")
    print("  [0] ❌ Keluar")
    print("-" * 70)

def main():
    """Main program"""
    print("\n" + "=" * 70)
    print("  🚗 DATABASE VIEWER - Remote MySQL Access")
    print("  Sistem Deteksi Plat Nomor SMK")
    print("=" * 70)
    print(f"\n  Server: {MACOS_SERVER_IP}:3306")
    print(f"  Database: {DB_CONFIG['database']}")
    print(f"  User: {DB_CONFIG['user']}")

    # Test koneksi awal
    print("\n  Testing connection...")
    network_ok = test_network_connectivity()

    if network_ok:
        mysql_ok = test_mysql_connection()

        if not mysql_ok:
            print("\n❌ Connection failed! Please fix the issues above.")
            input("\nPress Enter to exit...")
            return
    else:
        print("\n❌ Network not reachable! Please check network configuration.")
        input("\nPress Enter to exit...")
        return

    print("\n✅ Ready to use!")

    # Menu loop
    while True:
        show_menu()
        choice = input("  Pilih menu: ").strip()

        if choice == '1':
            view_vehicles()
        elif choice == '2':
            view_access_logs()
        elif choice == '3':
            view_statistics()
        elif choice == '4':
            view_top_vehicles()
        elif choice == '5':
            search_vehicle()
        elif choice == '6':
            test_network_connectivity()
            test_mysql_connection()
        elif choice == '0':
            print("\n👋 Terima kasih! Goodbye.\n")
            break
        else:
            print("\n❌ Pilihan tidak valid!")

        input("\nPress Enter to continue...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program dihentikan oleh user. Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
