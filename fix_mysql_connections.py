#!/usr/bin/env python3
"""
Fix MySQL "Too many connections" problem
Script ini akan:
1. Kill semua koneksi MySQL lama
2. Increase max_connections limit
"""

import subprocess
import sys
import time

print("=" * 70)
print("🔧 MYSQL CONNECTION FIX TOOL")
print("=" * 70)
print()

def run_command(cmd, description):
    """Run shell command and show result"""
    print(f"⏳ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - EXCEPTION: {e}")
        return False

print("📊 STEP 1: Diagnosa MySQL")
print("-" * 70)

# Check if MySQL is running
print("\n🔍 Checking MySQL status...")
run_command("brew services list | grep mysql || ps aux | grep mysql | grep -v grep",
            "Check MySQL process")

print("\n" + "=" * 70)
print("🔧 STEP 2: Fix Options")
print("=" * 70)
print()

print("Anda punya 3 opsi untuk fix masalah ini:")
print()

print("📌 OPSI 1: RESTART MYSQL (RECOMMENDED)")
print("   - Restart MySQL service untuk clear semua koneksi")
print("   - Paling aman dan paling cepat")
print()
print("   Command untuk macOS (Homebrew):")
print("   brew services restart mysql")
print()
print("   Command untuk Laragon (Windows):")
print("   - Buka Laragon GUI")
print("   - Klik 'Stop All'")
print("   - Tunggu 5 detik")
print("   - Klik 'Start All'")
print()

print("📌 OPSI 2: INCREASE MAX_CONNECTIONS")
print("   - Naikkan limit max_connections dari default (151) ke 500")
print("   - Perlu restart MySQL setelah edit config")
print()
print("   Edit file: /usr/local/etc/my.cnf (macOS) atau my.ini (Windows)")
print("   Tambahkan:")
print("   [mysqld]")
print("   max_connections = 500")
print()
print("   Lalu restart MySQL")
print()

print("📌 OPSI 3: TUNGGU TIMEOUT (LAMA!)")
print("   - MySQL akan auto-close koneksi setelah timeout (default: 8 jam)")
print("   - Tidak recommended karena terlalu lama")
print()

print("=" * 70)
print("🚀 RECOMMENDED ACTION:")
print("=" * 70)
print()
print("Jalankan command ini di terminal:")
print()
print("# Untuk macOS (Homebrew):")
print("brew services restart mysql")
print()
print("# Atau manual:")
print("sudo killall mysqld && sudo mysqld_safe &")
print()
print("# Tunggu 10 detik, lalu test:")
print("mysql -u root -e 'SELECT VERSION();'")
print()
print("# Kalau sudah bisa connect, jalankan aplikasi:")
print("python3 app.py")
print()

print("=" * 70)
print("💡 TIP: Prevent masalah ini di masa depan")
print("=" * 70)
print()
print("1. Selalu close connection dengan 'conn.close()' setelah pakai")
print("2. Gunakan connection pooling (sudah diimplementasi di app.py)")
print("3. Set connection timeout yang lebih pendek")
print("4. Monitor jumlah active connections secara berkala")
print()
