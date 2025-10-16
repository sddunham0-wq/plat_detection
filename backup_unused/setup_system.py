#!/usr/bin/env python3
"""
Setup System for Enhanced Indonesian License Plate Detection
Script untuk mempersiapkan sistem sebelum pertama kali dijalankan
"""

import os
import sys
import subprocess
import mysql.connector
from pathlib import Path

def print_header():
    """Print header setup"""
    print("=" * 60)
    print("🚗 ENHANCED INDONESIAN LICENSE PLATE DETECTION SYSTEM")
    print("🔧 SETUP & INITIALIZATION SCRIPT")
    print("=" * 60)
    print()

def check_python_version():
    """Cek versi Python"""
    print("📋 Checking Python version...")

    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ diperlukan!")
        print(f"   Versi saat ini: {sys.version}")
        return False

    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_directories():
    """Buat semua direktori yang diperlukan"""
    print("\n📁 Creating required directories...")

    directories = [
        'logs',
        'gambarplat',
        'utils'
    ]

    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Directory created/verified: {directory}/")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")
            return False

    return True

def check_env_file():
    """Cek dan setup file .env"""
    print("\n🔐 Checking environment configuration...")

    if not os.path.exists('.env'):
        print("⚠️ File .env tidak ditemukan!")
        print("📝 Membuat template .env...")

        template = '''# Konfigurasi Kamera CCTV
CAMERA_HOST=192.168.1.203
CAMERA_PORT=5503
CAMERA_USER=admin
CAMERA_PASSWORD=YourPassword123!
CAMERA_CHANNEL=1
CAMERA_SUBTYPE=0

# Konfigurasi Database
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=
DB_NAME=deteksi_plat_db

# Konfigurasi Sistem
SAVE_FOLDER=gambarplat
LOG_LEVEL=INFO
MAX_RETRY_CAMERA=5
'''

        try:
            with open('.env', 'w') as f:
                f.write(template)
            print("✅ Template .env berhasil dibuat")
            print("⚠️ PENTING: Edit file .env dengan konfigurasi Anda!")
            return True
        except Exception as e:
            print(f"❌ Error membuat .env: {e}")
            return False
    else:
        print("✅ File .env sudah ada")
        return True

def install_dependencies():
    """Install dependencies dari requirements.txt"""
    print("\n📦 Installing Python dependencies...")

    if not os.path.exists('requirements.txt'):
        print("❌ File requirements.txt tidak ditemukan!")
        return False

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Dependencies berhasil diinstall")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 Coba jalankan manual: pip install -r requirements.txt")
        return False

def check_tesseract():
    """Cek instalasi Tesseract OCR"""
    print("\n🔍 Checking Tesseract OCR...")

    try:
        result = subprocess.run(['tesseract', '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ Tesseract found: {version}")
            return True
        else:
            print("❌ Tesseract tidak ditemukan!")
            return False
    except FileNotFoundError:
        print("❌ Tesseract tidak terinstall!")
        print("💡 Install guide:")
        print("   - Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   - Ubuntu: sudo apt install tesseract-ocr")
        print("   - macOS: brew install tesseract")
        return False

def test_database_connection():
    """Test koneksi database"""
    print("\n💾 Testing database connection...")

    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv belum terinstall, menggunakan default values")

    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'deteksi_plat_db')
    }

    try:
        # Test koneksi ke MySQL server
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )

        cursor = conn.cursor()

        # Cek apakah database exists
        cursor.execute(f"SHOW DATABASES LIKE '{db_config['database']}'")
        db_exists = cursor.fetchone()

        if not db_exists:
            print(f"⚠️ Database '{db_config['database']}' belum ada")
            print("🔧 Membuat database...")

            cursor.execute(f"CREATE DATABASE {db_config['database']}")
            print(f"✅ Database '{db_config['database']}' berhasil dibuat")
        else:
            print(f"✅ Database '{db_config['database']}' sudah ada")

        # Test koneksi ke database
        conn.close()

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Cek table tb_kendaraan
        cursor.execute("SHOW TABLES LIKE 'tb_kendaraan'")
        table_exists = cursor.fetchone()

        if not table_exists:
            print("🔧 Membuat table tb_kendaraan...")

            create_table_sql = '''
            CREATE TABLE tb_kendaraan (
                Id_Kendaraan INT AUTO_INCREMENT PRIMARY KEY,
                Plat_Nomor VARCHAR(20) NOT NULL,
                Foto_Plat VARCHAR(255),
                Confidence_Score FLOAT DEFAULT 0.0,
                Waktu_Deteksi TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''

            cursor.execute(create_table_sql)
            print("✅ Table tb_kendaraan berhasil dibuat")
        else:
            print("✅ Table tb_kendaraan sudah ada")

        # Add Confidence_Score column if not exists (for compatibility)
        try:
            cursor.execute("ALTER TABLE tb_kendaraan ADD COLUMN Confidence_Score FLOAT DEFAULT 0.0")
            print("✅ Column Confidence_Score ditambahkan ke table existing")
        except mysql.connector.Error:
            pass  # Column sudah ada

        conn.close()
        print("✅ Database connection test berhasil")
        return True

    except mysql.connector.Error as e:
        print(f"❌ Database connection error: {e}")
        print("💡 Pastikan MySQL/XAMPP sudah running")
        print("💡 Cek kredensial database di file .env")
        return False

def validate_files():
    """Validasi file-file penting"""
    print("\n📋 Validating system files...")

    required_files = [
        'config.py',
        'utils/plate_validator.py',
        'deteksi_plat_enhanced.py',
        'app.py'
    ]

    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING!")
            all_ok = False

    return all_ok

def print_next_steps():
    """Print langkah selanjutnya"""
    print("\n" + "=" * 60)
    print("🎉 SETUP SELESAI!")
    print("=" * 60)
    print()
    print("📋 LANGKAH SELANJUTNYA:")
    print()
    print("1. 🔧 Edit file .env dengan konfigurasi kamera Anda:")
    print("   - CAMERA_HOST: IP address kamera")
    print("   - CAMERA_PASSWORD: Password kamera")
    print()
    print("2. 🚀 Jalankan sistem deteksi:")
    print("   python3 deteksi_plat_enhanced.py")
    print()
    print("3. 🌐 Jalankan web interface (terminal terpisah):")
    print("   python3 app.py")
    print()
    print("4. 🌟 Akses web dashboard:")
    print("   http://localhost:5001")
    print()
    print("=" * 60)

def main():
    """Main setup function"""
    print_header()

    # Checklist setup
    checks = [
        ("Python Version", check_python_version),
        ("Directories", create_directories),
        ("Environment File", check_env_file),
        ("Dependencies", install_dependencies),
        ("Tesseract OCR", check_tesseract),
        ("Database", test_database_connection),
        ("System Files", validate_files)
    ]

    failed_checks = []

    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            print(f"❌ Error during {check_name}: {e}")
            failed_checks.append(check_name)

    print("\n" + "=" * 60)
    print("📊 SETUP SUMMARY")
    print("=" * 60)

    if failed_checks:
        print("❌ SETUP FAILED!")
        print("🔴 Failed checks:")
        for check in failed_checks:
            print(f"   - {check}")
        print()
        print("💡 Perbaiki masalah di atas sebelum menjalankan sistem")
    else:
        print("✅ SETUP BERHASIL!")
        print_next_steps()

if __name__ == "__main__":
    main()