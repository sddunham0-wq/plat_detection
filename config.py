import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Konfigurasi sistem deteksi plat nomor"""

    # Konfigurasi Flask Secret Key
    # Generate dengan: python3 -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-please-change-in-production')

    # Konfigurasi Kamera
    CAMERA_HOST = os.getenv('CAMERA_HOST', '192.168.1.203')
    CAMERA_PORT = os.getenv('CAMERA_PORT', '5503')
    CAMERA_USER = os.getenv('CAMERA_USER', 'admin')
    CAMERA_PASSWORD = os.getenv('CAMERA_PASSWORD', '')
    CAMERA_CHANNEL = os.getenv('CAMERA_CHANNEL', '1')
    CAMERA_SUBTYPE = os.getenv('CAMERA_SUBTYPE', '0')

    @property
    def CAMERA_URL(self):
        """Generate RTSP URL dari konfigurasi"""
        return f"rtsp://{self.CAMERA_USER}:{self.CAMERA_PASSWORD}@{self.CAMERA_HOST}:{self.CAMERA_PORT}/cam/realmonitor?channel={self.CAMERA_CHANNEL}&subtype={self.CAMERA_SUBTYPE}"

    # Konfigurasi Database MySQL (Laragon)
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '3306'))
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_NAME = os.getenv('DB_NAME', 'sistem_parkir_smk')

    # Konfigurasi Sistem
    SAVE_FOLDER = os.getenv('SAVE_FOLDER', 'gambarplat')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_RETRY_CAMERA = int(os.getenv('MAX_RETRY_CAMERA', '5'))
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5001'))  # Default 5001 (avoid macOS AirPlay port 5000)

    # Konfigurasi Deteksi
    # Ukuran minimum plat yang akan dideteksi
    MIN_PLATE_WIDTH = 100
    MIN_PLATE_HEIGHT = 30
    MIN_ASPECT_RATIO = 1.5  # Plat nomor biasanya lebih lebar dari tinggi
    MAX_ASPECT_RATIO = 6.0  # Batas maksimal rasio lebar:tinggi

    # Konfigurasi OCR
    OCR_CONFIG = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Konfigurasi Preprocessing
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MORPH_KERNEL_SIZE = (3, 3)
    CANNY_THRESHOLD1 = 50
    CANNY_THRESHOLD2 = 150

    # Confidence threshold untuk deteksi
    MIN_CONFIDENCE = 0.6

    # Konfigurasi untuk validasi plat Indonesia
    INDONESIAN_PLATE_PATTERNS = [
        r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}$',  # Format umum: B 1234 ABC
        r'^[A-Z]{2}\s?\d{1,4}\s?[A-Z]{1,2}$',    # Format 2 huruf: AB 1234 CD
        r'^[A-Z]\s?\d{1,4}\s?[A-Z]{1,3}$'        # Format 1 huruf: B 1234 ABC
    ]

    def __init__(self):
        """Inisialisasi dan validasi konfigurasi"""
        self._create_folders()
        self._validate_config()

    def _create_folders(self):
        """Buat folder yang diperlukan jika belum ada"""
        if not os.path.exists(self.SAVE_FOLDER):
            os.makedirs(self.SAVE_FOLDER)
            print(f"✅ Folder {self.SAVE_FOLDER} dibuat")

    def _validate_config(self):
        """Validasi konfigurasi"""
        # Check .env file existence
        if not os.path.exists('.env'):
            print("⚠️  WARNING: File .env tidak ditemukan!")
            print("📝 SOLUSI:")
            print("   1. Copy file '.env.example' jadi '.env'")
            print("   2. Edit '.env' dan isi password kamera Anda")
            print("   3. Generate SECRET_KEY dengan: python3 -c \"import secrets; print(secrets.token_hex(32))\"")
            print("   Untuk sementara menggunakan default values (not recommended for production)\n")

        # Check camera password
        if not self.CAMERA_PASSWORD:
            print("⚠️  WARNING: CAMERA_PASSWORD belum diset!")
            print("📝 SOLUSI: Tambahkan 'CAMERA_PASSWORD=your_password' di file .env\n")

        # Check secret key (development warning)
        if self.SECRET_KEY == 'dev-secret-key-please-change-in-production':
            print("⚠️  WARNING: SECRET_KEY masih menggunakan default!")
            print("📝 SOLUSI: Generate random key dengan:")
            print("   python3 -c \"import secrets; print(secrets.token_hex(32))\"")
            print("   Lalu tambahkan ke .env file\n")

        # Info
        print(f"📹 Kamera: {self.CAMERA_HOST}:{self.CAMERA_PORT}")
        print(f"📁 Save folder: {self.SAVE_FOLDER}")
        print(f"💾 Database: MySQL ({self.DB_USER}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME})")
        print("")

# Instance global config
config = Config()