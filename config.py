"""
Configuration file untuk Live CCTV License Plate Detection System
Pengaturan lengkap untuk deteksi plat nomor real-time
"""

import os

class CCTVConfig:
    """Pengaturan CCTV dan Video Stream"""
    
    # Default video sources
    # Try different ports - uncomment to test
    DEFAULT_RTSP_URL = "rtsp://admin:H4nd4l9165!@192.168.1.195:554/85"      # Auto-detected working URL ✅
    # DEFAULT_RTSP_URL = "rtsp://:H4nd4l9165!@192.168.1.195:8554/85"   # Alternative RTSP port
    # DEFAULT_RTSP_URL = "rtsp://:H4nd4l9165!@192.168.1.195:8005/85"   # Original port
    DEFAULT_WEBCAM = 0      # Default to laptop camera (index 0)
    DEFAULT_LAPTOP_CAMERA = 0  # Laptop built-in camera index
    DEFAULT_VIDEO_FILE = "video_cctv.mp4"
    
    # Video processing settings
    FRAME_WIDTH = 480          # Lebar frame untuk processing (optimized for speed)
    FRAME_HEIGHT = 360         # Tinggi frame untuk processing (optimized for speed)
    FPS_LIMIT = 25             # Maksimal FPS untuk processing (increased for smoother streaming)
    BUFFER_SIZE = 5            # Maksimal frame di buffer (reduced for lower latency)
    
    # Frame skipping for performance (NEW)
    ENABLE_FRAME_SKIPPING = True   # Enable frame skipping for better performance
    PROCESS_EVERY_N_FRAMES = 2     # Process every 2nd frame (skip 1 frame)
    
    # RTSP connection settings
    RTSP_TIMEOUT = 10          # Timeout koneksi RTSP (detik)
    RECONNECT_DELAY = 5        # Delay sebelum reconnect (detik)
    MAX_RECONNECT_ATTEMPTS = 3 # Maksimal percobaan reconnect

class TesseractConfig:
    """Pengaturan Tesseract OCR untuk plat nomor Indonesia"""
    
    # Path ke tesseract executable (sesuaikan dengan instalasi Anda)
    # Windows: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Linux/Mac: '/usr/bin/tesseract' atau '/opt/homebrew/bin/tesseract'
    TESSERACT_PATH = '/opt/homebrew/bin/tesseract'  # Default untuk macOS Homebrew
    
    # OCR Configuration untuk plat nomor
    # --psm 7: Treat the image as a single text line (OPTIMIZED for Indonesian plates)
    # --oem 3: Default OCR Engine Mode
    # tessedit_char_whitelist: Hanya izinkan karakter ini
    OCR_CONFIG = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Alternative configurations untuk different scenarios
    OCR_CONFIG_SINGLE_WORD = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    OCR_CONFIG_RAW_LINE = '--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Language - gunakan bahasa Indonesia + English untuk plat nomor Indonesia
    LANGUAGE = 'ind+eng'  # Indonesia + English untuk hasil terbaik
    FALLBACK_LANGUAGE = 'eng'  # Fallback jika Indonesia tidak tersedia
    
    # Confidence threshold (0-100) - OPTIMIZED for Indonesian plates
    MIN_CONFIDENCE = 40        # Minimal confidence untuk accept hasil OCR (lowered for speed)
    INDONESIAN_MIN_CONFIDENCE = 25  # Lowered for faster processing
    
    # Auto language detection - OPTIMIZED thresholds
    ENABLE_AUTO_LANGUAGE = True  # Enable auto-detection berdasarkan confidence
    LANGUAGE_SWITCH_THRESHOLD = 20  # Lowered from 35 to 20 - less aggressive fallback
    
    # Indonesian plate specific optimizations
    USE_MULTIPLE_PSM = False    # Disabled for faster processing (use single PSM mode)
    PSM_PRIORITY = [7]          # Single PSM mode for speed

class IndonesianPlateConfig:
    """Pengaturan khusus untuk plat nomor Indonesia"""
    
    # Pattern validation untuk format plat Indonesia
    ENABLE_PATTERN_VALIDATION = True
    
    # Indonesian plate patterns (regex)
    # Format: X####XXX (misal: B1234ABC)
    PLATE_PATTERNS = [
        r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{2,3}$',  # Standard car plate
        r'^[ABDEFGHJKLNPRSTU]\s*\d{1,4}\s*[A-Z]{1,2}$',   # Some variations
        r'^\d{1,4}\s*[A-Z]{2,4}$',                        # Number first format
        r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'          # General format
    ]
    
    # Regional codes untuk validation
    REGIONAL_CODES = [
        'B', 'D', 'E', 'F', 'G', 'H',  # Jakarta, Bandung, Cirebon, etc
        'L', 'M', 'N', 'P', 'R', 'S',  # Surabaya, Madura, etc
        'T', 'U', 'AA', 'AB', 'AD',     # Other regions
        'AG', 'BA', 'BB', 'BD', 'BE'    # Sumatra, etc
    ]
    
    # Character corrections untuk Indonesian OCR
    CHAR_CORRECTIONS = {
        'O': '0',  # Often confused
        'I': '1',  # Often confused
        'S': '5',  # Sometimes confused
        'Z': '2',  # Sometimes confused
        '8': 'B',  # B and 8 confusion
        '6': 'G',  # G and 6 confusion
    }
    
    # Text cleaning rules
    REMOVE_CHARS = ['-', '_', '.', ',', ':', ';', '|', '/', '\\']
    REPLACE_MULTIPLE_SPACES = True
    MIN_PLATE_LENGTH = 6            # Increased from 5 to 6 (minimum B123CD format)
    MAX_PLATE_LENGTH = 10
    
    # Confidence boosting untuk Indonesian plates
    PATTERN_MATCH_BOOST = 10.0  # Boost confidence jika match pattern
    REGIONAL_CODE_BOOST = 5.0   # Boost jika ada regional code
    
    # Strict validation settings (RELAXED for debugging)
    ENABLE_STRICT_PATTERN_VALIDATION = False  # Temporarily disabled for CCTV debugging
    REJECT_NON_PATTERN_MATCHES = False        # Allow non-pattern matches for debugging
    MIN_REGIONAL_CODE_MATCH = False           # Temporarily disable regional code requirement
    
    # Preprocessing optimization untuk plat Indonesia
    CONTRAST_ENHANCEMENT = 1.5   # Enhance contrast untuk plat hitam-putih
    NOISE_REDUCTION_KERNEL = (2, 2)  # Kernel untuk noise reduction
    MORPHOLOGY_KERNEL = (3, 3)   # Kernel untuk morphology operations

class DetectionConfig:
    """Pengaturan deteksi plat nomor dengan enhanced algorithms"""
    
    # Enhanced preprocessing settings
    GAUSSIAN_BLUR_KERNEL = (5, 5)     # Kernel untuk blur
    ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11  # Block size untuk threshold
    ADAPTIVE_THRESHOLD_C = 2           # Constant untuk threshold
    
    # Enhanced contour detection (more accurate)
    MIN_CONTOUR_AREA = 500            # Reduced for better small plate detection
    MAX_CONTOUR_AREA = 35000          # Reduced to filter out large non-plate objects
    
    # Improved aspect ratio untuk plat nomor Indonesia
    MIN_ASPECT_RATIO = 1.5            # More tolerant for angled plates
    MAX_ASPECT_RATIO = 5.5            # Adjusted based on real plate observations
    
    # ROI (Region of Interest) - area deteksi dalam persen
    # Format: (x%, y%, width%, height%)
    ROI_AREA = (0.05, 0.2, 0.9, 0.6)  # Expanded area for better coverage
    
    # Enhanced duplicate detection
    DUPLICATE_THRESHOLD = 5           # Increased for faster processing
    MIN_PLATE_LENGTH = 5             # Minimal panjang karakter plat
    MAX_PLATE_LENGTH = 12            # Maksimal panjang karakter plat
    
    # Color-based detection thresholds
    ENABLE_COLOR_FILTERING = True     # Enable Indonesian plate color detection
    MIN_COLOR_CONFIDENCE = 15.0       # Minimum color confidence for regular plates
    MOTORCYCLE_MIN_COLOR_CONFIDENCE = 10.0  # Lower threshold for motorcycles
    
    # Geometric validation thresholds
    MIN_RECTANGULARITY = 0.7          # Minimum rectangularity score
    MIN_SOLIDITY = 0.8               # Minimum solidity score
    MIN_EXTENT = 0.7                 # Minimum extent score
    
    # Temporal smoothing settings
    ENABLE_TEMPORAL_SMOOTHING = True  # Enable detection tracking
    MIN_TRACKING_FRAMES = 3          # Minimum frames for stable detection
    TRACKING_TIMEOUT = 10.0          # Seconds to keep tracking history
    STABILITY_CONFIDENCE_BOOST = 10.0 # Confidence boost for stable detections

class MotorcycleDetectionConfig:
    """Pengaturan khusus deteksi plat motor dengan enhanced algorithms"""
    
    # Enhanced motorcycle-specific contour detection
    MIN_CONTOUR_AREA = 200            # Lowered from 500 to 200 for distant CCTV plates
    MAX_CONTOUR_AREA = 20000          # Reduced to filter large false positives
    
    # Enhanced aspect ratio untuk plat motor Indonesia
    MIN_ASPECT_RATIO = 1.2            # Tolerant for various angles
    MAX_ASPECT_RATIO = 4.5            # Adjusted for typical motorcycle plates
    
    # Enhanced ROI khusus motor (expanded coverage)
    ROI_AREA = (0.0, 0.0, 1.0, 1.0)  # Full frame for distant motorcycle capture
    
    # Enhanced size constraints untuk plat motor
    MIN_PLATE_WIDTH = 15              # Lowered from 25 to 15 for very distant CCTV plates
    MIN_PLATE_HEIGHT = 8              # Lowered from 10 to 8 for very distant CCTV plates
    MAX_PLATE_WIDTH = 180             # Slightly increased upper bound
    MAX_PLATE_HEIGHT = 90             # Slightly increased upper bound
    
    # Enhanced OCR optimization untuk plat motor kecil
    UPSCALE_FACTOR = 6.0              # Increased upscaling for better OCR
    MIN_OCR_HEIGHT = 20               # Further reduced for small plates
    
    # Enhanced detection confidence (optimized for stability)
    MIN_CONFIDENCE = 35               # Lowered from 50 to 35 for real-world CCTV conditions
    MOTORCYCLE_PRIORITY = True        # Prioritas deteksi untuk motor
    
    # Enhanced extreme distance detection settings
    ENABLE_EXTREME_UPSCALING = True   # Enable upscaling ekstrem
    EXTREME_UPSCALE_FACTOR = 10.0     # Increased for better quality
    USE_INTERPOLATION_CUBIC = True    # Gunakan cubic interpolation
    ENABLE_NOISE_REDUCTION = True     # Enable noise reduction untuk plat kecil
    
    # Enhanced geometric validation (more tolerant for motorcycles)
    MIN_RECTANGULARITY = 0.6          # More tolerant than regular plates
    MIN_SOLIDITY = 0.7               # More tolerant for small/distorted plates
    MIN_EXTENT = 0.6                 # More tolerant for perspective distortion

class DatabaseConfig:
    """Pengaturan database untuk menyimpan hasil"""
    
    DATABASE_PATH = "detected_plates.db"
    
    # Table schema
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        plate_text TEXT NOT NULL,
        confidence REAL,
        image_path TEXT,
        source_info TEXT,
        processed_time REAL
    )
    """

class SystemConfig:
    """Pengaturan sistem dan logging"""
    
    # Folders
    OUTPUT_FOLDER = "detected_plates"     # Folder untuk simpan gambar hasil
    LOG_FOLDER = "logs"                   # Folder untuk log files
    
    # Logging
    LOG_LEVEL = "INFO"                    # DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Performance (OPTIMIZED for streaming)
    MAX_THREADS = 2                       # Reduced threads for stability
    MEMORY_LIMIT_MB = 256                # Reduced memory usage for speed
    
    # Display settings
    SHOW_PREVIEW = True                  # Tampilkan preview window
    PREVIEW_WINDOW_SIZE = (800, 600)     # Ukuran preview window
    SHOW_ROI = True                      # Tampilkan ROI box di preview
    SHOW_DETECTIONS = True               # Tampilkan detection box

class AlertConfig:
    """Pengaturan alert dan notifikasi"""
    
    # Enable/disable alerts
    ENABLE_ALERTS = True
    
    # Watchlist plat nomor (untuk alert khusus)
    WATCHLIST_PLATES = [
        "B1234ABC",
        "D5678XYZ"
    ]
    
    # Blacklist plat nomor (untuk alert security)
    BLACKLIST_PLATES = [
        "B9999XXX"
    ]
    
    # Alert methods (bisa dikembangkan untuk email/telegram/dll)
    ALERT_CONSOLE = True
    ALERT_LOG_FILE = True

class TrackingConfig:
    """Pengaturan sistem tracking objek dan plat nomor"""
    
    # Enable/disable tracking system (DISABLED for streaming performance)
    ENABLE_TRACKING = False
    
    # Object tracking parameters
    MAX_DISAPPEARED_FRAMES = 30      # Max frame objek hilang sebelum dihapus
    MAX_TRACKING_DISTANCE = 100      # Max distance untuk matching (pixel)
    MIN_HITS_FOR_CONFIRMATION = 3    # Min deteksi berturut sebelum konfirmasi tracking
    IOU_THRESHOLD = 0.3              # IoU threshold untuk matching
    
    # Plate tracking parameters
    PLATE_CONFIRMATION_THRESHOLD = 3  # Min deteksi untuk konfirmasi plat
    MAX_PLATE_AGE = 10.0             # Max umur plat sebelum dihapus (detik)
    
    # Kalman filter settings
    USE_KALMAN_FILTER = True         # Enable Kalman filter untuk smooth tracking
    USE_ADAPTIVE_NOISE = True        # Enable adaptive noise adjustment
    KALMAN_MAX_AGE = 30              # Max frames untuk Kalman tracker
    KALMAN_MIN_HITS = 3              # Min hits untuk Kalman tracker confirmation
    
    # Temporal smoothing
    SMOOTHING_WINDOW = 5             # Frame window untuk temporal smoothing
    CONFIDENCE_ACCUMULATION = True    # Enable confidence accumulation over time
    
    # Visual tracking settings
    SHOW_TRACKING_IDS = True         # Tampilkan tracking IDs di display
    SHOW_TRACKING_TRAILS = False     # Tampilkan trails (belum implemented)
    SHOW_PREDICTION_BOXES = False    # Tampilkan predicted bounding boxes
    SHOW_TRACKING_STATS = True       # Tampilkan tracking statistics
    
    # Association settings
    VEHICLE_PLATE_ASSOCIATION = True  # Enable vehicle-plate association
    MAX_ASSOCIATION_DISTANCE = 80    # Max distance untuk associate plate dengan vehicle
    ASSOCIATION_IOU_THRESHOLD = 0.1   # Min IoU untuk association

class EnhancedDetectionConfig:
    """Pengaturan Enhanced Detection untuk jarak jauh dan kondisi sulit"""
    
    # Enable/disable enhanced detection (DISABLED for streaming performance)
    ENABLE_ENHANCED_DETECTION = False    # Disabled for faster streaming
    USE_SUPER_RESOLUTION = False         # Disabled for faster streaming
    USE_OCR_ENSEMBLE = False             # Disabled for faster streaming
    USE_ADAPTIVE_ENHANCEMENT = False     # Disabled for faster streaming
    
    # Super-resolution settings
    SUPER_RESOLUTION_FACTOR = 3.0        # Default upscaling factor
    AUTO_SCALE_FACTOR = True             # Auto-determine scale factor
    MIN_PLATE_SIZE_FOR_SR = (20, 40)     # Min size (h, w) untuk trigger super-resolution
    MAX_SR_PROCESSING_TIME = 0.5         # Max time untuk super-resolution (seconds)
    
    # Image quality thresholds
    BLUR_THRESHOLD = 100.0               # Laplacian variance threshold for blur detection
    CONTRAST_THRESHOLD = 50.0            # Standard deviation threshold for contrast
    NOISE_THRESHOLD = 20.0               # Variance threshold for noise detection
    QUALITY_THRESHOLD = 30.0             # Overall quality threshold (0-100)
    
    # Enhancement parameters
    CLAHE_CLIP_LIMIT = 3.0              # CLAHE clip limit
    CLAHE_GRID_SIZE = (8, 8)            # CLAHE tile grid size
    GAUSSIAN_SIGMA = 2.0                # Gaussian blur sigma for unsharp mask
    SHARPENING_STRENGTH = 1.5           # Unsharp mask strength
    GAMMA_CORRECTION = 1.2              # Gamma correction factor
    
    # Multi-scale detection
    ENABLE_MULTI_SCALE = True           # Enable multi-scale detection
    SCALE_FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]  # Scale factors for multi-scale
    NMS_THRESHOLD = 0.4                 # Non-maximum suppression threshold
    
    # OCR ensemble settings
    ENSEMBLE_METHODS = ['standard', 'single_line', 'single_word', 'character_level']
    MIN_ENSEMBLE_AGREEMENT = 0.6        # Minimum agreement ratio for consensus
    CHARACTER_CORRECTION = True         # Enable character correction
    PATTERN_VALIDATION = True           # Enable Indonesian plate pattern validation
    
    # Performance optimization
    PARALLEL_OCR = True                 # Enable parallel OCR processing
    MAX_OCR_THREADS = 3                 # Max threads for OCR ensemble
    CACHE_ENHANCEMENTS = True           # Cache enhancement results
    CACHE_SIZE = 100                    # Max cached enhancement results
    
    # Fallback settings
    FALLBACK_TO_STANDARD = True         # Fallback to standard detection if enhanced fails
    MAX_ENHANCEMENT_TIME = 2.0          # Max time for enhancement pipeline
    ENHANCED_CONFIDENCE_BOOST = 10.0    # Confidence boost for enhanced detections

class MultiCameraConfig:
    """Pengaturan multi-camera system"""
    
    # Multi-camera settings
    ENABLE_MULTI_CAMERA = True           # Enable multi-camera capability
    MAX_CAMERAS = 4                      # Maximum number of concurrent cameras
    AUTO_DISCOVER_CAMERAS = True         # Auto-discover available cameras on startup
    
    # Camera priorities (lower number = higher priority)
    CAMERA_PRIORITIES = {
        'laptop': 1,                     # Laptop built-in camera
        'usb': 2,                        # USB cameras  
        'rtsp': 3,                       # RTSP IP cameras
        'file': 4                        # Video files
    }
    
    # Default camera configurations
    DEFAULT_LAPTOP_CONFIG = {
        'resolution': (640, 480),
        'fps_limit': 10,
        'auto_exposure': True,
        'buffer_size': 30
    }
    
    DEFAULT_USB_CONFIG = {
        'resolution': (640, 480),
        'fps_limit': 10,
        'buffer_size': 30
    }
    
    DEFAULT_RTSP_CONFIG = {
        'fps_limit': 10,
        'buffer_size': 30,
        'reconnect_attempts': 3,
        'timeout': 10
    }
    
    # Multi-camera processing
    PARALLEL_DETECTION = True           # Enable parallel detection across cameras
    CROSS_CAMERA_DEDUPLICATION = True  # Remove duplicate plates across cameras
    DEDUPLICATION_THRESHOLD = 3.0       # Seconds threshold for cross-camera duplicates
    
    # UI settings
    DEFAULT_GRID_LAYOUT = (2, 2)        # 2x2 grid for 4 cameras
    SHOW_CAMERA_NAMES = True            # Show camera names in UI
    ENABLE_CAMERA_SWITCHING = True      # Allow switching between cameras
    
    # Performance settings
    MAX_CONCURRENT_STREAMS = 4          # Max concurrent video streams
    FRAME_SYNC_TIMEOUT = 5.0            # Timeout untuk frame synchronization
    AUTO_QUALITY_ADJUSTMENT = True      # Auto-adjust quality based on performance

class LaptopCameraConfig:
    """Pengaturan khusus untuk laptop camera / built-in camera"""
    
    # Laptop camera detection dan prioritas
    ENABLE_LAPTOP_CAMERA = True         # Enable laptop camera support
    LAPTOP_CAMERA_INDEX = 0             # Default laptop camera index
    AUTO_DETECT_LAPTOP_CAMERA = True    # Auto-detect laptop camera on startup
    
    # Optimal settings untuk laptop camera
    PREFERRED_RESOLUTION = (640, 480)   # Optimal resolution untuk detection
    PREFERRED_FPS = 15                  # Optimal FPS untuk laptop camera
    QUALITY_RESOLUTION = (1280, 720)    # High quality resolution option
    PERFORMANCE_RESOLUTION = (320, 240) # Performance-focused resolution
    
    # Camera optimization settings
    AUTO_EXPOSURE = True                # Enable auto exposure by default
    AUTO_WHITE_BALANCE = True           # Enable auto white balance
    BRIGHTNESS_ADJUSTMENT = 0.0         # Brightness adjustment (-100 to 100)
    CONTRAST_ADJUSTMENT = 0.0           # Contrast adjustment (-100 to 100)
    
    # Detection optimization untuk laptop camera
    ENHANCE_LOW_LIGHT = True            # Enable low light enhancement
    STABILIZATION = True                # Enable image stabilization if available
    NOISE_REDUCTION = True              # Enable noise reduction
    
    # Fallback settings
    FALLBACK_TO_ANY_CAMERA = True       # Fallback to any available camera
    MAX_CAMERA_SCAN_INDEX = 5           # Max camera index to scan
    CAMERA_TEST_DURATION = 2.0          # Duration to test camera (seconds)
    
    # Platform-specific settings
    MACOS_AVFOUNDATION = True           # Use AVFoundation on macOS
    WINDOWS_DIRECTSHOW = True           # Use DirectShow on Windows
    LINUX_V4L2 = True                   # Use Video4Linux2 on Linux

# Helper functions
def ensure_folders_exist():
    """Pastikan semua folder yang dibutuhkan ada"""
    folders = [
        SystemConfig.OUTPUT_FOLDER,
        SystemConfig.LOG_FOLDER,
        "utils"
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

def get_tesseract_path():
    """Auto-detect tesseract path"""
    common_paths = [
        '/opt/homebrew/bin/tesseract',  # macOS Homebrew
        '/usr/bin/tesseract',           # Linux
        '/usr/local/bin/tesseract',     # Linux alternative
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Windows
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', ''))  # Windows user install
    ]
    
    for path in common_paths:
        if os.path.isfile(path):
            return path
    
    return TesseractConfig.TESSERACT_PATH  # Return default jika tidak ketemu

# Update tesseract path saat import
TesseractConfig.TESSERACT_PATH = get_tesseract_path()

def get_laptop_camera_settings(scenario: str = 'default'):
    """
    Get optimal laptop camera settings berdasarkan scenario
    
    Args:
        scenario: 'default', 'quality', 'performance', atau 'detection'
        
    Returns:
        tuple: (resolution, fps, additional_settings)
    """
    settings_map = {
        'default': (
            LaptopCameraConfig.PREFERRED_RESOLUTION,
            LaptopCameraConfig.PREFERRED_FPS,
            {
                'auto_exposure': LaptopCameraConfig.AUTO_EXPOSURE,
                'brightness': LaptopCameraConfig.BRIGHTNESS_ADJUSTMENT,
                'contrast': LaptopCameraConfig.CONTRAST_ADJUSTMENT
            }
        ),
        'quality': (
            LaptopCameraConfig.QUALITY_RESOLUTION,
            max(10, LaptopCameraConfig.PREFERRED_FPS - 5),  # Reduce FPS for quality
            {
                'auto_exposure': False,
                'brightness': 10.0,
                'contrast': 20.0,
                'exposure': -6.0
            }
        ),
        'performance': (
            LaptopCameraConfig.PERFORMANCE_RESOLUTION,
            min(30, LaptopCameraConfig.PREFERRED_FPS + 10),  # Increase FPS for performance
            {
                'auto_exposure': True,
                'brightness': LaptopCameraConfig.BRIGHTNESS_ADJUSTMENT,
                'contrast': 5.0
            }
        ),
        'detection': (
            LaptopCameraConfig.PREFERRED_RESOLUTION,
            LaptopCameraConfig.PREFERRED_FPS,
            {
                'auto_exposure': False,
                'brightness': 15.0,
                'contrast': 25.0,
                'exposure': -5.0
            }
        )
    }
    
    return settings_map.get(scenario, settings_map['default'])

def is_laptop_camera_enabled():
    """Check if laptop camera support is enabled"""
    return LaptopCameraConfig.ENABLE_LAPTOP_CAMERA

def validate_indonesian_plate(text: str) -> bool:
    """
    Validate if text matches Indonesian license plate pattern
    
    Args:
        text: OCR text to validate
        
    Returns:
        bool: True if valid Indonesian plate pattern
    """
    import re
    
    if not text or len(text) < IndonesianPlateConfig.MIN_PLATE_LENGTH:
        return False
    
    if len(text) > IndonesianPlateConfig.MAX_PLATE_LENGTH:
        return False
    
    # Clean text
    cleaned_text = text.strip().upper()
    
    # Check against Indonesian plate patterns
    for pattern in IndonesianPlateConfig.PLATE_PATTERNS:
        if re.match(pattern, cleaned_text):
            # Additional check for valid regional code
            if IndonesianPlateConfig.MIN_REGIONAL_CODE_MATCH:
                # Extract first 1-2 characters as potential regional code
                potential_code = cleaned_text[:2] if len(cleaned_text) >= 2 else cleaned_text[:1]
                if potential_code in IndonesianPlateConfig.REGIONAL_CODES:
                    return True
                # Check single character codes
                if cleaned_text[:1] in IndonesianPlateConfig.REGIONAL_CODES:
                    return True
            else:
                return True
    
    return False

def calculate_plate_confidence_boost(text: str, base_confidence: float) -> float:
    """
    Calculate confidence boost based on Indonesian plate pattern matching
    
    Args:
        text: OCR text
        base_confidence: Original confidence score
        
    Returns:
        float: Boosted confidence score
    """
    if not text:
        return base_confidence
    
    boosted_confidence = base_confidence
    
    # Boost for pattern match
    if validate_indonesian_plate(text):
        boosted_confidence += IndonesianPlateConfig.PATTERN_MATCH_BOOST
    
    # Boost for regional code
    cleaned_text = text.strip().upper()
    potential_code = cleaned_text[:2] if len(cleaned_text) >= 2 else cleaned_text[:1]
    if potential_code in IndonesianPlateConfig.REGIONAL_CODES or cleaned_text[:1] in IndonesianPlateConfig.REGIONAL_CODES:
        boosted_confidence += IndonesianPlateConfig.REGIONAL_CODE_BOOST
    
    # Cap at 100%
    return min(boosted_confidence, 100.0)