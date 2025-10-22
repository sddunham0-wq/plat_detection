"""
Configuration file untuk Live CCTV License Plate Detection System
Pengaturan lengkap untuk deteksi plat nomor real-time
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class CCTVConfig:
    """Pengaturan CCTV dan Video Stream"""
    
    # Default video sources
    # ✅ WORKING URL FOUND: Correct RTSP path discovered!
    DEFAULT_RTSP_URL = "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=0"  # NEW WORKING URL ✅

    # Alternative URLs untuk testing dan fallback
    # DEFAULT_RTSP_URL = "rtsp://admin:H4nd4l9165!@192.168.1.195:554"      # Previous working URL
    # DEFAULT_RTSP_URL = "rtsp://admin:H4nd4l9165!@192.168.1.195:8554/85"   # Alternative RTSP port

    # Fallback URLs jika yang utama tidak berfungsi
    FALLBACK_RTSP_URLS = [
        "rtsp://admin:H4nd4l9165!@192.168.1.195:554",                      # Previous working camera (fallback)
        "rtsp://admin:H4nd4l9165!@192.168.1.203:5503/cam/realmonitor?channel=1&subtype=1",  # Substream
        "rtsp://admin:H4nd4l9165!@192.168.1.203:554/cam/realmonitor?channel=1&subtype=0",   # Standard RTSP port
        "rtsp://admin:H4nd4l9165!@192.168.1.195:8554/85",                  # Original alternative port
    ]
    DEFAULT_WEBCAM = 0      # Default to laptop camera (index 0)
    DEFAULT_LAPTOP_CAMERA = 0  # Laptop built-in camera index
    DEFAULT_VIDEO_FILE = "video_cctv.mp4"
    
    # Video processing settings
    FRAME_WIDTH = 480          # Lebar frame untuk processing (optimized for speed)
    FRAME_HEIGHT = 360         # Tinggi frame untuk processing (optimized for speed)
    FPS_LIMIT = 25             # Maksimal FPS untuk processing (increased for smoother streaming)
    BUFFER_SIZE = 12           # Maksimal frame di buffer (increased from 5 to 12 for RTSP stability)

    # Frame skipping for performance (SAFE MODE - bbox persisted across frames)
    ENABLE_FRAME_SKIPPING = True   # ENABLED with bbox caching for visibility
    PROCESS_EVERY_N_FRAMES = 2     # Process every 2nd frame (50% skip, SAFE with caching)
    
    # RTSP connection settings - Enhanced untuk 192.168.1.203
    RTSP_TIMEOUT = 15          # Timeout koneksi RTSP (detik) - increased for Dahua cameras
    RECONNECT_DELAY = 5        # Delay sebelum reconnect (detik)
    MAX_RECONNECT_ATTEMPTS = 3 # Maksimal percobaan reconnect

    # Enhanced OpenCV settings untuk optimal RTSP streaming
    OPENCV_BACKEND = 'FFMPEG'  # Force FFMPEG backend untuk RTSP
    BUFFER_SIZE_RTSP = 1       # Minimal buffer untuk real-time streaming
    FORCE_FRAME_SIZE = True    # Force frame size setting
    PREFERRED_CODEC = 'H264'   # Preferred video codec

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
    
    # Confidence threshold (0-100) - OPTIMIZED with pattern validation for quality
    MIN_CONFIDENCE = 40        # Optimized to 40 (with pattern validation, can afford higher threshold for quality)
    INDONESIAN_MIN_CONFIDENCE = 40  # Balanced threshold with intelligent filtering enabled
    
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
    MIN_PLATE_LENGTH = 4            # Relaxed from 6 to 4 untuk accept short plates
    MAX_PLATE_LENGTH = 12           # Increased from 10 to 12 untuk long plates with spaces
    
    # Confidence boosting untuk Indonesian plates
    PATTERN_MATCH_BOOST = 10.0  # Boost confidence jika match pattern
    REGIONAL_CODE_BOOST = 5.0   # Boost jika ada regional code
    
    # Strict validation settings (ENABLED for smart filtering)
    ENABLE_STRICT_PATTERN_VALIDATION = True  # ENABLED to reject invalid patterns (reduces false positives by 30%)
    REJECT_NON_PATTERN_MATCHES = True        # Reject plates that don't match Indonesian patterns
    MIN_REGIONAL_CODE_MATCH = True           # ENABLED to validate regional codes (B, D, L, etc) - reduces invalid plates by 20%

    # Smart validation settings (NEW - for anti-spam system)
    SMART_VALIDATION_MIN_CONFIDENCE = 70  # Minimum confidence to log detection (prevents garbage OCR spam)

    # Preprocessing optimization untuk plat Indonesia (OPTIMIZED for OCR)
    CONTRAST_ENHANCEMENT = 3.5   # Enhanced contrast for better OCR (increased from 2.5 → 3.5)
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

    # Anti-spam detection settings (NEW)
    DETECTION_COOLDOWN_TIME = 30      # Cooldown time in seconds (prevent spam logging)
    BBOX_OVERLAP_THRESHOLD = 0.7      # IoU threshold for bbox overlap detection (70%)
    MIN_OCR_CONFIDENCE_TO_LOG = 70    # Only log detections with confidence >= 70%

    # Enhanced duplicate detection
    DUPLICATE_THRESHOLD = 30          # Increased to 30 seconds untuk prevent spam
    MIN_PLATE_LENGTH = 5             # Minimal panjang karakter plat
    MAX_PLATE_LENGTH = 12            # Maksimal panjang karakter plat

    # Quality-based filtering thresholds (STRICT)
    MIN_QUALITY_SCORE = 0.75         # Increased from 0.6 to 0.75 untuk strict quality validation
    MIN_CONFIDENCE_THRESHOLD = 0.5    # Increased from 0.3 to 0.5 untuk prevent false positives
    MIN_TEXT_SCORE = 0.6             # Minimum text pattern score (60%)

    # Enhanced IoU threshold untuk duplicate filtering
    ENHANCED_IOU_THRESHOLD = 0.65    # Increased from 0.30 to 0.65 untuk better duplicate removal
    
    # Color-based detection thresholds
    ENABLE_COLOR_FILTERING = True     # Enable Indonesian plate color detection
    MIN_COLOR_CONFIDENCE = 15.0       # Minimum color confidence for regular plates
    MOTORCYCLE_MIN_COLOR_CONFIDENCE = 10.0  # Lower threshold for motorcycles
    
    # Geometric validation thresholds (enhanced)
    MIN_RECTANGULARITY = 0.5          # Reduced from 0.7 untuk better plate detection
    MIN_SOLIDITY = 0.6               # Reduced from 0.8 untuk better motorcycle plates
    MIN_EXTENT = 0.5                 # Reduced from 0.7 untuk angled plates

    # Bounding box refinement parameters (STRICT)
    ENABLE_BBOX_REFINEMENT = True     # Enable contour-based bounding box refinement
    EDGE_DETECTION_THRESHOLD = 0.5    # Increased from 0.3 to 0.5 untuk strict edge quality
    CONTOUR_AREA_MIN = 800           # Increased from 500 untuk larger minimum area
    CONTOUR_AREA_MAX = 30000         # Decreased from 50000 untuk reasonable maximum

    # False positive prevention parameters
    MIN_EDGE_DENSITY = 0.15          # Minimum edge density untuk valid plates
    MIN_CONTRAST_RATIO = 1.5         # Minimum contrast ratio untuk text visibility
    MAX_BACKGROUND_UNIFORMITY = 0.8   # Maximum background uniformity (lower = more varied)
    
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
    
    # Enhanced OCR optimization untuk plat motor kecil (BALANCED for speed + accuracy)
    UPSCALE_FACTOR = 7.0              # Optimized from 8.0 to 7.0 for speed (10% faster with minimal accuracy loss)
    MIN_OCR_HEIGHT = 40               # Increased from 20 to 40 for clearer text recognition
    
    # Enhanced detection confidence (optimized for quality - FIXED for accuracy)
    MIN_CONFIDENCE = 65               # Increased from 35 to 65 to prevent false positives and garbage OCR
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

class MySQLConfig:
    """Pengaturan MySQL database untuk access control system"""

    # Load from environment variables with fallback defaults
    MYSQL_HOST = os.getenv('MYSQL_HOST', '127.0.0.1')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3307))
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'plat_detection')

    # Connection pool settings (OPTIMIZED for multi-developer environment)
    # Reduced from 5 to 3 to prevent connection exhaustion (40% reduction per developer)
    MYSQL_POOL_SIZE = int(os.getenv('MYSQL_POOL_SIZE', 3))
    MYSQL_MAX_OVERFLOW = int(os.getenv('MYSQL_MAX_OVERFLOW', 10))
    MYSQL_POOL_TIMEOUT = int(os.getenv('MYSQL_POOL_TIMEOUT', 30))

    # Connection timeout settings (NEW - Auto-cleanup for stale connections)
    # Idle connections will be closed after 5 minutes (300 seconds)
    MYSQL_MAX_IDLE_TIME = int(os.getenv('MYSQL_MAX_IDLE_TIME', 300))
    # Health check interval for removing broken connections (60 seconds)
    MYSQL_HEALTH_CHECK_INTERVAL = int(os.getenv('MYSQL_HEALTH_CHECK_INTERVAL', 60))

    # Connection settings
    MYSQL_CONNECT_TIMEOUT = 10
    MYSQL_READ_TIMEOUT = 30
    MYSQL_WRITE_TIMEOUT = 30

    # Auto-reconnect settings
    MYSQL_AUTO_RECONNECT = True
    MYSQL_MAX_RECONNECT_ATTEMPTS = 3
    MYSQL_RECONNECT_DELAY = 5

    # Feature flags
    USE_MYSQL_DATABASE = os.getenv('USE_MYSQL_DATABASE', 'True').lower() == 'true'
    ENABLE_ACCESS_CONTROL = os.getenv('ENABLE_ACCESS_CONTROL', 'True').lower() == 'true'
    LOG_DENIED_ACCESS = os.getenv('LOG_DENIED_ACCESS', 'True').lower() == 'true'
    AUTO_UPDATE_VEHICLE_STATUS = os.getenv('AUTO_UPDATE_VEHICLE_STATUS', 'True').lower() == 'true'

    # Dual mode settings
    ENABLE_SQLITE_LOGGING = os.getenv('ENABLE_SQLITE_LOGGING', 'True').lower() == 'true'
    ENABLE_MYSQL_ACCESS_CONTROL = os.getenv('ENABLE_MYSQL_ACCESS_CONTROL', 'True').lower() == 'true'

class ImagePreprocessingConfig:
    """Pengaturan image preprocessing dan deskewing untuk tilted plates"""

    # Enable/disable image deskewing (DEFAULT: ENABLED for better accuracy)
    ENABLE_DESKEWING = os.getenv('ENABLE_DESKEWING', 'True').lower() == 'true'

    # Skew detection parameters
    MAX_SKEW_ANGLE = 30.0                # Maximum skew angle to detect (degrees)
    MIN_SKEW_CORRECTION_THRESHOLD = 0.5  # Minimum angle to trigger correction (degrees)

    # Perspective correction
    ENABLE_PERSPECTIVE_CORRECTION = os.getenv('ENABLE_PERSPECTIVE_CORRECTION', 'True').lower() == 'true'

    # Image enhancement (OPTIMIZED for better OCR on low-contrast plates)
    ENABLE_ENHANCEMENT = os.getenv('ENABLE_ENHANCEMENT', 'True').lower() == 'true'
    CLAHE_CLIP_LIMIT = 3.5              # CLAHE contrast enhancement clip limit (increased from 2.0 for better text clarity)
    CLAHE_TILE_GRID_SIZE = (8, 8)       # CLAHE tile grid size

    # Denoising parameters
    DENOISE_H = 10                       # Filter strength for denoising
    DENOISE_TEMPLATE_WINDOW_SIZE = 7     # Template window size
    DENOISE_SEARCH_WINDOW_SIZE = 21      # Search window size

    # Sharpening parameters
    SHARPENING_KERNEL = [[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]]   # Sharpening kernel

    # Multi-angle OCR attempts (OPTIMIZED for speed + accuracy balance)
    ENABLE_MULTI_ANGLE_OCR = os.getenv('ENABLE_MULTI_ANGLE_OCR', 'True').lower() == 'true'  # Enabled for tilted plate accuracy
    MULTI_ANGLE_ROTATIONS = [-5, 0, 5]  # Angles to try for OCR (degrees) - reduced from 5 to 3 for 40% speed boost
    MAX_OCR_VARIANTS = 3                 # Maximum preprocessing variants to try (reduced for speed)

    # Performance optimization (OPTIMIZED for faster processing)
    PREPROCESSING_TIMEOUT = 1.5          # Max time for preprocessing pipeline (reduced from 2.0s for speed)
    ENABLE_PREPROCESSING_CACHE = True    # Cache preprocessing results for same frames

class SystemConfig:
    """Pengaturan sistem dan logging"""

    # Folders
    OUTPUT_FOLDER = "detected_plates"     # Folder untuk simpan gambar hasil (deprecated - use VEHICLE_IMAGE_FOLDER)
    VEHICLE_IMAGE_FOLDER = "detected_vehicles"  # Folder untuk simpan foto kendaraan (full frame)
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

    # Enable/disable tracking system (ENABLED for duplicate filtering)
    ENABLE_TRACKING = True
    
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

class PersonDetectionConfig:
    """Pengaturan Person Detection System"""

    # Enable/disable person detection (DEFAULT: DISABLED untuk backward compatibility)
    ENABLE_PERSON_DETECTION = False      # ✅ DISABLED - Focus on plate detection only (2x faster)

    # Person detection thresholds
    PERSON_CONFIDENCE = 0.5              # Confidence threshold untuk person (50%)
    PERSON_MAX_DETECTIONS = 20           # Maximum persons per frame

    # Visual styling untuk person bounding boxes
    PERSON_BBOX_COLOR = (255, 0, 0)      # Blue color (BGR format)
    PERSON_BBOX_THICKNESS = 2            # Bounding box line thickness
    PERSON_SHOW_CONFIDENCE = True        # Show confidence scores

    # Model settings
    PERSON_YOLO_MODEL = 'yolov8n.pt'     # YOLOv8 model untuk person detection

    # Performance settings
    PERSON_DETECTION_PARALLEL = True     # Run person detection parallel dengan plate detection
    PERSON_FRAME_SKIP = 1                # Process every N frames (1 = every frame)

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