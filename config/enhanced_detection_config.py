"""
Enhanced Detection Configuration - Dual YOLO + OCR System
Konfigurasi untuk Vehicle YOLO + License Plate YOLO + OCR Fallback
"""

# Default configuration untuk Enhanced Hybrid Detector
DEFAULT_ENHANCED_CONFIG = {
    # Vehicle YOLO Configuration (existing)
    'vehicle_yolo': {
        'enabled': False,                        # Vehicle YOLO disabled by default
        'model_path': 'yolov8n.pt',             # YOLOv8 Nano model
        'confidence': 0.4,                      # Vehicle detection confidence
        'iou_threshold': 0.5,                   # IoU threshold for vehicle NMS
        'max_detections': 15,                   # Max vehicles per frame (optimized for CCTV)
        'device': 'auto',                       # 'auto', 'cpu', 'cuda'
        'vehicles_only': True,                  # Only detect vehicles (car, motorcycle, bus, truck)
        'crowded_scene_detection': True,        # Enable crowded motorcycle detection
        'enhanced_bbox': True                   # Enhanced bounding box for vehicles
    },

    # License Plate YOLO Configuration (NEW)
    'license_plate_yolo': {
        'enabled': False,                       # License Plate YOLO disabled by default
        'model_path': 'license_plate_yolo.pt', # Custom trained Indonesian plate model
        'confidence': 0.3,                     # Lower threshold for plate detection
        'iou_threshold': 0.4,                  # IoU threshold for plate NMS
        'max_detections': 25,                  # Max plates per frame
        'device': 'auto',                      # Device for inference

        # Indonesian Plate Specific Settings
        'indonesian_optimized': True,           # Enable Indonesian plate optimizations
        'minimum_plate_size': (30, 15),        # Minimum plate dimensions (w, h)
        'maximum_plate_size': (300, 150),      # Maximum plate dimensions (w, h)
        'aspect_ratio_range': (2.0, 6.0),     # Indonesian plate aspect ratio
        'min_plate_area': 800,                 # Minimum plate area in pixels
        'max_plate_area': 15000,               # Maximum plate area in pixels

        # Text Processing Settings
        'enable_text_extraction': True,        # Extract text from YOLO detections
        'text_confidence_boost': 0.1,          # Confidence boost for valid text
        'pattern_validation': True,            # Validate Indonesian plate patterns
        'ocr_fallback_integration': True       # Integrate with OCR for text extraction
    },

    # Enhanced Hybrid Detection System
    'enhanced_hybrid': {
        'enabled': False,                       # Enhanced hybrid system disabled by default
        'detection_priority': 'dual_yolo',     # 'vehicle_only', 'plate_only', 'dual_yolo', 'ocr_only'
        'use_vehicle_regions': True,            # Focus plate detection on vehicle regions
        'ocr_fallback': True,                   # Enable OCR fallback for missed plates
        'hybrid_validation': True,              # Cross-validate YOLO + OCR results

        # Confidence Enhancement
        'confidence_boost': {
            'plate_yolo': 0.1,                  # Boost for YOLO plate detections
            'ocr': -0.1,                        # Penalty for OCR-only detections
            'hybrid_match': 0.2,                # Bonus for YOLO + OCR agreement
            'indonesian_format': 0.15,          # Bonus for valid Indonesian format
            'vehicle_context': 0.05             # Bonus when plate found in vehicle region
        },

        # Duplicate Removal
        'duplicate_removal': {
            'enabled': True,                    # Enable intelligent duplicate removal
            'iou_threshold': 0.5,               # IoU threshold for spatial duplicates
            'text_similarity_threshold': 0.8,   # Text similarity threshold
            'temporal_window': 2.0,             # Temporal deduplication window (seconds)
            'cross_method_dedup': True          # Remove duplicates across detection methods
        },

        # Quality Filtering
        'quality_filtering': {
            'min_confidence': 0.3,              # Minimum overall confidence
            'min_text_length': 5,               # Minimum plate text length
            'max_text_length': 12,              # Maximum plate text length
            'require_indonesian_format': False, # Require Indonesian plate format
            'size_validation': True,            # Validate plate dimensions
            'aspect_ratio_validation': True     # Validate aspect ratio
        }
    },

    # OCR Configuration (enhanced for hybrid system)
    'ocr': {
        'enabled': True,                        # OCR always enabled for fallback
        'tesseract_path': '/opt/homebrew/bin/tesseract',  # Auto-detected
        'config': '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        'language': 'ind+eng',                 # Indonesian + English
        'min_confidence': 40,                  # Minimum OCR confidence

        # Enhanced OCR for hybrid system
        'preprocessing': {
            'upscale_factor': 3.0,              # Upscale small plates
            'contrast_enhancement': True,        # Enhance contrast
            'noise_reduction': True,            # Reduce noise
            'adaptive_threshold': True,         # Adaptive thresholding
            'morphology_cleanup': True          # Morphological operations
        },

        # Multi-method OCR
        'ensemble_ocr': {
            'enabled': False,                   # Disabled by default for speed
            'methods': ['standard', 'single_line', 'single_word'],
            'consensus_threshold': 0.6,         # Agreement threshold
            'max_processing_time': 1.0          # Max time per method
        }
    },

    # Indonesian Plate Validation
    'indonesian_plates': {
        'patterns': [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',    # B 1234 ABC
            r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$',          # B1234ABC
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{2,3}$',    # AA 1234 BB
        ],
        'regional_codes': [
            'B', 'D', 'E', 'F', 'G', 'H',      # Jakarta, Bandung, Cirebon, etc
            'L', 'M', 'N', 'P', 'R', 'S',      # Surabaya, Madura, etc
            'T', 'U', 'AA', 'AB', 'AD',        # Other regions
            'AG', 'BA', 'BB', 'BD', 'BE'       # Sumatra, etc
        ],
        'char_corrections': {
            'O': '0', 'I': '1', 'S': '5', 'Z': '2',
            '8': 'B', '6': 'G', '0': 'O'
        },
        'format_validation': True,              # Enable format validation
        'regional_code_validation': False       # Disabled for flexibility
    },

    # Performance Optimization
    'performance': {
        'max_processing_time': 5.0,            # Max processing time per frame
        'parallel_detection': True,            # Enable parallel detection
        'frame_skipping': True,                # Enable intelligent frame skipping
        'skip_factor': 2,                      # Process every N-th frame
        'memory_limit_mb': 512,                # Memory limit
        'gpu_optimization': True,              # Enable GPU optimizations if available
        'batch_processing': False,             # Batch processing (for multiple plates)
        'caching': {
            'enabled': True,                   # Enable result caching
            'max_cache_size': 100,             # Max cached results
            'cache_ttl': 60.0                  # Cache time-to-live (seconds)
        }
    },

    # Logging and Debugging
    'logging': {
        'level': 'INFO',                       # DEBUG, INFO, WARNING, ERROR
        'detection_stats': True,               # Log detection statistics
        'performance_metrics': True,           # Log performance metrics
        'method_breakdown': True,              # Log detection method breakdown
        'save_debug_images': False,            # Save debug images (for development)
        'debug_image_path': 'debug_images/'    # Path for debug images
    },

    # Integration Settings
    'integration': {
        'ultra_stable_counter': True,          # Integrate with ultra-stable counter
        'stream_manager': True,                # Integrate with stream manager
        'database_logging': True,              # Enable database logging
        'real_time_stats': True,              # Enable real-time statistics
        'api_compatible': True                # Maintain API compatibility
    }
}

# Preset configurations untuk different scenarios
PRESET_CONFIGS = {
    'laptop_camera': {
        'vehicle_yolo': {'enabled': False},     # Disable vehicle YOLO for laptop
        'license_plate_yolo': {'enabled': True, 'confidence': 0.4},
        'enhanced_hybrid': {'enabled': True, 'detection_priority': 'plate_only'},
        'performance': {'frame_skipping': True, 'skip_factor': 3}
    },

    'cctv_monitoring': {
        'vehicle_yolo': {'enabled': True, 'confidence': 0.3},
        'license_plate_yolo': {'enabled': True, 'confidence': 0.2},
        'enhanced_hybrid': {'enabled': True, 'detection_priority': 'dual_yolo'},
        'performance': {'frame_skipping': True, 'skip_factor': 2}
    },

    'high_accuracy': {
        'vehicle_yolo': {'enabled': True, 'confidence': 0.5},
        'license_plate_yolo': {'enabled': True, 'confidence': 0.4},
        'enhanced_hybrid': {
            'enabled': True,
            'detection_priority': 'dual_yolo',
            'hybrid_validation': True,
            'quality_filtering': {'require_indonesian_format': True}
        },
        'ocr': {'ensemble_ocr': {'enabled': True}},
        'performance': {'frame_skipping': False}
    },

    'performance_optimized': {
        'vehicle_yolo': {'enabled': False},     # Disable for speed
        'license_plate_yolo': {'enabled': True, 'confidence': 0.2},
        'enhanced_hybrid': {'enabled': True, 'detection_priority': 'plate_only'},
        'performance': {
            'frame_skipping': True,
            'skip_factor': 4,
            'parallel_detection': True
        },
        'ocr': {'ensemble_ocr': {'enabled': False}}
    },

    'development_debug': {
        'vehicle_yolo': {'enabled': True},
        'license_plate_yolo': {'enabled': True},
        'enhanced_hybrid': {'enabled': True},
        'logging': {
            'level': 'DEBUG',
            'save_debug_images': True,
            'detection_stats': True,
            'performance_metrics': True
        },
        'performance': {'frame_skipping': False}
    }
}

def get_config(preset: str = 'default') -> dict:
    """
    Get configuration for enhanced detection system

    Args:
        preset: Configuration preset - 'default', 'laptop_camera', 'cctv_monitoring',
                'high_accuracy', 'performance_optimized', 'development_debug'

    Returns:
        dict: Configuration dictionary
    """
    if preset == 'default':
        return DEFAULT_ENHANCED_CONFIG.copy()

    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")

    # Start with default config
    config = DEFAULT_ENHANCED_CONFIG.copy()
    preset_config = PRESET_CONFIGS[preset]

    # Deep merge preset configuration
    def deep_merge(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    deep_merge(config, preset_config)
    return config

def validate_config(config: dict) -> bool:
    """
    Validate enhanced detection configuration

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if valid
    """
    required_sections = ['vehicle_yolo', 'license_plate_yolo', 'enhanced_hybrid', 'ocr']

    for section in required_sections:
        if section not in config:
            print(f"Missing required section: {section}")
            return False

    # Validate logical combinations
    hybrid_config = config.get('enhanced_hybrid', {})
    if hybrid_config.get('enabled'):
        priority = hybrid_config.get('detection_priority', 'dual_yolo')

        if priority == 'dual_yolo':
            if not (config['vehicle_yolo'].get('enabled') and config['license_plate_yolo'].get('enabled')):
                print("dual_yolo priority requires both vehicle_yolo and license_plate_yolo to be enabled")
                return False
        elif priority == 'vehicle_only':
            if not config['vehicle_yolo'].get('enabled'):
                print("vehicle_only priority requires vehicle_yolo to be enabled")
                return False
        elif priority == 'plate_only':
            if not config['license_plate_yolo'].get('enabled'):
                print("plate_only priority requires license_plate_yolo to be enabled")
                return False

    return True

def create_stream_manager_config(enhanced_config: dict) -> dict:
    """
    Convert enhanced config to stream manager compatible format

    Args:
        enhanced_config: Enhanced detection configuration

    Returns:
        dict: Stream manager compatible configuration
    """
    return {
        'yolo': {
            'enabled': enhanced_config['vehicle_yolo']['enabled'],
            'model_path': enhanced_config['vehicle_yolo']['model_path'],
            'confidence': enhanced_config['vehicle_yolo']['confidence'],
            'iou_threshold': enhanced_config['vehicle_yolo']['iou_threshold'],
            'max_detections': enhanced_config['vehicle_yolo']['max_detections'],
            'device': enhanced_config['vehicle_yolo']['device']
        },
        'license_plate_yolo': enhanced_config['license_plate_yolo'],
        'enhanced_hybrid': enhanced_config['enhanced_hybrid'],
        'ocr': enhanced_config['ocr'],
        'indonesian_plates': enhanced_config['indonesian_plates'],
        'performance': enhanced_config['performance']
    }

# Example usage dan testing
if __name__ == "__main__":
    # Test different presets
    presets = ['default', 'laptop_camera', 'cctv_monitoring', 'high_accuracy', 'performance_optimized']

    for preset in presets:
        print(f"\n=== {preset.upper()} PRESET ===")
        config = get_config(preset)

        print(f"Vehicle YOLO: {'✅' if config['vehicle_yolo']['enabled'] else '❌'}")
        print(f"License Plate YOLO: {'✅' if config['license_plate_yolo']['enabled'] else '❌'}")
        print(f"Enhanced Hybrid: {'✅' if config['enhanced_hybrid']['enabled'] else '❌'}")
        print(f"Detection Priority: {config['enhanced_hybrid'].get('detection_priority', 'N/A')}")

        if validate_config(config):
            print("✅ Configuration valid")
        else:
            print("❌ Configuration invalid")

    # Test stream manager config conversion
    print(f"\n=== STREAM MANAGER INTEGRATION ===")
    enhanced_config = get_config('cctv_monitoring')
    stream_config = create_stream_manager_config(enhanced_config)
    print("Stream manager config created successfully")