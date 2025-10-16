"""
YOLO Configuration untuk Plate Detection
"""

YOLO_CONFIG = {
    # Model settings
    'model_path': 'models/best.pt',    # Path ke YOLO model
    'conf_threshold': 0.25,            # Confidence threshold (0.0-1.0)
    'iou_threshold': 0.45,             # IoU untuk NMS
    'max_detections': 3,               # Max plates per frame

    # Image size untuk YOLO
    'imgsz': 640,                      # 640x640 default

    # Device
    'device': 'cpu',                   # 'cpu' atau '0' untuk GPU

    # Performance
    'half': False,                     # FP16 inference (GPU only)
    'augment': False,                  # Augmented inference
    'verbose': False,                  # Print YOLO logs
}

# Fallback ke contour detection jika YOLO fails
ENABLE_FALLBACK = True

# Log settings
LOG_DETECTIONS = True
