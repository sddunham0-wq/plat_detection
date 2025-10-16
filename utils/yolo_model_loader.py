# -*- coding: utf-8 -*-
"""
YOLO Model Loader dengan Auto-Detection & Fallback
Mencari model yang tersedia dan gunakan yang terbaik
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def find_best_yolo_model():
    """
    Cari YOLO model yang tersedia dengan priority:
    1. Custom model (best.pt) - trained untuk plates
    2. Base YOLOv8n - general detection
    3. None - fallback ke contour detection

    Returns:
        tuple: (model_path, model_type) atau (None, None)
    """

    # Priority 1: Custom model (best accuracy)
    custom_model = Path('models/best.pt')
    if custom_model.exists():
        size_mb = custom_model.stat().st_size / 1024 / 1024
        logger.info(f"✅ Found custom model: {custom_model} ({size_mb:.1f}MB)")
        return str(custom_model), 'custom'

    # Priority 2: Base YOLOv8n (medium accuracy)
    # Check local models folder
    base_model_local = Path('models/yolov8n.pt')
    if base_model_local.exists():
        size_mb = base_model_local.stat().st_size / 1024 / 1024
        logger.info(f"✅ Found base model (local): {base_model_local} ({size_mb:.1f}MB)")
        return str(base_model_local), 'base'

    # Check ultralytics cache
    try:
        cache_dir = Path.home() / '.cache' / 'ultralytics'
        cached_model = cache_dir / 'yolov8n.pt'
        if cached_model.exists():
            size_mb = cached_model.stat().st_size / 1024 / 1024
            logger.info(f"✅ Found base model (cache): {cached_model} ({size_mb:.1f}MB)")
            return str(cached_model), 'base'
    except:
        pass

    # Priority 3: Try to use yolov8n (will auto-download)
    try:
        # Check if ultralytics installed
        from ultralytics import YOLO
        logger.info("⚠️  No local model found, will use yolov8n.pt (auto-download)")
        return 'yolov8n.pt', 'base_autodownload'
    except ImportError:
        logger.warning("❌ ultralytics not installed")

    # No models available
    logger.warning("⚠️  No YOLO models found")
    logger.info("ℹ️  Run: python3 download_yolo_model.py")
    return None, None

def get_model_info(model_path, model_type):
    """
    Get model information

    Returns:
        dict: Model metadata
    """
    info = {
        'path': model_path,
        'type': model_type,
        'name': Path(model_path).name if model_path else None,
        'accuracy': 'high' if model_type == 'custom' else 'medium',
        'description': ''
    }

    if model_type == 'custom':
        info['description'] = 'Custom model trained for Indonesian license plates'
    elif model_type in ['base', 'base_autodownload']:
        info['description'] = 'Base YOLOv8n model (general object detection)'
    else:
        info['description'] = 'Unknown model type'

    return info
