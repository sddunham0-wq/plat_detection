#!/usr/bin/env python3
"""
Download YOLO Model untuk License Plate Detection
Auto-download base YOLOv8 model jika custom model tidak tersedia
"""

import os
import sys
from pathlib import Path as PathLib

def download_yolo_model():
    """Download YOLO model untuk plate detection"""

    print("\n" + "="*60)
    print("🤖 YOLO MODEL SETUP")
    print("="*60 + "\n")

    models_dir = PathLib('models')
    models_dir.mkdir(exist_ok=True)

    custom_model = models_dir / 'best.pt'
    base_model = models_dir / 'yolov8n.pt'

    # Check if custom model exists
    if custom_model.exists():
        print(f"✅ Custom model found: {custom_model}")
        print(f"   Size: {custom_model.stat().st_size / 1024 / 1024:.2f} MB")
        return str(custom_model)

    print("⚠️  Custom model (best.pt) not found")
    print("ℹ️  Custom model untuk Indonesian plates lebih akurat")
    print()
    print("📥 Downloading base YOLOv8 model...")
    print("   Model: yolov8n.pt (~6MB)")
    print("   Note: Base model tidak spesifik plat, akurasi lebih rendah")
    print()

    try:
        from ultralytics import YOLO

        # Download base model (otomatis via ultralytics)
        print("🔄 Initializing YOLO...")
        model = YOLO('yolov8n.pt')  # Auto-download ke cache ultralytics

        print("✅ Base YOLOv8 model downloaded successfully!")
        print()
        print("⚠️  IMPORTANT: Untuk akurasi optimal, download custom model:")
        print()
        print("   1. Visit: https://universe.roboflow.com/")
        print("   2. Search: 'license plate detection indonesia'")
        print("   3. Download: best.pt model")
        print("   4. Save to: models/best.pt")
        print()
        print("   Atau gunakan model yang sudah dilatih dengan dataset:")
        print("   - Indonesian license plates")
        print("   - Various lighting conditions")
        print("   - Different distances and angles")
        print()

        return 'yolov8n.pt'  # Return base model name

    except ImportError:
        print("❌ ERROR: ultralytics not installed!")
        print()
        print("Install dependencies:")
        print("   pip3 install ultralytics torch torchvision")
        print()
        return None
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Manual download:")
        print("   1. Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        print("   2. Save to: models/yolov8n.pt")
        print()
        return None

def check_model_availability():
    """Check which models are available"""

    models_dir = PathLib('models')

    print("\n" + "="*60)
    print("📊 MODEL STATUS")
    print("="*60 + "\n")

    custom_model = models_dir / 'best.pt'
    base_model = models_dir / 'yolov8n.pt'

    models_found = []

    if custom_model.exists():
        size_mb = custom_model.stat().st_size / 1024 / 1024
        print(f"✅ Custom model (best.pt): {size_mb:.2f} MB")
        print(f"   Status: READY ✓")
        print(f"   Accuracy: HIGH (trained on plates)")
        models_found.append(str(custom_model))
    else:
        print("❌ Custom model (best.pt): NOT FOUND")
        print("   Recommendation: Download for better accuracy")

    print()

    # Check ultralytics cache
    try:
        from ultralytics import YOLO

        # Ultralytics cache location
        cache_dir = PathLib.home() / '.cache' / 'ultralytics'
        cached_model = cache_dir / 'yolov8n.pt'

        if cached_model.exists():
            size_mb = cached_model.stat().st_size / 1024 / 1024
            print(f"✅ Base model (yolov8n.pt): {size_mb:.2f} MB (cached)")
            print(f"   Status: READY ✓")
            print(f"   Accuracy: MEDIUM (general detection)")
            models_found.append('yolov8n.pt')
        else:
            print("⚠️  Base model (yolov8n.pt): Available for download")
            print("   Run this script to auto-download")

    except ImportError:
        print("❌ ultralytics not installed")
        print("   Install: pip3 install ultralytics")

    print()

    if models_found:
        print(f"✅ {len(models_found)} model(s) ready")
        print(f"   Recommended: {'Custom (best.pt)' if 'best.pt' in str(models_found[0]) else 'Base (yolov8n.pt)'}")
    else:
        print("❌ No models available")
        print("   Action required: Download model")

    print("="*60 + "\n")

    return models_found

if __name__ == '__main__':
    print()
    print("🚀 YOLO Model Setup for License Plate Detection")
    print()

    # Check current status
    available_models = check_model_availability()

    if not available_models:
        # No models found - download base
        print("🔄 No models found. Starting auto-download...")
        print()

        result = download_yolo_model()

        if result:
            print()
            print("="*60)
            print("✅ SETUP COMPLETE")
            print("="*60)
            print()
            print("Next steps:")
            print("   1. Test detection: python3 test_yolo_detection.py")
            print("   2. Run app: python3 app_simple.py")
            print()
            print("For better accuracy:")
            print("   - Download custom model (best.pt)")
            print("   - Place in models/ folder")
            print()
        else:
            print()
            print("="*60)
            print("❌ SETUP FAILED")
            print("="*60)
            print()
            print("Manual steps required:")
            print("   1. Install: pip3 install ultralytics torch")
            print("   2. Re-run: python3 download_yolo_model.py")
            print()
            sys.exit(1)
    else:
        print("✅ Models already available!")
        print()
        print("Ready to use:")
        for model in available_models:
            print(f"   - {model}")
        print()
        print("Test with:")
        print("   python3 test_yolo_detection.py image2.png")
        print()
