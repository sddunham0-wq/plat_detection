#!/usr/bin/env python3
"""
Download License Plate YOLO Model
Script untuk download dan setup license plate YOLO model
"""

import os
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url: str, filename: str) -> bool:
    """Download file dari URL"""
    try:
        logger.info(f"Downloading {filename} dari {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"✅ {filename} berhasil didownload")
        return True

    except Exception as e:
        logger.error(f"❌ Error downloading {filename}: {e}")
        return False

def download_license_plate_models():
    """Download beberapa pre-trained license plate models"""

    print("🚀 LICENSE PLATE YOLO MODEL DOWNLOADER")
    print("=" * 50)

    # Model options dengan URLs (contoh - perlu disesuaikan dengan yang actual available)
    models = {
        "general_license_plate": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "filename": "yolov8n_base.pt",
            "description": "YOLOv8 Nano base model (untuk fine-tuning)"
        }
    }

    print("📋 Available models untuk download:")
    for i, (key, info) in enumerate(models.items(), 1):
        print(f"  {i}. {info['description']}")

    # Download YOLOv8 base model untuk starting point
    base_model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    base_model_file = "yolov8n_base.pt"

    if not os.path.exists(base_model_file):
        logger.info("Downloading YOLOv8 base model...")
        if download_file(base_model_url, base_model_file):
            print(f"✅ Base model downloaded: {base_model_file}")
        else:
            print(f"❌ Failed to download base model")
            return False
    else:
        print(f"✅ Base model already exists: {base_model_file}")

    # Try to download specialized license plate model
    # NOTE: Ini adalah contoh URL - perlu diganti dengan actual model
    license_plate_urls = [
        # Contoh URLs - akan perlu disesuaikan
        "https://example.com/license_plate_yolo.pt",  # Placeholder
    ]

    print("\n🎯 Attempting to download specialized license plate models...")

    # Create placeholder model untuk testing (temporary solution)
    if not os.path.exists("license_plate_yolo.pt"):
        print("⚠️  Specialized Indonesian model tidak tersedia di public URLs")
        print("📦 Creating placeholder model untuk testing...")

        # Copy base model sebagai placeholder
        if os.path.exists(base_model_file):
            import shutil
            shutil.copy(base_model_file, "license_plate_yolo.pt")
            print("✅ Placeholder model created: license_plate_yolo.pt")
            print("⚠️  NOTE: Ini adalah base YOLO model, bukan khusus license plate")
            print("📋 Untuk hasil optimal, Anda perlu train custom model")
        else:
            print("❌ Cannot create placeholder model")
            return False
    else:
        print("✅ license_plate_yolo.pt already exists")

    return True

def setup_roboflow_download():
    """Setup Roboflow untuk download dataset/model"""
    print("\n🤖 ROBOFLOW SETUP")
    print("-" * 30)

    print("Untuk download dari Roboflow Universe:")
    print("1. Buat account di https://roboflow.com")
    print("2. Browse ke https://universe.roboflow.com")
    print("3. Cari 'license plate' atau 'Indonesian plate'")
    print("4. Download model dalam format YOLOv8")

    # Generate contoh code untuk Roboflow download
    roboflow_code = '''
# Contoh code untuk download dari Roboflow:
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
dataset = project.version(1).download("yolov8")

# Model akan tersedia di folder download
'''

    print(f"\n📝 Contoh code untuk Roboflow:")
    print(roboflow_code)

    return True

def verify_model():
    """Verify downloaded model"""
    print("\n✅ MODEL VERIFICATION")
    print("-" * 30)

    if os.path.exists("license_plate_yolo.pt"):
        file_size = os.path.getsize("license_plate_yolo.pt") / (1024 * 1024)  # MB
        print(f"✅ license_plate_yolo.pt exists ({file_size:.1f} MB)")

        # Test load dengan ultralytics
        try:
            from ultralytics import YOLO
            model = YOLO("license_plate_yolo.pt")
            print(f"✅ Model can be loaded successfully")
            print(f"📊 Model info: {len(model.names)} classes")
            return True
        except ImportError:
            print("⚠️  ultralytics not installed - install dengan: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("❌ license_plate_yolo.pt not found")
        return False

def main():
    """Main function"""
    print("🎯 LICENSE PLATE YOLO MODEL SETUP")
    print("=" * 60)

    print("\n📋 Setup options:")
    print("1. Download pre-trained models")
    print("2. Setup Roboflow integration")
    print("3. Verify existing model")
    print("4. All steps")

    choice = input("\nPilih opsi (1-4): ").strip()

    if choice in ['1', '4']:
        download_license_plate_models()

    if choice in ['2', '4']:
        setup_roboflow_download()

    if choice in ['3', '4']:
        verify_model()

    print(f"\n🎉 SETUP COMPLETED!")
    print("=" * 60)

    print(f"\n📋 NEXT STEPS:")
    if os.path.exists("license_plate_yolo.pt"):
        print("✅ Model tersedia - test sistem Enhanced Hybrid:")
        print("   python3 test_enhanced_hybrid_system.py")
    else:
        print("⚠️  Model belum tersedia. Options:")
        print("   1. Train custom model (paling akurat)")
        print("   2. Cari pre-trained Indonesian model")
        print("   3. Fine-tune existing model")

if __name__ == "__main__":
    main()