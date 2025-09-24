#!/usr/bin/env python3
"""
Fine-tune License Plate YOLO Model untuk Indonesian Plates
Quick adaptation dari existing license plate models
"""

import os
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_pretrained_plate_model():
    """Download pre-trained license plate model untuk fine-tuning"""
    print("📥 DOWNLOADING PRE-TRAINED LICENSE PLATE MODEL")
    print("-" * 50)

    # URLs untuk pre-trained models (contoh - perlu disesuaikan)
    model_options = {
        "yolov8n_coco": {
            "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "filename": "yolov8n_base.pt",
            "description": "YOLOv8 Nano COCO pre-trained"
        }
    }

    # Download base model jika belum ada
    for key, info in model_options.items():
        if not os.path.exists(info["filename"]):
            logger.info(f"Downloading {info['description']}...")
            try:
                response = requests.get(info["url"], stream=True)
                response.raise_for_status()

                with open(info["filename"], 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"✅ {info['filename']} downloaded successfully")
            except Exception as e:
                logger.error(f"❌ Error downloading {info['filename']}: {e}")
                return False
        else:
            logger.info(f"✅ {info['filename']} already exists")

    return True

def create_indonesian_fine_tuning_data():
    """Create sample Indonesian plate data untuk fine-tuning"""
    print("\n📋 CREATING INDONESIAN FINE-TUNING DATASET")
    print("-" * 50)

    # Create directories
    dirs = [
        "finetune_data/images",
        "finetune_data/labels"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Create dataset YAML untuk fine-tuning
    dataset_yaml = """# Indonesian License Plate Fine-tuning Dataset
path: ./finetune_data
train: images
val: images  # Same as train for small dataset
test: images

# Classes
nc: 1
names: ['license_plate']

# Indonesian plate characteristics
plate_formats:
  - "B1234ABC"   # Jakarta format
  - "D5678XYZ"   # Bandung format
  - "AA1234BB"   # Two letter prefix
  - "F9876GHI"   # Bogor format

regional_codes:
  - B   # Jakarta
  - D   # Bandung
  - E   # Cirebon
  - F   # Bogor
  - AA  # Kedu
  - AB  # Yogyakarta
  - AD  # Solo
"""

    with open('finetune_data/dataset.yaml', 'w') as f:
        f.write(dataset_yaml)

    # Create Indonesian plate annotation examples
    annotation_examples = """# INDONESIAN LICENSE PLATE ANNOTATION EXAMPLES

# File: B1234ABC_001.txt
# Image: B1234ABC_001.jpg (640x480)
# Plate location: center-bottom area
0 0.5 0.7 0.25 0.08

# File: D5678XYZ_002.txt
# Image: D5678XYZ_002.jpg (1280x720)
# Plate location: slightly left, center vertical
0 0.45 0.5 0.2 0.06

# File: AA9876BB_003.txt
# Image: AA9876BB_003.jpg (800x600)
# Plate location: right side, upper area
0 0.65 0.35 0.18 0.07

# Annotation format explanation:
# class_id x_center y_center width height (all normalized 0-1)
#
# class_id: 0 (license_plate)
# x_center: horizontal center (0=left, 1=right)
# y_center: vertical center (0=top, 1=bottom)
# width: plate width relative to image width
# height: plate height relative to image height
"""

    with open('finetune_data/annotation_examples.txt', 'w') as f:
        f.write(annotation_examples)

    print("✅ Fine-tuning dataset structure created")
    return True

def create_fine_tuning_script():
    """Create fine-tuning script"""
    print("\n🔧 CREATING FINE-TUNING SCRIPT")
    print("-" * 50)

    fine_tune_script = '''#!/usr/bin/env python3
"""
Fine-tune YOLOv8 for Indonesian License Plates
Lightweight training dengan small dataset
"""

from ultralytics import YOLO
import os

def finetune_for_indonesian_plates():
    """Fine-tune existing model untuk Indonesian characteristics"""

    print("🔧 FINE-TUNING FOR INDONESIAN LICENSE PLATES")
    print("=" * 50)

    # Load pre-trained model
    model_file = "yolov8n_base.pt"
    if not os.path.exists(model_file):
        print(f"❌ {model_file} not found. Run download script first.")
        return False

    model = YOLO(model_file)
    print(f"✅ Loaded base model: {model_file}")

    # Fine-tuning parameters (optimized untuk small dataset)
    print("🚀 Starting fine-tuning...")

    results = model.train(
        data='finetune_data/dataset.yaml',
        epochs=50,                      # Less epochs for fine-tuning
        imgsz=640,                     # Standard image size
        batch=8,                       # Smaller batch for fine-tuning
        lr0=0.001,                     # Lower learning rate
        lrf=0.1,                       # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=2,               # Shorter warmup
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        nbs=32,                        # Smaller nominal batch size
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
        save=True,
        save_period=10,                # Save every 10 epochs
        cache=False,
        device='',
        workers=4,                     # Fewer workers
        project='runs/finetune',
        name='indonesian_plates_ft',
        exist_ok=False,
        pretrained=True,               # Important: use pretrained weights
        optimizer='Adam',              # Adam often better for fine-tuning
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=True,               # Single class (license plate)
        rect=False,
        cos_lr=True,                   # Cosine learning rate (good for fine-tuning)
        close_mosaic=5,                # Close mosaic augmentation early
        resume=False,
        amp=True,
        fraction=1.0,
        profile=False,

        # Fine-tuning specific parameters
        freeze=10,                     # Freeze first 10 layers (backbone)
    )

    # Export fine-tuned model
    best_model = f"runs/finetune/indonesian_plates_ft/weights/best.pt"
    if os.path.exists(best_model):
        # Copy to main directory
        import shutil
        shutil.copy(best_model, 'license_plate_yolo.pt')
        print("✅ Fine-tuned model saved as: license_plate_yolo.pt")

        # Test the model
        test_model = YOLO('license_plate_yolo.pt')
        print(f"📊 Model loaded successfully with {len(test_model.names)} classes")
        print(f"🎯 Classes: {test_model.names}")

        return True
    else:
        print(f"❌ Training failed - best model not found")
        return False

def create_synthetic_data():
    """Create synthetic Indonesian plate data untuk demonstration"""

    print("🎨 CREATING SYNTHETIC TRAINING DATA")
    print("-" * 40)

    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        # Indonesian plate formats
        plate_formats = [
            "B 1234 ABC", "D 5678 XYZ", "F 9876 GHI",
            "AA 1234 BB", "AB 5678 CD", "AD 9999 EF",
            "E 1111 ABC", "G 2222 DEF", "H 3333 GHI"
        ]

        # Create synthetic plate images
        for i, plate_text in enumerate(plate_formats):
            # Create white background
            img = np.ones((120, 300, 3), dtype=np.uint8) * 255

            # Add black border
            cv2.rectangle(img, (5, 5), (295, 115), (0, 0, 0), 2)

            # Add text (simplified - real implementation would use better fonts)
            cv2.putText(img, plate_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

            # Save image
            img_path = f"finetune_data/images/synthetic_plate_{i:03d}.jpg"
            cv2.imwrite(img_path, img)

            # Create corresponding annotation
            # Plate occupies most of the image
            annotation = "0 0.5 0.5 0.9 0.8\\n"  # Full plate area

            label_path = f"finetune_data/labels/synthetic_plate_{i:03d}.txt"
            with open(label_path, 'w') as f:
                f.write(annotation)

        print(f"✅ Created {len(plate_formats)} synthetic plate images")
        print("⚠️  NOTE: Synthetic data untuk demo only - use real data untuk production")
        return True

    except ImportError as e:
        print(f"❌ Missing dependencies for synthetic data: {e}")
        print("   Install dengan: pip install opencv-python pillow")
        return False
    except Exception as e:
        print(f"❌ Error creating synthetic data: {e}")
        return False

if __name__ == "__main__":
    print("🎯 INDONESIAN LICENSE PLATE FINE-TUNING")
    print("=" * 50)

    print("\\n📋 Options:")
    print("1. Create synthetic demo data")
    print("2. Run fine-tuning")
    print("3. Both (recommended)")

    choice = input("\\nPilih opsi (1-3): ").strip()

    success = True

    if choice in ['1', '3']:
        success &= create_synthetic_data()

    if choice in ['2', '3']:
        success &= finetune_for_indonesian_plates()

    if success:
        print("\\n🎉 FINE-TUNING COMPLETED!")
        print("✅ Test model dengan: python3 test_enhanced_hybrid_system.py")
    else:
        print("\\n❌ FINE-TUNING FAILED")
        print("Please check the errors above")
'''

    with open('finetune_model.py', 'w') as f:
        f.write(fine_tune_script)

    print("✅ Fine-tuning script created: finetune_model.py")
    return True

def main():
    """Main fine-tuning setup"""
    print("🎯 LICENSE PLATE FINE-TUNING SETUP")
    print("=" * 50)

    steps = [
        ("Download pre-trained model", download_pretrained_plate_model),
        ("Create dataset structure", create_indonesian_fine_tuning_data),
        ("Create fine-tuning script", create_fine_tuning_script)
    ]

    print("\n🚀 Setting up fine-tuning environment...")

    success = True
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        success &= step_func()

    if success:
        print(f"\n🎉 FINE-TUNING SETUP COMPLETED!")
        print("=" * 60)

        print(f"\n📋 NEXT STEPS:")
        print("1. 📷 Add real Indonesian plate images ke finetune_data/images/")
        print("2. 🏷️  Create corresponding annotations in finetune_data/labels/")
        print("3. 🚀 Run fine-tuning: python3 finetune_model.py")
        print("4. ✅ Test result: python3 test_enhanced_hybrid_system.py")

        print(f"\n💡 ADVANTAGES OF FINE-TUNING:")
        print("   ✅ Faster training (30-60 minutes)")
        print("   ✅ Less data required (100+ images)")
        print("   ✅ Good accuracy for Indonesian plates")
        print("   ✅ Leverages existing knowledge")

    else:
        print(f"\n❌ SETUP FAILED")
        print("Please resolve issues and try again")

if __name__ == "__main__":
    main()