#!/usr/bin/env python3
"""
Train Indonesian License Plate YOLO Model
Complete training pipeline untuk Indonesian license plate detection
"""

import os
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check prerequisites untuk training"""
    print("🔧 CHECKING PREREQUISITES")
    print("-" * 40)

    requirements = {
        'ultralytics': 'pip install ultralytics',
        'torch': 'pip install torch torchvision',
        'opencv-python': 'pip install opencv-python',
        'Pillow': 'pip install Pillow'
    }

    missing = []
    for package, install_cmd in requirements.items():
        try:
            __import__(package if package != 'opencv-python' else 'cv2')
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - install with: {install_cmd}")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Please install missing packages first!")
        return False

    print("✅ All prerequisites satisfied")
    return True

def create_dataset_structure():
    """Create dataset directory structure"""
    print("\n📁 CREATING DATASET STRUCTURE")
    print("-" * 40)

    dirs = [
        "dataset/indonesian_plates",
        "dataset/indonesian_plates/train/images",
        "dataset/indonesian_plates/train/labels",
        "dataset/indonesian_plates/val/images",
        "dataset/indonesian_plates/val/labels",
        "dataset/indonesian_plates/test/images",
        "dataset/indonesian_plates/test/labels"
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

    # Create dataset.yaml
    dataset_yaml = {
        'path': './dataset/indonesian_plates',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,  # Number of classes (license plate)
        'names': ['license_plate']
    }

    with open('dataset/indonesian_plates/dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print("✅ Dataset structure created")
    return True

def generate_sample_annotations():
    """Generate sample annotation format"""
    print("\n📝 ANNOTATION FORMAT GUIDE")
    print("-" * 40)

    # Create sample annotation file
    sample_annotation = """# YOLO annotation format untuk license plate
# File: image001.txt (same name as image001.jpg)
# Format: class x_center y_center width height (normalized 0-1)

# Example annotation untuk plat "B 1234 ABC":
0 0.5 0.6 0.3 0.1

# Explanation:
# 0       = class ID (license_plate)
# 0.5     = x center (50% dari image width)
# 0.6     = y center (60% dari image height)
# 0.3     = width (30% dari image width)
# 0.1     = height (10% dari image height)
"""

    with open('dataset/annotation_guide.txt', 'w') as f:
        f.write(sample_annotation)

    print("✅ Annotation guide created: dataset/annotation_guide.txt")

    # Generate sample training script
    training_tips = """
# TIPS UNTUK COLLECT INDONESIAN PLATE DATA:

1. SUMBER DATA:
   - Screenshot dari CCTV traffic cameras
   - Video dashcam Indonesia
   - Google Street View Indonesia
   - Dataset kendaraan Indonesia
   - Foto manual di jalan raya

2. VARIASI YANG DIBUTUHKAN:
   - Format: B1234ABC, D5678XYZ, AA1234BB
   - Kendaraan: mobil, motor, truck, bus
   - Kondisi: siang/malam, hujan, berdebu
   - Angle: depan, belakang, miring
   - Jarak: dekat, sedang, jauh

3. JUMLAH MINIMUM:
   - Train: 800+ images dengan annotations
   - Validation: 200+ images
   - Test: 100+ images
   - Total: 1000+ annotated Indonesian plates

4. ANNOTATION TOOLS:
   - LabelImg (free, easy to use)
   - Roboflow (online, auto-augmentation)
   - CVAT (advanced, collaborative)
   - Labelbox (professional)
"""

    with open('dataset/data_collection_guide.txt', 'w') as f:
        f.write(training_tips)

    print("✅ Data collection guide created: dataset/data_collection_guide.txt")
    return True

def create_training_script():
    """Create training script"""
    print("\n🚀 CREATING TRAINING SCRIPT")
    print("-" * 40)

    training_script = '''#!/usr/bin/env python3
"""
Indonesian License Plate YOLOv8 Training Script
Run this after you have prepared the dataset
"""

from ultralytics import YOLO
import os

def train_indonesian_plate_model():
    """Train YOLOv8 model untuk Indonesian license plates"""

    print("🚀 Starting Indonesian License Plate Training")
    print("=" * 50)

    # Load base YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load YOLOv8 nano

    # Training parameters optimized untuk license plates
    results = model.train(
        data='dataset/indonesian_plates/dataset.yaml',
        epochs=100,                    # Adjust based on dataset size
        imgsz=640,                    # Image size
        batch=16,                     # Batch size (adjust based on GPU memory)
        lr0=0.01,                     # Initial learning rate
        lrf=0.1,                      # Final learning rate factor
        momentum=0.937,               # Momentum
        weight_decay=0.0005,          # Weight decay
        warmup_epochs=3,              # Warmup epochs
        warmup_momentum=0.8,          # Warmup momentum
        box=7.5,                      # Box loss gain
        cls=0.5,                      # Class loss gain (lower for single class)
        dfl=1.5,                      # DFL loss gain
        pose=12.0,                    # Pose loss gain (not used for detection)
        kobj=1.0,                     # Keypoint obj loss gain (not used)
        label_smoothing=0.0,          # Label smoothing
        nbs=64,                       # Nominal batch size
        overlap_mask=True,            # Overlap masks
        mask_ratio=4,                 # Mask downsample ratio
        dropout=0.0,                  # Use dropout
        val=True,                     # Validate during training
        plots=True,                   # Create training plots
        save=True,                    # Save model checkpoints
        save_period=-1,               # Save checkpoint every x epochs (-1 = disabled)
        cache=False,                  # Use dataset caching
        device='',                    # Device to run on ('' = auto-detect)
        workers=8,                    # Number of worker threads
        project='runs/detect',        # Project directory
        name='indonesian_plates',     # Experiment name
        exist_ok=False,               # Overwrite existing experiment
        pretrained=True,              # Use pretrained model
        optimizer='auto',             # Optimizer (auto, SGD, Adam, AdamW, RMSProp)
        verbose=True,                 # Verbose output
        seed=0,                       # Random seed
        deterministic=True,           # Deterministic training
        single_cls=True,              # Single class training (license plate only)
        rect=False,                   # Rectangular training
        cos_lr=False,                 # Cosine learning rate scheduler
        close_mosaic=10,              # Close mosaic augmentation (epochs)
        resume=False,                 # Resume training
        amp=True,                     # Automatic Mixed Precision
        fraction=1.0,                 # Dataset fraction to train on
        profile=False,                # Profile ONNX and TensorRT speeds
    )

    # Export trained model
    model.export(format='pt')  # Export to PyTorch format

    # Copy best model as license_plate_yolo.pt
    import shutil
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    if best_model_path.exists():
        shutil.copy(best_model_path, 'license_plate_yolo.pt')
        print(f"✅ Best model saved as: license_plate_yolo.pt")

    print("🎉 Training completed!")
    return True

if __name__ == "__main__":
    train_indonesian_plate_model()
'''

    with open('train_model.py', 'w') as f:
        f.write(training_script)

    print("✅ Training script created: train_model.py")
    return True

def create_quick_test_model():
    """Create quick test dengan existing model untuk demo"""
    print("\n⚡ CREATING QUICK TEST MODEL")
    print("-" * 40)

    try:
        from ultralytics import YOLO

        # Download base YOLOv8 model dan rename untuk testing
        if not os.path.exists("license_plate_yolo.pt"):
            print("📥 Downloading YOLOv8 base model untuk testing...")
            model = YOLO('yolov8n.pt')  # This will download if not exists

            # Save sebagai license_plate_yolo.pt untuk testing
            model.model.save('license_plate_yolo.pt')
            print("✅ Test model created: license_plate_yolo.pt")
            print("⚠️  NOTE: Ini adalah base YOLO, bukan trained untuk plates")
            print("📋 Untuk hasil optimal, train dengan Indonesian data")

        return True

    except ImportError:
        print("❌ ultralytics not installed")
        print("   Install dengan: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main training setup function"""
    print("🎯 INDONESIAN LICENSE PLATE YOLO TRAINING SETUP")
    print("=" * 60)

    print("\n📋 Setup options:")
    print("1. Check prerequisites")
    print("2. Create dataset structure")
    print("3. Create training scripts")
    print("4. Create quick test model")
    print("5. Full setup (all steps)")

    choice = input("\nPilih opsi (1-5): ").strip()

    success = True

    if choice in ['1', '5']:
        success &= check_prerequisites()

    if choice in ['2', '5']:
        success &= create_dataset_structure()
        success &= generate_sample_annotations()

    if choice in ['3', '5']:
        success &= create_training_script()

    if choice in ['4', '5']:
        success &= create_quick_test_model()

    if success:
        print(f"\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print(f"\n📋 NEXT STEPS:")
        if choice in ['2', '3', '5']:
            print("1. 📷 Collect Indonesian license plate images (1000+ recommended)")
            print("2. 🏷️  Annotate dengan LabelImg atau Roboflow")
            print("3. 📁 Place images di dataset/indonesian_plates/train/images/")
            print("4. 📁 Place annotations di dataset/indonesian_plates/train/labels/")
            print("5. 🚀 Run training: python3 train_model.py")
            print("6. ✅ Test hasil: python3 test_enhanced_hybrid_system.py")

        if choice in ['4']:
            print("1. ✅ Test basic functionality: python3 test_enhanced_hybrid_system.py")
            print("2. 📊 Untuk akurasi optimal, train dengan Indonesian data")

        print(f"\n💡 TIPS:")
        print("   - Gunakan GPU untuk training lebih cepat")
        print("   - Minimum 1000 annotated images untuk hasil optimal")
        print("   - Variasikan kondisi: siang/malam, berbagai angle, jarak")
        print("   - Format Indonesian: B1234ABC, D5678XYZ, AA1234BB")

    else:
        print(f"\n❌ SETUP INCOMPLETE")
        print("Please resolve the issues above and try again")

if __name__ == "__main__":
    main()