#!/usr/bin/env python3
"""
Download Custom License Plate Detection Model
Model khusus untuk detect plat nomor (bukan mobil!)
"""

import os
import sys
import urllib.request
from pathlib import Path

def download_with_progress(url, output_path):
    """Download file dengan progress bar"""
    print(f"\n⏬ Downloading from: {url}")
    print(f"💾 Saving to: {output_path}\n")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            sys.stdout.write(f"\r⏬ Progress: {percent}% ({downloaded_mb:.1f}MB / {total_mb:.1f}MB)")
            sys.stdout.flush()
        else:
            downloaded_mb = count * block_size / 1024 / 1024
            sys.stdout.write(f"\r⏬ Downloaded: {downloaded_mb:.1f}MB")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, reporthook)
        sys.stdout.write("\n")
        return True
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("🚗 CUSTOM LICENSE PLATE DETECTION MODEL DOWNLOADER")
    print("="*70 + "\n")

    # Create models folder
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("📋 Available Custom Models for License Plate Detection:\n")

    print("="*70)
    print("🏆 RECOMMENDED: General License Plate Model")
    print("="*70)
    print("Source: Ultralytics GitHub")
    print("Trained on: Multi-country license plates")
    print("Size: ~6-50MB (depends on version)")
    print("Accuracy: HIGH")
    print()
    print("URL Options:")
    print("1. YOLOv8n-plate (small, fast)")
    print("2. YOLOv8s-plate (medium, balanced)")
    print("3. YOLOv8m-plate (large, accurate)")
    print()

    print("="*70)
    print("📝 MANUAL DOWNLOAD (RECOMMENDED)")
    print("="*70)
    print()
    print("Karena model custom perlu authentication/specific version,")
    print("saya akan guide Anda download manual:\n")

    print("🔹 OPTION 1: Roboflow Universe (Easiest)")
    print("-" * 70)
    print("1. Visit: https://universe.roboflow.com/")
    print("2. Search: 'license plate detection'")
    print("3. Filter: YOLOv8, high images count")
    print("4. Recommended datasets:")
    print("   • 'yolo-plate' by new-workspace-ertfx (2702 images)")
    print("   • 'License Plates' by Sezgin KOC (926 images)")
    print("   • 'ALPR YOLOv8' by PPMG Burgas (675 images)")
    print()
    print("5. Click 'Download Dataset'")
    print("6. Select: 'YOLOv8' format")
    print("7. Download ZIP file")
    print("8. Extract and find 'weights/best.pt' or 'best.pt'")
    print("9. Copy to: models/best.pt")
    print()

    print("🔹 OPTION 2: Direct YOLOv8 Pretrained")
    print("-" * 70)
    print("Download from Ultralytics official:")
    print()

    models = {
        "1": {
            "name": "YOLOv8n (Nano - Fast)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            "size": "~6MB",
            "note": "General model - detect mobil, bukan plat khusus!"
        },
        "2": {
            "name": "YOLOv8s (Small - Balanced)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
            "size": "~22MB",
            "note": "General model - detect mobil, bukan plat khusus!"
        },
        "3": {
            "name": "YOLOv8m (Medium - Accurate)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
            "size": "~50MB",
            "note": "General model - detect mobil, bukan plat khusus!"
        }
    }

    for key, model in models.items():
        print(f"{key}. {model['name']}")
        print(f"   Size: {model['size']}")
        print(f"   Note: {model['note']}")
        print()

    print("⚠️  WARNING: Options 1-3 adalah general models!")
    print("   Mereka detect MOBIL, bukan PLAT khusus.")
    print("   Untuk detect plat, gunakan OPTION 1 (Roboflow) di atas!")
    print()

    print("="*70)
    choice = input("\nPilih (1-3 untuk general model, Enter untuk skip): ").strip()

    if choice in models:
        model = models[choice]
        output_path = models_dir / "best.pt"

        print(f"\n🔧 Downloading {model['name']}...")
        print(f"⚠️  Note: {model['note']}")

        proceed = input("\nLanjutkan download general model? (y/n): ").strip().lower()
        if proceed != 'y':
            print("❌ Download cancelled")
            return

        success = download_with_progress(model['url'], output_path)

        if success:
            print("\n✅ Model downloaded successfully!")
            print(f"📁 Saved to: {output_path}")
            print(f"📊 Size: {output_path.stat().st_size / 1024 / 1024:.2f}MB")
        else:
            print("\n❌ Download failed")
            return

    else:
        print("\n📝 Manual Download Guide:")
        print()
        print("Untuk mendapatkan model KHUSUS LICENSE PLATE:")
        print()
        print("1. Buka browser:")
        print("   https://universe.roboflow.com/search?q=license+plate+yolov8")
        print()
        print("2. Pilih dataset dengan:")
        print("   ✓ Images count: >500")
        print("   ✓ Model type: YOLOv8")
        print("   ✓ Health score: Good")
        print()
        print("3. Download dan extract")
        print()
        print("4. Copy file 'best.pt' ke folder 'models/':")
        print(f"   cp path/to/extracted/weights/best.pt {models_dir.absolute()}/best.pt")
        print()

    # Check final status
    best_pt = models_dir / "best.pt"
    if best_pt.exists():
        print("\n" + "="*70)
        print("✅ MODEL READY")
        print("="*70)
        print(f"\n📁 Model: {best_pt.absolute()}")
        print(f"📊 Size: {best_pt.stat().st_size / 1024 / 1024:.2f}MB")
        print()
        print("📋 Next Steps:")
        print("1. Test model:")
        print("   python3 test_yolo_detection.py image7.png")
        print()
        print("2. Run application:")
        print("   python3 app.py")
        print()
    else:
        print("\n" + "="*70)
        print("⚠️  MODEL NOT FOUND")
        print("="*70)
        print()
        print("Download model manually dan save ke:")
        print(f"   {models_dir.absolute()}/best.pt")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
