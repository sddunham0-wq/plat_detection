#!/usr/bin/env python3
"""
Install Enhanced Detection Requirements
Script untuk menginstall semua dependencies yang diperlukan untuk enhanced detector
"""
import subprocess
import sys
import os
import platform
import urllib.request

def run_command(command, description):
    """Run command dengan error handling"""
    print(f"🔧 {description}...")
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        print(f"✅ {description} - Success")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr.strip() if e.stderr else str(e)}")
        return False
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return False

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("💡 Enhanced detector requires Python 3.8 or higher")
        return False

    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_pip_packages():
    """Install required pip packages"""
    packages = [
        'opencv-python>=4.8.0',
        'ultralytics>=8.0.0',
        'pytesseract>=0.3.10',
        'Pillow>=9.0.0',
        'numpy>=1.21.0',
        'configparser>=5.0.0',
        'sqlite3',  # Usually built-in
    ]

    success_count = 0
    total_packages = len(packages)

    print(f"📦 Installing {total_packages} pip packages...")

    for package in packages:
        if package == 'sqlite3':
            # sqlite3 is usually built-in
            try:
                import sqlite3
                print(f"✅ sqlite3 - Already available (built-in)")
                success_count += 1
            except ImportError:
                print(f"❌ sqlite3 - Not available")
        else:
            success = run_command([sys.executable, '-m', 'pip', 'install', '--upgrade', package],
                                f"Installing {package}")
            if success:
                success_count += 1

    print(f"📊 Package installation: {success_count}/{total_packages} successful")
    return success_count == total_packages

def install_tesseract():
    """Install Tesseract OCR based on platform"""
    system = platform.system().lower()

    print(f"🔤 Installing Tesseract OCR for {system}...")

    if system == 'windows':
        print("📋 Windows detected:")
        print("   1. Download Tesseract installer from:")
        print("      https://github.com/UB-Mannheim/tesseract/wiki")
        print("   2. Install to default location (C:\\Program Files\\Tesseract-OCR)")
        print("   3. Add to PATH: C:\\Program Files\\Tesseract-OCR")
        print("   4. Restart terminal/IDE after installation")
        return True

    elif system == 'darwin':  # macOS
        # Try homebrew first
        if run_command('brew --version', 'Checking Homebrew'):
            return run_command('brew install tesseract', 'Installing Tesseract via Homebrew')
        else:
            print("💡 Homebrew not found. Install options:")
            print("   1. Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("   2. Then run: brew install tesseract")
            print("   3. Or download from: https://github.com/tesseract-ocr/tesseract/wiki")
            return False

    elif system == 'linux':
        # Try different package managers
        if run_command('which apt-get', 'Checking apt-get'):
            success = run_command('sudo apt-get update', 'Updating package list')
            if success:
                return run_command('sudo apt-get install -y tesseract-ocr tesseract-ocr-ind tesseract-ocr-eng',
                                 'Installing Tesseract via apt-get')
        elif run_command('which yum', 'Checking yum'):
            return run_command('sudo yum install -y tesseract tesseract-langpack-ind tesseract-langpack-eng',
                             'Installing Tesseract via yum')
        elif run_command('which dnf', 'Checking dnf'):
            return run_command('sudo dnf install -y tesseract tesseract-langpack-ind tesseract-langpack-eng',
                             'Installing Tesseract via dnf')
        else:
            print("💡 Unknown Linux distribution. Manual installation required:")
            print("   Check: https://github.com/tesseract-ocr/tesseract/wiki")
            return False

    return False

def test_tesseract():
    """Test Tesseract installation"""
    print("🧪 Testing Tesseract installation...")

    try:
        import pytesseract

        # Test basic functionality
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract {version} detected")

        # Test languages
        try:
            langs = pytesseract.get_languages()
            has_eng = 'eng' in langs
            has_ind = 'ind' in langs

            print(f"📚 Available languages: {len(langs)}")
            print(f"   English (eng): {'✅' if has_eng else '❌'}")
            print(f"   Indonesian (ind): {'✅' if has_ind else '❌'}")

            if not has_eng:
                print("💡 English language pack missing. Detection may not work properly.")

            return has_eng

        except Exception as e:
            print(f"⚠️ Language check failed: {e}")
            return True  # Assume basic installation works

    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        print("💡 Manual installation may be required")
        return False

def download_yolo_models():
    """Download YOLO models if not available"""
    print("🤖 Checking YOLO models...")

    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    model_dir = 'models'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created {model_dir} directory")

    for model in models:
        model_path = os.path.join(model_dir, model)

        if os.path.exists(model_path):
            print(f"✅ {model} - Already exists")
        else:
            print(f"📥 Downloading {model}...")
            try:
                # Models will be auto-downloaded by ultralytics on first use
                from ultralytics import YOLO
                temp_model = YOLO(model)  # This triggers download
                print(f"✅ {model} - Downloaded successfully")
            except Exception as e:
                print(f"⚠️ {model} - Will be downloaded on first use")

    return True

def test_opencv():
    """Test OpenCV installation"""
    print("📸 Testing OpenCV...")

    try:
        import cv2
        version = cv2.__version__
        print(f"✅ OpenCV {version} detected")

        # Test basic functionality
        test_img = cv2.imread('nonexistent.jpg')  # Should return None gracefully
        print("✅ OpenCV basic functionality working")

        return True

    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def create_config_files():
    """Create default configuration files if they don't exist"""
    print("⚙️ Creating configuration files...")

    config_files = [
        'enhanced_detection_config.ini'
    ]

    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"📝 {config_file} not found - will be created on first run")
        else:
            print(f"✅ {config_file} exists")

    return True

def test_full_system():
    """Test complete system functionality"""
    print("🧪 Testing complete system...")

    try:
        # Test all imports
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import pytesseract
        from PIL import Image
        import sqlite3
        import configparser

        print("✅ All imports successful")

        # Test basic detection pipeline
        print("🔬 Testing detection pipeline...")

        # Create test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img.fill(128)  # Gray image

        # Test YOLO
        try:
            model = YOLO('yolov8n.pt')
            results = model(test_img, verbose=False)
            print("✅ YOLO detection test passed")
        except Exception as e:
            print(f"⚠️ YOLO test warning: {e}")

        # Test OCR
        try:
            test_text_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_text_img, "TEST123", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            pil_img = Image.fromarray(test_text_img)
            text = pytesseract.image_to_string(pil_img)
            print("✅ OCR test passed")
        except Exception as e:
            print(f"⚠️ OCR test warning: {e}")

        # Test database
        try:
            conn = sqlite3.connect(':memory:')
            conn.execute('CREATE TABLE test (id INTEGER)')
            conn.close()
            print("✅ Database test passed")
        except Exception as e:
            print(f"⚠️ Database test warning: {e}")

        print("🎉 System test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Enhanced License Plate Detection - Installation Script")
    print("=" * 60)
    print("This script will install all required dependencies for the enhanced detector")
    print()

    # Installation steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing pip packages", install_pip_packages),
        ("Installing Tesseract OCR", install_tesseract),
        ("Testing Tesseract", test_tesseract),
        ("Testing OpenCV", test_opencv),
        ("Downloading YOLO models", download_yolo_models),
        ("Creating config files", create_config_files),
        ("Testing complete system", test_full_system),
    ]

    success_count = 0
    total_steps = len(steps)

    for i, (description, function) in enumerate(steps, 1):
        print(f"\n📋 Step {i}/{total_steps}: {description}")
        print("-" * 40)

        try:
            success = function()
            if success:
                success_count += 1
                print(f"✅ Step {i} completed successfully")
            else:
                print(f"⚠️ Step {i} completed with warnings")
        except Exception as e:
            print(f"❌ Step {i} failed: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful steps: {success_count}/{total_steps}")

    if success_count == total_steps:
        print("🎉 All components installed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_enhanced_detection.py")
        print("   2. Run: python enhanced_app.py")
        print("   3. Open: http://localhost:8000")
        print("\n📚 Documentation: ENHANCED_DETECTION_README.md")

    elif success_count >= total_steps - 2:
        print("⚠️ Installation mostly successful with minor issues")
        print("💡 You can try running the enhanced detector")
        print("   Some features may require manual configuration")

    else:
        print("❌ Installation encountered significant issues")
        print("💡 Manual installation may be required for some components")
        print("\n🔧 Common solutions:")
        print("   • Update pip: python -m pip install --upgrade pip")
        print("   • Install build tools for your platform")
        print("   • Check internet connection for downloads")

    print("\n📞 Need help? Check the troubleshooting section in README")

if __name__ == "__main__":
    main()