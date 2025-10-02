#!/usr/bin/env python3
"""
Start Enhanced License Plate Detection System
Script khusus untuk memulai sistem dengan enhanced detection
"""

import subprocess
import sys
import os
import argparse

def check_dependencies():
    """Check if enhanced detection dependencies are available"""
    required_packages = [
        'scikit-image',
        'scipy',
        'opencv-python',
        'pytesseract',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            elif package == 'scipy':
                import scipy
            elif package == 'pytesseract':
                import pytesseract
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies available")
    return True

def check_tesseract():
    """Check if Tesseract is properly installed"""
    try:
        import pytesseract
        # Try to get version
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract {version} detected")
        return True
    except Exception as e:
        print(f"❌ Tesseract not properly configured: {str(e)}")
        print("Install Tesseract:")
        print("  macOS: brew install tesseract tesseract-lang")
        print("  Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-ind")
        print("  Windows: Download from GitHub releases")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced License Plate Detection System')
    parser.add_argument('--source', '-s', type=str, default='0',
                        help='Video source: RTSP URL, webcam index (0), or video file path')
    parser.add_argument('--test-only', action='store_true',
                        help='Run test suite only')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip dependency checks')
    parser.add_argument('--motorcycle-mode', action='store_true',
                        help='Enable motorcycle detection mode')
    parser.add_argument('--super-resolution', action='store_true',
                        help='Enable super-resolution for very small plates')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable preview window')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced License Plate Detection System")
    print("=" * 50)
    
    # Check dependencies
    if not args.skip_checks:
        print("Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        if not check_tesseract():
            sys.exit(1)
    
    # Run test suite if requested
    if args.test_only:
        print("\n🧪 Running Enhanced Detection Test Suite...")
        try:
            subprocess.run([sys.executable, 'test_enhanced_detection.py'], check=True)
        except subprocess.CalledProcessError:
            print("❌ Test suite failed")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ test_enhanced_detection.py not found")
            sys.exit(1)
        return
    
    # Build command arguments
    cmd_args = [
        sys.executable, 'main.py',
        '--source', args.source,
        '--enhanced'  # Always use enhanced mode
    ]
    
    if args.motorcycle_mode:
        cmd_args.append('--motorcycle-mode')
    
    if args.super_resolution:
        cmd_args.append('--super-resolution')
    
    if args.no_preview:
        cmd_args.append('--no-preview')
    
    # Start the system
    print(f"\n🎥 Starting Enhanced Detection System...")
    print(f"Source: {args.source}")
    print("Features enabled:")
    print("  ✅ Enhanced Detection (super-resolution, ensemble OCR)")
    print("  ✅ Multi-scale detection")
    print("  ✅ Adaptive image enhancement")
    
    if args.motorcycle_mode:
        print("  ✅ Motorcycle detection mode")
    
    if args.super_resolution:
        print("  ✅ Super-resolution for small plates")
    
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'p' - Print statistics")
    
    print("\n" + "=" * 50)
    
    try:
        # Execute the main program
        subprocess.run(cmd_args, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ System failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("❌ main.py not found")
        sys.exit(1)

if __name__ == "__main__":
    main()