#!/usr/bin/env python3
"""
Quick Start Script dengan YOLOv8 Auto-Detection
Startup cepat dengan background YOLOv8 loading
"""

import subprocess
import sys
import time
import os
import threading

def check_dependencies():
    """Quick check dependencies"""
    print("🔍 Quick dependency check...")
    
    try:
        import cv2
        import flask
        import flask_socketio
        print("✅ Core dependencies OK")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Installing dependencies...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], check=True)
            print("✅ Dependencies installed")
            return True
        except:
            print("❌ Failed to install dependencies")
            return False

def check_ultralytics_background():
    """Check ultralytics in background"""
    def check_and_install():
        try:
            import ultralytics
            print("✅ YOLOv8 already available")
        except ImportError:
            print("📦 Installing YOLOv8 in background...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'ultralytics'
                ], capture_output=True, check=True)
                print("✅ YOLOv8 installed successfully")
            except:
                print("⚠️  YOLOv8 installation failed, will run without object detection")
    
    # Run in background
    thread = threading.Thread(target=check_and_install, daemon=True)
    thread.start()
    return thread

def start_server_fast():
    """Start server dengan optimasi"""
    print("\n🚀 Starting CCTV Detection Server...")
    print("=" * 50)
    
    # Quick dependency check
    if not check_dependencies():
        print("❌ Cannot start without core dependencies")
        return False
    
    # Start background YOLOv8 check
    yolo_thread = check_ultralytics_background()
    
    print("🌐 Starting web server...")
    print("⚡ YOLOv8 loading in background...")
    print("\n🎯 Once server is ready:")
    print("   1. Open: http://localhost:5000")
    print("   2. Start stream with your video source")
    print("   3. Object detection will auto-enable when ready!")
    print("\n" + "=" * 50)
    
    # Start main server
    try:
        from headless_stream import main
        main(host='0.0.0.0', port=5000, debug=False, no_yolo=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎥 CCTV Live Detection - Quick Start")
    print("🤖 With YOLOv8 Object Detection")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('headless_stream.py'):
        print("❌ Please run this script from the project directory")
        print("   cd /path/to/project-plat-detection-alfi")
        return
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Start server
    success = start_server_fast()
    
    if success:
        print("✅ Server started successfully!")
    else:
        print("❌ Failed to start server")
        print("\n🔧 Troubleshooting:")
        print("   1. Check dependencies: python install_yolo.py")
        print("   2. Test system: python test_integration.py")
        print("   3. Manual start: python headless_stream.py")

if __name__ == "__main__":
    main()