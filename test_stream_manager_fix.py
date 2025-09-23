#!/usr/bin/env python3
"""
Test Stream Manager Fix
Test apakah logger error sudah diperbaiki
"""
import sys
import os

def test_stream_manager_initialization():
    """Test stream manager initialization"""
    print("🧪 Testing Stream Manager Initialization")
    print("=" * 50)

    try:
        # Import stream manager
        from stream_manager import HeadlessStreamManager
        from database import PlateDatabase

        print("✅ Imports successful")

        # Test initialization
        database = PlateDatabase()

        # Test dengan webcam source (won't actually connect)
        print("🔧 Testing HeadlessStreamManager initialization...")

        stream_manager = HeadlessStreamManager(
            source=0,  # Webcam
            database=database,
            enable_yolo=True,
            enable_tracking=False
        )

        print("✅ HeadlessStreamManager initialized successfully")
        print(f"   Enhanced mode: {getattr(stream_manager, 'enhanced_mode', 'unknown')}")
        print(f"   Logger present: {hasattr(stream_manager, 'logger')}")

        # Test logger functionality
        if hasattr(stream_manager, 'logger'):
            stream_manager.logger.info("Test log message")
            print("✅ Logger working correctly")

        # Test start method (akan fail karena no camera, tapi logger harus work)
        print("🎬 Testing start method (expect camera failure, but no logger error)...")

        try:
            result = stream_manager.start()
            print(f"   Start result: {result}")
        except Exception as e:
            if 'logger' in str(e):
                print(f"❌ Logger error still present: {e}")
                return False
            else:
                print(f"✅ Expected camera error (not logger): {e}")

        print("✅ Stream manager initialization test PASSED")
        return True

    except Exception as e:
        print(f"❌ Stream manager test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_detector_loading():
    """Test enhanced detector loading specifically"""
    print(f"\n🔍 Testing Enhanced Detector Loading")
    print("-" * 40)

    try:
        from enhanced_plate_detector import EnhancedPlateDetector

        # Test dengan streaming config
        detector = EnhancedPlateDetector('enhanced_detection_streaming_config.ini')
        print("✅ Enhanced detector loaded with streaming config")

        # Test basic properties
        print(f"   Confidence threshold: {getattr(detector, 'enhanced_conf_threshold', 'unknown')}")
        print(f"   Use secondary: {getattr(detector, 'use_secondary', 'unknown')}")

        return True

    except Exception as e:
        print(f"❌ Enhanced detector loading failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Stream Manager Fix Verification")
    print("Testing logger initialization fix")
    print()

    # Test 1: Stream manager initialization
    test1_passed = test_stream_manager_initialization()

    # Test 2: Enhanced detector loading
    test2_passed = test_enhanced_detector_loading()

    print(f"\n📊 Test Results:")
    print(f"   Stream Manager Init: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Enhanced Detector:   {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n🎉 All tests PASSED! Logger error fixed!")
        print(f"💡 You can now run: python3 headless_stream.py")
    else:
        print(f"\n❌ Some tests failed. Check errors above.")

if __name__ == "__main__":
    main()