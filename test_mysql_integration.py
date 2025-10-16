"""
Test MySQL Integration - Comprehensive Testing
Test access control flow dengan simulated detections
"""

import logging
import time
from typing import Optional
from mysql_database import MySQLPlateDatabase
from access_controller import AccessController
from utils.plate_detector import PlateDetection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dummy_detection(plate_text: str, confidence: float = 85.0) -> PlateDetection:
    """
    Create dummy PlateDetection untuk testing

    Args:
        plate_text: Text plat nomor
        confidence: Confidence score

    Returns:
        PlateDetection: Dummy detection object
    """
    import numpy as np

    dummy_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
    return PlateDetection(
        text=plate_text,
        confidence=confidence,
        bbox=(100, 100, 200, 50),
        processed_image=dummy_image,
        timestamp=time.time()
    )

def test_access_granted():
    """Test access granted flow - kendaraan terdaftar"""
    print("\n" + "="*60)
    print("Test 1: ACCESS GRANTED (Registered Vehicle)")
    print("="*60)

    try:
        controller = AccessController()

        # Simulate detection untuk kendaraan terdaftar (F1818HG dari SQL)
        detection = create_dummy_detection("F1818HG", 88.5)

        print(f"\n📸 Simulating detection: {detection.text} ({detection.confidence:.1f}%)")

        # Process detection
        result = controller.process_detection(detection, image_path="test_image.jpg")

        print(f"\n📊 Result:")
        print(f"   Access: {result['access'].upper()}")
        print(f"   Plate: {result['plate_number']}")
        print(f"   Confidence: {result['confidence']:.1f}%")

        if result['access'] == 'Authorized':
            vehicle = result['vehicle']
            print(f"   ✅ Vehicle Info:")
            print(f"      - Owner: {vehicle['owner_name']}")
            print(f"      - Type: {vehicle['vehicle_type']}")
            print(f"      - Status: {vehicle['status']}")
            print(f"      - Message: {result['message']}")
            print(f"   ✅ Access Log ID: {result.get('access_log_id', 'N/A')}")
            return True
        else:
            print(f"   ❌ Unexpected result: {result.get('reason', 'Unknown')}")
            return False

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def test_access_denied():
    """Test access denied flow - kendaraan tidak terdaftar"""
    print("\n" + "="*60)
    print("Test 2: ACCESS DENIED (Unregistered Vehicle)")
    print("="*60)

    try:
        controller = AccessController()

        # Simulate detection untuk kendaraan TIDAK terdaftar
        detection = create_dummy_detection("TEST9999", 92.3)

        print(f"\n📸 Simulating detection: {detection.text} ({detection.confidence:.1f}%)")

        # Process detection
        result = controller.process_detection(detection, image_path="test_denied.jpg")

        print(f"\n📊 Result:")
        print(f"   Access: {result['access'].upper()}")
        print(f"   Plate: {result['plate_number']}")
        print(f"   Confidence: {result['confidence']:.1f}%")

        if result['access'] == 'Denied':
            print(f"   ❌ Reason: {result['reason']}")
            print(f"   📝 Message: {result['message']}")
            print(f"   📝 Access Log ID: {result.get('access_log_id', 'N/A')}")
            return True
        else:
            print(f"   ❌ Unexpected result: Expected 'Denied', got '{result['access']}'")
            return False

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def test_multiple_detections():
    """Test multiple detections dalam sequence"""
    print("\n" + "="*60)
    print("Test 3: Multiple Detections Sequence")
    print("="*60)

    try:
        controller = AccessController()

        test_plates = [
            ("F1818HG", 88.0, "registered"),    # Registered
            ("TEST1234", 90.0, "unregistered"), # Not registered
            ("F1818HG", 89.0, "registered"),    # Same registered (duplicate)
        ]

        results = []
        for plate, conf, expected_type in test_plates:
            print(f"\n📸 Processing: {plate} ({conf:.1f}%) - Expect: {expected_type}")
            detection = create_dummy_detection(plate, conf)
            result = controller.process_detection(detection)
            results.append(result)

            print(f"   → Result: {result['access'].upper()}")

        # Show statistics
        stats = controller.get_statistics()
        print(f"\n📊 Controller Statistics:")
        print(f"   Total Processed: {stats['controller']['total_processed']}")
        print(f"   Access Granted: {stats['controller']['access_granted']}")
        print(f"   Access Denied: {stats['controller']['access_denied']}")
        print(f"   Grant Rate: {stats['controller']['grant_rate']:.1f}%")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def test_vehicle_registration():
    """Test registrasi kendaraan baru"""
    print("\n" + "="*60)
    print("Test 4: Vehicle Registration")
    print("="*60)

    try:
        controller = AccessController()

        # Generate unique plate number with timestamp
        test_plate = f"TEST{int(time.time()) % 10000}"

        print(f"\n📝 Registering new vehicle: {test_plate}")

        # Register vehicle
        result = controller.register_vehicle(
            plate_number=test_plate,
            owner_name="Test User",
            vehicle_type="testing",
            contact_info="08123456789"
        )

        print(f"\n📊 Registration Result:")
        print(f"   Success: {result['success']}")
        print(f"   Message: {result['message']}")

        if result['success']:
            print(f"   Vehicle ID: {result['vehicle_id']}")

            # Verify registration
            vehicle = controller.check_vehicle_status(test_plate)
            if vehicle:
                print(f"\n✅ Verification successful:")
                print(f"   - Plate: {vehicle['plate_number']}")
                print(f"   - Owner: {vehicle['owner_name']}")
                print(f"   - Type: {vehicle['vehicle_type']}")
                return True
            else:
                print(f"\n❌ Verification failed: Vehicle not found after registration")
                return False
        else:
            print(f"\n⚠️ Registration failed (may already exist)")
            return False

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def test_access_history():
    """Test query access history"""
    print("\n" + "="*60)
    print("Test 5: Access History Query")
    print("="*60)

    try:
        controller = AccessController()

        # Get recent access history
        print(f"\n📋 Getting recent access history (limit 10)...")
        history = controller.get_access_history(limit=10)

        print(f"\n📊 Found {len(history)} access records:")

        for i, record in enumerate(history[:5], 1):  # Show first 5
            print(f"\n   {i}. Plate: {record['plate_number']}")
            print(f"      Status: {record['status']}")
            print(f"      Time: {record['acces_time']}")
            if record.get('owner_name'):
                print(f"      Owner: {record['owner_name']}")

        if len(history) > 5:
            print(f"\n   ... and {len(history) - 5} more records")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def test_database_statistics():
    """Test database statistics retrieval"""
    print("\n" + "="*60)
    print("Test 6: Database Statistics")
    print("="*60)

    try:
        db = MySQLPlateDatabase()
        stats = db.get_statistics()

        print(f"\n📊 Database Statistics:")
        print(f"   Total Vehicles: {stats.get('total_vehicles', 0)}")
        print(f"   Total Access Logs: {stats.get('total_access_logs', 0)}")
        print(f"   Access Today: {stats.get('access_today', 0)}")

        if stats.get('by_status'):
            print(f"\n   Vehicles by Status:")
            for status_info in stats['by_status']:
                print(f"      - {status_info['status']}: {status_info['count']}")

        return True

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "🚀 "*20)
    print("MySQL Integration Tests - Comprehensive Suite")
    print("🚀 "*20)

    tests = [
        ("Access Granted (Registered Vehicle)", test_access_granted),
        ("Access Denied (Unregistered Vehicle)", test_access_denied),
        ("Multiple Detections Sequence", test_multiple_detections),
        ("Vehicle Registration", test_vehicle_registration),
        ("Access History Query", test_access_history),
        ("Database Statistics", test_database_statistics),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")

    print(f"\n📊 Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n✅ All tests passed! MySQL integration is working perfectly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")

    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        logger.exception("Unexpected error in tests")
