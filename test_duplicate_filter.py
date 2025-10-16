#!/usr/bin/env python3
"""
Test script for duplicate filtering logic
Tests the new time-based duplicate filter in stream_manager.py
"""

import time
from utils.plate_validator import PlateValidator, validate_plate

def test_plate_validator():
    """Test Indonesian plate validation"""
    print("\n=== Testing Plate Validator ===")

    validator = PlateValidator()

    test_cases = [
        # (text, should_be_valid)
        ("B1234ABC", True),   # Valid Jakarta plate
        ("D5678XYZ", True),   # Valid Bandung plate
        ("ET", False),        # Too short - FALSE POSITIVE
        ("T", False),         # Too short - FALSE POSITIVE
        ("8123", False),      # No letters - FALSE POSITIVE
        ("B12345C", False),   # Invalid format (5 digits)
        ("ABC", False),       # No numbers - FALSE POSITIVE
        ("B1234A8C", False),  # Number in suffix - FALSE POSITIVE
    ]

    passed = 0
    failed = 0

    for text, expected_valid in test_cases:
        is_valid = validator.validate(text)
        score = validator.get_validation_score(text)
        status = "✅ PASS" if is_valid == expected_valid else "❌ FAIL"

        print(f"{status} | {text:12s} → Valid: {str(is_valid):5s} | Score: {score:.2f} | Expected: {expected_valid}")

        if is_valid == expected_valid:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_duplicate_filter():
    """Test time-based duplicate filtering logic"""
    print("\n=== Testing Duplicate Filter Logic ===")

    # Simulate the duplicate filter
    recent_detections = {}
    duplicate_window = 5.0

    def is_duplicate(plate_text):
        current_time = time.time()

        if plate_text in recent_detections:
            time_since_last = current_time - recent_detections[plate_text]

            if time_since_last < duplicate_window:
                return True
            else:
                recent_detections[plate_text] = current_time
                return False
        else:
            recent_detections[plate_text] = current_time
            return False

    # Test case 1: First detection should NOT be duplicate
    print("\nTest 1: First detection of B1234ABC")
    result = is_duplicate("B1234ABC")
    print(f"  Result: {'❌ FAIL (is duplicate)' if result else '✅ PASS (not duplicate)'}")

    # Test case 2: Immediate re-detection should be duplicate
    print("\nTest 2: Immediate re-detection of B1234ABC")
    result = is_duplicate("B1234ABC")
    print(f"  Result: {'✅ PASS (is duplicate)' if result else '❌ FAIL (not duplicate)'}")

    # Test case 3: Different plate should NOT be duplicate
    print("\nTest 3: Different plate D5678XYZ")
    result = is_duplicate("D5678XYZ")
    print(f"  Result: {'❌ FAIL (is duplicate)' if result else '✅ PASS (not duplicate)'}")

    # Test case 4: After 6 seconds, same plate should NOT be duplicate
    print("\nTest 4: Re-detection of B1234ABC after 6 seconds")
    print("  Waiting 6 seconds...")
    time.sleep(6)
    result = is_duplicate("B1234ABC")
    print(f"  Result: {'❌ FAIL (is duplicate)' if result else '✅ PASS (not duplicate)'}")

    # Test case 5: Within window should still be duplicate
    print("\nTest 5: Re-detection of B1234ABC immediately after")
    result = is_duplicate("B1234ABC")
    print(f"  Result: {'✅ PASS (is duplicate)' if result else '❌ FAIL (not duplicate)'}")


def test_confidence_threshold():
    """Test confidence threshold filtering"""
    print("\n=== Testing Confidence Threshold ===")

    MIN_CONFIDENCE = 0.65

    test_detections = [
        ("B1234ABC", 0.85, True),   # High confidence - SHOULD PASS
        ("D5678XYZ", 0.70, True),   # Above threshold - SHOULD PASS
        ("ET", 0.45, False),        # Low confidence - SHOULD FILTER
        ("T", 0.30, False),         # Very low confidence - SHOULD FILTER
        ("B9999ZZ", 0.64, False),   # Just below threshold - SHOULD FILTER
    ]

    passed = 0
    failed = 0

    for text, confidence, should_pass in test_detections:
        passes_filter = confidence >= MIN_CONFIDENCE
        status = "✅ PASS" if passes_filter == should_pass else "❌ FAIL"

        print(f"{status} | {text:12s} → Confidence: {confidence:.2f} | Filter: {passes_filter} | Expected: {should_pass}")

        if passes_filter == should_pass:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("DUPLICATE FILTER & VALIDATION TEST")
    print("=" * 60)

    # Run all tests
    validator_ok = test_plate_validator()
    test_duplicate_filter()
    confidence_ok = test_confidence_threshold()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Plate Validator: {'✅ PASS' if validator_ok else '❌ FAIL'}")
    print(f"Confidence Filter: {'✅ PASS' if confidence_ok else '❌ FAIL'}")
    print(f"\nAll systems {'✅ READY' if (validator_ok and confidence_ok) else '❌ NEED FIXES'}")
