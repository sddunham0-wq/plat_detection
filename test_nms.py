#!/usr/bin/env python3
"""
Test Non-Maximum Suppression (NMS) function

Penjelasan SMK: Script ini test apakah NMS bisa filter overlapping boxes dengan benar
"""

import sys
sys.path.append('.')

def calculate_iou(box1, box2):
    """Calculate IOU between 2 boxes"""
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def non_maximum_suppression(bboxes, iou_threshold=0.5):
    """NMS implementation"""
    if not bboxes or len(bboxes) == 0:
        return []

    if len(bboxes) == 1:
        return bboxes

    # Calculate area
    boxes_with_area = []
    for bbox in bboxes:
        x, y, w, h = bbox[:4]
        area = w * h
        boxes_with_area.append({
            'bbox': bbox,
            'area': area
        })

    # Sort by area (largest first)
    boxes_with_area.sort(key=lambda b: b['area'], reverse=True)

    # NMS algorithm
    keep = []

    while boxes_with_area:
        current = boxes_with_area.pop(0)
        keep.append(current['bbox'])

        remaining = []
        for other in boxes_with_area:
            iou = calculate_iou(current['bbox'], other['bbox'])

            if iou < iou_threshold:
                remaining.append(other)
            else:
                print(f"  Suppressed box (IOU={iou:.2f})")

        boxes_with_area = remaining

    print(f"✨ NMS: {len(bboxes)} boxes → {len(keep)} non-overlapping boxes")

    return keep

def test_case_1():
    """
    Test Case 1: Overlapping boxes dari multi-scale detection

    Scenario: Same plat detected at 3 scales (100%, 70%, 50%)
    Expected: Keep only 1 box (largest)
    """
    print("=" * 70)
    print("TEST CASE 1: Multi-Scale Overlapping Boxes")
    print("=" * 70)

    # Simulate 3 detections dari same plate di different scales
    boxes = [
        (100, 50, 200, 60),   # Scale 100% (largest)
        (105, 52, 140, 42),   # Scale 70% (medium, overlaps with #1)
        (108, 54, 100, 30),   # Scale 50% (smallest, overlaps with both)
    ]

    print(f"Input: {len(boxes)} overlapping boxes")
    for i, box in enumerate(boxes):
        x, y, w, h = box
        print(f"  Box #{i+1}: ({x},{y},{w},{h}) - area={w*h}")

    # Calculate IOUs
    print("\nOverlap analysis:")
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            iou = calculate_iou(boxes[i], boxes[j])
            print(f"  Box #{i+1} vs Box #{j+1}: IOU = {iou:.2f}")

    # Apply NMS
    print("\nApplying NMS (threshold=0.5):")
    result = non_maximum_suppression(boxes, iou_threshold=0.5)

    print(f"\nResult: {len(result)} box(es)")
    for i, box in enumerate(result):
        x, y, w, h = box
        print(f"  Box #{i+1}: ({x},{y},{w},{h}) - area={w*h}")

    # Verify
    success = len(result) == 1
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Expected 1 box, got {len(result)}")
    print()

    return success

def test_case_2():
    """
    Test Case 2: Non-overlapping boxes

    Scenario: 2 different plates far apart
    Expected: Keep both boxes
    """
    print("=" * 70)
    print("TEST CASE 2: Non-Overlapping Boxes (Different Plates)")
    print("=" * 70)

    boxes = [
        (100, 50, 200, 60),   # Plat #1 (left side)
        (500, 300, 200, 60),  # Plat #2 (right side, tidak overlap)
    ]

    print(f"Input: {len(boxes)} non-overlapping boxes")
    for i, box in enumerate(boxes):
        x, y, w, h = box
        print(f"  Box #{i+1}: ({x},{y},{w},{h}) - area={w*h}")

    # Calculate IOU
    iou = calculate_iou(boxes[0], boxes[1])
    print(f"\nOverlap: IOU = {iou:.2f} (no overlap)")

    # Apply NMS
    print("\nApplying NMS (threshold=0.5):")
    result = non_maximum_suppression(boxes, iou_threshold=0.5)

    print(f"\nResult: {len(result)} box(es)")
    for i, box in enumerate(result):
        x, y, w, h = box
        print(f"  Box #{i+1}: ({x},{y},{w},{h}) - area={w*h}")

    # Verify
    success = len(result) == 2
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Expected 2 boxes, got {len(result)}")
    print()

    return success

def test_case_3():
    """
    Test Case 3: Partial overlap (low IOU)

    Scenario: 2 plates close together with slight overlap
    Expected: Keep both (IOU < threshold)
    """
    print("=" * 70)
    print("TEST CASE 3: Partial Overlap (Low IOU)")
    print("=" * 70)

    boxes = [
        (100, 50, 200, 60),   # Plat #1
        (250, 50, 200, 60),   # Plat #2 (slight overlap)
    ]

    print(f"Input: {len(boxes)} partially overlapping boxes")
    for i, box in enumerate(boxes):
        x, y, w, h = box
        print(f"  Box #{i+1}: ({x},{y},{w},{h}) - area={w*h}")

    # Calculate IOU
    iou = calculate_iou(boxes[0], boxes[1])
    print(f"\nOverlap: IOU = {iou:.2f}")

    # Apply NMS
    print("\nApplying NMS (threshold=0.5):")
    result = non_maximum_suppression(boxes, iou_threshold=0.5)

    print(f"\nResult: {len(result)} box(es)")
    for i, box in enumerate(result):
        x, y, w, h = box
        print(f"  Box #{i+1}: ({x},{y},{w},{h}) - area={w*h}")

    # Verify
    success = (len(result) == 2 and iou < 0.5) or (len(result) == 1 and iou >= 0.5)
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}: Correct filtering based on IOU={iou:.2f}")
    print()

    return success

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "NON-MAXIMUM SUPPRESSION TEST SUITE" + " " * 19 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Multi-Scale Overlap", test_case_1()))
    results.append(("Non-Overlapping", test_case_2()))
    results.append(("Partial Overlap", test_case_3()))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_pass = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_pass = False

    print("=" * 70)
    print()

    if all_pass:
        print("🎉 ALL TESTS PASSED!")
        print("✅ NMS implementation is working correctly!")
        print()
        print("📝 What this means:")
        print("   • Overlapping boxes will be filtered (keep only 1)")
        print("   • Non-overlapping boxes will be kept (keep all)")
        print("   • Multi-scale detection akan lebih clean!")
        print()
    else:
        print("❌ SOME TESTS FAILED")
        print("⚠️  NMS implementation needs fixes!")
        print()

    return 0 if all_pass else 1

if __name__ == '__main__':
    exit(main())
