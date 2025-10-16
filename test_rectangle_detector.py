#!/usr/bin/env python3
"""
Test Rectangle Plate Detector
Optimized untuk plat Indonesia format PERSEGI PANJANG
"""

import cv2
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*70)
print("🧪 TEST: RECTANGLE PLATE DETECTOR")
print("   Optimized untuk plat persegi panjang (landscape)")
print("="*70 + "\n")

# 1. Import detector
print("📦 Importing Rectangle Plate Detector...")
try:
    from utils.plate_detector_rectangle import RectanglePlateDetector
    print("✅ Import successful\n")
except Exception as e:
    print(f"❌ Import failed: {e}\n")
    sys.exit(1)

# 2. Initialize detector
print("🔧 Initializing Rectangle Detector...")
detector = RectanglePlateDetector()
print()

# 3. Get test image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = 'image2.png'  # Default

print(f"📷 Loading test image: {image_path}")
frame = cv2.imread(image_path)

if frame is None:
    print(f"❌ Failed to load image: {image_path}\n")
    print("Usage: python3 test_rectangle_detector.py <image_path>")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"✅ Image loaded: {w}x{h}\n")

# 4. Run detection
print("🔍 Running rectangle detection...")
print("   Focus: LANDSCAPE plates (ratio 2.2-5.0)")
print("   Filters: Size, ratio, brightness, solidity, extent\n")

boxes = detector.detect(frame)

# 5. Show results
print("="*70)
print("📊 DETECTION RESULTS")
print("="*70 + "\n")

print(f"Total plates detected: {len(boxes)}\n")

if boxes:
    print("✅ SUCCESS - Rectangle plates detected!\n")

    for i, (x, y, w, h) in enumerate(boxes, 1):
        ratio = w / h if h > 0 else 0
        area = w * h

        print(f"Plate #{i}:")
        print(f"  📍 Position: ({x}, {y})")
        print(f"  📏 Size: {w}x{h} pixels")
        print(f"  📐 Aspect Ratio: {ratio:.2f}:1", end="")

        # Classification
        if 2.5 <= ratio <= 3.5:
            print(" ← PERFECT rectangle!")
        elif 2.2 <= ratio <= 4.0:
            print(" ← Good rectangle")
        else:
            print(" ← Acceptable")

        print(f"  🔲 Area: {area} pixels²")

        # Extract ROI for analysis
        roi = frame[y:y+h, x:x+w]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            brightness = gray_roi.mean()
            print(f"  💡 Brightness: {brightness:.1f}")

        print()

    # Draw and save result
    result = detector.draw(frame, boxes, "PLAT")

    output_path = image_path.replace('.png', '_rectangle.jpg').replace('.jpg', '_rectangle.jpg')
    if output_path == image_path:
        output_path = 'detected_rectangle.jpg'

    cv2.imwrite(output_path, result)
    print(f"💾 Result saved: {output_path}\n")

    # Comparison table
    print("="*70)
    print("📈 DETECTOR CHARACTERISTICS")
    print("="*70 + "\n")

    print("Rectangle Detector (Current):")
    print(f"  ✅ Ratio Range: 2.2 - 5.0 (landscape)")
    print(f"  ✅ Width Range: 40 - 600px")
    print(f"  ✅ Height Range: 12 - 150px")
    print(f"  ✅ Brightness: >= 50 (shadow support)")
    print(f"  ✅ Quality Scoring: Yes (best candidates first)")
    print(f"  ✅ Shape Filters: Solidity + Extent")
    print()

    print("vs Simple Detector (Previous):")
    print(f"  ⚠️  Ratio Range: 2.3 - 4.2")
    print(f"  ⚠️  Width Range: 50 - 600px")
    print(f"  ⚠️  No quality scoring")
    print(f"  ⚠️  No shape filters")
    print()

    print("="*70)
    print("✅ TEST PASSED")
    print("="*70 + "\n")

    print("Next steps:")
    print("  1. Check result image: " + output_path)
    print("  2. Verify bounding boxes pada plat persegi panjang")
    print("  3. Test dengan gambar plat lain")
    print()

    # Detector info
    info = detector.get_info()
    print("Detector Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

else:
    print("⚠️  No plates detected\n")

    # Debug info
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Image Analysis:")
    print(f"  Mean brightness: {gray.mean():.1f}")
    print(f"  Size: {w}x{h}")
    print(f"  Total pixels: {w*h}")
    print()

    print("Possible reasons:")
    print("  - No rectangle plates in image")
    print("  - Plates too small (< 40px width)")
    print("  - Plates not landscape oriented")
    print("  - Low contrast/brightness")
    print()

    print("="*70)
    print("❌ TEST FAILED")
    print("="*70 + "\n")

    print("Recommendations:")
    print("  1. Try different image with clear rectangle plate")
    print("  2. Use image with good lighting")
    print("  3. Ensure plate is visible and in focus")
    print()

print("="*70 + "\n")
