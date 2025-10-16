#!/usr/bin/env python3
"""
Debug script khusus untuk test CCTV image 15122022plat.jpg
"""

import cv2
import logging
import sys
import os

# Add project root to path
sys.path.append('/Users/andra/Documents/DWI/project-plat-detection-pai')

# Import our detectors
from utils.robust_plate_detector import RobustPlateDetector

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_candidates(image, candidates, output_path):
    """Visualize semua kandidat yang ditemukan"""
    result = image.copy()

    for i, candidate in enumerate(candidates):
        x, y, w, h = candidate['bbox']
        method = candidate.get('method', 'unknown')
        score = candidate.get('score', 0)

        # Draw bounding box dengan warna berbeda untuk setiap method
        if method == 'otsu':
            color = (0, 255, 0)  # Green
        elif method == 'adaptive_gaussian':
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 0, 255)  # Red

        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        # Add label
        label = f"{i+1}: {method} (score:{score:.1f})"
        cv2.putText(result, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        print(f"   Candidate {i+1}: {method} at ({x}, {y}, {w}, {h}) score: {score:.1f}")

    cv2.imwrite(output_path, result)
    print(f"💾 Candidates visualization saved: {output_path}")

def test_cctv_detection():
    """Test deteksi dengan gambar CCTV yang challenging"""
    image_path = "contoh/15122022plat.jpg"
    print(f"🔧 Testing CCTV Plate Detection: {image_path}")
    print("=" * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"📸 Image loaded: {image.shape}")

    # Test robust detector dengan settings yang sangat permisif
    detector = RobustPlateDetector(streaming_mode=True)

    # Override settings untuk extreme permissive detection
    detector.min_confidence = 5
    detector.min_text_likelihood = 5
    detector.min_area = 100
    detector.max_area = 30000

    print(f"\n🎯 Ultra Permissive Settings:")
    print(f"   Min confidence: {detector.min_confidence}")
    print(f"   Min text likelihood: {detector.min_text_likelihood}")
    print(f"   Min area: {detector.min_area}")
    print(f"   Max area: {detector.max_area}")

    # Get candidates before filtering untuk debug
    # We'll need to access internal method untuk get raw candidates
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ROI calculation
    height, width = gray.shape
    roi_x = int(width * 0.05)
    roi_y = int(height * 0.25)
    roi_w = int(width * 0.9)
    roi_h = int(height * 0.6)
    roi_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    print(f"\n🔍 ROI: ({roi_x}, {roi_y}, {roi_w}, {roi_h})")

    # Manual candidate detection
    candidates = []

    # Test horizontal detection
    try:
        horizontal_candidates = detector._detect_horizontal_plates(roi_image)
        # Adjust coordinates
        for candidate in horizontal_candidates:
            candidate['bbox'] = (
                candidate['bbox'][0] + roi_x,
                candidate['bbox'][1] + roi_y,
                candidate['bbox'][2],
                candidate['bbox'][3]
            )
            candidates.append(candidate)
        print(f"🔍 Horizontal candidates: {len(horizontal_candidates)}")
    except Exception as e:
        print(f"❌ Horizontal detection error: {e}")

    # Visualize all candidates
    if candidates:
        print(f"\n📊 Found {len(candidates)} candidates:")
        visualize_candidates(image, candidates, "contoh/cctv_candidates_debug.jpg")

        # Try OCR on each candidate manually dengan berbagai setting
        for i, candidate in enumerate(candidates[:3]):  # Test top 3 only
            x, y, w, h = candidate['bbox']
            roi = image[y:y+h, x:x+w]

            print(f"\n🔍 Testing OCR on candidate {i+1}: ({x}, {y}, {w}, {h})")

            # Save ROI untuk manual inspection
            roi_path = f"contoh/candidate_{i+1}_roi.jpg"
            cv2.imwrite(roi_path, roi)
            print(f"   ROI saved: {roi_path}")

            # Try different preprocessing
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing untuk license plates
            enhanced = cv2.bilateralFilter(gray_roi, 11, 17, 17)
            enhanced = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            enhanced_path = f"contoh/candidate_{i+1}_enhanced.jpg"
            cv2.imwrite(enhanced_path, enhanced)
            print(f"   Enhanced saved: {enhanced_path}")
    else:
        print("❌ No candidates found!")

        # Save full image dengan ROI marked
        debug_img = image.copy()
        cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        cv2.putText(debug_img, "ROI", (roi_x, roi_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("contoh/cctv_debug_roi.jpg", debug_img)
        print("💾 Debug ROI saved: contoh/cctv_debug_roi.jpg")

    # Run normal detection untuk comparison
    print(f"\n🔍 Running normal detection...")
    detections = detector.detect_plates(image)

    print(f"\n📊 Final Results:")
    print(f"   Candidates found: {len(candidates)}")
    print(f"   Final detections: {len(detections)}")

    if detections:
        for i, detection in enumerate(detections):
            print(f"   {i+1}. '{detection.text}' ({detection.confidence:.1f}%)")

    return detections

if __name__ == "__main__":
    test_cctv_detection()