#!/usr/bin/env python3

import cv2
import numpy as np
import sys

def debug_contour_detection(image_path):
    """Debug contour detection to see what's being found"""

    print(f"🔍 Debugging contour detection for: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return

    print(f"📐 Image size: {image.shape[1]}x{image.shape[0]}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

    # Find edges using Canny
    edged = cv2.Canny(bilateral, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"🔢 Found {len(contours)} total contours")

    # Filter contours by area and aspect ratio
    candidates = []
    debug_image = image.copy()

    for i, contour in enumerate(contours):
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Skip very small contours
        if area < 100:
            continue

        aspect_ratio = w / h if h > 0 else 0

        # Indonesian plate typical ratios: 2.5-4.5
        if 1.5 <= aspect_ratio <= 6.0 and area >= 100:
            candidates.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio
            })

            # Draw all candidates in different colors
            color = (0, 255, 0) if 2.0 <= aspect_ratio <= 5.0 else (0, 255, 255)
            cv2.rectangle(debug_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_image, f"#{len(candidates)} AR:{aspect_ratio:.1f}",
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    print(f"✅ Found {len(candidates)} potential candidates")

    # Sort by area (largest first)
    candidates.sort(key=lambda x: x['area'], reverse=True)

    # Show top candidates
    for i, candidate in enumerate(candidates[:10]):
        x, y, w, h = candidate['bbox']
        print(f"  Candidate {i+1}: bbox=({x},{y},{w},{h}) area={candidate['area']} AR={candidate['aspect_ratio']:.2f}")

    # Save debug image
    cv2.imwrite('contoh/debug_contours.jpg', debug_image)
    cv2.imwrite('contoh/debug_edges.jpg', edged)

    print(f"💾 Debug images saved:")
    print(f"  - contoh/debug_contours.jpg (candidates marked)")
    print(f"  - contoh/debug_edges.jpg (edge detection)")

    return candidates

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 debug_contour_detection.py <image_path>")
        sys.exit(1)

    debug_contour_detection(sys.argv[1])