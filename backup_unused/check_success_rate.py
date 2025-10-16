#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHECK SUCCESS RATE - Hitung Success Rate OCR

Penjelasan SMK: Script ini untuk hitung berapa persen OCR berhasil
Lihat folder gambarplat/ dan hitung SUCCESS vs total crop
"""

import os
import glob
from datetime import datetime

def check_success_rate():
    """Hitung success rate dari folder gambarplat/"""

    print("="*80)
    print("📊 SUCCESS RATE CALCULATOR")
    print("="*80)

    folder = "gambarplat"

    if not os.path.exists(folder):
        print(f"❌ Folder {folder}/ tidak ditemukan!")
        return

    # Hitung file
    crop_files = glob.glob(f"{folder}/crop_*.jpg")
    success_files = glob.glob(f"{folder}/SUCCESS_*.jpg")

    print(f"\n📁 Folder: {folder}/")
    print("-" * 80)
    print(f"Total crop images : {len(crop_files)}")
    print(f"Successful OCR    : {len(success_files)}")

    if len(crop_files) > 0:
        success_rate = (len(success_files) / len(crop_files)) * 100
        print(f"\n✅ SUCCESS RATE: {success_rate:.1f}%")

        if success_rate >= 90:
            print("🎉 EXCELLENT! OCR working very well!")
        elif success_rate >= 70:
            print("✅ GOOD! Most plates are detected successfully")
        elif success_rate >= 50:
            print("⚠️ OKAY - Need improvement")
        else:
            print("❌ POOR - Need major fixes")
    else:
        print("\n⚠️ No images found")
        print("💡 Run the app first: python3 app.py")

    # List SUCCESS files
    if success_files:
        print("\n" + "="*80)
        print("✅ SUCCESSFUL DETECTIONS:")
        print("-" * 80)

        for i, filepath in enumerate(sorted(success_files), 1):
            filename = os.path.basename(filepath)
            # Extract plate text from filename: SUCCESS_B1234ABC_timestamp.jpg
            parts = filename.replace('SUCCESS_', '').replace('.jpg', '').split('_')
            if len(parts) >= 2:
                plate_text = parts[0]
                timestamp = parts[1]
                size = os.path.getsize(filepath) / 1024  # KB

                print(f"{i:2d}. {plate_text:<15} (Time: {timestamp}, Size: {size:.1f}KB)")

    # List FAILED (crop without SUCCESS)
    failed_crops = []
    for crop_file in crop_files:
        crop_name = os.path.basename(crop_file)
        timestamp = crop_name.replace('crop_', '').replace('.jpg', '')

        # Check if ada SUCCESS dengan timestamp sama
        has_success = any(timestamp in s for s in success_files)

        if not has_success:
            failed_crops.append(crop_file)

    if failed_crops:
        print("\n" + "="*80)
        print("❌ FAILED DETECTIONS:")
        print("-" * 80)

        for i, filepath in enumerate(sorted(failed_crops), 1):
            filename = os.path.basename(filepath)
            size = os.path.getsize(filepath) / 1024  # KB
            # Get image dimensions
            import cv2
            img = cv2.imread(filepath)
            if img is not None:
                h, w = img.shape[:2]
                print(f"{i:2d}. {filename:<30} (Size: {w}x{h}, {size:.1f}KB)")

                # Analyze why failed
                if w < 100:
                    print(f"     ⚠️ Reason: Too small (width {w}px < 100px)")
                elif w > 500:
                    print(f"     ⚠️ Reason: Too large (width {w}px > 500px)")

    print("\n" + "="*80)

if __name__ == '__main__':
    check_success_rate()
