#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST OCR - Test OCR processor dengan gambar plat

Penjelasan SMK: Script ini untuk test OCR pakai gambar plat
yang sudah disimpan di folder gambarplat/
"""

import cv2
import os
import glob
from utils.ocr_processor import OCRProcessor

def test_ocr_on_images():
    """Test OCR pada semua gambar di folder gambarplat/"""

    print("="*80)
    print("🔍 TEST OCR PROCESSOR")
    print("="*80)

    # Initialize OCR
    ocr = OCRProcessor()
    print(f"✅ OCR Processor initialized\n")

    # Cari semua gambar di folder gambarplat/
    image_folder = "gambarplat"

    if not os.path.exists(image_folder):
        print(f"❌ Folder {image_folder}/ tidak ditemukan!")
        print(f"💡 Jalankan aplikasi dulu untuk generate gambar plat")
        return

    # Cari file gambar
    image_files = glob.glob(f"{image_folder}/*.jpg") + glob.glob(f"{image_folder}/*.png")

    if not image_files:
        print(f"❌ Tidak ada gambar di folder {image_folder}/")
        print(f"💡 Jalankan aplikasi dan biarkan deteksi plat untuk save gambar")
        return

    print(f"📁 Ditemukan {len(image_files)} gambar\n")
    print("="*80)

    # Test setiap gambar
    success_count = 0
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Testing: {os.path.basename(img_path)}")
        print("-" * 80)

        # Baca gambar
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Gagal baca gambar")
            continue

        height, width = img.shape[:2]
        print(f"📐 Size: {width}x{height}")

        # Test OCR dengan confidence
        plate_text, confidence = ocr.read_plate_with_confidence(img)

        if plate_text:
            print(f"✅ OCR BERHASIL!")
            print(f"   Text: {plate_text}")
            print(f"   Confidence: {confidence:.2f}")
            success_count += 1
        else:
            print(f"❌ OCR GAGAL")
            print(f"   Confidence: {confidence:.2f}")

            # Coba simple OCR (tanpa confidence)
            simple_text = ocr.read_plate_text(img)
            if simple_text:
                print(f"⚠️ Simple OCR: {simple_text}")

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Total gambar: {len(image_files)}")
    print(f"Berhasil: {success_count}")
    print(f"Gagal: {len(image_files) - success_count}")
    print(f"Success rate: {success_count/len(image_files)*100:.1f}%")
    print("="*80)

if __name__ == '__main__':
    test_ocr_on_images()
