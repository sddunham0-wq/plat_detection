#!/usr/bin/env python3
"""
Cleanup Script - Hapus file tidak berguna
Analisis dan hapus file duplicate/obsolete
"""

import os
import glob
from pathlib import Path

print("\n" + "="*70)
print("🧹 PROJECT CLEANUP SCRIPT")
print("="*70 + "\n")

# ==========================================
# 1. IDENTIFY FILES TO DELETE
# ==========================================

files_to_delete = []

# === DUPLICATE/OBSOLETE MD FILES ===
obsolete_docs = [
    # Old documentation (sudah ada yang lebih baru)
    'DATABASE_ACCESS_GUIDE.md',      # Obsolete - ada di README
    'FILTER_SETTINGS.md',            # Obsolete - settings di code
    'FORMAT_UPDATE.md',              # Obsolete - old update log
    'IMPROVEMENTS.md',               # Obsolete - old improvement log
    'OPTIMIZATION_CHANGELOG.md',     # Obsolete - old changelog
    'PANDUAN_SIMPLE.md',            # Obsolete - ada README & STATUS docs
    'QUICK_START_REMOTE.md',        # Obsolete - remote setup not used
    'SMOOTH_STREAMING_UPDATE.md',   # Obsolete - old update log
    'STABILITY_UPDATE.md',           # Obsolete - old update log
]

# === DUPLICATE TEST SCRIPTS ===
obsolete_tests = [
    'analyze_image2.py',             # Debug script - not needed anymore
    'quick_test_yolo.py',           # Duplicate - ada test_yolo_detection.py
    'test_contour_relaxed.py',      # Debug script - not needed anymore
]

# === OLD TEST IMAGES/RESULTS ===
obsolete_images = [
    'image2_contour_relaxed.jpg',    # Test result - not needed
    'image2_rectangle_rectangle.jpg', # Test result - not needed
    'image2_yolo.png',               # Test result - not needed
]

# === KEEP IMPORTANT FILES ===
keep_files = [
    'README.md',                     # Main documentation ✅
    'SETUP_MACOS.md',               # Setup guide ✅
    'GALLERY_FEATURE.md',           # Feature documentation ✅
    'PERBAIKAN_SELESAI.md',         # Latest fix summary ✅
    'STATUS_YOLO.md',               # YOLO status ✅
    'YOLO_SETUP.md',                # YOLO setup guide ✅
    'RECTANGLE_DETECTOR.md',        # Rectangle detector doc ✅

    'app.py',                        # Main app ✅
    'app_simple.py',                # Simple app ✅
    'config.py',                    # Configuration ✅
    'config_yolo.py',               # YOLO config ✅
    'download_yolo_model.py',       # Model downloader ✅
    'test_f1818hg.py',              # Database test ✅
    'test_yolo_detection.py',       # YOLO test ✅
    'test_rectangle_detector.py',   # Rectangle test ✅

    'image1.png',                   # Test images ✅
    'image2.png',                   # Test images ✅
    'image3.png',                   # Test images ✅
]

# ==========================================
# 2. CHECK EACH FILE
# ==========================================

print("📋 Scanning for obsolete files...\n")

# Scan obsolete docs
for doc in obsolete_docs:
    if os.path.exists(doc):
        size = os.path.getsize(doc) / 1024  # KB
        files_to_delete.append({
            'path': doc,
            'type': 'Documentation (obsolete)',
            'size': f"{size:.1f}KB"
        })

# Scan obsolete tests
for test in obsolete_tests:
    if os.path.exists(test):
        size = os.path.getsize(test) / 1024
        files_to_delete.append({
            'path': test,
            'type': 'Test Script (duplicate/debug)',
            'size': f"{size:.1f}KB"
        })

# Scan obsolete images
for img in obsolete_images:
    if os.path.exists(img):
        size = os.path.getsize(img) / 1024
        files_to_delete.append({
            'path': img,
            'type': 'Test Result Image',
            'size': f"{size:.1f}KB"
        })

# === SCAN __pycache__ DIRECTORIES ===
pycache_dirs = []
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        pycache_path = os.path.join(root, '__pycache__')
        # Count files
        pycache_files = glob.glob(f"{pycache_path}/*.pyc")
        if pycache_files:
            total_size = sum(os.path.getsize(f) for f in pycache_files) / 1024
            pycache_dirs.append({
                'path': pycache_path,
                'type': 'Python Cache',
                'size': f"{total_size:.1f}KB",
                'files': len(pycache_files)
            })

# ==========================================
# 3. DISPLAY RESULTS
# ==========================================

print("="*70)
print("📊 CLEANUP SUMMARY")
print("="*70 + "\n")

if files_to_delete:
    print(f"Found {len(files_to_delete)} obsolete files:\n")

    total_size = 0
    for i, item in enumerate(files_to_delete, 1):
        size_kb = float(item['size'].replace('KB', ''))
        total_size += size_kb
        print(f"{i}. {item['path']}")
        print(f"   Type: {item['type']}")
        print(f"   Size: {item['size']}")
        print()

    print(f"Total size to free: {total_size:.1f}KB\n")

if pycache_dirs:
    print(f"Found {len(pycache_dirs)} __pycache__ directories:\n")

    total_cache = 0
    for i, item in enumerate(pycache_dirs, 1):
        size_kb = float(item['size'].replace('KB', ''))
        total_cache += size_kb
        print(f"{i}. {item['path']}")
        print(f"   Files: {item['files']}")
        print(f"   Size: {item['size']}")
        print()

    print(f"Total cache size: {total_cache:.1f}KB\n")

if not files_to_delete and not pycache_dirs:
    print("✅ No obsolete files found - project is clean!\n")
    exit(0)

# ==========================================
# 4. CONFIRM DELETION
# ==========================================

print("="*70)
print("⚠️  DELETE CONFIRMATION")
print("="*70 + "\n")

print("Files that will be DELETED:")
for item in files_to_delete:
    print(f"  ❌ {item['path']}")

if pycache_dirs:
    print(f"\n__pycache__ directories:")
    for item in pycache_dirs:
        print(f"  ❌ {item['path']}")

print(f"\n⚠️  Total: {len(files_to_delete)} files + {len(pycache_dirs)} cache dirs")
print()

confirm = input("Proceed with deletion? (yes/no): ").strip().lower()

if confirm != 'yes':
    print("\n❌ Cleanup cancelled by user\n")
    exit(0)

# ==========================================
# 5. DELETE FILES
# ==========================================

print("\n" + "="*70)
print("🗑️  DELETING FILES...")
print("="*70 + "\n")

deleted_count = 0
failed_count = 0

# Delete obsolete files
for item in files_to_delete:
    try:
        os.remove(item['path'])
        print(f"✅ Deleted: {item['path']}")
        deleted_count += 1
    except Exception as e:
        print(f"❌ Failed: {item['path']} - {e}")
        failed_count += 1

# Delete __pycache__ directories
import shutil
for item in pycache_dirs:
    try:
        shutil.rmtree(item['path'])
        print(f"✅ Deleted: {item['path']}")
        deleted_count += 1
    except Exception as e:
        print(f"❌ Failed: {item['path']} - {e}")
        failed_count += 1

# ==========================================
# 6. FINAL SUMMARY
# ==========================================

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE")
print("="*70 + "\n")

print(f"Deleted: {deleted_count} items")
print(f"Failed: {failed_count} items")
print()

if deleted_count > 0:
    print("✅ Project cleaned successfully!")
    print()
    print("Remaining important files:")
    print("  📄 Documentation: README.md, SETUP_MACOS.md, STATUS_YOLO.md")
    print("  🐍 Main Apps: app.py, app_simple.py")
    print("  🧪 Tests: test_yolo_detection.py, test_rectangle_detector.py")
    print("  📷 Images: image1.png, image2.png, image3.png")
    print()

print("="*70 + "\n")
