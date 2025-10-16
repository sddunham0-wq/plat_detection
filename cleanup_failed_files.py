#!/usr/bin/env python3
"""
Cleanup Failed Detection Files
Script untuk membersihkan file-file deteksi yang gagal dan hemat storage
"""

import os
import shutil
import glob
from typing import List
import logging

class DetectionFileCleanup:
    """Manager untuk cleanup file deteksi yang gagal"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleaned_files = []
        self.total_size_freed = 0

    def cleanup_cache_files(self) -> int:
        """Clean Python cache files"""
        cache_patterns = [
            '__pycache__',
            '*.pyc',
            '*.pyo'
        ]

        freed_size = 0
        for pattern in cache_patterns:
            if pattern == '__pycache__':
                # Remove __pycache__ directories
                for root, dirs, files in os.walk('.'):
                    if '__pycache__' in dirs:
                        cache_dir = os.path.join(root, '__pycache__')
                        try:
                            dir_size = self._get_dir_size(cache_dir)
                            shutil.rmtree(cache_dir)
                            freed_size += dir_size
                            self.cleaned_files.append(cache_dir)
                            self.logger.info(f"🗑️ Removed cache: {cache_dir}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove {cache_dir}: {e}")
            else:
                # Remove .pyc files
                for file_path in glob.glob(f"**/{pattern}", recursive=True):
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_size += file_size
                        self.cleaned_files.append(file_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to remove {file_path}: {e}")

        return freed_size

    def cleanup_temp_files(self) -> int:
        """Clean temporary detection files"""
        temp_patterns = [
            '*temp*.jpg',
            '*tmp*.jpg',
            '*verification_frame*.jpg',
            'hybrid_plate_result.jpg'
        ]

        freed_size = 0
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern, recursive=True):
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    freed_size += file_size
                    self.cleaned_files.append(file_path)
                    self.logger.info(f"🗑️ Removed temp file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove {file_path}: {e}")

        return freed_size

    def cleanup_old_logs(self, days_old: int = 7) -> int:
        """Clean old log files"""
        freed_size = 0

        if not os.path.exists('logs'):
            return 0

        import time
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 3600)

        for log_file in glob.glob('logs/*.log'):
            try:
                file_stat = os.stat(log_file)
                if file_stat.st_mtime < cutoff_time and file_stat.st_size > 1024*1024:  # > 1MB
                    file_size = file_stat.st_size
                    os.remove(log_file)
                    freed_size += file_size
                    self.cleaned_files.append(log_file)
                    self.logger.info(f"🗑️ Removed old log: {log_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {log_file}: {e}")

        return freed_size

    def cleanup_empty_files(self) -> int:
        """Clean empty or corrupt image files"""
        freed_size = 0

        image_patterns = ['*.jpg', '*.png', '*.bmp']
        for pattern in image_patterns:
            for file_path in glob.glob(f"**/{pattern}", recursive=True):
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:  # Empty file
                        os.remove(file_path)
                        self.cleaned_files.append(file_path)
                        self.logger.info(f"🗑️ Removed empty file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to check {file_path}: {e}")

        return freed_size

    def _get_dir_size(self, path: str) -> int:
        """Get total size of directory"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
        except:
            pass
        return total_size

    def run_full_cleanup(self) -> dict:
        """Run complete cleanup process"""
        self.logger.info("🧹 Starting detection file cleanup...")

        # Cleanup different types of files
        cache_freed = self.cleanup_cache_files()
        temp_freed = self.cleanup_temp_files()
        logs_freed = self.cleanup_old_logs()
        empty_freed = self.cleanup_empty_files()

        total_freed = cache_freed + temp_freed + logs_freed + empty_freed
        self.total_size_freed = total_freed

        results = {
            'cache_freed_mb': cache_freed / (1024*1024),
            'temp_freed_mb': temp_freed / (1024*1024),
            'logs_freed_mb': logs_freed / (1024*1024),
            'empty_freed_mb': empty_freed / (1024*1024),
            'total_freed_mb': total_freed / (1024*1024),
            'files_cleaned': len(self.cleaned_files),
            'cleaned_files': self.cleaned_files
        }

        self.logger.info(f"✅ Cleanup completed!")
        self.logger.info(f"   Files cleaned: {results['files_cleaned']}")
        self.logger.info(f"   Total space freed: {results['total_freed_mb']:.2f} MB")

        return results

def main():
    """Main cleanup function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run cleanup
    cleanup = DetectionFileCleanup()
    results = cleanup.run_full_cleanup()

    # Print summary
    print("\n" + "="*50)
    print("🧹 DETECTION FILE CLEANUP SUMMARY")
    print("="*50)
    print(f"📁 Cache files freed: {results['cache_freed_mb']:.2f} MB")
    print(f"🗄️ Temp files freed: {results['temp_freed_mb']:.2f} MB")
    print(f"📋 Log files freed: {results['logs_freed_mb']:.2f} MB")
    print(f"🗑️ Empty files freed: {results['empty_freed_mb']:.2f} MB")
    print("-" * 50)
    print(f"💾 TOTAL SPACE FREED: {results['total_freed_mb']:.2f} MB")
    print(f"📊 FILES CLEANED: {results['files_cleaned']}")
    print("="*50)

    if results['total_freed_mb'] > 0:
        print("✅ Cleanup successful! Storage optimized.")
    else:
        print("ℹ️ No files needed cleanup - system already clean.")

if __name__ == "__main__":
    main()