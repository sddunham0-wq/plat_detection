#!/usr/bin/env python3
"""
Easy Vehicle Detection Restore System
Script untuk restore vehicle detection dengan mudah kapan saja diperlukan
"""

import os
import re
import shutil
from typing import Dict, List

class VehicleDetectionRestore:
    """Manager untuk restore vehicle detection system"""

    def __init__(self):
        self.backup_file = "backup_vehicle_detection.py"
        self.files_to_modify = {
            "config.py": self._restore_config,
            "utils/hybrid_plate_detector.py": self._restore_hybrid_detector,
            "stream_manager.py": self._restore_stream_manager
        }

    def check_backup_exists(self) -> bool:
        """Check if backup file exists"""
        return os.path.exists(self.backup_file)

    def create_enable_flag(self) -> bool:
        """Create enable flag in config.py"""
        try:
            config_path = "config.py"

            # Read current config
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Enable vehicle detection
            content = content.replace(
                "ENABLE_VEHICLE_DETECTION = False    # DISABLED - fokus pure plate detection untuk stabilitas",
                "ENABLE_VEHICLE_DETECTION = True     # ENABLED - hybrid vehicle + plate detection"
            )

            # Enable vehicle-plate association
            content = content.replace(
                "VEHICLE_PLATE_ASSOCIATION = False  # DISABLED - pure plate detection mode",
                "VEHICLE_PLATE_ASSOCIATION = True   # ENABLED - hybrid detection mode"
            )

            # Write back
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("✅ Vehicle detection ENABLED in config.py")
            return True

        except Exception as e:
            print(f"❌ Failed to enable vehicle detection in config: {e}")
            return False

    def _restore_config(self) -> bool:
        """Restore config.py settings"""
        return self.create_enable_flag()

    def _restore_hybrid_detector(self) -> bool:
        """Restore hybrid_plate_detector.py"""
        try:
            file_path = "utils/hybrid_plate_detector.py"

            # Read current file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Restore YOLO initialization
            yolo_disabled = """        # VEHICLE DETECTION DISABLED FOR STABILITY - PURE PLATE DETECTION MODE
        self.yolo_detector = None
        self.yolo_enabled = False  # PERMANENTLY DISABLED untuk maximum stability
        self.logger.info("🎯 Vehicle detection DISABLED - Using pure plate detection for stability")"""

            yolo_enabled = """        # Initialize YOLO for vehicle detection - PERFORMANCE OPTIMIZED
        try:
            from config import YOLOVehicleConfig
            if YOLOVehicleConfig.ENABLE_VEHICLE_DETECTION:
                self.yolo_detector = YOLOObjectDetector(
                    confidence=0.4,  # Increased for speed (less processing)
                    max_detections=10  # Reduced from 20 to 10 untuk speed
                )
                self.yolo_enabled = True  # ENABLED for hybrid detection
                self.logger.info("✅ YOLO vehicle detector initialized (hybrid mode)")
            else:
                self.yolo_detector = None
                self.yolo_enabled = False
                self.logger.info("🎯 Vehicle detection DISABLED - Using pure plate detection")
        except Exception as e:
            self.yolo_detector = None
            self.yolo_enabled = False
            self.logger.warning(f"YOLO not available: {e}")"""

            content = content.replace(yolo_disabled, yolo_enabled)

            # Restore initialization message
            content = content.replace(
                'self.logger.info("🔧 Pure Plate Detector initialized (OpenCV only - STABLE MODE)")',
                'self.logger.info("🔧 Hybrid Plate Detector initialized (YOLO + OpenCV)")'
            )

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("✅ Vehicle detection RESTORED in hybrid_plate_detector.py")
            return True

        except Exception as e:
            print(f"❌ Failed to restore hybrid detector: {e}")
            return False

    def _restore_stream_manager(self) -> bool:
        """Restore stream_manager.py with complete vehicle detection"""
        try:
            file_path = "stream_manager.py"

            # Read current file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Restore YOLO object detection in processing loop
            content = content.replace(
                "# 🎯 VEHICLE DETECTION DISABLED - Pure plate detection mode\n                object_detections = []  # No vehicle detection untuk maximum stability",
                "# Process YOLO object detection untuk vehicle detection\n                object_detections = []\n                if self.yolo_enabled and self.yolo_detector:\n                    object_detections = self.yolo_detector.detect_objects(frame)"
            )

            # Restore vehicle drawing
            content = content.replace(
                "# 🎯 PURE PLATE DETECTION MODE: Show only plate detections\n                # Vehicle detection disabled untuk stability",
                "# Draw object detections (vehicles)\n                if object_detections:\n                    for detection in object_detections:\n                        cv2.rectangle(annotated_frame, \n                                    (int(detection.bbox[0]), int(detection.bbox[1])), \n                                    (int(detection.bbox[2]), int(detection.bbox[3])), \n                                    (0, 255, 0), 2)"
            )

            # Restore YOLO background loading function
            content = content.replace(
                "def _start_yolo_background_loading(self):\n        \"\"\"DISABLED - YOLO background loading disabled untuk pure plate detection mode\"\"\"\n        self.logger.info(\"🎯 YOLO background loading DISABLED - Pure plate detection mode\")",
                "def _start_yolo_background_loading(self):\n        \"\"\"Start YOLO loading in background thread\"\"\"\n        def load_yolo():\n            try:\n                from utils.yolo_detector import YOLOObjectDetector\n                self.yolo_detector = YOLOObjectDetector()\n                self.logger.info(\"✅ YOLO loaded successfully in background\")\n            except Exception as e:\n                self.logger.error(f\"Failed to load YOLO: {e}\")\n                self.yolo_detector = None\n        \n        thread = threading.Thread(target=load_yolo, daemon=True)\n        thread.start()"
            )

            # Restore tracking system
            content = content.replace(
                "# TRACKING DISABLED - Pure plate detection mode untuk maximum stability\n        self.tracking_manager = None\n        self.tracking_enabled = False\n        self.stats['tracking_enabled'] = False\n        self.logger.info(\"🎯 Object tracking DISABLED - Pure plate detection mode\")",
                "if self.tracking_enabled:\n            self.logger.info(\"Initializing tracking system...\")\n            tracking_config = {\n                'max_disappeared': TrackingConfig.MAX_DISAPPEARED_FRAMES,\n                'max_distance': TrackingConfig.MAX_TRACKING_DISTANCE,\n                'min_hits': TrackingConfig.MIN_HITS_FOR_CONFIRMATION,\n                'iou_threshold': TrackingConfig.IOU_THRESHOLD\n            }\n            \n            self.tracking_manager = TrackingManager(\n                tracking_config=tracking_config,\n                plate_confirmation_threshold=TrackingConfig.PLATE_CONFIRMATION_THRESHOLD,\n                max_plate_age=TrackingConfig.MAX_PLATE_AGE\n            )\n            self.stats['tracking_enabled'] = True\n            self.logger.info(\"✅ Tracking system initialized\")"
            )

            # Restore YOLO enabling
            content = content.replace(
                "self.yolo_enabled = False  # DISABLED - pure plate detection mode untuk stability",
                "self.yolo_enabled = enable_yolo  # RESTORED - hybrid detection available"
            )

            # Restore vehicle detection message
            content = content.replace(
                "# VEHICLE DETECTION DISABLED - Pure plate detection mode\n        # Background YOLO loading disabled untuk stability\n        self.logger.info(\"🎯 Vehicle detection DISABLED - Pure plate detection mode active\")",
                "# Start YOLOv8 loading in background untuk faster startup\n        if self.yolo_enabled:\n            self.logger.info(\"Starting YOLOv8 background loading...\")\n            self._start_yolo_background_loading()\n        else:\n            self.logger.info(\"🎯 Vehicle detection DISABLED - Pure plate detection mode active\")"
            )

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("✅ Vehicle detection RESTORED in stream_manager.py")
            return True

        except Exception as e:
            print(f"❌ Failed to restore stream manager: {e}")
            return False

    def restore_vehicle_detection(self) -> bool:
        """Full restore process"""
        print("🔄 Starting vehicle detection restore process...")

        if not self.check_backup_exists():
            print("❌ Backup file not found! Cannot restore vehicle detection.")
            return False

        success_count = 0
        total_files = len(self.files_to_modify)

        for file_path, restore_func in self.files_to_modify.items():
            if os.path.exists(file_path):
                if restore_func():
                    success_count += 1
                else:
                    print(f"⚠️ Failed to restore {file_path}")
            else:
                print(f"⚠️ File not found: {file_path}")

        if success_count == total_files:
            print("\n🎉 VEHICLE DETECTION RESTORED SUCCESSFULLY!")
            print("✅ All files updated")
            print("✅ Vehicle detection re-enabled")
            print("✅ Hybrid mode available")
            print("\n🚀 Run dengan: python3 headless_stream.py")
            return True
        else:
            print(f"\n⚠️ Partial restore: {success_count}/{total_files} files restored")
            return False

    def disable_vehicle_detection(self) -> bool:
        """Disable vehicle detection again"""
        print("🔄 Disabling vehicle detection...")

        try:
            config_path = "config.py"

            # Read current config
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Disable vehicle detection
            content = content.replace(
                "ENABLE_VEHICLE_DETECTION = True     # ENABLED - hybrid vehicle + plate detection",
                "ENABLE_VEHICLE_DETECTION = False    # DISABLED - fokus pure plate detection untuk stabilitas"
            )

            # Disable vehicle-plate association
            content = content.replace(
                "VEHICLE_PLATE_ASSOCIATION = True   # ENABLED - hybrid detection mode",
                "VEHICLE_PLATE_ASSOCIATION = False  # DISABLED - pure plate detection mode"
            )

            # Write back
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print("✅ Vehicle detection DISABLED")
            return True

        except Exception as e:
            print(f"❌ Failed to disable vehicle detection: {e}")
            return False

def main():
    """Main restore function"""
    import sys

    restore_manager = VehicleDetectionRestore()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "enable" or command == "restore":
            success = restore_manager.restore_vehicle_detection()
            sys.exit(0 if success else 1)

        elif command == "disable":
            success = restore_manager.disable_vehicle_detection()
            sys.exit(0 if success else 1)

        elif command == "status":
            if restore_manager.check_backup_exists():
                print("📋 Backup available - vehicle detection can be restored")
            else:
                print("❌ No backup found")
            sys.exit(0)
        else:
            print("❌ Unknown command. Use: enable, disable, or status")
            sys.exit(1)
    else:
        print("🔧 Vehicle Detection Restore System")
        print("Usage:")
        print("  python3 restore_vehicle_detection.py enable   - Enable vehicle detection")
        print("  python3 restore_vehicle_detection.py disable  - Disable vehicle detection")
        print("  python3 restore_vehicle_detection.py status   - Check backup status")

if __name__ == "__main__":
    main()