# -*- coding: utf-8 -*-
"""
YOLO PLATE DETECTOR
Deteksi plat nomor menggunakan YOLOv8
"""

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from .yolo_model_loader import find_best_yolo_model, get_model_info

logger = logging.getLogger(__name__)

class YOLOPlateDetector:
    """Detector plat nomor menggunakan YOLOv8"""

    def __init__(self, model_path='models/best.pt', conf_threshold=0.25):
        """
        Initialize YOLO plate detector

        Args:
            model_path: Path ke YOLO model (.pt file)
            conf_threshold: Confidence threshold (0.0-1.0)
        """
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.model = None
        self.model_type = None

        # Try specified model first
        try:
            logger.info(f"🔧 Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            self.model_type = 'custom' if 'best.pt' in model_path else 'base'
            logger.info(f"✅ YOLO model loaded: {model_path}")
            logger.info(f"   Model type: {self.model_type}")
            logger.info(f"   Confidence threshold: {conf_threshold}")
            return

        except Exception as e:
            logger.warning(f"⚠️  Could not load {model_path}: {e}")
            logger.info("🔍 Searching for alternative models...")

            # Try to find best available model
            best_model, model_type = find_best_yolo_model()

            if best_model:
                try:
                    logger.info(f"🔧 Loading alternative model: {best_model}")
                    self.model = YOLO(best_model)
                    self.model_path = best_model
                    self.model_type = model_type

                    model_info = get_model_info(best_model, model_type)
                    logger.info(f"✅ YOLO model loaded: {best_model}")
                    logger.info(f"   Type: {model_type} ({model_info['accuracy']} accuracy)")
                    logger.info(f"   Description: {model_info['description']}")
                    logger.info(f"   Confidence threshold: {conf_threshold}")

                except Exception as e2:
                    logger.error(f"❌ Failed to load alternative model: {e2}")
                    logger.error("💡 Run: python3 download_yolo_model.py")
                    raise RuntimeError(f"No YOLO models available. Run: python3 download_yolo_model.py")
            else:
                logger.error("❌ No YOLO models found!")
                logger.error("💡 Solutions:")
                logger.error("   1. Run: python3 download_yolo_model.py")
                logger.error("   2. Or install: pip3 install ultralytics")
                raise RuntimeError(f"No YOLO models available. Run: python3 download_yolo_model.py")

    def detect(self, frame):
        """
        Deteksi plat nomor di frame

        Args:
            frame: Input image (BGR format from OpenCV)

        Returns:
            List of (x, y, w, h) bounding boxes, sorted by confidence
        """
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device='cpu'  # Use CPU (change to '0' for GPU)
            )

            boxes = []

            # Extract bounding boxes from results
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())

                        # Convert to (x, y, w, h) format
                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)

                        # Validate box size (minimum 30x10 pixels)
                        if w > 30 and h > 10:
                            boxes.append((x, y, w, h))
                            logger.debug(f"✅ Plate detected: conf={conf:.3f}, bbox=({x},{y},{w},{h})")

            # Boxes are already sorted by YOLO confidence
            logger.info(f"📊 YOLO detected {len(boxes)} plate(s)")

            # Return top 3 detections
            return boxes[:3]

        except Exception as e:
            logger.error(f"❌ YOLO detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def detect_with_confidence(self, frame):
        """
        Deteksi plat dengan confidence score

        Args:
            frame: Input image

        Returns:
            List of {'bbox': (x,y,w,h), 'confidence': float}
        """
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                conf=self.conf_threshold,
                verbose=False,
                device='cpu'
            )

            detections = []

            # Extract detections with confidence
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())

                        x = int(x1)
                        y = int(y1)
                        w = int(x2 - x1)
                        h = int(y2 - y1)

                        if w > 30 and h > 10:
                            detections.append({
                                'bbox': (x, y, w, h),
                                'confidence': conf
                            })
                            logger.debug(f"Detection: conf={conf:.3f}, bbox=({x},{y},{w},{h})")

            # Sort by confidence (highest first)
            detections.sort(key=lambda d: d['confidence'], reverse=True)

            logger.info(f"📊 {len(detections)} detection(s) with confidence")

            return detections[:3]

        except Exception as e:
            logger.error(f"❌ Detection with confidence error: {e}")
            return []

    def draw(self, frame, boxes, vehicle_type="KENDARAAN"):
        """
        Gambar bounding boxes di frame dengan label

        Args:
            frame: Input frame
            boxes: List of (x,y,w,h) tuples OR list of dicts with 'bbox' key
            vehicle_type: Label untuk vehicle type

        Returns:
            Annotated frame
        """
        if not boxes:
            return frame

        # Hijau untuk semua box (konsisten)
        GREEN = (0, 255, 0)

        for i, box in enumerate(boxes):
            # Handle both tuple and dict format
            if isinstance(box, dict):
                x, y, w, h = box['bbox']
                conf = box.get('confidence', 0)
                # Label dengan confidence
                if i == 0:
                    label = f"{vehicle_type} {conf:.2f}"
                else:
                    label = f"PLATE #{i+1} {conf:.2f}"
            else:
                # Tuple format (x, y, w, h)
                x, y, w, h = box
                if i == 0:
                    label = vehicle_type
                else:
                    label = f"PLATE #{i+1}"

            # Draw rectangle (hijau)
            cv2.rectangle(frame, (x, y), (x+w, y+h), GREEN, 2)

            # Draw label di atas box
            cv2.putText(frame, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        return frame

    def get_model_info(self):
        """Get YOLO model information"""
        try:
            info = {
                'model_path': self.model_path,
                'conf_threshold': self.conf_threshold,
                'model_type': type(self.model).__name__,
            }
            return info
        except:
            return {'error': 'Unable to get model info'}
