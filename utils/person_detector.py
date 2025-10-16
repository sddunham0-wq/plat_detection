"""
Person Detector Module
Isolated YOLO-based person detection system yang tidak mengganggu plate detection
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Check YOLOv8 availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

@dataclass
class PersonDetection:
    """Data class untuk hasil deteksi person"""
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    person_id: Optional[int] = None  # For tracking purposes
    timestamp: float = 0.0

class PersonDetector:
    """
    Isolated Person Detector menggunakan YOLOv8
    Tidak mengganggu sistem plate detection yang ada
    """

    def __init__(self, model_path='yolov8n.pt', confidence=0.5, max_detections=20):
        """
        Initialize Person Detector

        Args:
            model_path: Path to YOLO model (default: yolov8n.pt)
            confidence: Confidence threshold untuk person detection (default: 0.5)
            max_detections: Maximum person detections per frame (default: 20)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.max_detections = max_detections
        self.model = None
        self.enabled = False
        self.logger = logging.getLogger(__name__)

        # Person class ID dalam COCO dataset
        self.PERSON_CLASS_ID = 0

        # Statistics
        self.total_detections = 0
        self.detection_times = []

        # Bounding box styling
        self.bbox_color = (255, 0, 0)  # Blue (BGR format)
        self.bbox_thickness = 2
        self.label_font = cv2.FONT_HERSHEY_SIMPLEX
        self.label_scale = 0.6
        self.label_thickness = 2

        # Initialize model
        self.initialize()

    def initialize(self) -> bool:
        """
        Initialize YOLOv8 model dengan error handling
        Returns True jika berhasil, False jika gagal
        """
        if not YOLO_AVAILABLE:
            self.logger.warning("❌ YOLOv8 (ultralytics) not available for person detection")
            self.logger.info("To enable: pip install ultralytics")
            return False

        try:
            self.logger.info(f"🔄 Loading YOLOv8 model for person detection: {self.model_path}")

            # Load model with error isolation
            self.model = YOLO(self.model_path)
            self.enabled = True

            self.logger.info("✅ Person Detector initialized successfully")
            self.logger.info(f"   - Confidence threshold: {self.confidence}")
            self.logger.info(f"   - Max detections: {self.max_detections}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load YOLOv8 model for person detection: {str(e)}")
            self.logger.warning("Person detection will be disabled")
            self.enabled = False
            return False

    def is_enabled(self) -> bool:
        """Check if person detection is enabled and ready"""
        return self.enabled and self.model is not None

    def detect_persons(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Detect persons dalam frame dengan error isolation

        Args:
            frame: Input frame (numpy array)

        Returns:
            List of PersonDetection objects (empty list jika error)
        """
        if not self.is_enabled():
            return []

        start_time = time.time()
        detections = []

        try:
            # Run YOLO detection dengan error handling
            results = self.model(
                frame,
                conf=self.confidence,
                classes=[self.PERSON_CLASS_ID],  # Only detect person class
                max_det=self.max_detections,
                verbose=False
            )

            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])

                        # Verify it's a person (double check)
                        if class_id != self.PERSON_CLASS_ID:
                            continue

                        # Convert to (x, y, w, h) format
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1

                        # Create detection object
                        detection = PersonDetection(
                            confidence=confidence,
                            bbox=(x, y, w, h),
                            timestamp=time.time()
                        )

                        detections.append(detection)

            # Update statistics
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            self.total_detections += len(detections)

            # Keep only last 100 detection times for moving average
            if len(self.detection_times) > 100:
                self.detection_times = self.detection_times[-100:]

            # Log detection (optional, for debugging)
            if detections:
                self.logger.debug(f"👤 Detected {len(detections)} person(s) in {detection_time:.3f}s")

            return detections

        except Exception as e:
            # ERROR ISOLATION: Jangan crash sistem jika person detection error
            self.logger.error(f"❌ Error in person detection: {str(e)}")
            return []  # Return empty list instead of crashing

    def draw_detections(self, frame: np.ndarray, detections: List[PersonDetection],
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw person detections pada frame

        Args:
            frame: Input frame
            detections: List of PersonDetection objects
            show_confidence: Show confidence scores on labels

        Returns:
            Annotated frame dengan person bounding boxes
        """
        if not detections:
            return frame

        try:
            annotated_frame = frame.copy()

            for detection in detections:
                x, y, w, h = detection.bbox
                x1, y1, x2, y2 = x, y, x + w, y + h

                # Draw bounding box (BLUE)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    self.bbox_color,
                    self.bbox_thickness
                )

                # Prepare label
                if show_confidence:
                    label = f'Person: {detection.confidence:.2f}'
                else:
                    label = 'Person'

                # Calculate label size
                label_size = cv2.getTextSize(
                    label,
                    self.label_font,
                    self.label_scale,
                    self.label_thickness
                )[0]

                # Draw label background (BLUE)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    self.bbox_color,
                    -1  # Filled rectangle
                )

                # Draw label text (WHITE)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    self.label_font,
                    self.label_scale,
                    (255, 255, 255),  # White text
                    self.label_thickness
                )

            return annotated_frame

        except Exception as e:
            # ERROR ISOLATION: Return original frame jika drawing error
            self.logger.error(f"❌ Error drawing person detections: {str(e)}")
            return frame

    def get_statistics(self) -> Dict:
        """
        Get person detection statistics

        Returns:
            Dictionary dengan detection stats
        """
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0

        return {
            'enabled': self.enabled,
            'total_detections': self.total_detections,
            'avg_detection_time': round(avg_detection_time, 3),
            'detection_fps': round(1.0 / avg_detection_time, 1) if avg_detection_time > 0 else 0,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence,
            'max_detections': self.max_detections
        }

    def reset_statistics(self):
        """Reset detection statistics"""
        self.total_detections = 0
        self.detection_times = []
        self.logger.info("Person detection statistics reset")

    def set_confidence(self, confidence: float):
        """
        Update confidence threshold

        Args:
            confidence: New confidence threshold (0.0 - 1.0)
        """
        self.confidence = max(0.0, min(1.0, confidence))
        self.logger.info(f"Person detection confidence threshold set to: {self.confidence}")

    def set_max_detections(self, max_detections: int):
        """
        Update maximum detections per frame

        Args:
            max_detections: New max detections limit
        """
        self.max_detections = max(1, min(100, max_detections))
        self.logger.info(f"Person detection max detections set to: {self.max_detections}")

    def enable(self):
        """Enable person detection (if model available)"""
        if YOLO_AVAILABLE and self.model is not None:
            self.enabled = True
            self.logger.info("✅ Person detection enabled")
        else:
            self.logger.warning("❌ Cannot enable person detection - model not available")

    def disable(self):
        """Disable person detection"""
        self.enabled = False
        self.logger.info("⏸️  Person detection disabled")

# Factory function untuk easy initialization
def create_person_detector(confidence=0.5, max_detections=20) -> Optional[PersonDetector]:
    """
    Factory function untuk create PersonDetector dengan error handling

    Args:
        confidence: Confidence threshold (default: 0.5)
        max_detections: Maximum detections per frame (default: 20)

    Returns:
        PersonDetector instance atau None jika gagal
    """
    try:
        detector = PersonDetector(confidence=confidence, max_detections=max_detections)
        if detector.is_enabled():
            return detector
        else:
            logging.getLogger(__name__).warning("Person detector created but not enabled")
            return detector  # Return detector anyway untuk future enable
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create person detector: {str(e)}")
        return None
