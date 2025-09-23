#!/usr/bin/env python3
"""
YOLOv8 License Plate Detector
Menggunakan YOLO untuk detection plat nomor yang akurat seperti object detection
"""

import cv2
import numpy as np
import logging
import time
import requests
import os
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Check YOLOv8 availability
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

@dataclass
class PlateDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    angle: float = 0.0
    processed_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    vehicle_type: str = "unknown"
    detection_method: str = "yolo"

class YOLOPlateDetector:
    """
    YOLOv8-based license plate detector yang akurat seperti object detection
    """
    
    def __init__(self, confidence=0.5, streaming_mode=True):
        """
        Initialize YOLO plate detector
        
        Args:
            confidence: Confidence threshold for plate detection
            streaming_mode: Enable optimizations for real-time streaming
        """
        self.confidence = confidence
        self.streaming_mode = streaming_mode
        self.model = None
        self.enabled = False
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.total_detections = 0
        self.successful_ocr = 0
        self.failed_ocr = 0
        self.false_positives = 0
        
        # OCR setup
        try:
            import pytesseract
            self.ocr_available = True
            # Configure Tesseract for Indonesian plates
            self.ocr_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        except ImportError:
            self.ocr_available = False
            self.logger.warning("Tesseract not available for text extraction")
        
        self._setup_model()
    
    def _setup_model(self):
        """Setup YOLO license plate detection model"""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLOv8 not available. Install ultralytics package.")
            return
        
        try:
            # Try to download pre-trained license plate model
            model_path = self._download_license_plate_model()
            
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                self.enabled = True
                self.logger.info(f"✅ YOLO license plate model loaded: {model_path}")
            else:
                # Fallback: Use YOLOv8n and train it for license plates
                self.logger.warning("Pre-trained license plate model not found. Using fallback approach.")
                self._setup_fallback_detection()
                
        except Exception as e:
            self.logger.error(f"Failed to setup YOLO plate model: {e}")
            self.enabled = False
    
    def _download_license_plate_model(self) -> Optional[str]:
        """
        Download pre-trained license plate detection model
        """
        model_path = "license_plate_detector.pt"
        
        if os.path.exists(model_path):
            return model_path
        
        # URLs for pre-trained models (ordered by preference)
        model_urls = [
            "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"  # fallback
        ]
        
        for url in model_urls:
            try:
                self.logger.info(f"Downloading license plate model from: {url}")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    self.logger.info(f"✅ Downloaded license plate model: {model_path}")
                    return model_path
                    
            except Exception as e:
                self.logger.warning(f"Failed to download from {url}: {e}")
                continue
        
        return None
    
    def _setup_fallback_detection(self):
        """
        Setup fallback detection using YOLOv8n + custom training approach
        """
        try:
            # Use standard YOLOv8n as base
            self.model = YOLO('yolov8n.pt')
            self.enabled = True
            self.use_fallback = True
            self.logger.info("✅ Using YOLOv8n as fallback for license plate detection")
            
        except Exception as e:
            self.logger.error(f"Failed to setup fallback detection: {e}")
            self.enabled = False
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """
        Detect license plates using YOLO
        
        Args:
            image: Input image
            
        Returns:
            List of PlateDetection objects
        """
        if not self.enabled or self.model is None:
            return []
        
        detections = []
        start_time = time.time()
        
        try:
            # Run YOLO inference
            results = self.model(
                image,
                conf=self.confidence,
                verbose=False,
                imgsz=640 if self.streaming_mode else 1280
            )
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get bounding box coordinates
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        
                        # Get confidence
                        conf = float(boxes.conf[i].cpu().numpy())
                        
                        # Get class (for license plate specific models)
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Filter for license plate detections
                        if self._is_license_plate_detection(class_id, w, h, conf):
                            # Extract text using OCR
                            plate_roi = image[y1:y2, x1:x2]
                            text, ocr_conf = self._extract_text_with_ocr(plate_roi)
                            
                            # Create detection
                            detection = PlateDetection(
                                text=text,
                                confidence=conf * 100,
                                bbox=(x1, y1, w, h),
                                angle=0.0,
                                timestamp=time.time(),
                                detection_method="yolo"
                            )
                            
                            detections.append(detection)
                            
                            # Update statistics
                            self.total_detections += 1
                            if text and len(text) >= 3:
                                self.successful_ocr += 1
                            else:
                                self.failed_ocr += 1
            
            # Sort by confidence
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit for streaming mode
            if self.streaming_mode and len(detections) > 3:
                detections = detections[:3]
            
            detection_time = time.time() - start_time
            self.logger.info(f"🎯 YOLO plate detection: {len(detections)} plates in {detection_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in YOLO plate detection: {e}")
        
        return detections
    
    def _is_license_plate_detection(self, class_id: int, width: int, height: int, confidence: float) -> bool:
        """
        Determine if detection is a license plate
        """
        # If using dedicated license plate model, accept all detections above confidence
        if not hasattr(self, 'use_fallback'):
            return confidence >= self.confidence
        
        # For fallback mode with YOLOv8n, use heuristics
        # Look for any object that could contain license plates
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        
        # More permissive criteria - look for vehicles and objects that might contain plates
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        return (
            confidence >= self.confidence and
            (class_id in vehicle_classes or  # Vehicle detection
             (1.5 <= aspect_ratio <= 6.0 and  # Could be plate
              500 <= area <= 50000 and  # Reasonable size range
              width >= 30 and height >= 15))  # Minimum size
        )
    
    def _extract_text_with_ocr(self, plate_roi: np.ndarray) -> Tuple[str, float]:
        """
        Extract text from plate region using OCR
        """
        if not self.ocr_available or plate_roi.size == 0:
            return "", 0.0
        
        try:
            import pytesseract
            
            # Preprocess image for better OCR
            preprocessed = self._preprocess_for_ocr(plate_roi)
            
            # Extract text
            text = pytesseract.image_to_string(
                preprocessed, 
                config=self.ocr_config
            ).strip().upper()
            
            # Clean text
            text = ''.join(c for c in text if c.isalnum())
            
            # Simple confidence based on text characteristics
            confidence = self._calculate_ocr_confidence(text)
            
            return text, confidence
            
        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return "", 0.0
    
    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess ROI for better OCR results
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Resize for better OCR
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def _calculate_ocr_confidence(self, text: str) -> float:
        """
        Calculate OCR confidence based on text characteristics
        """
        if not text:
            return 0.0
        
        confidence = 50.0  # Base confidence
        
        # Length bonus
        if 5 <= len(text) <= 8:
            confidence += 30
        elif 3 <= len(text) <= 10:
            confidence += 15
        
        # Character composition bonus
        has_letters = any(c.isalpha() for c in text)
        has_numbers = any(c.isdigit() for c in text)
        
        if has_letters and has_numbers:
            confidence += 20
        elif has_letters or has_numbers:
            confidence += 10
        
        return min(100.0, confidence)
    
    def draw_detections(self, frame: np.ndarray, detections: List[PlateDetection], 
                       show_roi: bool = True) -> np.ndarray:
        """
        Draw YOLO plate detections with distinctive styling
        """
        result = frame.copy()
        
        # Sort detections by confidence untuk prioritas visual
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        for i, detection in enumerate(sorted_detections):
            x, y, w, h = detection.bbox
            
            # YOLO PLATE BOUNDING BOX - Distinctive colors
            if i == 0:  # Best detection - bright cyan
                plate_color = (255, 255, 0)  # CYAN untuk plat terbaik YOLO
                thickness = 4
            else:  # Other detections - blue
                plate_color = (255, 128, 0)  # BLUE untuk plat lainnya YOLO
                thickness = 3
            
            # Double border dengan YOLO signature
            # Border luar (hitam)
            cv2.rectangle(result, (x-2, y-2), (x + w + 2, y + h + 2), (0, 0, 0), thickness+1)
            # Border dalam (warna YOLO)
            cv2.rectangle(result, (x, y), (x + w, y + h), plate_color, thickness)
            
            # YOLO-style corner markers
            corner_size = 12
            corner_thickness = 3
            # Top-left
            cv2.line(result, (x, y), (x + corner_size, y), plate_color, corner_thickness)
            cv2.line(result, (x, y), (x, y + corner_size), plate_color, corner_thickness)
            # Top-right
            cv2.line(result, (x + w, y), (x + w - corner_size, y), plate_color, corner_thickness)
            cv2.line(result, (x + w, y), (x + w, y + corner_size), plate_color, corner_thickness)
            # Bottom-left
            cv2.line(result, (x, y + h), (x + corner_size, y + h), plate_color, corner_thickness)
            cv2.line(result, (x, y + h), (x, y + h - corner_size), plate_color, corner_thickness)
            # Bottom-right
            cv2.line(result, (x + w, y + h), (x + w - corner_size, y + h), plate_color, corner_thickness)
            cv2.line(result, (x + w, y + h), (x + w, y + h - corner_size), plate_color, corner_thickness)
            
            # YOLO PLATE LABEL - Distinctive
            if detection.text:
                if i == 0:
                    label = f"🎯 YOLO-PLATE: {detection.text} ({detection.confidence:.0f}%)"
                    font_scale = 0.8
                else:
                    label = f"YOLO-PLATE: {detection.text} ({detection.confidence:.0f}%)"
                    font_scale = 0.7
                
                font = cv2.FONT_HERSHEY_DUPLEX
                font_thickness = 2
                
                # Get text size
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Background dengan YOLO styling
                bg_padding = 5
                bg_x1 = x - bg_padding
                bg_y1 = y - text_h - 15 - bg_padding  
                bg_x2 = x + text_w + bg_padding
                bg_y2 = y - 5 + bg_padding
                
                # Background hitam dengan border YOLO
                cv2.rectangle(result, (bg_x1-1, bg_y1-1), (bg_x2+1, bg_y2+1), (0, 0, 0), -1)
                cv2.rectangle(result, (bg_x1, bg_y1), (bg_x2, bg_y2), plate_color, -1)
                
                # Text putih tebal
                cv2.putText(result, label, (x, y - 8), font, font_scale, (255, 255, 255), font_thickness)
        
        return result
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get YOLO plate detection statistics
        """
        success_rate = (self.successful_ocr / self.total_detections * 100) if self.total_detections > 0 else 0
        
        return {
            "total_detections": self.total_detections,
            "successful_ocr": self.successful_ocr,
            "failed_ocr": self.failed_ocr,
            "false_positives": self.false_positives,
            "success_rate": round(success_rate, 1),
            "detection_method": "YOLO"
        }

def check_and_download_license_plate_model():
    """
    Check and download license plate detection model
    """
    detector = YOLOPlateDetector()
    return detector.enabled

if __name__ == "__main__":
    # Test YOLO plate detector
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is not None:
            detector = YOLOPlateDetector()
            detections = detector.detect_plates(image)
            
            print(f"🎯 YOLO detected {len(detections)} license plates:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. '{det.text}' ({det.confidence:.1f}%)")
            
            # Draw and save result
            result = detector.draw_detections(image, detections)
            cv2.imwrite("yolo_plate_result.jpg", result)
            print("💾 Result saved: yolo_plate_result.jpg")
        else:
            print(f"❌ Could not load image: {image_path}")
    else:
        print("Usage: python yolo_plate_detector.py <image_path>")