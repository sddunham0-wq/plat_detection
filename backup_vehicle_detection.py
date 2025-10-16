#!/usr/bin/env python3
"""
BACKUP: Vehicle Detection Code
File ini berisi backup lengkap semua kode vehicle detection yang di-disable
Gunakan file ini untuk restore vehicle detection kapan saja diperlukan
"""

# ===== BACKUP dari hybrid_plate_detector.py =====

# Original YOLO initialization code:
ORIGINAL_YOLO_INIT = '''
# Initialize YOLO for vehicle detection - PERFORMANCE OPTIMIZED
try:
    self.yolo_detector = YOLOObjectDetector(
        confidence=0.4,  # Increased for speed (less processing)
        max_detections=10  # Reduced from 20 to 10 untuk speed
    )
    self.yolo_enabled = False  # Disabled by default untuk speed - enable manually if needed
    self.logger.info("✅ YOLO vehicle detector initialized (performance mode)")
except Exception as e:
    self.yolo_detector = None
    self.yolo_enabled = False
    self.logger.warning(f"YOLO not available: {e}")
'''

# Original vehicle detection method:
ORIGINAL_DETECT_VEHICLE_REGIONS = '''
def _detect_vehicle_regions(self, image: np.ndarray) -> List[Dict]:
    """
    Enhanced YOLO vehicle detection dengan intelligent region expansion
    """
    vehicle_regions = []

    try:
        # Get YOLO detections dengan vehicles only
        object_detections = self.yolo_detector.detect_objects(image, vehicles_only=True)

        for detection in object_detections:
            x, y, w, h = detection.bbox

            # Intelligent expansion based on vehicle type
            if detection.class_name == 'motorcycle':
                # Motorcycles: smaller expansion, focus on front/rear
                expansion_factor = 0.2
                front_expansion = int(w * 0.3)  # More expansion in front
                side_expansion = int(w * 0.1)   # Less on sides

                expanded_x = max(0, x - side_expansion)
                expanded_y = max(0, y - int(h * 0.1))
                expanded_w = min(image.shape[1] - expanded_x, w + front_expansion + side_expansion)
                expanded_h = min(image.shape[0] - expanded_y, h + int(h * 0.2))

            elif detection.class_name in ['car', 'bus', 'truck']:
                # Cars/buses/trucks: larger expansion, more uniform
                front_expansion = int(w * 0.25)
                side_expansion = int(w * 0.15)

                expanded_x = max(0, x - side_expansion)
                expanded_y = max(0, y - int(h * 0.1))
                expanded_w = min(image.shape[1] - expanded_x, w + front_expansion + side_expansion)
                expanded_h = min(image.shape[0] - expanded_y, h + int(h * 0.15))

            else:
                # Default expansion
                expansion = 25
                expanded_x = max(0, x - expansion)
                expanded_y = max(0, y - expansion)
                expanded_w = min(image.shape[1] - expanded_x, w + 2*expansion)
                expanded_h = min(image.shape[0] - expanded_y, h + 2*expansion)

            # Add region to list dengan enhanced metadata
            vehicle_regions.append({
                'bbox': (expanded_x, expanded_y, expanded_w, expanded_h),
                'vehicle_type': detection.class_name,
                'confidence': detection.confidence,
                'original_bbox': detection.bbox,
                'class_id': detection.class_id,
                'expansion_applied': True
            })

        self.logger.info(f"🚗 Found {len(vehicle_regions)} vehicle regions")

    except Exception as e:
        self.logger.warning(f"Vehicle detection failed: {e}")

    return vehicle_regions
'''

# Original detect plates in regions method:
ORIGINAL_DETECT_PLATES_IN_REGIONS = '''
def _detect_plates_in_regions(self, image: np.ndarray, vehicle_regions: List[Dict]) -> List[PlateDetection]:
    """
    Detect plates within vehicle regions
    """
    all_detections = []

    for i, region in enumerate(vehicle_regions):
        try:
            x, y, w, h = region['bbox']

            # Extract region
            roi = image[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Apply plate detection to region
            region_detections = self.plate_detector.detect_plates(roi)

            # Enhanced OCR post-processing untuk better text recognition
            if self.enhanced_ocr_enabled and region_detections:
                region_detections = self._enhance_detections_with_ocr(region_detections, roi)

            # Adjust coordinates back to full image
            for detection in region_detections:
                det_x, det_y, det_w, det_h = detection.bbox

                # Enhanced stability filtering
                base_confidence = detection.confidence

                # Apply spatial validation dan stability bonus - EXTREME RELAXED for CCTV
                spatial_score = self._validate_plate_in_vehicle(detection.bbox, region, roi.shape)
                if spatial_score < 0.05:  # Further lowered from 0.08 to 0.05 for maximum CCTV permissiveness
                    continue

                stability_bonus = self._calculate_stability_bonus(detection, region['vehicle_type'])
                spatial_bonus = spatial_score * 10  # Convert to bonus points
                final_confidence = min(100.0, base_confidence + stability_bonus + spatial_bonus)

                # Create hybrid detection dengan enhanced validation
                if self._validate_detection_stability(detection, final_confidence):
                    hybrid_detection = PlateDetection(
                        text=detection.text,
                        confidence=final_confidence,
                        bbox=(x + det_x, y + det_y, det_w, det_h),
                        angle=detection.angle,
                        vehicle_type=region['vehicle_type'],
                        detection_method=f"hybrid_{region['vehicle_type']}"
                    )

                    all_detections.append(hybrid_detection)

                # Update stats
                self.total_detections += 1
                if detection.text and len(detection.text) >= 3:
                    self.successful_ocr += 1
                else:
                    self.failed_ocr += 1

            self.logger.debug(f"Region {i+1} ({region['vehicle_type']}): {len(region_detections)} plates")

        except Exception as e:
            self.logger.warning(f"Failed to process vehicle region {i+1}: {e}")

    return all_detections
'''

# Original validate plate in vehicle method:
ORIGINAL_VALIDATE_PLATE_IN_VEHICLE = '''
def _validate_plate_in_vehicle(self, plate_bbox: Tuple[int, int, int, int],
                             vehicle_region: Dict, roi_shape: Tuple[int, int, int]) -> float:
    """
    Validate spatial relationship antara plate dan vehicle
    Returns score 0.0-1.0 dimana 1.0 = perfect spatial fit
    """
    try:
        plate_x, plate_y, plate_w, plate_h = plate_bbox
        vehicle_type = vehicle_region['vehicle_type']
        roi_h, roi_w = roi_shape[:2]

        # Calculate plate position relative to vehicle region
        plate_center_x = plate_x + plate_w / 2
        plate_center_y = plate_y + plate_h / 2

        # Normalize to ROI dimensions
        norm_x = plate_center_x / roi_w if roi_w > 0 else 0.5
        norm_y = plate_center_y / roi_h if roi_h > 0 else 0.5

        # Vehicle-specific spatial expectations
        if vehicle_type == 'motorcycle':
            # Motorcycles: plate biasanya di front atau rear, center horizontally
            expected_regions = [
                (0.2, 0.7, 0.3, 0.9),  # Front area
                (0.7, 0.9, 0.3, 0.9),  # Rear area
            ]
        elif vehicle_type in ['car', 'bus', 'truck']:
            # Cars: plate bisa front atau rear, lebih wide area
            expected_regions = [
                (0.1, 0.9, 0.2, 0.8),  # Front bumper area
                (0.1, 0.9, 0.6, 0.95), # Rear area
            ]
        else:
            # Default: anywhere in bottom half
            expected_regions = [
                (0.0, 1.0, 0.4, 1.0),  # Bottom half
            ]

        # Calculate score based on proximity to expected regions
        max_score = 0.0
        for x_min, x_max, y_min, y_max in expected_regions:
            if x_min <= norm_x <= x_max and y_min <= norm_y <= y_max:
                # Perfect match
                max_score = 1.0
                break
            else:
                # Calculate distance to region
                dx = min(abs(norm_x - x_min), abs(norm_x - x_max)) if not (x_min <= norm_x <= x_max) else 0
                dy = min(abs(norm_y - y_min), abs(norm_y - y_max)) if not (y_min <= norm_y <= y_max) else 0
                distance = (dx**2 + dy**2)**0.5
                score = max(0, 1.0 - distance)
                max_score = max(max_score, score)

        return max_score

    except Exception as e:
        self.logger.warning(f"Spatial validation error: {e}")
        return 0.1  # Low but non-zero score
'''

# Original hybrid detect method dengan vehicle support:
ORIGINAL_DETECT_WITH_VEHICLES = '''
def detect_plates_with_vehicles(self, image: np.ndarray) -> List[PlateDetection]:
    """
    Hybrid detection: YOLO vehicle detection + OpenCV plate detection
    """
    detections = []
    start_time = time.time()

    try:
        if self.yolo_enabled and self.yolo_detector:
            # Enhanced hybrid approach dengan vehicle detection
            self.logger.info("🎯 Using enhanced hybrid detection (YOLO+OpenCV)")

            # Step 1: Detect vehicle regions using YOLO
            vehicle_regions = self._detect_vehicle_regions(image)
            self.vehicle_regions_found = len(vehicle_regions)

            if vehicle_regions:
                # Step 2: Detect plates within vehicle regions
                detections = self._detect_plates_in_regions(image, vehicle_regions)
                self.logger.info(f"🚗 Found {len(detections)} plates in {len(vehicle_regions)} vehicle regions")

            # Step 3: Fallback to full image detection jika tidak ada vehicle regions
            if not detections:
                self.logger.info("🔄 No plates in vehicle regions, falling back to full detection")
                detections = self._fallback_full_detection(image)
        else:
            # Pure plate detection tanpa vehicle guidance
            self.logger.info("🎯 Using direct plate detection (no vehicle detection)")
            detections = self._fallback_full_detection(image)
            self.vehicle_regions_found = 0

        # Post-process detections
        detections = self._post_process_detections(detections)

        detection_time = time.time() - start_time
        self.logger.info(f"🎯 Hybrid detection: {len(detections)} plates in {detection_time:.2f}s")

    except Exception as e:
        self.logger.error(f"Error in hybrid detection: {e}")

    return detections
'''

# ===== RESTORE INSTRUCTIONS =====

RESTORE_INSTRUCTIONS = '''
=== CARA RESTORE VEHICLE DETECTION ===

1. ENABLE di config.py:
   Set: ENABLE_VEHICLE_DETECTION = True

2. RESTORE di hybrid_plate_detector.py:
   - Uncomment semua YOLO initialization code
   - Restore _detect_vehicle_regions method
   - Restore _detect_plates_in_regions method
   - Restore _validate_plate_in_vehicle method
   - Change detect_plates() method untuk support vehicle detection

3. RESTORE di stream_manager.py:
   - Enable YOLO loading background
   - Restore vehicle detection statistics
   - Enable vehicle-plate association

4. RUN dengan vehicle detection:
   python3 headless_stream.py --enable-vehicles

=== QUICK RESTORE FUNCTION ===
Copy code dari backup ini dan paste kembali ke file original
'''

def restore_vehicle_detection():
    """
    Function untuk restore vehicle detection
    Call function ini untuk enable kembali vehicle detection
    """
    print("📋 Vehicle Detection Restore Instructions:")
    print("1. Set ENABLE_VEHICLE_DETECTION = True di config.py")
    print("2. Uncomment YOLO code di hybrid_plate_detector.py")
    print("3. Restore vehicle detection methods dari backup ini")
    print("4. Run dengan: python3 headless_stream.py --enable-vehicles")
    print("\n✅ Semua kode ter-backup di file ini!")

if __name__ == "__main__":
    restore_vehicle_detection()