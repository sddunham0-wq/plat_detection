"""
Plate Counter Manager
Centralized system untuk accurate plate counting dengan deduplication dan tracking integration
Mengatasi masalah over-counting pada detection statistics
"""

import time
import numpy as np
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import difflib

@dataclass
class UniquePlate:
    """Data class untuk plat unik yang sudah di-track"""
    id: int
    text: str
    normalized_text: str  # Text yang sudah dinormalisasi untuk comparison
    bbox: Tuple[int, int, int, int]
    confidence: float
    first_seen: float
    last_seen: float
    detection_count: int = 0
    tracking_id: Optional[int] = None
    vehicle_type: str = "unknown"
    is_confirmed: bool = False
    center_history: List[Tuple[float, float]] = field(default_factory=list)

    def update_position(self, bbox: Tuple[int, int, int, int]):
        """Update position dan center history"""
        self.bbox = bbox
        self.last_seen = time.time()
        x, y, w, h = bbox
        center = (x + w/2, y + h/2)
        self.center_history.append(center)

        # Keep only last 10 positions untuk velocity calculation
        if len(self.center_history) > 10:
            self.center_history.pop(0)

    def get_velocity(self) -> Tuple[float, float]:
        """Calculate velocity dari position history"""
        if len(self.center_history) < 2:
            return (0.0, 0.0)

        recent = self.center_history[-5:]  # Last 5 positions
        if len(recent) < 2:
            return (0.0, 0.0)

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = len(recent) * 0.1  # Assume 10fps, so 0.1s per frame

        return (dx/dt if dt > 0 else 0.0, dy/dt if dt > 0 else 0.0)

    def is_expired(self, max_age: float) -> bool:
        """Check apakah plate sudah expired"""
        return (time.time() - self.last_seen) > max_age

class PlateCounterManager:
    """
    Manager untuk accurate plate counting dengan deduplication dan tracking
    """

    def __init__(self,
                 similarity_threshold: float = 0.8,
                 spatial_proximity_distance: float = 50.0,
                 plate_expiry_time: float = 5.0,
                 confirmation_threshold: int = 3,
                 confidence_filter_min: float = 0.6):
        """
        Initialize plate counter manager

        Args:
            similarity_threshold: Text similarity threshold (0-1)
            spatial_proximity_distance: Max pixel distance untuk same plate
            plate_expiry_time: Seconds before plate considered expired
            confirmation_threshold: Min detection count untuk confirmation
            confidence_filter_min: Min confidence untuk counting
        """
        self.similarity_threshold = similarity_threshold
        self.spatial_proximity_distance = spatial_proximity_distance
        self.plate_expiry_time = plate_expiry_time
        self.confirmation_threshold = confirmation_threshold
        self.confidence_filter_min = confidence_filter_min

        # Tracking storage
        self.unique_plates: Dict[int, UniquePlate] = {}
        self.next_plate_id = 1

        # Statistics
        self.session_stats = {
            'total_raw_detections': 0,
            'total_unique_plates': 0,
            'current_visible_plates': 0,
            'confirmed_plates_count': 0,
            'false_positives_filtered': 0,
            'duplicate_detections_filtered': 0,
            'expired_plates_count': 0
        }

        # Performance tracking
        self.processing_times = []
        self.logger = logging.getLogger(__name__)

        # Deduplication cache
        self.recent_detections_cache = []  # For fast duplicate checking
        self.cache_max_size = 100

        self.logger.info(f"PlateCounterManager initialized: similarity={similarity_threshold}, "
                        f"proximity={spatial_proximity_distance}px, expiry={plate_expiry_time}s")

    def normalize_text(self, text: str) -> str:
        """Normalize text untuk consistent comparison - Enhanced for Indonesian plates"""
        if not text:
            return ""

        # Enhanced normalization untuk Indonesian plates
        text = text.strip().upper()

        # Clean up common noise characters
        noise_chars = ['.', ',', ';', ':', '!', '?', '-', '_', '|', '\\', '/', '*']
        for char in noise_chars:
            text = text.replace(char, ' ')

        # Handle Indonesian plate format: "B 1234 XYZ" or "B1234XYZ"
        # Keep alphanumeric dan normalize spaces
        normalized_chars = []
        for c in text:
            if c.isalnum():
                normalized_chars.append(c)
            elif c == ' ' and normalized_chars and normalized_chars[-1] != ' ':
                normalized_chars.append(' ')

        normalized = ''.join(normalized_chars).strip()

        # Standardize Indonesian plate format: ensure space separation
        import re
        # Pattern: Letter(s) + Numbers + Letter(s)
        plate_pattern = r'^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})$'
        no_space_match = re.match(plate_pattern, normalized.replace(' ', ''))

        if no_space_match:
            # Reformat to standard: "B 1234 XYZ"
            area, number, suffix = no_space_match.groups()
            normalized = f"{area} {number} {suffix}"

        # Smart OCR error correction untuk Indonesian context
        # Handle cases where text doesn't have spaces yet (like "81234ABC")
        if ' ' not in normalized and len(normalized) >= 6:
            # Try to identify Indonesian pattern in no-space format
            # Look for pattern: Letter(s) + Numbers + Letter(s)
            import re
            pattern_match = re.match(r'^([A-Z0-9]{1,2})(\d{1,4})([A-Z0-9]{1,3})$', normalized)

            if pattern_match:
                area_raw, number, suffix_raw = pattern_match.groups()

                # Fix OCR errors in area code
                area_fixed = area_raw
                for i, char in enumerate(area_raw):
                    if char.isdigit() and i == 0:  # First character should likely be letter
                        if char == '0': area_fixed = area_fixed[:i] + 'O' + area_fixed[i+1:]
                        elif char == '1': area_fixed = area_fixed[:i] + 'I' + area_fixed[i+1:]
                        elif char == '5': area_fixed = area_fixed[:i] + 'S' + area_fixed[i+1:]
                        elif char == '8': area_fixed = area_fixed[:i] + 'B' + area_fixed[i+1:]

                # Fix OCR errors in suffix
                suffix_fixed = suffix_raw
                for i, char in enumerate(suffix_raw):
                    if char.isdigit():
                        if char == '0': suffix_fixed = suffix_fixed[:i] + 'O' + suffix_fixed[i+1:]
                        elif char == '1': suffix_fixed = suffix_fixed[:i] + 'I' + suffix_fixed[i+1:]
                        elif char == '5': suffix_fixed = suffix_fixed[:i] + 'S' + suffix_fixed[i+1:]
                        elif char == '8': suffix_fixed = suffix_fixed[:i] + 'B' + suffix_fixed[i+1:]

                normalized = f"{area_fixed} {number} {suffix_fixed}"

        # Apply corrections to already spaced format
        parts = normalized.split()
        if len(parts) == 3:  # Standard "B 1234 XYZ" format
            area_code, number, suffix = parts

            # Fix common OCR errors dalam area code (first part - should be letters)
            area_fixed = area_code
            for i, char in enumerate(area_code):
                if char.isdigit():
                    if char == '0': area_fixed = area_fixed[:i] + 'O' + area_fixed[i+1:]
                    elif char == '1': area_fixed = area_fixed[:i] + 'I' + area_fixed[i+1:]
                    elif char == '5': area_fixed = area_fixed[:i] + 'S' + area_fixed[i+1:]
                    elif char == '8': area_fixed = area_fixed[:i] + 'B' + area_fixed[i+1:]

            # Fix common OCR errors dalam suffix (last part - should be letters)
            suffix_fixed = suffix
            for i, char in enumerate(suffix):
                if char.isdigit():
                    if char == '0': suffix_fixed = suffix_fixed[:i] + 'O' + suffix_fixed[i+1:]
                    elif char == '1': suffix_fixed = suffix_fixed[:i] + 'I' + suffix_fixed[i+1:]
                    elif char == '5': suffix_fixed = suffix_fixed[:i] + 'S' + suffix_fixed[i+1:]
                    elif char == '8': suffix_fixed = suffix_fixed[:i] + 'B' + suffix_fixed[i+1:]

            normalized = f"{area_fixed} {number} {suffix_fixed}"

        return normalized

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity menggunakan difflib"""
        if not text1 or not text2:
            return 0.0

        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)

        if norm1 == norm2:
            return 1.0

        # Use sequence matcher
        similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        return similarity

    def calculate_spatial_distance(self, bbox1: Tuple[int, int, int, int],
                                 bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)

        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance

    def find_matching_plate(self, detection_text: str, detection_bbox: Tuple[int, int, int, int],
                           confidence: float) -> Optional[int]:
        """
        Find matching plate dari existing unique plates
        Returns plate_id jika found, None jika tidak ada match
        """
        best_match_id = None
        best_similarity = 0.0

        normalized_text = self.normalize_text(detection_text)

        for plate_id, plate in self.unique_plates.items():
            # Skip expired plates
            if plate.is_expired(self.plate_expiry_time):
                continue

            # Calculate text similarity
            text_similarity = self.calculate_text_similarity(detection_text, plate.text)

            # Calculate spatial distance
            spatial_distance = self.calculate_spatial_distance(detection_bbox, plate.bbox)

            # Matching criteria
            text_match = text_similarity >= self.similarity_threshold
            spatial_match = spatial_distance <= self.spatial_proximity_distance

            # Combined scoring
            if text_match and spatial_match:
                # Weighted score (text similarity more important)
                combined_score = text_similarity * 0.7 + (1.0 - spatial_distance/self.spatial_proximity_distance) * 0.3

                if combined_score > best_similarity:
                    best_similarity = combined_score
                    best_match_id = plate_id

        return best_match_id if best_similarity > 0.6 else None

    def add_or_update_detection(self, detection_text: str, detection_bbox: Tuple[int, int, int, int],
                               confidence: float, tracking_id: Optional[int] = None,
                               vehicle_type: str = "unknown") -> Optional[int]:
        """
        Add new detection atau update existing plate
        Returns plate_id jika successfully processed, None jika filtered out
        """
        start_time = time.time()

        self.session_stats['total_raw_detections'] += 1

        # Filter by confidence
        if confidence < self.confidence_filter_min:
            self.session_stats['false_positives_filtered'] += 1
            return None

        # Filter by text quality - Enhanced for Indonesian plates
        normalized_text = self.normalize_text(detection_text)
        if len(normalized_text.replace(' ', '')) < 4:  # Indonesian plates minimum 4 chars (B123X format)
            self.session_stats['false_positives_filtered'] += 1
            return None

        # Additional quality check: Indonesian plate pattern validation
        import re
        indonesia_pattern = r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$'
        if not re.match(indonesia_pattern, normalized_text):
            # Allow some flexibility but filter obvious non-plates
            if len(normalized_text) < 6 or not any(c.isdigit() for c in normalized_text):
                self.session_stats['false_positives_filtered'] += 1
                return None

        # Check for recent duplicate dalam cache
        current_time = time.time()
        for cached_detection in self.recent_detections_cache:
            if current_time - cached_detection['timestamp'] > 1.0:  # 1 second cache
                continue

            if (self.calculate_text_similarity(detection_text, cached_detection['text']) > 0.9 and
                self.calculate_spatial_distance(detection_bbox, cached_detection['bbox']) < 20):
                self.session_stats['duplicate_detections_filtered'] += 1
                return None

        # Add to cache
        self.recent_detections_cache.append({
            'text': detection_text,
            'bbox': detection_bbox,
            'timestamp': current_time
        })

        # Keep cache size manageable
        if len(self.recent_detections_cache) > self.cache_max_size:
            self.recent_detections_cache.pop(0)

        # Find matching existing plate
        matching_plate_id = self.find_matching_plate(detection_text, detection_bbox, confidence)

        if matching_plate_id:
            # Update existing plate
            plate = self.unique_plates[matching_plate_id]
            plate.update_position(detection_bbox)
            plate.detection_count += 1

            # Update text jika confidence lebih tinggi
            if confidence > plate.confidence:
                plate.text = detection_text
                plate.confidence = confidence
                plate.normalized_text = normalized_text

            # Update tracking info
            if tracking_id:
                plate.tracking_id = tracking_id

            plate.vehicle_type = vehicle_type

            # Check confirmation status
            if plate.detection_count >= self.confirmation_threshold and not plate.is_confirmed:
                plate.is_confirmed = True
                self.session_stats['confirmed_plates_count'] += 1
                self.logger.debug(f"Plate confirmed: {plate.text} (ID: {plate.id})")

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return matching_plate_id

        else:
            # Create new unique plate
            new_plate = UniquePlate(
                id=self.next_plate_id,
                text=detection_text,
                normalized_text=normalized_text,
                bbox=detection_bbox,
                confidence=confidence,
                first_seen=current_time,
                last_seen=current_time,
                detection_count=1,
                tracking_id=tracking_id,
                vehicle_type=vehicle_type
            )

            self.unique_plates[self.next_plate_id] = new_plate
            self.session_stats['total_unique_plates'] += 1

            self.logger.debug(f"New unique plate: {detection_text} (ID: {self.next_plate_id})")

            plate_id = self.next_plate_id
            self.next_plate_id += 1

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            return plate_id

    def cleanup_expired_plates(self):
        """Remove expired plates dari tracking"""
        current_time = time.time()
        expired_ids = []

        for plate_id, plate in self.unique_plates.items():
            if plate.is_expired(self.plate_expiry_time):
                expired_ids.append(plate_id)

        for plate_id in expired_ids:
            plate = self.unique_plates[plate_id]
            if plate.is_confirmed:
                self.session_stats['confirmed_plates_count'] -= 1

            del self.unique_plates[plate_id]
            self.session_stats['expired_plates_count'] += 1

            self.logger.debug(f"Expired plate removed: {plate.text} (ID: {plate_id})")

    def get_current_counts(self) -> Dict[str, int]:
        """Get current accurate counts"""
        self.cleanup_expired_plates()

        current_visible = len(self.unique_plates)
        confirmed_visible = len([p for p in self.unique_plates.values() if p.is_confirmed])

        # Update current stats
        self.session_stats['current_visible_plates'] = current_visible

        return {
            'current_visible_plates': current_visible,
            'confirmed_visible_plates': confirmed_visible,
            'total_unique_plates_session': self.session_stats['total_unique_plates'],
            'confirmed_plates_total': self.session_stats['confirmed_plates_count'],
            'raw_detections_processed': self.session_stats['total_raw_detections'],
            'false_positives_filtered': self.session_stats['false_positives_filtered'],
            'duplicates_filtered': self.session_stats['duplicate_detections_filtered'],
            'expired_plates': self.session_stats['expired_plates_count']
        }

    def get_current_plates(self) -> List[UniquePlate]:
        """Get list of currently visible plates"""
        self.cleanup_expired_plates()
        return list(self.unique_plates.values())

    def get_confirmed_plates(self) -> List[UniquePlate]:
        """Get list of confirmed plates only"""
        self.cleanup_expired_plates()
        return [plate for plate in self.unique_plates.values() if plate.is_confirmed]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        counts = self.get_current_counts()

        # Calculate accuracy metrics
        total_processed = self.session_stats['total_raw_detections']
        if total_processed > 0:
            deduplication_rate = (self.session_stats['duplicate_detections_filtered'] / total_processed) * 100
            false_positive_rate = (self.session_stats['false_positives_filtered'] / total_processed) * 100
            unique_plate_rate = (self.session_stats['total_unique_plates'] / total_processed) * 100
        else:
            deduplication_rate = false_positive_rate = unique_plate_rate = 0.0

        # Performance metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0

        # Tracking quality metrics
        confirmed_rate = 0.0
        avg_detection_count = 0.0
        if self.unique_plates:
            confirmed_rate = len([p for p in self.unique_plates.values() if p.is_confirmed]) / len(self.unique_plates) * 100
            avg_detection_count = np.mean([p.detection_count for p in self.unique_plates.values()])

        return {
            **counts,
            'accuracy_metrics': {
                'deduplication_rate_percent': round(deduplication_rate, 1),
                'false_positive_filter_rate_percent': round(false_positive_rate, 1),
                'unique_plate_extraction_rate_percent': round(unique_plate_rate, 1),
                'confirmation_rate_percent': round(confirmed_rate, 1)
            },
            'performance_metrics': {
                'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
                'processing_samples': len(self.processing_times),
                'avg_detections_per_plate': round(avg_detection_count, 1)
            },
            'configuration': {
                'similarity_threshold': self.similarity_threshold,
                'spatial_proximity_distance': self.spatial_proximity_distance,
                'plate_expiry_time': self.plate_expiry_time,
                'confirmation_threshold': self.confirmation_threshold,
                'confidence_filter_min': self.confidence_filter_min
            }
        }

    def reset_session_stats(self):
        """Reset session statistics untuk new test"""
        self.unique_plates.clear()
        self.recent_detections_cache.clear()
        self.processing_times.clear()

        self.session_stats = {
            'total_raw_detections': 0,
            'total_unique_plates': 0,
            'current_visible_plates': 0,
            'confirmed_plates_count': 0,
            'false_positives_filtered': 0,
            'duplicate_detections_filtered': 0,
            'expired_plates_count': 0
        }

        self.next_plate_id = 1
        self.logger.info("Session statistics reset")

    def update_configuration(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                self.logger.info(f"Updated {key}: {old_value} → {value}")
            else:
                self.logger.warning(f"Unknown configuration parameter: {key}")

    def get_plate_by_id(self, plate_id: int) -> Optional[UniquePlate]:
        """Get plate by ID"""
        return self.unique_plates.get(plate_id)

    def get_plates_by_text(self, text: str, similarity_threshold: float = None) -> List[UniquePlate]:
        """Find plates by text similarity"""
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        matching_plates = []
        for plate in self.unique_plates.values():
            similarity = self.calculate_text_similarity(text, plate.text)
            if similarity >= similarity_threshold:
                matching_plates.append(plate)

        return matching_plates

def create_plate_counter_manager(config: Optional[Dict] = None) -> PlateCounterManager:
    """
    Factory function untuk create PlateCounterManager dengan config optimized for Indonesian plates
    """
    # Optimized config untuk Indonesian license plates
    default_config = {
        'similarity_threshold': 0.85,  # Higher threshold untuk Indonesian plates (better text matching)
        'spatial_proximity_distance': 60.0,  # Slightly larger distance untuk vehicle movement
        'plate_expiry_time': 3.0,  # Shorter expiry untuk responsive counting (vehicles move fast)
        'confirmation_threshold': 2,  # Lower threshold untuk responsiveness (2 hits = confirmed)
        'confidence_filter_min': 0.45  # Lower untuk Indonesian OCR challenges tapi still filter noise
    }

    if config:
        default_config.update(config)

    return PlateCounterManager(**default_config)