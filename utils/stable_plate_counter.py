#!/usr/bin/env python3
"""
Stable Plate Counter - Ultra Robust Indonesian License Plate Counting System
Designed untuk maximum stability dan accuracy dengan real-world traffic conditions
"""

import time
import numpy as np
import logging
import difflib
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class PlateTrack:
    """Advanced plate tracking dengan temporal stability"""
    id: int
    original_text: str
    normalized_text: str
    clean_text: str  # Text yang sudah di-clean untuk matching
    confidence_scores: List[float] = field(default_factory=list)
    positions: deque = field(default_factory=lambda: deque(maxlen=10))  # Last 10 positions
    timestamps: deque = field(default_factory=lambda: deque(maxlen=10))  # Corresponding timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    detection_count: int = 0
    is_stable: bool = False
    stability_score: float = 0.0
    movement_pattern: str = "stationary"  # stationary, moving_left, moving_right, moving_up, moving_down
    area_history: List[int] = field(default_factory=list)  # Bbox area history untuk size consistency

    def add_detection(self, text: str, bbox: Tuple[int, int, int, int], confidence: float):
        """Add new detection dengan temporal tracking"""
        self.last_seen = time.time()
        self.detection_count += 1
        self.confidence_scores.append(confidence)

        # Track position
        x, y, w, h = bbox
        center = (x + w/2, y + h/2)
        self.positions.append(center)
        self.timestamps.append(self.last_seen)
        self.area_history.append(w * h)

        # Update text jika confidence lebih tinggi
        if confidence > max(self.confidence_scores[:-1], default=0):
            self.original_text = text

        # Update stability
        self._update_stability()
        self._update_movement_pattern()

    def _update_stability(self):
        """Update stability score berdasarkan consistency"""
        if len(self.confidence_scores) < 2:
            self.stability_score = 0.0
            return

        # Factor 1: Confidence consistency (30%)
        conf_std = np.std(self.confidence_scores)
        conf_score = max(0, 1.0 - conf_std / 0.3)  # Lower std = higher stability

        # Factor 2: Position stability (40%)
        if len(self.positions) >= 2:
            distances = []
            for i in range(1, len(self.positions)):
                dx = self.positions[i][0] - self.positions[i-1][0]
                dy = self.positions[i][1] - self.positions[i-1][1]
                distances.append(np.sqrt(dx**2 + dy**2))

            pos_consistency = 1.0 - min(1.0, np.mean(distances) / 50.0)  # Normalize by 50px
        else:
            pos_consistency = 1.0

        # Factor 3: Size consistency (20%)
        if len(self.area_history) >= 2:
            area_std = np.std(self.area_history)
            area_mean = np.mean(self.area_history)
            size_consistency = 1.0 - min(1.0, area_std / area_mean) if area_mean > 0 else 0.0
        else:
            size_consistency = 1.0

        # Factor 4: Detection frequency (10%)
        time_span = self.last_seen - self.first_seen
        freq_score = min(1.0, self.detection_count / max(1, time_span * 2))  # Expected 2 detections per second

        # Weighted stability score
        self.stability_score = (
            conf_score * 0.3 +
            pos_consistency * 0.4 +
            size_consistency * 0.2 +
            freq_score * 0.1
        )

        # Mark as stable jika score tinggi dan detection count cukup
        self.is_stable = self.stability_score > 0.7 and self.detection_count >= 3

    def _update_movement_pattern(self):
        """Analyze movement pattern"""
        if len(self.positions) < 3:
            self.movement_pattern = "stationary"
            return

        # Calculate average movement vector
        recent_positions = list(self.positions)[-5:]  # Last 5 positions
        if len(recent_positions) < 3:
            return

        dx_total = recent_positions[-1][0] - recent_positions[0][0]
        dy_total = recent_positions[-1][1] - recent_positions[0][1]

        # Threshold untuk movement detection
        movement_threshold = 20  # pixels

        if abs(dx_total) > movement_threshold or abs(dy_total) > movement_threshold:
            if abs(dx_total) > abs(dy_total):
                self.movement_pattern = "moving_right" if dx_total > 0 else "moving_left"
            else:
                self.movement_pattern = "moving_down" if dy_total > 0 else "moving_up"
        else:
            self.movement_pattern = "stationary"

    def get_predicted_position(self) -> Tuple[float, float]:
        """Predict next position berdasarkan movement pattern"""
        if len(self.positions) < 2:
            return self.positions[-1] if self.positions else (0, 0)

        recent_positions = list(self.positions)[-3:]
        if len(recent_positions) < 2:
            return recent_positions[-1]

        # Calculate velocity
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]
        dt = len(recent_positions) - 1

        # Predict next position
        next_x = recent_positions[-1][0] + dx / dt
        next_y = recent_positions[-1][1] + dy / dt

        return (next_x, next_y)

    def is_expired(self, max_age: float) -> bool:
        """Check if plate track expired"""
        return (time.time() - self.last_seen) > max_age

    def get_average_confidence(self) -> float:
        """Get average confidence score"""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0

class StablePlateCounter:
    """
    Ultra-stable plate counter dengan advanced Indonesian plate recognition
    """

    def __init__(self,
                 text_similarity_threshold: float = 0.8,
                 spatial_distance_threshold: float = 80.0,  # Larger untuk vehicle movement
                 temporal_window: float = 5.0,  # Longer window untuk stability
                 min_stability_score: float = 0.6,
                 min_confidence: float = 0.4,  # Lower untuk Indonesian OCR
                 confirmation_detections: int = 3):  # Need 3 detections untuk stability

        self.text_similarity_threshold = text_similarity_threshold
        self.spatial_distance_threshold = spatial_distance_threshold
        self.temporal_window = temporal_window
        self.min_stability_score = min_stability_score
        self.min_confidence = min_confidence
        self.confirmation_detections = confirmation_detections

        # Active tracking
        self.active_tracks: Dict[int, PlateTrack] = {}
        self.next_track_id = 1

        # Statistics
        self.session_stats = {
            'total_raw_detections': 0,
            'stable_unique_plates': 0,
            'unstable_filtered': 0,
            'low_confidence_filtered': 0,
            'duplicate_filtered': 0,
            'false_positive_filtered': 0
        }

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)

        # Indonesian plate patterns
        self.indonesia_patterns = [
            r'^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]{1,3}$',  # Standard: B 1234 ABC
            r'^[A-Z]{1}\s*\d{4}\s*[A-Z]{2,3}$',      # Jakarta: B 1234 AB
            r'^[A-Z]{2}\s*\d{1,4}\s*[A-Z]{1,2}$',    # Some regions: AB 1234 C
        ]

        self.logger.info(f"StablePlateCounter initialized with ultra-stable parameters")

    def clean_and_normalize_text(self, text: str) -> str:
        """Advanced text cleaning untuk Indonesian plates"""
        if not text or len(text) < 3:
            return ""

        # Basic cleaning
        text = text.strip().upper()

        # Remove noise characters
        noise_chars = ['.', ',', ';', ':', '!', '?', '-', '_', '|', '\\', '/', '*', '+', '=', '(', ')', '[', ']', '{', '}']
        for char in noise_chars:
            text = text.replace(char, ' ')

        # Normalize spaces
        text = ' '.join(text.split())

        # Indonesian-specific cleaning
        text = self._fix_indonesian_ocr_errors(text)
        text = self._normalize_indonesian_format(text)

        return text

    def _fix_indonesian_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors untuk Indonesian plates"""

        # Common substitutions
        ocr_fixes = {
            # Numbers yang sering salah dibaca sebagai letters
            '0O': '0', 'O0': 'O',  # Context-dependent
            '1I': '1', 'I1': 'I',  # Context-dependent
            '5S': '5', 'S5': 'S',  # Context-dependent
            '8B': '8', 'B8': 'B',  # Context-dependent
            # Common confusions
            'G': '6', 'Z': '2', 'Q': '0'
        }

        # Apply context-aware fixes
        parts = text.split()
        if len(parts) >= 2:
            # Try to identify number part
            for i, part in enumerate(parts):
                if part.isdigit() or any(c.isdigit() for c in part):
                    # This is likely the number part - fix letter intrusions
                    fixed_part = part
                    for old, new in [('O', '0'), ('I', '1'), ('S', '5'), ('B', '8')]:
                        if old in fixed_part and any(c.isdigit() for c in fixed_part):
                            fixed_part = fixed_part.replace(old, new)
                    parts[i] = fixed_part
                else:
                    # This is likely letter part - fix number intrusions
                    fixed_part = part
                    for old, new in [('0', 'O'), ('1', 'I'), ('5', 'S'), ('8', 'B')]:
                        if old in fixed_part and any(c.isalpha() for c in fixed_part):
                            fixed_part = fixed_part.replace(old, new)
                    parts[i] = fixed_part

        return ' '.join(parts)

    def _normalize_indonesian_format(self, text: str) -> str:
        """Normalize ke format Indonesian standard"""
        # Remove extra spaces
        text = ' '.join(text.split())

        # Try to match Indonesian patterns dan reformat
        no_space_text = text.replace(' ', '')

        for pattern in self.indonesia_patterns:
            # Test dengan spaces
            if re.match(pattern, text):
                return text  # Already in correct format

            # Test tanpa spaces dan reformat
            pattern_no_space = pattern.replace(r'\s*', '')
            match = re.match(pattern_no_space, no_space_text)
            if match:
                # Extract components dan reformat
                if len(no_space_text) >= 6:
                    # Standard format: 1-2 letters + 1-4 numbers + 1-3 letters
                    area_match = re.match(r'^([A-Z]{1,2})', no_space_text)
                    if area_match:
                        area = area_match.group(1)
                        remaining = no_space_text[len(area):]

                        number_match = re.match(r'^(\d{1,4})', remaining)
                        if number_match:
                            number = number_match.group(1)
                            suffix = remaining[len(number):]

                            if suffix and suffix.isalpha():
                                return f"{area} {number} {suffix}"

        return text

    def calculate_advanced_similarity(self, text1: str, text2: str) -> float:
        """Advanced similarity calculation untuk Indonesian plates"""
        if not text1 or not text2:
            return 0.0

        clean1 = self.clean_and_normalize_text(text1)
        clean2 = self.clean_and_normalize_text(text2)

        if clean1 == clean2:
            return 1.0

        # Multiple similarity metrics
        # 1. Exact match
        if clean1.replace(' ', '') == clean2.replace(' ', ''):
            return 0.95

        # 2. Sequence similarity
        seq_sim = difflib.SequenceMatcher(None, clean1, clean2).ratio()

        # 3. Jaro-Winkler like similarity (better for short strings)
        jaro_sim = self._jaro_similarity(clean1, clean2)

        # 4. Character position similarity
        pos_sim = self._positional_similarity(clean1, clean2)

        # 5. Number-letter separation similarity (Indonesian specific)
        pattern_sim = self._pattern_similarity(clean1, clean2)

        # Weighted combination
        final_similarity = (
            seq_sim * 0.4 +
            jaro_sim * 0.3 +
            pos_sim * 0.2 +
            pattern_sim * 0.1
        )

        return min(1.0, final_similarity)

    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """Jaro similarity implementation"""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        match_window = max(len1, len2) // 2 - 1
        if match_window < 1:
            match_window = 1

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len1):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        return (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3

    def _positional_similarity(self, s1: str, s2: str) -> float:
        """Character position based similarity"""
        if not s1 or not s2:
            return 0.0

        max_len = max(len(s1), len(s2))
        matches = 0

        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                matches += 1

        return matches / max_len if max_len > 0 else 0.0

    def _pattern_similarity(self, s1: str, s2: str) -> float:
        """Indonesian plate pattern similarity"""
        # Extract patterns: letters, numbers, letters
        def extract_pattern(text):
            parts = text.split()
            if len(parts) >= 3:
                return (len(parts[0]), len(parts[1]), len(parts[2]))
            return (0, 0, 0)

        pattern1 = extract_pattern(s1)
        pattern2 = extract_pattern(s2)

        if pattern1 == pattern2 and pattern1 != (0, 0, 0):
            return 1.0
        elif pattern1[1] == pattern2[1] and pattern1[1] > 0:  # Same number length
            return 0.7
        else:
            return 0.3

    def calculate_spatial_compatibility(self, bbox1: Tuple[int, int, int, int],
                                      bbox2: Tuple[int, int, int, int],
                                      predicted_pos: Optional[Tuple[float, float]] = None) -> float:
        """Advanced spatial compatibility dengan prediction"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)

        # Distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

        # Use predicted position jika available
        if predicted_pos:
            pred_distance = np.sqrt((predicted_pos[0] - center2[0])**2 + (predicted_pos[1] - center2[1])**2)
            distance = min(distance, pred_distance)

        # Size compatibility
        area1, area2 = w1 * h1, w2 * h2
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0

        # Aspect ratio compatibility
        aspect1, aspect2 = w1/h1 if h1 > 0 else 0, w2/h2 if h2 > 0 else 0
        aspect_ratio = min(aspect1, aspect2) / max(aspect1, aspect2) if max(aspect1, aspect2) > 0 else 0

        # Distance score (closer = better)
        distance_score = max(0, 1 - distance / self.spatial_distance_threshold)

        # Combined spatial score
        spatial_score = distance_score * 0.6 + size_ratio * 0.25 + aspect_ratio * 0.15

        return spatial_score

    def find_matching_track(self, text: str, bbox: Tuple[int, int, int, int], confidence: float) -> Optional[int]:
        """Find best matching track dengan comprehensive scoring"""

        best_track_id = None
        best_score = 0.0

        cleaned_text = self.clean_and_normalize_text(text)

        for track_id, track in self.active_tracks.items():
            if track.is_expired(self.temporal_window):
                continue

            # Text similarity
            text_sim = self.calculate_advanced_similarity(text, track.original_text)

            # Spatial compatibility dengan prediction
            predicted_pos = track.get_predicted_position() if len(track.positions) >= 2 else None
            spatial_score = self.calculate_spatial_compatibility(bbox,
                                                               (int(track.positions[-1][0] - 50), int(track.positions[-1][1] - 25), 100, 50),
                                                               predicted_pos)

            # Confidence compatibility
            avg_conf = track.get_average_confidence()
            conf_diff = abs(confidence - avg_conf)
            conf_score = max(0, 1 - conf_diff / 0.3)  # Normalize by 30%

            # Temporal score (more recent = better)
            time_diff = time.time() - track.last_seen
            temporal_score = max(0, 1 - time_diff / self.temporal_window)

            # Stability bonus
            stability_bonus = track.stability_score * 0.1

            # Combined score dengan weighted factors
            combined_score = (
                text_sim * 0.50 +           # Text similarity most important
                spatial_score * 0.25 +      # Spatial compatibility
                conf_score * 0.10 +         # Confidence consistency
                temporal_score * 0.10 +     # Recency
                stability_bonus * 0.05      # Stability bonus
            )

            # Threshold checks
            if (text_sim >= self.text_similarity_threshold and
                spatial_score >= 0.3 and
                combined_score > best_score):

                best_score = combined_score
                best_track_id = track_id

        # Return match only jika score cukup tinggi
        return best_track_id if best_score >= 0.6 else None

    def add_detection(self, text: str, bbox: Tuple[int, int, int, int], confidence: float,
                     vehicle_type: str = "unknown") -> Optional[int]:
        """Add detection dengan ultra-stable processing"""

        start_time = time.time()
        self.session_stats['total_raw_detections'] += 1

        # Quality filters
        if confidence < self.min_confidence:
            self.session_stats['low_confidence_filtered'] += 1
            return None

        cleaned_text = self.clean_and_normalize_text(text)
        if len(cleaned_text.replace(' ', '')) < 4:
            self.session_stats['false_positive_filtered'] += 1
            return None

        # Pattern validation
        is_valid_pattern = any(re.match(pattern, cleaned_text) for pattern in self.indonesia_patterns)
        if not is_valid_pattern and confidence < 0.7:  # Allow non-pattern jika confidence tinggi
            self.session_stats['false_positive_filtered'] += 1
            return None

        # Find matching track
        matching_track_id = self.find_matching_track(text, bbox, confidence)

        if matching_track_id:
            # Update existing track
            track = self.active_tracks[matching_track_id]
            track.add_detection(text, bbox, confidence)

            # Update normalization dengan detection terbaru
            track.clean_text = cleaned_text

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            return matching_track_id

        else:
            # Create new track
            new_track = PlateTrack(
                id=self.next_track_id,
                original_text=text,
                normalized_text=cleaned_text,
                clean_text=cleaned_text
            )
            new_track.add_detection(text, bbox, confidence)

            self.active_tracks[self.next_track_id] = new_track
            self.session_stats['stable_unique_plates'] += 1

            track_id = self.next_track_id
            self.next_track_id += 1

            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            return track_id

    def get_stable_counts(self) -> Dict[str, int]:
        """Get ultra-stable plate counts"""
        # Cleanup expired tracks
        expired_ids = []
        for track_id, track in self.active_tracks.items():
            if track.is_expired(self.temporal_window):
                expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.active_tracks[track_id]

        # Count stable tracks
        current_visible = len(self.active_tracks)
        stable_plates = len([t for t in self.active_tracks.values() if t.is_stable])
        high_confidence_plates = len([t for t in self.active_tracks.values() if t.get_average_confidence() > 0.7])

        return {
            'current_visible_plates': current_visible,
            'stable_confirmed_plates': stable_plates,
            'high_confidence_plates': high_confidence_plates,
            'total_unique_session': self.session_stats['stable_unique_plates'],
            'total_raw_processed': self.session_stats['total_raw_detections'],
            'low_quality_filtered': self.session_stats['low_confidence_filtered'],
            'false_positives_filtered': self.session_stats['false_positive_filtered'],
            'duplicates_filtered': self.session_stats['duplicate_filtered']
        }

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        counts = self.get_stable_counts()

        # Calculate advanced metrics
        total_processed = self.session_stats['total_raw_detections']
        if total_processed > 0:
            stability_rate = (counts['stable_confirmed_plates'] / counts['current_visible_plates'] * 100) if counts['current_visible_plates'] > 0 else 0
            efficiency_rate = (counts['total_unique_session'] / total_processed * 100)
            quality_filter_rate = ((self.session_stats['low_confidence_filtered'] + self.session_stats['false_positive_filtered']) / total_processed * 100)
        else:
            stability_rate = efficiency_rate = quality_filter_rate = 0.0

        # Performance metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0

        return {
            **counts,
            'stability_metrics': {
                'stability_rate_percent': round(stability_rate, 1),
                'efficiency_rate_percent': round(efficiency_rate, 1),
                'quality_filter_rate_percent': round(quality_filter_rate, 1)
            },
            'performance_metrics': {
                'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
                'total_tracks_created': self.next_track_id - 1,
                'active_tracks': len(self.active_tracks)
            },
            'configuration': {
                'text_similarity_threshold': self.text_similarity_threshold,
                'spatial_distance_threshold': self.spatial_distance_threshold,
                'temporal_window': self.temporal_window,
                'min_stability_score': self.min_stability_score,
                'min_confidence': self.min_confidence,
                'confirmation_detections': self.confirmation_detections
            }
        }

    def reset_session(self):
        """Reset untuk session baru"""
        self.active_tracks.clear()
        self.processing_times.clear()
        self.next_track_id = 1

        self.session_stats = {
            'total_raw_detections': 0,
            'stable_unique_plates': 0,
            'unstable_filtered': 0,
            'low_confidence_filtered': 0,
            'duplicate_filtered': 0,
            'false_positive_filtered': 0
        }

        self.logger.info("Session reset - StablePlateCounter ready")

def create_stable_plate_counter(config: Optional[Dict] = None) -> StablePlateCounter:
    """Factory untuk ultra-stable plate counter"""
    default_config = {
        'text_similarity_threshold': 0.75,  # Slightly lower untuk flexibility
        'spatial_distance_threshold': 100.0,  # Larger untuk Indonesian traffic
        'temporal_window': 4.0,  # 4 second window
        'min_stability_score': 0.6,
        'min_confidence': 0.35,  # Lower untuk handle Indonesian OCR
        'confirmation_detections': 3
    }

    if config:
        default_config.update(config)

    return StablePlateCounter(**default_config)