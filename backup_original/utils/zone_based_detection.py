"""
Zone-Based Motorcycle Detection
Optimized untuk jarak jauh dengan pembagian area deteksi
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from .motorcycle_plate_detector import MotorcyclePlateDetector, PlateDetection
from .yolo_detector import YOLOObjectDetector

@dataclass
class ZoneConfig:
    """Configuration untuk zone detection"""
    zone_id: int
    zone_name: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence_threshold: float
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low

class ZoneBasedMotorcycleDetector:
    """
    Zone-based motorcycle detection untuk area yang luas
    Membagi frame menjadi beberapa zone untuk detection yang lebih fokus
    """
    
    def __init__(self, frame_width=1280, frame_height=720):
        """
        Initialize zone-based detector
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
        """
        self.logger = logging.getLogger(__name__)
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Initialize base detector
        self.motorcycle_detector = MotorcyclePlateDetector(confidence=0.3)
        
        # Zone configuration
        self.zones = []
        self.current_active_zone = 0
        self.zone_switch_interval = 2.0  # seconds
        self.last_zone_switch = time.time()
        
        # Detection modes
        self.mode = "sequential"  # "sequential", "parallel", "adaptive"
        self.adaptive_threshold = 0.5  # untuk adaptive mode
        
        # Statistics
        self.zone_stats = {}
        self.total_detections_per_zone = {}
        
        # Auto-configure zones untuk parking lot layout
        self.setup_default_zones()
        
        self.logger.info("Zone-based motorcycle detector initialized")
    
    def setup_default_zones(self):
        """Setup default zones untuk parking lot seperti screenshot"""
        # Clear existing zones
        self.zones = []
        
        # Zone layout untuk parking lot (3x3 grid optimal)
        zone_configs = [
            # Row 1 (top)
            ZoneConfig(0, "Top-Left", self._calculate_zone_bbox(0, 0, 3, 3), 0.25, True, 1),
            ZoneConfig(1, "Top-Center", self._calculate_zone_bbox(1, 0, 3, 3), 0.3, True, 1),
            ZoneConfig(2, "Top-Right", self._calculate_zone_bbox(2, 0, 3, 3), 0.25, True, 1),
            
            # Row 2 (middle) - highest priority
            ZoneConfig(3, "Mid-Left", self._calculate_zone_bbox(0, 1, 3, 3), 0.2, True, 1),
            ZoneConfig(4, "Mid-Center", self._calculate_zone_bbox(1, 1, 3, 3), 0.2, True, 1),
            ZoneConfig(5, "Mid-Right", self._calculate_zone_bbox(2, 1, 3, 3), 0.2, True, 1),
            
            # Row 3 (bottom) - closest to camera
            ZoneConfig(6, "Bot-Left", self._calculate_zone_bbox(0, 2, 3, 3), 0.15, True, 1),
            ZoneConfig(7, "Bot-Center", self._calculate_zone_bbox(1, 2, 3, 3), 0.15, True, 1),
            ZoneConfig(8, "Bot-Right", self._calculate_zone_bbox(2, 2, 3, 3), 0.15, True, 1),
        ]
        
        self.zones = zone_configs
        
        # Initialize statistics
        for zone in self.zones:
            self.zone_stats[zone.zone_id] = {
                'detections': 0,
                'motorcycles': 0,
                'plates_read': 0,
                'last_detection': 0,
                'avg_confidence': 0.0
            }
            self.total_detections_per_zone[zone.zone_id] = 0
        
        self.logger.info(f"Setup {len(self.zones)} detection zones")
    
    def _calculate_zone_bbox(self, col, row, total_cols, total_rows):
        """Calculate bounding box untuk zone tertentu"""
        zone_width = self.frame_width // total_cols
        zone_height = self.frame_height // total_rows
        
        x = col * zone_width
        y = row * zone_height
        w = zone_width
        h = zone_height
        
        return (x, y, w, h)
    
    def set_detection_mode(self, mode: str):
        """Set detection mode"""
        if mode in ["sequential", "parallel", "adaptive"]:
            self.mode = mode
            self.logger.info(f"Detection mode set to: {mode}")
        else:
            self.logger.warning(f"Invalid mode: {mode}")
    
    def detect_in_zones(self, frame: np.ndarray) -> Dict[int, List[PlateDetection]]:
        """
        Detect motorcycles in configured zones
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary mapping zone_id to detection results
        """
        results = {}
        
        if self.mode == "sequential":
            results = self._detect_sequential(frame)
        elif self.mode == "parallel":
            results = self._detect_parallel(frame)
        elif self.mode == "adaptive":
            results = self._detect_adaptive(frame)
        
        # Update statistics
        self._update_zone_statistics(results)
        
        return results
    
    def _detect_sequential(self, frame: np.ndarray) -> Dict[int, List[PlateDetection]]:
        """Sequential zone detection - satu zone per frame"""
        current_time = time.time()
        
        # Switch zone berdasarkan interval
        if current_time - self.last_zone_switch >= self.zone_switch_interval:
            self._switch_to_next_zone()
            self.last_zone_switch = current_time
        
        results = {}
        current_zone = self.zones[self.current_active_zone]
        
        if current_zone.enabled:
            zone_results = self._detect_in_zone(frame, current_zone)
            if zone_results:
                results[current_zone.zone_id] = zone_results
        
        return results
    
    def _detect_parallel(self, frame: np.ndarray) -> Dict[int, List[PlateDetection]]:
        """Parallel zone detection - semua zone sekaligus"""
        results = {}
        
        for zone in self.zones:
            if zone.enabled:
                zone_results = self._detect_in_zone(frame, zone)
                if zone_results:
                    results[zone.zone_id] = zone_results
        
        return results
    
    def _detect_adaptive(self, frame: np.ndarray) -> Dict[int, List[PlateDetection]]:
        """Adaptive zone detection - prioritas berdasarkan aktivitas"""
        results = {}
        
        # Sort zones berdasarkan priority dan recent activity
        sorted_zones = sorted(
            self.zones,
            key=lambda z: (z.priority, -self.zone_stats[z.zone_id]['detections'])
        )
        
        # Detect di high-priority zones atau zones dengan aktivitas tinggi
        for zone in sorted_zones[:5]:  # Top 5 zones
            if zone.enabled:
                zone_results = self._detect_in_zone(frame, zone)
                if zone_results:
                    results[zone.zone_id] = zone_results
                    # Boost priority untuk zone dengan detection
                    self.zone_stats[zone.zone_id]['last_detection'] = time.time()
        
        return results
    
    def _detect_in_zone(self, frame: np.ndarray, zone: ZoneConfig) -> List[PlateDetection]:
        """Detect motorcycles dalam zone spesifik"""
        try:
            # Extract zone dari frame
            x, y, w, h = zone.bbox
            zone_frame = frame[y:y+h, x:x+w]
            
            if zone_frame.size == 0:
                return []
            
            # Set confidence untuk zone ini
            original_confidence = self.motorcycle_detector.yolo_detector.confidence
            self.motorcycle_detector.yolo_detector.set_confidence(zone.confidence_threshold)
            
            # Detect di zone
            zone_results = self.motorcycle_detector.detect_motorcycles_and_plates(zone_frame)
            
            # Adjust coordinates ke full frame
            adjusted_results = []
            for result in zone_results:
                # Adjust vehicle bbox
                vx, vy, vw, vh = result.vehicle_detection.bbox
                adjusted_vehicle_bbox = (vx + x, vy + y, vw, vh)
                result.vehicle_detection.bbox = adjusted_vehicle_bbox
                
                # Adjust plate bboxes
                for plate in result.plate_detections:
                    px, py, pw, ph = plate.bbox
                    plate.bbox = (px + x, py + y, pw, ph)
                
                adjusted_results.append(result)
            
            # Restore original confidence
            self.motorcycle_detector.yolo_detector.set_confidence(original_confidence)
            
            return adjusted_results
            
        except Exception as e:
            self.logger.error(f"Error detecting in zone {zone.zone_id}: {str(e)}")
            return []
    
    def _switch_to_next_zone(self):
        """Switch ke zone berikutnya untuk sequential mode"""
        enabled_zones = [z for z in self.zones if z.enabled]
        if not enabled_zones:
            return
            
        # Find current zone index in enabled zones
        current_zone_id = self.zones[self.current_active_zone].zone_id
        enabled_zone_ids = [z.zone_id for z in enabled_zones]
        
        try:
            current_idx = enabled_zone_ids.index(current_zone_id)
            next_idx = (current_idx + 1) % len(enabled_zone_ids)
            next_zone_id = enabled_zone_ids[next_idx]
            
            # Find actual zone index
            for i, zone in enumerate(self.zones):
                if zone.zone_id == next_zone_id:
                    self.current_active_zone = i
                    break
                    
        except ValueError:
            self.current_active_zone = 0
        
        self.logger.debug(f"Switched to zone {self.zones[self.current_active_zone].zone_name}")
    
    def _update_zone_statistics(self, results: Dict[int, List[PlateDetection]]):
        """Update statistics untuk setiap zone"""
        for zone_id, zone_results in results.items():
            stats = self.zone_stats[zone_id]
            stats['detections'] += len(zone_results)
            stats['motorcycles'] += len(zone_results)
            
            # Count plates
            total_plates = sum(len(r.plate_detections) for r in zone_results)
            stats['plates_read'] += total_plates
            
            # Update average confidence
            if zone_results:
                confidences = [r.combined_confidence for r in zone_results]
                stats['avg_confidence'] = np.mean(confidences)
    
    def draw_zone_overlay(self, frame: np.ndarray, show_stats: bool = True) -> np.ndarray:
        """Draw zone overlay pada frame"""
        overlay_frame = frame.copy()
        
        for zone in self.zones:
            x, y, w, h = zone.bbox
            
            # Zone border color berdasarkan status
            if zone.zone_id == self.zones[self.current_active_zone].zone_id and self.mode == "sequential":
                color = (0, 255, 255)  # Yellow untuk active zone
                thickness = 3
            elif zone.enabled:
                color = (0, 255, 0)    # Green untuk enabled
                thickness = 2
            else:
                color = (128, 128, 128)  # Gray untuk disabled
                thickness = 1
            
            # Draw zone rectangle
            cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Zone label
            label = f"Z{zone.zone_id}"
            if self.mode == "sequential" and zone.zone_id == self.zones[self.current_active_zone].zone_id:
                label += " (ACTIVE)"
            
            cv2.putText(overlay_frame, label, (x + 5, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show statistics jika diminta
            if show_stats:
                stats = self.zone_stats[zone.zone_id]
                stats_text = f"M:{stats['motorcycles']} P:{stats['plates_read']}"
                cv2.putText(overlay_frame, stats_text, (x + 5, y + h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Mode indicator
        mode_text = f"Mode: {self.mode.upper()}"
        cv2.putText(overlay_frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay_frame
    
    def get_zone_statistics(self) -> Dict:
        """Get comprehensive zone statistics"""
        total_detections = sum(stats['detections'] for stats in self.zone_stats.values())
        total_motorcycles = sum(stats['motorcycles'] for stats in self.zone_stats.values())
        total_plates = sum(stats['plates_read'] for stats in self.zone_stats.values())
        
        return {
            'total_detections': total_detections,
            'total_motorcycles': total_motorcycles,
            'total_plates_read': total_plates,
            'active_zones': len([z for z in self.zones if z.enabled]),
            'detection_mode': self.mode,
            'current_active_zone': self.zones[self.current_active_zone].zone_name if self.zones else None,
            'zone_breakdown': self.zone_stats,
            'plate_success_rate': (total_plates / total_motorcycles * 100) if total_motorcycles > 0 else 0
        }
    
    def configure_zone(self, zone_id: int, enabled: bool = None, confidence: float = None, priority: int = None):
        """Configure specific zone"""
        for zone in self.zones:
            if zone.zone_id == zone_id:
                if enabled is not None:
                    zone.enabled = enabled
                if confidence is not None:
                    zone.confidence_threshold = confidence
                if priority is not None:
                    zone.priority = priority
                
                self.logger.info(f"Zone {zone_id} configured: enabled={zone.enabled}, conf={zone.confidence_threshold}")
                break
    
    def reset_statistics(self):
        """Reset all zone statistics"""
        for zone_id in self.zone_stats:
            self.zone_stats[zone_id] = {
                'detections': 0,
                'motorcycles': 0,
                'plates_read': 0,
                'last_detection': 0,
                'avg_confidence': 0.0
            }
        self.logger.info("Zone statistics reset")

def create_zone_detector(frame_width=1280, frame_height=720) -> ZoneBasedMotorcycleDetector:
    """Factory function untuk create zone-based detector"""
    return ZoneBasedMotorcycleDetector(frame_width, frame_height)