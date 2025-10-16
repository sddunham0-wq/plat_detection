#!/usr/bin/env python3
"""
Production-Ready Optimized Plate Detector untuk Closer Camera
Fast and effective detector for closer camera distances
"""

import cv2
import numpy as np
import logging
from typing import List
from utils.robust_plate_detector import RobustPlateDetector, PlateDetection

class OptimizedPlateDetector(RobustPlateDetector):
    """
    Production optimized detector untuk closer camera
    """

    def __init__(self):
        # Initialize as full mode but with optimized parameters
        super().__init__(streaming_mode=False)

        # Override dengan production-optimized parameters
        self.min_area = 300          # Catch smaller plates
        self.max_area = 25000        # Full range
        self.min_aspect_ratio = 1.5  # Full range
        self.max_aspect_ratio = 6.0  # Full range
        self.min_width = 25          # Permissive
        self.max_width = 400         # Full range
        self.min_height = 10         # Permissive
        self.max_height = 150        # Full range
        self.min_confidence = 8      # Very low threshold
        self.min_text_likelihood = 15 # Very low threshold
        self.max_candidates = 20     # More coverage

        # Very permissive texture
        self.min_edge_density = 0.8
        self.min_texture_variance = 2

        self.logger.info("🚀 Production Optimized Detector ready")

# Replace the detector in hybrid_plate_detector
def patch_hybrid_detector():
    """
    Patch HybridPlateDetector to use optimized detector
    """
    from utils import hybrid_plate_detector

    # Replace the robust detector with optimized version
    original_init = hybrid_plate_detector.HybridPlateDetector.__init__

    def new_init(self, streaming_mode=True):
        original_init(self, streaming_mode)
        # Replace with optimized detector
        self.plate_detector = OptimizedPlateDetector()
        self.logger.info("🎯 Hybrid detector using optimized plate detector")

    hybrid_plate_detector.HybridPlateDetector.__init__ = new_init

    return hybrid_plate_detector.HybridPlateDetector

# Auto-patch on import
if __name__ != "__main__":
    patch_hybrid_detector()