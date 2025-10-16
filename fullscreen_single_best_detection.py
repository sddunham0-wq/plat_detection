#!/usr/bin/env python3
"""
Full Screen Plate Detection - Mengembalikan 1 hasil terbaik saja
Comprehensive scanning dengan scoring system untuk mendapatkan deteksi plat nomor terbaik
"""

import cv2
import numpy as np
import sys
import os
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from utils.ocr_ensemble import OCREnsemble

@dataclass
class PlateCandidate:
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    text: str
    ocr_confidence: float
    visual_score: float
    combined_score: float
    roi: np.ndarray
    detection_method: str

class FullScreenPlateDetector:
    """Comprehensive full screen plate detector yang return 1 hasil terbaik"""

    def __init__(self):
        self.ocr_ensemble = OCREnsemble()
        self.logger = logging.getLogger(__name__)

    def detect_best_plate(self, image: np.ndarray) -> Optional[PlateCandidate]:
        """Deteksi plat nomor terbaik dari seluruh gambar"""

        print("🔍 Memulai full screen plate detection...")

        # Kumpulkan semua kandidat dari berbagai metode
        all_candidates = []

        # Method 1: Grid-based scanning
        grid_candidates = self._grid_scan_detection(image)
        all_candidates.extend(grid_candidates)
        print(f"   📊 Grid scan: {len(grid_candidates)} kandidat")

        # Method 2: Edge-based detection
        edge_candidates = self._edge_based_detection(image)
        all_candidates.extend(edge_candidates)
        print(f"   📊 Edge detection: {len(edge_candidates)} kandidat")

        # Method 3: Color-based detection (white/light regions)
        color_candidates = self._color_based_detection(image)
        all_candidates.extend(color_candidates)
        print(f"   📊 Color detection: {len(color_candidates)} kandidat")

        # Method 4: MSER text region detection
        mser_candidates = self._mser_detection(image)
        all_candidates.extend(mser_candidates)
        print(f"   📊 MSER detection: {len(mser_candidates)} kandidat")

        print(f"🎯 Total kandidat ditemukan: {len(all_candidates)}")

        if not all_candidates:
            print("❌ Tidak ada kandidat plat nomor ditemukan")
            return None

        # Remove overlapping candidates
        filtered_candidates = self._remove_overlapping_candidates(all_candidates)
        print(f"🔧 Setelah filter overlap: {len(filtered_candidates)} kandidat")

        # Score dan rank semua kandidat
        scored_candidates = self._score_and_rank_candidates(image, filtered_candidates)

        if not scored_candidates:
            print("❌ Tidak ada kandidat yang valid setelah scoring")
            return None

        # Return kandidat terbaik
        best_candidate = scored_candidates[0]
        print(f"🏆 Kandidat terbaik: '{best_candidate.text}' (score: {best_candidate.combined_score:.2f})")

        return best_candidate

    def _grid_scan_detection(self, image: np.ndarray) -> List[PlateCandidate]:
        """Scan gambar dengan grid pattern untuk mencari area plat nomor"""
        candidates = []
        height, width = image.shape[:2]

        # Berbagai ukuran window untuk scanning
        window_configs = [
            (0.15, 0.04),  # Small plate
            (0.20, 0.05),  # Medium plate
            (0.25, 0.06),  # Large plate
            (0.30, 0.08),  # Extra large plate
        ]

        for w_ratio, h_ratio in window_configs:
            window_w = int(width * w_ratio)
            window_h = int(height * h_ratio)

            # Step size (overlap untuk tidak miss detection)
            step_x = window_w // 3
            step_y = window_h // 2

            for y in range(0, height - window_h, step_y):
                for x in range(0, width - window_w, step_x):
                    roi = image[y:y+window_h, x:x+window_w]

                    # Quick visual assessment
                    visual_score = self._assess_visual_quality(roi)

                    if visual_score > 0.3:  # Threshold untuk visual quality
                        candidates.append(PlateCandidate(
                            bbox=(x, y, window_w, window_h),
                            text="",
                            ocr_confidence=0.0,
                            visual_score=visual_score,
                            combined_score=0.0,
                            roi=roi,
                            detection_method="grid_scan"
                        ))

        return candidates

    def _edge_based_detection(self, image: np.ndarray) -> List[PlateCandidate]:
        """Deteksi berbasis edge untuk mencari rectangular regions"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multi-threshold edge detection
        for low_thresh, high_thresh in [(50, 150), (75, 200), (100, 250)]:
            edges = cv2.Canny(gray, low_thresh, high_thresh)

            # Morphological operations untuk connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500 or area > 50000:  # Filter by reasonable size
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Indonesian license plates aspect ratio
                if 2.0 <= aspect_ratio <= 6.0:
                    roi = image[y:y+h, x:x+w]
                    visual_score = self._assess_visual_quality(roi)

                    candidates.append(PlateCandidate(
                        bbox=(x, y, w, h),
                        text="",
                        ocr_confidence=0.0,
                        visual_score=visual_score,
                        combined_score=0.0,
                        roi=roi,
                        detection_method="edge_detection"
                    ))

        return candidates

    def _color_based_detection(self, image: np.ndarray) -> List[PlateCandidate]:
        """Deteksi berbasis warna untuk mencari region putih/terang"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Multiple threshold untuk light regions
        for threshold in [120, 140, 160, 180, 200]:
            _, light_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            morph = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 400 or area > 40000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                if 2.0 <= aspect_ratio <= 6.0:
                    roi = image[y:y+h, x:x+w]
                    visual_score = self._assess_visual_quality(roi)

                    candidates.append(PlateCandidate(
                        bbox=(x, y, w, h),
                        text="",
                        ocr_confidence=0.0,
                        visual_score=visual_score,
                        combined_score=0.0,
                        roi=roi,
                        detection_method="color_based"
                    ))

        return candidates

    def _mser_detection(self, image: np.ndarray) -> List[PlateCandidate]:
        """MSER detection untuk text-like regions"""
        candidates = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create MSER detector
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        for region in regions:
            if len(region) < 50:  # Too few points
                continue

            x, y, w, h = cv2.boundingRect(region)
            area = w * h

            if area < 300 or area > 30000:
                continue

            aspect_ratio = w / h if h > 0 else 0

            if 2.0 <= aspect_ratio <= 6.0:
                roi = image[y:y+h, x:x+w]
                visual_score = self._assess_visual_quality(roi)

                candidates.append(PlateCandidate(
                    bbox=(x, y, w, h),
                    text="",
                    ocr_confidence=0.0,
                    visual_score=visual_score,
                    combined_score=0.0,
                    roi=roi,
                    detection_method="mser"
                ))

        return candidates

    def _assess_visual_quality(self, roi: np.ndarray) -> float:
        """Assess visual quality ROI untuk preliminary filtering"""
        if roi.size == 0:
            return 0.0

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi

        # Calculate various quality metrics
        scores = []

        # 1. Contrast score
        contrast = gray_roi.std()
        contrast_score = min(1.0, contrast / 50.0)  # Normalize
        scores.append(contrast_score)

        # 2. Edge density score
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = edges.sum() / (gray_roi.shape[0] * gray_roi.shape[1] * 255)
        edge_score = min(1.0, edge_density * 10)  # Scale
        scores.append(edge_score)

        # 3. Size score (prefer medium sizes)
        area = gray_roi.shape[0] * gray_roi.shape[1]
        ideal_area = 3000  # pixels
        size_score = 1.0 - abs(area - ideal_area) / ideal_area
        size_score = max(0.0, size_score)
        scores.append(size_score)

        # Combined score
        return sum(scores) / len(scores)

    def _remove_overlapping_candidates(self, candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Remove overlapping candidates, keep the one with higher visual score"""
        if len(candidates) <= 1:
            return candidates

        # Sort by visual score descending
        candidates.sort(key=lambda x: x.visual_score, reverse=True)

        filtered = []
        for candidate in candidates:
            x1, y1, w1, h1 = candidate.bbox

            # Check overlap with existing filtered candidates
            is_overlapping = False
            for existing in filtered:
                x2, y2, w2, h2 = existing.bbox

                # Calculate intersection
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area1 = w1 * h1
                    area2 = w2 * h2

                    # If intersection > 30% of either area, consider overlapping
                    if inter_area > 0.3 * min(area1, area2):
                        is_overlapping = True
                        break

            if not is_overlapping:
                filtered.append(candidate)

        return filtered

    def _score_and_rank_candidates(self, image: np.ndarray, candidates: List[PlateCandidate]) -> List[PlateCandidate]:
        """Score dan rank semua kandidat, return sorted by score"""
        scored_candidates = []

        print(f"🔍 Scoring {len(candidates)} kandidat...")

        for i, candidate in enumerate(candidates):
            # Enhanced ROI untuk OCR
            enhanced_roi = self._enhance_roi_for_ocr(candidate.roi)

            # Run OCR
            try:
                text, ocr_conf, details = self.ocr_ensemble.ensemble_ocr(enhanced_roi)

                # Clean text
                if text:
                    clean_text = ''.join(c for c in text if c.isalnum() or c.isspace()).strip()
                    # Validate text length untuk plat Indonesia
                    if len(clean_text.replace(' ', '')) < 4 or len(clean_text.replace(' ', '')) > 15:
                        ocr_conf *= 0.5  # Penalty untuk length yang tidak wajar
                else:
                    clean_text = ""
                    ocr_conf = 0.0

            except Exception as e:
                clean_text = ""
                ocr_conf = 0.0

            # Combined scoring
            combined_score = self._calculate_combined_score(
                candidate.visual_score, ocr_conf, candidate.bbox, clean_text
            )

            # Update candidate
            candidate.text = clean_text
            candidate.ocr_confidence = ocr_conf
            candidate.combined_score = combined_score

            if combined_score > 0.2:  # Minimum threshold
                scored_candidates.append(candidate)
                print(f"   Kandidat {i+1}: '{clean_text}' (OCR: {ocr_conf:.1f}%, Combined: {combined_score:.2f})")

        # Sort by combined score descending
        scored_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        return scored_candidates

    def _enhance_roi_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Enhance ROI untuk OCR yang lebih baik"""
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Resize untuk OCR optimal
        target_height = 64
        if gray.shape[0] < target_height:
            scale = target_height / gray.shape[0]
            new_width = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Apply CLAHE untuk better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)

        # Convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _calculate_combined_score(self, visual_score: float, ocr_conf: float,
                                bbox: Tuple[int, int, int, int], text: str) -> float:
        """Calculate combined score dari berbagai faktor"""

        x, y, w, h = bbox
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Component scores
        scores = {
            'visual': visual_score * 0.25,  # 25%
            'ocr': (ocr_conf / 100.0) * 0.35,  # 35%
            'aspect': self._score_aspect_ratio(aspect_ratio) * 0.15,  # 15%
            'size': self._score_size(area) * 0.15,  # 15%
            'text_quality': self._score_text_quality(text) * 0.10  # 10%
        }

        return sum(scores.values())

    def _score_aspect_ratio(self, aspect_ratio: float) -> float:
        """Score aspect ratio (Indonesian plates: ~3:1 to 4:1)"""
        ideal_ratio = 3.5
        deviation = abs(aspect_ratio - ideal_ratio) / ideal_ratio
        return max(0.0, 1.0 - deviation)

    def _score_size(self, area: int) -> float:
        """Score size (prefer medium sizes)"""
        ideal_area = 4000
        if area == 0:
            return 0.0
        deviation = abs(area - ideal_area) / ideal_area
        return max(0.0, 1.0 - deviation)

    def _score_text_quality(self, text: str) -> float:
        """Score text quality berdasarkan karakteristik plat Indonesia"""
        if not text:
            return 0.0

        clean_text = text.replace(' ', '')

        # Length score
        if 5 <= len(clean_text) <= 10:  # Typical Indonesian plate length
            length_score = 1.0
        else:
            length_score = 0.5

        # Character composition score
        has_letters = any(c.isalpha() for c in clean_text)
        has_numbers = any(c.isdigit() for c in clean_text)

        if has_letters and has_numbers:
            composition_score = 1.0
        elif has_letters or has_numbers:
            composition_score = 0.7
        else:
            composition_score = 0.0

        return (length_score + composition_score) / 2.0

def draw_best_detection(image: np.ndarray, best_candidate: PlateCandidate) -> np.ndarray:
    """Draw bounding box dan label untuk deteksi terbaik"""
    result = image.copy()

    x, y, w, h = best_candidate.bbox

    # Draw bounding box (hijau tebal)
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Prepare label
    if best_candidate.text:
        label = f"{best_candidate.text} ({best_candidate.combined_score:.2f})"
    else:
        label = f"Detected ({best_candidate.combined_score:.2f})"

    # Draw label dengan background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Position label
    label_x = x
    label_y = y - 15

    if label_y - text_height < 0:
        label_y = y + h + text_height + 15

    # Background rectangle
    cv2.rectangle(result,
                 (label_x - 5, label_y - text_height - 5),
                 (label_x + text_width + 10, label_y + baseline + 5),
                 (0, 0, 0), -1)

    # White text
    cv2.putText(result, label, (label_x, label_y),
               font, font_scale, (255, 255, 255), font_thickness)

    # Arrow pointing to plate
    arrow_start = (label_x + text_width // 2, label_y + 10)
    arrow_end = (x + w // 2, y + 5)
    cv2.arrowedLine(result, arrow_start, arrow_end, (0, 255, 0), 3)

    return result

def main():
    """Main function"""

    print("=" * 80)
    print("🚗 FULL SCREEN PLATE DETECTION - SINGLE BEST RESULT")
    print("=" * 80)

    # Input dan output
    input_path = "contoh/15122022plat.jpg"
    output_path = "contoh/15122022plat_fullscreen_single_best.jpg"

    # Load image
    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Tidak dapat memuat gambar: {input_path}")
        return

    print(f"📷 Memproses gambar: {image.shape}")

    # Initialize detector
    detector = FullScreenPlateDetector()

    # Detect best plate
    best_candidate = detector.detect_best_plate(image)

    if best_candidate:
        # Draw detection
        result_image = draw_best_detection(image, best_candidate)

        # Save result
        cv2.imwrite(output_path, result_image)

        # Print results
        print("\n" + "=" * 80)
        print("🏆 HASIL DETEKSI TERBAIK")
        print("=" * 80)
        print(f"📋 Plat nomor: '{best_candidate.text}'")
        print(f"📊 OCR Confidence: {best_candidate.ocr_confidence:.1f}%")
        print(f"🎯 Visual Score: {best_candidate.visual_score:.3f}")
        print(f"⭐ Combined Score: {best_candidate.combined_score:.3f}")
        print(f"📍 Bounding Box: x={best_candidate.bbox[0]}, y={best_candidate.bbox[1]}")
        print(f"📏 Ukuran: {best_candidate.bbox[2]}x{best_candidate.bbox[3]} pixels")
        print(f"🔍 Method: {best_candidate.detection_method}")
        print(f"💾 Output: {output_path}")
        print("=" * 80)
        print("✅ DETEKSI BERHASIL!")
        print("=" * 80)

    else:
        print("\n❌ Tidak ditemukan plat nomor yang valid")

if __name__ == "__main__":
    main()