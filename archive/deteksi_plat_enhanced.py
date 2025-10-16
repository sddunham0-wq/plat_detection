import cv2
import pytesseract
import os
import logging
import time
import numpy as np
from datetime import datetime
from PIL import Image
import mysql.connector
from typing import Optional, Tuple, List, Dict

# Import konfigurasi dan validator yang sudah kita buat
from config import config
from utils.plate_validator import plate_validator

# Buat folder yang diperlukan terlebih dahulu
def ensure_directories():
    """Pastikan semua direktori yang diperlukan sudah ada"""
    directories = ['logs', config.SAVE_FOLDER]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Folder '{directory}' berhasil dibuat")

# Pastikan direktori ada sebelum setup logging
ensure_directories()

# Setup logging setelah folder sudah pasti ada
try:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/plate_detection.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    # Fallback ke console logging saja jika ada masalah
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Tidak bisa setup file logging: {e}")

class EnhancedPlateDetector:
    """
    Sistem deteksi plat nomor yang telah ditingkatkan dengan:
    - Preprocessing gambar yang lebih baik
    - Validasi format plat Indonesia
    - Error handling yang robust
    - Confidence scoring
    - Auto-reconnect untuk kamera
    """

    def __init__(self):
        self.camera = None
        self.retry_count = 0
        self.max_retries = config.MAX_RETRY_CAMERA
        self.detection_stats = {
            'total_frames': 0,
            'detections': 0,
            'valid_plates': 0,
            'false_positives': 0
        }

        # Folder sudah dipastikan ada oleh ensure_directories()

        logger.info("🚀 Enhanced Plate Detector dimulai")
        logger.info(f"📁 Save folder: {config.SAVE_FOLDER}")
        logger.info(f"📹 Camera URL: {config.CAMERA_URL}")

    def connect_camera(self) -> bool:
        """
        Koneksi ke kamera dengan retry mechanism
        """
        try:
            if self.camera is not None:
                self.camera.release()

            logger.info(f"🔌 Mencoba koneksi ke kamera... (percobaan {self.retry_count + 1})")
            self.camera = cv2.VideoCapture(config.CAMERA_URL)

            if self.camera.isOpened():
                # Test baca frame
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    logger.info("✅ Kamera berhasil terhubung")
                    self.retry_count = 0
                    return True
                else:
                    logger.warning("⚠️ Kamera terhubung tapi tidak bisa baca frame")
                    return False
            else:
                logger.error("❌ Gagal koneksi ke kamera")
                return False

        except Exception as e:
            logger.error(f"❌ Error koneksi kamera: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing gambar untuk meningkatkan akurasi deteksi
        """
        # 1. Konversi ke grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian blur untuk mengurangi noise
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)

        # 3. Histogram equalization untuk meningkatkan kontras
        equalized = cv2.equalizeHist(blurred)

        # 4. Morphological operations untuk cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE)
        processed = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)

        return processed

    def detect_plate_contours(self, processed_image: np.ndarray) -> List[Tuple]:
        """
        Deteksi kontur yang mungkin adalah plat nomor
        """
        # Edge detection dengan parameter yang dapat dikonfigurasi
        edges = cv2.Canny(processed_image, config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2)

        # Cari kontur
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plate_candidates = []

        for contour in contours:
            # Approximate contour to polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), closed=True)

            # Filter berdasarkan jumlah titik (plat biasanya 4 sudut)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter berdasarkan ukuran minimum
                if w >= config.MIN_PLATE_WIDTH and h >= config.MIN_PLATE_HEIGHT:
                    # Filter berdasarkan aspect ratio
                    aspect_ratio = w / h
                    if config.MIN_ASPECT_RATIO <= aspect_ratio <= config.MAX_ASPECT_RATIO:
                        # Hitung area dan solidity untuk filter tambahan
                        area = cv2.contourArea(contour)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0

                        # Filter berdasarkan solidity (objek yang solid seperti plat)
                        if solidity > 0.3:
                            plate_candidates.append((x, y, w, h, area, solidity, aspect_ratio))

        # Sort berdasarkan area (yang terbesar dulu)
        plate_candidates.sort(key=lambda x: x[4], reverse=True)

        return plate_candidates

    def extract_text_from_roi(self, image: np.ndarray, bbox: Tuple) -> Dict:
        """
        Ekstrak teks dari region of interest dengan preprocessing tambahan
        """
        x, y, w, h = bbox[:4]

        # Crop area plat
        roi = image[y:y+h, x:x+w]

        if roi.size == 0:
            return {'text': '', 'confidence': 0.0}

        # Resize untuk OCR yang lebih baik (jika terlalu kecil)
        if roi.shape[1] < 200:
            scale_factor = 200 / roi.shape[1]
            new_width = int(roi.shape[1] * scale_factor)
            new_height = int(roi.shape[0] * scale_factor)
            roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Convert ke grayscale jika belum
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi

        # Adaptive threshold untuk teks yang lebih clear
        roi_thresh = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # OCR dengan konfigurasi optimized
        try:
            # OCR dengan confidence data
            ocr_data = pytesseract.image_to_data(
                roi_thresh,
                config=config.OCR_CONFIG,
                output_type=pytesseract.Output.DICT
            )

            # Ekstrak teks dengan confidence tinggi
            text_parts = []
            confidences = []

            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])

                if text and conf > 30:  # Hanya ambil yang confidence > 30
                    text_parts.append(text)
                    confidences.append(conf)

            # Gabungkan teks
            full_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

            return {
                'text': full_text.strip().upper(),
                'confidence': avg_confidence,
                'roi': roi
            }

        except Exception as e:
            logger.error(f"❌ Error OCR: {e}")
            return {'text': '', 'confidence': 0.0}

    def save_detection(self, plate_text: str, roi_image: np.ndarray, confidence: float) -> Optional[str]:
        """
        Simpan hasil deteksi ke database dan file
        """
        try:
            # Buat timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"plat_{timestamp}.jpg"
            filepath = os.path.join(config.SAVE_FOLDER, filename)

            # Simpan gambar
            cv2.imwrite(filepath, roi_image)

            # Simpan ke database
            self.save_to_database(plate_text, filepath, confidence)

            logger.info(f"💾 Deteksi disimpan: {plate_text} -> {filename}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Error simpan detection: {e}")
            return None

    def save_to_database(self, plate_text: str, filepath: str, confidence: float):
        """
        Simpan data ke database dengan error handling
        """
        try:
            conn = mysql.connector.connect(
                host=config.DB_HOST,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME
            )

            cursor = conn.cursor()

            # Query dengan confidence score
            query = """
            INSERT INTO tb_kendaraan (Plat_Nomor, Foto_Plat, Confidence_Score, Waktu_Deteksi)
            VALUES (%s, %s, %s, NOW())
            """

            cursor.execute(query, (plate_text, filepath, confidence))
            conn.commit()

            logger.info(f"📊 Data tersimpan ke database: {plate_text} (confidence: {confidence:.2f})")

        except mysql.connector.Error as e:
            logger.error(f"❌ Error database: {e}")
        except Exception as e:
            logger.error(f"❌ Error database (general): {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Proses satu frame untuk deteksi plat
        """
        self.detection_stats['total_frames'] += 1

        # Preprocessing
        processed = self.preprocess_image(frame)

        # Deteksi kandidat plat
        candidates = self.detect_plate_contours(processed)

        results = []

        for candidate in candidates[:3]:  # Maksimal 3 kandidat terbaik
            bbox = candidate[:4]
            area = candidate[4]
            solidity = candidate[5]
            aspect_ratio = candidate[6]

            # Ekstrak teks
            ocr_result = self.extract_text_from_roi(frame, bbox)
            plate_text = ocr_result['text']
            ocr_confidence = ocr_result['confidence']

            if plate_text:
                self.detection_stats['detections'] += 1

                # Validasi dengan sistem validator Indonesia
                validation = plate_validator.validate_plate(plate_text, config.MIN_CONFIDENCE)

                if validation['is_valid']:
                    self.detection_stats['valid_plates'] += 1

                    # Hitung confidence gabungan
                    final_confidence = (ocr_confidence + validation['confidence']) / 2

                    result = {
                        'text': validation['cleaned_text'],
                        'bbox': bbox,
                        'confidence': final_confidence,
                        'validation': validation,
                        'ocr_confidence': ocr_confidence,
                        'area': area,
                        'solidity': solidity,
                        'aspect_ratio': aspect_ratio
                    }

                    results.append(result)

                    # Simpan jika confidence cukup tinggi
                    if final_confidence >= config.MIN_CONFIDENCE and 'roi' in ocr_result:
                        self.save_detection(validation['cleaned_text'], ocr_result['roi'], final_confidence)

                    logger.info(
                        f"✅ Plat terdeteksi: {validation['cleaned_text']} "
                        f"(confidence: {final_confidence:.2f})"
                    )
                else:
                    self.detection_stats['false_positives'] += 1
                    logger.debug(f"❌ False positive: {plate_text} (confidence: {validation['confidence']:.2f})")

        return {
            'results': results,
            'frame_processed': processed,
            'candidates_found': len(candidates)
        }

    def draw_detections(self, frame: np.ndarray, results: List[Dict]) -> np.ndarray:
        """
        Gambar hasil deteksi di frame
        """
        annotated_frame = frame.copy()

        for result in results:
            x, y, w, h = result['bbox']
            confidence = result['confidence']
            text = result['text']

            # Warna berdasarkan confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Hijau untuk confidence tinggi
            elif confidence >= 0.6:
                color = (0, 255, 255)  # Kuning untuk confidence sedang
            else:
                color = (0, 165, 255)  # Orange untuk confidence rendah

            # Gambar rectangle
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)

            # Gambar teks
            label = f"{text} ({confidence:.2f})"
            cv2.putText(
                annotated_frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        return annotated_frame

    def print_stats(self):
        """
        Print statistik deteksi
        """
        stats = self.detection_stats
        if stats['total_frames'] > 0:
            detection_rate = (stats['detections'] / stats['total_frames']) * 100
            accuracy_rate = (stats['valid_plates'] / max(stats['detections'], 1)) * 100

            logger.info(f"""
            📊 STATISTIK DETEKSI:
            - Total frames: {stats['total_frames']}
            - Total deteksi: {stats['detections']}
            - Plat valid: {stats['valid_plates']}
            - False positives: {stats['false_positives']}
            - Detection rate: {detection_rate:.1f}%
            - Accuracy rate: {accuracy_rate:.1f}%
            """)

    def run(self):
        """
        Main loop untuk menjalankan deteksi
        """
        logger.info("🎬 Memulai deteksi plat nomor...")

        last_stats_time = time.time()

        while True:
            try:
                # Cek koneksi kamera
                if self.camera is None or not self.camera.isOpened():
                    if not self.connect_camera():
                        self.retry_count += 1
                        if self.retry_count >= self.max_retries:
                            logger.error(f"❌ Gagal koneksi setelah {self.max_retries} percobaan")
                            break

                        logger.info(f"⏳ Menunggu {self.retry_count * 2} detik sebelum retry...")
                        time.sleep(self.retry_count * 2)
                        continue

                # Baca frame
                ret, frame = self.camera.read()

                if not ret or frame is None:
                    logger.warning("⚠️ Gagal baca frame, reconnecting...")
                    self.camera = None
                    continue

                # Process frame
                process_result = self.process_frame(frame)

                # Draw detections
                if process_result['results']:
                    annotated_frame = self.draw_detections(frame, process_result['results'])
                else:
                    annotated_frame = frame

                # Tampilkan frame
                cv2.imshow('Enhanced Plate Detection', annotated_frame)

                # Print stats setiap 30 detik
                current_time = time.time()
                if current_time - last_stats_time >= 30:
                    self.print_stats()
                    last_stats_time = current_time

                # Check untuk exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("👋 Program dihentikan oleh user")
                    break

            except KeyboardInterrupt:
                logger.info("👋 Program dihentikan oleh user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"❌ Error dalam main loop: {e}")
                time.sleep(1)

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """
        Cleanup resources
        """
        if self.camera is not None:
            self.camera.release()

        cv2.destroyAllWindows()
        self.print_stats()
        logger.info("🧹 Cleanup selesai")

def main():
    """
    Main function
    """
    try:
        detector = EnhancedPlateDetector()
        detector.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        logger.info("👋 Program selesai")

if __name__ == "__main__":
    main()