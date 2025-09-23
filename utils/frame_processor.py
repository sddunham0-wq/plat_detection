"""
Frame Processor untuk Live Video Processing
Multi-threading processing untuk performance optimal
"""

import cv2
import threading
import queue
import time
import logging
from typing import List, Optional, Callable, Any
from dataclasses import dataclass
from utils.plate_detector import LicensePlateDetector, PlateDetection
from utils.detection_manager import detection_manager
from utils.bounding_box_refiner import bounding_box_refiner

@dataclass
class ProcessingResult:
    """Result dari frame processing"""
    frame: cv2.Mat
    detections: List[PlateDetection]
    processing_time: float
    timestamp: float
    frame_id: int

class FrameProcessor:
    """
    Multi-threaded frame processor untuk live video
    
    Architecture:
    - Input Thread: Menerima frame dari video stream
    - Processing Thread(s): Process frame untuk deteksi plat
    - Output Thread: Handle hasil dan simpan ke database
    """
    
    def __init__(self, plate_detector: LicensePlateDetector, 
                 max_threads: int = 2, queue_size: int = 30,
                 motorcycle_detector=None):
        """
        Initialize frame processor
        
        Args:
            plate_detector: Instance dari LicensePlateDetector
            max_threads: Maksimal processing threads
            queue_size: Ukuran queue untuk buffer
            motorcycle_detector: Optional motorcycle plate detector
        """
        self.detector = plate_detector
        self.motorcycle_detector = motorcycle_detector
        self.max_threads = max_threads
        
        # Queues untuk inter-thread communication
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)
        
        # Threading control
        self.running = False
        self.input_thread = None
        self.processing_threads = []
        self.output_thread = None
        
        # Statistics
        self.frames_processed = 0
        self.frames_skipped = 0
        self.total_processing_time = 0
        self.frame_counter = 0
        
        # Callbacks
        self.detection_callback = None
        self.frame_callback = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FrameProcessor initialized with {max_threads} processing threads")
    
    def set_detection_callback(self, callback: Callable[[List[PlateDetection]], None]):
        """
        Set callback function untuk handle hasil deteksi
        
        Args:
            callback: Function yang akan dipanggil dengan parameter List[PlateDetection]
        """
        self.detection_callback = callback
    
    def set_frame_callback(self, callback: Callable[[ProcessingResult], None]):
        """
        Set callback function untuk handle processed frame
        
        Args:
            callback: Function yang akan dipanggil dengan parameter ProcessingResult
        """
        self.frame_callback = callback
    
    def start(self):
        """Start semua processing threads"""
        if self.running:
            self.logger.warning("FrameProcessor already running")
            return
        
        self.running = True
        self.frames_processed = 0
        self.frames_skipped = 0
        self.total_processing_time = 0
        self.frame_counter = 0
        
        # Start processing threads
        self.processing_threads = []
        for i in range(self.max_threads):
            thread = threading.Thread(
                target=self._processing_worker, 
                name=f"ProcessingThread-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        # Start output thread
        self.output_thread = threading.Thread(
            target=self._output_worker,
            name="OutputThread", 
            daemon=True
        )
        self.output_thread.start()
        
        self.logger.info("FrameProcessor started")
    
    def stop(self):
        """Stop semua threads"""
        if not self.running:
            return
            
        self.logger.info("Stopping FrameProcessor...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.processing_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2.0)
        
        # Clear queues
        self._clear_queue(self.input_queue)
        self._clear_queue(self.output_queue)
        
        self.logger.info("FrameProcessor stopped")
    
    def _clear_queue(self, q: queue.Queue):
        """Clear queue contents"""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    
    def add_frame(self, frame: cv2.Mat) -> bool:
        """
        Add frame untuk diproses
        
        Args:
            frame: Input frame
            
        Returns:
            bool: True jika berhasil ditambahkan ke queue
        """
        if not self.running:
            return False
        
        try:
            # Jika queue penuh, skip frame lama
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()  # Remove old frame
                    self.frames_skipped += 1
                except queue.Empty:
                    pass
            
            # Add new frame dengan timestamp
            self.frame_counter += 1
            frame_data = {
                'frame': frame.copy(),
                'timestamp': time.time(),
                'frame_id': self.frame_counter
            }
            
            self.input_queue.put_nowait(frame_data)
            return True
            
        except queue.Full:
            self.frames_skipped += 1
            return False
        except Exception as e:
            self.logger.error(f"Error adding frame: {str(e)}")
            return False
    
    def _processing_worker(self):
        """Worker thread untuk processing frames"""
        thread_name = threading.current_thread().name
        self.logger.debug(f"{thread_name} started")
        
        while self.running:
            try:
                # Get frame dari input queue
                frame_data = self.input_queue.get(timeout=1.0)
                
                if frame_data is None:
                    continue
                
                # Extract data
                frame = frame_data['frame']
                timestamp = frame_data['timestamp']
                frame_id = frame_data['frame_id']
                
                # Process frame dengan dual detector
                start_time = time.time()
                
                # Get detections from all sources
                detections_dict = {}

                # General detector
                try:
                    general_detections = self.detector.detect_plates(frame)
                    detections_dict['general'] = general_detections
                except Exception as e:
                    self.logger.warning(f"General detection failed: {str(e)}")
                    detections_dict['general'] = []

                # Motorcycle detector if available
                if self.motorcycle_detector and self.motorcycle_detector.is_enabled():
                    try:
                        motorcycle_detections = self.motorcycle_detector.detect_plates(frame)
                        # Mark as motorcycle detections
                        for detection in motorcycle_detections:
                            detection.vehicle_type = "motorcycle"
                            detection.detection_method = "motorcycle_optimized"
                        detections_dict['motorcycle'] = motorcycle_detections
                    except Exception as e:
                        self.logger.warning(f"Motorcycle detection failed: {str(e)}")
                        detections_dict['motorcycle'] = []
                else:
                    detections_dict['motorcycle'] = []

                # Use DetectionManager untuk process dan deduplicate
                detections = detection_manager.process_detections(detections_dict, frame, frame_id)

                # Enhance bounding boxes untuk valid detections
                enhanced_detections = []
                for detection in detections:
                    try:
                        refined_bbox = bounding_box_refiner.refine_bounding_box(
                            frame, detection.bbox, detection.confidence
                        )
                        if refined_bbox and refined_bbox.confidence > 0.3:  # Minimum threshold
                            # Update detection dengan refined bbox
                            detection.bbox = (refined_bbox.x, refined_bbox.y,
                                            refined_bbox.width, refined_bbox.height)
                            detection.confidence = refined_bbox.confidence
                            enhanced_detections.append(detection)
                    except Exception as e:
                        # Fallback ke original detection
                        self.logger.debug(f"Bbox refinement failed: {str(e)}")
                        enhanced_detections.append(detection)

                detections = enhanced_detections
                
                processing_time = time.time() - start_time
                
                # Create result
                result = ProcessingResult(
                    frame=frame,
                    detections=detections,
                    processing_time=processing_time,
                    timestamp=timestamp,
                    frame_id=frame_id
                )
                
                # Put ke output queue
                if not self.output_queue.full():
                    self.output_queue.put_nowait(result)
                else:
                    # Skip jika output queue penuh
                    self.frames_skipped += 1
                
                # Update statistics
                self.frames_processed += 1
                self.total_processing_time += processing_time
                
                self.logger.debug(f"{thread_name} processed frame {frame_id} in {processing_time:.3f}s")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in {thread_name}: {str(e)}")
                time.sleep(0.1)
    
    def _output_worker(self):
        """Worker thread untuk handle output results"""
        self.logger.debug("OutputThread started")
        
        while self.running:
            try:
                # Get result dari output queue
                result = self.output_queue.get(timeout=1.0)
                
                if result is None:
                    continue
                
                # Call detection callback jika ada deteksi
                if result.detections and self.detection_callback:
                    try:
                        self.detection_callback(result.detections)
                    except Exception as e:
                        self.logger.error(f"Error in detection callback: {str(e)}")
                
                # Call frame callback
                if self.frame_callback:
                    try:
                        self.frame_callback(result)
                    except Exception as e:
                        self.logger.error(f"Error in frame callback: {str(e)}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in OutputThread: {str(e)}")
                time.sleep(0.1)
    
    def get_queue_sizes(self) -> dict:
        """Get ukuran semua queue"""
        return {
            'input_queue': self.input_queue.qsize(),
            'output_queue': self.output_queue.qsize()
        }
    
    def get_statistics(self) -> dict:
        """Get processing statistics including detection manager stats"""
        avg_processing_time = (self.total_processing_time / self.frames_processed
                              if self.frames_processed > 0 else 0)

        fps = self.frames_processed / self.total_processing_time if self.total_processing_time > 0 else 0

        # Get detection manager statistics
        detection_stats = detection_manager.get_statistics()

        return {
            'frames_processed': self.frames_processed,
            'frames_skipped': self.frames_skipped,
            'avg_processing_time': round(avg_processing_time, 3),
            'processing_fps': round(fps, 1),
            'queue_sizes': self.get_queue_sizes(),
            'detection_stats': detection_stats
        }
    
    def is_running(self) -> bool:
        """Check apakah processor masih berjalan"""
        return self.running
    
    def get_latest_result(self, timeout: float = 0.1) -> Optional[ProcessingResult]:
        """
        Get hasil processing terbaru
        
        Args:
            timeout: Timeout untuk waiting
            
        Returns:
            ProcessingResult atau None jika tidak ada
        """
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

class BatchFrameProcessor(FrameProcessor):
    """
    Extended processor untuk batch processing
    """
    
    def __init__(self, *args, batch_size: int = 5, **kwargs):
        """
        Initialize batch processor
        
        Args:
            batch_size: Jumlah frame per batch
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.frame_batch = []
        self.batch_lock = threading.Lock()
    
    def add_frame_to_batch(self, frame: cv2.Mat) -> bool:
        """
        Add frame ke batch untuk batch processing
        
        Args:
            frame: Input frame
            
        Returns:
            bool: True jika batch sudah penuh dan diproses
        """
        with self.batch_lock:
            self.frame_batch.append({
                'frame': frame.copy(),
                'timestamp': time.time(),
                'frame_id': self.frame_counter
            })
            self.frame_counter += 1
            
            # Process batch jika sudah penuh
            if len(self.frame_batch) >= self.batch_size:
                batch = self.frame_batch.copy()
                self.frame_batch.clear()
                
                # Process batch di background
                threading.Thread(
                    target=self._process_batch,
                    args=(batch,),
                    daemon=True
                ).start()
                
                return True
        
        return False
    
    def _process_batch(self, batch: List[dict]):
        """Process batch frames"""
        results = []
        
        for frame_data in batch:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            frame_id = frame_data['frame_id']
            
            # Process frame dengan dual detector
            start_time = time.time()
            
            # Get detections from general detector
            general_detections = self.detector.detect_plates(frame)
            
            # Get detections from motorcycle detector if available
            motorcycle_detections = []
            if self.motorcycle_detector and self.motorcycle_detector.is_enabled():
                try:
                    motorcycle_detections = self.motorcycle_detector.detect_plates(frame)
                    # Mark as motorcycle detections
                    for detection in motorcycle_detections:
                        detection.vehicle_type = "motorcycle"
                        detection.detection_method = "motorcycle_optimized"
                except Exception as e:
                    self.logger.warning(f"Motorcycle detection failed: {str(e)}")
            
            # Combine and prioritize detections
            detections = self._combine_detections(general_detections, motorcycle_detections)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                frame=frame,
                detections=detections,
                processing_time=processing_time,
                timestamp=timestamp,
                frame_id=frame_id
            )
            
            results.append(result)
            self.frames_processed += 1
            self.total_processing_time += processing_time
        
        # Process all results
        for result in results:
            if result.detections and self.detection_callback:
                try:
                    self.detection_callback(result.detections)
                except Exception as e:
                    self.logger.error(f"Error in batch detection callback: {str(e)}")
            
            if self.frame_callback:
                try:
                    self.frame_callback(result)
                except Exception as e:
                    self.logger.error(f"Error in batch frame callback: {str(e)}")
    
    def _combine_detections(self, general_detections, motorcycle_detections):
        """
        Combine detections from general and motorcycle detectors
        Prioritize motorcycle detections for better accuracy
        
        Args:
            general_detections: List of detections from general detector
            motorcycle_detections: List of detections from motorcycle detector
            
        Returns:
            List of combined detections with duplicates removed
        """
        combined = []
        
        # Add motorcycle detections first (higher priority)
        for moto_detection in motorcycle_detections:
            combined.append(moto_detection)
        
        # Add general detections that don't overlap with motorcycle detections
        for general_detection in general_detections:
            # Check for overlap with motorcycle detections
            overlaps = False
            gx, gy, gw, gh = general_detection.bbox
            
            for moto_detection in motorcycle_detections:
                mx, my, mw, mh = moto_detection.bbox
                
                # Calculate intersection over union (IoU)
                x1 = max(gx, mx)
                y1 = max(gy, my)
                x2 = min(gx + gw, mx + mw)
                y2 = min(gy + gh, my + mh)
                
                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = gw * gh
                    area2 = mw * mh
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > 0.3:  # 30% overlap threshold
                        overlaps = True
                        break
            
            # Add general detection if no overlap
            if not overlaps:
                # Set vehicle_type based on detection characteristics if not set
                if not hasattr(general_detection, 'vehicle_type') or general_detection.vehicle_type == "unknown":
                    general_detection.vehicle_type = "car"  # Default to car
                combined.append(general_detection)
        
        return combined

def test_frame_processor():
    """Test function untuk frame processor"""
    print("Testing FrameProcessor...")
    
    # Create test detector
    from utils.plate_detector import LicensePlateDetector
    detector = LicensePlateDetector()
    
    # Create processor
    processor = FrameProcessor(detector, max_threads=2)
    
    # Setup callbacks
    def detection_callback(detections):
        print(f"Callback: Found {len(detections)} plates")
        for det in detections:
            print(f"  - {det.text} ({det.confidence:.1f}%)")
    
    def frame_callback(result):
        print(f"Frame {result.frame_id} processed in {result.processing_time:.3f}s")
    
    processor.set_detection_callback(detection_callback)
    processor.set_frame_callback(frame_callback)
    
    # Start processor
    processor.start()
    
    # Add some test frames
    for i in range(5):
        test_frame = cv2.imread('test_image.jpg') if i == 0 else None
        if test_frame is None:
            # Create dummy frame
            test_frame = cv2.rectangle(
                np.ones((480, 640, 3), dtype=np.uint8) * 50,
                (200, 200), (400, 250), (255, 255, 255), -1
            )
        
        success = processor.add_frame(test_frame)
        print(f"Frame {i+1} added: {success}")
        time.sleep(0.1)
    
    # Wait for processing
    time.sleep(2)
    
    # Show statistics
    stats = processor.get_statistics()
    print(f"Statistics: {stats}")
    
    # Stop processor
    processor.stop()
    print("Test completed")

if __name__ == "__main__":
    # Setup logging untuk testing
    logging.basicConfig(level=logging.INFO)
    
    # Import numpy untuk test
    import numpy as np
    
    # Run test
    test_frame_processor()