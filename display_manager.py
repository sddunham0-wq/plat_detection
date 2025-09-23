"""
Display Manager untuk OpenCV Thread-Safe Display
Mengatasi threading issues dengan cv2.imshow()
"""

import cv2
import threading
import queue
import time
import logging
import os
from datetime import datetime
from typing import Optional, Tuple

class ThreadSafeDisplayManager:
    """
    Thread-safe display manager untuk OpenCV
    Mengatasi "Unknown C++ exception" di threading context
    """
    
    def __init__(self, window_name: str = "Live Plate Detection"):
        self.window_name = window_name
        self.display_queue = queue.Queue(maxsize=5)  # Small buffer
        self.display_thread = None
        self.running = False
        self.display_available = True
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Test display capability
        self._test_display_capability()
    
    def _test_display_capability(self):
        """Test if display is available in current environment"""
        try:
            import numpy as np
            
            # Check environment first
            if os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'):
                self.logger.info("SSH environment detected - disabling display")
                self.display_available = False
                return
                
            if not os.environ.get('DISPLAY') and not os.environ.get('WINDOWID'):
                # No display environment on Linux/macOS
                pass  # macOS doesn't always need DISPLAY
            
            # Quick test with threading simulation
            test_img = np.zeros((100, 100, 3), dtype='uint8')
            
            # Test in thread context
            import threading
            test_error = None
            
            def test_thread():
                nonlocal test_error
                try:
                    cv2.imshow("test_window", test_img)
                    cv2.waitKey(1)  # This is where error usually occurs
                    cv2.destroyWindow("test_window")
                except Exception as e:
                    test_error = e
            
            thread = threading.Thread(target=test_thread)
            thread.start()
            thread.join(timeout=2.0)
            
            if test_error:
                raise test_error
            
            self.display_available = True
            self.logger.info("Display capability: Available (thread-safe)")
            
        except Exception as e:
            self.display_available = False
            self.logger.warning(f"Display not available: {str(e)}")
            self.logger.info("Will use fallback display mode")
    
    def start(self):
        """Start display thread"""
        if not self.display_available:
            self.logger.info("Display not available - preview disabled")
            return False
            
        self.running = True
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.display_thread.start()
        self.logger.info("Display thread started")
        return True
    
    def stop(self):
        """Stop display thread"""
        self.running = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        # Clean up OpenCV windows
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        self.logger.info("Display thread stopped")
    
    def _display_worker(self):
        """Worker thread for display operations - with robust error handling"""
        self.logger.debug("Display worker started")
        
        cv2_error_count = 0
        max_cv2_errors = 10
        
        try:
            while self.running:
                try:
                    # Get frame from queue (non-blocking)
                    frame_data = self.display_queue.get(timeout=0.1)
                    
                    if frame_data is None:
                        continue
                    
                    frame = frame_data['frame']
                    callback = frame_data.get('callback', None)
                    
                    # Try display operations with error recovery
                    try:
                        # Display frame
                        cv2.imshow(self.window_name, frame)
                        
                        # Handle keyboard input (this often causes the error)
                        key = cv2.waitKey(1) & 0xFF
                        
                        # Reset error count on success
                        cv2_error_count = 0
                        
                        # Call callback with key if provided
                        if callback and callable(callback):
                            try:
                                callback(key)
                            except Exception as e:
                                self.logger.error(f"Display callback error: {str(e)}")
                    
                    except cv2.error as e:
                        cv2_error_count += 1
                        if cv2_error_count <= 3:  # Log first few errors
                            self.logger.warning(f"OpenCV display error ({cv2_error_count}): {str(e)}")
                        
                        if cv2_error_count >= max_cv2_errors:
                            self.logger.error("Too many OpenCV errors - disabling display")
                            self.display_available = False
                            break
                        
                        # Call callback with no key on error
                        if callback and callable(callback):
                            try:
                                callback(255)  # No key
                            except:
                                pass
                    
                except queue.Empty:
                    # No frame to display - try minimal key handling
                    if self.running and cv2_error_count < max_cv2_errors:
                        try:
                            key = cv2.waitKey(1) & 0xFF
                            cv2_error_count = 0  # Reset on success
                        except cv2.error:
                            cv2_error_count += 1
                            if cv2_error_count >= max_cv2_errors:
                                self.logger.error("OpenCV waitKey consistently failing - disabling display")
                                self.display_available = False
                                break
                    continue
                    
                except Exception as e:
                    self.logger.error(f"Display worker unexpected error: {str(e)}")
                    time.sleep(0.1)  # Brief pause on error
        
        except Exception as e:
            self.logger.error(f"Display worker fatal error: {str(e)}")
        
        finally:
            # Cleanup
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            self.logger.debug("Display worker finished")
    
    def show_frame(self, frame, callback=None) -> bool:
        """
        Show frame in display (thread-safe)
        
        Args:
            frame: OpenCV image to display
            callback: Function to call with key press (optional)
            
        Returns:
            bool: True if frame queued for display
        """
        if not self.display_available or not self.running:
            return False
        
        try:
            # Prepare frame data
            frame_data = {
                'frame': frame.copy(),
                'callback': callback,
                'timestamp': time.time()
            }
            
            # Add to queue (non-blocking)
            if self.display_queue.full():
                # Remove old frame if queue is full
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.display_queue.put_nowait(frame_data)
            return True
            
        except Exception as e:
            self.logger.error(f"Error queuing frame for display: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Check if display is available"""
        return self.display_available
    
    def is_running(self) -> bool:
        """Check if display thread is running"""
        return self.running and self.display_thread and self.display_thread.is_alive()

class FallbackDisplayManager:
    """
    Fallback display manager - saves images instead of showing
    """
    
    def __init__(self, output_dir: str = "preview_frames"):
        self.output_dir = output_dir
        self.frame_count = 0
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Fallback display: saving frames to {output_dir}")
    
    def start(self):
        """Start fallback display"""
        return True
    
    def stop(self):
        """Stop fallback display"""
        pass
    
    def show_frame(self, frame, callback=None) -> bool:
        """Save frame instead of displaying"""
        try:
            self.frame_count += 1
            
            # Save every 30th frame to avoid spam
            if self.frame_count % 30 == 0:
                timestamp = datetime.now().strftime('%H%M%S')
                filename = f"frame_{timestamp}_{self.frame_count}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                cv2.imwrite(filepath, frame)
                self.logger.debug(f"Saved frame: {filepath}")
            
            # Simulate callback for key handling
            if callback and callable(callback):
                callback(255)  # No key pressed
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fallback display error: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Fallback is always available"""
        return True
    
    def is_running(self) -> bool:
        """Fallback is always running"""
        return True

def create_display_manager(window_name: str = "Live Plate Detection", 
                         force_fallback: bool = False):
    """
    Factory function untuk create display manager
    
    Args:
        window_name: Name for display window
        force_fallback: Force use fallback display
        
    Returns:
        Display manager instance
    """
    if force_fallback:
        return FallbackDisplayManager()
    
    # Try thread-safe display first
    display_mgr = ThreadSafeDisplayManager(window_name)
    
    if display_mgr.is_available():
        return display_mgr
    else:
        # Fallback to image saving
        return FallbackDisplayManager()

# Example usage and testing
def test_display_manager():
    """Test display manager functionality"""
    import numpy as np
    
    print("Testing Display Manager...")
    
    # Create display manager
    display_mgr = create_display_manager()
    
    def key_callback(key):
        if key == ord('q'):
            print("Q key pressed!")
        elif key == ord('s'):
            print("S key pressed!")
    
    # Start display
    if display_mgr.start():
        print("Display started")
        
        # Test with some frames
        for i in range(10):
            # Create test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Test Frame {i+1}", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Show frame
            success = display_mgr.show_frame(frame, key_callback)
            print(f"Frame {i+1} displayed: {success}")
            
            time.sleep(0.1)
        
        time.sleep(2)  # Let it run for a bit
        display_mgr.stop()
        print("Display stopped")
    else:
        print("Failed to start display")

if __name__ == "__main__":
    test_display_manager()