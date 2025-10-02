"""
Super Resolution Module untuk License Plate Enhancement
Lightweight real-time image enhancement untuk deteksi plat jarak jauh
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional
from scipy import ndimage
from skimage import restoration, filters, exposure, transform
import time

class SuperResolutionEnhancer:
    """
    Real-time super resolution enhancer untuk license plates
    Menggunakan teknik klasik yang dioptimalkan untuk performa
    """
    
    def __init__(self):
        """Initialize super resolution enhancer"""
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.enhancement_times = []
        self.enhancement_count = 0
        
        # Pre-computed kernels untuk efficiency
        self.sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1], 
                                       [-1, -1, -1]])
        
        self.edge_kernel = np.array([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]])
        
        self.logger.info("Super Resolution Enhancer initialized")
    
    def bicubic_upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        High-quality bicubic upscaling
        
        Args:
            image: Input image
            scale_factor: Upscaling factor
            
        Returns:
            np.ndarray: Upscaled image
        """
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    def lanczos_upscale(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Lanczos resampling untuk better quality
        
        Args:
            image: Input image
            scale_factor: Upscaling factor
            
        Returns:
            np.ndarray: Upscaled image using Lanczos
        """
        try:
            height, width = image.shape[:2]
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            
            # Convert to float for better precision
            float_image = image.astype(np.float32) / 255.0
            
            # Use skimage transform with Lanczos
            upscaled = transform.resize(
                float_image,
                (new_height, new_width),
                order=3,  # Lanczos-like
                preserve_range=True,
                anti_aliasing=True
            )
            
            return (upscaled * 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"Lanczos upscaling failed, fallback to bicubic: {str(e)}")
            return self.bicubic_upscale(image, scale_factor)
    
    def edge_directed_interpolation(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Edge-directed interpolation untuk preserve edges
        
        Args:
            image: Input image
            scale_factor: Upscaling factor
            
        Returns:
            np.ndarray: Edge-preserved upscaled image
        """
        # Step 1: Basic upscaling
        upscaled = self.bicubic_upscale(image, scale_factor)
        
        # Step 2: Edge detection on original
        edges = cv2.Canny(image, 50, 150)
        edges_upscaled = self.bicubic_upscale(edges, scale_factor)
        
        # Step 3: Edge-guided enhancement
        # Apply stronger sharpening pada edge areas
        edge_mask = edges_upscaled > 50
        
        # Sharpen entire image
        sharpened = cv2.filter2D(upscaled, -1, self.sharpen_kernel)
        
        # Blend based on edge mask
        enhanced = np.where(edge_mask[..., np.newaxis], sharpened, upscaled)
        
        return enhanced.astype(np.uint8)
    
    def unsharp_mask_enhancement(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Unsharp mask untuk detail enhancement
        
        Args:
            image: Input image
            strength: Enhancement strength
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Create unsharp mask
        unsharp = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        
        return np.clip(unsharp, 0, 255).astype(np.uint8)
    
    def wiener_deconvolution(self, image: np.ndarray, noise_ratio: float = 0.01) -> np.ndarray:
        """
        Simplified Wiener deconvolution untuk blur reduction
        
        Args:
            image: Input blurred image
            noise_ratio: Noise-to-signal ratio
            
        Returns:
            np.ndarray: Deblurred image
        """
        try:
            # Convert to float
            float_image = image.astype(np.float32) / 255.0
            
            # Simple blur kernel (motion blur simulation)
            kernel = np.ones((3, 3)) / 9
            
            # Wiener deconvolution
            deblurred = restoration.wiener(float_image, kernel, noise_ratio)
            
            return (np.clip(deblurred, 0, 1) * 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.warning(f"Wiener deconvolution failed: {str(e)}")
            return image
    
    def adaptive_enhancement(self, image: np.ndarray, enhancement_factor: float = 2.0) -> np.ndarray:
        """
        Adaptive enhancement berdasarkan local image characteristics
        
        Args:
            image: Input image
            enhancement_factor: Enhancement factor
            
        Returns:
            np.ndarray: Adaptively enhanced image
        """
        # Calculate local variance untuk identify detail areas
        kernel = np.ones((5, 5))
        local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel / 25)
        local_variance = cv2.filter2D((image.astype(np.float32) - local_mean) ** 2, -1, kernel / 25)
        
        # Normalize variance
        var_normalized = local_variance / (local_variance.max() + 1e-6)
        
        # Create enhancement mask
        enhancement_mask = var_normalized * enhancement_factor
        enhancement_mask = np.clip(enhancement_mask, 0.5, 2.0)
        
        # Apply adaptive enhancement
        enhanced = image.astype(np.float32) * enhancement_mask
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def super_resolve_esrgan_style(self, image: np.ndarray, scale_factor: float = 4.0) -> np.ndarray:
        """
        ESRGAN-style super resolution menggunakan classical methods
        
        Args:
            image: Input low-resolution image
            scale_factor: Upscaling factor
            
        Returns:
            np.ndarray: Super-resolved image
        """
        start_time = time.time()
        
        # Step 1: Edge-directed interpolation
        upscaled = self.edge_directed_interpolation(image, scale_factor)
        
        # Step 2: Deblur jika perlu
        if scale_factor > 2.0:
            upscaled = self.wiener_deconvolution(upscaled, noise_ratio=0.005)
        
        # Step 3: Adaptive enhancement
        enhanced = self.adaptive_enhancement(upscaled, enhancement_factor=1.5)
        
        # Step 4: Final sharpening
        final = self.unsharp_mask_enhancement(enhanced, strength=1.2)
        
        # Track performance
        processing_time = time.time() - start_time
        self.enhancement_times.append(processing_time)
        self.enhancement_count += 1
        
        return final
    
    def real_time_enhance(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Real-time enhancement untuk live video processing
        Optimized untuk speed vs quality trade-off
        
        Args:
            image: Input image
            target_size: Target size (width, height), None for auto-scaling
            
        Returns:
            np.ndarray: Enhanced image
        """
        start_time = time.time()
        
        # Determine scale factor
        if target_size:
            height, width = image.shape[:2]
            scale_x = target_size[0] / width
            scale_y = target_size[1] / height
            scale_factor = min(scale_x, scale_y)
        else:
            # Auto-scale berdasarkan image size
            height, width = image.shape[:2]
            if height < 30 or width < 60:
                scale_factor = 4.0
            elif height < 50 or width < 100:
                scale_factor = 3.0
            else:
                scale_factor = 2.0
        
        # Quick enhancement pipeline
        if scale_factor > 1.0:
            # Method 1: Fast upscaling
            enhanced = self.lanczos_upscale(image, scale_factor)
            
            # Method 2: Quick sharpening
            enhanced = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
            
            # Method 3: Contrast enhancement
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
        else:
            # Just apply sharpening jika tidak perlu upscaling
            enhanced = cv2.filter2D(image, -1, self.sharpen_kernel)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)
        
        # Track performance
        processing_time = time.time() - start_time
        self.enhancement_times.append(processing_time)
        self.enhancement_count += 1
        
        return enhanced
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Specialized enhancement untuk OCR accuracy
        
        Args:
            image: Input plate image
            
        Returns:
            np.ndarray: OCR-optimized image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Upscale jika terlalu kecil
        height, width = gray.shape
        if height < 40 or width < 80:
            scale_factor = max(40 / height, 80 / width)
            gray = self.lanczos_upscale(gray, scale_factor)
        
        # Step 2: Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Step 3: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Step 4: Sharpening
        sharpened = cv2.filter2D(enhanced, -1, self.sharpen_kernel)
        
        # Step 5: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        
        return final
    
    def batch_enhance(self, images: list, method: str = "real_time") -> list:
        """
        Batch enhancement untuk multiple images
        
        Args:
            images: List of images to enhance
            method: Enhancement method ("real_time", "esrgan_style", "ocr")
            
        Returns:
            list: List of enhanced images
        """
        enhanced_images = []
        
        for image in images:
            if method == "real_time":
                enhanced = self.real_time_enhance(image)
            elif method == "esrgan_style":
                enhanced = self.super_resolve_esrgan_style(image)
            elif method == "ocr":
                enhanced = self.enhance_for_ocr(image)
            else:
                enhanced = self.real_time_enhance(image)
            
            enhanced_images.append(enhanced)
        
        return enhanced_images
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.enhancement_times:
            return {"avg_time": 0, "total_enhancements": 0}
        
        avg_time = np.mean(self.enhancement_times)
        return {
            "avg_time": round(avg_time, 4),
            "total_enhancements": self.enhancement_count,
            "fps_estimate": round(1.0 / avg_time, 2) if avg_time > 0 else 0
        }

class AdaptiveSuperResolution:
    """
    Adaptive super resolution yang memilih method terbaik berdasarkan kondisi
    """
    
    def __init__(self):
        """Initialize adaptive super resolution"""
        self.enhancer = SuperResolutionEnhancer()
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.blur_threshold = 100.0
        self.contrast_threshold = 50.0
        self.size_threshold = (50, 100)  # min height, width
    
    def assess_image_needs(self, image: np.ndarray) -> dict:
        """
        Assess image dan tentukan enhancement strategy
        
        Args:
            image: Input image
            
        Returns:
            dict: Assessment results
        """
        # Ensure grayscale untuk assessment
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        
        # Blur assessment
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = blur_score < self.blur_threshold
        
        # Contrast assessment
        contrast_score = gray.std()
        is_low_contrast = contrast_score < self.contrast_threshold
        
        # Size assessment
        is_small = height < self.size_threshold[0] or width < self.size_threshold[1]
        
        # Noise assessment
        noise_score = filters.rank.variance(gray, morphology.disk(1)).mean()
        is_noisy = noise_score > 20
        
        return {
            "is_blurry": is_blurry,
            "is_low_contrast": is_low_contrast,
            "is_small": is_small,
            "is_noisy": is_noisy,
            "blur_score": blur_score,
            "contrast_score": contrast_score,
            "size": (height, width),
            "noise_score": noise_score
        }
    
    def adaptive_enhance(self, image: np.ndarray, priority: str = "quality") -> np.ndarray:
        """
        Adaptive enhancement berdasarkan image assessment
        
        Args:
            image: Input image
            priority: "quality" atau "speed"
            
        Returns:
            np.ndarray: Adaptively enhanced image
        """
        assessment = self.assess_image_needs(image)
        
        # Choose enhancement strategy
        if priority == "speed":
            # Fast enhancement
            return self.enhancer.real_time_enhance(image)
        
        elif assessment["is_small"] and assessment["is_blurry"]:
            # Small and blurry - need aggressive enhancement
            enhanced = self.enhancer.super_resolve_esrgan_style(image, scale_factor=4.0)
            return self.enhancer.wiener_deconvolution(enhanced)
        
        elif assessment["is_small"]:
            # Just small - upscale with good quality
            return self.enhancer.super_resolve_esrgan_style(image, scale_factor=3.0)
        
        elif assessment["is_blurry"]:
            # Blurry but good size - focus on deblur
            deblurred = self.enhancer.wiener_deconvolution(image)
            return self.enhancer.unsharp_mask_enhancement(deblurred)
        
        elif assessment["is_low_contrast"]:
            # Low contrast - focus on enhancement
            return self.enhancer.adaptive_enhancement(image, enhancement_factor=2.0)
        
        else:
            # Good quality - light enhancement
            return self.enhancer.unsharp_mask_enhancement(image, strength=1.0)

# Test functions
def test_super_resolution():
    """Test super resolution module"""
    print("Testing Super Resolution Module...")
    
    # Create test image
    test_image = np.ones((30, 60), dtype=np.uint8) * 128
    cv2.putText(test_image, "ABC123", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    
    # Add some blur and noise
    blurred = cv2.GaussianBlur(test_image, (3, 3), 1.0)
    noise = np.random.normal(0, 10, blurred.shape).astype(np.uint8)
    noisy = cv2.add(blurred, noise)
    
    print(f"Original size: {noisy.shape}")
    
    # Test different enhancement methods
    enhancer = SuperResolutionEnhancer()
    
    # Method 1: Real-time
    start_time = time.time()
    enhanced_rt = enhancer.real_time_enhance(noisy)
    rt_time = time.time() - start_time
    print(f"Real-time enhancement: {enhanced_rt.shape}, time: {rt_time:.4f}s")
    
    # Method 2: ESRGAN-style
    start_time = time.time()
    enhanced_esrgan = enhancer.super_resolve_esrgan_style(noisy, scale_factor=4.0)
    esrgan_time = time.time() - start_time
    print(f"ESRGAN-style enhancement: {enhanced_esrgan.shape}, time: {esrgan_time:.4f}s")
    
    # Method 3: OCR-optimized
    start_time = time.time()
    enhanced_ocr = enhancer.enhance_for_ocr(noisy)
    ocr_time = time.time() - start_time
    print(f"OCR-optimized enhancement: {enhanced_ocr.shape}, time: {ocr_time:.4f}s")
    
    # Test adaptive enhancement
    adaptive = AdaptiveSuperResolution()
    assessment = adaptive.assess_image_needs(noisy)
    print(f"Image assessment: {assessment}")
    
    enhanced_adaptive = adaptive.adaptive_enhance(noisy, priority="quality")
    print(f"Adaptive enhancement: {enhanced_adaptive.shape}")
    
    # Performance stats
    stats = enhancer.get_performance_stats()
    print(f"Performance stats: {stats}")

if __name__ == "__main__":
    test_super_resolution()