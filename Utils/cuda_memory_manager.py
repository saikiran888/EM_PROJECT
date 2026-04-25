"""
CUDA Memory Manager for non-blocking GPU operations.
Provides utilities for managing CUDA memory efficiently without causing UI freezing.
"""

import torch
import threading
from contextlib import contextmanager


class CUDAMemoryManager:
    """Manages CUDA memory efficiently to prevent blocking operations"""
    
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize memory manager.
        
        Args:
            device: CUDA device string (e.g., "cuda:0")
        """
        self.device = device
        self.is_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self._lock = threading.Lock()
        self._background_thread = None
    
    def get_memory_stats(self) -> dict:
        """Get current CUDA memory statistics (non-blocking)"""
        if not self.is_cuda:
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "available_mb": 0,
            }
        
        try:
            device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            allocated = torch.cuda.memory_allocated(device_idx) / 1024**2
            reserved = torch.cuda.memory_reserved(device_idx) / 1024**2
            total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**2
            available = total - allocated
            
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "available_mb": available,
                "total_mb": total,
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {
                "allocated_mb": 0,
                "reserved_mb": 0,
                "available_mb": 0,
            }
    
    @contextmanager
    def auto_cleanup(self):
        """
        Context manager for automatic CUDA memory cleanup.
        Do NOT use torch.cuda.synchronize() - it causes blocking.
        """
        try:
            yield
        finally:
            # Cleanup happens asynchronously via garbage collection
            # No explicit synchronize - that's what causes the hang!
            if self.is_cuda:
                try:
                    # This is non-blocking - just releases cached memory
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    
    def cleanup_async(self):
        """
        Asynchronously clean up CUDA memory in a background thread.
        This is truly non-blocking - perfect for UI updates.
        """
        if not self.is_cuda:
            return
        
        def cleanup_worker():
            try:
                # No synchronize here - that causes the hang!
                torch.cuda.empty_cache()
                print("[CUDA Memory] Async cleanup completed")
            except Exception as e:
                print(f"[CUDA Memory] Async cleanup error: {e}")
        
        # Start cleanup in background thread
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def clear_all(self):
        """
        Clear all CUDA memory caches (blocking - only call when necessary).
        This is a last resort - avoid using during inference!
        """
        if not self.is_cuda:
            return
        
        try:
            with self._lock:
                # Only synchronize if absolutely necessary
                # torch.cuda.synchronize()  # DON'T DO THIS - causes hangs!
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                print("[CUDA Memory] All caches cleared")
        except Exception as e:
            print(f"[CUDA Memory] Clear error: {e}")
    
    def optimize_for_inference(self):
        """
        Configure CUDA for optimal inference performance (non-blocking).
        This should be called once at startup.
        """
        if not self.is_cuda:
            return
        
        try:
            # Enable TF32 for faster operations without reduced precision concerns
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Disable cuDNN benchmark to save memory
            # torch.backends.cudnn.benchmark = False  # Actually, benchmark can help
            
            # Enable persistent kernels for better performance
            torch.backends.cudnn.enabled = True
            
            print("[CUDA Memory] Optimized CUDA settings for inference")
        except Exception as e:
            print(f"[CUDA Memory] Error optimizing CUDA: {e}")
    
    def estimate_memory_needed(self, image_size: tuple, num_boxes: int) -> float:
        """
        Estimate CUDA memory needed for SAM3 inference.
        
        Args:
            image_size: (height, width) tuple
            num_boxes: number of bounding boxes
            
        Returns:
            Estimated memory in MB
        """
        h, w = image_size
        
        # Rough estimates based on SAM3 model
        # Image tensor: 3 * h * w * 4 bytes
        image_mem = (3 * h * w * 4) / (1024**2)
        
        # Embeddings: 256 * 64 * 64 * 4 bytes (for 1008x1008 input)
        embedding_mem = (256 * 64 * 64 * 4) / (1024**2)
        
        # Mask decoder per box: 256 * h * w * 4 bytes
        decoder_mem = (256 * h * w * 4) / (1024**2) * num_boxes
        
        # Safety margin
        total = (image_mem + embedding_mem + decoder_mem) * 1.5
        
        return total
    
    def check_available_memory(self, required_mb: float) -> bool:
        """Check if sufficient CUDA memory is available"""
        if not self.is_cuda:
            return True
        
        stats = self.get_memory_stats()
        available = stats.get("available_mb", 0)
        
        return available >= required_mb
    
    def reduce_batch_size(self, current_batch: int, target_memory_mb: float) -> int:
        """
        Calculate reduced batch size if memory is constrained.
        
        Args:
            current_batch: Current batch size
            target_memory_mb: Target memory usage in MB
            
        Returns:
            Recommended batch size
        """
        if not self.is_cuda or current_batch <= 1:
            return current_batch
        
        stats = self.get_memory_stats()
        available = stats.get("available_mb", 1000)
        
        if available < target_memory_mb:
            # Reduce batch size proportionally
            reduction_ratio = available / target_memory_mb
            new_batch = max(1, int(current_batch * reduction_ratio))
            print(f"[CUDA Memory] Reduced batch from {current_batch} to {new_batch} (available: {available:.1f}MB)")
            return new_batch
        
        return current_batch
