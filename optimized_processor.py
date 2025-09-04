"""
Optimized Video Processor with Smart Memory Management
Implements C++-inspired memory pooling, frame buffering, and efficient resource management
"""

import os
import torch
import numpy as np
from PIL import Image
from collections import deque
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import logging
from typing import Optional, Tuple, Dict, Any
import psutil

# Configure logging to file only
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/video_processor.log'),
    ]
)
logger = logging.getLogger(__name__)

class MemoryPool:
    """Memory pool for efficient frame buffer management"""
    def __init__(self, pool_size: int = 5, frame_shape: Tuple[int, int, int] = (1080, 1920, 3)):
        self.pool_size = pool_size
        self.frame_shape = frame_shape
        self.available_buffers = deque()
        self.in_use_buffers = set()
        self.lock = threading.Lock()
        
        # Pre-allocate buffers
        for _ in range(pool_size):
            buffer = np.empty(frame_shape, dtype=np.uint8)
            self.available_buffers.append(buffer)
    
    def acquire(self) -> Optional[np.ndarray]:
        """Get a buffer from the pool"""
        with self.lock:
            if self.available_buffers:
                buffer = self.available_buffers.popleft()
                self.in_use_buffers.add(id(buffer))
                return buffer
            return None
    
    def release(self, buffer: np.ndarray):
        """Return a buffer to the pool"""
        with self.lock:
            buffer_id = id(buffer)
            if buffer_id in self.in_use_buffers:
                self.in_use_buffers.remove(buffer_id)
                self.available_buffers.append(buffer)

class FrameBufferManager:
    """Manages frame buffering with sliding window for efficient memory usage"""
    def __init__(self, buffer_size: int = 3):
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.processed_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to buffer using copy-on-write semantics"""
        with self.lock:
            # Use view instead of copy when possible
            self.frame_buffer.append(frame.view())
    
    def get_batch(self, batch_size: int = 1) -> list:
        """Get batch of frames for processing"""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.frame_buffer))):
                if self.frame_buffer:
                    batch.append(self.frame_buffer.popleft())
            return batch

class OptimizedVideoProcessor:
    """
    Optimized video processor with smart resource management
    Inspired by C++ memory management techniques
    """
    
    def __init__(self, model=None, device='cpu'):
        self.model = model
        self.device = device
        self.memory_pool = None
        self.frame_buffer_manager = FrameBufferManager(buffer_size=3)
        self.batch_size = 1  # Process frames in batches
        
        # Resource monitoring
        self.max_memory_percent = 70  # Max memory usage allowed
        self.max_gpu_memory_percent = 80  # Max GPU memory usage allowed
        
        # Initialize memory pool based on available memory
        self._initialize_memory_pool()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Optimization flags
        self.use_mixed_precision = False  # Disabled for stability
        self.use_tensorrt = self._check_tensorrt()
        
    def _initialize_memory_pool(self):
        """Initialize memory pool based on available system resources"""
        try:
            # Get available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Calculate optimal pool size
            # Each 1080p frame is approximately 6MB
            frame_size_mb = 6
            max_frames = int((available_gb * 1024 * self.max_memory_percent / 100) / frame_size_mb)
            pool_size = min(max(3, max_frames // 4), 10)  # Between 3 and 10 frames
            
            self.memory_pool = MemoryPool(pool_size=pool_size)
            logger.info(f"Memory pool initialized with {pool_size} buffers")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory pool: {e}")
            self.memory_pool = MemoryPool(pool_size=3)  # Fallback to minimal pool
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available for optimization"""
        try:
            import torch_tensorrt
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def optimize_model(self):
        """Apply model optimizations"""
        if not self.model:
            return
        
        try:
            # Set model to eval mode
            self.model.eval()
            
            # Apply torch optimizations
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                # PyTorch 2.0+ compile for better performance
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("Model compiled with torch.compile")
            
            # Quantization for CPU (reduces memory and increases speed)
            if self.device == 'cpu':
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                logger.info("Model quantized for CPU")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
    
    def process_frame_batch(self, frames: list) -> list:
        """Process a batch of frames efficiently"""
        if not frames or not self.model:
            return frames
        
        try:
            # Stack frames into batch tensor
            batch_tensor = torch.stack([self._prepare_frame(f) for f in frames])
            
            # Move to device
            batch_tensor = batch_tensor.to(self.device)
            
            # Process with model
            with torch.no_grad():
                # Use autocast for mixed precision if available (disabled for now)
                outputs = self.model(batch_tensor)
                
                # Handle different output formats
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[-1]
                
                # Apply sigmoid and move to CPU
                outputs = outputs.sigmoid().cpu()
            
            # Clear GPU cache immediately
            if self.device != 'cpu':
                torch.cuda.empty_cache()
            
            # Convert outputs back to frames
            processed_frames = []
            for i, output in enumerate(outputs):
                processed_frame = self._apply_mask(frames[i], output)
                processed_frames.append(processed_frame)
            
            return processed_frames
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return frames
    
    def _prepare_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Prepare frame for model input"""
        # Convert to PIL
        if isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        
        # Resize if needed
        if frame.size != (1024, 1024):
            frame = frame.resize((1024, 1024), Image.LANCZOS)
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(np.array(frame)).float() / 255.0
        
        # Add batch dimension and reorder channels
        if len(frame_tensor.shape) == 3:
            frame_tensor = frame_tensor.permute(2, 0, 1)
        
        return frame_tensor
    
    def _apply_mask(self, original_frame: np.ndarray, mask: torch.Tensor) -> np.ndarray:
        """Apply segmentation mask to frame"""
        # Convert mask to numpy
        mask_np = mask.squeeze().numpy()
        
        # Resize mask to original frame size
        h, w = original_frame.shape[:2]
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((w, h), Image.LANCZOS)
        mask_array = np.array(mask_resized) / 255.0
        
        # Apply mask
        if len(mask_array.shape) == 2:
            mask_array = np.stack([mask_array] * 3, axis=-1)
        
        # Composite with transparency or background
        result = original_frame * mask_array
        
        return result.astype(np.uint8)
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor system resources"""
        stats = {}
        
        try:
            # CPU and RAM
            stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            stats['ram_percent'] = memory.percent
            stats['ram_used_gb'] = memory.used / (1024**3)
            
            # GPU and VRAM
            if torch.cuda.is_available():
                stats['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
                stats['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024**3)
                
                # Try to get GPU utilization
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        stats['gpu_percent'] = gpu.load * 100
                        stats['vram_percent'] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                except:
                    pass
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear frame buffers
            self.frame_buffer_manager.frame_buffer.clear()
            self.frame_buffer_manager.processed_buffer.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Shutdown executor
            self.executor.shutdown(wait=False)
            
            logger.info("Resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# Singleton instance for global access
_processor_instance = None

def get_optimized_processor(model=None, device='cpu') -> OptimizedVideoProcessor:
    """Get or create the optimized processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = OptimizedVideoProcessor(model, device)
    elif model is not None:
        _processor_instance.model = model
        _processor_instance.device = device
    return _processor_instance

def cleanup_processor():
    """Clean up the processor instance"""
    global _processor_instance
    if _processor_instance:
        _processor_instance.cleanup()
        _processor_instance = None
