from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from transformers import AutoModelForImageSegmentation, AutoImageProcessor
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
import tempfile
import uuid
import time
import sys
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
# Optimization modules removed - using built-in optimizations
OPTIMIZATIONS_AVAILABLE = False
from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips, vfx
from werkzeug.utils import secure_filename

# Set FFMPEG path for moviepy
os.environ['IMAGEIO_FFMPEG_EXE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin', 'ffmpeg.exe')

# Create bin directory if it doesn't exist
bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')
if not os.path.exists(bin_dir):
    os.makedirs(bin_dir)
    print(f"Created bin directory at {bin_dir}")
    print("Please download ffmpeg.exe and place it in the bin directory")
    print("Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip")
    print("Extract ffmpeg.exe from the zip and place it in the bin directory")
    sys.exit(1)

# Check if FFMPEG exists
ffmpeg_path = os.path.join(bin_dir, 'ffmpeg.exe')
if not os.path.exists(ffmpeg_path):
    print(f"ERROR: ffmpeg.exe not found at {ffmpeg_path}")
    print("Please download ffmpeg.exe and place it in the bin directory")
    print("Download from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip")
    print("Extract ffmpeg.exe from the zip and place it in the bin directory")
    sys.exit(1)
else:
    print(f"Found ffmpeg at {ffmpeg_path}")

import io
import base64
import json
import multiprocessing
import signal
import atexit
import psutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-background-remover'

# CORS: allow all origins to support Electron (file://) and localhost
# Keeping it permissive because this is a local desktop app.
CORS(app, resources={r"/*": {"origins": "*"}})
# Allow all origins for SocketIO to support Electron's file:// origin
# Use polling transport only for better stability
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', 
                   transports=['polling'], # Polling only for stability
                   ping_timeout=60, ping_interval=25,
                   allow_upgrades=False) # Prevent WebSocket upgrade issues

# Debug mode - set to True to debug GPU detection issues
DEBUG_MODE = True  # Enable full debugging to identify GPU issues [[memory:6712208]]

# Import hardware optimizer
try:
    from hardware_optimizer import get_hardware_optimizer
    hardware_optimizer = get_hardware_optimizer()
    hw_settings = hardware_optimizer.get_optimized_settings()
    hardware_optimizer.print_optimization_report()
except Exception as e:
    print(f"Warning: Could not load hardware optimizer: {e}")
    print("Using default settings...")
    hw_settings = {
        'device': 'cpu',
        'torch_threads': 2,
        'max_workers': 2,
        'batch_size': 1,
        'frame_buffer_size': 30,
        'use_mixed_precision': False,
        'use_model_quantization': False,
        'model_preference': 'fast'
    }

# Configure PyTorch based on hardware
torch.set_float32_matmul_precision("high" if hw_settings['device'] != 'cpu' else "medium")
torch.set_num_threads(hw_settings['torch_threads'])
torch.set_num_interop_threads(hw_settings['torch_threads'])

# Enable CPU optimizations if needed
if hw_settings['device'] == 'cpu':
    try:
        torch.set_flush_denormal(True)
    except:
        pass

# Set device based on hardware detection
print("\n" + "="*60)
print("GPU DETECTION DEBUG LOG")
print("="*60)

# Step 1: Check PyTorch version and CUDA compilation
print(f"1. PyTorch version: {torch.__version__}")
print(f"2. PyTorch compiled with CUDA: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'No CUDA support'}")

# Step 2: Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"3. torch.cuda.is_available(): {cuda_available}")

if cuda_available:
    # Step 3: Get GPU details
    print(f"4. Number of GPUs: {torch.cuda.device_count()}")
    print(f"5. Current GPU index: {torch.cuda.current_device()}")
    print(f"6. GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"7. GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Step 4: Test GPU tensor creation
    try:
        test_tensor = torch.zeros(1, 1).cuda()
        print(f"8. Test tensor on GPU: Success - {test_tensor.device}")
        del test_tensor
    except Exception as e:
        print(f"8. Test tensor on GPU: Failed - {e}")
        cuda_available = False
    
    if cuda_available:
        device = 'cuda'
        print(f"\nGPU DETECTED AND WORKING! Using: {torch.cuda.get_device_name(0)}")
        
        # Update hw_settings to reflect actual GPU availability
        hw_settings['device'] = 'cuda'
        
        # Force reload hardware settings to get GPU-optimized parameters
        try:
            hardware_optimizer = get_hardware_optimizer()
            gpu_settings = hardware_optimizer.get_optimized_settings()
            if gpu_settings['device'] == 'cuda':
                hw_settings.update(gpu_settings)
                print(f"9. Loaded GPU-optimized settings:")
                print(f"   - Batch size: {hw_settings['batch_size']}")
                print(f"   - Workers: {hw_settings['max_workers']}")
                print(f"   - Mixed precision: {hw_settings.get('use_mixed_precision', False)}")
        except Exception as e:
            print(f"9. Failed to load GPU settings: {e}")
    else:
        device = 'cpu'
        print("\nWarning: GPU detected but not functional, falling back to CPU")
        hw_settings['device'] = 'cpu'
else:
    device = 'cpu'
    print("\nWarning: CUDA not available, using CPU")
    print("Possible reasons:")
    print("  - No NVIDIA GPU present")
    print("  - NVIDIA drivers not installed")
    print("  - PyTorch installed without CUDA support")
    print("  - CUDA version mismatch")
    hw_settings['device'] = 'cpu'

print("="*60)
print(f"FINAL DEVICE: {device}")
print("="*60 + "\n")

# Global settings from hardware optimizer
MAX_BATCH_SIZE = hw_settings['batch_size'] if device != 'cpu' else 1
FRAME_BUFFER_SIZE = hw_settings['frame_buffer_size']
USE_MIXED_PRECISION = hw_settings['use_mixed_precision'] and device == 'cuda'
USE_QUANTIZATION = hw_settings['use_model_quantization']
MODEL_PREFERENCE = hw_settings['model_preference']

print(f"Settings: Batch size={MAX_BATCH_SIZE}, Mixed precision={USE_MIXED_PRECISION}")

# Model cache directory
MODEL_CACHE_DIR = "./model_cache"

# Initialize model variables - these will be cached
birefnet = None
birefnet_lite = None
models_available = True
models_loaded = False  # Track if models have been loaded

def has_sufficient_memory_for_full_model():
    try:
        import psutil
        memory = psutil.virtual_memory()
        # Require at least 8GB available for full model
        return memory.available >= 8 * (1024 ** 3)
    except Exception:
        return False

def check_memory_available():
    """Check if there's enough memory to load models"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        print(f"Available memory: {available_gb:.2f} GB")
        # Require at least 2GB free memory to load lite models (more reasonable)
        # BiRefNet_lite should work with 2-3GB available memory
        return available_gb >= 2.0
    except Exception as e:
        print(f"Error checking memory: {e}")
        return True  # Assume it's OK if we can't check

def monitor_memory_usage(session_id):
    """Monitor memory usage during processing"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"[{session_id}] DEBUG: Current memory usage: {memory.percent:.1f}%")
        
        if memory.percent > 95:  # Increased threshold to 95%
            print(f"[{session_id}] CRITICAL: Memory usage at {memory.percent:.1f}%")
            # Force garbage collection before stopping
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        elif memory.percent > 85:  # Warning at 85%
            print(f"[{session_id}] WARNING: Memory usage at {memory.percent:.1f}%")
            # Proactive cleanup at warning level
            gc.collect()
        return True
    except Exception as e:
        print(f"[{session_id}] DEBUG: Error checking memory: {e}")
        return True

def force_memory_cleanup():
    """Aggressive memory cleanup"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Additional cleanup for PyTorch tensors
        import torch
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_emptyCache'):
            torch._C._cuda_emptyCache()
    except Exception as e:
        print(f"Memory cleanup warning: {e}")

def cleanup_on_exit():
    """Cleanup function called on exit"""
    try:
        # Kill any remaining threads
        import threading
        for thread in threading.enumerate():
            if thread.name and "processing" in thread.name:
                print(f"Stopping thread: {thread.name}")
        
        # Clear models from memory
        global birefnet, birefnet_lite
        birefnet = None
        birefnet_lite = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Cleanup completed")
    except Exception as e:
        print(f"Cleanup warning: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

def quantize_model(model):
    """Apply quantization to reduce memory and improve speed"""
    if not USE_QUANTIZATION:
        return model
    
    try:
        if device == "cpu":
            # Dynamic quantization for CPU
            import torch.quantization as quantization
            model_int8 = quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
            print("Model quantized for CPU (INT8)")
            return model_int8
        elif device == "cuda":
            # Half precision for GPU - much faster on GTX 1650
            model = model.half()
            print("Model converted to FP16 for GPU acceleration")
            return model
        else:
            return model
    except Exception as e:
        print(f"Warning: Could not quantize model: {e}")
        return model

def preload_models():
    """Preload models based on hardware capabilities"""
    global models_loaded, birefnet, birefnet_lite
    
    if models_loaded:
        return True
    
    print("\n" + "="*60)
    print("MODEL LOADING DEBUG LOG")
    print("="*60)
    print(f"Device for models: {device}")
    print(f"Hardware profile: {hw_settings.get('profile', 'unknown')}")
    print(f"Model preference: {MODEL_PREFERENCE}")
    print(f"CUDA available at model load: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU for models: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    try:
        # Load models based on hardware profile
        if MODEL_PREFERENCE in ['ultra_fast', 'fast'] or hw_settings.get('profile') in ['low', 'potato']:
            # Only load lite model for low-end systems
            if birefnet_lite is None:
                print("Loading BiRefNet_lite model (optimized for your hardware)...")
                birefnet_lite = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet_lite",
                    trust_remote_code=True,
                    cache_dir=MODEL_CACHE_DIR
                )
                birefnet_lite.to(device).eval()
                
                # Apply quantization if needed
                birefnet_lite = quantize_model(birefnet_lite)
                print("Lite model ready!")
        else:
            # Load both models for high-end systems
            if birefnet_lite is None:
                print(f"Loading BiRefNet_lite model...")
                print(f"Target device: {device}")
                birefnet_lite = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet_lite",
                    trust_remote_code=True,
                    cache_dir=MODEL_CACHE_DIR
                )
                print(f"Model loaded, moving to device: {device}")
                birefnet_lite = birefnet_lite.to(device)
                birefnet_lite.eval()
                
                # Verify model is on correct device
                if hasattr(birefnet_lite, 'device'):
                    print(f"Model device after .to(): {birefnet_lite.device}")
                
                # Check a parameter to verify device
                for name, param in list(birefnet_lite.named_parameters())[:1]:
                    print(f"First parameter '{name}' is on: {param.device}")
                    break
                
                if device == 'cuda':
                    print(f" BiRefNet_lite loaded to GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print(f"Model loaded to: {device}")
                    
                birefnet_lite = quantize_model(birefnet_lite)
                
            if birefnet is None and hw_settings.get('profile') in ['ultra', 'high']:
                print("Loading BiRefNet full model for quality mode...")
                try:
                    birefnet = AutoModelForImageSegmentation.from_pretrained(
                        "ZhengPeng7/BiRefNet",
                        trust_remote_code=True,
                        cache_dir=MODEL_CACHE_DIR
                    )
                    birefnet.to(device).eval()
                    birefnet = quantize_model(birefnet)
                    print("Full model ready!")
                except Exception as e:
                    print(f"Could not load full model: {e}")
                    birefnet = None
        
        models_loaded = True
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def load_models_if_needed():
    global birefnet, birefnet_lite, models_available, models_loaded
    
    debug_log("\n=== MODEL LOADING DEBUG ===", "system")
    debug_log(f"models_loaded: {models_loaded}", "system")
    debug_log(f"models_available: {models_available}", "system")
    debug_log(f"birefnet_lite is None: {birefnet_lite is None}", "system")
    debug_log(f"birefnet is None: {birefnet is None}", "system")
    
    # Return immediately if models are already loaded
    if models_loaded:
        debug_log("Models already loaded and cached", "system")
        return True
    
    # Only load models if they haven't been loaded yet and are available
    if not models_available:
        debug_log("Models are not available due to system constraints", "system")
        return False
    
    # Check memory before attempting to load
    if not check_memory_available():
        debug_log("Insufficient memory to load models. Please free up memory and try again.", "system")
        models_available = False
        return False
    
    debug_log("Loading BiRefNet models (this will be cached)...", "system")
    try:
        # Load the lite model with memory optimizations
        if birefnet_lite is None:
            print("Loading BiRefNet_lite model with memory optimizations...")
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Attempting to load from HuggingFace...")
            birefnet_lite = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_lite", 
                trust_remote_code=True,
                cache_dir="./model_cache"  # Cache models locally
            )
            print(f"Model loaded, type: {type(birefnet_lite)}")
            
            birefnet_lite.to(device)
            birefnet_lite.eval()  # Set to evaluation mode
            
            # Force garbage collection after loading
            gc.collect()
            print("BiRefNet_lite model loaded with optimizations")
        
        # Always try to load full model for quality mode support
        if birefnet is None:
            try:
                print("Loading BiRefNet full model for quality mode...")
                gc.collect()
                birefnet = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet", 
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )
                birefnet.to(device)
                birefnet.eval()
                gc.collect()
                print("BiRefNet full model loaded successfully")
            except Exception as e2:
                print(f"Could not load full model, quality mode will use lite model: {e2}")
                birefnet = None  # Ensure it's None if loading failed
        
        models_loaded = True
        print(f"\nFinal state - models_loaded: {models_loaded}, birefnet_lite available: {birefnet_lite is not None}")
        print("All models loaded successfully and cached!")
        print("=== END MODEL LOADING DEBUG ===\n")
        return True
    except Exception as e:
        print(f"Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        models_available = False
        models_loaded = False
        return False

# Defer model loading to first use to avoid startup memory issues
print("Initializing API server...")
try:
    # Create model cache directory if it doesn't exist
    os.makedirs("./model_cache", exist_ok=True)
    print("Model cache directory ready")
    print("Models will be loaded on first request to conserve memory")
    # Do NOT load models at startup - wait for first request
    models_available = True
except Exception as e:
    print(f"Error setting up model cache: {e}")
    models_available = True  # Still allow trying to load later

transform_image = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Store for active processing sessions
active_sessions = {}

# Store temp files for cleanup
temp_files = set()

def cleanup_temp_file(filepath):
    """Clean up a temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up temp file: {filepath}")
            if filepath in temp_files:
                temp_files.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up {filepath}: {e}")

def cleanup_all_temp_files():
    """Clean up all temporary files"""
    print("Cleaning up all temporary files...")
    # Clean up tracked temp files
    for filepath in list(temp_files):
        cleanup_temp_file(filepath)
    
    # Also clean up any temp files matching our patterns
    patterns = [
        'temp_output_*.mp4', 'temp_output_*.webm', 'temp_output_*.mov',
        'temp_output_*_transparent.mov',  # Add pattern for transparent MOV files
        'temp_output_*_preview.webm',     # Add pattern for MOV preview files
        'temp_video_*.mp4', 'input_*.*', 'bg_*.*'
    ]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            cleanup_temp_file(filepath)
    
    # Clean up frame directories
    import shutil
    for frame_dir in glob.glob('frames_*'):
        try:
            shutil.rmtree(frame_dir)
            print(f"Removed frame directory: {frame_dir}")
        except Exception as e:
            print(f"Could not remove frame directory {frame_dir}: {e}")
    
    print("Temp file cleanup complete")

def process_frames_batch_gpu(frames, model, batch_size=4):
    """Process multiple frames in batches on GPU for better performance"""
    if device != "cuda" or not frames:
        return None
    
    try:
        processor = AutoImageProcessor.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
        results = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:min(i+batch_size, len(frames))]
            
            # Prepare batch
            batch_inputs = processor(images=batch, return_tensors="pt")
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            # Process batch
            with torch.no_grad():
                if USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch_inputs)
                else:
                    outputs = model(**batch_inputs)
            
            # Extract predictions
            predictions = outputs.logits.sigmoid()
            
            for pred in predictions:
                # Convert to mask
                mask = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
                results.append(mask)
        
        return results
    except Exception as e:
        print(f"GPU batch processing failed: {e}")
        return None

def process_frame_simple(frame, bg_type, bg, fast_mode, bg_frame_index, background_frames, color):
    """Simplified frame processing that maintains consistent dimensions"""
    try:
        # Keep original frame dimensions
        original_shape = frame.shape
        frame_id = f"Frame_{bg_frame_index}"
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Process based on background type
        if bg_type == "Transparent":
            processed_image = process_image(pil_image, None, fast_mode, transparent=True)
        elif bg_type == "Color":
            processed_image = process_image(pil_image, color, fast_mode, transparent=False)
        elif bg_type == "Image":
            processed_image = process_image(pil_image, bg, fast_mode, transparent=False)
        elif bg_type == "Video":
            if background_frames and len(background_frames) > 0 and bg_frame_index < len(background_frames):
                background_frame = background_frames[bg_frame_index]
                background_image = Image.fromarray(background_frame)
                processed_image = process_image(pil_image, background_image, fast_mode, transparent=False)
            else:
                processed_image = process_image(pil_image, color, fast_mode, transparent=False)
        else:
            processed_image = pil_image
        
        # Convert back to numpy array
        if isinstance(processed_image, Image.Image):
            result = np.array(processed_image)
        else:
            result = processed_image
            
        # Ensure dimensions match original
        if result.shape[:2] != original_shape[:2]:
            pil_result = Image.fromarray(result)
            pil_result = pil_result.resize((original_shape[1], original_shape[0]), Image.LANCZOS)
            result = np.array(pil_result)
        
        # No verbose frame verification logging
        return result, bg_frame_index
    except Exception as e:
        # Frame processing error - return original
        return frame, bg_frame_index

# REMOVED DUPLICATE - Using process_frame_simple instead



def process_image(image, bg, fast_mode=False, transparent=False):
    """
    Process an image by removing its background and replacing with the specified background.
    CRITICAL: Returns consistent image format for video processing.
    """
    debug_log(f"[AI] process_image called - fast_mode={fast_mode}, transparent={transparent}, bg type={type(bg)}", "ai_process")
    
    # Store original size and mode
    image_size = image.size
    original_mode = image.mode
    debug_log(f"[AI] Original image size: {image_size}, mode: {original_mode}", "ai_process")
    
    # Check if models are loaded
    debug_log(f"[AI] Model status - birefnet: {'Loaded' if birefnet is not None else 'Not loaded'}, birefnet_lite: {'Loaded' if birefnet_lite is not None else 'Not loaded'}", "ai_process")
    
    # Caching disabled for simplicity
    
    debug_log(f"[AI] Preparing image tensor for AI model...", "ai_process")
    input_images = transform_image(image).unsqueeze(0)
    debug_log(f"[AI] Tensor shape: {input_images.shape}, initial device: {input_images.device}", "ai_process")
    debug_log(f"[AI] Moving tensor to device: {device}", "ai_process")
    input_images = input_images.to(device)
    debug_log(f"[AI] Tensor now on device: {input_images.device}", "ai_process")
    
    # Use mixed precision if enabled
    if USE_MIXED_PRECISION and device != 'cpu':
        debug_log(f"[AI] Using mixed precision (FP16) on GPU", "ai_process")
        with torch.cuda.amp.autocast():
            input_images = input_images.half()
    
    debug_log(f"[AI] Input tensor ready - shape: {input_images.shape}, device: {input_images.device}, dtype: {input_images.dtype}", "ai_process")
    
    # FIXED MODEL SELECTION: Use BiRefNet for quality, BiRefNet_lite for fast
    if fast_mode:
        # Fast mode: prefer lite model
        model = birefnet_lite if birefnet_lite is not None else birefnet
        model_name = "birefnet_lite (fast)" if birefnet_lite is not None else "birefnet (fallback)"
    else:
        # Quality mode: prefer full model
        model = birefnet if birefnet is not None else birefnet_lite
        model_name = "birefnet (quality)" if birefnet is not None else "birefnet_lite (fallback)"
    
    if model is None:
        debug_log("[AI] ERROR: No AI model available, returning original image", "ai_process")
        return image
    
    debug_log(f"[AI] Using model: {model_name}", "ai_process")
    
    # Check model device
    for name, param in list(model.named_parameters())[:1]:
        debug_log(f"[AI] Model parameter '{name}' is on device: {param.device}", "ai_process")
        break
    
    try:
        with torch.no_grad():
            debug_log("[AI] Running AI inference to detect foreground...", "ai_process")
            start_inference = time.time()
            
            # Use mixed precision if available
            if USE_MIXED_PRECISION and device != 'cpu':
                with torch.cuda.amp.autocast():
                    preds = model(input_images)[-1].sigmoid()
            else:
                preds = model(input_images)[-1].sigmoid()
            
            preds = preds.cpu()
            inference_time = time.time() - start_inference
            debug_log(f"[AI] AI inference complete in {inference_time:.2f}s - predictions shape: {preds.shape}", "ai_process")
    except Exception as e:
        debug_log(f"[AI] ERROR in model inference: {e}", "ai_process")
        import traceback
        debug_log(f"[AI] Stack trace: {traceback.format_exc()}", "ai_process")
        return image
    
    debug_log("[AI] Creating mask from AI predictions...", "ai_process")
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    debug_log(f"[AI] Mask created - size: {mask.size}, mode: {mask.mode}", "ai_process")
    
    # Immediately free memory
    debug_log("[AI] Freeing GPU/CPU memory...", "ai_process")
    del input_images, preds, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # If transparent background requested, return RGBA image
    debug_log(f"[AI]  TRANSPARENCY DECISION IN process_image:", "ai_process")
    debug_log(f"[AI]   - transparent param: {transparent}", "ai_process")
    debug_log(f"[AI]   - type(transparent): {type(transparent)}", "ai_process")
    if transparent:
        debug_log("[AI]  TRANSPARENT PATH SELECTED - Creating RGBA with alpha channel", "ai_process")
        image_rgba = image.convert("RGBA")
        
        # CRITICAL FIX: Also clear RGB channels where transparent!
        # Convert mask to numpy for processing
        import numpy as np
        mask_np = np.array(mask)
        image_np = np.array(image_rgba)
        
        # Where mask is 0 (transparent), set RGB to black (or could be white)
        # This ensures that if alpha is lost, we see black, not original background
        transparent_pixels = mask_np < 128  # Threshold for transparency
        image_np[transparent_pixels] = [0, 0, 0, 0]  # Black with full transparency
        
        # Convert back to PIL Image
        image_rgba = Image.fromarray(image_np, 'RGBA')
        
        # Apply the mask properly
        image_rgba.putalpha(mask)
        
        debug_log(f"[AI]  Transparent RGBA image created", "ai_process")
        debug_log(f"[AI]   - Size: {image_rgba.size}", "ai_process")
        debug_log(f"[AI]   - Mode: {image_rgba.mode}", "ai_process")
        debug_log(f"[AI]   - Has alpha: {'A' in image_rgba.mode}", "ai_process")
        # Check alpha channel values
        alpha_array = np.array(image_rgba)[:, :, 3]
        unique_alpha = np.unique(alpha_array)
        debug_log(f"[AI]   - Unique alpha values (first 10): {unique_alpha[:10] if len(unique_alpha) > 10 else unique_alpha}", "ai_process")
        debug_log(f"[AI]   - Total unique alpha values: {len(unique_alpha)}", "ai_process")
        return image_rgba
    
    # For non-transparent, ensure RGB format for video consistency
    debug_log("[AI] ❌ NON-TRANSPARENT PATH - Applying background to image...", "ai_process")
    debug_log(f"[AI]   - transparent was False or None", "ai_process")
    if isinstance(bg, str) and bg.startswith("#"):
        debug_log(f"[AI] Creating solid COLOR background: {bg}", "ai_process")
        color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
        debug_log(f"[AI] RGB values: {color_rgb}", "ai_process")
        background = Image.new("RGB", image_size, color_rgb)
    elif isinstance(bg, Image.Image):
        debug_log(f"[AI] Using IMAGE background - size: {bg.size}, mode: {bg.mode}", "ai_process")
        background = bg.convert("RGB").resize(image_size)
        debug_log(f"[AI] Background resized to: {background.size}", "ai_process")
    elif bg is None:
        debug_log(f"[AI] No background specified, using white", "ai_process")
        background = Image.new("RGB", image_size, (255, 255, 255))
    else:
        debug_log(f"[AI] Loading background from file: {bg}", "ai_process")
        try:
            background = Image.open(bg).convert("RGB").resize(image_size)
            debug_log(f"[AI] Background loaded from file successfully", "ai_process")
        except Exception as e:
            debug_log(f"[AI] Failed to open background file: {e}, using white fallback", "ai_process")
            background = Image.new("RGB", image_size, (255, 255, 255))
    
    # Composite the image - ensure RGB output
    debug_log("[AI] Compositing foreground over background...", "ai_process")
    image_rgb = image.convert("RGB")
    result = Image.composite(image_rgb, background, mask)
    
    debug_log(f"[AI] Final image created - size: {result.size}, mode: {result.mode}", "ai_process")
    return result

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def process_video_async(session_id, video_path, bg_type, bg_path, color, fps, video_handling, fast_mode, max_workers, output_format='mp4'):
    debug_log(f"\n[{session_id}] === ASYNC PROCESSING STARTED ===", session_id)
    debug_log(f"[{session_id}] Thread ID: {threading.current_thread().ident}", session_id)
    debug_log(f"[{session_id}] Video path exists: {os.path.exists(video_path)}", session_id)
    debug_log(f"[{session_id}]  CRITICAL PARAMETERS:", session_id)
    debug_log(f"[{session_id}]   - bg_type = '{bg_type}' (type: {type(bg_type).__name__})", session_id)
    debug_log(f"[{session_id}]   - output_format = '{output_format}' (type: {type(output_format).__name__})", session_id)
    debug_log(f"[{session_id}]   - fast_mode = {fast_mode}", session_id)
    debug_log(f"[{session_id}]   - Is bg_type == 'Transparent'? {bg_type == 'Transparent'}", session_id)
    debug_log(f"[{session_id}]   - output_format.lower() = '{output_format.lower()}'", session_id)
    
    # Check if models are loaded
    global birefnet, birefnet_lite
    debug_log(f"[{session_id}] Model check - birefnet: {birefnet is not None}, birefnet_lite: {birefnet_lite is not None}", session_id)
    if birefnet is None and birefnet_lite is None:
        debug_log(f"[{session_id}] ERROR: No models loaded!", session_id)
        socketio.emit('processing_error', {
            'session_id': session_id,
            'status': 'error',
            'message': 'AI models not loaded. Please wait for models to initialize.',
            'elapsed_time': 0
        })
        return
    
    video = None
    background_video = None
    processed_video = None
    
    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        debug_log(f"[{session_id}] Starting video processing with settings: bg_type={bg_type}, fast_mode={fast_mode}, max_workers={max_workers}", session_id)
        start_time = time.time()
        debug_log(f"[{session_id}] Loading video from: {video_path}", session_id)
        video = VideoFileClip(video_path)
        debug_log(f"[{session_id}] Video loaded successfully: duration={video.duration}s, fps={video.fps}", session_id)
        if fps == 0:
            fps = video.fps
            debug_log(f"[{session_id}] Using original video FPS: {fps}", session_id)
        else:
            debug_log(f"[{session_id}] Using custom FPS: {fps}", session_id)
        
        # Store audio reference before extracting frames
        audio = video.audio
        debug_log(f"[{session_id}] Loading video frames...", session_id)
        
        # Load frames based on available memory
        debug_log(f"[{session_id}] Loading video frames...", session_id)
        frames = list(video.iter_frames(fps=fps))
        total_frames = len(frames)
        debug_log(f"[{session_id}] Total frames to process: {total_frames}", session_id)
        
        debug_log(f"[{session_id}] Emitting processing_update event to client", session_id)
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'started',
            'message': 'Processing started... Extracting frames',
            'progress': 0,
            'elapsed_time': 0,
            'currentFrame': 0,
            'totalFrames': total_frames
        })
        debug_log(f"[{session_id}] Event emitted", session_id)
        
        processed_frames = []
        
        # Handle background setup
        background_frames = None
        bg_image = None
        
        if bg_type == "Video" and bg_path:
            debug_log(f"[{session_id}] Loading background video: {bg_path}", session_id)
            try:
                background_video = VideoFileClip(bg_path)
                video_duration = video.duration
                bg_duration = background_video.duration
                
                if bg_duration < video_duration:
                    if video_handling == "slow_down":
                        print(f"[{session_id}] Slowing down background video to match duration")
                        # Create a new slowed down video instead of modifying the original
                        speed_factor = video_duration / bg_duration
                        old_bg = background_video
                        background_video = old_bg.fx(vfx.speedx, factor=speed_factor)
                        old_bg.close()  # Close the original
                    else:  # loop
                        print(f"[{session_id}] Looping background video to match duration")
                        # Calculate how many times to loop
                        loop_count = int(video_duration / bg_duration + 1)
                        clips_to_concat = [background_video] * loop_count
                        old_bg = background_video
                        background_video = concatenate_videoclips(clips_to_concat)
                        old_bg.close()  # Close the original
                print(f"[{session_id}] Background video loaded (duration adjusted)")
                socketio.emit('processing_update', {
                    'session_id': session_id,
                    'status': 'processing',
                    'message': 'Background video prepared',
                    'progress': 0,
                    'elapsed_time': time.time() - start_time,
                    'currentFrame': 0,
                    'totalFrames': total_frames
                })
            except Exception as bg_load_err:
                print(f"[{session_id}] Error loading background video: {bg_load_err}")
                if background_video:
                    background_video.close()
                background_video = None
                bg_type = "Color"  # Fallback to color background
        elif bg_type == "Image" and bg_path:
            print(f"[{session_id}] Loading background image: {bg_path}")
            bg_image = Image.open(bg_path)
        elif bg_type == "Color":
            print(f"[{session_id}] Using background color: {color}")
        
        bg_frame_index = 0
        
        # Limit to 200 frames max like the original Gradio implementation
        if total_frames > 200:
            print(f"[{session_id}] Warning: Video has {total_frames} frames, limiting to 200 for memory safety")
            frames = frames[:200]
            total_frames = len(frames)
        
        # Check if frames are very high resolution and suggest optimization
        if len(frames) > 0:
            frame_shape = frames[0].shape
            if frame_shape[0] > 2000 or frame_shape[1] > 2000:
                debug_log(f"[{session_id}] HIGH RESOLUTION DETECTED: {frame_shape[1]}x{frame_shape[0]} - This will be slow!", session_id)
                debug_log(f"[{session_id}] Each frame will take ~60 seconds to process with BiRefNet quality model", session_id)
                debug_log(f"[{session_id}] Consider using Fast Mode for faster processing of high-res videos", session_id)
        
        # Process all frames in a single batch like the original Gradio
        print(f"[{session_id}] Processing all {total_frames} frames with ThreadPoolExecutor")
        
        # Check if session was cancelled
        if session_id not in active_sessions:
            print(f"[{session_id}] Processing cancelled")
            return  # Session was cancelled
                
        # Prepare background frames if using video background
        background_frames = None
        if bg_type == "Video" and background_video is not None:
            try:
                print(f"[{session_id}] Loading background video frames")
                background_frames = list(background_video.iter_frames(fps=fps))
                if len(background_frames) < total_frames:
                    # Handle looping or speed adjustment
                    if video_handling == "loop":
                        # Loop background frames
                        multiplier = (total_frames // len(background_frames)) + 1
                        background_frames = background_frames * multiplier
                    # For "slow_down", we'll handle frame selection in process_frame
                print(f"[{session_id}] Loaded {len(background_frames)} background frames")
            except Exception as bg_err:
                print(f"[{session_id}] Error loading background frames: {bg_err}")
                background_frames = None

        # Check memory before processing
        if not monitor_memory_usage(session_id):
            print(f"[{session_id}] CRITICAL: Memory usage too high, cancelling processing")
            return
        
        # Use ThreadPoolExecutor for parallel processing - exactly like Gradio
        debug_log(f"[{session_id}] Starting ThreadPoolExecutor with max_workers={max_workers}", session_id)
        bg_frame_index = 0  # Initialize background frame index
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frames for parallel processing - exactly like Gradio
            futures = []
            debug_log(f"[{session_id}] Submitting {total_frames} frames for processing", session_id)
            for i in range(total_frames):
                futures.append(
                    executor.submit(
                        process_frame_simple,
                        frames[i],
                        bg_type,
                        bg_image,  # This will be None for video backgrounds
                        fast_mode,
                        bg_frame_index + i,
                        background_frames,
                        color
                    )
                )
            debug_log(f"[{session_id}] All frames submitted to executor", session_id)
            
            # Process results as they complete
            successfully_processed = 0
            failed_frames = []
            
            debug_log(f"[{session_id}] Starting to collect processed frames", session_id)
            for i, future in enumerate(futures):
                # Check if session was cancelled before processing each result
                if session_id not in active_sessions:
                    debug_log(f"[{session_id}] Processing cancelled during frame collection", session_id)
                    # Cancel all remaining futures
                    for remaining_future in futures[i:]:
                        remaining_future.cancel()
                    return  # Exit processing
                
                try:
                    debug_log(f"[{session_id}] Waiting for frame {i+1}/{total_frames}", session_id)
                    result, _ = future.result(timeout=120)  # Increased timeout for 4K processing
                    if result is None:
                        debug_log(f"[{session_id}] WARNING: Frame {i} returned None, using original as fallback!", session_id)
                        result = frames[i]  # Use original frame as fallback
                    else:
                        # Verify we got the processed frame with correct shape
                        debug_log(f"[{session_id}] Got processed frame {i+1} - Shape: {result.shape}, dtype: {result.dtype}", session_id)
                    
                    # DIAGNOSTIC: For first frame, save comparison to debug folder
                    if i == 0:  # Only save the first frame to avoid too many files
                        try:
                            # Note that we'll save all frames at the end of processing
                            # This is just to catch any early issues
                            
                            # Compare frames
                            if frames[i].shape != result.shape:
                                debug_log(f"[{session_id}] FRAME {i+1} CHANGED: {frames[i].shape} -> {result.shape}", session_id)
                            else:
                                are_identical = np.array_equal(frames[i], result)
                                debug_log(f"[{session_id}] FRAME {i+1} IDENTICAL: {are_identical}", session_id)
                            
                            # Report to console
                            if result.shape[2] == 4:
                                # Check if alpha channel has real transparency
                                alpha = result[:, :, 3]
                                unique_alpha = len(np.unique(alpha))
                                debug_log(f"[{session_id}] Frame {i+1} has alpha channel with {unique_alpha} unique values", session_id)
                                if unique_alpha <= 1:
                                    debug_log(f"[{session_id}] WARNING: Frame {i+1} alpha channel is uniform - no transparency!", session_id)
                                else:
                                    debug_log(f"[{session_id}] Frame {i+1} has proper transparency", session_id)
                        except Exception as compare_err:
                            debug_log(f"[{session_id}] Error comparing frames: {compare_err}", session_id)
                    
                    processed_frames.append(result)
                    successfully_processed += 1
                    debug_log(f"[{session_id}] Frame {i+1} collected successfully", session_id)
                    
                    # Save each frame as an image for frame slider [[memory:6712208]]
                    frame_dir = f"frames_{session_id}"
                    if not os.path.exists(frame_dir):
                        os.makedirs(frame_dir)
                        debug_log(f"[{session_id}] Created frame directory: {frame_dir}", session_id)
                    
                    # Save frame as image
                    frame_filename = f"{frame_dir}/frame_{i+1:04d}.png"
                    try:
                        if isinstance(result, np.ndarray):
                            frame_image = Image.fromarray(result.astype(np.uint8))
                        else:
                            frame_image = result
                        
                        # Save with appropriate format based on transparency
                        if bg_type == "Transparent" and frame_image.mode == 'RGBA':
                            frame_image.save(frame_filename, 'PNG')
                        else:
                            # Convert to RGB if needed for non-transparent
                            if frame_image.mode == 'RGBA':
                                frame_image = frame_image.convert('RGB')
                            frame_image.save(frame_filename, 'PNG')
                        
                        debug_log(f"[{session_id}]  Saved frame {i+1} to {frame_filename} - File exists: {os.path.exists(frame_filename)}", session_id)
                    except Exception as save_err:
                        debug_log(f"[{session_id}] ❌ ERROR saving frame {i+1}: {str(save_err)}", session_id)
                    
                    elapsed_time = time.time() - start_time
                    progress = ((i + 1) / total_frames) * 100
                    
                    # Convert frame to base64 for preview
                    preview_base64 = image_to_base64(frame_image)
                    
                    # Send update for EVERY frame (not just every 5 frames) for the slider
                    socketio.emit('processing_update', {
                        'session_id': session_id,
                        'status': 'processing',
                        'message': f'Processing frame {i+1}/{total_frames}',
                        'progress': progress,
                        'elapsed_time': elapsed_time,
                        'preview_image': preview_base64,
                        'currentFrame': i + 1,
                        'totalFrames': total_frames,
                        'frame_url': f"/frames/{session_id}/{i+1:04d}",  # URL to access saved frame
                        'all_frames': [f"/frames/{session_id}/{j+1:04d}" for j in range(i+1)]  # All processed frames so far
                    })
                    # Send progress to debug console [[memory:6712208]]
                    debug_log(f"[{session_id}] Progress: {progress:.1f}%, Frame: {i+1}/{total_frames} (Success: {successfully_processed}, Failed: {len(failed_frames)})", session_id)
                except Exception as frame_error:
                    debug_log(f"[{session_id}] ERROR processing frame {i}: {str(frame_error)}", session_id)
                    failed_frames.append(i)
                    # Use original frame as fallback
                    if i < len(frames):
                        processed_frames.append(frames[i])
                    else:
                        processed_frames.append(frames[0])
            
            debug_log(f"[{session_id}] Frame processing complete: {successfully_processed} succeeded, {len(failed_frames)} failed", session_id)
            if failed_frames:
                debug_log(f"[{session_id}] Failed frames: {failed_frames}", session_id)
                
        # Force cleanup after processing all frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if session_id not in active_sessions:
            print(f"[{session_id}] Processing cancelled before video creation")
            return  # Session was cancelled
        
        # Clear memory before final video creation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create final video
        debug_log(f"[{session_id}] Creating final video with {len(processed_frames)} processed frames", session_id)

        
        # Basic validation of processed frames
        if processed_frames and len(processed_frames) > 0:
            first_frame_shape = processed_frames[0].shape
            if bg_type == "Transparent" and len(first_frame_shape) == 3 and first_frame_shape[2] == 4:
                debug_log(f"[{session_id}] Using RGBA frames with transparency", session_id)
            elif bg_type == "Transparent" and len(first_frame_shape) == 3 and first_frame_shape[2] == 3:
                debug_log(f"[{session_id}] WARNING: Expected RGBA but got RGB - transparency may be lost", session_id)
        
        # Additional debug info
        debug_log(f"[{session_id}] 🔍 PRE-PROCESSING CHECK:", session_id)
        debug_log(f"[{session_id}]   - Output format: '{output_format}' (lower: '{output_format.lower()}')", session_id)
        debug_log(f"[{session_id}]   - Background type: '{bg_type}'", session_id)
        debug_log(f"[{session_id}]   - Is Transparent? {bg_type == 'Transparent'}", session_id)
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'processing',
            'message': 'Compiling frames into final video',
            'progress': 99,
            'elapsed_time': time.time() - start_time,
            'currentFrame': total_frames,
            'totalFrames': total_frames
        })
        
        try:
            # Ensure all frames have the same dimensions AND CHANNELS
            if processed_frames:
                first_shape = processed_frames[0].shape
                print(f"[{session_id}] First frame shape: {first_shape}, dtype: {processed_frames[0].dtype}")
                
                # Check for channel consistency
                has_transparency = False
                rgba_count = 0
                rgb_count = 0
                
                # Count how many frames are RGBA vs RGB
                for frame in processed_frames:
                    if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
                        rgba_count += 1
                    elif len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB
                        rgb_count += 1
                
                # Determine if we need transparency based on majority
                if rgba_count > rgb_count:
                    has_transparency = True
                    debug_log(f"[{session_id}] Most frames have transparency ({rgba_count} RGBA vs {rgb_count} RGB) - preserving transparency", session_id)
                else:
                    debug_log(f"[{session_id}] Most frames are RGB ({rgb_count} RGB vs {rgba_count} RGBA) - removing transparency", session_id)
                
                # Normalize all frames to the same shape and channel count
                normalized_frames = []
                for idx, frame in enumerate(processed_frames):
                    # First handle resolution differences
                    if frame.shape[:2] != first_shape[:2]:
                        debug_log(f"[{session_id}] Frame {idx} has different resolution: {frame.shape[:2]}, resizing to {first_shape[:2]}", session_id)
                        # Resize frame to match first frame resolution
                        pil_frame = Image.fromarray(frame.astype(np.uint8), mode='RGBA' if frame.shape[2] == 4 else 'RGB')
                        pil_frame = pil_frame.resize((first_shape[1], first_shape[0]), Image.LANCZOS)
                        frame = np.array(pil_frame)
                    
                    # Then handle channel differences
                    if has_transparency and frame.shape[2] == 3:  # Add alpha channel to RGB
                        debug_log(f"[{session_id}] Adding alpha channel to frame {idx}", session_id)
                        height, width = frame.shape[:2]
                        rgba_frame = np.ones((height, width, 4), dtype=np.uint8) * 255
                        rgba_frame[:, :, :3] = frame  # Copy RGB data
                        frame = rgba_frame
                    elif not has_transparency and frame.shape[2] == 4:  # Remove alpha channel from RGBA
                        debug_log(f"[{session_id}] Removing alpha channel from frame {idx}", session_id)
                        frame = frame[:, :, :3]  # Keep only RGB channels
                    
                    # Ensure uint8 dtype
                    if frame.dtype != np.uint8:
                        debug_log(f"[{session_id}] Converting frame {idx} from dtype {frame.dtype} to uint8", session_id)
                        frame = frame.astype(np.uint8)
                        
                    normalized_frames.append(frame)

                
                debug_log(f"[{session_id}] Creating ImageSequenceClip with {len(normalized_frames)} frames at {fps} fps", session_id)
                
                # Validate transparency support
                if normalized_frames and len(normalized_frames) > 0:
                    sample_frame = normalized_frames[0]
                    if bg_type == "Transparent" and len(sample_frame.shape) == 3:
                        if sample_frame.shape[2] == 4:
                            debug_log(f"[{session_id}] Frames have alpha channel for transparency", session_id)
                        else:
                            debug_log(f"[{session_id}] WARNING: Frames missing alpha channel", session_id)
                
                # CRITICAL FIX: Handle transparency by BYPASSING MoviePy and using direct FFmpeg
                # For ANY format with transparent background, we'll use MOV with PNG codec
                debug_log(f"[{session_id}] 🚨 TRANSPARENCY DECISION POINT 1:", session_id)
                debug_log(f"[{session_id}]   - bg_type = '{bg_type}'", session_id)
                debug_log(f"[{session_id}]   - bg_type == 'Transparent'? {bg_type == 'Transparent'}", session_id)
                debug_log(f"[{session_id}]   - Has normalized_frames? {normalized_frames is not None and len(normalized_frames) > 0}", session_id)
                if bg_type == "Transparent" and normalized_frames and len(normalized_frames) > 0:
                    sample = normalized_frames[0]
                    if len(sample.shape) == 3 and sample.shape[2] == 4:
                        debug_log(f"[{session_id}] Creating video with TRANSPARENCY support (RGBA frames detected)", session_id)
                        
                        # BYPASS MoviePy for transparency - save frames as PNG sequence and use direct FFmpeg
                        debug_log(f"[{session_id}] BYPASSING MoviePy - saving RGBA frames as PNG sequence for direct FFmpeg", session_id)
                        
                        # Create temporary directory for PNG frames
                        import tempfile
                        temp_dir = tempfile.mkdtemp(prefix=f"frames_{session_id}_")
                        debug_log(f"[{session_id}] Created temp directory: {temp_dir}", session_id)
                        
                        # Save all RGBA frames as PNG files (preserves transparency)
                        frame_paths = []
                        for idx, frame in enumerate(normalized_frames):
                            frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                            # Convert numpy RGBA to PIL and save as PNG
                            pil_frame = Image.fromarray(frame.astype(np.uint8), mode='RGBA')
                            pil_frame.save(frame_path, 'PNG')
                            frame_paths.append(frame_path)
                        
                        debug_log(f"[{session_id}] Saved {len(frame_paths)} RGBA frames as PNG files", session_id)
                        
                        # Try WebM with proper settings from example.py
                        debug_log(f"[{session_id}] Attempting WebM transparency with image2 format", session_id)
                        
                        # Determine output format based on what user selected
                        debug_log(f"[{session_id}]  FORMAT SELECTION FOR TRANSPARENCY:", session_id)
                        debug_log(f"[{session_id}]   - output_format = '{output_format}'", session_id)
                        debug_log(f"[{session_id}]   - output_format.lower() = '{output_format.lower()}'", session_id)
                        debug_log(f"[{session_id}]   - Is output_format.lower() == 'webm'? {output_format.lower() == 'webm'}", session_id)
                        if output_format.lower() == 'webm':
                            output_path = f"temp_output_{session_id}.webm"
                            use_webm = True
                            debug_log(f"[{session_id}]    SELECTED: WebM with transparency (use_webm=True)", session_id)
                        else:
                            output_path = f"temp_output_{session_id}_transparent.mov"
                            use_webm = False
                            debug_log(f"[{session_id}]    SELECTED: MOV with transparency (use_webm=False)", session_id)
                        debug_log(f"[{session_id}]   - output_path = '{output_path}'", session_id)
                        
                        abs_output_path = os.path.join(os.getcwd(), output_path)
                        
                        ffmpeg_path = os.path.join(os.getcwd(), 'bin', 'ffmpeg.exe')
                        if not os.path.exists(ffmpeg_path):
                            ffmpeg_path = 'ffmpeg'
                        
                        # Build FFmpeg command - FIXED like example.py!
                        # CRITICAL: Set framerate and format BEFORE input
                        ffmpeg_cmd = [
                            ffmpeg_path,
                            '-y',  # Overwrite output
                            '-framerate', str(fps),
                            '-f', 'image2',  # CRITICAL: Tell FFmpeg this is an image sequence
                            '-i', os.path.join(temp_dir, 'frame_%06d.png')
                        ]

                        debug_log(f"[{session_id}] 🎞️ CODEC SELECTION:", session_id)
                        debug_log(f"[{session_id}]   - use_webm = {use_webm}", session_id)
                        # Use standard encoding
                        if use_webm:
                            debug_log(f"[{session_id}]    USING WEBM WITH VP9 TRANSPARENCY", session_id)
                            ffmpeg_cmd.extend([
                                '-c:v', 'libvpx-vp9',
                                '-pix_fmt', 'yuva420p',
                                '-b:v', '0',
                                '-crf', '10',
                                '-row-mt', '1'
                            ])
                        else:
                            debug_log(f"[{session_id}]    USING MOV PATH WITH TRANSPARENCY", session_id)
                            ffmpeg_cmd.extend([
                                '-c:v', 'png',
                                '-pred', 'mixed'
                            ])
                        
                        # AUDIO DISABLED FOR DEBUGGING TRANSPARENCY!
                        # if audio is not None:
                        #     debug_log(f"[{session_id}] Adding audio to output...", session_id)
                        #     audio_path = os.path.join(temp_dir, 'audio.wav')
                        #     audio.write_audiofile(audio_path, logger=None)
                        #     
                        #     # Add audio input AFTER video settings
                        #     ffmpeg_cmd.extend(['-i', audio_path])
                        #     
                        #     # Audio codec based on format
                        #     if use_webm:
                        #         ffmpeg_cmd.extend(['-c:a', 'libopus'])
                        #     else:
                        #         ffmpeg_cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
                        #     
                        #     # Map both video and audio streams
                        #     ffmpeg_cmd.extend(['-map', '0:v', '-map', '1:a'])
                        #     debug_log(f"[{session_id}] Audio will be included in output", session_id)
                        debug_log(f"[{session_id}] Warning: AUDIO DISABLED FOR TRANSPARENCY DEBUGGING!", session_id)
                        
                        # Output file (must be last)
                        ffmpeg_cmd.append(abs_output_path)
                        
                        debug_log(f"[{session_id}] Running FFmpeg to create {'WebM' if use_webm else 'MOV'} with transparency", session_id)
                        debug_log(f"[{session_id}] FFmpeg command: {' '.join(ffmpeg_cmd)}", session_id)
                        
                        import subprocess
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            debug_log(f"[{session_id}] Direct FFmpeg encoding successful!", session_id)
                        else:
                            debug_log(f"[{session_id}] FFmpeg error: {result.stderr}", session_id)
                            raise Exception(f"FFmpeg failed: {result.stderr}")
                        
                        # Cleanup PNG frames - DISABLED FOR DEBUGGING!
                        import shutil
                        # shutil.rmtree(temp_dir)  # COMMENTED OUT TO INSPECT FRAMES!
                        debug_log(f"[{session_id}] 🔍 TEMP FRAMES KEPT FOR INSPECTION: {temp_dir}", session_id)
                        debug_log(f"[{session_id}] Warning: Remember to manually delete temp folders later!", session_id)
                        
                        # Create a dummy video object for compatibility
                        processed_video = None  # We'll skip MoviePy export
                        debug_log(f"[{session_id}] Direct transparency encoding complete - skipping MoviePy export", session_id)
                    else:
                        debug_log(f"[{session_id}] WARNING: Expected RGBA frames for transparency but got shape {sample.shape}", session_id)
                        processed_video = ImageSequenceClip(normalized_frames, fps=fps)
                else:
                    # Non-transparent video - use frames as-is
                    debug_log(f"[{session_id}] Creating standard video (no transparency)", session_id)
                    processed_video = ImageSequenceClip(normalized_frames, fps=fps)
            
            # Only log duration if processed_video exists (not None from direct FFmpeg path)
            if processed_video is not None:
                debug_log(f"[{session_id}] ImageSequenceClip created successfully, duration: {processed_video.duration}s", session_id)
            
            # Try to add audio if available
            if audio is not None and processed_video is not None:
                try:
                    debug_log(f"[{session_id}] Attempting to add audio...", session_id)
                    processed_video = processed_video.set_audio(audio)
                    debug_log(f"[{session_id}] Audio added to video successfully", session_id)
                except Exception as audio_error:
                    debug_log(f"[{session_id}] Could not add audio: {audio_error}", session_id)
            elif audio is None:
                debug_log(f"[{session_id}] No audio in original video", session_id)
            
            # Check if we have valid frames
            if not processed_frames and processed_video is None:
                debug_log(f"[{session_id}] ERROR: No processed frames available!", session_id)
                raise ValueError("No processed frames to create video")
                
        except Exception as video_error:
            debug_log(f"[{session_id}] Error creating video: {video_error}", session_id)
            import traceback
            traceback.print_exc()
            
            # Check if we already successfully created a transparent video with direct FFmpeg
            debug_log(f"[{session_id}] 🔍 CHECKING FOR EXISTING TRANSPARENT VIDEO:", session_id)
            debug_log(f"[{session_id}]   - bg_type = '{bg_type}'", session_id)
            debug_log(f"[{session_id}]   - Is Transparent? {bg_type == 'Transparent'}", session_id)
            output_path = f"temp_output_{session_id}_transparent.mov" if bg_type == "Transparent" else f"temp_output_{session_id}.webm"
            abs_output_path = os.path.join(os.getcwd(), output_path)
            debug_log(f"[{session_id}]   - Looking for: {abs_output_path}", session_id)
            debug_log(f"[{session_id}]   - File exists? {os.path.exists(abs_output_path)}", session_id)
            if bg_type == "Transparent" and os.path.exists(abs_output_path):
                debug_log(f"[{session_id}]  Direct FFmpeg transparent video already exists - skipping fallback", session_id)
                processed_video = None  # Keep it None to skip MoviePy export
            # Try without audio as fallback
            elif processed_frames:
                debug_log(f"[{session_id}] Attempting fallback: creating video without audio", session_id)
                # IMPORTANT: Preserve transparency in fallback too!
                if bg_type == "Transparent" and len(processed_frames[0].shape) == 3 and processed_frames[0].shape[2] == 4:
                    debug_log(f"[{session_id}] Using transparency-preserving fallback for RGBA frames", session_id)
                    # Use direct RGBA frames in fallback too
                    processed_video = ImageSequenceClip(processed_frames, fps=fps, with_mask=True)
                    debug_log(f"[{session_id}] Fallback video created with RGBA transparency", session_id)
                else:
                    processed_video = ImageSequenceClip(processed_frames, fps=fps)
                    debug_log(f"[{session_id}] Fallback video created successfully (no transparency)", session_id)
            else:
                raise
        
        # Skip MoviePy export if we already used direct FFmpeg for transparency
        debug_log(f"[{session_id}] 🚨 MOVIEPY EXPORT PATH DECISION:", session_id)
        debug_log(f"[{session_id}]   - bg_type = '{bg_type}'", session_id)
        debug_log(f"[{session_id}]   - Is Transparent? {bg_type == 'Transparent'}", session_id)
        debug_log(f"[{session_id}]   - output_format = '{output_format}'", session_id)
        debug_log(f"[{session_id}]   - output_format.lower() = '{output_format.lower()}'", session_id)
        if bg_type == "Transparent":
            debug_log(f"[{session_id}]   ➡️ TRANSPARENT PATH CHOSEN", session_id)
            # For transparent video, use appropriate format
            if output_format.lower() == "webm":
                output_path = f"temp_output_{session_id}.webm"  # Try WebM with transparency
                debug_log(f"[{session_id}]     - Selected: WebM for transparency", session_id)
            else:
                output_path = f"temp_output_{session_id}_transparent.mov"  # MOV for other formats
                debug_log(f"[{session_id}]     - Selected: MOV for transparency", session_id)
        else:
            output_path = f"temp_output_{session_id}.{output_format.lower()}"
            debug_log(f"[{session_id}]   ➡️ NON-TRANSPARENT PATH: {output_path}", session_id)
        abs_output_path = os.path.join(os.getcwd(), output_path)
        debug_log(f"[{session_id}] 🔍 CHECKING IF SHOULD SKIP MOVIEPY EXPORT:", session_id)
        debug_log(f"[{session_id}]   - abs_output_path = '{abs_output_path}'", session_id)
        debug_log(f"[{session_id}]   - File exists? {os.path.exists(abs_output_path)}", session_id)
        debug_log(f"[{session_id}]   - processed_video is None? {processed_video is None}", session_id)
        debug_log(f"[{session_id}]   - bg_type == 'Transparent'? {bg_type == 'Transparent'}", session_id)
        skip_moviepy = processed_video is None and bg_type == "Transparent" and os.path.exists(abs_output_path)
        debug_log(f"[{session_id}]   - WILL SKIP MOVIEPY? {skip_moviepy}", session_id)
        if skip_moviepy:
            debug_log(f"[{session_id}]  SKIPPING MoviePy export - Video already created with direct FFmpeg", session_id)
        else:
            debug_log(f"[{session_id}] Warning: ENTERING MOVIEPY EXPORT (THIS MAY OVERWRITE TRANSPARENCY!)", session_id)
            # Determine output format and codec with error handling
            output_ext = output_format.lower()
            debug_log(f"[{session_id}] STARTING VIDEO EXPORT - Format: {output_ext}, Background type: {bg_type}", session_id)
            
            try:
                debug_log(f"[{session_id}]  FINAL EXPORT LOGIC:", session_id)
                debug_log(f"[{session_id}]   - output_ext = '{output_ext}'", session_id)
                debug_log(f"[{session_id}]   - bg_type = '{bg_type}'", session_id)
                if output_ext == 'webm':
                    debug_log(f"[{session_id}]   🔍 WebM export branch", session_id)
                    if bg_type == "Transparent":
                        debug_log(f"[{session_id}]   Warning: TRANSPARENT WEBM - SHOULD BE ALREADY CREATED!", session_id)
                        # WebM transparency is broken, already handled above with MOV
                        debug_log(f"[{session_id}] Transparent WebM already created", session_id)
                        # Skip writing another file, we already have the MOV
                    else:
                        debug_log(f"[{session_id}]   ❌ NON-TRANSPARENT WEBM PATH (THIS IS THE PROBLEM!)", session_id)
                        # Non-transparent WebM export
                        output_path = f"temp_output_{session_id}.webm"
                        debug_log(f"[{session_id}] Configuring WebM export (non-transparent)...", session_id)
                        
                        codec = "libvpx-vp9"
                        codec_params = []
                        debug_log(f"[{session_id}] 🔴 WebM standard - codec: {codec}", session_id)
                        debug_log(f"[{session_id}] 🔴 NO PIXEL FORMAT SPECIFIED - WILL DEFAULT TO yuv420p!", session_id)
                        
                        # High quality settings
                        codec_params += ["-b:v", "0", "-crf", "10", "-deadline", "best", "-cpu-used", "0"]
                        debug_log(f"[{session_id}] 🔴 WebM codec params: {codec_params} (NO -pix_fmt!)", session_id)
                        
                        # Ensure absolute path
                        abs_output_path = os.path.join(os.getcwd(), output_path)
                        debug_log(f"[{session_id}] 🔴 WRITING NON-TRANSPARENT WEBM TO: {abs_output_path}", session_id)
                        debug_log(f"[{session_id}] 🔴 THIS WILL OVERWRITE ANY EXISTING TRANSPARENT VIDEO!", session_id)
                        debug_log(f"[{session_id}] 🔴 Calling MoviePy's write_videofile (uses FFmpeg internally)...", session_id)
                        debug_log(f"[{session_id}] 🔴 Codec params: {codec_params} (NO PIX_FMT - WILL DEFAULT TO yuv420p!)", session_id)
                        processed_video.write_videofile(abs_output_path, codec=codec, threads=2, ffmpeg_params=codec_params, logger=None)
                        debug_log(f"[{session_id}] 🔴 NON-TRANSPARENT WebM written - transparency LOST if it existed!", session_id)
                elif output_ext == 'mov':
                    if bg_type == "Transparent":
                        # Transparent MOV already created with PNG codec
                        debug_log(f"[{session_id}] Transparent MOV already created with PNG codec", session_id)
                        # Create WebM preview for transparent MOV
                        webm_preview_path = f"temp_output_{session_id}_preview.webm"
                        abs_webm_preview_path = os.path.join(os.getcwd(), webm_preview_path)
                        debug_log(f"[{session_id}] Creating WebM preview for MOV transparency...", session_id)
                        debug_log(f"[{session_id}] WebM preview path: {abs_webm_preview_path}", session_id)
                        try:
                            processed_video.write_videofile(abs_webm_preview_path, codec="libvpx-vp9", threads=2, 
                                ffmpeg_params=["-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "10", "-row-mt", "1"], logger=None)
                            debug_log(f"[{session_id}]  WebM preview created successfully for transparent MOV", session_id)
                            debug_log(f"[{session_id}]  WebM preview file exists: {os.path.exists(abs_webm_preview_path)}", session_id)
                            debug_log(f"[{session_id}]  WebM preview file size: {os.path.getsize(abs_webm_preview_path) if os.path.exists(abs_webm_preview_path) else 0} bytes", session_id)
                        except Exception as e:
                            debug_log(f"[{session_id}] ❌ Warning: Failed to create WebM preview: {str(e)}", session_id)
                    else:
                        # Create both MOV (for download) and WebM (for preview)
                        output_path = f"temp_output_{session_id}.mov"
                        webm_preview_path = f"temp_output_{session_id}_preview.webm"
                        debug_log(f"[{session_id}] Configuring MOV export (non-transparent) + WebM preview...", session_id)
                        
                        # Create MOV file for download
                        codec = "libx264"
                        codec_params = ["-crf", "10"]
                        debug_log(f"[{session_id}] MOV standard - codec: libx264", session_id)
                        # Ensure absolute path
                        abs_output_path = os.path.join(os.getcwd(), output_path)
                        debug_log(f"[{session_id}] Writing MOV to: {abs_output_path}", session_id)
                        debug_log(f"[{session_id}] Calling FFmpeg to encode MOV video...", session_id)
                        processed_video.write_videofile(abs_output_path, codec=codec, threads=2, ffmpeg_params=codec_params, logger=None)
                        debug_log(f"[{session_id}] MOV video written successfully", session_id)
                        
                        # Create WebM preview file
                        abs_webm_preview_path = os.path.join(os.getcwd(), webm_preview_path)
                        debug_log(f"[{session_id}] Creating WebM preview for MOV...", session_id)
                        debug_log(f"[{session_id}] WebM preview path: {abs_webm_preview_path}", session_id)
                        try:
                            processed_video.write_videofile(abs_webm_preview_path, codec="libvpx-vp9", threads=2, 
                                ffmpeg_params=["-b:v", "0", "-crf", "10", "-row-mt", "1"], logger=None)
                            debug_log(f"[{session_id}]  WebM preview created successfully for MOV", session_id)
                            debug_log(f"[{session_id}]  WebM preview file exists: {os.path.exists(abs_webm_preview_path)}", session_id)
                            debug_log(f"[{session_id}]  WebM preview file size: {os.path.getsize(abs_webm_preview_path) if os.path.exists(abs_webm_preview_path) else 0} bytes", session_id)
                        except Exception as e:
                            debug_log(f"[{session_id}] ❌ Warning: Failed to create WebM preview: {str(e)}", session_id)
                else:  # Default to MP4
                    if bg_type == "Transparent":
                        # MP4 doesn't support transparency, already created MOV
                        debug_log(f"[{session_id}] MP4 doesn't support transparency - MOV already created with PNG codec", session_id)
                        # Skip writing another file
                    else:
                        output_path = f"temp_output_{session_id}.mp4"
                        debug_log(f"[{session_id}] Configuring MP4 export (standard format)...", session_id)
                        # Ensure absolute path
                        abs_output_path = os.path.join(os.getcwd(), output_path)
                        debug_log(f"[{session_id}] Writing MP4 to: {abs_output_path}", session_id)
                        debug_log(f"[{session_id}] MP4 codec: libx264, preset: veryslow, crf: 10 (high quality)", session_id)
                        debug_log(f"[{session_id}] Calling FFmpeg to encode video...", session_id)
                        processed_video.write_videofile(abs_output_path, codec="libx264", threads=2, ffmpeg_params=["-preset", "veryslow", "-crf", "10"], logger=None)
                        debug_log(f"[{session_id}] MP4 video written successfully", session_id)
            
                # Check if file was created
                debug_log(f"[{session_id}] Verifying output file...", session_id)
                if os.path.exists(abs_output_path):
                    file_size = os.path.getsize(abs_output_path)
                    file_size_mb = file_size / (1024 * 1024)
                    debug_log(f"[{session_id}] VIDEO SAVED SUCCESSFULLY!", session_id)
                    debug_log(f"[{session_id}] Format: {output_ext.upper()}", session_id)
                    debug_log(f"[{session_id}] File size: {file_size} bytes ({file_size_mb:.2f} MB)", session_id)
                    debug_log(f"[{session_id}] Location: {abs_output_path}", session_id)
                    
                    # Verify pixel format (alpha presence) using ffprobe if available
                    try:
                        import subprocess, json, shutil
                        # Prefer bundled ffprobe in bin
                        candidates = [
                            os.path.join(os.getcwd(), 'bin', 'ffprobe.exe'),
                            os.path.join(os.getcwd(), 'bin', 'ffprobe'),
                            shutil.which('ffprobe')
                        ]
                        ffprobe_path = next((p for p in candidates if p and os.path.exists(p)), None)
                        if ffprobe_path:
                            cmd = [
                                ffprobe_path,
                                '-v','error',
                                '-select_streams','v:0',
                                '-show_entries','stream=pix_fmt,codec_name,width,height,nb_frames',
                                '-of','json',
                                abs_output_path
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0 and result.stdout:
                                info = json.loads(result.stdout)
                                stream = (info.get('streams') or [{}])[0]
                                pix_fmt = stream.get('pix_fmt')
                                codec = stream.get('codec_name')
                                debug_log(f"[{session_id}] FFPROBE: codec={codec}, pix_fmt={pix_fmt}", session_id)
                                if pix_fmt and ('a' in pix_fmt or 'yuva' in pix_fmt or pix_fmt.endswith('a')):
                                    debug_log(f"[{session_id}] FFPROBE: Alpha channel PRESENT (pix_fmt {pix_fmt})", session_id)
                                else:
                                    debug_log(f"[{session_id}] FFPROBE: Alpha channel NOT detected (pix_fmt {pix_fmt})", session_id)
                            else:
                                debug_log(f"[{session_id}] FFPROBE warning: {result.stderr.strip()}", session_id)
                        else:
                            debug_log(f"[{session_id}] FFPROBE not found; skipping pixel format verification", session_id)
                    except Exception as probe_err:
                        debug_log(f"[{session_id}] FFPROBE error: {probe_err}", session_id)
                    # DO NOT track output file for cleanup - keep for download
                    # temp_files.add(abs_output_path)  # Commented out to prevent premature deletion
                    debug_log(f"[{session_id}] Output file preserved for download (not tracked for cleanup)", session_id)
                else:
                    debug_log(f"[{session_id}] ERROR: Output file was not created!", session_id)
                    debug_log(f"[{session_id}] Expected location: {abs_output_path}", session_id)
                    raise FileNotFoundError(f"Output file {abs_output_path} was not created")
                    
            except Exception as write_error:
                debug_log(f"[{session_id}] Error writing video: {write_error}", session_id)
                import traceback
                error_trace = traceback.format_exc()
                debug_log(f"[{session_id}] Stack trace:\n{error_trace}", session_id)
                # Fallback to basic MP4
                output_path = f"temp_output_{session_id}.mp4"
                debug_log(f"[{session_id}] Attempting fallback to basic MP4...", session_id)
                try:
                    # Ensure absolute path for fallback
                    abs_output_path = os.path.join(os.getcwd(), output_path)
                    debug_log(f"[{session_id}] Fallback path: {abs_output_path}", session_id)
                    processed_video.write_videofile(abs_output_path, codec="libx264", threads=1, ffmpeg_params=["-preset", "veryslow", "-crf", "10"], logger=None)
                    debug_log(f"[{session_id}] Fallback video saved successfully!", session_id)
                    # Track output file for cleanup
                    if os.path.exists(abs_output_path):
                        # DO NOT track output file for cleanup - keep for download
                        # temp_files.add(abs_output_path)  # Commented out to prevent premature deletion
                        debug_log(f"[{session_id}] Fallback output file preserved for download", session_id)
                except Exception as fallback_error:
                    debug_log(f"[{session_id}] CRITICAL: Fallback also failed: {fallback_error}", session_id)
                    import traceback
                    fallback_trace = traceback.format_exc()
                    debug_log(f"[{session_id}] Fallback error trace:\n{fallback_trace}", session_id)
                    raise
        
        elapsed_time = time.time() - start_time
        debug_log(f"[{session_id}] PROCESSING COMPLETED in {elapsed_time:.2f} seconds", session_id)
        
        # Final file verification and info
        debug_log(f"[{session_id}] ============================", session_id)
        debug_log(f"[{session_id}] FINAL OUTPUT FILE INFO:", session_id)
        debug_log(f"[{session_id}] Filename: {output_path}", session_id)
        debug_log(f"[{session_id}] Full path: {abs_output_path}", session_id)
        debug_log(f"[{session_id}] File exists: {os.path.exists(abs_output_path)}", session_id)
        
        if os.path.exists(abs_output_path):
            file_size = os.path.getsize(abs_output_path)
            file_size_mb = file_size / (1024 * 1024)
            debug_log(f"[{session_id}] File size: {file_size} bytes ({file_size_mb:.2f} MB)", session_id)
        else:
            debug_log(f"[{session_id}] WARNING: File does not exist!", session_id)
        
        debug_log(f"[{session_id}] Download URL will be: /api/download/{os.path.basename(abs_output_path)}", session_id)
        debug_log(f"[{session_id}] ============================", session_id)
        
        completion_data = {
            'session_id': session_id,
            'status': 'completed',
            'message': 'Processing complete!',
            'elapsed_time': elapsed_time,
            'output_file': os.path.basename(abs_output_path),
            'debug_info': {
                'full_path': abs_output_path,
                'exists': os.path.exists(abs_output_path),
                'size': os.path.getsize(abs_output_path) if os.path.exists(abs_output_path) else 0
            }
        }
        debug_log(f"[{session_id}] Sending completion event to frontend...", session_id)
        socketio.emit('processing_complete', completion_data)
        debug_log(f"[{session_id}] Completion event sent!", session_id)
        
    except Exception as e:
        debug_log(f"[{session_id}] ERROR processing video: {e}", session_id)
        import traceback
        error_trace = traceback.format_exc()
        debug_log(f"[{session_id}] Stack trace:\n{error_trace}", session_id)
        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0
        socketio.emit('processing_error', {
            'session_id': session_id,
            'status': 'error',
            'message': f'Error processing video: {str(e)}',
            'elapsed_time': elapsed_time
        })
    finally:
        # Clean up video objects to prevent memory leaks and access violations
        debug_log(f"[{session_id}] Starting cleanup...", session_id)
        try:
            if processed_video is not None:
                processed_video.close()
                debug_log(f"[{session_id}] Closed processed video", session_id)
        except Exception as e:
            debug_log(f"[{session_id}] Error closing processed video: {e}", session_id)
        
        try:
            if background_video is not None:
                background_video.close()
                debug_log(f"[{session_id}] Closed background video", session_id)
        except Exception as e:
            debug_log(f"[{session_id}] Error closing background video: {e}", session_id)
        
        try:
            if video is not None:
                if hasattr(video, 'audio') and video.audio is not None:
                    video.audio.close()
                video.close()
                debug_log(f"[{session_id}] Closed input video", session_id)
        except Exception as e:
            debug_log(f"[{session_id}] Error closing input video: {e}", session_id)
        
        if session_id in active_sessions:
            del active_sessions[session_id]
            debug_log(f"[{session_id}] Session removed from active sessions", session_id)
        
        # Only cleanup INPUT files (not output files)
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                debug_log(f"[{session_id}] Removed input video file", session_id)
            if bg_path and os.path.exists(bg_path):
                os.remove(bg_path)
                debug_log(f"[{session_id}] Removed background file", session_id)
            debug_log(f"[{session_id}] NOTE: Output file preserved for download - will cleanup on app exit or manual removal", session_id)
        except Exception as cleanup_err:
            debug_log(f"[{session_id}] Cleanup warning: {cleanup_err}", session_id)

# REMOVED DUPLICATE FUNCTION - Using the first one above

@app.route('/api/get_video_data/<filename>', methods=['GET'])
def get_video_data(filename):
    """Get raw video data for direct browser processing"""
    print(f"\n[DIRECT_DOWNLOAD] ==================== DIRECT VIDEO DATA REQUEST ====================")
    print(f"[DIRECT_DOWNLOAD] Request for file data: {filename}")
    
    try:
        # Prevent path traversal by forcing a basename and whitelisting pattern
        safe_name = os.path.basename(filename)
        print(f"[DIRECT_DOWNLOAD] Sanitized filename: {safe_name}")
        
        # Accept mp4, webm, or mov files (including transparent versions)
        if not safe_name.startswith('temp_output_') or not (safe_name.endswith('.mp4') or safe_name.endswith('.webm') or safe_name.endswith('.mov')):
            print(f"[DIRECT_DOWNLOAD] Invalid filename pattern: {safe_name}")
            return jsonify({'error': 'Invalid filename'}), 400
        
        # Try multiple locations
        potential_paths = [
            os.path.join(os.getcwd(), safe_name),  # Current working directory
            safe_name,  # Relative path
            os.path.abspath(safe_name),  # Absolute path
            os.path.join(tempfile.gettempdir(), safe_name)  # Temp directory
        ]
        
        # Find first existing file
        file_path = None
        for path in potential_paths:
            print(f"[DIRECT_DOWNLOAD] Checking path: {path}")
            if os.path.exists(path):
                file_path = path
                print(f"[DIRECT_DOWNLOAD] Found file at: {file_path}")
                break
        
        if not file_path:
            print(f"[DIRECT_DOWNLOAD] File not found in any location")
            return jsonify({'error': 'File not found'}), 404
        
        # Read file data
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
                print(f"[DIRECT_DOWNLOAD] Read {len(file_data)} bytes from file")
                
                # Convert to base64
                import base64
                encoded_data = base64.b64encode(file_data).decode('utf-8')
                print(f"[DIRECT_DOWNLOAD] Encoded to base64 string of length {len(encoded_data)}")
                
                # Determine MIME type
                if safe_name.endswith('.webm'):
                    mimetype = 'video/webm'
                elif safe_name.endswith('.mov'):
                    mimetype = 'video/quicktime'
                else:
                    mimetype = 'video/mp4'
                
                # Return data URI
                return jsonify({
                    'success': True,
                    'data': f"data:{mimetype};base64,{encoded_data}",
                    'filename': f"processed_{safe_name}",
                    'mimetype': mimetype,
                    'size': len(file_data)
                })
        except Exception as read_err:
            print(f"[DIRECT_DOWNLOAD] Error reading file: {read_err}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Error reading file', 'details': str(read_err)}), 500
    except Exception as e:
        print(f"[DIRECT_DOWNLOAD] Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Unhandled error', 'details': str(e)}), 500

@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        debug_log("\n=== VIDEO PROCESSING REQUEST ===", "system")
        debug_log("Processing video request received...", "system")
        debug_log(f"Request method: {request.method}", "system")
        debug_log(f"Request files: {request.files.keys()}", "system")
        debug_log(f"Request form: {request.form.to_dict()}", "system")
        
        # Check if models are available
        if not models_available:
            debug_log("ERROR: Models not available", "system")
            return jsonify({'error': 'AI models are not available due to system constraints. Please restart the application or increase virtual memory.'}), 503
        
        debug_log("Models are available, attempting to load...", "system")
        
        # Load models if needed
        if not load_models_if_needed():
            debug_log("ERROR: Failed to load models", "system")
            return jsonify({'error': 'Failed to load AI models. Please restart the application or increase virtual memory.'}), 503
            
        debug_log("Models loaded successfully, proceeding with video processing...", "system")
            
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            'client_id': request.remote_addr or 'unknown',
            'status': 'processing'
        }
        debug_log(f"Created session: {session_id} for client: {request.remote_addr}", session_id)
        
        # Get form data
        video_file = request.files.get('video')
        bg_type = request.form.get('bg_type', 'Color')
        debug_log(f"\n BACKGROUND TYPE FROM FORM:", session_id)
        debug_log(f"  - Raw value: '{bg_type}'", session_id)
        debug_log(f"  - Type: {type(bg_type).__name__}", session_id)
        debug_log(f"  - Is 'Transparent'? {bg_type == 'Transparent'}", session_id)
        debug_log(f"  - Default if missing: 'Color'", session_id)
        
        color = request.form.get('color', '#00FF00')
        fps = int(request.form.get('fps', 0))
        video_handling = request.form.get('video_handling', 'slow_down')
        fast_mode = request.form.get('fast_mode', 'true').lower() == 'true'
        output_format = request.form.get('output_format', 'mp4')
        debug_log(f"\n OUTPUT FORMAT FROM FORM:", session_id)
        debug_log(f"  - Raw value: '{output_format}'", session_id)
        debug_log(f"  - Lowercase: '{output_format.lower()}'", session_id)
        debug_log(f"  - Is 'webm'? {output_format.lower() == 'webm'}", session_id)
        # Get max_workers from hardware optimizer
        try:
            # Use hardware-optimized worker count
            max_workers_requested = int(request.form.get('max_workers', 0))
            if max_workers_requested > 0:
                # User specified a value, but cap it based on hardware
                max_workers = min(max_workers_requested, hw_settings.get('max_workers', 4))
            else:
                # Use hardware-optimized default
                max_workers = hw_settings.get('max_workers', 4)
            debug_log(f"Using max_workers={max_workers} (optimized for your hardware)", session_id)
        except Exception as e:
            debug_log(f"Error setting max_workers: {e}, using hardware default", session_id)
            max_workers = hw_settings.get('max_workers', 2)
        
        if not video_file:
            return jsonify({'error': 'No video file provided'}), 400
        
        # Save uploaded video
        video_filename = secure_filename(f"input_{session_id}_{video_file.filename}")
        video_path = os.path.join(tempfile.gettempdir(), video_filename)
        video_file.save(video_path)
        
        # Track temp file for cleanup
        temp_files.add(video_path)
        
        # Video resolution optimization can be added here if needed in future
        
        # Handle background file if provided
        bg_path = None
        if bg_type in ['Image', 'Video']:
            bg_file = request.files.get('background')
            if bg_file:
                bg_filename = secure_filename(f"bg_{session_id}_{bg_file.filename}")
                bg_path = os.path.join(tempfile.gettempdir(), bg_filename)
                bg_file.save(bg_path)
                # Track temp file for cleanup
                temp_files.add(bg_path)
        
        # Start processing in background
        debug_log(f"Starting background thread for session {session_id}", session_id)
        debug_log(f"Thread args: video_path={video_path}, bg_type={bg_type}, bg_path={bg_path}, color={color}, fps={fps}, video_handling={video_handling}, fast_mode={fast_mode}, max_workers={max_workers}, output_format={output_format}", session_id)
        
        thread = threading.Thread(
            target=process_video_async,
            args=(session_id, video_path, bg_type, bg_path, color, fps, video_handling, fast_mode, max_workers, output_format),
            name=f"processing-{session_id}"  # Name the thread for easier tracking
        )
        thread.start()
        
        print(f"Thread started for session {session_id}")
        print("=== END VIDEO PROCESSING REQUEST ===\n")
        
        return jsonify({
            'session_id': session_id,
            'message': 'Processing started'
        })
        
    except Exception as e:
        print(f"ERROR in process_video endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/cancel_processing', methods=['POST'])
def cancel_processing():
    session_id = request.json.get('session_id')
    print(f"[CANCEL] Received cancellation request for session: {session_id}")
    
    if session_id in active_sessions:
        # Mark session as cancelled
        active_sessions[session_id] = False
        del active_sessions[session_id]
        
        # Force aggressive memory cleanup
        print(f"[CANCEL] Forcing memory cleanup for session: {session_id}")
        force_memory_cleanup()
        
        # Kill any lingering threads
        import threading
        for thread in threading.enumerate():
            if thread.name and session_id in thread.name:
                print(f"[CANCEL] Found thread for session {session_id}, attempting to stop")
                # Note: We can't forcefully kill threads in Python, but marking session as cancelled
                # will cause the processing loop to exit
        
        # Additional cleanup
        gc.collect()
        
        # Send cancellation event to client
        socketio.emit('processing_cancelled', {
            'session_id': session_id,
            'message': 'Processing cancelled and resources cleaned'
        })
        
        print(f"[CANCEL] Successfully cancelled session: {session_id}")
        return jsonify({'message': 'Processing cancelled and resources cleaned'})
    
    print(f"[CANCEL] Session not found: {session_id}")
    return jsonify({'error': 'Session not found'}), 404

# Send debug info to frontend via socket
def send_debug_to_frontend(session_id, message):
    try:
        print(f"[{session_id}] BROWSER_DEBUG: {message}")
        socketio.emit('debug_log', {
            'session_id': session_id,
            'message': message,
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"Error sending debug to frontend: {e}")

# Global debug function that sends all messages to frontend
def debug_log(message, session_id=None):
    """Send debug message to both console and frontend"""
    # Skip verbose debug messages
    if session_id in ['frame_process', 'frame_serve', 'download']:
        return
    
    # Skip most debug messages if not in debug mode (except critical ones)
    if not DEBUG_MODE and session_id != 'system':
        return
    
    if session_id is None:
        session_id = 'system'
    
    # Print to console with proper error handling for Windows
    try:
        print(message)
    except (OSError, UnicodeEncodeError) as e:
        # Handle Windows console encoding issues
        try:
            # Try to encode as UTF-8 and ignore problematic characters
            safe_message = str(message).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
            print(safe_message)
        except:
            # If all else fails, just print a simple message
            print(f"[{session_id}] Message could not be displayed (encoding error)")
    
    # Send to frontend - broadcast to all clients
    try:
        # Ensure message is a string and handle encoding
        if not isinstance(message, str):
            message = str(message)
        
        # Clean the message to remove problematic characters
        # Remove null bytes and other control characters that can cause issues
        message = message.replace('\x00', '').encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        # Truncate very long messages
        if len(message) > 5000:
            message = message[:4997] + "..."
        
        # Ensure session_id is also clean
        session_id_str = str(session_id).replace('\x00', '').encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        # Try to emit with error handling
        socketio.emit('debug_log', {
            'session_id': session_id_str,
            'message': message,
            'timestamp': time.time()
        })
    except (OSError, Exception) as e:
        # Handle OSError [Errno 22] and other exceptions
        # Don't try to print the full error as it might cause another encoding issue
        try:
            # Try to at least log that there was an issue
            print(f"[{session_id}] Debug message could not be sent to frontend")
        except:
            # Complete failure - silently continue
            pass

@app.route('/frames/<session_id>/<frame_number>')
def get_frame(session_id, frame_number):
    """Serve individual frame images for the frame slider"""
    try:
        # Sanitize inputs
        safe_session_id = os.path.basename(session_id)
        safe_frame_number = os.path.basename(frame_number)
        
        # Construct path to frame
        frame_path = f"frames_{safe_session_id}/frame_{safe_frame_number}.png"
        
        debug_log(f"[FRAME] Request for frame: session={safe_session_id}, frame={safe_frame_number}", "frame_serve")
        debug_log(f"[FRAME] Looking for file: {frame_path}", "frame_serve")
        debug_log(f"[FRAME] File exists: {os.path.exists(frame_path)}", "frame_serve")
        
        if not os.path.exists(frame_path):
            debug_log(f"[FRAME] Frame not found: {frame_path}", "frame_serve")
            # List available frames in directory for debugging
            frame_dir = f"frames_{safe_session_id}"
            if os.path.exists(frame_dir):
                available_frames = os.listdir(frame_dir)
                debug_log(f"[FRAME] Available frames in {frame_dir}: {available_frames[:5]}...", "frame_serve")
            return jsonify({'error': 'Frame not found', 'path': frame_path}), 404
        
        # Return the frame image
        debug_log(f"[FRAME] Serving frame: {frame_path}", "frame_serve")
        return send_file(frame_path, mimetype='image/png')
    except Exception as e:
        debug_log(f"[FRAME] Error serving frame: {str(e)}", "frame_serve")
        return jsonify({'error': 'Failed to load frame', 'details': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    debug_log(f"\n[DOWNLOAD] ==================== DOWNLOAD REQUEST ====================", "download")
    debug_log(f"[DOWNLOAD] Client requested file: {filename}", "download")
    debug_log(f"[DOWNLOAD] Request method: {request.method}", "download")
    debug_log(f"[DOWNLOAD] Request headers: {dict(request.headers)}", "download")
    
    try:
        # Prevent path traversal by forcing a basename and whitelisting pattern
        safe_name = os.path.basename(filename)
        debug_log(f"[DOWNLOAD] Sanitized filename: {safe_name}", "download")
        
        # Accept mp4, webm, or mov files (including transparent versions)
        if not safe_name.startswith('temp_output_') or not (safe_name.endswith('.mp4') or safe_name.endswith('.webm') or safe_name.endswith('.mov')):
            debug_log(f"[DOWNLOAD] Invalid filename pattern: {safe_name}", "download")
            debug_log(f"[DOWNLOAD] Expected pattern: temp_output_*.mp4/webm/mov", "download")
            return jsonify({
                'error': 'Invalid filename',
                'details': {
                    'filename': safe_name,
                    'valid_pattern': False,
                    'starts_with_temp_output': safe_name.startswith('temp_output_'),
                    'valid_extension': safe_name.endswith('.mp4') or safe_name.endswith('.webm') or safe_name.endswith('.mov')
                }
            }), 400
        
        # Try multiple locations
        potential_paths = [
            os.path.join(os.getcwd(), safe_name),  # Current working directory
            safe_name,  # Relative path
            os.path.abspath(safe_name),  # Absolute path
            os.path.join(tempfile.gettempdir(), safe_name)  # Temp directory
        ]
        
        debug_log(f"[DOWNLOAD] Searching for file in {len(potential_paths)} locations...", "download")
        
        # Find first existing file
        file_path = None
        for idx, path in enumerate(potential_paths, 1):
            debug_log(f"[DOWNLOAD] Location {idx}: {path}", "download")
            if os.path.exists(path):
                file_path = path
                debug_log(f"[DOWNLOAD] FOUND file at location {idx}: {file_path}", "download")
                break
            else:
                debug_log(f"[DOWNLOAD] Not found at location {idx}", "download")
        
        if not file_path:
            print(f"[DOWNLOAD] File not found in any location")
            # List files in directory to help debug
            try:
                print(f"[DOWNLOAD] Current working directory: {os.getcwd()}")
                dir_contents = os.listdir(os.getcwd())
                temp_files = [f for f in dir_contents if f.startswith('temp_output_')]
                print(f"[DOWNLOAD] Available temp files in CWD: {temp_files}")
                
                # Also check temp directory
                temp_dir = tempfile.gettempdir()
                print(f"[DOWNLOAD] Temp directory: {temp_dir}")
                temp_dir_contents = os.listdir(temp_dir)
                temp_dir_files = [f for f in temp_dir_contents if f.startswith('temp_output_')]
                print(f"[DOWNLOAD] Available temp files in temp dir: {temp_dir_files}")
                
                return jsonify({
                    'error': 'File not found',
                    'details': {
                        'requested_file': safe_name,
                        'checked_paths': potential_paths,
                        'cwd': os.getcwd(),
                        'cwd_temp_files': temp_files,
                        'temp_dir': temp_dir,
                        'temp_dir_files': temp_dir_files
                    }
                }), 404
            except Exception as dir_err:
                print(f"[DOWNLOAD] Error listing directory: {dir_err}")
                return jsonify({'error': 'File not found', 'directory_error': str(dir_err)}), 404
        
        file_size = os.path.getsize(file_path)
        print(f"[DOWNLOAD] File exists, size: {file_size} bytes")
        
        # Set appropriate mime type
        if safe_name.endswith('.webm'):
            mimetype = 'video/webm'
        elif safe_name.endswith('.mov'):
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'
        
        print(f"[DOWNLOAD] Serving file with mimetype: {mimetype}")
        
        # Create a direct file response
        try:
            # Use direct file streaming for better performance
            response = send_file(
                file_path, 
                as_attachment=True,  # Force download
                download_name=f"processed_{safe_name}", 
                mimetype=mimetype,
                conditional=True  # Enable range requests
            )
            
            # Set headers for better download experience
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition, Content-Length'
            response.headers['Content-Length'] = str(file_size)
            
            print(f"[DOWNLOAD] Successfully created file response")
            print(f"[DOWNLOAD] Response headers: {dict(response.headers)}")
            print(f"[DOWNLOAD] ==================== END DOWNLOAD REQUEST ====================\n")
            return response
        except Exception as send_err:
            print(f"[DOWNLOAD] Error creating send_file response: {send_err}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'Error serving file', 'send_file_error': str(send_err)}), 500
    except Exception as e:
        print(f"[DOWNLOAD] Unhandled error in download_file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Unhandled error',
            'message': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/')
def root():
    return jsonify({
        'status': 'ok',
        'message': 'Video Background Removal API is running'
    })

@app.route('/health')
@app.route('/api/health')
def health_check():
    # Check system resources
    import psutil
    
    try:
        # Get memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return jsonify({
            'status': 'healthy',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'models_available': models_available,
            'models_loaded': birefnet is not None and birefnet_lite is not None,
            'system_info': {
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent': memory.percent,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_free_gb': round(swap.free / (1024**3), 2),
                'swap_percent': swap.percent,
                'cpu_percent': psutil.cpu_percent(interval=0.1)
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'models_available': models_available,
            'models_loaded': birefnet is not None and birefnet_lite is not None,
            'error': str(e)
        })

@app.route('/api/hw_status', methods=['GET'])
def hw_status():
    """Alternative hardware status endpoint"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        # Check actual CUDA availability in real-time
        actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return jsonify({
            'status': 'ok',
            'hardware': {
                'processing_device': actual_device.upper(),
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not Available',
                'memory_gb': round(memory.total / (1024**3), 1),
                'memory_available_gb': round(memory.available / (1024**3), 1),
                'memory_usage_percent': memory.percent
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify server is responding"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is responding',
        'models_available': models_available,
        'models_loaded': models_loaded
    })



@app.route('/api/hardware_status', methods=['GET'])
def hardware_status():
    """Get current hardware configuration and status"""
    try:
        # Get GPU name if available
        gpu_name = "Not Available"
        if device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        elif device == 'mps':
            gpu_name = "Apple M1/M2 GPU"
        
        # Get current memory usage
        import psutil
        memory = psutil.virtual_memory()
        
        return jsonify({
            'status': 'ok',
            'hardware': {
                'processing_device': device.upper(),
                'gpu_name': gpu_name,
                'gpu_available': torch.cuda.is_available(),
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'memory_gb': round(memory.total / (1024**3), 1),
                'memory_available_gb': round(memory.available / (1024**3), 1),
                'memory_usage_percent': memory.percent
            },
            'settings': {
                'batch_size': MAX_BATCH_SIZE,
                'max_workers': hw_settings.get('max_workers', 2),
                'mixed_precision': USE_MIXED_PRECISION,
                'model_quantization': USE_QUANTIZATION,
                'model_preference': MODEL_PREFERENCE,
                'optimization_profile': hw_settings.get('profile', 'unknown')
            },
            'models': {
                'birefnet_loaded': birefnet is not None,
                'birefnet_lite_loaded': birefnet_lite is not None,
                'models_available': models_available,
                'models_loaded': models_loaded
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'processing_device': device
        })

@app.route('/api/gpu_diagnostics', methods=['GET'])
def gpu_diagnostics():
    """Detailed GPU and PyTorch diagnostics for troubleshooting CUDA availability."""
    import platform
    info = {
        'platform': platform.platform(),
        'python_executable': sys.executable,
        'env': {
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
        },
        'torch': {
            'imported': False,
            'version': None,
            'compiled_with_cuda': None,
            'cuda_is_available': False,
            'device_count': 0,
            'device_name_0': None
        },
        'nvidia_smi': {
            'available': False,
            'raw': None
        }
    }
    try:
        import torch  # noqa: F401
        info['torch']['imported'] = True
        info['torch']['version'] = getattr(torch, '__version__', None)
        info['torch']['compiled_with_cuda'] = getattr(getattr(torch, 'version', None), 'cuda', None)
        try:
            info['torch']['cuda_is_available'] = bool(torch.cuda.is_available())
        except Exception:
            info['torch']['cuda_is_available'] = False
        try:
            info['torch']['device_count'] = int(torch.cuda.device_count())
            if info['torch']['device_count'] > 0:
                info['torch']['device_name_0'] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    except Exception as e:
        info['torch']['error'] = str(e)

    # Try nvidia-smi
    try:
        import shutil, subprocess
        nvsmi = shutil.which('nvidia-smi')
        if nvsmi:
            result = subprocess.run([nvsmi], capture_output=True, text=True, timeout=10)
            info['nvidia_smi']['available'] = result.returncode == 0
            info['nvidia_smi']['raw'] = result.stdout
    except Exception as e:
        info['nvidia_smi']['error'] = str(e)

    return jsonify(info)

@app.route('/api/reset_models', methods=['POST'])
def reset_models():
    """Reset model availability flag and try loading again"""
    global models_available, models_loaded
    
    # Reset flags
    models_available = True
    models_loaded = False
    
    # Try to load models
    success = load_models_if_needed()
    
    return jsonify({
        'status': 'success' if success else 'failed',
        'models_available': models_available,
        'models_loaded': models_loaded,
        'message': 'Models reset and load attempted'
    })

@app.route('/api/cleanup_files', methods=['POST'])
def cleanup_files():
    """Manual cleanup of processed output files"""
    try:
        cleaned_files = []
        patterns = ['temp_output_*.mp4', 'temp_output_*.webm', 'temp_output_*.mov', 'temp_output_*_transparent.mov', 'temp_output_*_preview.webm']
        
        for pattern in patterns:
            for filepath in glob.glob(pattern):
                try:
                    os.remove(filepath)
                    cleaned_files.append(filepath)
                    debug_log(f"[CLEANUP] Manually removed: {filepath}", "system")
                except Exception as e:
                    debug_log(f"[CLEANUP] Failed to remove {filepath}: {e}", "system")
        
        debug_log(f"[CLEANUP] Manual cleanup complete - removed {len(cleaned_files)} files", "system")
        return jsonify({
            'status': 'success',
            'message': f'Cleaned up {len(cleaned_files)} processed video files',
            'files_removed': cleaned_files
        })
    except Exception as e:
        debug_log(f"[CLEANUP] Error during manual cleanup: {e}", "system")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    debug_log(f'\n=== CLIENT CONNECTED: {client_id} ===', 'system')
    emit('connected', {'message': 'Connected to video processing server'})
    
    # Send initial system status to debug console
    # Check actual GPU availability in real-time
    actual_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debug_log(f'Server Status: Ready', 'system')
    debug_log(f'Python Backend: Running on port 5000', 'system')
    debug_log(f'Device: {actual_device}', 'system')
    debug_log(f'Models Available: {models_available}', 'system')
    debug_log(f'Models Loaded: {models_loaded}', 'system')
    debug_log(f'BiRefNet: {"Loaded" if birefnet is not None else "Not loaded"}', 'system')
    debug_log(f'BiRefNet Lite: {"Loaded" if birefnet_lite is not None else "Not loaded"}', 'system')
    debug_log(f'FFmpeg Path: {ffmpeg_path}', 'system')
    debug_log(f'Ready to process videos!', 'system')

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    debug_log(f'\n=== CLIENT DISCONNECTED: {client_id} ===', 'system')
    
    # Cancel any active processing for this client
    sessions_to_remove = []
    for session_id, session_info in active_sessions.items():
        if session_info.get('client_id') == client_id:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        print(f"Cancelling session {session_id} for disconnected client")
        if session_id in active_sessions:
            del active_sessions[session_id]
        # Only clean up INPUT files on disconnect, preserve OUTPUT files for download
        for pattern in [f'input_{session_id}.*', f'bg_{session_id}.*']:
            for filepath in glob.glob(pattern):
                cleanup_temp_file(filepath)
        debug_log(f"[DISCONNECT] Preserved output files for session {session_id} - use /api/cleanup_files to remove manually", "system")

import atexit
import signal

def shutdown_handler(signum=None, frame=None):
    """Handle shutdown gracefully"""
    print("\n=== SHUTTING DOWN SERVER ===")
    cleanup_all_temp_files()
    print("Server shutdown complete")
    if signum:
        sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_all_temp_files)
signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

if __name__ == '__main__':
    debug_log(f"Found ffmpeg at {ffmpeg_path}", "system")
    debug_log(f"Using device: {device}", "system")
    debug_log("Initializing API server...", "system")
    debug_log("Model cache directory ready", "system")
    
    # Clean up any leftover temp files from previous runs
    cleanup_all_temp_files()
    
    # Don't preload on startup - load on first use
    debug_log("Server ready - models will load on first use", "system")
    
    try:
        # Start the socket server
        socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        shutdown_handler()
    except Exception as e:
        print(f"Server error: {e}")
        cleanup_all_temp_files()
        raise
