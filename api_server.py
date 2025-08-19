from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import tempfile
import uuid
import time
import sys
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
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

# Optimize memory usage
torch.set_float32_matmul_precision("medium")
torch.set_num_threads(2)  # Limit CPU threads to reduce memory usage

# Force CPU to avoid GPU memory issues
device = "cpu"  # Always use CPU to avoid GPU memory allocation issues
print(f"Using device: {device}")

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

def preload_models():
    """Preload ONLY lite model for faster startup"""
    global models_loaded, birefnet, birefnet_lite
    
    if models_loaded:
        return True
    
    try:
        # Only load lite model for faster startup
        if birefnet_lite is None:
            print("Loading BiRefNet_lite model...")
            birefnet_lite = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_lite",
                trust_remote_code=True,
                cache_dir=MODEL_CACHE_DIR
            )
            birefnet_lite.to(device).eval()
            print("Model ready!")
        
        models_loaded = True
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def load_models_if_needed():
    global birefnet, birefnet_lite, models_available, models_loaded
    
    print("\n=== MODEL LOADING DEBUG ===")
    print(f"models_loaded: {models_loaded}")
    print(f"models_available: {models_available}")
    print(f"birefnet_lite is None: {birefnet_lite is None}")
    print(f"birefnet is None: {birefnet is None}")
    
    # Return immediately if models are already loaded
    if models_loaded:
        print("Models already loaded and cached")
        return True
    
    # Only load models if they haven't been loaded yet and are available
    if not models_available:
        print("Models are not available due to system constraints")
        return False
    
    # Check memory before attempting to load
    if not check_memory_available():
        print("Insufficient memory to load models. Please free up memory and try again.")
        models_available = False
        return False
    
    print("Loading BiRefNet models (this will be cached)...")
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
        
        # Optionally load full model if enough memory
        if birefnet is None and has_sufficient_memory_for_full_model():
            try:
                print("Loading BiRefNet full model (optional)...")
                gc.collect()
                birefnet = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet", 
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                )
                birefnet.to(device)
                birefnet.eval()
                gc.collect()
                print("BiRefNet full model loaded")
            except Exception as e2:
                print(f"Skipping full model due to error: {e2}")
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

def process_frame_simple(frame, bg_type, bg, fast_mode, bg_frame_index, background_frames, color):
    """Simplified frame processing that maintains consistent dimensions"""
    try:
        # Keep original frame dimensions
        original_shape = frame.shape
        pil_image = Image.fromarray(frame)
        
        # Process based on background type
        if bg_type == "Transparent":
            processed_image = process_image(pil_image, None, fast_mode, transparent=True)
            # Convert RGBA to RGB for video consistency
            if isinstance(processed_image, Image.Image) and processed_image.mode == "RGBA":
                # Create RGB version with white background
                rgb_image = Image.new("RGB", processed_image.size, (255, 255, 255))
                rgb_image.paste(processed_image, mask=processed_image.split()[3])
                processed_image = rgb_image
        else:
            # For other types, use simplified processing
            processed_image = process_image(pil_image, color if bg_type == "Color" else bg, fast_mode)
        
        # Convert to array and ensure same shape as input
        if isinstance(processed_image, Image.Image):
            result = np.array(processed_image)
        else:
            result = processed_image
            
        # Ensure dimensions match original
        if result.shape[:2] != original_shape[:2]:
            pil_result = Image.fromarray(result)
            pil_result = pil_result.resize((original_shape[1], original_shape[0]), Image.LANCZOS)
            result = np.array(pil_result)
        
        return result, bg_frame_index
    except Exception as e:
        print(f"Frame processing error: {e}")
        return frame, bg_frame_index

def process_frame(frame, bg_type, bg, fast_mode, bg_frame_index, background_frames, color):
    """
    Process a single frame, closely matching the original Gradio implementation.
    
    Args:
        frame: The video frame as numpy array
        bg_type: "Color", "Image", "Video", or "Transparent"
        bg: Background image or color
        fast_mode: Whether to use BiRefNet_lite for faster processing
        bg_frame_index: Current background frame index
        background_frames: List of background frames if using video background
        color: Background color if using color background
        
    Returns:
        Tuple of processed frame and updated bg_frame_index
    """
    try:
        print(f"DEBUG: Processing frame with bg_type={bg_type}, fast_mode={fast_mode}")
        
        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Process based on background type
        if bg_type == "Transparent":
            # Special case for transparent background
            processed_image = process_image(pil_image, None, fast_mode, transparent=True)
        elif bg_type == "Color":
            # Use color as background
            processed_image = process_image(pil_image, color, fast_mode)
        elif bg_type == "Image":
            # Use image as background
            processed_image = process_image(pil_image, bg, fast_mode)
        elif bg_type == "Video":
            # Use video frame as background
            if background_frames and len(background_frames) > 0:
                safe_index = bg_frame_index % len(background_frames)
                background_frame = background_frames[safe_index]
                bg_frame_index += 1
                background_image = Image.fromarray(background_frame)
                processed_image = process_image(pil_image, background_image, fast_mode)
            else:
                # Fallback to color if no background frames
                processed_image = process_image(pil_image, color, fast_mode)
        else:
            # Default to original image
            processed_image = pil_image
        
        # Convert back to numpy array and ensure consistent shape
        if isinstance(processed_image, Image.Image):
            result_array = np.array(processed_image)
        else:
            result_array = processed_image
        
        # Ensure consistent shape for video frames
        if len(result_array.shape) == 2:  # Grayscale
            result_array = np.stack([result_array] * 3, axis=-1)
        
        return result_array, bg_frame_index
    except Exception as e:
        print(f"ERROR processing frame: {e}")
        import traceback
        traceback.print_exc()
        # Return original frame on error
        return frame, bg_frame_index

def process_image(image, bg, fast_mode=False, transparent=False):
    """
    Process an image by removing its background and replacing with the specified background.
    CRITICAL: Returns consistent image format for video processing.
    """
    print(f"DEBUG: process_image called - fast_mode={fast_mode}, transparent={transparent}")
    
    # Store original size and mode
    image_size = image.size
    original_mode = image.mode
    print(f"DEBUG: Original image size: {image_size}, mode: {original_mode}")
    
    input_images = transform_image(image).unsqueeze(0).to(device)
    
    # FIXED MODEL SELECTION: Use BiRefNet for quality, BiRefNet_lite for fast
    if fast_mode:
        # Fast mode: prefer lite model
        model = birefnet_lite if birefnet_lite is not None else birefnet
        model_name = "birefnet_lite (fast)"
    else:
        # Quality mode: prefer full model
        model = birefnet if birefnet is not None else birefnet_lite
        model_name = "birefnet (quality)"
    
    if model is None:
        print("ERROR: Model is None, returning original image")
        return image
    
    print(f"DEBUG: Using model: {model_name}")
    print(f"DEBUG: Input tensor shape: {input_images.shape}")
    
    with torch.no_grad():
        print("DEBUG: Running model inference...")
        preds = model(input_images)[-1].sigmoid().cpu()
        print("DEBUG: Model inference complete")
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    # Immediately free memory
    del input_images, preds, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # If transparent background requested, return RGBA image
    if transparent:
        # For WebM format, we need RGBA
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask)
        print(f"DEBUG: Returning transparent RGBA image, size: {image_rgba.size}")
        return image_rgba
    
    # For non-transparent, ensure RGB format for video consistency
    if isinstance(bg, str) and bg.startswith("#"):
        print(f"DEBUG: Using color background: {bg}")
        color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
        background = Image.new("RGB", image_size, color_rgb)
    elif isinstance(bg, Image.Image):
        print("DEBUG: Using Image background")
        background = bg.convert("RGB").resize(image_size)
    else:
        print(f"DEBUG: Using file background: {bg}")
        try:
            background = Image.open(bg).convert("RGB").resize(image_size)
        except:
            # Fallback to white if file can't be opened
            background = Image.new("RGB", image_size, (255, 255, 255))
    
    # Composite the image - ensure RGB output
    image_rgb = image.convert("RGB")
    result = Image.composite(image_rgb, background, mask)
    
    print(f"DEBUG: Result image size: {result.size}, mode: {result.mode}")
    return result

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def process_video_async(session_id, video_path, bg_type, bg_path, color, fps, video_handling, fast_mode, max_workers, output_format='mp4'):
    print(f"\n[{session_id}] === ASYNC PROCESSING STARTED ===")
    print(f"[{session_id}] Thread ID: {threading.current_thread().ident}")
    print(f"[{session_id}] Video path exists: {os.path.exists(video_path)}")
    
    video = None
    background_video = None
    processed_video = None
    
    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        print(f"[{session_id}] Starting video processing with settings: bg_type={bg_type}, fast_mode={fast_mode}, max_workers={max_workers}")
        start_time = time.time()
        print(f"[{session_id}] Loading video from: {video_path}")
        video = VideoFileClip(video_path)
        print(f"[{session_id}] Video loaded successfully: duration={video.duration}s, fps={video.fps}")
        if fps == 0:
            fps = video.fps
            print(f"[{session_id}] Using original video FPS: {fps}")
        else:
            print(f"[{session_id}] Using custom FPS: {fps}")
        
        # Store audio reference before extracting frames
        audio = video.audio
        print(f"[{session_id}] Loading video frames...")
        
        # Load all frames at once like the original Gradio implementation
        print(f"[{session_id}] Loading all video frames...")
        frames = list(video.iter_frames(fps=fps))
        total_frames = len(frames)
        print(f"[{session_id}] Total frames to process: {total_frames}")
        
        print(f"[{session_id}] Emitting processing_update event to client")
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'started',
            'message': 'Processing started... Extracting frames',
            'progress': 0,
            'elapsed_time': 0,
            'currentFrame': 0,
            'totalFrames': total_frames
        })
        print(f"[{session_id}] Event emitted")
        
        processed_frames = []
        
        # Handle background setup
        background_frames = None
        bg_image = None
        
        if bg_type == "Video" and bg_path:
            print(f"[{session_id}] Loading background video: {bg_path}")
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
        print(f"[{session_id}] Starting ThreadPoolExecutor with max_workers={max_workers}")
        bg_frame_index = 0  # Initialize background frame index
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frames for parallel processing - exactly like Gradio
            futures = []
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
            
            # Process results as they complete
            successfully_processed = 0
            failed_frames = []
            
            for i, future in enumerate(futures):
                # Check if session was cancelled before processing each result
                if session_id not in active_sessions:
                    print(f"[{session_id}] Processing cancelled during frame collection")
                    # Cancel all remaining futures
                    for remaining_future in futures[i:]:
                        remaining_future.cancel()
                    return  # Exit processing
                
                try:
                    result, _ = future.result(timeout=30)  # Increased timeout
                    processed_frames.append(result)
                    successfully_processed += 1
                    
                    elapsed_time = time.time() - start_time
                    progress = ((i + 1) / total_frames) * 100
                    
                    # Send preview for first frame, then every 5 frames or for the last frame
                    if i == 0 or i % 5 == 0 or i == total_frames - 1:
                        # Ensure result is a numpy array
                        if isinstance(result, np.ndarray):
                            preview_image = Image.fromarray(result.astype(np.uint8))
                        else:
                            preview_image = result
                        preview_base64 = image_to_base64(preview_image)
                        
                        socketio.emit('processing_update', {
                            'session_id': session_id,
                            'status': 'processing',
                            'message': f'Processing frame {i+1}/{total_frames}',
                            'progress': progress,
                            'elapsed_time': elapsed_time,
                            'preview_image': preview_base64,
                            'currentFrame': i + 1,
                            'totalFrames': total_frames
                        })
                        print(f"[{session_id}] Progress: {progress:.1f}%, Frame: {i+1}/{total_frames} (Success: {successfully_processed}, Failed: {len(failed_frames)})")
                except Exception as frame_error:
                    print(f"[{session_id}] ERROR processing frame {i}: {str(frame_error)}")
                    failed_frames.append(i)
                    # Use original frame as fallback
                    if i < len(frames):
                        processed_frames.append(frames[i])
                    else:
                        processed_frames.append(frames[0])
            
            print(f"[{session_id}] Frame processing complete: {successfully_processed} succeeded, {len(failed_frames)} failed")
            if failed_frames:
                print(f"[{session_id}] Failed frames: {failed_frames}")
                
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
        print(f"[{session_id}] Creating final video with {len(processed_frames)} processed frames")
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
            # Ensure all frames have the same dimensions
            if processed_frames:
                first_shape = processed_frames[0].shape
                print(f"[{session_id}] First frame shape: {first_shape}")
                
                # Normalize all frames to the same shape
                normalized_frames = []
                for idx, frame in enumerate(processed_frames):
                    if frame.shape != first_shape:
                        print(f"[{session_id}] Frame {idx} has different shape: {frame.shape}, resizing to {first_shape}")
                        # Resize frame to match first frame
                        pil_frame = Image.fromarray(frame.astype(np.uint8))
                        pil_frame = pil_frame.resize((first_shape[1], first_shape[0]), Image.LANCZOS)
                        frame = np.array(pil_frame)
                    normalized_frames.append(frame)
                
                processed_video = ImageSequenceClip(normalized_frames, fps=fps)
                
                # Try to add audio if available
                if audio is not None:
                    try:
                        processed_video = processed_video.set_audio(audio)
                        print(f"[{session_id}] Audio added to video")
                    except:
                        print(f"[{session_id}] Could not add audio, continuing without it")
                else:
                    print(f"[{session_id}] No audio in original video")
        except Exception as video_error:
            print(f"[{session_id}] Error creating video: {video_error}")
            import traceback
            traceback.print_exc()
            # Try without audio as fallback
            processed_video = ImageSequenceClip(processed_frames, fps=fps)
            print(f"[{session_id}] Created video without audio due to error")
        
        # Determine output format and codec with error handling
        output_ext = output_format.lower()
        try:
            if output_ext == 'webm':
                output_path = f"temp_output_{session_id}.webm"
                # Use VP9 codec for WebM with alpha channel support
                if bg_type == "Transparent":
                    codec = "libvpx-vp9"
                    codec_params = ["-pix_fmt", "yuva420p"]  # Support alpha channel
                else:
                    codec = "libvpx-vp9"
                    codec_params = []
                # Speed/quality tradeoffs for faster export
                codec_params += ["-b:v", "0", "-crf", "32", "-deadline", "good", "-cpu-used", "4"]
                print(f"[{session_id}] Writing WebM video to {output_path}")
                processed_video.write_videofile(output_path, codec=codec, threads=2, ffmpeg_params=codec_params, logger=None)
            elif output_ext == 'mov':
                output_path = f"temp_output_{session_id}.mov"
                # Use ProRes 4444 for MOV with alpha channel support
                if bg_type == "Transparent":
                    codec = "prores_ks"
                    codec_params = ["-profile:v", "4444", "-pix_fmt", "yuva444p10le"]
                else:
                    codec = "libx264"
                    codec_params = []
                print(f"[{session_id}] Writing MOV video to {output_path}")
                processed_video.write_videofile(output_path, codec=codec, threads=2, ffmpeg_params=codec_params, logger=None)
            else:  # Default to MP4
                output_path = f"temp_output_{session_id}.mp4"
                print(f"[{session_id}] Writing MP4 video to {output_path}")
                processed_video.write_videofile(output_path, codec="libx264", threads=2, ffmpeg_params=["-preset", "veryfast", "-crf", "20"], logger=None)
            
            print(f"[{session_id}] Video saved successfully as {output_ext.upper()}")
        except Exception as write_error:
            print(f"[{session_id}] Error writing video: {write_error}")
            # Fallback to basic MP4
            output_path = f"temp_output_{session_id}.mp4"
            print(f"[{session_id}] Attempting fallback to basic MP4")
            processed_video.write_videofile(output_path, codec="libx264", threads=1, logger=None)
            print(f"[{session_id}] Fallback video saved successfully")
        
        elapsed_time = time.time() - start_time
        print(f"[{session_id}] Processing completed in {elapsed_time:.2f} seconds")
        
        socketio.emit('processing_complete', {
            'session_id': session_id,
            'status': 'completed',
            'message': 'Processing complete!',
            'elapsed_time': elapsed_time,
            'output_file': output_path
        })
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        elapsed_time = time.time() - start_time
        socketio.emit('processing_error', {
            'session_id': session_id,
            'status': 'error',
            'message': f'Error processing video: {str(e)}',
            'elapsed_time': elapsed_time
        })
    finally:
        # Clean up video objects to prevent memory leaks and access violations
        try:
            if processed_video is not None:
                processed_video.close()
                print(f"[{session_id}] Closed processed video")
        except Exception as e:
            print(f"[{session_id}] Error closing processed video: {e}")
        
        try:
            if background_video is not None:
                background_video.close()
                print(f"[{session_id}] Closed background video")
        except Exception as e:
            print(f"[{session_id}] Error closing background video: {e}")
        
        try:
            if video is not None:
                if hasattr(video, 'audio') and video.audio is not None:
                    video.audio.close()
                video.close()
                print(f"[{session_id}] Closed input video")
        except Exception as e:
            print(f"[{session_id}] Error closing input video: {e}")
        
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        # Cleanup temporary input files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[{session_id}] Removed input video file")
            if bg_path and os.path.exists(bg_path):
                os.remove(bg_path)
                print(f"[{session_id}] Removed background file")
        except Exception as cleanup_err:
            print(f"[{session_id}] Cleanup warning: {cleanup_err}")

# REMOVED DUPLICATE FUNCTION - Using the first one above

@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        print("\n=== VIDEO PROCESSING REQUEST ===")
        print("Processing video request received...")
        print(f"Request method: {request.method}")
        print(f"Request files: {request.files.keys()}")
        print(f"Request form: {request.form.to_dict()}")
        
        # Check if models are available
        if not models_available:
            print("ERROR: Models not available")
            return jsonify({'error': 'AI models are not available due to system constraints. Please restart the application or increase virtual memory.'}), 503
        
        print("Models are available, attempting to load...")
        
        # Load models if needed
        if not load_models_if_needed():
            print("ERROR: Failed to load models")
            return jsonify({'error': 'Failed to load AI models. Please restart the application or increase virtual memory.'}), 503
            
        print("Models loaded successfully, proceeding with video processing...")
            
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = True
        print(f"Created session: {session_id}")
        
        # Get form data
        video_file = request.files.get('video')
        bg_type = request.form.get('bg_type', 'Color')
        color = request.form.get('color', '#00FF00')
        fps = int(request.form.get('fps', 0))
        video_handling = request.form.get('video_handling', 'slow_down')
        fast_mode = request.form.get('fast_mode', 'true').lower() == 'true'
        output_format = request.form.get('output_format', 'mp4')
        # Get max_workers parameter - default to 4 for parallel processing 
        try:
            max_workers = int(request.form.get('max_workers', 4))
            # Limit to reasonable value based on CPU cores
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            max_workers = min(max_workers, max(1, cpu_count - 1))  # Use at most CPU count - 1
            print(f"Using max_workers={max_workers} (system has {cpu_count} CPU cores)")
        except Exception as e:
            print(f"Error setting max_workers: {e}, defaulting to 4")
            max_workers = 4
        
        if not video_file:
            return jsonify({'error': 'No video file provided'}), 400
        
        # Save uploaded video
        video_filename = secure_filename(f"input_{session_id}_{video_file.filename}")
        video_path = os.path.join(tempfile.gettempdir(), video_filename)
        video_file.save(video_path)
        
        # Handle background file if provided
        bg_path = None
        if bg_type in ['Image', 'Video']:
            bg_file = request.files.get('background')
            if bg_file:
                bg_filename = secure_filename(f"bg_{session_id}_{bg_file.filename}")
                bg_path = os.path.join(tempfile.gettempdir(), bg_filename)
                bg_file.save(bg_path)
        
        # Start processing in background
        print(f"Starting background thread for session {session_id}")
        print(f"Thread args: video_path={video_path}, bg_type={bg_type}, bg_path={bg_path}, color={color}, fps={fps}, video_handling={video_handling}, fast_mode={fast_mode}, max_workers={max_workers}, output_format={output_format}")
        
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

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        # Prevent path traversal by forcing a basename and whitelisting pattern
        safe_name = os.path.basename(filename)
        # Accept mp4, webm, or mov files
        if not safe_name.startswith('temp_output_') or not (safe_name.endswith('.mp4') or safe_name.endswith('.webm') or safe_name.endswith('.mov')):
            return jsonify({'error': 'Invalid filename'}), 400
        file_path = os.path.join(os.getcwd(), safe_name)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Set appropriate mime type
        if safe_name.endswith('.webm'):
            mimetype = 'video/webm'
        elif safe_name.endswith('.mov'):
            mimetype = 'video/quicktime'
        else:
            mimetype = 'video/mp4'
        
        # Serve inline so the <video> tag can play it; also allow CORS
        response = send_file(file_path, as_attachment=False, download_name=f"processed_{safe_name}", mimetype=mimetype)
        try:
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        except Exception:
            pass
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 404

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
            'device': device,
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
            'device': device,
            'models_available': models_available,
            'models_loaded': birefnet is not None and birefnet_lite is not None,
            'error': str(e)
        })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify server is responding"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is responding',
        'models_available': models_available,
        'models_loaded': models_loaded
    })

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

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'\n=== CLIENT CONNECTED: {client_id} ==="')
    emit('connected', {'message': 'Connected to video processing server'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    print(f'\n=== CLIENT DISCONNECTED: {client_id} ==="')

if __name__ == '__main__':
    print(f"Found ffmpeg at {ffmpeg_path}")
    print(f"Using device: {device}")
    print("Initializing API server...")
    print("Model cache directory ready")
    
    # Don't preload on startup - load on first use
    print("Server ready - models will load on first use")
    
    # Start the socket server
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
