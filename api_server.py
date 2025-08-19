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
from moviepy.editor import VideoFileClip, ImageSequenceClip
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'video-background-remover'

# Configure CORS from environment or default to localhost and file protocols used by Electron
allowed_origins_env = os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://localhost:5000')
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',') if origin.strip()]
CORS(app, origins=allowed_origins)
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
        if memory.percent > 95:  # Increased threshold to 95%
            print(f"[{session_id}] CRITICAL: Memory usage at {memory.percent}%")
            # Force garbage collection before stopping
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        elif memory.percent > 85:  # Warning at 85%
            print(f"[{session_id}] WARNING: Memory usage at {memory.percent}%")
            # Proactive cleanup at warning level
            gc.collect()
        return True
    except Exception:
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup completed on exit")
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

def load_models_if_needed():
    global birefnet, birefnet_lite, models_available, models_loaded
    
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
            
            birefnet_lite = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_lite", 
                trust_remote_code=True,
                cache_dir="./model_cache"  # Cache models locally
            )
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
        
        models_loaded = True
        print("All models loaded successfully and cached!")
        return True
    except Exception as e:
        print(f"Failed to load models: {e}")
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

def process_frame(frame, bg_type, bg, fast_mode, bg_frame_index, background_frames, color):
    try:
        pil_image = Image.fromarray(frame)
        if bg_type == "Transparent":
            processed_image = process_image(pil_image, None, fast_mode, transparent=True)
        elif bg_type == "Color":
            processed_image = process_image(pil_image, color, fast_mode)
        elif bg_type == "Image":
            processed_image = process_image(pil_image, bg, fast_mode)
        elif bg_type == "Video":
            if background_frames and len(background_frames) > 0:
                safe_index = bg_frame_index % len(background_frames)
                background_frame = background_frames[safe_index]
                bg_frame_index += 1
                background_image = Image.fromarray(background_frame)
                processed_image = process_image(pil_image, background_image, fast_mode)
            else:
                processed_image = process_image(pil_image, color, fast_mode)
        else:
            processed_image = pil_image
        return np.array(processed_image), bg_frame_index
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, bg_frame_index

def process_image(image, bg, fast_mode=False, transparent=False):
    image_size = image.size
    
    # Prepare input with memory optimization
    input_images = transform_image(image).unsqueeze(0).to(device)
    model = birefnet_lite if (fast_mode or birefnet is None) else birefnet
    if model is None:
        # Model failed to load; return original image to avoid crashing
        return image
    
    # Process with strict memory management
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
        
        # Immediately free memory
        del input_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    # Clean up tensors
    del preds, pred
    
    # If transparent background requested, return image with alpha channel
    if transparent:
        # Convert image to RGBA if not already
        image_rgba = image.convert("RGBA")
        # Apply mask as alpha channel
        image_rgba.putalpha(mask)
        return image_rgba
    
    # Otherwise composite with background
    if isinstance(bg, str) and bg.startswith("#"):
        color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
        background = Image.new("RGBA", image_size, color_rgb + (255,))
    elif isinstance(bg, Image.Image):
        background = bg.convert("RGBA").resize(image_size)
    else:
        background = Image.open(bg).convert("RGBA").resize(image_size)
    
    image = Image.composite(image, background, mask)
    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def process_video_async(session_id, video_path, bg_type, bg_path, color, fps, video_handling, fast_mode, max_workers, output_format='mp4'):
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
        video = VideoFileClip(video_path)
        if fps == 0:
            fps = video.fps
            print(f"[{session_id}] Using original video FPS: {fps}")
        else:
            print(f"[{session_id}] Using custom FPS: {fps}")
        
        # Store audio reference before extracting frames
        audio = video.audio
        print(f"[{session_id}] Loading video frames...")
        
        # For large videos, avoid loading all frames at once
        # First, get the total frame count
        total_frames = int(video.duration * fps) if fps else int(video.duration * video.fps)
        print(f"[{session_id}] Total frames to process: {total_frames}")
        
        # Create a frame generator to avoid memory issues
        frame_generator = video.iter_frames(fps=fps)
        frames = []
        
        socketio.emit('processing_update', {
            'session_id': session_id,
            'status': 'started',
            'message': 'Processing started...',
            'progress': 0,
            'elapsed_time': 0,
            'currentFrame': 0,
            'totalFrames': total_frames
        })
        
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
        
        # Process frames in batches to avoid memory issues - use very small batches
        batch_size = min(5, total_frames)  # Process max 5 frames at a time (extremely conservative)
        num_batches = (total_frames + batch_size - 1) // batch_size
        
        # Limit total frames for memory safety
        if total_frames > 200:
            print(f"[{session_id}] Warning: Video has {total_frames} frames, limiting to 200 for memory safety")
            frames = frames[:200]
            total_frames = 200
            num_batches = (total_frames + batch_size - 1) // batch_size
        
        print(f"[{session_id}] Processing in {num_batches} batches of up to {batch_size} frames each")
        
        frame_idx_global = 0
        for batch_idx in range(num_batches):
            if session_id not in active_sessions:
                print(f"[{session_id}] Processing cancelled")
                return  # Session was cancelled
                
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            batch_size_current = end_idx - start_idx
            
            # Load frames for this batch only
            batch_frames = []
            for _ in range(batch_size_current):
                try:
                    frame = next(frame_generator)
                    batch_frames.append(frame)
                except StopIteration:
                    break  # No more frames
            
            if not batch_frames:
                break  # No frames loaded
            
            print(f"[{session_id}] Processing batch {batch_idx+1}/{num_batches}, frames {start_idx}-{end_idx}")
            
            # Use single-threaded processing for maximum stability
            futures = []
            actual_batch_size = len(batch_frames)
            # Process frames one by one for maximum stability
            for i in range(actual_batch_size):
                current_frame = batch_frames[i]
                frame_global_idx = start_idx + i
                effective_bg_type = bg_type
                effective_bg = bg_image
                effective_background_frames = background_frames
                
                # Check memory before processing each frame
                if not monitor_memory_usage(session_id):
                    print(f"[{session_id}] Stopping processing due to memory constraints")
                    break
                
                if bg_type == "Video" and background_video is not None:
                    # Compute background frame on-the-fly to avoid storing all frames
                    try:
                        t = (frame_global_idx) / fps if fps else (frame_global_idx) / video.fps
                        # Ensure we don't go beyond the background video duration
                        t = min(t, background_video.duration - 0.001)  # Small offset to avoid edge case
                        bg_np = background_video.get_frame(t)
                        bg_pil = Image.fromarray(bg_np)
                        effective_bg_type = "Image"
                        effective_bg = bg_pil
                        effective_background_frames = None
                    except Exception as bg_err:
                        print(f"[{session_id}] Warning: failed to get background frame at t={t}: {bg_err}")
                        effective_bg_type = "Color"
                        effective_bg = color
                        effective_background_frames = None
                
                # Process frame synchronously
                try:
                    result, _ = process_frame(
                        current_frame,
                        effective_bg_type,
                        effective_bg,
                        fast_mode,
                        bg_frame_index + frame_global_idx,
                        effective_background_frames,
                        color
                    )
                    processed_frames.append(result)
                    
                    frame_idx = start_idx + i
                    elapsed_time = time.time() - start_time
                    progress = ((frame_idx + 1) / total_frames) * 100
                    
                    # Aggressive garbage collection after each frame
                    if (frame_idx + 1) % 1 == 0:  # Every frame
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Send preview every 3 frames or for the last frame
                    if frame_idx % 3 == 0 or frame_idx == total_frames - 1:
                        # Send preview of current frame
                        preview_image = Image.fromarray(result)
                        preview_base64 = image_to_base64(preview_image)
                        
                        socketio.emit('processing_update', {
                            'session_id': session_id,
                            'status': 'processing',
                            'message': f'Processing frame {frame_idx+1}/{total_frames}',
                            'progress': progress,
                            'elapsed_time': elapsed_time,
                            'preview_image': preview_base64,
                            'currentFrame': frame_idx + 1,
                            'totalFrames': total_frames
                        })
                        print(f"[{session_id}] Progress: {progress:.1f}%, Frame: {frame_idx+1}/{total_frames}")
                        
                except Exception as frame_error:
                    print(f"[{session_id}] Error processing frame {frame_idx}: {frame_error}")
                    # Use original frame as fallback
                    processed_frames.append(current_frame)
                
            # Single-threaded processing completed above
        
        if session_id not in active_sessions:
            print(f"[{session_id}] Processing cancelled before video creation")
            return  # Session was cancelled
        
        # Clear memory before final video creation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create final video
        print(f"[{session_id}] Creating final video with {len(processed_frames)} processed frames")
        try:
            processed_video = ImageSequenceClip(processed_frames, fps=fps)
            if audio is not None:
                processed_video = processed_video.with_audio(audio)
                print(f"[{session_id}] Audio added to video")
            else:
                print(f"[{session_id}] No audio in original video")
        except Exception as video_error:
            print(f"[{session_id}] Error creating video: {video_error}")
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
                print(f"[{session_id}] Writing WebM video to {output_path}")
                processed_video.write_videofile(output_path, codec=codec, threads=2, ffmpeg_params=codec_params)
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
                processed_video.write_videofile(output_path, codec=codec, threads=2, ffmpeg_params=codec_params)
            else:  # Default to MP4
                output_path = f"temp_output_{session_id}.mp4"
                print(f"[{session_id}] Writing MP4 video to {output_path}")
                processed_video.write_videofile(output_path, codec="libx264", threads=2)
            
            print(f"[{session_id}] Video saved successfully as {output_ext.upper()}")
        except Exception as write_error:
            print(f"[{session_id}] Error writing video: {write_error}")
            # Fallback to basic MP4
            output_path = f"temp_output_{session_id}.mp4"
            print(f"[{session_id}] Attempting fallback to basic MP4")
            processed_video.write_videofile(output_path, codec="libx264", threads=1)
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

def process_video_async(session_id, video_path, bg_type, bg_path, color, fps, video_handling, fast_mode, max_workers, output_format):
    """Asynchronous video processing function"""
    try:
        print(f"[{session_id}] Starting video processing with settings: bg_type={bg_type}, fast_mode={fast_mode}, max_workers={max_workers}")
        
        # Load video
        clip = VideoFileClip(video_path)
        if fps == 0:
            fps = clip.fps
        print(f"[{session_id}] Using original video FPS: {fps}")
        
        # Extract audio
        audio = None
        try:
            audio = clip.audio
        except Exception as e:
            print(f"[{session_id}] No audio track or audio extraction failed: {e}")
        
        # Load frames
        print(f"[{session_id}] Loading video frames...")
        frames = list(clip.iter_frames())
        total_frames = len(frames)
        print(f"[{session_id}] Total frames to process: {total_frames}")
        
        # Process frames in small batches to manage memory
        batch_size = 2  # Very small batches for memory management
        processed_frames = []
        
        # Calculate batches
        num_batches = (total_frames + batch_size - 1) // batch_size
        print(f"[{session_id}] Processing in {num_batches} batches of up to {batch_size} frames each")
        
        model = birefnet_lite if fast_mode else birefnet
        if model is None:
            raise Exception("Model not loaded")
        
        for batch_idx in range(num_batches):
            if session_id not in active_sessions:
                print(f"[{session_id}] Processing cancelled by user")
                break
                
            # Check memory before processing batch
            if not monitor_memory_usage(session_id):
                print(f"[{session_id}] Stopping processing due to memory constraints")
                continue  # Skip this batch but continue with others
                
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_frames)
            batch_frames = frames[start_idx:end_idx]
            
            print(f"[{session_id}] Processing batch {batch_idx + 1}/{num_batches}, frames {start_idx}-{end_idx}")
            
            # Process each frame in the batch
            for i, frame in enumerate(batch_frames):
                frame_idx = start_idx + i
                
                # Check memory before each frame
                if not monitor_memory_usage(session_id):
                    print(f"[{session_id}] Stopping processing due to memory constraints")
                    break
                
                try:
                    # Convert frame to PIL Image
                    pil_image = Image.fromarray(frame)
                    
                    # Process with model
                    with torch.no_grad():
                        # Preprocess
                        transform = transforms.Compose([
                            transforms.Resize((1024, 1024)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        input_tensor = transform(pil_image).unsqueeze(0).to(device)
                        
                        # Get prediction
                        prediction = model(input_tensor)
                        prediction = torch.sigmoid(prediction)
                        prediction = prediction.cpu().numpy().squeeze()
                        
                        # Resize mask back to original size
                        mask = Image.fromarray((prediction * 255).astype(np.uint8)).resize(pil_image.size)
                        mask_array = np.array(mask) / 255.0
                        
                        # Apply background replacement
                        if bg_type == 'Transparent':
                            # Create RGBA image
                            result = Image.new('RGBA', pil_image.size)
                            result_array = np.array(result)
                            pil_array = np.array(pil_image)
                            
                            result_array[:, :, :3] = pil_array
                            result_array[:, :, 3] = (mask_array * 255).astype(np.uint8)
                            
                            processed_frame = Image.fromarray(result_array, 'RGBA')
                        else:
                            # Solid color background
                            if bg_type == 'Color':
                                # Convert hex color to RGB
                                bg_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                                bg_image = Image.new('RGB', pil_image.size, bg_color)
                            else:
                                # For now, fallback to green screen for other types
                                bg_image = Image.new('RGB', pil_image.size, (0, 255, 0))
                            
                            # Composite foreground and background
                            pil_array = np.array(pil_image)
                            bg_array = np.array(bg_image)
                            
                            # Apply mask
                            mask_3d = np.stack([mask_array] * 3, axis=2)
                            result_array = pil_array * mask_3d + bg_array * (1 - mask_3d)
                            
                            processed_frame = Image.fromarray(result_array.astype(np.uint8))
                        
                        # Convert back to numpy array for MoviePy
                        processed_frames.append(np.array(processed_frame.convert('RGB')))
                        
                        # Progress update
                        progress = ((frame_idx + 1) / total_frames) * 100
                        socketio.emit('progress', {
                            'session_id': session_id,
                            'progress': progress,
                            'current_frame': frame_idx + 1,
                            'total_frames': total_frames
                        })
                        print(f"[{session_id}] Progress: {progress:.1f}%, Frame: {frame_idx + 1}/{total_frames}")
                        
                except Exception as frame_error:
                    print(f"[{session_id}] Error processing frame {frame_idx}: {frame_error}")
                    continue
            
            # Cleanup after each batch
            force_memory_cleanup()
        
        # Create final video
        print(f"[{session_id}] Creating final video with {len(processed_frames)} processed frames")
        
        if len(processed_frames) == 0:
            raise Exception("No frames were successfully processed")
        
        # Create video clip
        processed_video = ImageSequenceClip(processed_frames, fps=fps)
        
        # Add audio if available
        if audio is not None:
            try:
                processed_video = processed_video.with_audio(audio)
                print(f"[{session_id}] Audio added to video")
            except Exception as audio_error:
                print(f"[{session_id}] Could not add audio: {audio_error}")
        
        # Generate output filename
        output_filename = f"temp_output_{session_id}.mp4"  # Always use MP4 to avoid WebM issues
        output_path = output_filename
        
        # Write video file (always MP4 for compatibility)
        print(f"[{session_id}] Writing MP4 video to {output_filename}")
        processed_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac' if audio is not None else None,
            temp_audiofile=f"temp_audio_{session_id}.m4a",
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # Cleanup
        clip.close()
        processed_video.close()
        
        # Notify completion
        socketio.emit('processing_complete', {
            'session_id': session_id,
            'download_url': f'/api/download/{output_filename}',
            'message': 'Video processing completed successfully'
        })
        
        print(f"[{session_id}] Processing completed successfully")
        
    except Exception as e:
        print(f"[{session_id}] Processing error: {e}")
        socketio.emit('processing_error', {
            'session_id': session_id,
            'error': str(e)
        })
    finally:
        # Cleanup session
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        # Cleanup files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if bg_path and os.path.exists(bg_path):
                os.remove(bg_path)
        except Exception as cleanup_error:
            print(f"[{session_id}] Cleanup error: {cleanup_error}")
        
        # Final memory cleanup
        force_memory_cleanup()

@app.route('/api/process_video', methods=['POST'])
def process_video():
    try:
        print("Processing video request received...")
        
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
        
        # Get form data
        video_file = request.files.get('video')
        bg_type = request.form.get('bg_type', 'Color')
        color = request.form.get('color', '#00FF00')
        fps = int(request.form.get('fps', 0))
        video_handling = request.form.get('video_handling', 'slow_down')
        fast_mode = request.form.get('fast_mode', 'true').lower() == 'true'
        output_format = request.form.get('output_format', 'mp4')
        # Extremely conservative worker limits to avoid memory issues
        try:
            max_workers = int(request.form.get('max_workers', 1))
        except Exception:
            max_workers = 1
        # Maximum 1 worker to avoid memory issues (single-threaded processing)
        max_workers = 1  # Force single-threaded for maximum stability
        
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
        thread = threading.Thread(
            target=process_video_async,
            args=(session_id, video_path, bg_type, bg_path, color, fps, video_handling, fast_mode, max_workers, output_format)
        )
        thread.start()
        
        return jsonify({
            'session_id': session_id,
            'message': 'Processing started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cancel_processing', methods=['POST'])
def cancel_processing():
    session_id = request.json.get('session_id')
    if session_id in active_sessions:
        del active_sessions[session_id]
        return jsonify({'message': 'Processing cancelled'})
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
        
        return send_file(file_path, as_attachment=True, download_name=f"processed_{safe_name}", mimetype=mimetype)
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
    print('Client connected')
    emit('connected', {'message': 'Connected to video processing server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
