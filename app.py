import gradio as gr
from loadimg import load_img
import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from pydub import AudioSegment
from PIL import Image
import numpy as np
import os
import tempfile
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from moviepy import VideoFileClip, vfx, concatenate_videoclips, ImageSequenceClip

torch.set_float32_matmul_precision("medium")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load both BiRefNet models
birefnet = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True)
birefnet.to(device)
birefnet_lite = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
birefnet_lite.to(device)

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Function to process a single frame
def process_frame(frame, bg_type, bg, fast_mode, bg_frame_index, background_frames, color):
    try:
        pil_image = Image.fromarray(frame)
        if bg_type == "Color":
            processed_image = process(pil_image, color, fast_mode)
        elif bg_type == "Image":
            processed_image = process(pil_image, bg, fast_mode)
        elif bg_type == "Video":
            background_frame = background_frames[bg_frame_index]  # Access the correct background frame
            bg_frame_index += 1
            background_image = Image.fromarray(background_frame)
            processed_image = process(pil_image, background_image, fast_mode)
        else:
            processed_image = pil_image  # Default to original image if no background is selected
        return np.array(processed_image), bg_frame_index
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, bg_frame_index

@spaces.GPU
def fn(vid, bg_type="Color", bg_image=None, bg_video=None, color="#00FF00", fps=0, video_handling="slow_down", fast_mode=True, max_workers=6):
    try:
        start_time = time.time()  # Start the timer
        video = VideoFileClip(vid)
        if fps == 0:
            fps = video.fps
        
        audio = video.audio
        frames = list(video.iter_frames(fps=fps))
        
        processed_frames = []
        yield gr.update(visible=True), gr.update(visible=False), f"Processing started... Elapsed time: 0 seconds"
        
        if bg_type == "Video":
            background_video = VideoFileClip(bg_video)
            if background_video.duration < video.duration:
                if video_handling == "slow_down":
                    background_video = background_video.fx(vfx.speedx, factor=video.duration / background_video.duration)
                else:  # video_handling == "loop"
                    background_video = concatenate_videoclips([background_video] * int(video.duration / background_video.duration + 1))
            background_frames = list(background_video.iter_frames(fps=fps))
        else:
            background_frames = None
        
        bg_frame_index = 0  # Initialize background frame index

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Pass bg_frame_index as part of the function arguments
            futures = [executor.submit(process_frame, frames[i], bg_type, bg_image, fast_mode, bg_frame_index + i, background_frames, color) for i in range(len(frames))] 
            for i, future in enumerate(futures):
                result, _ = future.result() #  No need to update bg_frame_index here
                processed_frames.append(result)
                elapsed_time = time.time() - start_time
                yield result, None, f"Processing frame {i+1}/{len(frames)}... Elapsed time: {elapsed_time:.2f} seconds"
        
        processed_video = ImageSequenceClip(processed_frames, fps=fps)
        processed_video = processed_video.set_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_filepath = temp_file.name
            processed_video.write_videofile(temp_filepath, codec="libx264")
        
        elapsed_time = time.time() - start_time
        yield gr.update(visible=False), gr.update(visible=True), f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds"
        yield processed_frames[-1], temp_filepath, f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds"
    
    except Exception as e:
        print(f"Error: {e}")
        elapsed_time = time.time() - start_time
        yield gr.update(visible=False), gr.update(visible=True), f"Error processing video: {e}. Elapsed time: {elapsed_time:.2f} seconds"
        yield None, f"Error processing video: {e}", f"Error processing video: {e}. Elapsed time: {elapsed_time:.2f} seconds"

def process(image, bg, fast_mode=False):
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(device)
    model = birefnet_lite if fast_mode else birefnet
    
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    if isinstance(bg, str) and bg.startswith("#"):
        color_rgb = tuple(int(bg[i:i+2], 16) for i in (1, 3, 5))
        background = Image.new("RGBA", image_size, color_rgb + (255,))
    elif isinstance(bg, Image.Image):
        background = bg.convert("RGBA").resize(image_size)
    else:
        background = Image.open(bg).convert("RGBA").resize(image_size)
    
    image = Image.composite(image, background, mask)
    return image

with gr.Blocks(theme=gr.themes.Ocean()) as demo:
    gr.Markdown("# Video Background Remover & Changer\n### You can replace image background with any color, image or video.\nNOTE: As this Space is running on ZERO GPU it has limit. It can handle approx 200 frames at once. So, if you have a big video than use small chunks or Duplicate this space.")
    
    with gr.Row():
        in_video = gr.Video(label="Input Video", interactive=True)
        stream_image = gr.Image(label="Streaming Output", visible=False)
        out_video = gr.Video(label="Final Output Video")
    
    submit_button = gr.Button("Change Background", interactive=True)
    
    with gr.Row():
        fps_slider = gr.Slider(
            minimum=0,
            maximum=60,
            step=1,
            value=0,
            label="Output FPS (0 will inherit the original fps value)",
            interactive=True
        )
        bg_type = gr.Radio(["Color", "Image", "Video"], label="Background Type", value="Color", interactive=True)
        color_picker = gr.ColorPicker(label="Background Color", value="#00FF00", visible=True, interactive=True)
        bg_image = gr.Image(label="Background Image", type="filepath", visible=False, interactive=True)
        bg_video = gr.Video(label="Background Video", visible=False, interactive=True)
        
        with gr.Column(visible=False) as video_handling_options:
            video_handling_radio = gr.Radio(["slow_down", "loop"], label="Video Handling", value="slow_down", interactive=True)
        
        fast_mode_checkbox = gr.Checkbox(label="Fast Mode (Use BiRefNet_lite)", value=True, interactive=True)
        max_workers_slider = gr.Slider( minimum=1, maximum=32, step=1, value=6, label="Max Workers", info="Determines how many frames to process in parallel", interactive=True )

    time_textbox = gr.Textbox(label="Time Elapsed", interactive=False)

    def update_visibility(bg_type):
        if bg_type == "Color":
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif bg_type == "Image":
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif bg_type == "Video":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    bg_type.change(update_visibility, inputs=bg_type, outputs=[color_picker, bg_image, bg_video, video_handling_options])

    examples = gr.Examples(
        [
            ["rickroll-2sec.mp4", "Video", None, "background.mp4"],
            ["rickroll-2sec.mp4", "Image", "images.webp", None],
            ["rickroll-2sec.mp4", "Color", None, None],
        ],
        inputs=[in_video, bg_type, bg_image, bg_video],
        outputs=[stream_image, out_video, time_textbox],
        fn=fn,
        cache_examples=True,
        cache_mode="eager",
    )

    submit_button.click(
        fn,
        inputs=[in_video, bg_type, bg_image, bg_video, color_picker, fps_slider, video_handling_radio, fast_mode_checkbox, max_workers_slider],
        outputs=[stream_image, out_video, time_textbox],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)