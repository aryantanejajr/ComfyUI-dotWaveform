import torch
import torchaudio
import numpy as np
from PIL import Image, ImageDraw
import librosa
import math
import tempfile
import os
from typing import Dict, List, Tuple, Any
import comfy.model_management
from comfy.utils import ProgressBar
import copy

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def safe_hex_to_rgb(hex_color: str, fallback: Tuple[int, int, int] = (0, 255, 255)) -> Tuple[int, int, int]:
    try:
        return hex_to_rgb(hex_color)
    except:
        return fallback


class DottedWaveformVisualizer:
    
    DESCRIPTION = "Creates animated dotted waveform visualizations from audio input with customizable appearance, colors, and animation styles"
    
    OUTPUT_TOOLTIPS = (
        "Generated waveform animation frames as image sequence",
        "Original audio data passed through unchanged", 
        "Actual frames per second value used for the animation"
    )
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio file to create waveform visualization from"
                }),
                
                "width": ("INT", {
                    "default": 1280,
                    "min": 100,
                    "max": 1920,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Output image width in pixels. Default 1280 (landscape HD). Max 1920 for Full HD."
                }),
                "height": ("INT", {
                    "default": 720,
                    "min": 100,
                    "max": 1920,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Output image height in pixels. Default 720 (landscape HD). Max 1920 for square/portrait."
                }),
                
                "size": ("INT", {
                    "default": 6,
                    "min": 1,
                    "max": 30,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Size of individual dots/bars in pixels. Larger size = bolder appearance. Range: 1-30px."
                }),
                "spacing": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 40,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Distance between dot/bar centers. Smaller values = denser waveform. Range: 1-40px."
                }),
                
                "dot_color": ("STRING", {
                    "default": "#00FFFF",
                    "multiline": False,
                    "tooltip": "Hex color for waveform dots. Examples: #00FFFF (cyan), #FF0000 (red), #00FF00 (green), #FFFFFF (white)"
                }),
                "background_color": ("STRING", {
                    "default": "#000000", 
                    "multiline": False,
                    "tooltip": "Hex color for background. Examples: #000000 (black), #FFFFFF (white), #333333 (dark gray)"
                }),
                
                "animation_style": (["scrolling", "breathing", "radial", "bars", "wave"], {
                    "tooltip": "Choose animation style. SCROLLING: Continuous waveform. BREATHING: Pulsing dots. RADIAL: Expanding rings. BARS: Frequency bars. WAVE: Sine wave patterns."
                }),
                "max_height": ("INT", {
                    "default": 60,
                    "min": 5,
                    "max": 150,
                    "step": 5,
                    "display": "slider",
                    "tooltip": "Maximum waveform size as % of image height. Higher = taller waveforms. Affects all animation styles including radial radius."
                }),
                "fps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Animation speed in frames per second. Higher FPS = smoother but larger file size. Range: 1-60 FPS."
                }),
                
                "max_frames": ("INT", {
                    "default": 300,
                    "min": 0,
                    "max": 1000,
                    "step": 10,
                    "display": "slider",
                    "tooltip": "Frame limit to prevent hangs on long audio. 0 = unlimited. 300 frames ≈ 30 seconds at 10 FPS."
                }),
                
                "opacity_mode": (["uniform", "3_levels", "5_levels", "10_levels"], {
                    "default": "uniform",
                    "tooltip": "UNIFORM: All dots same brightness (fastest). 3_LEVELS: 0%/50%/100% opacity. 5_LEVELS: More variation. 10_LEVELS: Smoothest gradients (densest look)."
                }),
            },
            "optional": {
                "window_size": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "SCROLLING ONLY: Time window in seconds. How much audio timeline visible at once. Smaller = more detailed, larger = more overview."
                }),
                
                "preview_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Toggle ON for fast preview with sine wave pattern (no audio processing). Toggle OFF for normal audio-based animation."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "audio", "fps_output")
    OUTPUT_NODE = False
    FUNCTION = "generate_waveform"
    CATEGORY = "audio/visualization"

    def load_audio_with_fallback(self, audio_data: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        try:
            # Handle LazyAudioMap from VideoHelperSuite
            if hasattr(audio_data, '__class__') and 'LazyAudioMap' in str(type(audio_data)):
                print("Detected LazyAudioMap from VideoHelperSuite, converting...")
                
                # First, try to inspect the LazyAudioMap object
                print(f"LazyAudioMap attributes: {[attr for attr in dir(audio_data) if not attr.startswith('_')]}")
                
                waveform = None
                sample_rate = 22050
                
                # Method 1: Try calling it as a function
                try:
                    waveform = audio_data()
                    print(f"Method 1 (call): Success - {type(waveform)}")
                except Exception as e:
                    print(f"Method 1 (call) failed: {e}")
                
                # Method 2: Try accessing common attributes
                if waveform is None:
                    for attr_name in ['waveform', 'audio', 'data', 'tensor']:
                        try:
                            waveform = getattr(audio_data, attr_name)
                            print(f"Method 2 (attr '{attr_name}'): Success - {type(waveform)}")
                            break
                        except Exception as e:
                            print(f"Method 2 (attr '{attr_name}') failed: {e}")
                
                # Method 3: Try dictionary-like access (LazyAudioMap acts like a dict)
                if waveform is None:
                    try:
                        # Get available keys
                        if hasattr(audio_data, 'keys'):
                            available_keys = list(audio_data.keys())
                            print(f"Available LazyAudioMap keys: {available_keys}")
                            
                            # Try common audio keys
                            for key in ['waveform', 'audio', 'data', 'tensor']:
                                if key in available_keys:
                                    waveform = audio_data.get(key)
                                    print(f"Method 3 (dict key '{key}'): Success - {type(waveform)}")
                                    break
                            
                            # If no common keys found, try the first available key
                            if waveform is None and available_keys:
                                first_key = available_keys[0]
                                waveform = audio_data.get(first_key)
                                print(f"Method 3 (dict first key '{first_key}'): Success - {type(waveform)}")
                    except Exception as e:
                        print(f"Method 3 (dict access) failed: {e}")
                
                # Method 4: Try slicing (LazyAudioMap might support indexing)
                if waveform is None:
                    try:
                        # Try to get all data using slice notation
                        waveform = audio_data[:]
                        print(f"Method 4 (slice): Success - {type(waveform)}")
                    except Exception as e:
                        print(f"Method 4 (slice) failed: {e}")
                
                # Method 5: Try to convert to tensor directly
                if waveform is None:
                    try:
                        # Check if it's iterable and has length
                        if hasattr(audio_data, '__len__') and hasattr(audio_data, '__getitem__'):
                            # Try to access as array-like
                            waveform = torch.tensor([float(x) for x in audio_data])
                            print(f"Method 5 (array-like): Success - shape {waveform.shape}")
                    except Exception as e:
                        print(f"Method 5 (array-like) failed: {e}")
                
                # Get sample rate
                try:
                    sample_rate = getattr(audio_data, 'sample_rate', 22050)
                    if not isinstance(sample_rate, (int, float)):
                        sample_rate = 22050
                except:
                    sample_rate = 22050
                
                # If we still don't have waveform data, create fallback
                if waveform is None:
                    print("All LazyAudioMap conversion methods failed, using fallback")
                    return np.zeros(1024), 22050
                    
                print(f"LazyAudioMap converted successfully - Type: {type(waveform)}, Sample rate: {sample_rate}")
            elif isinstance(audio_data, dict):
                if 'waveform' in audio_data and 'sample_rate' in audio_data:
                    waveform = audio_data['waveform']
                    sample_rate = audio_data['sample_rate']
                elif 'audio' in audio_data:
                    waveform = audio_data['audio']
                    sample_rate = audio_data.get('sample_rate', 22050)
                else:
                    print("Audio format not recognized, trying first dict value")
                    first_key = list(audio_data.keys())[0]
                    waveform = audio_data[first_key]
                    sample_rate = audio_data.get('sample_rate', 22050)
            else:
                print("Audio data is not a dict, using as waveform directly")
                waveform = audio_data
                sample_rate = 22050
            
            # Convert to numpy array
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.detach().cpu().numpy()
            else:
                audio_np = np.array(waveform, dtype=np.float32)
                
            # Handle different tensor shapes
            if len(audio_np.shape) == 3:
                audio_np = audio_np[0]
            
            if len(audio_np.shape) == 2 and audio_np.shape[0] > 1:
                audio_np = np.mean(audio_np, axis=0)
            elif len(audio_np.shape) == 2:
                audio_np = audio_np[0]
                
            print(f"Final audio shape: {audio_np.shape}, Sample rate: {sample_rate}")
            return audio_np, sample_rate
            
        except Exception as e:
            print(f"Audio loading error: {e}")
            print(f"Audio data keys: {list(audio_data.keys()) if isinstance(audio_data, dict) else 'Not a dict'}")
            print(f"Audio data type: {type(audio_data)}")
            return np.zeros(1024), 22050

    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int, fps: int) -> List[np.ndarray]:
        try:
            frame_duration = 1.0 / fps
            samples_per_frame = int(sample_rate * frame_duration)
            total_frames = len(audio_data) // samples_per_frame
            
            amplitude_frames = []
            
            for i in range(total_frames):
                start_idx = i * samples_per_frame
                end_idx = min(start_idx + samples_per_frame, len(audio_data))
                frame_data = audio_data[start_idx:end_idx]
                
                if len(frame_data) > 0:
                    rms = np.sqrt(np.mean(frame_data ** 2))
                    amplitude_frames.append(rms)
                else:
                    amplitude_frames.append(0.0)
            
            if len(amplitude_frames) > 0:
                max_amp = max(amplitude_frames) if max(amplitude_frames) > 0 else 1.0
                amplitude_frames = [amp / max_amp for amp in amplitude_frames]
            
            return amplitude_frames
            
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return [0.5] * 10

    def create_dotted_waveform(self, amplitude: float, width: int, height: int, 
                             size: int, spacing: int, max_height: int, 
                             dot_color: Tuple[int, int, int], background_color: Tuple[int, int, int]) -> Image.Image:
        
        bg_color = background_color
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        rgb_color = dot_color
        
        center_y = height // 2
        max_vis_height = (height * max_height) // 200
        
        dots_per_row = width // spacing
        
        current_height = int(amplitude * max_vis_height)
        
        for x in range(spacing // 2, width, spacing):
            dots_to_draw = current_height // spacing
            
            for i in range(dots_to_draw):
                dot_y = center_y - (i + 1) * spacing
                if dot_y >= 0:
                    self.draw_dot(draw, x, dot_y, size, rgb_color)
            
            for i in range(dots_to_draw):
                dot_y = center_y + (i + 1) * spacing
                if dot_y < height:
                    self.draw_dot(draw, x, dot_y, size, rgb_color)
        
        return img

    def draw_dot(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: Tuple[int, int, int]):
        half_size = size // 2
        draw.ellipse([
            x - half_size, y - half_size,
            x + half_size, y + half_size
        ], fill=color)

    def draw_dot_with_opacity_fast(self, img_array: np.ndarray, x: int, y: int, size: int, color: Tuple[int, int, int], opacity: float):
        if opacity <= 0:
            return
        
        height, width, _ = img_array.shape
        half_size = size // 2
        
        x_min = max(0, x - half_size)
        x_max = min(width, x + half_size + 1)
        y_min = max(0, y - half_size)
        y_max = min(height, y + half_size + 1)
        
        if x_min >= x_max or y_min >= y_max:
            return
        
        for py in range(y_min, y_max):
            for px in range(x_min, x_max):
                distance = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if distance <= half_size:
                    alpha = opacity
                    current_color = img_array[py, px]
                    new_color = [
                        int(current_color[i] * (1 - alpha) + color[i] * alpha)
                        for i in range(3)
                    ]
                    img_array[py, px] = new_color

    def get_discrete_opacity_color(self, base_color: Tuple[int, int, int], bg_color: Tuple[int, int, int], amplitude: float, levels: int = 5) -> Tuple[int, int, int]:
        
        if levels == 3:
            if amplitude <= 0.15:
                opacity = 0.0
            elif amplitude <= 0.65:
                opacity = 0.5
            else:
                opacity = 1.0
        elif levels == 5:
            if amplitude <= 0.05:
                opacity = 0.0
            elif amplitude <= 0.25:
                opacity = 0.25
            elif amplitude <= 0.5:
                opacity = 0.5
            elif amplitude <= 0.75:
                opacity = 0.75
            else:
                opacity = 1.0
        else:
            if amplitude <= 0.1:
                opacity = 0.1
            elif amplitude <= 0.2:
                opacity = 0.2
            elif amplitude <= 0.3:
                opacity = 0.3
            elif amplitude <= 0.4:
                opacity = 0.4
            elif amplitude <= 0.5:
                opacity = 0.5
            elif amplitude <= 0.6:
                opacity = 0.6
            elif amplitude <= 0.7:
                opacity = 0.7
            elif amplitude <= 0.8:
                opacity = 0.8
            elif amplitude <= 0.9:
                opacity = 0.9
            else:
                opacity = 1.0
        
        if levels == 10 and opacity < 0.1:
            opacity = 0.1
        
        if opacity <= 0.0:
            return bg_color
        else:
            return tuple(int(bg_color[i] * (1 - opacity) + base_color[i] * opacity) for i in range(3))

    def create_frame_with_opacity_batch(self, width: int, height: int, bg_color: Tuple[int, int, int], 
                                       dots: List[Tuple[int, int, int, Tuple[int, int, int], float]]) -> Image.Image:
        img_array = np.full((height, width, 3), bg_color, dtype=np.float32) / 255.0
        
        for x, y, size, color, opacity in dots:
            if opacity <= 0:
                continue
                
            half_size = size // 2
            color_float = [c / 255.0 for c in color]
            
            x_min = max(0, x - half_size)
            x_max = min(width, x + half_size + 1)
            y_min = max(0, y - half_size)
            y_max = min(height, y + half_size + 1)
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            py, px = np.mgrid[y_min:y_max, x_min:x_max]
            distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)
            mask = distance <= half_size
            
            if np.any(mask):
                for i in range(3):
                    img_array[py[mask], px[mask], i] = (
                        img_array[py[mask], px[mask], i] * (1 - opacity) + 
                        color_float[i] * opacity
                    )
        
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def draw_dot_with_opacity(self, img: Image.Image, x: int, y: int, size: int, color: Tuple[int, int, int], opacity: float):
        if opacity <= 0:
            return img
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        color_float = [c / 255.0 for c in color]
        
        self.draw_dot_with_opacity_fast(img_array, x, y, size, color_float, opacity)
        
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def generate_scrolling_animation(self, audio_np, sample_rate, width, height, size, spacing, 
                                   max_height, fps, dot_color, bg_color, window_size, opacity_mode, max_frames):
        frames = []
        audio_duration = len(audio_np) / sample_rate
        total_frames = int(audio_duration * fps)
        
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)
        
        print(f"Generating {total_frames} frames...")
        pbar = ProgressBar(total_frames)
        
        for frame_i in range(total_frames):
            if comfy.model_management.processing_interrupted():
                print("Animation generation cancelled by user")
                break
            pbar.update(1)
            current_time = frame_i / fps
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            
            center_y = height // 2
            max_vis_height = (height * max_height) // 200
            
            for x in range(spacing // 2, width, spacing):
                time_offset = (x / width) * window_size - window_size / 2
                sample_time = current_time + time_offset
                
                if 0 <= sample_time < audio_duration:
                    sample_idx = int(sample_time * sample_rate)
                    if sample_idx < len(audio_np):
                        amplitude = abs(audio_np[sample_idx])
                    else:
                        amplitude = 0.0
                else:
                    amplitude = 0.0
                
                if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                    base_opacity = 0.5
                    
                    import random
                    time_quantized = int(sample_time * 50) / 50.0
                    audio_time_seed = int(time_quantized * 1000)
                    random.seed(audio_time_seed)
                    is_random_peak = random.random() < 0.25
                    
                    if is_random_peak:
                        center_opacity = 1.0
                    elif amplitude > 0:
                        center_opacity = min(1.0, base_opacity + amplitude * 0.5)
                    else:
                        center_opacity = base_opacity
                    
                    levels = 5 if opacity_mode == "5_levels" else 10
                    center_color = self.get_discrete_opacity_color(dot_color, bg_color, center_opacity, levels)
                else:
                    center_color = dot_color
                
                self.draw_dot(draw, x, center_y, size, center_color)
                
                max_dots_per_column = int(max_vis_height / (spacing * 0.8))
                amplitude_dots_count = int(amplitude * max_dots_per_column)
                
                for i in range(amplitude_dots_count):
                    dot_offset = (i + 1) * (spacing * 0.8)
                    y_up = center_y - dot_offset
                    y_down = center_y + dot_offset
                    
                    import random
                    random.seed(int((sample_time + i * 0.1) * 1000))
                    is_bright_spot = random.random() < 0.18
                    
                    if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                        if is_bright_spot:
                            dot_opacity = 1.0
                        else:
                            distance_factor = 1.0 - (i / amplitude_dots_count) if amplitude_dots_count > 0 else 1.0
                            dot_opacity = max(0.3, distance_factor * 0.8)
                        
                        levels = 5 if opacity_mode == "5_levels" else 10
                        discrete_color = self.get_discrete_opacity_color(dot_color, bg_color, dot_opacity, levels)
                    else:
                        discrete_color = dot_color
                    
                    if y_up >= 0:
                        self.draw_dot(draw, x, y_up, size, discrete_color)
                    if y_down < height:
                        self.draw_dot(draw, x, y_down, size, discrete_color)
            
            frame_array = np.array(img).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame_array).unsqueeze(0))
        
        return frames

    def generate_breathing_animation(self, audio_np, sample_rate, width, height, size, spacing, 
                                   max_height, fps, dot_color, bg_color, opacity_mode, max_frames):
        amplitude_frames = self.analyze_audio(audio_np, sample_rate, fps)
        
        if max_frames > 0:
            amplitude_frames = amplitude_frames[:max_frames]
        
        frames = []
        total_frames = len(amplitude_frames)
        print(f"Generating {total_frames} breathing frames...")
        pbar = ProgressBar(total_frames)
        
        for i, amplitude in enumerate(amplitude_frames):
            if comfy.model_management.processing_interrupted():
                print("Breathing animation cancelled by user")
                break
            pbar.update(1)
            
            center_y = height // 2
            max_vis_height = (height * max_height) // 200
            
            if opacity_mode in ["5_levels", "10_levels"]:
                img = Image.new('RGB', (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                
                levels = 5 if opacity_mode == "5_levels" else 10
                final_dot_color = self.get_discrete_opacity_color(dot_color, bg_color, amplitude, levels)
                
                should_draw_frame = (opacity_mode == "10_levels") or (final_dot_color != bg_color)
                
                if should_draw_frame:
                    for x in range(spacing // 2, width, spacing):
                        current_height = int(amplitude * max_vis_height)
                        dots_count = current_height // spacing
                        
                        for j in range(dots_count):
                            y_up = center_y - (j + 1) * spacing
                            y_down = center_y + (j + 1) * spacing
                            
                            if y_up >= 0:
                                self.draw_dot(draw, x, y_up, size, final_dot_color)
                            if y_down < height:
                                self.draw_dot(draw, x, y_down, size, final_dot_color)
            else:
                img = Image.new('RGB', (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                
                for x in range(spacing // 2, width, spacing):
                    current_height = int(amplitude * max_vis_height)
                    dots_count = current_height // spacing
                    
                    for j in range(dots_count):
                        y_up = center_y - (j + 1) * spacing
                        y_down = center_y + (j + 1) * spacing
                        
                        if y_up >= 0:
                            self.draw_dot(draw, x, y_up, size, dot_color)
                        if y_down < height:
                            self.draw_dot(draw, x, y_down, size, dot_color)
            
            frame_array = np.array(img).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame_array).unsqueeze(0))
        
        return frames

    def generate_radial_animation(self, audio_np, sample_rate, width, height, size, spacing, 
                                max_height, fps, dot_color, bg_color, opacity_mode, max_frames):
        amplitude_frames = self.analyze_audio(audio_np, sample_rate, fps)
        
        if max_frames > 0:
            amplitude_frames = amplitude_frames[:max_frames]
        
        frames = []
        total_frames = len(amplitude_frames)
        print(f"Generating {total_frames} radial frames...")
        pbar = ProgressBar(total_frames)
        
        center_x, center_y = width // 2, height // 2
        max_possible_radius = min(width, height) // 2
        max_radius = int(max_possible_radius * (max_height / 100.0))
        
        for i, amplitude in enumerate(amplitude_frames):
            if comfy.model_management.processing_interrupted():
                print("Radial animation cancelled by user")
                break
            pbar.update(1)
            img = Image.new('RGB', (width, height), bg_color)
            
            ring_count = int(amplitude * max_radius / spacing)
            
            for ring in range(ring_count):
                radius = ring * spacing
                circumference = 2 * math.pi * radius
                dots_in_ring = max(1, int(circumference / spacing))
                
                if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                    ring_fade = (1.0 - ring / ring_count) if ring_count > 0 else 1.0
                    ring_opacity = amplitude * ring_fade
                    levels = 5 if opacity_mode == "5_levels" else 10
                    discrete_color = self.get_discrete_opacity_color(dot_color, bg_color, ring_opacity, levels)
                else:
                    discrete_color = dot_color
                
                should_draw_ring = (opacity_mode == "10_levels") or (discrete_color != bg_color) or (opacity_mode == "uniform")
                
                if should_draw_ring:
                    for i in range(dots_in_ring):
                        angle = (2 * math.pi * i) / dots_in_ring
                        x = int(center_x + radius * math.cos(angle))
                        y = int(center_y + radius * math.sin(angle))
                        
                        if 0 <= x < width and 0 <= y < height:
                            draw = ImageDraw.Draw(img)
                            self.draw_dot(draw, x, y, size, discrete_color)
            
            frame_array = np.array(img).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame_array).unsqueeze(0))
        
        return frames

    def generate_bars_animation(self, audio_np, sample_rate, width, height, size, spacing, 
                               max_height, fps, dot_color, bg_color, opacity_mode, max_frames):
        amplitude_frames = self.analyze_audio(audio_np, sample_rate, fps)
        
        if max_frames > 0:
            amplitude_frames = amplitude_frames[:max_frames]
        
        frames = []
        total_frames = len(amplitude_frames)
        print(f"Generating {total_frames} bars frames...")
        pbar = ProgressBar(total_frames)
        
        num_bars = width // spacing
        bar_heights = [0.0] * num_bars
        decay_factor = 0.85
        attack_factor = 0.6
        
        for frame_i, amplitude in enumerate(amplitude_frames):
            if comfy.model_management.processing_interrupted():
                print("Bars animation cancelled by user")
                break
            pbar.update(1)
            
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            
            center_x = width // 2
            center_y = height // 2
            max_vis_height = (height * max_height) // 200
            half_bars = num_bars // 2
            
            for i in range(num_bars):
                x = i * spacing + spacing // 2
                
                distance_from_center = abs(i - half_bars) / half_bars if half_bars > 0 else 0
                bell_curve = math.exp(-4 * distance_from_center ** 2)
                # Make max_height effect 2x stronger for bars animation
                height_boost = 1.0 + (max_height / 100.0)  # Additional height boost based on max_height setting
                target_amplitude = amplitude * bell_curve * height_boost
                
                if target_amplitude > bar_heights[i]:
                    bar_heights[i] = bar_heights[i] * attack_factor + target_amplitude * (1 - attack_factor)
                else:
                    bar_heights[i] *= decay_factor
                
                bar_height = int(bar_heights[i] * max_vis_height)
                
                if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                    levels = 3 if opacity_mode == "3_levels" else (5 if opacity_mode == "5_levels" else 10)
                    bar_color = self.get_discrete_opacity_color(dot_color, bg_color, bar_heights[i], levels)
                else:
                    bar_color = dot_color
                
                y_start = center_y - bar_height // 2
                y_end = center_y + bar_height // 2
                
                if bar_height > 0:
                    draw.rectangle([x - size//2, y_start, x + size//2, y_end], fill=bar_color)
            
            frame_array = np.array(img).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame_array).unsqueeze(0))
        
        return frames


    def generate_wave_animation(self, audio_np, sample_rate, width, height, size, spacing, 
                              dot_color, bg_color, max_height, fps, max_frames, opacity_mode, **kwargs):
        frames = []
        max_vis_height = (height * max_height) // 200
        
        frame_duration = len(audio_np) / sample_rate
        frame_count = min(int(frame_duration * fps), max_frames) if max_frames > 0 else int(frame_duration * fps)
        
        num_waves = 3
        wave_colors = [dot_color, dot_color, dot_color]
        
        for frame_idx in range(frame_count):
            time_offset = (frame_idx / fps) if fps > 0 else 0
            sample_idx = int(time_offset * sample_rate)
            
            if sample_idx < len(audio_np):
                amplitude = abs(audio_np[sample_idx])
            else:
                amplitude = 0
            
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            
            for wave_idx in range(num_waves):
                wave_offset = wave_idx * math.pi * 0.5
                wave_amplitude = amplitude * (0.8 - wave_idx * 0.2)
                
                for x in range(0, width, spacing):
                    wave_phase = (x / width) * 4 * math.pi + time_offset * 3 + wave_offset
                    y = height // 2 + math.sin(wave_phase) * max_vis_height * wave_amplitude
                    
                    if 0 <= y < height:
                        local_amplitude = wave_amplitude * (0.5 + 0.5 * math.sin(wave_phase))
                        
                        if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                            levels = 3 if opacity_mode == "3_levels" else (5 if opacity_mode == "5_levels" else 10)
                            point_color = self.get_discrete_opacity_color(wave_colors[wave_idx], bg_color, local_amplitude, levels)
                        else:
                            point_color = wave_colors[wave_idx]
                        
                        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=point_color)
            
            frame_array = np.array(img).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame_array).unsqueeze(0))
        
        return frames


    def generate_preview(self, width, height, size, spacing, 
                        dot_color, bg_color, animation_style, max_height, opacity_mode, window_size):
        
        preview_img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(preview_img)
        
        center_y = height // 2
        max_vis_height = (height * max_height) // 200
        
        if animation_style == "scrolling":
            for x in range(spacing // 2, width, spacing):
                normalized_x = x / width
                fake_amplitude = 0.3 + 0.7 * abs(math.sin(normalized_x * math.pi * 3))
                
                if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                    import random
                    random.seed(x)
                    is_random_peak = random.random() < 0.15
                    
                    if is_random_peak:
                        center_opacity = 1.0
                    else:
                        center_opacity = 0.5 + fake_amplitude * 0.3
                    
                    levels = 5 if opacity_mode == "5_levels" else 10
                    center_color = self.get_discrete_opacity_color(dot_color, bg_color, center_opacity, levels)
                else:
                    center_color = dot_color
                
                self.draw_dot(draw, x, center_y, size, center_color)
                
                max_dots_per_column = max_vis_height // spacing
                for i in range(max_dots_per_column):
                    y_up = center_y - (i + 1) * spacing
                    y_down = center_y + (i + 1) * spacing
                    
                    threshold = (i + 1) / max_dots_per_column
                    if fake_amplitude >= threshold:
                        if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                            excess = (fake_amplitude - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
                            dot_opacity = max(0.1, excess)
                            levels = 3 if opacity_mode == "3_levels" else (5 if opacity_mode == "5_levels" else 10)
                            discrete_color = self.get_discrete_opacity_color(dot_color, bg_color, dot_opacity, levels)
                        else:
                            discrete_color = dot_color
                        
                        if y_up >= 0:
                            self.draw_dot(draw, x, y_up, size, discrete_color)
                        if y_down < height:
                            self.draw_dot(draw, x, y_down, size, discrete_color)
                            
        elif animation_style == "breathing":
            fake_amplitude = 0.6
            
            if opacity_mode in ["5_levels", "10_levels"]:
                levels = 5 if opacity_mode == "5_levels" else 10
                preview_color = self.get_discrete_opacity_color(dot_color, bg_color, fake_amplitude, levels)
            else:
                preview_color = dot_color
            
            for x in range(spacing // 2, width, spacing):
                current_height = int(fake_amplitude * max_vis_height)
                dots_count = current_height // spacing
                
                for j in range(dots_count):
                    y_up = center_y - (j + 1) * spacing
                    y_down = center_y + (j + 1) * spacing
                    
                    if y_up >= 0:
                        self.draw_dot(draw, x, y_up, size, preview_color)
                    if y_down < height:
                        self.draw_dot(draw, x, y_down, size, preview_color)
        
        elif animation_style == "bars":
            fake_amplitude = 0.7
            center_x = width // 2
            num_bars = width // spacing
            half_bars = num_bars // 2
            
            for i in range(num_bars):
                x = i * spacing + spacing // 2
                distance_from_center = abs(i - half_bars) / half_bars if half_bars > 0 else 0
                bell_curve = math.exp(-4 * distance_from_center ** 2)
                # Apply same 2x height boost to preview
                height_boost = 1.0 + (max_height / 100.0)
                bar_amplitude = fake_amplitude * bell_curve * height_boost
                
                bar_height = int(bar_amplitude * max_vis_height)
                
                if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                    levels = 3 if opacity_mode == "3_levels" else (5 if opacity_mode == "5_levels" else 10)
                    bar_color = self.get_discrete_opacity_color(dot_color, bg_color, bar_amplitude, levels)
                else:
                    bar_color = dot_color
                
                y_start = center_y - bar_height // 2
                y_end = center_y + bar_height // 2
                
                draw.rectangle([x - size//2, y_start, x + size//2, y_end], fill=bar_color)
        
        elif animation_style == "wave":
            fake_amplitude = 0.6
            max_vis_height = (height * max_height) // 200
            
            num_waves = 3
            wave_colors = [dot_color, dot_color, dot_color]
            
            for wave_idx in range(num_waves):
                wave_offset = wave_idx * math.pi * 0.5
                wave_amplitude = fake_amplitude * (0.8 - wave_idx * 0.2)
                
                for x in range(0, width, spacing):
                    wave_phase = (x / width) * 4 * math.pi + wave_offset
                    y = height // 2 + math.sin(wave_phase) * max_vis_height * wave_amplitude
                    
                    if 0 <= y < height:
                        local_amplitude = wave_amplitude * (0.5 + 0.5 * math.sin(wave_phase))
                        
                        if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                            levels = 3 if opacity_mode == "3_levels" else (5 if opacity_mode == "5_levels" else 10)
                            point_color = self.get_discrete_opacity_color(wave_colors[wave_idx], bg_color, local_amplitude, levels)
                        else:
                            point_color = wave_colors[wave_idx]
                        
                        draw.ellipse([x-size//2, y-size//2, x+size//2, y+size//2], fill=point_color)
                        
        else:
            center_x = width // 2
            max_possible_radius = min(width, height) // 2
            max_radius = int(max_possible_radius * (max_height / 100.0))
            fake_amplitude = 0.7
            
            ring_count = int(fake_amplitude * max_radius / spacing)
            
            for ring in range(ring_count):
                radius = ring * spacing
                circumference = 2 * math.pi * radius
                dots_in_ring = max(1, int(circumference / spacing))
                
                if opacity_mode in ["3_levels", "5_levels", "10_levels"]:
                    ring_fade = (1.0 - ring / ring_count) if ring_count > 0 else 1.0
                    ring_opacity = fake_amplitude * ring_fade
                    levels = 5 if opacity_mode == "5_levels" else 10
                    discrete_color = self.get_discrete_opacity_color(dot_color, bg_color, ring_opacity, levels)
                else:
                    discrete_color = dot_color
                
                for i in range(dots_in_ring):
                    angle = (2 * math.pi * i) / dots_in_ring
                    x = int(center_x + radius * math.cos(angle))
                    y = int(center_y + radius * math.sin(angle))
                    
                    if 0 <= x < width and 0 <= y < height:
                        self.draw_dot(draw, x, y, size, discrete_color)
        
        preview_array = np.array(preview_img).astype(np.float32) / 255.0
        return torch.from_numpy(preview_array).unsqueeze(0)


    def generate_waveform(self, audio, width, height, size, spacing, 
                         dot_color, background_color, animation_style, max_height, fps, max_frames, opacity_mode, **kwargs):
        
        try:
            # CRITICAL: Store the EXACT original audio object - don't modify it at all
            # This preserves the tensor references, device placement, and ComfyUI format
            original_audio_passthrough = audio
            
            window_size = kwargs.get('window_size', 2.0)
            preview_mode = kwargs.get('preview_mode', False)
            
            if window_size <= 0.0:
                window_size = 2.0
            elif window_size < 0.5:
                window_size = 0.5
            
            final_color = safe_hex_to_rgb(dot_color, (0, 255, 255))
            final_bg_color = safe_hex_to_rgb(background_color, (0, 0, 0))
            
            if preview_mode:
                preview_image = self.generate_preview(width, height, size, spacing, 
                                                    final_color, final_bg_color, animation_style, 
                                                    max_height, opacity_mode, window_size)
                return (preview_image, original_audio_passthrough, float(fps))
            
            audio_np, sample_rate = self.load_audio_with_fallback(audio)
            
            if animation_style == "scrolling":
                frames = self.generate_scrolling_animation(audio_np, sample_rate, width, height, 
                                                         size, spacing, max_height, fps,
                                                         final_color, final_bg_color, window_size, opacity_mode, max_frames)
            elif animation_style == "breathing":
                frames = self.generate_breathing_animation(audio_np, sample_rate, width, height,
                                                         size, spacing, max_height, fps,
                                                         final_color, final_bg_color, opacity_mode, max_frames)
            elif animation_style == "bars":
                frames = self.generate_bars_animation(audio_np, sample_rate, width, height,
                                                    size, spacing, max_height, fps,
                                                    final_color, final_bg_color, opacity_mode, max_frames)
            elif animation_style == "wave":
                frames = self.generate_wave_animation(audio_np, sample_rate, width, height,
                                                    size, spacing, final_color, final_bg_color, 
                                                    max_height, fps, max_frames, opacity_mode)
            else:
                # Default to radial for backwards compatibility
                frames = self.generate_radial_animation(audio_np, sample_rate, width, height,
                                                      size, spacing, max_height, fps,
                                                      final_color, final_bg_color, opacity_mode, max_frames)
            
            if frames:
                output_images = torch.cat(frames, dim=0)
                print(f"✅ Generated {len(frames)} waveform frames successfully!")
            else:
                fallback_img = Image.new('RGB', (width, height), final_bg_color)
                fallback_array = np.array(fallback_img).astype(np.float32) / 255.0
                output_images = torch.from_numpy(fallback_array).unsqueeze(0)
                print("⚠️ No frames generated, returning fallback")
            
            # Fix audio format for VideoHelperSuite compatibility
            if isinstance(original_audio_passthrough, dict) and 'waveform' in original_audio_passthrough:
                wf = original_audio_passthrough['waveform']
                print(f"🔊 Original audio - Shape: {wf.shape}, Device: {wf.device}, Dtype: {wf.dtype}")
                
                # Convert to format compatible with VideoHelperSuite
                fixed_waveform = wf.float()  # Convert to float32
                
                # Convert mono to stereo if needed (VideoHelperSuite expects stereo)
                if fixed_waveform.shape[1] == 1:  # If mono (1 channel)
                    fixed_waveform = fixed_waveform.repeat(1, 2, 1)  # Duplicate to make stereo
                    print(f"🔊 Converted mono to stereo")
                
                audio_output = {
                    'waveform': fixed_waveform,
                    'sample_rate': original_audio_passthrough['sample_rate']
                }
                print(f"🔊 Fixed audio - Shape: {fixed_waveform.shape}, Dtype: {fixed_waveform.dtype}")
            else:
                audio_output = original_audio_passthrough
            
            return (output_images, audio_output, float(fps))
            
        except Exception as e:
            print(f"Waveform generation error: {e}")
            fallback_bg_color = safe_hex_to_rgb(background_color, (0, 0, 0))
            fallback_img = Image.new('RGB', (width, height), fallback_bg_color)
            fallback_array = np.array(fallback_img).astype(np.float32) / 255.0
            fallback_tensor = torch.from_numpy(fallback_array).unsqueeze(0)
            # Use the original audio parameter even in error case
            return (fallback_tensor, audio, float(fps))

NODE_CLASS_MAPPINGS = {
    "DottedWaveformVisualizer": DottedWaveformVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DottedWaveformVisualizer": "Dotted Waveform Visualizer"
}
