"""
Hardware Detection and Optimization Module
Automatically detects system capabilities and optimizes settings for best performance
"""

import os
import sys
import platform
import subprocess
import psutil
import torch
import json
import logging
from typing import Dict, Tuple, Optional
import GPUtil
import cpuinfo

class HardwareOptimizer:
    """
    Detects hardware capabilities and provides optimized settings
    Works on both high-end and low-end PCs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hardware_info = self._detect_hardware()
        self.optimization_profile = self._determine_profile()
        
    def _detect_hardware(self) -> Dict:
        """Comprehensive hardware detection"""
        info = {
            'cpu': self._get_cpu_info(),
            'gpu': self._get_gpu_info(),
            'memory': self._get_memory_info(),
            'os': platform.system(),
            'ffmpeg': self._detect_ffmpeg_capabilities()
        }
        return info
    
    def _get_cpu_info(self) -> Dict:
        """Get detailed CPU information"""
        try:
            cpu_info_data = cpuinfo.get_cpu_info()
            return {
                'name': cpu_info_data.get('brand_raw', 'Unknown'),
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'arch': platform.machine(),
                'has_avx': 'avx' in cpu_info_data.get('flags', []),
                'has_avx2': 'avx2' in cpu_info_data.get('flags', [])
            }
        except:
            return {
                'cores': psutil.cpu_count(logical=False) or 2,
                'threads': psutil.cpu_count(logical=True) or 4,
                'frequency': 2000  # Default 2GHz
            }
    
    def _get_gpu_info(self) -> Dict:
        """Detect GPU capabilities"""
        gpu_info = {
            'available': False,
            'type': 'none',
            'name': 'None',
            'memory': 0,
            'compute_capability': None
        }
        
        # Check for NVIDIA GPU
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info.update({
                        'available': True,
                        'type': 'nvidia',
                        'name': gpu.name,
                        'memory': gpu.memoryTotal / 1024,  # Convert MB to GB
                        'compute_capability': torch.cuda.get_device_capability(0)
                    })
            except:
                gpu_info.update({
                    'available': True,
                    'type': 'nvidia',
                    'name': torch.cuda.get_device_name(0),
                    'memory': torch.cuda.get_device_properties(0).total_memory / (1024**3)
                })
        
        # Check for Apple Silicon
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info.update({
                'available': True,
                'type': 'apple',
                'name': 'Apple Silicon GPU',
                'memory': psutil.virtual_memory().total / (1024**3) * 0.7  # Estimate
            })
        
        # Check for AMD GPU (ROCm)
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            gpu_info.update({
                'available': True,
                'type': 'amd',
                'name': 'AMD GPU',
                'memory': 4  # Default estimate
            })
        
        # Check for Intel GPU
        try:
            result = subprocess.run(['clinfo'], capture_output=True, text=True)
            if 'Intel' in result.stdout:
                gpu_info.update({
                    'available': True,
                    'type': 'intel',
                    'name': 'Intel Integrated Graphics'
                })
        except:
            pass
        
        return gpu_info
    
    def _get_memory_info(self) -> Dict:
        """Get system memory information"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'percent_used': mem.percent
        }
    
    def _detect_ffmpeg_capabilities(self) -> Dict:
        """Detect available FFmpeg hardware encoders"""
        encoders = {
            'h264': [],
            'hevc': [],
            'vp9': []
        }
        
        try:
            # Get list of available encoders
            result = subprocess.run(['ffmpeg', '-encoders'], 
                                  capture_output=True, text=True)
            output = result.stdout
            
            # NVIDIA encoders
            if 'h264_nvenc' in output:
                encoders['h264'].append('h264_nvenc')
            if 'hevc_nvenc' in output:
                encoders['hevc'].append('hevc_nvenc')
            
            # Intel QuickSync
            if 'h264_qsv' in output:
                encoders['h264'].append('h264_qsv')
            if 'hevc_qsv' in output:
                encoders['hevc'].append('hevc_qsv')
            if 'vp9_qsv' in output:
                encoders['vp9'].append('vp9_qsv')
            
            # AMD AMF
            if 'h264_amf' in output:
                encoders['h264'].append('h264_amf')
            if 'hevc_amf' in output:
                encoders['hevc'].append('hevc_amf')
            
            # Apple VideoToolbox
            if 'h264_videotoolbox' in output:
                encoders['h264'].append('h264_videotoolbox')
            if 'hevc_videotoolbox' in output:
                encoders['hevc'].append('hevc_videotoolbox')
            
            # Check for VAAPI (Linux)
            if 'h264_vaapi' in output:
                encoders['h264'].append('h264_vaapi')
            if 'hevc_vaapi' in output:
                encoders['hevc'].append('hevc_vaapi')
            if 'vp9_vaapi' in output:
                encoders['vp9'].append('vp9_vaapi')
                
        except Exception as e:
            self.logger.warning(f"Could not detect FFmpeg capabilities: {e}")
        
        return encoders
    
    def _determine_profile(self) -> str:
        """
        Determine optimization profile based on hardware
        Returns: 'ultra', 'high', 'medium', 'low', 'potato'
        """
        score = 0
        
        # CPU scoring (0-40 points)
        cpu_cores = self.hardware_info['cpu'].get('cores', 2)
        if cpu_cores >= 16:
            score += 40
        elif cpu_cores >= 8:
            score += 30
        elif cpu_cores >= 4:
            score += 20
        elif cpu_cores >= 2:
            score += 10
        else:
            score += 5
        
        # GPU scoring (0-40 points)
        gpu = self.hardware_info['gpu']
        if gpu['available']:
            if gpu['type'] == 'nvidia' and gpu.get('memory', 0) >= 8:
                score += 40
            elif gpu['type'] == 'nvidia' and gpu.get('memory', 0) >= 4:
                score += 30
            elif gpu['type'] in ['apple', 'amd']:
                score += 25
            elif gpu['type'] == 'intel':
                score += 15
            else:
                score += 20
        
        # Memory scoring (0-20 points)
        total_mem = self.hardware_info['memory']['total_gb']
        if total_mem >= 32:
            score += 20
        elif total_mem >= 16:
            score += 15
        elif total_mem >= 8:
            score += 10
        elif total_mem >= 4:
            score += 5
        else:
            score += 2
        
        # Determine profile based on score
        if score >= 80:
            return 'ultra'
        elif score >= 60:
            return 'high'
        elif score >= 40:
            return 'medium'
        elif score >= 20:
            return 'low'
        else:
            return 'potato'  # For very low-end systems
    
    def get_optimized_settings(self) -> Dict:
        """
        Get optimized settings based on hardware profile
        """
        profile = self.optimization_profile
        
        profiles = {
            'ultra': {
                'device': self._get_best_device(),
                'max_workers': min(8, self.hardware_info['cpu']['threads'] - 2),
                'batch_size': 8,
                'frame_buffer_size': 120,  # Process 120 frames at once
                'torch_threads': max(4, self.hardware_info['cpu']['threads'] // 2),
                'use_mixed_precision': True,
                'use_model_quantization': False,
                'model_preference': 'quality',  # Use full model
                'video_cache_frames': True,
                'max_resolution': None,  # No limit
                'encoder_preset': 'fast',
                'encoder_quality': 'high'
            },
            'high': {
                'device': self._get_best_device(),
                'max_workers': min(6, self.hardware_info['cpu']['threads'] - 2),
                'batch_size': 4,
                'frame_buffer_size': 60,
                'torch_threads': max(2, self.hardware_info['cpu']['threads'] // 3),
                'use_mixed_precision': True,
                'use_model_quantization': False,
                'model_preference': 'balanced',
                'video_cache_frames': True,
                'max_resolution': (3840, 2160),  # 4K max
                'encoder_preset': 'medium',
                'encoder_quality': 'high'
            },
            'medium': {
                'device': self._get_best_device(),
                'max_workers': min(4, self.hardware_info['cpu']['threads'] - 1),
                'batch_size': 2,
                'frame_buffer_size': 30,
                'torch_threads': max(2, self.hardware_info['cpu']['threads'] // 4),
                'use_mixed_precision': self.hardware_info['gpu']['available'],
                'use_model_quantization': True,
                'model_preference': 'fast',  # Prefer lite model
                'video_cache_frames': False,
                'max_resolution': (1920, 1080),  # 1080p max
                'encoder_preset': 'fast',
                'encoder_quality': 'medium'
            },
            'low': {
                'device': 'cpu',  # Force CPU for stability
                'max_workers': 2,
                'batch_size': 1,
                'frame_buffer_size': 15,
                'torch_threads': 2,
                'use_mixed_precision': False,
                'use_model_quantization': True,
                'model_preference': 'fast',
                'video_cache_frames': False,
                'max_resolution': (1280, 720),  # 720p max
                'encoder_preset': 'ultrafast',
                'encoder_quality': 'low'
            },
            'potato': {
                'device': 'cpu',
                'max_workers': 1,
                'batch_size': 1,
                'frame_buffer_size': 5,
                'torch_threads': 1,
                'use_mixed_precision': False,
                'use_model_quantization': True,
                'model_preference': 'ultra_fast',  # Most aggressive optimizations
                'video_cache_frames': False,
                'max_resolution': (854, 480),  # 480p max
                'encoder_preset': 'ultrafast',
                'encoder_quality': 'lowest'
            }
        }
        
        settings = profiles[profile].copy()
        settings['profile'] = profile
        settings['hardware_info'] = self.hardware_info
        
        # Add encoder recommendations
        settings['recommended_encoder'] = self._get_recommended_encoder()
        
        return settings
    
    def _get_best_device(self) -> str:
        """Determine the best processing device"""
        gpu = self.hardware_info['gpu']
        
        if gpu['available']:
            if gpu['type'] == 'nvidia':
                return 'cuda'
            elif gpu['type'] == 'apple':
                return 'mps'
            elif gpu['type'] == 'amd':
                return 'cuda'  # ROCm uses same interface
        
        return 'cpu'
    
    def _get_recommended_encoder(self) -> Dict[str, str]:
        """Get recommended video encoder based on available hardware"""
        encoders = self.hardware_info['ffmpeg']
        recommendations = {}
        
        # MP4/H264
        if encoders['h264']:
            # Prefer NVIDIA > Intel > AMD > Apple > VAAPI
            if 'h264_nvenc' in encoders['h264']:
                recommendations['h264'] = 'h264_nvenc'
            elif 'h264_qsv' in encoders['h264']:
                recommendations['h264'] = 'h264_qsv'
            elif 'h264_amf' in encoders['h264']:
                recommendations['h264'] = 'h264_amf'
            elif 'h264_videotoolbox' in encoders['h264']:
                recommendations['h264'] = 'h264_videotoolbox'
            elif 'h264_vaapi' in encoders['h264']:
                recommendations['h264'] = 'h264_vaapi'
            else:
                recommendations['h264'] = encoders['h264'][0]
        else:
            recommendations['h264'] = 'libx264'  # Software fallback
        
        # HEVC
        if encoders['hevc']:
            if 'hevc_nvenc' in encoders['hevc']:
                recommendations['hevc'] = 'hevc_nvenc'
            elif 'hevc_qsv' in encoders['hevc']:
                recommendations['hevc'] = 'hevc_qsv'
            else:
                recommendations['hevc'] = encoders['hevc'][0]
        else:
            recommendations['hevc'] = 'libx265'
        
        # VP9
        if encoders['vp9']:
            recommendations['vp9'] = encoders['vp9'][0]
        else:
            recommendations['vp9'] = 'libvpx-vp9'
        
        return recommendations
    
    def get_adaptive_batch_size(self, available_memory_gb: float) -> int:
        """
        Calculate optimal batch size based on available memory
        Used for dynamic adjustment during processing
        """
        if self.hardware_info['gpu']['available']:
            # GPU memory-based calculation
            gpu_mem = self.hardware_info['gpu'].get('memory', 4)
            if gpu_mem >= 8:
                return 8
            elif gpu_mem >= 4:
                return 4
            else:
                return 2
        else:
            # CPU memory-based calculation
            if available_memory_gb >= 8:
                return 4
            elif available_memory_gb >= 4:
                return 2
            else:
                return 1
    
    def should_downsample_input(self, input_resolution: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Determine if input should be downsampled based on hardware capabilities
        Returns target resolution or None if no downsampling needed
        """
        settings = self.get_optimized_settings()
        max_res = settings.get('max_resolution')
        
        if max_res and (input_resolution[0] > max_res[0] or input_resolution[1] > max_res[1]):
            # Calculate scaling factor
            scale = min(max_res[0] / input_resolution[0], max_res[1] / input_resolution[1])
            new_width = int(input_resolution[0] * scale)
            new_height = int(input_resolution[1] * scale)
            # Make dimensions even for video encoding
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            return (new_width, new_height)
        
        return None
    
    def print_optimization_report(self):
        """Print a human-readable optimization report"""
        print("\n" + "="*60)
        print("HARDWARE DETECTION & OPTIMIZATION REPORT")
        print("="*60)
        
        # CPU Info
        cpu = self.hardware_info['cpu']
        print(f"\nCPU:")
        print(f"  • Cores: {cpu.get('cores', 'Unknown')} physical, {cpu.get('threads', 'Unknown')} logical")
        print(f"  • Frequency: {cpu.get('frequency', 0)/1000:.1f} GHz")
        if cpu.get('name'):
            print(f"  • Model: {cpu['name']}")
        
        # GPU Info
        gpu = self.hardware_info['gpu']
        print(f"\nGPU:")
        if gpu['available']:
            print(f"  • Type: {gpu['type'].upper()}")
            print(f"  • Name: {gpu['name']}")
            if gpu.get('memory'):
                print(f"  • Memory: {gpu['memory']:.1f} GB")
        else:
            print("  • No compatible GPU detected")
        
        # Memory Info
        mem = self.hardware_info['memory']
        print(f"\nMemory:")
        print(f"  • Total: {mem['total_gb']:.1f} GB")
        print(f"  • Available: {mem['available_gb']:.1f} GB")
        
        # FFmpeg Encoders
        ffmpeg = self.hardware_info['ffmpeg']
        print(f"\nHardware Encoders:")
        if any(ffmpeg.values()):
            for format_type, encoders in ffmpeg.items():
                if encoders:
                    print(f"  • {format_type.upper()}: {', '.join(encoders)}")
        else:
            print("  • No hardware encoders detected (using software encoding)")
        
        # Optimization Profile
        print(f"\nOptimization Profile: {self.optimization_profile.upper()}")
        
        settings = self.get_optimized_settings()
        print(f"\nRecommended Settings:")
        print(f"  • Processing Device: {settings['device'].upper()}")
        print(f"  • Worker Threads: {settings['max_workers']}")
        print(f"  • Batch Size: {settings['batch_size']}")
        print(f"  • Model: {'Full quality' if settings['model_preference'] == 'quality' else 'Fast/Lite'}")
        if settings.get('max_resolution'):
            print(f"  • Max Resolution: {settings['max_resolution'][0]}x{settings['max_resolution'][1]}")
        print(f"  • Encoder Quality: {settings['encoder_quality']}")
        
        print("\n" + "="*60 + "\n")


# Utility function for easy integration
def get_hardware_optimizer():
    """Get or create singleton hardware optimizer instance"""
    if not hasattr(get_hardware_optimizer, '_instance'):
        get_hardware_optimizer._instance = HardwareOptimizer()
    return get_hardware_optimizer._instance


# Test the module
if __name__ == "__main__":
    optimizer = HardwareOptimizer()
    optimizer.print_optimization_report()
    
    # Save settings to file for debugging
    settings = optimizer.get_optimized_settings()
    with open('hardware_settings.json', 'w') as f:
        # Convert non-serializable items
        clean_settings = {}
        for k, v in settings.items():
            if k != 'hardware_info':
                clean_settings[k] = v
        json.dump(clean_settings, f, indent=2)
    print(f"Settings saved to hardware_settings.json")
