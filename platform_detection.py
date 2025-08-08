#!/usr/bin/env python3
"""
DARG Platform Detection Module
Detects and configures GPU acceleration (CUDA/MPS) and system resources
"""

import os
import sys
import platform
import subprocess
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PlatformConfig:
    """Platform configuration for DARG"""
    system: str
    gpu_available: bool
    gpu_type: str  # 'cuda', 'mps', 'none'
    gpu_devices: List[str]
    cpu_cores: int
    memory_gb: float
    acceleration_available: bool
    recommended_threads: int
    recommended_batch_size: int

class PlatformDetector:
    """Platform detection and configuration for DARG"""
    
    def __init__(self):
        self.config = None
        self._detect_platform()
    
    def _detect_platform(self) -> None:
        """Detect platform capabilities"""
        system = platform.system()
        cpu_cores = os.cpu_count() or 4
        memory_gb = self._get_memory_size()
        
        # GPU Detection
        gpu_available, gpu_type, gpu_devices = self._detect_gpu()
        
        # Calculate recommendations
        recommended_threads = min(cpu_cores, 16)  # Cap at 16 for most workloads
        recommended_batch_size = self._calculate_batch_size(memory_gb, gpu_available)
        
        self.config = PlatformConfig(
            system=system,
            gpu_available=gpu_available,
            gpu_type=gpu_type,
            gpu_devices=gpu_devices,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            acceleration_available=gpu_available,
            recommended_threads=recommended_threads,
            recommended_batch_size=recommended_batch_size
        )
        
        logger.info(f"Platform detected: {system} with {cpu_cores} cores, {memory_gb:.1f}GB RAM")
        if gpu_available:
            logger.info(f"GPU acceleration: {gpu_type} with {len(gpu_devices)} device(s)")
        else:
            logger.info("No GPU acceleration available")
    
    def _detect_gpu(self) -> Tuple[bool, str, List[str]]:
        """Detect GPU capabilities"""
        gpu_devices = []
        
        # Check for NVIDIA CUDA
        if self._check_cuda():
            try:
                # Try to get CUDA device info
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_devices = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                    return True, 'cuda', gpu_devices
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        # Check for Apple MPS (Metal Performance Shaders)
        if self._check_mps():
            try:
                # Check if MPS is available
                import subprocess
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and 'Metal' in result.stdout:
                    gpu_devices = ['Apple GPU (MPS)']
                    return True, 'mps', gpu_devices
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                pass
        
        return False, 'none', []
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            # Check nvidia-smi
            subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _check_mps(self) -> bool:
        """Check if Apple MPS is available"""
        return platform.system() == 'Darwin' and platform.machine() in ['arm64', 'x86_64']
    
    def _get_memory_size(self) -> float:
        """Get system memory size in GB"""
        try:
            if platform.system() == 'Darwin':  # macOS
                result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                if result.returncode == 0:
                    memory_bytes = int(result.stdout.split(':')[1].strip())
                    return memory_bytes / (1024**3)
            elif platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            memory_kb = int(line.split()[1])
                            return memory_kb / (1024**2)
            elif platform.system() == 'Windows':
                import psutil
                return psutil.virtual_memory().total / (1024**3)
        except Exception as e:
            logger.warning(f"Could not detect memory size: {e}")
        
        return 8.0  # Default assumption
    
    def _calculate_batch_size(self, memory_gb: float, gpu_available: bool) -> int:
        """Calculate recommended batch size based on available memory"""
        if gpu_available:
            # GPU memory is typically more limited
            base_batch = min(10000, int(memory_gb * 1000))
        else:
            # CPU memory is more abundant
            base_batch = min(50000, int(memory_gb * 2000))
        
        return max(1000, base_batch)
    
    def get_acceleration_config(self) -> Dict[str, any]:
        """Get acceleration configuration"""
        if not self.config:
            return {'enabled': False, 'type': 'none'}
        
        config = {
            'enabled': self.config.gpu_available,
            'type': self.config.gpu_type,
            'devices': self.config.gpu_devices,
            'threads': self.config.recommended_threads,
            'batch_size': self.config.recommended_batch_size
        }
        
        return config
    
    def setup_environment(self) -> None:
        """Setup environment variables for acceleration"""
        if not self.config:
            return
        
        if self.config.gpu_type == 'cuda':
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(len(self.config.gpu_devices))])
            # Set CUDA memory growth to avoid OOM
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
        elif self.config.gpu_type == 'mps':
            # Enable MPS if available
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Set OpenMP threads for CPU operations
        os.environ['OMP_NUM_THREADS'] = str(self.config.recommended_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.config.recommended_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.recommended_threads)
    
    def print_platform_info(self) -> None:
        """Print detailed platform information"""
        if not self.config:
            print("Platform detection not completed")
            return
        
        print("=" * 60)
        print("DARG PLATFORM CONFIGURATION")
        print("=" * 60)
        print(f"System: {self.config.system}")
        print(f"CPU Cores: {self.config.cpu_cores}")
        print(f"Memory: {self.config.memory_gb:.1f} GB")
        print(f"Recommended Threads: {self.config.recommended_threads}")
        print(f"Recommended Batch Size: {self.config.recommended_batch_size:,}")
        
        if self.config.gpu_available:
            print(f"GPU Acceleration: {self.config.gpu_type.upper()}")
            print(f"GPU Devices: {len(self.config.gpu_devices)}")
            for i, device in enumerate(self.config.gpu_devices):
                print(f"  Device {i}: {device}")
        else:
            print("GPU Acceleration: Not Available")
        
        print("=" * 60)

# Global platform detector instance
platform_detector = PlatformDetector()

def get_platform_config() -> PlatformConfig:
    """Get platform configuration"""
    return platform_detector.config

def setup_acceleration() -> Dict[str, any]:
    """Setup acceleration and return configuration"""
    platform_detector.setup_environment()
    return platform_detector.get_acceleration_config()

def print_platform_info() -> None:
    """Print platform information"""
    platform_detector.print_platform_info()

if __name__ == "__main__":
    print_platform_info()
    config = setup_acceleration()
    print(f"\nAcceleration Config: {config}")
