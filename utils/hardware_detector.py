"""
Detector de hardware para determinar el tipo de sistema
"""

import os
import platform
import psutil
from pathlib import Path
from typing import Dict, Any

class HardwareDetector:
    """Detector de tipo de hardware y capacidades del sistema"""
    
    def __init__(self):
        self.system_info = self._gather_system_info()
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Recopila información del sistema"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': platform.python_version(),
        }
        
        # Detectar si es Raspberry Pi
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                info['is_raspberry_pi'] = 'raspberry pi' in cpuinfo.lower()
        except:
            info['is_raspberry_pi'] = False
        
        # Detectar si es Jetson
        info['is_jetson'] = os.path.exists('/etc/nv_tegra_release')
        
        # Detectar GPU
        info['has_gpu'] = self._detect_gpu()
        
        return info
    
    def _detect_gpu(self) -> bool:
        """Detecta si hay GPU disponible"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        # Verificar archivos del sistema para GPU
        gpu_paths = [
            '/dev/nvidia0',
            '/proc/driver/nvidia',
            '/sys/class/drm'
        ]
        
        for path in gpu_paths:
            if os.path.exists(path):
                return True
        
        return False
    
    def get_hardware_type(self) -> str:
        """
        Determina el tipo de hardware
        
        Returns:
            'embedded' para dispositivos embebidos, 'desktop' para PCs
        """
        if self.system_info['is_raspberry_pi'] or self.system_info['is_jetson']:
            return 'embedded'
        
        # Criterios para considerar como embedded
        if (self.system_info['cpu_count'] <= 4 and 
            self.system_info['memory_gb'] <= 4):
            return 'embedded'
        
        return 'desktop'
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """Obtiene configuración recomendada basada en el hardware"""
        hardware_type = self.get_hardware_type()
        
        if hardware_type == 'embedded':
            return {
                'camera': {
                    'resolution': (640, 480),
                    'fps': 10
                },
                'ai': {
                    'model': 'lightweight',
                    'batch_size': 1,
                    'use_gpu': False
                },
                'processing': {
                    'threads': min(2, self.system_info['cpu_count']),
                    'queue_size': 10
                }
            }
        else:
            return {
                'camera': {
                    'resolution': (1280, 720),
                    'fps': 30
                },
                'ai': {
                    'model': 'full',
                    'batch_size': 4,
                    'use_gpu': self.system_info['has_gpu']
                },
                'processing': {
                    'threads': min(8, self.system_info['cpu_count']),
                    'queue_size': 50
                }
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información completa del sistema"""
        return self.system_info.copy()
    
    def print_system_info(self):
        """Imprime información del sistema"""
        print("=== INFORMACIÓN DEL SISTEMA ===")
        print(f"Plataforma: {self.system_info['platform']}")
        print(f"Arquitectura: {self.system_info['machine']}")
        print(f"Procesador: {self.system_info['processor']}")
        print(f"CPUs: {self.system_info['cpu_count']}")
        print(f"RAM: {self.system_info['memory_gb']} GB")
        print(f"Python: {self.system_info['python_version']}")
        print(f"Raspberry Pi: {'Sí' if self.system_info['is_raspberry_pi'] else 'No'}")
        print(f"Jetson: {'Sí' if self.system_info['is_jetson'] else 'No'}")
        print(f"GPU: {'Sí' if self.system_info['has_gpu'] else 'No'}")
        print(f"Tipo de hardware: {self.get_hardware_type()}")
        print("=" * 35)