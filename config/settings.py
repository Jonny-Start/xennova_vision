import json
import os
from pathlib import Path
from typing import Dict, Any
from utils.hardware_detector import HardwareDetector

class ConfigManager:
    """Gestor de configuración que selecciona automáticamente el archivo correcto"""
    
    def __init__(self):
        self.hardware_detector = HardwareDetector()
        self.config_dir = Path(__file__).parent
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración apropiada según el hardware"""
        hardware_type = self.hardware_detector.get_hardware_type()
        
        if hardware_type == "embedded":
            config_file = self.config_dir / "embedded_config.json"
        else:
            config_file = self.config_dir / "desktop_config.json"
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear configuración: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Obtiene configuración de cámara"""
        return self.get('camera', {})
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Obtiene configuración de IA"""
        return self.get('ai', {})
    
    def get_network_config(self) -> Dict[str, Any]:
        """Obtiene configuración de red"""
        return self.get('network', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Obtiene configuración de almacenamiento"""
        return self.get('storage', {})
    
    def is_embedded(self) -> bool:
        """Verifica si estamos en modo embedded"""
        return self.hardware_detector.get_hardware_type() == "embedded"