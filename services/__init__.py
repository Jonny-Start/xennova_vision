from core.interfaces import ICameraService, IPlateDetector, IEventStorage, INetworkService

# Importar servicios ultra-optimizados
from .ultra_smart_detector import UltraSmartDetector
from .ultra_camera_service import UltraCameraService
from .ultra_service_factory import UltraServiceFactory  # ← AGREGADO

# Legacy services for compatibility
SmartPlateDetector = None
OptimizedPlateDetector = None

try:
    from .camera_service import CameraService
except ImportError:
    CameraService = None

__all__ = [
    # Interfaces
    'ICameraService', 
    'IPlateDetector', 
    'IEventStorage', 
    'INetworkService',
    
    # Servicios Ultra (principales)
    'UltraSmartDetector',
    'UltraCameraService', 
    'UltraServiceFactory',  # ← AGREGADO
    
    # Servicios legacy (compatibilidad)
    'SmartPlateDetector',
    'OptimizedPlateDetector',
    'CameraService'
]