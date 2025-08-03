from core.interfaces import ICameraService, IPlateDetector, IEventStorage, INetworkService

# Importar servicios ultra-optimizados
from .detector_service import UltraSmartDetector
from .camera_service import UltraCameraService
from .factory_service import UltraServiceFactory

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