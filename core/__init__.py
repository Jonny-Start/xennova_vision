"""
Core package - Interfaces, modelos y excepciones principales
"""

from .models import (
    PlateDetection,
    PlateEvent,
    SystemStatus,
    ProcessingResult
)

from .interfaces import (
    ICameraService,
    IPlateDetector,
    IEventStorage,
    INetworkService,
    IEventManager
)

from .exceptions import (
    PlateSystemError,
    CameraError,
    AIModelError,
    NetworkError,
    StorageError
)

__all__ = [
    # Models
    'PlateDetection',
    'PlateEvent',
    'SystemStatus',
    'ProcessingResult',
    
    # Interfaces
    'ICameraService',
    'IPlateDetector',
    'IEventStorage',
    'INetworkService',
    'IEventManager',
    
    # Exceptions
    'PlateSystemError',
    'CameraError',
    'AIModelError',
    'NetworkError',
    'StorageError'
]