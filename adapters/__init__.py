"""
Adapters package - Implementaciones concretas de las interfaces
"""

from .camera_adapters import (
    OpenCVCameraAdapter,
    MockCameraAdapter
)

from .ai_adapters import (
    EasyOCRAdapter,
    YOLOAdapter,
    MockAIAdapter
)

from .storage_adapters import (
    JSONStorageAdapter,
    SQLiteStorageAdapter
)

__all__ = [
    'OpenCVCameraAdapter',
    'MockCameraAdapter',
    'EasyOCRAdapter',
    'YOLOAdapter',
    'MockAIAdapter',
    'JSONStorageAdapter',
    'SQLiteStorageAdapter'
]