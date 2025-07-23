from typing import Dict, Any
from core.interfaces import ICameraService, IPlateDetector, IEventStorage, INetworkService
from adapters.camera_adapters import OpenCVCameraAdapter, MockCameraAdapter
from adapters.ai_adapters import EasyOCRAdapter, YOLOAdapter, MockAIAdapter
from adapters.storage_adapters import JSONStorageAdapter, SQLiteStorageAdapter
from services.network_service import HTTPNetworkService
from utils.hardware_detector import HardwareDetector

class ServiceFactory:
    """Factory para crear servicios según la configuración"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hardware_detector = HardwareDetector()
    
    def create_camera_service(self) -> ICameraService:
        """Crea el servicio de cámara apropiado"""
        camera_config = self.config.get('camera', {})
        camera_type = camera_config.get('type', 'opencv')
        
        if camera_type == 'mock':
            return MockCameraAdapter()
        else:
            device_id = camera_config.get('device_id', 0)
            return OpenCVCameraAdapter(device_id)
    
    def create_plate_detector(self) -> IPlateDetector:
        """Crea el detector de placas apropiado"""
        ai_config = self.config.get('ai', {})
        detector_type = ai_config.get('detector', 'easyocr')
        
        if detector_type == 'mock':
            return MockAIAdapter()
        elif detector_type == 'yolo':
            model_path = ai_config.get('model_path', 'yolov8n.pt')
            return YOLOAdapter(model_path)
        else:
            languages = ai_config.get('languages', ['en'])
            return EasyOCRAdapter(languages)
    
    def create_storage_service(self) -> IEventStorage:
        """Crea el servicio de almacenamiento apropiado"""
        storage_config = self.config.get('storage', {})
        storage_type = storage_config.get('type', 'json')
        
        if storage_type == 'sqlite':
            db_path = storage_config.get('path', 'events.db')
            return SQLiteStorageAdapter(db_path)
        else:
            file_path = storage_config.get('path', 'events.json')
            return JSONStorageAdapter(file_path)
    
    def create_network_service(self) -> INetworkService:
        """Crea el servicio de red"""
        network_config = self.config.get('network', {})
        return HTTPNetworkService(
            endpoint=network_config.get('endpoint', ''),
            timeout=network_config.get('timeout', 30),
            retry_attempts=network_config.get('retry_attempts', 3)
        )