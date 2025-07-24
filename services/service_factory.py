from typing import Dict, Any
from core.interfaces import ICameraService, IPlateDetector, IEventStorage, INetworkService
from adapters.camera_adapters import EnhancedOpenCVCameraAdapter, IPCameraAdapter, MockCameraAdapter
from adapters.ai_adapters import EnhancedEasyOCRAdapter, HybridPlateDetector, MockAIAdapter
from adapters.storage_adapters import JSONStorageAdapter, SQLiteStorageAdapter
from services.network_service import HTTPNetworkService
from utils.hardware_detector import HardwareDetector
from utils.logger import get_logger

logger = get_logger(__name__)

class ServiceFactory:
    """Factory mejorado para crear servicios según la configuración"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hardware_detector = HardwareDetector()

    def create_camera_service(self) -> ICameraService:
        """Crea el servicio de cámara apropiado"""
        camera_config = self.config.get('camera', {})
        camera_type = camera_config.get('type', 'opencv')

        logger.info(f"Creando servicio de cámara tipo: {camera_type}")

        if camera_type == 'mock':
            simulate_errors = camera_config.get('simulate_errors', False)
            return MockCameraAdapter(simulate_errors=simulate_errors)

        elif camera_type == 'ip':
            url = camera_config.get('url', '')
            username = camera_config.get('username')
            password = camera_config.get('password')
            return IPCameraAdapter(url, username, password)

        else:  # opencv o usb
            device_id = camera_config.get('device_id', 0)
            backup_devices = camera_config.get('backup_devices', [1, 2])
            return EnhancedOpenCVCameraAdapter(device_id, backup_devices)

    def create_plate_detector(self) -> IPlateDetector:
        """Crea el detector de placas apropiado"""
        ai_config = self.config.get('ai_model', {})  # Cambio de 'ai' a 'ai_model'
        detector_type = ai_config.get('type', 'advanced')  # Cambio de 'detector' a 'type'

        logger.info(f"Creando detector de placas tipo: {detector_type}")

        if detector_type == 'mock':
            return MockAIAdapter()

        elif detector_type == 'hybrid':
            yolo_model = ai_config.get('yolo_model', 'yolov8n.pt')
            languages = ai_config.get('languages', ['en'])
            return HybridPlateDetector(yolo_model, languages)

        elif detector_type == 'advanced':
            # Usar detector híbrido como avanzado por defecto
            yolo_model = ai_config.get('yolo_model', 'yolov8n.pt')
            languages = ai_config.get('languages', ['en'])

            # Intentar crear detector híbrido, si falla usar EasyOCR mejorado
            try:
                return HybridPlateDetector(yolo_model, languages)
            except Exception as e:
                logger.warning(f"No se pudo crear detector híbrido: {e}")
                logger.info("Usando detector EasyOCR mejorado como respaldo")
                return EnhancedEasyOCRAdapter(languages)

        else:  # 'basic' o cualquier otro
            languages = ai_config.get('languages', ['en'])
            return EnhancedEasyOCRAdapter(languages)

    def create_storage_service(self) -> IEventStorage:
        """Crea el servicio de almacenamiento apropiado"""
        storage_config = self.config.get('storage', {})
        storage_type = storage_config.get('type', 'sqlite')

        logger.info(f"Creando servicio de almacenamiento tipo: {storage_type}")

        if storage_type == 'sqlite':
            db_path = storage_config.get('database_path', 'plate_events.db')
            return SQLiteStorageAdapter(db_path)
        else:
            file_path = storage_config.get('path', 'events.json')
            return JSONStorageAdapter(file_path)

    def create_network_service(self) -> INetworkService:
        """Crea el servicio de red"""
        network_config = self.config.get('network', {})

        logger.info("Creando servicio de red HTTP")

        return HTTPNetworkService(
            endpoint=network_config.get('endpoint', ''),
            timeout=network_config.get('timeout', 30),
            retry_attempts=network_config.get('retry_attempts', 3)
        )

    def get_recommended_config(self) -> Dict[str, Any]:
        """Obtiene configuración recomendada basada en el hardware"""
        hardware_type = self.hardware_detector.get_hardware_type()

        if hardware_type == 'embedded':
            return {
                'camera': {
                    'type': 'opencv',
                    'device_id': 0,
                    'backup_devices': [1],
                    'resolution': [640, 480],
                    'fps': 15
                },
                'ai_model': {
                    'type': 'basic',
                    'languages': ['en'],
                    'confidence_threshold': 0.6
                },
                'capture_interval': 1.0
            }
        else:  # desktop
            return {
                'camera': {
                    'type': 'opencv',
                    'device_id': 0,
                    'backup_devices': [1, 2],
                    'resolution': [1280, 720],
                    'fps': 30
                },
                'ai_model': {
                    'type': 'advanced',
                    'yolo_model': 'yolov8n.pt',
                    'languages': ['en'],
                    'confidence_threshold': 0.7
                },
                'capture_interval': 0.5
            }
