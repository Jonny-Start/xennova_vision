from typing import Dict, Any
from core.interfaces import ICameraAdapter, IPlateDetector, IStorageAdapter
from adapters.camera_adapters import EnhancedOpenCVCameraAdapter, IPCameraAdapter, MockCameraAdapter
from adapters.ai_adapters import (
    TesseractOCRAdapter, 
    EasyOCRAdapter, 
    YOLOPlateDetector,
    PlateValidator
)
# Importar nuevos detectores inteligentes
from services.smart_plate_detector import SmartPlateDetector, OptimizedPlateDetector
from adapters.storage_adapters import SQLiteStorageAdapter, InMemoryStorageAdapter

class ServiceFactory:
    """Factory mejorado para crear servicios optimizados"""

    @staticmethod
    def create_camera_adapter(config: Dict[str, Any]) -> ICameraAdapter:
        """Crea adaptador de cámara con detección automática"""
        camera_config = config.get('camera', {})
        camera_type = camera_config.get('type', 'usb').lower()

        print(f"🎥 Configurando cámara tipo: {camera_type}")

        if camera_type == 'usb':
            device_id = camera_config.get('device_id', 0)
            backup_devices = camera_config.get('backup_devices', [])

            # Intentar dispositivo principal
            adapter = EnhancedOpenCVCameraAdapter(camera_config)

            # Probar dispositivos de respaldo si falla el principal
            if not adapter.is_connected() and backup_devices:
                print(f"⚠️ Cámara USB {device_id} no disponible, probando respaldos...")
                for backup_id in backup_devices:
                    backup_config = camera_config.copy()
                    backup_config['device_id'] = backup_id
                    backup_adapter = EnhancedOpenCVCameraAdapter(backup_config)
                    if backup_adapter.is_connected():
                        print(f"✅ Usando cámara USB de respaldo: {backup_id}")
                        return backup_adapter

            return adapter

        elif camera_type == 'ip':
            return IPCameraAdapter(camera_config)
        elif camera_type == 'mock':
            return MockCameraAdapter(camera_config)
        else:
            raise ValueError(f"Tipo de cámara no soportado: {camera_type}")

    @staticmethod
    def create_plate_detector(config: Dict[str, Any]) -> IPlateDetector:
        """Crea detector inteligente basado en configuración"""
        ai_config = config.get('ai_model', {})
        detector_mode = ai_config.get('detector_mode', 'smart').lower()

        print(f"🧠 Configurando detector modo: {detector_mode}")

        if detector_mode == 'smart':
            # Detector inteligente con análisis múltiple
            return SmartPlateDetector(ai_config)
        elif detector_mode == 'fast':
            # Detector optimizado para velocidad
            return OptimizedPlateDetector(ai_config)
        elif detector_mode == 'advanced':
            # Detector con YOLO (si está disponible)
            try:
                return YOLOPlateDetector(ai_config)
            except:
                print("⚠️ YOLO no disponible, usando detector inteligente")
                return SmartPlateDetector(ai_config)
        else:
            # Por defecto, usar detector inteligente
            return SmartPlateDetector(ai_config)

    @staticmethod
    def create_storage_adapter(config: Dict[str, Any]) -> IStorageAdapter:
        """Crea adaptador de almacenamiento"""
        storage_config = config.get('storage', {})
        storage_type = storage_config.get('type', 'sqlite').lower()

        if storage_type == 'sqlite':
            return SQLiteStorageAdapter(storage_config)
        elif storage_type == 'memory':
            return InMemoryStorageAdapter(storage_config)
        else:
            return SQLiteStorageAdapter(storage_config)  # Por defecto

    @staticmethod
    def get_available_cameras() -> Dict[int, str]:
        """Detecta cámaras disponibles en el sistema"""
        import cv2
        available = {}

        print("🔍 Detectando cámaras disponibles...")

        # Probar hasta 10 dispositivos
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Obtener información de la cámara
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))

                        available[i] = f"USB Camera {i} ({width}x{height} @ {fps}fps)"
                        print(f"  ✅ Dispositivo {i}: {available[i]}")
                cap.release()
            except:
                continue

        if not available:
            print("  ❌ No se encontraron cámaras disponibles")
        else:
            print(f"  📊 Total encontradas: {len(available)}")

        return available

    @staticmethod
    def test_camera(device_id: int) -> bool:
        """Prueba si una cámara específica funciona"""
        import cv2

        try:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except:
            return False

    @staticmethod
    def create_optimized_services(config_path: str = None):
        """Crea todos los servicios optimizados"""
        import json

        # Cargar configuración
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # Configuración por defecto optimizada
            config = {
                "camera": {"type": "usb", "device_id": 0},
                "ai_model": {"detector_mode": "smart"},
                "storage": {"type": "sqlite"}
            }

        print("🚀 Creando servicios optimizados...")

        # Detectar cámaras disponibles primero
        available_cameras = ServiceFactory.get_available_cameras()

        if not available_cameras:
            print("⚠️ No hay cámaras disponibles, usando mock")
            config['camera']['type'] = 'mock'

        # Crear servicios
        camera = ServiceFactory.create_camera_adapter(config)
        detector = ServiceFactory.create_plate_detector(config)
        storage = ServiceFactory.create_storage_adapter(config)

        return {
            'camera': camera,
            'detector': detector,
            'storage': storage,
            'config': config,
            'available_cameras': available_cameras
        }
