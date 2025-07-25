from typing import Dict, Any, List
import cv2

# Imports absolutos para evitar problemas
from core.interfaces import ICameraService, IPlateDetector, IEventStorage, INetworkService
from fixed_smart_detector import SmartPlateDetector, FastPlateDetector

# Crear una clase CameraService compatible
class CameraService(ICameraService):
    """Servicio de cámara compatible con la interfaz existente"""

    def __init__(self, config: dict):
        self.config = config
        self.device_id = config.get('device_id', 0)
        self.resolution = config.get('resolution', [1280, 720])
        self.fps = config.get('fps', 30)
        self.cap = None
        self._is_initialized = False

    def initialize(self) -> bool:
        """Inicializa la cámara"""
        try:
            print(f"🎥 Inicializando cámara USB {self.device_id}...")
            self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                print(f"❌ No se pudo abrir cámara {self.device_id}")
                return False

            # Configurar resolución y FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Verificar que funciona
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"❌ Cámara {self.device_id} no puede capturar frames")
                self.cap.release()
                return False

            self._is_initialized = True
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"✅ Cámara {self.device_id} inicializada: {actual_width}x{actual_height}@{actual_fps}fps")
            return True

        except Exception as e:
            print(f"❌ Error inicializando cámara: {e}")
            return False

    def capture_frame(self):
        """Captura un frame de la cámara"""
        if not self._is_initialized or not self.cap or not self.cap.isOpened():
            return None

        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame  # Retornar numpy array directamente
            return None
        except Exception as e:
            print(f"⚠️ Error capturando frame: {e}")
            return None

    def get_resolution(self):
        """Obtiene la resolución actual"""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return self.resolution

    def release(self):
        """Libera los recursos de la cámara"""
        if self.cap:
            self.cap.release()
            self._is_initialized = False
            print("🔄 Cámara liberada")

# Adaptadores de almacenamiento simples
class SimpleStorageAdapter(IEventStorage):
    """Adaptador de almacenamiento simple"""

    def __init__(self, config: dict):
        self.config = config
        self.events = []

    def store_event(self, event) -> bool:
        """Almacena un evento"""
        try:
            self.events.append(event)
            return True
        except:
            return False

    def get_pending_events(self):
        """Obtiene eventos pendientes"""
        return [e for e in self.events if not getattr(e, 'sent', False)]

    def mark_as_sent(self, event_id: str) -> bool:
        """Marca evento como enviado"""
        for event in self.events:
            if getattr(event, 'id', None) == event_id:
                event.sent = True
                return True
        return False

class SimpleNetworkService(INetworkService):
    """Servicio de red simple"""

    def __init__(self, config: dict):
        self.config = config
        self.endpoint = config.get('endpoint', 'http://localhost:3000/api/placa')

    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """Simula envío de evento"""
        print(f"📡 Enviando evento: {event_data}")
        return True

    def is_connected(self) -> bool:
        """Simula verificación de conexión"""
        return True

class ServiceFactory:
    """Factory corregido con imports absolutos"""

    @staticmethod
    def create_camera_service(config: Dict[str, Any]) -> ICameraService:
        """Crea servicio de cámara"""
        camera_config = config.get('camera', {})
        return CameraService(camera_config)

    @staticmethod
    def create_plate_detector(config: Dict[str, Any]) -> IPlateDetector:
        """Crea detector de placas inteligente"""
        ai_config = config.get('ai_model', {})
        detector_mode = ai_config.get('detector_mode', 'smart').lower()

        print(f"🧠 Creando detector modo: {detector_mode}")

        if detector_mode == 'smart':
            return SmartPlateDetector(ai_config)
        elif detector_mode == 'fast':
            return FastPlateDetector(ai_config)
        else:
            return SmartPlateDetector(ai_config)

    @staticmethod
    def create_event_storage(config: Dict[str, Any]) -> IEventStorage:
        """Crea adaptador de almacenamiento"""
        return SimpleStorageAdapter(config.get('storage', {}))

    @staticmethod
    def create_network_service(config: Dict[str, Any]) -> INetworkService:
        """Crea servicio de red"""
        return SimpleNetworkService(config.get('network', {}))

    @staticmethod
    def get_available_cameras() -> Dict[int, str]:
        """Detecta cámaras disponibles"""
        available = {}

        print("🔍 Buscando cámaras disponibles...")

        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        available[i] = f"Camera {i} ({width}x{height}@{fps}fps)"
                        print(f"  ✅ {available[i]}")
                cap.release()
            except:
                continue

        if not available:
            print("  ❌ No se encontraron cámaras")

        return available

    @staticmethod
    def test_camera(device_id: int) -> bool:
        """Prueba si una cámara específica funciona"""
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
    def create_all_services(config: Dict[str, Any] = None):
        """Crea todos los servicios necesarios"""
        if config is None:
            config = {
                "camera": {"device_id": 0, "resolution": [1280, 720], "fps": 30},
                "ai_model": {"detector_mode": "smart", "confidence_threshold": 0.85},
                "storage": {"type": "simple"},
                "network": {"endpoint": "http://localhost:3000/api/placa"}
            }

        print("🚀 Creando todos los servicios...")

        # Verificar cámaras disponibles
        available_cameras = ServiceFactory.get_available_cameras()

        if not available_cameras:
            print("⚠️ No hay cámaras disponibles")
            return None

        # Crear servicios
        try:
            camera_service = ServiceFactory.create_camera_service(config)
            plate_detector = ServiceFactory.create_plate_detector(config)
            event_storage = ServiceFactory.create_event_storage(config)
            network_service = ServiceFactory.create_network_service(config)

            # Inicializar servicios críticos
            if not camera_service.initialize():
                print("❌ Error inicializando cámara")
                return None

            if not plate_detector.initialize():
                print("❌ Error inicializando detector")
                return None

            print("✅ Todos los servicios creados correctamente")

            return {
                'camera': camera_service,
                'detector': plate_detector,
                'storage': event_storage,
                'network': network_service,
                'config': config,
                'available_cameras': available_cameras
            }

        except Exception as e:
            print(f"❌ Error creando servicios: {e}")
            return None
