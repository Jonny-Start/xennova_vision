import cv2
import time
from typing import Optional
from ..core.interfaces import ICameraService

class USBCameraService(ICameraService):
    """Servicio de cámara USB para sistemas desktop"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device_id = config.get('device_id', 0)
        self.resolution = config.get('resolution', [640, 480])
        self.fps = config.get('fps', 30)
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Inicializa la cámara USB"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception as e:
            print(f"Error inicializando cámara USB: {e}")
            self.cap = None
    
    def capture_frame(self) -> Optional[bytes]:
        """Captura un frame de la cámara USB"""
        if not self.is_available():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        return None
    
    def is_available(self) -> bool:
        """Verifica si la cámara está disponible"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Libera recursos de la cámara"""
        if self.cap:
            self.cap.release()

class CSICameraService(ICameraService):
    """Servicio de cámara CSI para sistemas embebidos (Raspberry Pi, Jetson)"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device_id = config.get('device_id', 0)
        self.resolution = config.get('resolution', [640, 480])
        self.fps = config.get('fps', 15)
        self.cap = None
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Inicializa la cámara CSI"""
        try:
            # GStreamer pipeline para cámara CSI
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={self.device_id} ! "
                f"video/x-raw(memory:NVMM), width={self.resolution[0]}, height={self.resolution[1]}, "
                f"format=NV12, framerate={self.fps}/1 ! "
                "nvvidconv flip-method=0 ! "
                "video/x-raw, width=640, height=480, format=BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! appsink"
            )
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        except Exception as e:
            print(f"Error inicializando cámara CSI: {e}")
            # Fallback a cámara USB
            self.cap = cv2.VideoCapture(self.device_id)
    
    def capture_frame(self) -> Optional[bytes]:
        """Captura un frame de la cámara CSI"""
        if not self.is_available():
            return None
        
        ret, frame = self.cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        return None
    
    def is_available(self) -> bool:
        """Verifica si la cámara está disponible"""
        return self.cap is not None and self.cap.isOpened()
    
    def release(self) -> None:
        """Libera recursos de la cámara"""
        if self.cap:
            self.cap.release()