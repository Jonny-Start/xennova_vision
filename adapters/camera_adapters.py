"""
Adaptadores para diferentes tipos de cámaras
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import time

from core.interfaces import ICameraService
from core.exceptions import CameraError
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenCVCameraAdapter(ICameraService):
    """Adaptador para cámaras usando OpenCV"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
    
    def initialize(self) -> bool:
        """Inicializa la cámara"""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise CameraError(f"No se pudo abrir la cámara {self.device_id}")
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_active = True
            logger.info(f"Cámara {self.device_id} inicializada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando cámara: {e}")
            raise CameraError(f"Error inicializando cámara: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un frame de la cámara"""
        if not self.is_active or not self.cap:
            raise CameraError("Cámara no inicializada")
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("No se pudo capturar frame")
            return None
        
        return frame
    
    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución actual"""
        if not self.cap:
            return (0, 0)
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def release(self) -> None:
        """Libera los recursos de la cámara"""
        if self.cap:
            self.cap.release()
            self.is_active = False
            logger.info("Cámara liberada")

class MockCameraAdapter(ICameraService):
    """Adaptador mock para pruebas"""
    
    def __init__(self):
        self.is_active = False
        self.frame_count = 0
    
    def initialize(self) -> bool:
        """Inicializa la cámara mock"""
        self.is_active = True
        logger.info("Cámara mock inicializada")
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Genera un frame sintético"""
        if not self.is_active:
            raise CameraError("Cámara mock no inicializada")
        
        # Crear imagen sintética con placa
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Simular una placa
        cv2.rectangle(frame, (400, 300), (880, 420), (255, 255, 255), -1)
        cv2.rectangle(frame, (400, 300), (880, 420), (0, 0, 0), 3)
        
        # Texto de placa simulada
        plates = ["ABC123", "XYZ789", "DEF456", "GHI012"]
        plate_text = plates[self.frame_count % len(plates)]
        
        cv2.putText(frame, plate_text, (450, 370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        self.frame_count += 1
        time.sleep(0.1)  # Simular tiempo de captura
        
        return frame
    
    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución mock"""
        return (1280, 720)
    
    def release(self) -> None:
        """Libera recursos mock"""
        self.is_active = False
        logger.info("Cámara mock liberada")