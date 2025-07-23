"""
Interfaces para el sistema de reconocimiento de placas
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from .models import PlateEvent, PlateDetection

class ICameraService(ABC):
    """Interface para servicios de cámara"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializa la cámara"""
        pass
    
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un frame de la cámara"""
        pass
    
    @abstractmethod
    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución de la cámara"""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Libera los recursos de la cámara"""
        pass

class IPlateDetector(ABC):
    """Interface para detectores de placas"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializa el detector"""
        pass
    
    @abstractmethod
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placas en una imagen"""
        pass

class IEventStorage(ABC):
    """Interface para almacenamiento de eventos"""
    
    @abstractmethod
    def store_event(self, event: PlateEvent) -> bool:
        """Almacena un evento"""
        pass
    
    @abstractmethod
    def get_pending_events(self) -> List[PlateEvent]:
        """Obtiene eventos pendientes de envío"""
        pass
    
    @abstractmethod
    def mark_as_sent(self, event_id: str) -> bool:
        """Marca un evento como enviado"""
        pass

class INetworkService(ABC):
    """Interface para servicios de red"""
    
    @abstractmethod
    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """Envía un evento al servidor"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Verifica si hay conexión a internet"""
        pass

class IEventManager(ABC):
    """Interface para gestión de eventos"""
    
    @abstractmethod
    def process_detection(self, detection: PlateDetection, image_path: Optional[str] = None) -> bool:
        """Procesa una detección y crea un evento"""
        pass
    
    @abstractmethod
    def sync_pending_events(self) -> int:
        """Sincroniza eventos pendientes con el servidor"""
        pass
    
    @abstractmethod
    def get_event_count(self) -> int:
        """Obtiene el número de eventos pendientes"""
        pass