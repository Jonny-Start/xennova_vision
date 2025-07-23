"""
Modelos de datos para el sistema de reconocimiento de placas
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List
import uuid

@dataclass
class PlateDetection:
    """Representa una detección de placa"""
    plate_number: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    
    def __post_init__(self):
        """Validación post-inicialización"""
        if not self.plate_number:
            raise ValueError("El número de placa no puede estar vacío")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("La confianza debe estar entre 0.0 y 1.0")
        
        # Limpiar número de placa
        self.plate_number = self.plate_number.upper().strip()

@dataclass
class PlateEvent:
    """Representa un evento de detección de placa"""
    plate_number: str
    confidence: float
    timestamp: datetime
    bbox: Optional[Tuple[int, int, int, int]] = None
    image_path: Optional[str] = None
    sent: bool = False
    id: str = None
    
    def __post_init__(self):
        """Validación y asignación de ID post-inicialización"""
        if self.id is None:
            self.id = str(uuid.uuid4())
        
        if not self.plate_number:
            raise ValueError("El número de placa no puede estar vacío")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("La confianza debe estar entre 0.0 y 1.0")
        
        # Limpiar número de placa
        self.plate_number = self.plate_number.upper().strip()
    
    @classmethod
    def from_detection(
        cls, 
        detection: PlateDetection, 
        image_path: Optional[str] = None
    ) -> 'PlateEvent':
        """Crea un PlateEvent desde una PlateDetection"""
        return cls(
            plate_number=detection.plate_number,
            confidence=detection.confidence,
            timestamp=datetime.now(),
            bbox=detection.bbox,
            image_path=image_path,
            sent=False
        )
    
    def to_dict(self) -> dict:
        """Convierte el evento a diccionario para serialización"""
        return {
            'id': self.id,
            'plate_number': self.plate_number,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'bbox': list(self.bbox) if self.bbox else None,
            'image_path': self.image_path,
            'sent': self.sent
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PlateEvent':
        """Crea un PlateEvent desde un diccionario"""
        return cls(
            id=data['id'],
            plate_number=data['plate_number'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            bbox=tuple(data['bbox']) if data['bbox'] else None,
            image_path=data.get('image_path'),
            sent=data.get('sent', False)
        )

@dataclass
class SystemStatus:
    """Representa el estado del sistema"""
    camera_active: bool = False
    ai_model_loaded: bool = False
    network_connected: bool = False
    events_pending: int = 0
    last_detection: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convierte el estado a diccionario"""
        return {
            'camera_active': self.camera_active,
            'ai_model_loaded': self.ai_model_loaded,
            'network_connected': self.network_connected,
            'events_pending': self.events_pending,
            'last_detection': self.last_detection.isoformat() if self.last_detection else None
        }

@dataclass
class ProcessingResult:
    """Resultado del procesamiento de una imagen"""
    success: bool
    detections: List[PlateDetection]
    processing_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convierte el resultado a diccionario"""
        return {
            'success': self.success,
            'detections': [
                {
                    'plate_number': det.plate_number,
                    'confidence': det.confidence,
                    'bbox': list(det.bbox) if det.bbox else None
                }
                for det in self.detections
            ],
            'processing_time': self.processing_time,
            'error_message': self.error_message
        }