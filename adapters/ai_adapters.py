"""
Adaptadores para diferentes modelos de IA
"""

import cv2
import numpy as np
from typing import List, Optional
import easyocr
import re
from ultralytics import YOLO

from core.interfaces import IPlateDetector
from core.models import PlateDetection
from core.exceptions import AIModelError
from utils.logger import get_logger

logger = get_logger(__name__)

class EasyOCRAdapter(IPlateDetector):
    """Adaptador para EasyOCR"""
    
    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['en']
        self.reader = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Inicializa EasyOCR"""
        try:
            logger.info(f"Inicializando EasyOCR con idiomas: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            self.is_initialized = True
            logger.info("EasyOCR inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error inicializando EasyOCR: {e}")
            raise AIModelError(f"Error inicializando EasyOCR: {e}")
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placas en la imagen"""
        if not self.is_initialized:
            raise AIModelError("EasyOCR no inicializado")
        
        try:
            results = self.reader.readtext(image)
            detections = []
            
            for (bbox, text, confidence) in results:
                # Filtrar texto que parece placa
                if self._is_plate_like(text) and confidence > 0.5:
                    # Convertir bbox a formato estándar
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    detection = PlateDetection(
                        plate_number=text.upper().replace(' ', ''),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2)
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error en detección EasyOCR: {e}")
            return []
    
    def _is_plate_like(self, text: str) -> bool:
        """Verifica si el texto parece una placa"""
        # Limpiar texto
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Verificar longitud típica de placas
        if len(clean_text) < 4 or len(clean_text) > 8:
            return False
        
        # Verificar que tenga letras y números
        has_letters = bool(re.search(r'[A-Z]', clean_text))
        has_numbers = bool(re.search(r'[0-9]', clean_text))
        
        return has_letters and has_numbers

class YOLOAdapter(IPlateDetector):
    """Adaptador para YOLO"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Inicializa YOLO"""
        try:
            logger.info(f"Inicializando YOLO con modelo: {self.model_path}")
            self.model = YOLO(self.model_path)
            self.is_initialized = True
            logger.info("YOLO inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error inicializando YOLO: {e}")
            raise AIModelError(f"Error inicializando YOLO: {e}")
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placas usando YOLO"""
        if not self.is_initialized:
            raise AIModelError("YOLO no inicializado")
        
        try:
            results = self.model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extraer información de la detección
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence > 0.5:
                            # Extraer región de la placa
                            plate_region = image[int(y1):int(y2), int(x1):int(x2)]
                            
                            # Usar OCR en la región detectada
                            plate_text = self._extract_text_from_region(plate_region)
                            
                            if plate_text:
                                detection = PlateDetection(
                                    plate_number=plate_text,
                                    confidence=float(confidence),
                                    bbox=(int(x1), int(y1), int(x2), int(y2))
                                )
                                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error en detección YOLO: {e}")
            return []
    
    def _extract_text_from_region(self, region: np.ndarray) -> Optional[str]:
        """Extrae texto de una región usando OCR simple"""
        # Implementación básica - en producción usarías un OCR más sofisticado
        try:
            # Preprocesar imagen
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Aquí podrías usar pytesseract o EasyOCR
            # Por simplicidad, retornamos un placeholder
            return "ABC123"  # Placeholder
            
        except:
            return None

class MockAIAdapter(IPlateDetector):
    """Adaptador mock para pruebas"""
    
    def __init__(self):
        self.is_initialized = False
        self.detection_count = 0
    
    def initialize(self) -> bool:
        """Inicializa el adaptador mock"""
        self.is_initialized = True
        logger.info("Adaptador AI mock inicializado")
        return True
    
    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Genera detecciones mock"""
        if not self.is_initialized:
            raise AIModelError("Adaptador mock no inicializado")
        
        # Simular detecciones
        mock_plates = ["ABC123", "XYZ789", "DEF456", "GHI012"]
        plate = mock_plates[self.detection_count % len(mock_plates)]
        
        detection = PlateDetection(
            plate_number=plate,
            confidence=0.95,
            bbox=(400, 300, 880, 420)
        )
        
        self.detection_count += 1
        return [detection]