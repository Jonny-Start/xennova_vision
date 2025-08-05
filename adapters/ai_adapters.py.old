"""
Adaptadores para diferentes modelos de IA - MEJORADO
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import easyocr
import re
from ultralytics import YOLO
import pytesseract

from core.interfaces import IPlateDetector
from core.models import PlateDetection
from core.exceptions import AIModelError
from utils.logger import get_logger

logger = get_logger(__name__)

class PlateValidator:
    """Validador de placas colombianas"""

    # Patrones de placas colombianas
    PATTERNS = [
        r'^[A-Z]{3}[0-9]{3}$',  # ABC123 (formato estándar)
        r'^[A-Z]{3}[0-9]{2}[A-Z]$',  # ABC12D (formato nuevo)
        r'^[A-Z]{2}[0-9]{4}$',  # AB1234 (motos)
        r'^[A-Z]{4}[0-9]{2}$',  # ABCD12 (diplomáticos)
    ]

    @classmethod
    def is_valid_plate(cls, text: str) -> bool:
        """Valida si el texto es una placa válida"""
        if not text:
            return False

        # Limpiar texto
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Verificar longitud
        if len(clean_text) < 5 or len(clean_text) > 6:
            return False

        # Verificar patrones
        for pattern in cls.PATTERNS:
            if re.match(pattern, clean_text):
                return True

        return False

    @classmethod
    def clean_plate_text(cls, text: str) -> str:
        """Limpia y formatea el texto de la placa"""
        # Remover caracteres no alfanuméricos
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Correcciones comunes de OCR
        corrections = {
            '0': 'O',  # En posiciones de letras
            'O': '0',  # En posiciones de números
            '1': 'I',  # En posiciones de letras
            'I': '1',  # En posiciones de números
            '5': 'S',  # En posiciones de letras
            'S': '5',  # En posiciones de números
        }

        # Aplicar correcciones basadas en posición
        if len(clean) == 6:  # Formato ABC123
            # Primeras 3 posiciones deben ser letras
            for i in range(3):
                if clean[i].isdigit() and clean[i] in corrections:
                    clean = clean[:i] + corrections[clean[i]] + clean[i+1:]

            # Últimas 3 posiciones deben ser números
            for i in range(3, 6):
                if clean[i].isalpha() and clean[i] in corrections:
                    clean = clean[:i] + corrections[clean[i]] + clean[i+1:]

        return clean

class EnhancedEasyOCRAdapter(IPlateDetector):
    """Adaptador mejorado para EasyOCR con preprocesamiento avanzado"""

    def __init__(self, languages: List[str] = None):
        self.languages = languages or ['en']
        self.reader = None
        self.is_initialized = False
        self.validator = PlateValidator()

    def initialize(self) -> bool:
        """Inicializa EasyOCR"""
        try:
            logger.info(f"Inicializando EasyOCR mejorado con idiomas: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            self.is_initialized = True
            logger.info("EasyOCR mejorado inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error inicializando EasyOCR: {e}")
            raise AIModelError(f"Error inicializando EasyOCR: {e}")

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placas en la imagen con preprocesamiento mejorado"""
        if not self.is_initialized:
            raise AIModelError("EasyOCR no inicializado")

        try:
            # Preprocesar imagen para mejorar OCR
            processed_images = self._preprocess_image(image)
            all_detections = []

            # Procesar cada variante de la imagen
            for processed_img in processed_images:
                results = self.reader.readtext(processed_img)

                for (bbox, text, confidence) in results:
                    # Limpiar y validar texto
                    clean_text = self.validator.clean_plate_text(text)

                    if self.validator.is_valid_plate(clean_text) and confidence > 0.3:
                        # Convertir bbox a formato estándar
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]

                        x1, y1 = int(min(x_coords)), int(min(y_coords))
                        x2, y2 = int(max(x_coords)), int(max(y_coords))

                        detection = PlateDetection(
                            plate_number=clean_text,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2)
                        )
                        all_detections.append(detection)

            # Filtrar duplicados y mantener los de mayor confianza
            unique_detections = self._filter_duplicates(all_detections)

            return unique_detections

        except Exception as e:
            logger.error(f"Error en detección EasyOCR mejorada: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Preprocesa la imagen para mejorar la detección"""
        processed_images = []

        # Imagen original
        processed_images.append(image)

        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Variante 1: Mejora de contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        # Variante 2: Filtro bilateral para reducir ruido
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        processed_images.append(cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR))

        # Variante 3: Threshold adaptativo
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2BGR))

        # Variante 4: Detección de bordes + dilatación
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        processed_images.append(cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR))

        return processed_images

    def _filter_duplicates(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """Filtra detecciones duplicadas manteniendo las de mayor confianza"""
        if not detections:
            return []

        # Agrupar por texto de placa
        plate_groups = {}
        for detection in detections:
            plate_text = detection.plate_number
            if plate_text not in plate_groups:
                plate_groups[plate_text] = []
            plate_groups[plate_text].append(detection)

        # Mantener solo la detección con mayor confianza por grupo
        unique_detections = []
        for plate_text, group in plate_groups.items():
            best_detection = max(group, key=lambda x: x.confidence)
            unique_detections.append(best_detection)

        return unique_detections

class HybridPlateDetector(IPlateDetector):
    """Detector híbrido que combina YOLO + OCR mejorado"""

    def __init__(self, yolo_model_path: str = "yolov8n.pt", languages: List[str] = None):
        self.yolo_model_path = yolo_model_path
        self.languages = languages or ['en']
        self.yolo_model = None
        self.ocr_reader = None
        self.is_initialized = False
        self.validator = PlateValidator()

    def initialize(self) -> bool:
        """Inicializa YOLO y OCR"""
        try:
            logger.info("Inicializando detector híbrido YOLO + OCR...")

            # Inicializar YOLO
            self.yolo_model = YOLO(self.yolo_model_path)

            # Inicializar EasyOCR
            self.ocr_reader = easyocr.Reader(self.languages, gpu=False)

            self.is_initialized = True
            logger.info("Detector híbrido inicializado correctamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando detector híbrido: {e}")
            raise AIModelError(f"Error inicializando detector híbrido: {e}")

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placas usando YOLO para localización y OCR para texto"""
        if not self.is_initialized:
            raise AIModelError("Detector híbrido no inicializado")

        try:
            detections = []

            # Paso 1: Detectar regiones de placas con YOLO
            yolo_results = self.yolo_model(image)
            plate_regions = []

            for result in yolo_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        if confidence > 0.3:  # Umbral más bajo para YOLO
                            plate_regions.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': float(confidence)
                            })

            # Paso 2: Si no se detectan regiones con YOLO, usar toda la imagen
            if not plate_regions:
                h, w = image.shape[:2]
                plate_regions = [{'bbox': (0, 0, w, h), 'confidence': 0.5}]

            # Paso 3: Aplicar OCR a cada región detectada
            for region_info in plate_regions:
                x1, y1, x2, y2 = region_info['bbox']
                region_confidence = region_info['confidence']

                # Extraer región
                plate_region = image[y1:y2, x1:x2]

                if plate_region.size == 0:
                    continue

                # Aplicar OCR con múltiples procesamientos
                ocr_results = self._extract_text_from_region(plate_region)

                for text, ocr_confidence, rel_bbox in ocr_results:
                    # Limpiar y validar texto
                    clean_text = self.validator.clean_plate_text(text)

                    if self.validator.is_valid_plate(clean_text):
                        # Calcular bbox absoluto
                        if rel_bbox:
                            abs_x1 = x1 + rel_bbox[0]
                            abs_y1 = y1 + rel_bbox[1]
                            abs_x2 = x1 + rel_bbox[2]
                            abs_y2 = y1 + rel_bbox[3]
                        else:
                            abs_x1, abs_y1, abs_x2, abs_y2 = x1, y1, x2, y2

                        # Combinar confianzas
                        combined_confidence = (region_confidence + ocr_confidence) / 2

                        detection = PlateDetection(
                            plate_number=clean_text,
                            confidence=combined_confidence,
                            bbox=(abs_x1, abs_y1, abs_x2, abs_y2)
                        )
                        detections.append(detection)

            # Filtrar duplicados
            unique_detections = self._filter_duplicates(detections)

            return unique_detections

        except Exception as e:
            logger.error(f"Error en detección híbrida: {e}")
            return []

    def _extract_text_from_region(self, region: np.ndarray) -> List[Tuple[str, float, Optional[Tuple[int, int, int, int]]]]:
        """Extrae texto de una región usando múltiples métodos"""
        results = []

        # Redimensionar región si es muy pequeña
        h, w = region.shape[:2]
        if h < 50 or w < 150:
            scale_factor = max(50/h, 150/w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            region = cv2.resize(region, (new_w, new_h))

        # Método 1: EasyOCR
        try:
            ocr_results = self.ocr_reader.readtext(region)
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.3:
                    # Convertir bbox relativo
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    rel_bbox = (int(min(x_coords)), int(min(y_coords)), 
                              int(max(x_coords)), int(max(y_coords)))
                    results.append((text, confidence, rel_bbox))
        except:
            pass

        # Método 2: Pytesseract con diferentes configuraciones
        try:
            # Preprocesar para Tesseract
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Configuraciones de Tesseract
            configs = [
                '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]

            for config in configs:
                text = pytesseract.image_to_string(gray, config=config).strip()
                if text and len(text) >= 5:
                    results.append((text, 0.7, None))  # Confianza fija para Tesseract
        except:
            pass

        return results

    def _filter_duplicates(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """Filtra detecciones duplicadas"""
        if not detections:
            return []

        # Agrupar por texto de placa
        plate_groups = {}
        for detection in detections:
            plate_text = detection.plate_number
            if plate_text not in plate_groups:
                plate_groups[plate_text] = []
            plate_groups[plate_text].append(detection)

        # Mantener solo la detección con mayor confianza por grupo
        unique_detections = []
        for plate_text, group in plate_groups.items():
            best_detection = max(group, key=lambda x: x.confidence)
            unique_detections.append(best_detection)

        return sorted(unique_detections, key=lambda x: x.confidence, reverse=True)

# Mantener adaptadores originales para compatibilidad
class EasyOCRAdapter(EnhancedEasyOCRAdapter):
    """Alias para compatibilidad hacia atrás"""
    pass

class YOLOAdapter(IPlateDetector):
    """Adaptador YOLO básico mantenido para compatibilidad"""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.is_initialized = False

    def initialize(self) -> bool:
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
        if not self.is_initialized:
            raise AIModelError("YOLO no inicializado")

        try:
            results = self.model(image)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()

                        if confidence > 0.5:
                            detection = PlateDetection(
                                plate_number="DETECTADO",  # Placeholder
                                confidence=float(confidence),
                                bbox=(int(x1), int(y1), int(x2), int(y2))
                            )
                            detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error en detección YOLO: {e}")
            return []

class MockAIAdapter(IPlateDetector):
    """Adaptador mock mejorado para pruebas"""

    def __init__(self):
        self.is_initialized = False
        self.detection_count = 0
        self.validator = PlateValidator()

    def initialize(self) -> bool:
        self.is_initialized = True
        logger.info("Adaptador AI mock mejorado inicializado")
        return True

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        if not self.is_initialized:
            raise AIModelError("Adaptador mock no inicializado")

        # Placas de prueba con formato colombiano válido
        mock_plates = ["ABC123", "XYZ789", "DEF456", "GHI012", "JKL345", "MNO678"]
        plate = mock_plates[self.detection_count % len(mock_plates)]

        detection = PlateDetection(
            plate_number=plate,
            confidence=0.95,
            bbox=(400, 300, 880, 420)
        )

        self.detection_count += 1
        return [detection]
