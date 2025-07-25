from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import time
import re
from collections import deque, Counter

# Imports absolutos para evitar problemas
from core.interfaces import IPlateDetector
from core.models import PlateDetection

class SmartPlateDetector(IPlateDetector):
    """Detector inteligente compatible con interfaces existentes"""

    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)

        # Sistema de cache y análisis múltiple
        self.recent_detections = deque(maxlen=10)
        self.plate_candidates = Counter()
        self.last_confirmed_plate = None
        self.confirmation_count = 0
        self.min_confirmations = config.get('min_confirmations', 3)

        # Estado de inicialización
        self.is_initialized = False

        # Patrones de placas colombianas
        self.plate_patterns = [
            r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            r'^[A-Z]{3}-[0-9]{3}$', # ABC-123
            r'^[A-Z]{2}[0-9]{4}$',  # AB1234 (motos)
        ]

    def initialize(self) -> bool:
        """Inicializa el detector"""
        try:
            # Intentar importar dependencias
            import easyocr
            import pytesseract

            # Inicializar motores OCR
            print("🔄 Inicializando EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

            print("🔄 Configurando Tesseract...")
            self.ocr_engine = pytesseract
            self.tesseract_config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            self.is_initialized = True
            print("✅ SmartPlateDetector iniciado correctamente")
            return True

        except ImportError as e:
            print(f"❌ Error importando dependencias: {e}")
            print("💡 Instala: pip install easyocr pytesseract")
            self.is_initialized = False
            return False
        except Exception as e:
            print(f"❌ Error inesperado inicializando: {e}")
            self.is_initialized = False
            return False

    def _preprocess_image(self, image: np.ndarray) -> tuple:
        """Preprocesamiento optimizado para placas"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Mejorar contraste
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)

            # Filtro bilateral
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)

            return gray, binary
        except Exception as e:
            print(f"⚠️ Error en preprocesamiento: {e}")
            return image, image

    def _detect_plate_regions(self, image: np.ndarray) -> List[tuple]:
        """Detecta regiones candidatas a placas"""
        try:
            gray, binary = self._preprocess_image(image)

            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            plate_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000 or area > 50000:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                # Placas suelen tener ratio entre 1.5:1 y 5:1
                if 1.5 < aspect_ratio < 5.0:
                    plate_regions.append((x, y, w, h))

            # Si no encuentra regiones, usar centro de la imagen
            if not plate_regions:
                h, w = image.shape[:2]
                plate_regions.append((w//4, h//4, w//2, h//2))

            return plate_regions

        except Exception as e:
            print(f"⚠️ Error detectando regiones: {e}")
            # Fallback: usar imagen completa
            h, w = image.shape[:2]
            return [(0, 0, w, h)]

    def _extract_text_multiple_methods(self, roi: np.ndarray) -> List[tuple]:
        """Extrae texto usando múltiples métodos OCR"""
        results = []

        # Método 1: EasyOCR
        try:
            if hasattr(self, 'easyocr_reader'):
                easyocr_results = self.easyocr_reader.readtext(roi, detail=0)
                for text in easyocr_results:
                    if text.strip():
                        clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                        if len(clean_text) >= 4:
                            results.append((clean_text, 'easyocr', 0.9))
        except Exception as e:
            print(f"⚠️ EasyOCR error: {e}")

        # Método 2: Tesseract
        try:
            if hasattr(self, 'ocr_engine'):
                text = self.ocr_engine.image_to_string(roi, config=self.tesseract_config)
                clean_text = re.sub(r'[^A-Z0-9]', '', text.strip().upper())
                if len(clean_text) >= 4:
                    results.append((clean_text, 'tesseract', 0.8))
        except Exception as e:
            print(f"⚠️ Tesseract error: {e}")

        return results

    def _validate_plate_format(self, text: str) -> float:
        """Valida formato de placa colombiana"""
        if not text or len(text) < 4:
            return 0.0

        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Verificar patrones exactos
        for pattern in self.plate_patterns:
            if re.match(pattern, clean_text):
                return 0.95

            # Con guión
            if len(clean_text) == 6:
                formatted = clean_text[:3] + '-' + clean_text[3:]
                if re.match(pattern, formatted):
                    return 0.90

        # Formato aproximado (3 letras + 3 números)
        if len(clean_text) == 6:
            letters = clean_text[:3]
            numbers = clean_text[3:]
            if letters.isalpha() and numbers.isdigit():
                return 0.80

        # Formato de moto (2 letras + 4 números)
        if len(clean_text) == 6:
            letters = clean_text[:2]
            numbers = clean_text[2:]
            if letters.isalpha() and numbers.isdigit():
                return 0.75

        return 0.0

    def _analyze_confidence(self, plate_text: str) -> tuple:
        """Analiza confianza y determina si confirmar"""
        if not plate_text:
            return False, 0.0, "Sin texto"

        # Agregar a candidatos
        self.plate_candidates[plate_text] += 1
        confirmations = self.plate_candidates[plate_text]

        # Verificar confirmación
        if confirmations >= self.min_confirmations:
            if plate_text != self.last_confirmed_plate:
                self.last_confirmed_plate = plate_text
                return True, 0.95, f"CONFIRMADA - {plate_text}"

        return False, 0.7, f"Analizando {plate_text} ({confirmations}/{self.min_confirmations})"

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detecta placas en una imagen (método requerido por la interfaz)"""
        if not self.is_initialized:
            return []

        if image is None or image.size == 0:
            return []

        try:
            start_time = time.time()

            # Detectar regiones
            plate_regions = self._detect_plate_regions(image)
            detections = []

            for x, y, w, h in plate_regions:
                # Extraer ROI con validación
                if y+h > image.shape[0] or x+w > image.shape[1]:
                    continue

                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                # Extraer texto
                text_results = self._extract_text_multiple_methods(roi)

                for text, method, base_confidence in text_results:
                    # Validar formato
                    format_confidence = self._validate_plate_format(text)

                    if format_confidence > 0.7:
                        # Analizar con histórico
                        is_confirmed, final_confidence, status = self._analyze_confidence(text)

                        # Crear detección
                        detection = PlateDetection(
                            plate_number=text,
                            confidence=final_confidence,
                            bbox=(x, y, x+w, y+h)
                        )

                        detections.append(detection)

                        # Log para debug
                        processing_time = time.time() - start_time
                        if is_confirmed:
                            print(f"🚀 {status} (método: {method}, tiempo: {processing_time:.3f}s)")
                        else:
                            print(f"🔍 {status} (método: {method})")

                        # Si está confirmada, retornar inmediatamente
                        if is_confirmed:
                            return [detection]

            return detections[:3]  # Máximo 3 detecciones

        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []

    def reset_history(self):
        """Reinicia historial de detecciones"""
        self.recent_detections.clear()
        self.plate_candidates.clear()
        self.last_confirmed_plate = None
        self.confirmation_count = 0
        print("🔄 Historial reiniciado")

class FastPlateDetector(IPlateDetector):
    """Detector rápido y simple"""

    def __init__(self, config: dict):
        self.config = config
        self.is_initialized = False
        self.frame_cache = {}

    def initialize(self) -> bool:
        """Inicializa detector rápido"""
        try:
            import pytesseract
            self.ocr_engine = pytesseract
            self.fast_config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            self.is_initialized = True
            print("✅ FastPlateDetector iniciado")
            return True
        except ImportError:
            print("❌ Error: instala pytesseract")
            return False

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Detección rápida"""
        if not self.is_initialized or image is None:
            return []

        try:
            # Solo centro de imagen para velocidad
            h, w = image.shape[:2]
            center_roi = image[h//4:3*h//4, w//4:3*w//4]

            if len(center_roi.shape) == 3:
                gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = center_roi

            # OCR directo
            text = self.ocr_engine.image_to_string(gray, config=self.fast_config)
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

            # Validación rápida
            if len(clean_text) == 6 and clean_text[:3].isalpha() and clean_text[3:].isdigit():
                return [PlateDetection(
                    plate_number=clean_text,
                    confidence=0.85,
                    bbox=(w//4, h//4, 3*w//4, 3*h//4)
                )]

            return []

        except Exception as e:
            print(f"❌ Error detección rápida: {e}")
            return []
