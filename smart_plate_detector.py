from typing import Optional, List, Dict
import cv2
import numpy as np
from ..core.interfaces import IPlateDetector
from ..core.models import DetectionResult
import time
import re
from collections import deque, Counter
import threading

class SmartPlateDetector(IPlateDetector):
    """Detector inteligente con análisis múltiple y cache"""

    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.8)

        # Sistema de cache y análisis múltiple
        self.recent_detections = deque(maxlen=10)  # Últimas 10 detecciones
        self.plate_candidates = Counter()  # Contador de placas candidatas
        self.last_confirmed_plate = None
        self.confirmation_count = 0
        self.min_confirmations = 3  # Mínimo 3 detecciones iguales

        # Cache para optimización
        self.frame_cache = {}
        self.cache_size = 5

        self._initialize_models()

        # Patrones de placas colombianas
        self.plate_patterns = [
            r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            r'^[A-Z]{3}-[0-9]{3}$', # ABC-123
            r'^[A-Z]{2}[0-9]{4}$',  # AB1234 (motos)
        ]

    def _initialize_models(self):
        """Inicializa modelos optimizados"""
        try:
            import easyocr
            import pytesseract

            # EasyOCR con configuración optimizada
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            self.ocr_engine = pytesseract

            # Configuración de Tesseract optimizada para placas
            self.tesseract_config = '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

            self.ready = True
            print("✅ Detector inteligente iniciado correctamente")
        except ImportError as e:
            print(f"❌ Error inicializando detector: {e}")
            self.ready = False

    def _preprocess_image(self, image):
        """Preprocesamiento optimizado para placas"""
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Filtro bilateral para reducir ruido manteniendo bordes
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Binarización adaptativa
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

        return gray, binary

    def _detect_plate_regions(self, image):
        """Detecta regiones candidatas a placas"""
        gray, binary = self._preprocess_image(image)

        # Encontrar contornos
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        plate_regions = []
        for contour in contours:
            # Filtrar por área y aspecto
            area = cv2.contourArea(contour)
            if area < 1000 or area > 50000:  # Filtrar por tamaño
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Placas suelen tener ratio entre 2:1 y 4:1
            if 1.5 < aspect_ratio < 5.0:
                plate_regions.append((x, y, w, h))

        return plate_regions

    def _extract_text_multiple_methods(self, roi):
        """Extrae texto usando múltiples métodos OCR"""
        results = []

        try:
            # Método 1: EasyOCR
            easyocr_results = self.easyocr_reader.readtext(roi, detail=0)
            for text in easyocr_results:
                if text.strip():
                    results.append((text.strip().upper(), 'easyocr'))
        except:
            pass

        try:
            # Método 2: Tesseract optimizado
            text = self.ocr_engine.image_to_string(roi, config=self.tesseract_config)
            if text.strip():
                results.append((text.strip().upper(), 'tesseract'))
        except:
            pass

        return results

    def _validate_plate_format(self, text: str) -> float:
        """Valida formato de placa colombiana y retorna confianza"""
        if not text or len(text) < 6:
            return 0.0

        # Limpiar texto
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

        # Verificar patrones
        for pattern in self.plate_patterns:
            if re.match(pattern, clean_text):
                return 0.95

            # Verificar con guiones
            formatted_text = clean_text[:3] + '-' + clean_text[3:]
            if re.match(pattern, formatted_text):
                return 0.90

        # Verificar formato aproximado (3 letras + 3 números)
        if len(clean_text) == 6:
            letters = clean_text[:3]
            numbers = clean_text[3:]
            if letters.isalpha() and numbers.isdigit():
                return 0.80

        return 0.0

    def _analyze_detection_confidence(self, plate_text: str) -> bool:
        """Analiza si tenemos suficiente confianza en la detección"""
        if not plate_text:
            return False

        # Agregar a candidatos
        self.plate_candidates[plate_text] += 1

        # Verificar si tenemos suficientes confirmaciones
        if self.plate_candidates[plate_text] >= self.min_confirmations:
            if plate_text != self.last_confirmed_plate:
                self.last_confirmed_plate = plate_text
                self.confirmation_count = 0
                return True  # Nueva placa confirmada!

        return False

    def detect_plate(self, image_data: bytes) -> DetectionResult:
        """Detección principal optimizada"""
        start_time = time.time()

        if not self.ready:
            return DetectionResult(None, 0.0, None, 0.0, False)

        try:
            # Convertir bytes a imagen
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return DetectionResult(None, 0.0, None, time.time() - start_time, False)

            # Detectar regiones de placas
            plate_regions = self._detect_plate_regions(image)

            best_plate = None
            best_confidence = 0.0

            # Analizar cada región
            for x, y, w, h in plate_regions:
                roi = image[y:y+h, x:x+w]

                # Extraer texto con múltiples métodos
                text_results = self._extract_text_multiple_methods(roi)

                for text, method in text_results:
                    # Validar formato
                    format_confidence = self._validate_plate_format(text)

                    if format_confidence > best_confidence:
                        best_plate = text
                        best_confidence = format_confidence

            processing_time = time.time() - start_time

            # Analizar confianza con histórico
            if best_plate and best_confidence > self.confidence_threshold:
                self.recent_detections.append(best_plate)

                # Verificar si tenemos confirmación
                is_confirmed = self._analyze_detection_confidence(best_plate)

                if is_confirmed:
                    print(f"🚀 RAPIDO! Placa confirmada: {best_plate}")
                    return DetectionResult(
                        best_plate, 
                        best_confidence, 
                        f"RAPIDO - {best_plate}", 
                        processing_time, 
                        True
                    )
                else:
                    confirmations = self.plate_candidates[best_plate]
                    print(f"🔍 Analizando: {best_plate} ({confirmations}/{self.min_confirmations})")
                    return DetectionResult(
                        best_plate, 
                        best_confidence * 0.7,  # Confianza reducida hasta confirmar
                        f"Analizando {best_plate} ({confirmations}/{self.min_confirmations})", 
                        processing_time, 
                        False
                    )

            return DetectionResult(None, 0.0, "Buscando placa...", processing_time, False)

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Error en detección: {e}")
            return DetectionResult(None, 0.0, f"Error: {str(e)}", processing_time, False)

    def is_ready(self) -> bool:
        return self.ready

    def reset_detection_history(self):
        """Reinicia el historial de detecciones"""
        self.recent_detections.clear()
        self.plate_candidates.clear()
        self.last_confirmed_plate = None
        self.confirmation_count = 0
        print("🔄 Historial de detecciones reiniciado")

class OptimizedPlateDetector(IPlateDetector):
    """Detector super optimizado para máximo rendimiento"""

    def __init__(self, config: dict):
        self.config = config
        self.ready = False
        self._initialize_fast_ocr()

        # Cache para frames similares
        self.frame_hash_cache = {}
        self.cache_hits = 0
        self.total_frames = 0

    def _initialize_fast_ocr(self):
        """Inicializa OCR optimizado para velocidad"""
        try:
            import pytesseract
            self.ocr_engine = pytesseract
            # Configuración super rápida
            self.fast_config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            self.ready = True
        except ImportError:
            self.ready = False

    def _quick_frame_hash(self, image):
        """Genera hash rápido del frame para cache"""
        small = cv2.resize(image, (64, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return hash(gray.tobytes())

    def detect_plate(self, image_data: bytes) -> DetectionResult:
        start_time = time.time()
        self.total_frames += 1

        if not self.ready:
            return DetectionResult(None, 0.0, None, 0.0, False)

        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Check cache first
            frame_hash = self._quick_frame_hash(image)
            if frame_hash in self.frame_hash_cache:
                self.cache_hits += 1
                cached_result = self.frame_hash_cache[frame_hash]
                print(f"💨 Cache hit! ({self.cache_hits}/{self.total_frames})")
                return cached_result

            # Procesamiento súper rápido
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Solo centro de la imagen (más rápido)
            h, w = gray.shape
            center_roi = gray[h//4:3*h//4, w//4:3*w//4]

            # OCR directo y rápido
            text = self.ocr_engine.image_to_string(center_roi, config=self.fast_config)
            text = re.sub(r'[^A-Z0-9]', '', text.upper())

            processing_time = time.time() - start_time

            # Validación rápida
            confidence = 0.0
            if len(text) == 6 and text[:3].isalpha() and text[3:].isdigit():
                confidence = 0.85

            result = DetectionResult(
                text if confidence > 0 else None, 
                confidence, 
                f"Rápido: {text}" if confidence > 0 else "Buscando...", 
                processing_time, 
                confidence > 0.8
            )

            # Cache result
            if len(self.frame_hash_cache) > 20:  # Limitar cache
                self.frame_hash_cache.clear()
            self.frame_hash_cache[frame_hash] = result

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            return DetectionResult(None, 0.0, f"Error: {str(e)}", processing_time, False)

    def is_ready(self) -> bool:
        return self.ready
