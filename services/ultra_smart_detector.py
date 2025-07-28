from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
from core.interfaces import IPlateDetector
from core.models import PlateDetection
import time
import re
from collections import deque, Counter
import threading
from concurrent.futures import ThreadPoolExecutor

class UltraSmartDetector(IPlateDetector):
    """
    Detector ultra-optimizado con múltiples mejoras:
    - Detección en tiempo real con cache inteligente
    - Múltiples motores OCR con validación cruzada
    - Sistema de confirmación adaptativo
    - Reducción drástica de falsos positivos
    - Compatible con interfaces existentes
    """

    def __init__(self, config: dict):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.85)
        
        # Sistema de cache y análisis múltiple
        self.recent_detections = deque(maxlen=15)
        self.plate_candidates = Counter()
        self.confirmed_plates = {}
        self.last_confirmed_plate = None
        self.confirmation_count = 0
        self.min_confirmations = config.get('min_confirmations', 3)
        
        # Control de rendimiento
        self.frame_skip_count = 0
        self.skip_frames = config.get('skip_frames', 2)  # Procesar 1 de cada 3 frames
        self.processing_time_history = deque(maxlen=10)
        
        # Threading para OCR paralelo
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Estado de inicialización
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        # Patrones de placas colombianas optimizados
        self.plate_patterns = [
            r'^[A-Z]{3}[0-9]{3}$',      # ABC123 (carros)
            r'^[A-Z]{3}-[0-9]{3}$',     # ABC-123 (con guión)
            r'^[A-Z]{2}[0-9]{4}$',      # AB1234 (motos)
            r'^[A-Z]{2}-[0-9]{4}$',     # AB-1234 (motos con guión)
            r'^[A-Z]{1}[0-9]{2}[A-Z]{1}[0-9]{2}$',  # A12B34 (diplomáticas)
        ]
        
        # Estadísticas
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'confirmed_detections': 0,
            'false_positives_filtered': 0,
            'avg_processing_time': 0.0
        }

    def initialize(self) -> bool:
        """Inicializa el detector con todos los motores OCR"""
        with self.initialization_lock:
            if self.is_initialized:
                return True
                
            try:
                print("🚀 Inicializando UltraSmartDetector...")
                
                # Intentar importar EasyOCR
                try:
                    import easyocr
                    self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                    print("  ✅ EasyOCR cargado")
                except ImportError:
                    print("  ⚠️ EasyOCR no disponible")
                    self.easyocr_reader = None
                
                # Intentar importar Tesseract
                try:
                    import pytesseract
                    self.tesseract = pytesseract
                    self.tesseract_config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    print("  ✅ Tesseract cargado")
                except ImportError:
                    print("  ⚠️ Tesseract no disponible")
                    self.tesseract = None
                
                # Verificar que al menos un motor esté disponible
                if not self.easyocr_reader and not self.tesseract:
                    print("  ❌ No hay motores OCR disponibles")
                    return False
                
                # Inicializar filtros de imagen
                self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                
                self.is_initialized = True
                print("  ✅ UltraSmartDetector inicializado correctamente")
                return True
                
            except Exception as e:
                print(f"  ❌ Error inicializando detector: {e}")
                self.is_initialized = False
                return False

    def _should_process_frame(self) -> bool:
        """Determina si debe procesar este frame (optimización de rendimiento)"""
        self.stats['total_frames'] += 1
        
        # Skip frames para mejorar rendimiento
        self.frame_skip_count += 1
        if self.frame_skip_count <= self.skip_frames:
            return False
        
        self.frame_skip_count = 0
        return True

    def _preprocess_image_advanced(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesamiento avanzado optimizado para placas"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Mejorar contraste con CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Filtro bilateral para reducir ruido manteniendo bordes
        filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
        
        # Múltiples binarizaciones para diferentes condiciones de luz
        binary1 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        binary2 = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Combinar binarizaciones
        binary_combined = cv2.bitwise_or(binary1, binary2)
        
        return enhanced, filtered, binary_combined

    def _detect_plate_regions_advanced(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detecta regiones candidatas con scoring avanzado"""
        try:
            enhanced, filtered, binary = self._preprocess_image_advanced(image)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            plate_regions = []
            h, w = image.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filtrar por área
                if area < 800 or area > w * h * 0.3:
                    continue
                
                # Obtener rectángulo delimitador
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                
                # Calcular métricas
                aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                fill_ratio = area / (w_rect * h_rect) if (w_rect * h_rect) > 0 else 0
                
                # Scoring basado en características típicas de placas
                score = 0.0
                
                # Aspect ratio ideal para placas (1.5:1 a 5:1)
                if 1.5 <= aspect_ratio <= 5.0:
                    score += 0.4
                elif 1.2 <= aspect_ratio <= 6.0:
                    score += 0.2
                
                # Fill ratio (qué tan lleno está el rectángulo)
                if 0.3 <= fill_ratio <= 0.9:
                    score += 0.3
                
                # Posición (placas suelen estar en la parte inferior)
                if y > h * 0.3:
                    score += 0.2
                
                # Tamaño relativo
                relative_area = area / (w * h)
                if 0.005 <= relative_area <= 0.1:
                    score += 0.1
                
                # Solo considerar regiones con score mínimo
                if score >= 0.5:
                    plate_regions.append((x, y, w_rect, h_rect, score))
            
            # Ordenar por score descendente
            plate_regions.sort(key=lambda x: x[4], reverse=True)
            
            # Si no hay regiones buenas, usar estrategia de fallback
            if not plate_regions:
                # Dividir imagen en regiones y evaluar cada una
                regions = [
                    (w//4, h//2, w//2, h//4, 0.3),      # Centro-inferior
                    (w//6, h//3, 2*w//3, h//3, 0.2),    # Centro
                    (0, h//2, w, h//2, 0.1)             # Mitad inferior completa
                ]
                plate_regions.extend(regions)
            
            return plate_regions[:5]  # Máximo 5 regiones
            
        except Exception as e:
            print(f"⚠️ Error detectando regiones: {e}")
            # Fallback básico
            h, w = image.shape[:2]
            return [(w//4, h//4, w//2, h//2, 0.1)]

    def _extract_text_parallel(self, roi: np.ndarray) -> List[Tuple[str, str, float]]:
        """Extrae texto usando múltiples métodos OCR en paralelo"""
        results = []
        futures = []
        
        # Lanzar OCR en paralelo
        if self.easyocr_reader:
            future = self.thread_pool.submit(self._ocr_easyocr, roi)
            futures.append(('easyocr', future))
        
        if self.tesseract:
            future = self.thread_pool.submit(self._ocr_tesseract, roi)
            futures.append(('tesseract', future))
        
        # Recoger resultados
        for method, future in futures:
            try:
                result = future.result(timeout=2.0)  # Timeout de 2 segundos
                if result:
                    results.extend(result)
            except Exception as e:
                print(f"⚠️ Error en {method}: {e}")
        
        return results

    def _ocr_easyocr(self, roi: np.ndarray) -> List[Tuple[str, str, float]]:
        """OCR con EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(roi, detail=1)
            extracted = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(clean_text) >= 4:
                        extracted.append((clean_text, 'easyocr', confidence))
            
            return extracted
        except Exception:
            return []

    def _ocr_tesseract(self, roi: np.ndarray) -> List[Tuple[str, str, float]]:
        """OCR con Tesseract"""
        try:
            text = self.tesseract.image_to_string(roi, config=self.tesseract_config)
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            if len(clean_text) >= 4:
                # Tesseract no da confidence, estimamos basado en longitud y formato
                confidence = min(0.9, len(clean_text) / 8.0 + 0.5)
                return [(clean_text, 'tesseract', confidence)]
            
            return []
        except Exception:
            return []

    def _validate_plate_format_advanced(self, text: str) -> Tuple[float, str]:
        """Validación avanzada de formato de placa"""
        if not text or len(text) < 4:
            return 0.0, "Texto muy corto"
        
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Verificar patrones exactos
        for i, pattern in enumerate(self.plate_patterns):
            if re.match(pattern, clean_text):
                confidence = 0.95 - (i * 0.05)  # Patrones más comunes tienen mayor confianza
                return confidence, f"Patrón exacto {i+1}"
            
            # Probar con guión insertado
            if len(clean_text) == 6:
                formatted = clean_text[:3] + '-' + clean_text[3:]
                if re.match(pattern, formatted):
                    return 0.90 - (i * 0.05), f"Patrón con guión {i+1}"
        
        # Validaciones aproximadas
        if len(clean_text) == 6:
            letters = clean_text[:3]
            numbers = clean_text[3:]
            if letters.isalpha() and numbers.isdigit():
                return 0.80, "Formato aproximado ABC123"
        
        if len(clean_text) == 6:
            letters = clean_text[:2]
            numbers = clean_text[2:]
            if letters.isalpha() and numbers.isdigit():
                return 0.75, "Formato moto AB1234"
        
        # Validación de caracteres válidos
        valid_chars = sum(1 for c in clean_text if c.isalnum())
        char_ratio = valid_chars / len(clean_text) if clean_text else 0
        
        if char_ratio > 0.8 and len(clean_text) >= 5:
            return 0.60, "Caracteres válidos"
        
        return 0.0, "No válido"

    def _analyze_confidence_advanced(self, plate_text: str, method: str, base_confidence: float) -> Tuple[bool, float, str]:
        """Análisis avanzado de confianza con histórico"""
        if not plate_text:
            return False, 0.0, "Sin texto"
        
        # Agregar a candidatos con peso por método
        method_weight = {'easyocr': 1.2, 'tesseract': 1.0}.get(method, 1.0)
        self.plate_candidates[plate_text] += method_weight
        
        confirmations = self.plate_candidates[plate_text]
        
        # Calcular confianza final
        final_confidence = min(0.98, base_confidence * (1 + confirmations * 0.1))
        
        # Verificar si debe confirmar
        threshold = self.min_confirmations
        
        # Reducir threshold para placas con alta confianza
        if base_confidence > 0.9:
            threshold = max(1, threshold - 1)
        
        if confirmations >= threshold:
            if plate_text != self.last_confirmed_plate:
                self.last_confirmed_plate = plate_text
                self.confirmed_plates[plate_text] = {
                    'timestamp': time.time(),
                    'confidence': final_confidence,
                    'confirmations': confirmations
                }
                self.stats['confirmed_detections'] += 1
                return True, final_confidence, f"🚀 CONFIRMADO: {plate_text}"
        
        status = f"Analizando {plate_text} ({confirmations:.1f}/{threshold}) - {method}"
        return False, final_confidence, status

    def _filter_false_positives(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """Filtro avanzado de falsos positivos"""
        if not detections:
            return detections
        
        filtered = []
        
        for detection in detections:
            plate = detection.plate_number
            
            # Filtros básicos
            if len(plate) < 4 or len(plate) > 8:
                self.stats['false_positives_filtered'] += 1
                continue
            
            # Filtrar secuencias repetitivas
            if len(set(plate)) < 3:  # Muy pocos caracteres únicos
                self.stats['false_positives_filtered'] += 1
                continue
            
            # Filtrar patrones obviamente incorrectos
            if re.match(r'^[0-9]+$', plate) or re.match(r'^[A-Z]+$', plate):
                self.stats['false_positives_filtered'] += 1
                continue
            
            # Filtrar caracteres problemáticos comunes en OCR
            problematic = ['O0O', '1I1', 'B8B', 'S5S']
            if any(prob in plate for prob in problematic):
                if detection.confidence < 0.9:  # Solo filtrar si confianza baja
                    self.stats['false_positives_filtered'] += 1
                    continue
            
            filtered.append(detection)
        
        return filtered

    def detect_plates(self, image: np.ndarray) -> List[PlateDetection]:
        """Método principal de detección optimizado"""
        if not self.is_initialized:
            print("⚠️ Detector no inicializado")
            return []
        
        if image is None or image.size == 0:
            return []
        
        # Control de rendimiento
        if not self._should_process_frame():
            return []
        
        start_time = time.time()
        
        try:
            self.stats['processed_frames'] += 1
            
            # Detectar regiones candidatas
            plate_regions = self._detect_plate_regions_advanced(image)
            detections = []
            
            for x, y, w, h, region_score in plate_regions:
                # Extraer ROI con padding
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # Extraer texto con múltiples métodos
                text_results = self._extract_text_parallel(roi)
                
                for text, method, ocr_confidence in text_results:
                    # Validar formato
                    format_confidence, format_reason = self._validate_plate_format_advanced(text)
                    
                    if format_confidence > 0.6:
                        # Combinar confianzas
                        combined_confidence = (ocr_confidence * 0.6 + format_confidence * 0.4)
                        
                        # Analizar con histórico
                        is_confirmed, final_confidence, status = self._analyze_confidence_advanced(
                            text, method, combined_confidence
                        )
                        
                        # Crear detección
                        detection = PlateDetection(
                            plate_number=text,
                            confidence=final_confidence,
                            bbox=(x1, y1, x2, y2)
                        )
                        
                        detections.append(detection)
                        self.stats['detections'] += 1
                        
                        # Log detallado
                        processing_time = time.time() - start_time
                        print(f"🔍 {status} | {format_reason} | {processing_time:.3f}s")
                        
                        # Si está confirmada, retornar inmediatamente
                        if is_confirmed:
                            self._update_processing_stats(processing_time)
                            return [detection]
            
            # Filtrar falsos positivos
            filtered_detections = self._filter_false_positives(detections)
            
            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            return filtered_detections[:3]  # Máximo 3 detecciones
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []

    def _update_processing_stats(self, processing_time: float):
        """Actualiza estadísticas de rendimiento"""
        self.processing_time_history.append(processing_time)
        self.stats['avg_processing_time'] = sum(self.processing_time_history) / len(self.processing_time_history)

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del detector"""
        return {
            **self.stats,
            'confirmed_plates_count': len(self.confirmed_plates),
            'candidate_plates_count': len(self.plate_candidates),
            'processing_rate': self.stats['processed_frames'] / max(1, self.stats['total_frames']),
            'detection_rate': self.stats['detections'] / max(1, self.stats['processed_frames']),
            'confirmation_rate': self.stats['confirmed_detections'] / max(1, self.stats['detections'])
        }

    def reset_history(self):
        """Reinicia historial y estadísticas"""
        self.recent_detections.clear()
        self.plate_candidates.clear()
        self.confirmed_plates.clear()
        self.last_confirmed_plate = None
        self.confirmation_count = 0
        self.processing_time_history.clear()
        
        # Reiniciar estadísticas
        for key in self.stats:
            self.stats[key] = 0 if 'count' in key or 'total' in key else 0.0
        
        print("🔄 Historial y estadísticas reiniciados")

    def cleanup(self):
        """Limpia recursos"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        print("🧹 Recursos limpiados")