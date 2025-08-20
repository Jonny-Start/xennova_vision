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
        import os
        max_workers = 1 if os.getenv('XENNOVA_MODE') == 'single' else 2
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Estado de inicialización
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        # Patrones de placas colombianas optimizados
        self.plate_patterns = [
            r'^[A-Z]{3}[0-9]{3}$',    # ABC123 (carros)
            r'^[A-Z]{3}-[0-9]{3}$',    # ABC-123 (con guión)
            r'^[A-Z]{2}[0-9]{4}$',    # AB1234 (motos)
            r'^[A-Z]{2}-[0-9]{4}$',    # AB-1234 (motos con guión)
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

    def _create_mser_detector(self):
        """Crea detector MSER con compatibilidad de versiones OpenCV"""
        params = {
            'delta': 5,
            'min_area': 60, 
            'max_area': 14400,
            'max_variation': 0.25,
            'min_diversity': 0.2,
            'max_evolution': 200,
            'area_threshold': 1.01,
            'min_margin': 0.003,
            'edge_blur_size': 5
        }
        
        try:
            return cv2.MSER_create(**params)
        except TypeError:
            # Versiones antiguas usan underscore
            old_params = {f'_{k}': v for k, v in params.items()}
            return cv2.MSER_create(**old_params)
        
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
                    (w//4, h//2, w//2, h//4, 0.3),    # Centro-inferior
                    (w//6, h//3, 2*w//3, h//3, 0.2),    # Centro
                    (0, h//2, w, h//2, 0.1)    # Mitad inferior completa
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
        """Sistema principal de detección robusto y universal"""
        
        if not self.is_initialized:
            print("⚠️ Detector no inicializado")
            return []
        
        if image is None or image.size == 0:
            return []
        
        # Marcar modo imagen individual
        self._single_image_mode = True
        
        # Pre-análisis rápido de la imagen
        if not self._quick_plate_check(image):
            print("🔍 Pre-análisis: No se detectan características de placa")
            return []
        
        # Control de rendimiento (mejorado para imágenes individuales)
        if not self._should_process_frame_enhanced():
            return []
        
        start_time = time.time()
        
        try:
            self.stats['processed_frames'] += 1
            
            # HÍBRIDO: Usar detección universal + detección avanzada original
            # 1. Detección universal (nueva)
            universal_candidates = self._universal_plate_detection(image)
            
            # 2. Detección avanzada original (mantener)
            advanced_regions = self._detect_plate_regions_advanced(image)

            # AGREGAR: Detección por color como método adicional
            color_regions = self._detect_by_color_enhanced(image)
            if color_regions:
                print(f"🎨 Detectadas {len(color_regions)} regiones por color")
                # Agregar regiones de color a las regiones existentes
                for x, y, w, h, score in color_regions:
                    advanced_regions.append((x, y, w, h, score))
            
            # Combinar ambos enfoques
            all_candidates = []
            
            # Procesar candidatos universales
            for candidate in universal_candidates:
                x, y, w, h = candidate['bbox']
                all_candidates.append((x, y, w, h, candidate['confidence'], candidate['method']))
            
            # Procesar regiones avanzadas originales
            for x, y, w, h, region_score in advanced_regions:
                all_candidates.append((x, y, w, h, region_score, 'advanced_detection'))
            
            detections = []
            
            for x, y, w, h, region_score, detection_method in all_candidates:
                # Extraer ROI con padding (MANTENER ORIGINAL)
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image.shape[1], x + w + padding)
                y2 = min(image.shape[0], y + h + padding)
                
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                # HÍBRIDO: OCR múltiple (original) + OCR mejorado (nuevo)
                text_results = []
                
                # 1. OCR paralelo original (mantener)
                original_results = self._extract_text_parallel(roi)
                text_results.extend(original_results)
                
                # 2. OCR multi-escala nuevo (agregar)
                enhanced_text, enhanced_conf = self._multi_scale_ocr(roi)
                if enhanced_text:
                    text_results.append((enhanced_text, 'enhanced_ocr', enhanced_conf / 100.0))
                
                for text, method, ocr_confidence in text_results:
                    # HÍBRIDO: Validación original + validación flexible
                    # 1. Validación avanzada original
                    format_confidence, format_reason = self._validate_plate_format_advanced(text)
                    
                    # 2. Validación flexible nueva (como fallback)
                    if format_confidence <= 0.6:
                        is_valid, clean_text, validation_status = self._flexible_plate_validation(text)
                        if is_valid:
                            format_confidence = 0.7  # Confianza moderada para formatos desconocidos
                            format_reason = f"Flexible validation: {validation_status}"
                            text = clean_text
                    
                    if format_confidence > 0.6:
                        # Combinar confianzas (MANTENER ORIGINAL)
                        combined_confidence = (ocr_confidence * 0.6 + format_confidence * 0.4)
                        
                        # MANTENER: Análisis con histórico (CRÍTICO para confirmación)
                        is_confirmed, final_confidence, status = self._analyze_confidence_advanced(
                            text, method, combined_confidence
                        )
                        
                        # NUEVA: Clasificación de vehículo
                        vehicle_info = self._classify_vehicle_type(text, (w, h))
                        
                        # Crear detección con información completa
                        detection = PlateDetection(
                            plate_number=text,
                            confidence=final_confidence,
                            bbox=(x1, y1, x2, y2)
                        )
                        
                        # NUEVOS: Atributos adicionales
                        detection.vehicle_type = vehicle_info['type']
                        detection.vehicle_confidence = vehicle_info['confidence']
                        detection.country_hint = vehicle_info['country_hint']
                        detection.validation_status = getattr(locals(), 'validation_status', 'validated')
                        detection.detection_method = detection_method
                        
                        detections.append(detection)
                        self.stats['detections'] += 1
                        
                        # Log detallado (MANTENER)
                        processing_time = time.time() - start_time
                        vehicle_info_str = f"🚗 {vehicle_info['type']} ({vehicle_info['country_hint']})"
                        print(f"🔍 {status} | {format_reason} | {vehicle_info_str} | {processing_time:.3f}s")
                        
                        # CRÍTICO: Si está confirmada, retornar inmediatamente (MANTENER)
                        if is_confirmed:
                            self._update_processing_stats(processing_time)
                            return [detection]
            
            # MANTENER: Filtrar falsos positivos
            filtered_detections = self._filter_false_positives(detections)
            
            # Actualizar estadísticas
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time)
            
            return filtered_detections[:3]  # Máximo 3 detecciones
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []

    def _universal_plate_detection(self, image):
        """Detección universal de placas - cualquier país/color"""

        candidates = []

        # 1. Detección por contornos (genérica)
        candidates.extend(self._detect_by_contours_universal(image))

        # 2. Detección por bordes y geometría
        candidates.extend(self._detect_by_edges_geometry(image))

        # 3. Detección por textura (caracteres alfanuméricos)
        candidates.extend(self._detect_by_text_regions(image))

        # 4. Detección por múltiples colores (no solo amarillo)
        candidates.extend(self._detect_by_multiple_colors(image))

        return self._rank_candidates_universal(candidates, image)

    def _detect_by_multiple_colors(self, image):
        """Detección por múltiples colores comunes de placas"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates = []

        # Rangos de colores comunes en placas mundiales
        color_ranges = {
            'yellow': ([15, 100, 100], [35, 255, 255]),    # Colombia, Alemania
            'white': ([0, 0, 180], [180, 30, 255]),    # USA, México, Europa
            'blue': ([100, 50, 50], [130, 255, 255]),    # Europa (algunos)
            'green': ([40, 50, 50], [80, 255, 255]),    # Diplomáticas
            'black': ([0, 0, 0], [180, 255, 80])    # Algunos países
        }

        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            candidates.extend(self._extract_rectangular_regions(mask, f'color_{color_name}'))

        return candidates

    def _detect_by_contours_universal(self, image):
        """Detección genérica por contornos"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Preprocesamiento
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Filtros universales
            if (1.5 < aspect_ratio < 6.0 and 
                1000 < area < 50000 and
                w > 60 and h > 15):
                candidates.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.7,
                    'method': 'contours'
                })
        
        return candidates

    def _detect_by_text_regions(self, image):
        """Detecta regiones con alta densidad de texto alfanumérico"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # MSER (Maximally Stable Extremal Regions) para texto
        try:
            mser = self._create_mser_detector()
        except Exception as e:
            print(f"⚠️ Error creando MSER: {e}")
            return []

        regions, _ = mser.detectRegions(gray)
        candidates = []

        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            aspect_ratio = w / h

            # Filtros genéricos para regiones tipo placa
            if (1.5 < aspect_ratio < 6.0 and w > 60 and h > 15):
                candidates.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.7,
                    'method': 'text_regions'
                })

        return candidates

    def _extract_rectangular_regions(self, mask, method_name):
        """Extrae regiones rectangulares de una máscara"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            if (1.5 < aspect_ratio < 6.0 and 
                800 < area < 30000 and
                w > 60 and h > 15):
                candidates.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.75,
                    'method': method_name
                })
        
        return candidates

    def _rank_candidates_universal(self, candidates, image):
        """Rankea y filtra candidatos"""
        if not candidates:
            return []
        
        # Eliminar duplicados cercanos
        filtered = []
        for candidate in candidates:
            x, y, w, h = candidate['bbox']
            is_duplicate = False
            
            for existing in filtered:
                ex, ey, ew, eh = existing['bbox']
                # Verificar solapamiento
                overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                overlap_area = overlap_x * overlap_y
                
                if overlap_area > 0.5 * min(w * h, ew * eh):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(candidate)
        
        # Ordenar por confianza
        return sorted(filtered, key=lambda x: x['confidence'], reverse=True)[:5]

    def _multi_scale_ocr(self, plate_region):
        """OCR optimizado - menos escalas para imágenes individuales"""
        if plate_region is None or plate_region.size == 0:
            return "", 0
        
        if hasattr(self, '_single_image_mode'):
            cv2.imwrite('debug_region.jpg', plate_region)
            print(f"🔍 Región guardada: debug_region.jpg (tamaño: {plate_region.shape})")

        best_text = ""
        best_confidence = 0
        
        # Escalas reducidas para mejor rendimiento
        scales = [1.0] if hasattr(self, '_single_image_mode') else [0.8, 1.0, 1.2, 1.5, 2.0]
        
        for scale in scales:
            # Redimensionar
            height, width = plate_region.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if new_width < 20 or new_height < 10:
                continue
            
            # EARLY STOPPING: Si encuentra algo bueno, parar
            if best_confidence > 50:
                break
            
            resized = cv2.resize(plate_region, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Diferentes configuraciones de Tesseract
            configs = [
                "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "--psm 6 --oem 3"  # Sin restricción para casos difíciles
            ] if hasattr(self, '_single_image_mode') else [
                "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "--psm 6"
            ]
            
            for config in configs:
                try:
                    if self.tesseract:
                        text = self.tesseract.image_to_string(resized, config=config).strip()
                        text = re.sub(r'[^A-Z0-9]', '', text.upper())
                        
                        if len(text) >= 4:
                            # Estimar confianza basada en longitud y formato
                            confidence = min(90, len(text) * 10 + 30)
                            
                            if confidence > best_confidence:
                                best_text = text
                                best_confidence = confidence
                except:
                    continue
        
        return best_text, best_confidence

    def _flexible_plate_validation(self, text):
        """Validación flexible que no falla con formatos desconocidos"""

        # Limpiar texto básico
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())

        # Filtros básicos universales
        if len(clean_text) < 4:  # Muy corto
            return False, clean_text, "too_short"

        if len(clean_text) > 10:  # Muy largo
            return False, clean_text, "too_long"

        # Debe tener al menos una letra Y un número
        has_letter = bool(re.search(r'[A-Z]', clean_text))
        has_number = bool(re.search(r'[0-9]', clean_text))

        if not (has_letter and has_number):
            return False, clean_text, "invalid_format"

        # Correcciones comunes OCR (universales)
        corrections = {
            '0': 'O', 'O': '0',  # Bidireccional
            '1': 'I', 'I': '1',
            '5': 'S', 'S': '5',
            '8': 'B', 'B': '8',
            '6': 'G', 'G': '6'
        }

        # Intentar correcciones solo si es necesario
        corrected_versions = [clean_text]

        # Generar versiones corregidas
        for original, replacement in corrections.items():
            if original in clean_text:
                corrected = clean_text.replace(original, replacement)
                corrected_versions.append(corrected)

        # Retornar la mejor versión
        for version in corrected_versions:
            if len(version) >= 5:  # Longitud mínima razonable
                return True, version, "valid"

        return True, clean_text, "unknown_format"  # Aceptar formatos desconocidos

    def _classify_vehicle_type(self, plate_text, bbox_dimensions):
        """Clasifica el tipo de vehículo basado en formato de placa y dimensiones"""

        clean_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        width, height = bbox_dimensions
        aspect_ratio = width / height

        vehicle_info = {
            'type': 'unknown',
            'confidence': 0.0,
            'country_hint': 'unknown',
            'format': clean_text
        }

        # Patrones globales comunes
        patterns = {
            # Colombia
            'colombia_car': r'^[A-Z]{3}[0-9]{3}$',    # ABC123
            'colombia_car_new': r'^[A-Z]{3}[0-9]{2}[A-Z]$',  # ABC12D
            'colombia_moto': r'^[A-Z]{2}[0-9]{4}$',    # AB1234

            # México
            'mexico_car': r'^[A-Z]{3}[0-9]{4}$',    # ABC1234
            'mexico_moto': r'^[0-9]{2}[A-Z]{3}$',    # 12ABC

            # USA (simplificado)
            'usa_car': r'^[A-Z0-9]{6,8}$',    # Variable

            # Europa (genérico)
            'europe_car': r'^[A-Z]{1,3}[0-9]{2,4}[A-Z]{0,2}$',

            # Brasil
            'brazil_car': r'^[A-Z]{3}[0-9]{4}$',    # ABC1234
            'brazil_moto': r'^[A-Z]{3}[0-9]{3}$',    # ABC123
        }

        # Evaluar patrones
        for pattern_name, pattern in patterns.items():
            if re.match(pattern, clean_text):
                if 'moto' in pattern_name:
                    vehicle_info['type'] = 'motorcycle'
                    vehicle_info['confidence'] = 0.9
                else:
                    vehicle_info['type'] = 'car'
                    vehicle_info['confidence'] = 0.9

                vehicle_info['country_hint'] = pattern_name.split('_')[0]
                break

        # Clasificación por dimensiones si no hay patrón claro
        if vehicle_info['type'] == 'unknown':
            if aspect_ratio > 3.5:  # Placas muy alargadas suelen ser de moto
                vehicle_info['type'] = 'motorcycle'
                vehicle_info['confidence'] = 0.6
            elif 2.0 < aspect_ratio < 3.5:  # Proporción típica de carro
                vehicle_info['type'] = 'car'
                vehicle_info['confidence'] = 0.6

        return vehicle_info

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

    def _detect_by_color_enhanced(self, image):
        """Detección específica por color amarillo (Colombia) + fallback universal"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates = []
        
        # Rango para amarillo de placas colombianas (más amplio)
        lower_yellow = np.array([18, 50, 50])  # Más permisivo
        upper_yellow = np.array([35, 255, 255])
        
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # También detectar blancas (muchas placas son blancas)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combinar máscaras
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        
        # Operaciones morfológicas para limpiar
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = cv2.contourArea(contour)
            
            # Filtros específicos para placas
            if (1.8 < aspect_ratio < 5.0 and  # Proporción de placa
                800 < area < 20000 and    # Área razonable
                w > 60 and h > 20):    # Tamaño mínimo
                candidates.append((x, y, w, h, 0.8))
        
        return candidates

    def _should_process_frame_enhanced(self) -> bool:
        """Optimizado: detecta automáticamente modo de operación"""
        self.stats['total_frames'] += 1
        
        # Auto-detectar si es imagen individual (no hay atributo manual)
        if not hasattr(self, 'last_frame_time'):
            self.last_frame_time = time.time()
            return True
        
        # Si han pasado más de 2 segundos, es imagen individual
        current_time = time.time()
        if current_time - self.last_frame_time > 2.0:
            self.last_frame_time = current_time
            return True
        
        # Para video continuo, aplicar skip
        self.frame_skip_count += 1
        if self.frame_skip_count <= self.skip_frames:
            self.last_frame_time = current_time
            return False
        
        self.frame_skip_count = 0
        self.last_frame_time = current_time
        return True
        
    def _detect_by_edges_geometry(self, image):
        """Detección básica por bordes - función faltante"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detección de bordes simple
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Encontrar líneas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                minLineLength=50, maxLineGap=10)
        
        candidates = []
        if lines is not None:
            # Análisis básico de líneas para encontrar rectángulos
            h, w = image.shape[:2]
            # Región central como candidato básico
            candidates.append({
                'bbox': (w//4, h//3, w//2, h//3),
                'confidence': 0.5,
                'method': 'edges_geometry'
            })
        
        return candidates
        
    def _quick_plate_check(self, image: np.ndarray) -> bool:
        """Pre-análisis rápido para determinar si vale la pena procesar"""
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Detectar bordes básicos
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Si hay muy pocos bordes, probablemente no hay placas
            if edge_density < 0.01:
                return False
            
            # Buscar regiones rectangulares básicas
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangular_regions = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Área mínima
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 1.5 < aspect_ratio < 6.0:  # Proporción de placa
                        rectangular_regions += 1
            
            # Si no hay regiones rectangulares, no procesar
            return rectangular_regions > 0
            
        except Exception as e:
            print(f"⚠️ Error en pre-análisis: {e}")
            return True  # En caso de error, procesar por seguridad