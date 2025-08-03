#!/usr/bin/env python3
"""
Visualizador Ultra en Tiempo Real
- Ventana con detecciones visuales
- Estadísticas en tiempo real
- Control de cámaras
- Interfaz optimizada
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, List, Optional
import argparse

from services.factory_service import UltraServiceFactory
from core.models import PlateDetection
from utils.logger import get_logger

logger = get_logger(__name__)

class UltraLiveViewer:
    """
    Visualizador Ultra en Tiempo Real
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.system = None
        self.is_running = False
        
        # Configuración de visualización
        self.window_name = "Xennova Vision - Ultra Live Viewer"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        
        # Colores
        self.colors = {
            'confirmed': (0, 255, 0),      # Verde para confirmadas
            'candidate': (0, 255, 255),    # Amarillo para candidatas
            'text': (255, 255, 255),       # Blanco para texto
            'background': (0, 0, 0),       # Negro para fondo
            'stats_bg': (50, 50, 50)       # Gris para estadísticas
        }
        
        # Estadísticas
        self.stats = {
            'frames_shown': 0,
            'detections_total': 0,
            'confirmations_total': 0,
            'fps_display': 0.0,
            'last_fps_time': time.time()
        }
    
    def initialize(self) -> bool:
        """Inicializa el sistema de visualización"""
        try:
            logger.info("🚀 Inicializando Ultra Live Viewer...")
            
            # Crear sistema completo
            self.system = UltraServiceFactory.create_complete_system(self.config)
            
            if not self.system:
                logger.error("❌ No se pudo crear sistema")
                return False
            
            # Configurar ventana
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            
            logger.info("✅ Ultra Live Viewer inicializado")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando viewer: {e}")
            return False
    
    def run(self):
        """Ejecuta el visualizador en tiempo real"""
        if not self.system:
            logger.error("Sistema no inicializado")
            return
        
        logger.info("🎬 Iniciando visualización en tiempo real...")
        logger.info("Controles:")
        logger.info("  - ESC/Q: Salir")
        logger.info("  - SPACE: Cambiar cámara")
        logger.info("  - R: Reiniciar detector")
        logger.info("  - S: Mostrar estadísticas")
        
        self.is_running = True
        camera = self.system['camera']
        detector = self.system['detector']
        
        try:
            while self.is_running:
                # Capturar frame
                frame = camera.capture_frame()
                
                if frame is None:
                    logger.warning("Frame nulo, continuando...")
                    time.sleep(0.1)
                    continue
                
                # Detectar placas
                detections = detector.detect_plates(frame)
                
                # Crear frame de visualización
                display_frame = self._create_display_frame(frame, detections)
                
                # Mostrar frame
                cv2.imshow(self.window_name, display_frame)
                
                # Actualizar estadísticas
                self._update_display_stats(detections)
                
                # Manejar teclas
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_press(key):
                    break
            
        except KeyboardInterrupt:
            logger.info("Interrumpido por usuario")
        except Exception as e:
            logger.error(f"Error en visualización: {e}")
        finally:
            self._cleanup()
    
    def _create_display_frame(self, frame: np.ndarray, detections: List[PlateDetection]) -> np.ndarray:
        """Crea frame de visualización con detecciones y estadísticas"""
        display_frame = frame.copy()
        
        # Dibujar detecciones
        for detection in detections:
            self._draw_detection(display_frame, detection)
        
        # Dibujar estadísticas
        self._draw_stats_overlay(display_frame)
        
        # Dibujar información del sistema
        self._draw_system_info(display_frame)
        
        return display_frame
    
    def _draw_detection(self, frame: np.ndarray, detection: PlateDetection):
        """Dibuja una detección en el frame"""
        if not detection.bbox:
            return
        
        x1, y1, x2, y2 = detection.bbox
        
        # Determinar color según confianza
        if detection.confidence > 0.9:
            color = self.colors['confirmed']
            status = "CONFIRMADA"
        else:
            color = self.colors['candidate']
            status = "CANDIDATA"
        
        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
        
        # Preparar texto
        plate_text = f"{detection.plate_number}"
        confidence_text = f"{detection.confidence:.2f}"
        status_text = f"{status}"
        
        # Calcular posición del texto
        text_y = y1 - 10 if y1 > 30 else y2 + 25
        
        # Dibujar fondo para texto
        text_size = cv2.getTextSize(plate_text, self.font, self.font_scale, self.thickness)[0]
        cv2.rectangle(frame, 
                     (x1, text_y - text_size[1] - 10), 
                     (x1 + text_size[0] + 10, text_y + 5), 
                     color, -1)
        
        # Dibujar texto de placa
        cv2.putText(frame, plate_text, (x1 + 5, text_y - 5), 
                   self.font, self.font_scale, (0, 0, 0), self.thickness)
        
        # Dibujar confianza y estado
        info_text = f"{confidence_text} - {status_text}"
        cv2.putText(frame, info_text, (x1, y2 + 20), 
                   self.font, 0.5, color, 1)
    
    def _draw_stats_overlay(self, frame: np.ndarray):
        """Dibuja overlay de estadísticas"""
        h, w = frame.shape[:2]
        
        # Área de estadísticas
        stats_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, stats_height), self.colors['stats_bg'], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Obtener estadísticas del detector
        detector_stats = self.system['detector'].get_stats()
        camera_stats = self.system['camera'].get_performance_stats()
        
        # Textos de estadísticas
        stats_lines = [
            f"FPS: {camera_stats.get('current_fps', 0):.1f}",
            f"Frames: {self.stats['frames_shown']}",
            f"Detecciones: {self.stats['detections_total']}",
            f"Confirmadas: {self.stats['confirmations_total']}",
            f"Candidatos activos: {detector_stats.get('active_candidates', 0)}"
        ]
        
        # Dibujar estadísticas
        for i, line in enumerate(stats_lines):
            y_pos = 30 + i * 20
            cv2.putText(frame, line, (20, y_pos), 
                       self.font, 0.6, self.colors['text'], 1)
    
    def _draw_system_info(self, frame: np.ndarray):
        """Dibuja información del sistema"""
        h, w = frame.shape[:2]
        
        # Información en la esquina superior derecha
        camera_info = self.system['camera'].get_performance_stats()
        device_id = camera_info.get('device_id', 'N/A')
        resolution = camera_info.get('resolution', (0, 0))
        
        info_text = f"Camara {device_id} - {resolution[0]}x{resolution[1]}"
        
        # Calcular posición
        text_size = cv2.getTextSize(info_text, self.font, 0.6, 1)[0]
        x_pos = w - text_size[0] - 20
        
        # Fondo
        cv2.rectangle(frame, (x_pos - 10, 10), (w - 10, 40), self.colors['stats_bg'], -1)
        
        # Texto
        cv2.putText(frame, info_text, (x_pos, 30), 
                   self.font, 0.6, self.colors['text'], 1)
    
    def _update_display_stats(self, detections: List[PlateDetection]):
        """Actualiza estadísticas de visualización"""
        self.stats['frames_shown'] += 1
        self.stats['detections_total'] += len(detections)
        
        # Contar confirmaciones
        for detection in detections:
            if detection.confidence > 0.9:
                self.stats['confirmations_total'] += 1
        
        # Actualizar FPS de display
        current_time = time.time()
        if current_time - self.stats['last_fps_time'] >= 1.0:
            self.stats['fps_display'] = self.stats['frames_shown'] / (current_time - self.stats['last_fps_time'])
            self.stats['frames_shown'] = 0
            self.stats['last_fps_time'] = current_time
    
    def _handle_key_press(self, key: int) -> bool:
        """Maneja pulsaciones de teclas"""
        if key == 27 or key == ord('q'):  # ESC o Q
            logger.info("Saliendo...")
            return False
        
        elif key == ord(' '):  # SPACE - cambiar cámara
            self._switch_to_next_camera()
        
        elif key == ord('r'):  # R - reiniciar detector
            logger.info("Reiniciando detector...")
            detector = self.system['detector']
            if hasattr(detector, 'reset_history'):
                detector.reset_history()
        
        elif key == ord('s'):  # S - mostrar estadísticas detalladas
            self._print_detailed_stats()
        
        return True
    
    def _switch_to_next_camera(self):
        """Cambia a la siguiente cámara disponible"""
        try:
            available_cameras = self.system['available_cameras']
            current_id = self.system['camera'].current_device_id
            
            # Encontrar siguiente cámara
            camera_ids = [cam.device_id for cam in available_cameras if cam.is_available]
            
            if len(camera_ids) <= 1:
                logger.info("Solo hay una cámara disponible")
                return
            
            current_index = camera_ids.index(current_id) if current_id in camera_ids else 0
            next_index = (current_index + 1) % len(camera_ids)
            next_camera_id = camera_ids[next_index]
            
            logger.info(f"Cambiando de cámara {current_id} → {next_camera_id}")
            
            if self.system['camera'].switch_camera(next_camera_id):
                logger.info(f"✅ Cambio exitoso a cámara {next_camera_id}")
            else:
                logger.error(f"❌ Error cambiando a cámara {next_camera_id}")
        
        except Exception as e:
            logger.error(f"Error cambiando cámara: {e}")
    
    def _print_detailed_stats(self):
        """Imprime estadísticas detalladas en consola"""
        logger.info("📊 ESTADÍSTICAS DETALLADAS:")
        logger.info(f"  Frames mostrados: {self.stats['frames_shown']}")
        logger.info(f"  Detecciones totales: {self.stats['detections_total']}")
        logger.info(f"  Confirmaciones: {self.stats['confirmations_total']}")
        
        detector_stats = self.system['detector'].get_stats()
        logger.info(f"  Detector - Candidatos activos: {detector_stats.get('active_candidates', 0)}")
        logger.info(f"  Detector - FPS estimado: {detector_stats.get('fps_estimate', 0):.1f}")
        
        camera_stats = self.system['camera'].get_performance_stats()
        logger.info(f"  Cámara - FPS actual: {camera_stats.get('current_fps', 0):.1f}")
        logger.info(f"  Cámara - Tiempo promedio frame: {camera_stats.get('avg_frame_time_ms', 0):.1f}ms")
    
    def _cleanup(self):
        """Limpia recursos"""
        self.is_running = False
        
        if self.system:
            if 'camera' in self.system:
                self.system['camera'].release()
        
        cv2.destroyAllWindows()
        logger.info("✅ Recursos liberados")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Xennova Vision - Ultra Live Viewer')
    parser.add_argument('--camera', type=int, default=None, 
                       help='ID de cámara específica (auto-detecta si no se especifica)')
    parser.add_argument('--config', type=str, default=None,
                       help='Archivo de configuración JSON')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = None
    if args.config:
        config = UltraServiceFactory.load_config(args.config)
    
    if not config:
        config = UltraServiceFactory.create_optimized_config(args.camera)
    
    # Crear y ejecutar viewer
    viewer = UltraLiveViewer(config)
    
    if viewer.initialize():
        viewer.run()
    else:
        logger.error("❌ No se pudo inicializar el viewer")

if __name__ == "__main__":
    main()