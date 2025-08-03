import cv2
import time
import threading
from typing import Dict, Any, Optional
import numpy as np
from services.factory_service import UltraServiceFactory

class UltraRealTimeViewer:
    """
    Visualizador en tiempo real ultra-optimizado:
    - Ventana de visualización con estadísticas
    - Overlay de información en tiempo real
    - Controles de teclado
    - Grabación de video opcional
    - Dashboard de rendimiento
    """
    
    def __init__(self, system: Dict[str, Any]):
        self.system = system
        self.camera = system['camera']
        self.detector = system['detector']
        self.storage = system['storage']
        self.network = system['network']
        self.config = system['config']
        
        # Estado del visualizador
        self.running = False
        self.paused = False
        self.show_stats = True
        self.show_debug = False
        self.recording = False
        
        # Configuración de ventana
        display_config = self.config.get('display', {})
        self.window_size = display_config.get('window_size', [1024, 768])
        self.show_confidence = display_config.get('show_confidence', True)
        self.show_processing_time = display_config.get('show_processing_time', True)
        
        # Video writer para grabación
        self.video_writer = None
        self.recording_path = None
        
        # Estadísticas en tiempo real
        self.stats = {
            'frames_displayed': 0,
            'total_detections': 0,
            'confirmed_detections': 0,
            'start_time': time.time(),
            'last_detection_time': 0,
            'fps_display': 0.0
        }
        
        # Buffer para FPS suavizado
        self.fps_history = []
        self.fps_history_size = 10
        
        # Colores para UI
        self.colors = {
            'detection_box': (0, 255, 0),      # Verde para detecciones
            'confirmed_box': (0, 255, 255),    # Amarillo para confirmadas
            'text_bg': (0, 0, 0),              # Negro para fondo de texto
            'text_fg': (255, 255, 255),        # Blanco para texto
            'stats_bg': (50, 50, 50),          # Gris para panel de stats
            'warning': (0, 165, 255),          # Naranja para advertencias
            'error': (0, 0, 255)               # Rojo para errores
        }

    def start(self):
        """Inicia el visualizador en tiempo real"""
        print("🖥️ Iniciando visualizador en tiempo real...")
        
        # Crear ventana
        cv2.namedWindow('Xennova Vision - Ultra Real Time', cv2.WINDOW_RESIZABLE)
        cv2.resizeWindow('Xennova Vision - Ultra Real Time', self.window_size[0], self.window_size[1])
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        print("✅ Visualizador iniciado")
        print("🎮 Controles:")
        print("  ESPACIO: Pausar/Reanudar")
        print("  S: Mostrar/Ocultar estadísticas")
        print("  D: Mostrar/Ocultar debug")
        print("  R: Iniciar/Detener grabación")
        print("  C: Capturar screenshot")
        print("  ESC/Q: Salir")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n⚠️ Interrupción por teclado")
        finally:
            self.stop()

    def _main_loop(self):
        """Loop principal del visualizador"""
        last_fps_time = time.time()
        fps_counter = 0
        
        while self.running:
            try:
                if not self.paused:
                    # Capturar frame
                    frame = self.camera.capture_frame()
                    
                    if frame is not None:
                        # Procesar detecciones
                        detections = self.detector.detect_plates(frame)
                        
                        # Crear frame de visualización
                        display_frame = self._create_display_frame(frame, detections)
                        
                        # Mostrar frame
                        cv2.imshow('Xennova Vision - Ultra Real Time', display_frame)
                        
                        # Grabar si está habilitado
                        if self.recording and self.video_writer:
                            self.video_writer.write(display_frame)
                        
                        # Actualizar estadísticas
                        self.stats['frames_displayed'] += 1
                        self.stats['total_detections'] += len(detections)
                        
                        for detection in detections:
                            if detection.confidence > 0.9:
                                self.stats['confirmed_detections'] += 1
                                self.stats['last_detection_time'] = time.time()
                        
                        fps_counter += 1
                    
                    else:
                        # Mostrar frame de error si no hay imagen
                        error_frame = self._create_error_frame("No hay señal de cámara")
                        cv2.imshow('Xennova Vision - Ultra Real Time', error_frame)
                
                else:
                    # Mostrar frame de pausa
                    pause_frame = self._create_pause_frame()
                    cv2.imshow('Xennova Vision - Ultra Real Time', pause_frame)
                
                # Calcular FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.stats['fps_display'] = fps_counter / (current_time - last_fps_time)
                    self.fps_history.append(self.stats['fps_display'])
                    if len(self.fps_history) > self.fps_history_size:
                        self.fps_history.pop(0)
                    
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Manejar eventos de teclado
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard(key):
                    break
                    
            except Exception as e:
                print(f"❌ Error en loop principal: {e}")
                time.sleep(0.1)

    def _create_display_frame(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Crea frame de visualización con overlays"""
        display_frame = frame.copy()
        
        # Dibujar detecciones
        for detection in detections:
            self._draw_detection(display_frame, detection)
        
        # Dibujar panel de estadísticas
        if self.show_stats:
            self._draw_stats_panel(display_frame)
        
        # Dibujar información de debug
        if self.show_debug:
            self._draw_debug_info(display_frame, detections)
        
        # Dibujar indicadores de estado
        self._draw_status_indicators(display_frame)
        
        return display_frame

    def _draw_detection(self, frame: np.ndarray, detection):
        """Dibuja una detección en el frame"""
        x1, y1, x2, y2 = detection.bbox
        
        # Color según confianza
        if detection.confidence > 0.9:
            color = self.colors['confirmed_box']
            thickness = 3
        else:
            color = self.colors['detection_box']
            thickness = 2
        
        # Dibujar rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Preparar texto
        if self.show_confidence:
            text = f"{detection.plate_number} ({detection.confidence:.2f})"
        else:
            text = detection.plate_number
        
        # Calcular tamaño del texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        
        # Dibujar fondo del texto
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), 
                     self.colors['text_bg'], -1)
        
        # Dibujar texto
        cv2.putText(frame, text, (x1 + 5, y1 - 5), 
                   font, font_scale, self.colors['text_fg'], text_thickness)

    def _draw_stats_panel(self, frame: np.ndarray):
        """Dibuja panel de estadísticas"""
        h, w = frame.shape[:2]
        panel_width = 300
        panel_height = 200
        
        # Posición del panel (esquina superior derecha)
        x = w - panel_width - 10
        y = 10
        
        # Dibujar fondo del panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), 
                     self.colors['stats_bg'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Preparar estadísticas
        uptime = time.time() - self.stats['start_time']
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        camera_stats = self.camera.get_stats()
        detector_stats = self.detector.get_stats()
        
        stats_text = [
            f"XENNOVA VISION ULTRA",
            f"Tiempo: {uptime:.0f}s",
            f"FPS: {avg_fps:.1f}",
            f"Frames: {self.stats['frames_displayed']}",
            f"Detecciones: {self.stats['total_detections']}",
            f"Confirmadas: {self.stats['confirmed_detections']}",
            f"Cámara: {camera_stats.get('current_device', 'N/A')}",
            f"Resolución: {camera_stats.get('resolution', 'N/A')}",
            f"Errores: {camera_stats.get('errors', 0)}"
        ]
        
        # Dibujar texto de estadísticas
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        
        for i, text in enumerate(stats_text):
            y_pos = y + 20 + (i * line_height)
            cv2.putText(frame, text, (x + 10, y_pos), 
                       font, font_scale, self.colors['text_fg'], 1)

    def _draw_debug_info(self, frame: np.ndarray, detections: list):
        """Dibuja información de debug"""
        h, w = frame.shape[:2]
        
        # Panel de debug en la esquina inferior izquierda
        debug_info = [
            f"Debug Mode: ON",
            f"Processing Time: {self.detector.get_stats().get('avg_processing_time', 0):.3f}s",
            f"Detection Rate: {self.detector.get_stats().get('detection_rate', 0):.2f}",
            f"Confirmation Rate: {self.detector.get_stats().get('confirmation_rate', 0):.2f}",
            f"False Positives Filtered: {self.detector.get_stats().get('false_positives_filtered', 0)}"
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        line_height = 15
        
        for i, text in enumerate(debug_info):
            y_pos = h - 100 + (i * line_height)
            cv2.putText(frame, text, (10, y_pos), 
                       font, font_scale, self.colors['warning'], 1)

    def _draw_status_indicators(self, frame: np.ndarray):
        """Dibuja indicadores de estado"""
        h, w = frame.shape[:2]
        
        # Indicador de grabación
        if self.recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 50, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Indicador de pausa
        if self.paused:
            cv2.putText(frame, "PAUSADO", (w//2 - 50, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['warning'], 2)
        
        # Indicador de última detección
        time_since_detection = time.time() - self.stats['last_detection_time']
        if time_since_detection < 5.0:  # Mostrar por 5 segundos
            alpha = max(0, 1 - (time_since_detection / 5.0))
            color = tuple(int(c * alpha) for c in self.colors['confirmed_box'])
            cv2.putText(frame, "PLACA DETECTADA!", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _create_error_frame(self, message: str) -> np.ndarray:
        """Crea frame de error"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dibujar mensaje de error
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Calcular posición centrada
        (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
        x = (640 - text_width) // 2
        y = (480 + text_height) // 2
        
        cv2.putText(frame, message, (x, y), font, font_scale, self.colors['error'], thickness)
        
        return frame

    def _create_pause_frame(self) -> np.ndarray:
        """Crea frame de pausa"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dibujar símbolo de pausa
        cv2.rectangle(frame, (280, 200), (310, 280), self.colors['text_fg'], -1)
        cv2.rectangle(frame, (330, 200), (360, 280), self.colors['text_fg'], -1)
        
        # Texto de pausa
        cv2.putText(frame, "PAUSADO", (250, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text_fg'], 2)
        
        return frame

    def _handle_keyboard(self, key: int) -> bool:
        """Maneja eventos de teclado"""
        if key == 27 or key == ord('q'):  # ESC o Q
            return False
        
        elif key == ord(' '):  # ESPACIO
            self.paused = not self.paused
            print(f"{'⏸️ Pausado' if self.paused else '▶️ Reanudado'}")
        
        elif key == ord('s'):  # S
            self.show_stats = not self.show_stats
            print(f"📊 Estadísticas: {'ON' if self.show_stats else 'OFF'}")
        
        elif key == ord('d'):  # D
            self.show_debug = not self.show_debug
            print(f"🐛 Debug: {'ON' if self.show_debug else 'OFF'}")
        
        elif key == ord('r'):  # R
            self._toggle_recording()
        
        elif key == ord('c'):  # C
            self._capture_screenshot()
        
        elif key == ord('h'):  # H
            self._show_help()
        
        return True

    def _toggle_recording(self):
        """Alterna grabación de video"""
        if not self.recording:
            # Iniciar grabación
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.recording_path = f"recording_{timestamp}.mp4"
            
            # Configurar video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20.0
            frame_size = tuple(self.window_size)
            
            self.video_writer = cv2.VideoWriter(self.recording_path, fourcc, fps, frame_size)
            
            if self.video_writer.isOpened():
                self.recording = True
                print(f"🎬 Grabación iniciada: {self.recording_path}")
            else:
                print("❌ Error iniciando grabación")
                self.video_writer = None
        else:
            # Detener grabación
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.recording = False
            print(f"⏹️ Grabación detenida: {self.recording_path}")

    def _capture_screenshot(self):
        """Captura screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = f"screenshot_{timestamp}.jpg"
        
        # Capturar frame actual
        frame = self.camera.capture_frame()
        if frame is not None:
            detections = self.detector.detect_plates(frame)
            display_frame = self._create_display_frame(frame, detections)
            
            if cv2.imwrite(screenshot_path, display_frame):
                print(f"📸 Screenshot guardado: {screenshot_path}")
            else:
                print("❌ Error guardando screenshot")

    def _show_help(self):
        """Muestra ayuda en consola"""
        print("\n🎮 CONTROLES DISPONIBLES:")
        print("  ESPACIO: Pausar/Reanudar")
        print("  S: Mostrar/Ocultar estadísticas")
        print("  D: Mostrar/Ocultar debug")
        print("  R: Iniciar/Detener grabación")
        print("  C: Capturar screenshot")
        print("  H: Mostrar esta ayuda")
        print("  ESC/Q: Salir")

    def stop(self):
        """Detiene el visualizador"""
        print("🔄 Deteniendo visualizador...")
        
        self.running = False
        
        # Detener grabación si está activa
        if self.recording and self.video_writer:
            self.video_writer.release()
            print(f"💾 Grabación guardada: {self.recording_path}")
        
        # Cerrar ventanas
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        uptime = time.time() - self.stats['start_time']
        print(f"📊 Estadísticas finales:")
        print(f"  Tiempo total: {uptime:.1f}s")
        print(f"  Frames mostrados: {self.stats['frames_displayed']}")
        print(f"  Total detecciones: {self.stats['total_detections']}")
        print(f"  Detecciones confirmadas: {self.stats['confirmed_detections']}")
        print(f"  FPS promedio: {self.stats['frames_displayed']/uptime:.1f}")
        
        print("✅ Visualizador detenido")

def main():
    """Función principal para ejecutar el visualizador"""
    print("🚀 Iniciando Xennova Vision Ultra Real Time Viewer...")
    
    # Crear sistema completo
    system = UltraServiceFactory.create_all_services()
    
    if not system:
        print("❌ Error: No se pudo inicializar el sistema")
        return
    
    # Crear y ejecutar visualizador
    viewer = UltraRealTimeViewer(system)
    
    try:
        viewer.start()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupción por usuario")
    finally:
        # Limpiar recursos
        system['camera'].release()
        system['detector'].cleanup()
        print("🧹 Recursos liberados")

if __name__ == "__main__":
    main()