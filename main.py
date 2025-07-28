"""
Sistema principal de reconocimiento de placas - ULTRA MEJORADO
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import threading
from queue import Queue

from config.settings import ConfigManager
from services import UltraServiceFactory  # ← CORREGIDO: Usar UltraServiceFactory
from core.models import SystemStatus
from utils.logger import get_logger
from utils.hardware_detector import HardwareDetector

logger = get_logger(__name__)

class PlateRecognitionSystem:
    """Sistema principal de reconocimiento de placas ultra-mejorado"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.service_factory = UltraServiceFactory(self.config_manager.config)  # ← CORREGIDO
        self.is_running = False
        self.status = SystemStatus()
        self.show_window = False

        # Cola para frames y detecciones
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=100)

        # Inicializar servicios
        self.camera_service = None
        self.plate_detector = None
        self.storage_service = None
        self.network_service = None

    def initialize_services(self) -> bool:
        """Inicializa todos los servicios ultra-optimizados"""
        try:
            logger.info("🚀 Inicializando servicios ultra-optimizados...")

            # Crear servicios ultra
            self.camera_service = self.service_factory.create_camera_service()
            self.plate_detector = self.service_factory.create_plate_detector()
            self.storage_service = self.service_factory.create_storage_service()
            self.network_service = self.service_factory.create_network_service()

            # Inicializar cámara con reintentos
            camera_initialized = False
            for attempt in range(3):
                try:
                    if self.camera_service.initialize():
                        camera_initialized = True
                        logger.info("✅ Cámara inicializada correctamente")
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Intento {attempt + 1} de inicialización de cámara falló: {e}")
                    time.sleep(2)

            if not camera_initialized:
                logger.error("❌ Error inicializando cámara después de 3 intentos")
                return False

            self.status.camera_active = True

            # Inicializar detector de placas ultra
            if not self.plate_detector.initialize():
                logger.error("❌ Error inicializando detector ultra de placas")
                return False
            self.status.ai_model_loaded = True
            logger.info("✅ Detector ultra de placas inicializado")

            # Verificar conectividad de red
            self.status.network_connected = self.network_service.is_connected()
            logger.info(f"🌐 Red: {'Conectada' if self.status.network_connected else 'Desconectada'}")

            logger.info("🎉 Todos los servicios ultra inicializados correctamente")
            return True

        except Exception as e:
            logger.error(f"❌ Error inicializando servicios: {e}")
            return False

    def process_image(self, image_path: str = None) -> bool:
        """Procesa una imagen específica con detección ultra-optimizada"""
        try:
            if image_path:
                # Cargar imagen desde archivo
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"❌ No se pudo cargar la imagen: {image_path}")
                    return False
                logger.info(f"📷 Procesando imagen: {image_path}")
            else:
                # Capturar desde cámara con reintentos
                image = None
                for attempt in range(3):
                    try:
                        image = self.camera_service.capture_frame()
                        if image is not None:
                            break
                    except Exception as e:
                        logger.warning(f"⚠️ Error capturando frame (intento {attempt + 1}): {e}")
                        time.sleep(0.5)

                if image is None:
                    logger.warning("⚠️ No se pudo capturar frame de la cámara después de 3 intentos")
                    return False
                logger.info("📷 Procesando frame de la cámara")

            # Detectar placas con detector ultra
            detections = self.plate_detector.detect_plates(image)

            # Mostrar imagen con detecciones si está habilitado
            if self.show_window:
                self._display_frame_with_detections(image, detections)

            if detections:
                logger.info(f"🎯 Detectadas {len(detections)} placas:")
                for detection in detections:
                    logger.info(f"  - Placa: {detection.plate_number}, Confianza: {detection.confidence:.2f}")

                # Crear evento desde detección
                from core.models import PlateEvent
                event = PlateEvent.from_detection(detections[0], image_path)  # Usar la mejor detección

                # Almacenar evento
                if self.storage_service.store_event(event):
                    logger.info(f"💾 Evento almacenado: {event.plate_number}")

                # Intentar enviar inmediatamente
                if self.status.network_connected:
                    try:
                        if self.network_service.send_event(event.to_dict()):
                            self.storage_service.mark_as_sent(event.id)
                            logger.info(f"📤 Evento enviado: {event.plate_number}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error enviando evento: {e}")

                self.status.last_detection = detections[0].plate_number if detections else None
                return True
            else:
                logger.info("ℹ️ No se detectaron placas en la imagen")
                return True

        except Exception as e:
            logger.error(f"❌ Error procesando imagen: {e}")
            return False

    def _display_frame_with_detections(self, frame: np.ndarray, detections: list):
        """Muestra el frame con las detecciones dibujadas - ULTRA VERSION"""
        display_frame = frame.copy()

        # Información de rendimiento
        fps_text = f"ULTRA MODE | FPS: {30:.1f}"
        cv2.putText(display_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Dibujar detecciones
        for i, detection in enumerate(detections):
            if detection.bbox:
                x1, y1, x2, y2 = detection.bbox

                # Color según confianza
                confidence = detection.confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Verde - Alta confianza
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Amarillo - Media confianza
                else:
                    color = (0, 0, 255)  # Rojo - Baja confianza

                # Dibujar rectángulo con grosor según confianza
                thickness = int(confidence * 4) + 1
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                # Dibujar texto mejorado
                label = f"{detection.plate_number} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                # Fondo para el texto
                cv2.rectangle(display_frame, 
                             (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), 
                             color, -1)

                # Texto
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Número de detección
                cv2.putText(display_frame, f"#{i+1}", (x1-20, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Información del sistema en la parte inferior
        info_lines = [
            f"Detecciones: {len(detections)}",
            f"Modo: ULTRA",
            f"Cámara: {'OK' if self.status.camera_active else 'ERROR'}",
            f"Red: {'OK' if self.status.network_connected else 'OFFLINE'}",
            "Presiona 'q' para salir"
        ]

        y_offset = display_frame.shape[0] - len(info_lines) * 25 - 10
        for i, line in enumerate(info_lines):
            y_pos = y_offset + i * 25
            cv2.putText(display_frame, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostrar frame
        cv2.imshow('Xennova Vision ULTRA - Detección de Placas', display_frame)

        # Procesar eventos de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.is_running = False
        elif key == ord('s'):  # Screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"detection_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display_frame)
            logger.info(f"📸 Screenshot guardado: {filename}")

    def run_continuous(self, show_window: bool = False):
        """Ejecuta el sistema en modo continuo ultra-optimizado"""
        logger.info("🚀 Iniciando modo continuo ULTRA...")
        self.is_running = True
        self.show_window = show_window

        # Crear ventana si es necesario
        if self.show_window:
            cv2.namedWindow('Xennova Vision ULTRA - Detección de Placas', cv2.WINDOW_NORMAL)

        try:
            frame_count = 0
            start_time = time.time()
            last_process_time = time.time()
            fps_counter = 0
            fps_start = time.time()

            while self.is_running:
                try:
                    current_time = time.time()

                    # Procesar frame con intervalo optimizado
                    if current_time - last_process_time >= 0.33:  # ~3 FPS para procesamiento
                        self.process_image()
                        last_process_time = current_time
                        frame_count += 1
                        fps_counter += 1

                    # Mostrar estadísticas cada 30 frames
                    if frame_count % 30 == 0 and frame_count > 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        logger.info(f"📊 Procesados {frame_count} frames | FPS: {fps:.1f}")

                    # Pequeña pausa optimizada
                    time.sleep(0.05)

                except KeyboardInterrupt:
                    logger.info("⚠️ Deteniendo sistema...")
                    break
                except Exception as e:
                    logger.error(f"❌ Error en loop principal: {e}")
                    time.sleep(1)

        finally:
            self.is_running = False
            if self.show_window:
                cv2.destroyAllWindows()
            
            # Estadísticas finales
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(f"📊 Sesión finalizada: {frame_count} frames en {total_time:.1f}s (FPS promedio: {avg_fps:.1f})")
            
            self.shutdown()

    def sync_pending_events(self):
        """Sincroniza eventos pendientes con red"""
        try:
            pending_events = self.storage_service.get_pending_events()
            if not pending_events:
                logger.info("ℹ️ No hay eventos pendientes")
                return

            logger.info(f"🔄 Sincronizando {len(pending_events)} eventos pendientes...")

            success_count = 0
            for event in pending_events:
                try:
                    if self.network_service.send_event(event.to_dict()):
                        self.storage_service.mark_as_sent(event.id)
                        logger.info(f"✅ Evento sincronizado: {event.plate_number}")
                        success_count += 1
                    else:
                        logger.warning(f"❌ Error sincronizando evento: {event.plate_number}")
                except Exception as e:
                    logger.error(f"❌ Error enviando evento {event.plate_number}: {e}")

            # Actualizar eventos pendientes
            remaining = len(self.storage_service.get_pending_events())
            logger.info(f"📊 Sincronizados: {success_count}/{len(pending_events)} | Pendientes: {remaining}")

        except Exception as e:
            logger.error(f"❌ Error sincronizando eventos: {e}")

    def show_status(self):
        """Muestra el estado del sistema ultra-mejorado"""
        hardware_detector = HardwareDetector()

        print("\n" + "="*60)
        print("🚗 XENNOVA VISION - SISTEMA ULTRA DE RECONOCIMIENTO DE PLACAS")
        print("="*60)
        print(f"🖥️  Hardware: {hardware_detector.get_hardware_type().upper()}")
        print(f"📷 Cámara: {'✅ ACTIVA' if self.status.camera_active else '❌ INACTIVA'}")
        print(f"🤖 IA Ultra: {'✅ CARGADA' if self.status.ai_model_loaded else '❌ ERROR'}")
        print(f"🌐 Red: {'✅ CONECTADA' if self.status.network_connected else '❌ DESCONECTADA'}")

        try:
            pending = len(self.storage_service.get_pending_events()) if self.storage_service else 0
            print(f"📤 Eventos pendientes: {pending}")
        except:
            print("📤 Eventos pendientes: Error obteniendo datos")

        if self.status.last_detection:
            print(f"🎯 Última detección: {self.status.last_detection}")

        print("="*60)
        print("💡 Comandos disponibles:")
        print("  python main.py --continuous --window  # Modo continuo con ventana")
        print("  python main.py --image imagen.jpg     # Procesar imagen específica")
        print("  python main.py --sync                 # Sincronizar eventos")
        print("="*60)

    def shutdown(self):
        """Cierra el sistema limpiamente"""
        logger.info("🔄 Cerrando sistema ultra...")

        if self.camera_service:
            self.camera_service.release()
            logger.info("✅ Cámara liberada")

        logger.info("✅ Sistema ultra cerrado correctamente")

def main():
    """Función principal mejorada"""
    parser = argparse.ArgumentParser(
        description="Xennova Vision - Sistema Ultra de Reconocimiento de Placas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                           # Capturar y procesar una imagen
  python main.py --continuous              # Modo continuo
  python main.py --continuous --window     # Modo continuo con visualización
  python main.py --image foto.jpg          # Procesar imagen específica
  python main.py --sync                    # Sincronizar eventos pendientes
  python main.py --status                  # Mostrar estado del sistema
        """
    )
    
    parser.add_argument('--image', '-i', help='Procesar imagen específica')
    parser.add_argument('--continuous', '-c', action='store_true', help='Modo continuo')
    parser.add_argument('--window', '-w', action='store_true', help='Mostrar ventana de visualización')
    parser.add_argument('--sync', '-s', action='store_true', help='Sincronizar eventos pendientes')
    parser.add_argument('--status', action='store_true', help='Mostrar estado del sistema')

    args = parser.parse_args()

    # Crear sistema ultra
    system = PlateRecognitionSystem()

    try:
        # Inicializar servicios ultra
        if not system.initialize_services():
            logger.error("❌ Error inicializando sistema ultra")
            sys.exit(1)

        # Mostrar estado
        system.show_status()

        if args.status:
            return
        elif args.sync:
            system.sync_pending_events()
        elif args.image:
            if args.window:
                system.show_window = True
                cv2.namedWindow('Xennova Vision ULTRA - Detección de Placas', cv2.WINDOW_NORMAL)
            system.process_image(args.image)
            if args.window:
                print("\n💡 Presiona cualquier tecla para cerrar la ventana...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif args.continuous:
            system.run_continuous(show_window=args.window)
        else:
            # Modo por defecto: procesar una imagen de la cámara
            logger.info("📷 Capturando y procesando una imagen...")
            if args.window:
                system.show_window = True
                cv2.namedWindow('Xennova Vision ULTRA - Detección de Placas', cv2.WINDOW_NORMAL)
            system.process_image()
            if args.window:
                print("\n💡 Presiona cualquier tecla para cerrar la ventana...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except KeyboardInterrupt:
        logger.info("⚠️ Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()