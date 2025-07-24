"""
Sistema principal de reconocimiento de placas - MEJORADO
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
from services import ServiceFactory
from core.models import SystemStatus
from utils.logger import get_logger
from utils.hardware_detector import HardwareDetector

logger = get_logger(__name__)

class PlateRecognitionSystem:
    """Sistema principal de reconocimiento de placas mejorado"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.service_factory = ServiceFactory(self.config_manager.config)
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
        """Inicializa todos los servicios"""
        try:
            logger.info("Inicializando servicios...")

            # Crear servicios
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
                        break
                except Exception as e:
                    logger.warning(f"Intento {attempt + 1} de inicialización de cámara falló: {e}")
                    time.sleep(2)

            if not camera_initialized:
                logger.error("Error inicializando cámara después de 3 intentos")
                return False

            self.status.camera_active = True

            # Inicializar detector de placas
            if not self.plate_detector.initialize():
                logger.error("Error inicializando detector de placas")
                return False
            self.status.ai_model_loaded = True

            # Verificar conectividad de red
            self.status.network_connected = self.network_service.is_connected()

            logger.info("Todos los servicios inicializados correctamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando servicios: {e}")
            return False

    def process_image(self, image_path: str = None) -> bool:
        """Procesa una imagen específica"""
        try:
            if image_path:
                # Cargar imagen desde archivo
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"No se pudo cargar la imagen: {image_path}")
                    return False
                logger.info(f"Procesando imagen: {image_path}")
            else:
                # Capturar desde cámara con reintentos
                image = None
                for attempt in range(3):
                    try:
                        image = self.camera_service.capture_frame()
                        if image is not None:
                            break
                    except Exception as e:
                        logger.warning(f"Error capturando frame (intento {attempt + 1}): {e}")
                        time.sleep(0.5)

                if image is None:
                    logger.warning("No se pudo capturar frame de la cámara después de 3 intentos")
                    return False
                logger.info("Procesando frame de la cámara")

            # Detectar placas
            detections = self.plate_detector.detect_plates(image)

            # Mostrar imagen con detecciones si está habilitado
            if self.show_window:
                self._display_frame_with_detections(image, detections)

            if detections:
                logger.info(f"Detectadas {len(detections)} placas:")
                for detection in detections:
                    logger.info(f"  - Placa: {detection.plate_number}, Confianza: {detection.confidence:.2f}")

                    # Crear evento desde detección
                    from core.models import PlateEvent
                    event = PlateEvent.from_detection(detection, image_path)

                    # Almacenar evento
                    if self.storage_service.store_event(event):
                        logger.info(f"Evento almacenado: {event.plate_number}")

                        # Intentar enviar inmediatamente
                        if self.status.network_connected:
                            try:
                                if self.network_service.send_event(event.to_dict()):
                                    self.storage_service.mark_as_sent(event.id)
                                    logger.info(f"Evento enviado: {event.plate_number}")
                            except Exception as e:
                                logger.warning(f"Error enviando evento: {e}")

                self.status.last_detection = detections[0].plate_number if detections else None
                return True
            else:
                logger.info("No se detectaron placas en la imagen")
                return True

        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            return False

    def _display_frame_with_detections(self, frame: np.ndarray, detections: list):
        """Muestra el frame con las detecciones dibujadas"""
        display_frame = frame.copy()

        # Dibujar detecciones
        for detection in detections:
            if detection.bbox:
                x1, y1, x2, y2 = detection.bbox

                # Dibujar rectángulo
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Dibujar texto
                label = f"{detection.plate_number} ({detection.confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

                # Fondo para el texto
                cv2.rectangle(display_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            (0, 255, 0), -1)

                # Texto
                cv2.putText(display_frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Información del sistema
        info_text = f"Placas detectadas: {len(detections)} | Presiona 'q' para salir"
        cv2.putText(display_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar frame
        cv2.imshow('Xennova Vision - Detección de Placas', display_frame)

        # Procesar eventos de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.is_running = False

    def run_continuous(self, show_window: bool = False):
        """Ejecuta el sistema en modo continuo"""
        logger.info("Iniciando modo continuo...")
        self.is_running = True
        self.show_window = show_window

        # Crear ventana si es necesario
        if self.show_window:
            cv2.namedWindow('Xennova Vision - Detección de Placas', cv2.WINDOW_AUTOSIZE)

        try:
            frame_count = 0
            last_process_time = time.time()

            while self.is_running:
                try:
                    current_time = time.time()

                    # Procesar frame cada cierto intervalo
                    if current_time - last_process_time >= 0.5:  # 2 FPS para procesamiento
                        self.process_image()
                        last_process_time = current_time
                        frame_count += 1

                        if frame_count % 100 == 0:
                            logger.info(f"Procesados {frame_count} frames")

                    # Pequeña pausa para evitar sobrecargar el sistema
                    time.sleep(0.1)

                except KeyboardInterrupt:
                    logger.info("Deteniendo sistema...")
                    break
                except Exception as e:
                    logger.error(f"Error en loop principal: {e}")
                    time.sleep(1)  # Esperar antes de continuar

        finally:
            self.is_running = False
            if self.show_window:
                cv2.destroyAllWindows()
            self.shutdown()

    def sync_pending_events(self):
        """Sincroniza eventos pendientes"""
        try:
            pending_events = self.storage_service.get_pending_events()
            if not pending_events:
                logger.info("No hay eventos pendientes")
                return

            logger.info(f"Sincronizando {len(pending_events)} eventos pendientes...")

            for event in pending_events:
                try:
                    if self.network_service.send_event(event.to_dict()):
                        self.storage_service.mark_as_sent(event.id)
                        logger.info(f"Evento sincronizado: {event.plate_number}")
                    else:
                        logger.warning(f"Error sincronizando evento: {event.plate_number}")
                except Exception as e:
                    logger.error(f"Error enviando evento {event.plate_number}: {e}")

            # Actualizar eventos pendientes
            remaining = len(self.storage_service.get_pending_events())
            logger.info(f"Eventos pendientes restantes: {remaining}")

        except Exception as e:
            logger.error(f"Error sincronizando eventos: {e}")

    def show_status(self):
        """Muestra el estado del sistema"""
        hardware_detector = HardwareDetector()

        print("\n" + "="*50)
        print("🚗 SISTEMA DE RECONOCIMIENTO DE PLACAS")
        print("="*50)
        print(f"Hardware: {hardware_detector.get_hardware_type().upper()}")
        print(f"Cámara activa: {'✅' if self.status.camera_active else '❌'}")
        print(f"IA cargada: {'✅' if self.status.ai_model_loaded else '❌'}")
        print(f"Conectado: {'✅' if self.status.network_connected else '❌'}")

        try:
            pending = len(self.storage_service.get_pending_events()) if self.storage_service else 0
            print(f"Eventos pendientes: {pending}")
        except:
            print("Eventos pendientes: Error obteniendo datos")

        print("="*50)

    def shutdown(self):
        """Cierra el sistema limpiamente"""
        logger.info("Cerrando sistema...")

        if self.camera_service:
            self.camera_service.release()

        logger.info("Sistema cerrado correctamente")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Sistema de reconocimiento de placas")
    parser.add_argument('--image', '-i', help='Procesar imagen específica')
    parser.add_argument('--continuous', '-c', action='store_true', help='Modo continuo')
    parser.add_argument('--window', '-w', action='store_true', help='Mostrar ventana de visualización')
    parser.add_argument('--sync', '-s', action='store_true', help='Sincronizar eventos pendientes')
    parser.add_argument('--status', action='store_true', help='Mostrar estado del sistema')

    args = parser.parse_args()

    # Crear sistema
    system = PlateRecognitionSystem()

    try:
        # Inicializar servicios
        if not system.initialize_services():
            logger.error("Error inicializando sistema")
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
                cv2.namedWindow('Xennova Vision - Detección de Placas', cv2.WINDOW_AUTOSIZE)
            system.process_image(args.image)
            if args.window:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif args.continuous:
            system.run_continuous(show_window=args.window)
        else:
            # Modo por defecto: procesar una imagen de la cámara
            logger.info("Capturando y procesando una imagen...")
            if args.window:
                system.show_window = True
                cv2.namedWindow('Xennova Vision - Detección de Placas', cv2.WINDOW_AUTOSIZE)
            system.process_image()
            if args.window:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()
