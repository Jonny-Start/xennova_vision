"""
Sistema principal de reconocimiento de placas
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from config.settings import ConfigManager
from services import ServiceFactory
from core.models import SystemStatus
from utils.logger import get_logger
from utils.hardware_detector import HardwareDetector

logger = get_logger(__name__)

class PlateRecognitionSystem:
    """Sistema principal de reconocimiento de placas"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.service_factory = ServiceFactory(self.config_manager.config)
        self.is_running = False
        self.status = SystemStatus()
        
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
            
            # Inicializar cámara
            if not self.camera_service.initialize():
                logger.error("Error inicializando cámara")
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
                # Capturar desde cámara
                image = self.camera_service.capture_frame()
                if image is None:
                    logger.warning("No se pudo capturar frame de la cámara")
                    return False
                logger.info("Procesando frame de la cámara")
            
            # Detectar placas
            detections = self.plate_detector.detect_plates(image)
            
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
    
    def run_continuous(self):
        """Ejecuta el sistema en modo continuo"""
        logger.info("Iniciando modo continuo...")
        self.is_running = True
        
        try:
            frame_count = 0
            while self.is_running:
                try:
                    # Procesar frame
                    self.process_image()
                    
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
            system.process_image(args.image)
        elif args.continuous:
            system.run_continuous()
        else:
            # Modo por defecto: procesar una imagen de la cámara
            logger.info("Capturando y procesando una imagen...")
            system.process_image()
            
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()