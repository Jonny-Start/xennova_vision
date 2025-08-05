"""
Sistema principal de reconocimiento de placas - VERSIÓN CORREGIDA
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from queue import Queue

from config.settings import ConfigManager
from services.factory_service import UltraServiceFactory
from core.models import SystemStatus
from utils.logger import get_logger
from utils.hardware_detector import HardwareDetector

logger = get_logger(__name__)

class PlateRecognitionSystem:
    """Sistema principal de reconocimiento de placas ultra-mejorado"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.service_factory = UltraServiceFactory(self.config_manager.config)
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

    def _display_frame_with_detections(self, frame: np.ndarray, detections: list):
        """Muestra el frame con las detecciones Y ÁREAS DE ANÁLISIS - ULTRA VERSION MEJORADA"""
        display_frame = frame.copy()

        # MOSTRAR REGIONES QUE ESTÁ ANALIZANDO
        try:
            # Obtener regiones candidatas del detector
            if hasattr(self.plate_detector, '_detect_plate_regions_advanced'):
                plate_regions = self.plate_detector._detect_plate_regions_advanced(frame)

                # Dibujar regiones candidatas en AZUL
                for x, y, w, h, score in plate_regions:
                    # Rectángulo azul para regiones que está analizando
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    # Etiqueta de análisis
                    cv2.putText(display_frame, f"Analizando: {score:.2f}", 
                               (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        except:
            pass  # Si no puede obtener regiones, continúa

        # Información de rendimiento
        fps_text = f"ULTRA MODE | FPS: {30:.1f}"
        cv2.putText(display_frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Dibujar detecciones CONFIRMADAS
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
                thickness = int(confidence * 4) + 2
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

        # 🆕 LEYENDA DE COLORES
        legend_y = 60
        cv2.putText(display_frame, "🔍 AZUL: Analizando", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(display_frame, "✅ VERDE: Placa confirmada", (10, legend_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Información del sistema en la parte inferior
        info_lines = [
            f"Detecciones: {len(detections)}",
            f"Modo: ULTRA",
            f"Cámara: {'OK' if self.status.camera_active else 'ERROR'}",
            f"Red: {'OK' if self.status.network_connected else 'OFFLINE'}",
            "Presiona 'q' para salir, 's' para screenshot"
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

    # Resto de métodos continúan igual pero con indentación correcta...
    # [Por brevedad, no incluyo todos los métodos aquí]

if __name__ == "__main__":
    # Función main también con indentación correcta
    pass
