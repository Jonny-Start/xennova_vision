"""
Adaptadores para diferentes tipos de cámaras - MEJORADO
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image
import time
import threading
from queue import Queue, Empty

from core.interfaces import ICameraService
from core.exceptions import CameraError
from utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedOpenCVCameraAdapter(ICameraService):
    """Adaptador mejorado para cámaras usando OpenCV con manejo robusto de errores"""

    def __init__(self, device_id: int = 0, backup_devices: List[int] = None):
        self.device_id = device_id
        self.backup_devices = backup_devices or [1, 2]  # Dispositivos de respaldo
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
        self.last_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.max_errors = 10

        # Buffer de frames para suavizar la captura
        self.frame_buffer = Queue(maxsize=5)
        self.capture_thread = None
        self.stop_capture = False

    def initialize(self) -> bool:
        """Inicializa la cámara con reintentos y dispositivos de respaldo"""
        devices_to_try = [self.device_id] + self.backup_devices

        for device_id in devices_to_try:
            try:
                logger.info(f"Intentando inicializar cámara {device_id}...")

                # Probar diferentes backends
                backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]

                for backend in backends:
                    try:
                        self.cap = cv2.VideoCapture(device_id, backend)

                        if not self.cap.isOpened():
                            continue

                        # Configurar propiedades de la cámara
                        self._configure_camera()

                        # Probar captura
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.device_id = device_id
                            self.is_active = True
                            self.last_frame = frame

                            # Iniciar hilo de captura
                            self._start_capture_thread()

                            logger.info(f"Cámara {device_id} inicializada correctamente con backend {backend}")
                            return True

                    except Exception as e:
                        logger.warning(f"Error con backend {backend}: {e}")
                        if self.cap:
                            self.cap.release()
                        continue

            except Exception as e:
                logger.warning(f"Error inicializando cámara {device_id}: {e}")
                continue

        raise CameraError("No se pudo inicializar ninguna cámara disponible")

    def _configure_camera(self):
        """Configura las propiedades de la cámara"""
        if not self.cap:
            return

        # Configuraciones básicas
        configurations = [
            (cv2.CAP_PROP_FRAME_WIDTH, 1280),
            (cv2.CAP_PROP_FRAME_HEIGHT, 720),
            (cv2.CAP_PROP_FPS, 30),
            (cv2.CAP_PROP_BUFFERSIZE, 1),  # Reducir buffer para menor latencia
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),  # Desactivar auto exposición
        ]

        for prop, value in configurations:
            try:
                self.cap.set(prop, value)
            except:
                pass  # Ignorar errores de configuración no críticos

    def _start_capture_thread(self):
        """Inicia el hilo de captura continua"""
        self.stop_capture = False
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        """Loop de captura en hilo separado"""
        consecutive_errors = 0

        while not self.stop_capture and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Resetear contador de errores
                    consecutive_errors = 0
                    self.error_count = 0

                    # Agregar frame al buffer (no bloqueante)
                    try:
                        if self.frame_buffer.full():
                            # Remover frame más antiguo
                            try:
                                self.frame_buffer.get_nowait()
                            except Empty:
                                pass

                        self.frame_buffer.put_nowait(frame)
                        self.last_frame = frame
                        self.frame_count += 1

                    except:
                        pass  # Buffer lleno, continuar

                else:
                    consecutive_errors += 1
                    self.error_count += 1

                    if consecutive_errors > 5:
                        logger.warning(f"Múltiples errores de captura consecutivos: {consecutive_errors}")
                        time.sleep(0.1)

                    if self.error_count > self.max_errors:
                        logger.error("Demasiados errores de captura, reiniciando cámara...")
                        self._reinitialize_camera()
                        consecutive_errors = 0

                time.sleep(0.01)  # Pequeña pausa para no saturar CPU

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error en loop de captura: {e}")
                time.sleep(0.1)

    def _reinitialize_camera(self):
        """Reinicializa la cámara en caso de errores"""
        try:
            if self.cap:
                self.cap.release()

            time.sleep(1)  # Esperar antes de reinicializar

            self.cap = cv2.VideoCapture(self.device_id)
            if self.cap.isOpened():
                self._configure_camera()
                self.error_count = 0
                logger.info("Cámara reinicializada correctamente")
            else:
                logger.error("Error reinicializando cámara")

        except Exception as e:
            logger.error(f"Error en reinicialización: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un frame de la cámara"""
        if not self.is_active:
            raise CameraError("Cámara no inicializada")

        try:
            # Intentar obtener frame del buffer
            try:
                frame = self.frame_buffer.get(timeout=1.0)
                return frame
            except Empty:
                # Si no hay frames en buffer, usar último frame disponible
                if self.last_frame is not None:
                    logger.warning("Usando último frame disponible")
                    return self.last_frame.copy()
                else:
                    logger.warning("No hay frames disponibles")
                    return None

        except Exception as e:
            logger.error(f"Error capturando frame: {e}")
            return None

    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución actual"""
        if not self.cap:
            return (0, 0)

        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        except:
            return (0, 0)

    def get_fps(self) -> float:
        """Obtiene los FPS actuales"""
        if not self.cap:
            return 0.0

        try:
            return self.cap.get(cv2.CAP_PROP_FPS)
        except:
            return 0.0

    def get_stats(self) -> dict:
        """Obtiene estadísticas de la cámara"""
        return {
            'device_id': self.device_id,
            'is_active': self.is_active,
            'frame_count': self.frame_count,
            'error_count': self.error_count,
            'resolution': self.get_resolution(),
            'fps': self.get_fps(),
            'buffer_size': self.frame_buffer.qsize() if self.frame_buffer else 0
        }

    def release(self) -> None:
        """Libera los recursos de la cámara"""
        logger.info("Liberando recursos de cámara...")

        # Detener hilo de captura
        self.stop_capture = True
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)

        # Liberar cámara
        if self.cap:
            self.cap.release()

        # Limpiar buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except Empty:
                break

        self.is_active = False
        logger.info("Cámara liberada correctamente")

class IPCameraAdapter(ICameraService):
    """Adaptador para cámaras IP"""

    def __init__(self, url: str, username: str = None, password: str = None):
        self.url = url
        self.username = username
        self.password = password
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False

        # Construir URL con credenciales si se proporcionan
        if username and password:
            # Formato: rtsp://username:password@ip:port/path
            if '://' in url:
                protocol, rest = url.split('://', 1)
                self.full_url = f"{protocol}://{username}:{password}@{rest}"
            else:
                self.full_url = f"rtsp://{username}:{password}@{url}"
        else:
            self.full_url = url

    def initialize(self) -> bool:
        """Inicializa la cámara IP"""
        try:
            logger.info(f"Inicializando cámara IP: {self.url}")

            self.cap = cv2.VideoCapture(self.full_url)

            if not self.cap.isOpened():
                raise CameraError(f"No se pudo conectar a la cámara IP: {self.url}")

            # Configurar buffer
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Probar captura
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise CameraError("No se pudo capturar frame de la cámara IP")

            self.is_active = True
            logger.info("Cámara IP inicializada correctamente")
            return True

        except Exception as e:
            logger.error(f"Error inicializando cámara IP: {e}")
            raise CameraError(f"Error inicializando cámara IP: {e}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura un frame de la cámara IP"""
        if not self.is_active or not self.cap:
            raise CameraError("Cámara IP no inicializada")

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("No se pudo capturar frame de cámara IP")
            return None

        return frame

    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución actual"""
        if not self.cap:
            return (0, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def release(self) -> None:
        """Libera los recursos de la cámara IP"""
        if self.cap:
            self.cap.release()
            self.is_active = False
            logger.info("Cámara IP liberada")

class MockCameraAdapter(ICameraService):
    """Adaptador mock mejorado para pruebas"""

    def __init__(self, simulate_errors: bool = False):
        self.is_active = False
        self.frame_count = 0
        self.simulate_errors = simulate_errors
        self.error_probability = 0.1 if simulate_errors else 0.0

    def initialize(self) -> bool:
        """Inicializa la cámara mock"""
        self.is_active = True
        logger.info("Cámara mock mejorada inicializada")
        return True

    def capture_frame(self) -> Optional[np.ndarray]:
        """Genera un frame sintético con placas realistas"""
        if not self.is_active:
            raise CameraError("Cámara mock no inicializada")

        # Simular errores ocasionales
        if self.simulate_errors and np.random.random() < self.error_probability:
            logger.warning("Simulando error de captura")
            return None

        # Crear imagen sintética más realista
        frame = self._generate_realistic_frame()

        self.frame_count += 1
        time.sleep(0.033)  # Simular 30 FPS

        return frame

    def _generate_realistic_frame(self) -> np.ndarray:
        """Genera un frame realista con placa"""
        # Crear fondo con textura
        frame = np.random.randint(20, 60, (720, 1280, 3), dtype=np.uint8)

        # Agregar ruido
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Simular diferentes condiciones de iluminación
        brightness_factor = 0.7 + 0.6 * np.random.random()
        frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)

        # Agregar placa realista
        self._add_realistic_plate(frame)

        # Agregar otros elementos del entorno
        self._add_environment_elements(frame)

        return frame

    def _add_realistic_plate(self, frame: np.ndarray):
        """Agrega una placa realista al frame"""
        plates = ["ABC123", "XYZ789", "DEF456", "GHI012", "JKL345", "MNO678"]
        plate_text = plates[self.frame_count % len(plates)]

        # Posición y tamaño variables
        base_x = 400 + np.random.randint(-50, 50)
        base_y = 300 + np.random.randint(-30, 30)
        width = 480 + np.random.randint(-40, 40)
        height = 120 + np.random.randint(-10, 10)

        # Fondo de la placa (blanco con variaciones)
        plate_color = (240 + np.random.randint(-15, 15), 
                      240 + np.random.randint(-15, 15), 
                      240 + np.random.randint(-15, 15))

        cv2.rectangle(frame, (base_x, base_y), (base_x + width, base_y + height), plate_color, -1)

        # Borde de la placa
        border_color = (0, 0, 0)
        cv2.rectangle(frame, (base_x, base_y), (base_x + width, base_y + height), border_color, 3)

        # Texto de la placa con variaciones
        font_scale = 2.0 + np.random.uniform(-0.3, 0.3)
        thickness = 3 + np.random.randint(-1, 1)

        # Calcular posición centrada del texto
        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = base_x + (width - text_size[0]) // 2
        text_y = base_y + (height + text_size[1]) // 2

        # Agregar sombra al texto
        cv2.putText(frame, plate_text, (text_x + 2, text_y + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), thickness)

        # Texto principal
        cv2.putText(frame, plate_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Simular desgaste o suciedad ocasional
        if np.random.random() < 0.3:
            self._add_plate_wear(frame, base_x, base_y, width, height)

    def _add_plate_wear(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        """Agrega efectos de desgaste a la placa"""
        # Manchas aleatorias
        for _ in range(np.random.randint(1, 4)):
            spot_x = x + np.random.randint(0, w)
            spot_y = y + np.random.randint(0, h)
            spot_size = np.random.randint(5, 15)
            spot_color = (np.random.randint(100, 200), np.random.randint(100, 200), np.random.randint(100, 200))
            cv2.circle(frame, (spot_x, spot_y), spot_size, spot_color, -1)

    def _add_environment_elements(self, frame: np.ndarray):
        """Agrega elementos del entorno"""
        # Simular luces de fondo
        for _ in range(np.random.randint(2, 5)):
            light_x = np.random.randint(0, frame.shape[1])
            light_y = np.random.randint(0, frame.shape[0])
            light_radius = np.random.randint(20, 50)
            light_intensity = np.random.randint(100, 255)
            cv2.circle(frame, (light_x, light_y), light_radius, 
                      (light_intensity, light_intensity, light_intensity), -1)

    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución mock"""
        return (1280, 720)

    def release(self) -> None:
        """Libera recursos mock"""
        self.is_active = False
        logger.info("Cámara mock liberada")

# Alias para compatibilidad hacia atrás
OpenCVCameraAdapter = EnhancedOpenCVCameraAdapter
