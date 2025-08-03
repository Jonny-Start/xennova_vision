import cv2
import time
import threading
from typing import Optional, Dict, List, Tuple
from collections import deque
import numpy as np
from core.interfaces import ICameraService

class UltraCameraService(ICameraService):
    """
    Servicio de cámara ultra-optimizado:
    - Auto-detección de cámaras USB
    - Buffer inteligente para frames
    - Configuración automática de parámetros
    - Recuperación automática de errores
    - Estadísticas de rendimiento
    """

    def __init__(self, config: dict):
        self.config = config
        self.device_id = config.get('device_id', 0)
        self.backup_devices = config.get('backup_devices', [1, 2])
        self.resolution = config.get('resolution', [1280, 720])
        self.fps = config.get('fps', 30)
        self.auto_detect = config.get('auto_detect', True)
        
        # Estado de la cámara
        self.cap = None
        self.is_initialized = False
        self.current_device = None
        
        # Buffer de frames
        self.frame_buffer = deque(maxlen=config.get('frame_buffer_size', 5))
        self.buffer_lock = threading.Lock()
        
        # Thread para captura continua
        self.capture_thread = None
        self.capture_running = False
        
        # Estadísticas
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'fps_actual': 0.0,
            'last_frame_time': 0.0,
            'errors': 0,
            'reconnections': 0
        }
        
        # Control de errores
        self.consecutive_errors = 0
        self.max_errors = 5
        self.last_error_time = 0
        
        # Configuraciones optimizadas por resolución
        self.resolution_configs = {
            (640, 480): {'fps': 60, 'buffer_size': 3},
            (1280, 720): {'fps': 30, 'buffer_size': 5},
            (1920, 1080): {'fps': 15, 'buffer_size': 7}
        }

    def initialize(self) -> bool:
        """Inicializa la cámara con auto-detección"""
        try:
            print("🎥 Inicializando UltraCameraService...")
            
            # Auto-detectar cámaras si está habilitado
            if self.auto_detect:
                available_cameras = self._detect_available_cameras()
                if not available_cameras:
                    print("❌ No se encontraron cámaras disponibles")
                    return False
                
                print(f"📹 Cámaras detectadas: {list(available_cameras.keys())}")
            
            # Intentar inicializar cámara principal
            if self._initialize_camera(self.device_id):
                self.current_device = self.device_id
                print(f"✅ Cámara principal {self.device_id} inicializada")
            else:
                # Intentar cámaras de respaldo
                for backup_id in self.backup_devices:
                    if self._initialize_camera(backup_id):
                        self.current_device = backup_id
                        print(f"✅ Cámara de respaldo {backup_id} inicializada")
                        break
                else:
                    print("❌ No se pudo inicializar ninguna cámara")
                    return False
            
            # Configurar parámetros optimizados
            self._configure_camera_parameters()
            
            # Iniciar captura continua
            self._start_continuous_capture()
            
            self.is_initialized = True
            print("✅ UltraCameraService inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando cámara: {e}")
            return False

    def _detect_available_cameras(self) -> Dict[int, Dict]:
        """Detecta cámaras disponibles con información detallada"""
        available = {}
        
        print("🔍 Detectando cámaras disponibles...")
        
        for i in range(10):  # Buscar en los primeros 10 dispositivos
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Probar captura
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Obtener información de la cámara
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        available[i] = {
                            'resolution': (width, height),
                            'fps': fps,
                            'name': f"Camera {i}",
                            'working': True
                        }
                        
                        print(f"  ✅ Cámara {i}: {width}x{height}@{fps}fps")
                
                cap.release()
                
            except Exception as e:
                print(f"  ⚠️ Error probando cámara {i}: {e}")
                continue
        
        return available

    def _initialize_camera(self, device_id: int) -> bool:
        """Inicializa una cámara específica"""
        try:
            if self.cap:
                self.cap.release()
            
            # Crear captura con configuración optimizada
            self.cap = cv2.VideoCapture(device_id)
            
            if not self.cap.isOpened():
                return False
            
            # Configurar resolución y FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Configuraciones adicionales para mejor rendimiento
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Probar captura
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.release()
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error inicializando cámara {device_id}: {e}")
            return False

    def _configure_camera_parameters(self):
        """Configura parámetros optimizados según la resolución"""
        if not self.cap:
            return
        
        try:
            # Obtener resolución actual
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution_key = (actual_width, actual_height)
            
            # Aplicar configuración optimizada si existe
            if resolution_key in self.resolution_configs:
                config = self.resolution_configs[resolution_key]
                self.cap.set(cv2.CAP_PROP_FPS, config['fps'])
                self.frame_buffer = deque(maxlen=config['buffer_size'])
                print(f"📐 Configuración optimizada aplicada para {resolution_key}")
            
            # Configuraciones adicionales para calidad
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Exposición manual
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)
            
            print(f"🎛️ Parámetros de cámara configurados: {actual_width}x{actual_height}")
            
        except Exception as e:
            print(f"⚠️ Error configurando parámetros: {e}")

    def _start_continuous_capture(self):
        """Inicia captura continua en thread separado"""
        if self.capture_thread and self.capture_thread.is_alive():
            return
        
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._continuous_capture_loop, daemon=True)
        self.capture_thread.start()
        print("🔄 Captura continua iniciada")

    def _continuous_capture_loop(self):
        """Loop principal de captura continua"""
        last_fps_time = time.time()
        fps_counter = 0
        
        while self.capture_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Agregar frame al buffer
                    with self.buffer_lock:
                        if len(self.frame_buffer) >= self.frame_buffer.maxlen:
                            self.stats['frames_dropped'] += 1
                        
                        self.frame_buffer.append({
                            'frame': frame.copy(),
                            'timestamp': time.time()
                        })
                    
                    self.stats['frames_captured'] += 1
                    self.stats['last_frame_time'] = time.time()
                    fps_counter += 1
                    
                    # Calcular FPS actual
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        self.stats['fps_actual'] = fps_counter / (current_time - last_fps_time)
                        fps_counter = 0
                        last_fps_time = current_time
                    
                    # Reset contador de errores
                    self.consecutive_errors = 0
                    
                else:
                    self._handle_capture_error()
                
                # Pequeña pausa para no saturar CPU
                time.sleep(0.001)
                
            except Exception as e:
                self._handle_capture_error(str(e))

    def _handle_capture_error(self, error_msg: str = ""):
        """Maneja errores de captura con recuperación automática"""
        self.consecutive_errors += 1
        self.stats['errors'] += 1
        self.last_error_time = time.time()
        
        if error_msg:
            print(f"⚠️ Error de captura: {error_msg}")
        
        # Si hay muchos errores consecutivos, intentar reconectar
        if self.consecutive_errors >= self.max_errors:
            print("🔄 Intentando reconectar cámara...")
            
            if self._reconnect_camera():
                self.stats['reconnections'] += 1
                self.consecutive_errors = 0
                print("✅ Cámara reconectada")
            else:
                print("❌ No se pudo reconectar la cámara")
                time.sleep(1)  # Esperar antes del siguiente intento

    def _reconnect_camera(self) -> bool:
        """Intenta reconectar la cámara"""
        try:
            # Liberar cámara actual
            if self.cap:
                self.cap.release()
                time.sleep(0.5)
            
            # Intentar reconectar con dispositivo actual
            if self._initialize_camera(self.current_device):
                self._configure_camera_parameters()
                return True
            
            # Intentar con dispositivos de respaldo
            for backup_id in self.backup_devices:
                if backup_id != self.current_device:
                    if self._initialize_camera(backup_id):
                        self.current_device = backup_id
                        self._configure_camera_parameters()
                        return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error en reconexión: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captura frame del buffer (método requerido por interfaz)"""
        if not self.is_initialized:
            return None
        
        try:
            with self.buffer_lock:
                if self.frame_buffer:
                    # Obtener frame más reciente
                    frame_data = self.frame_buffer[-1]
                    return frame_data['frame']
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error capturando frame: {e}")
            return None

    def get_latest_frame_with_timestamp(self) -> Optional[Tuple[np.ndarray, float]]:
        """Obtiene el frame más reciente con timestamp"""
        try:
            with self.buffer_lock:
                if self.frame_buffer:
                    frame_data = self.frame_buffer[-1]
                    return frame_data['frame'], frame_data['timestamp']
            
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️ Error obteniendo frame con timestamp: {e}")
            return None, 0.0

    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene la resolución actual de la cámara"""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return tuple(self.resolution)

    def get_fps(self) -> float:
        """Obtiene el FPS actual"""
        return self.stats.get('fps_actual', 0.0)

    def is_available(self) -> bool:
        """Verifica si la cámara está disponible"""
        return (self.is_initialized and 
                self.cap and 
                self.cap.isOpened() and 
                self.capture_running)

    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la cámara"""
        return {
            **self.stats,
            'current_device': self.current_device,
            'resolution': self.get_resolution(),
            'buffer_size': len(self.frame_buffer),
            'is_available': self.is_available(),
            'consecutive_errors': self.consecutive_errors
        }

    def reset_stats(self):
        """Reinicia estadísticas"""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0 if isinstance(self.stats[key], int) else 0.0
        print("📊 Estadísticas de cámara reiniciadas")

    def release(self):
        """Libera recursos de la cámara"""
        print("🔄 Liberando recursos de cámara...")
        
        # Detener captura continua
        self.capture_running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Liberar cámara
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Limpiar buffer
        with self.buffer_lock:
            self.frame_buffer.clear()
        
        self.is_initialized = False
        print("✅ Recursos de cámara liberados")

    def __del__(self):
        """Destructor para asegurar liberación de recursos"""
        self.release()