"""
Ultra Service Factory - Fábrica optimizada de servicios
Crea todos los servicios ultra-optimizados del sistema
"""

import sqlite3
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time
import requests

from core.interfaces import ICameraService, IPlateDetector, IEventStorage, INetworkService
from utils.logger import get_logger
from .ultra_camera_service import UltraCameraService
from .ultra_smart_detector import UltraSmartDetector

logger = get_logger(__name__)

class UltraEventStorage(IEventStorage):
    """Sistema ultra-optimizado de almacenamiento con SQLite y caché"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('storage', {}).get('database', 'data/events.db')
        self.cache = {}
        self.cache_size = 1000
        self._ensure_database()
    
    def _ensure_database(self):
        """Asegura que la base de datos existe"""
        try:
            # Crear directorio si no existe
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Crear tabla si no existe
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp INTEGER NOT NULL,
                        image_path TEXT,
                        bbox TEXT,
                        sent BOOLEAN DEFAULT FALSE,
                        created_at INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # Crear índices para optimizar consultas
                conn.execute('CREATE INDEX IF NOT EXISTS idx_plate_number ON events(plate_number)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_sent ON events(sent)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creando base de datos: {e}")
    
    def store_event(self, event) -> bool:
        """Almacena un evento en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (plate_number, confidence, timestamp, image_path, bbox)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    event.plate_number,
                    event.confidence,
                    int(time.time()),
                    getattr(event, 'image_path', None),
                    str(getattr(event, 'bbox', ''))
                ))
                conn.commit()
                
                # Actualizar caché
                event_id = cursor.lastrowid
                self.cache[event_id] = event
                
                # Limpiar caché si es muy grande
                if len(self.cache) > self.cache_size:
                    # Eliminar los más antiguos
                    oldest_keys = sorted(self.cache.keys())[:100]
                    for key in oldest_keys:
                        del self.cache[key]
                
                return True
                
        except Exception as e:
            logger.error(f"Error almacenando evento: {e}")
            return False
    
    def get_pending_events(self) -> list:
        """Obtiene eventos pendientes de envío"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, plate_number, confidence, timestamp, image_path, bbox
                    FROM events 
                    WHERE sent = FALSE 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                ''')
                
                events = []
                for row in cursor.fetchall():
                    # Crear objeto de evento simple
                    event = type('Event', (), {
                        'id': row[0],
                        'plate_number': row[1],
                        'confidence': row[2],
                        'timestamp': row[3],
                        'image_path': row[4],
                        'bbox': row[5],
                        'to_dict': lambda self: {
                            'plate_number': self.plate_number,
                            'confidence': self.confidence,
                            'timestamp': self.timestamp,
                            'image_path': self.image_path
                        }
                    })()
                    events.append(event)
                
                return events
                
        except Exception as e:
            logger.error(f"Error obteniendo eventos pendientes: {e}")
            return []
    
    def mark_as_sent(self, event_id: int) -> bool:
        """Marca un evento como enviado"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('UPDATE events SET sent = TRUE WHERE id = ?', (event_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error marcando evento como enviado: {e}")
            return False

class UltraNetworkService(INetworkService):
    """Servicio ultra-optimizado de red con reintentos y caché"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.network_config = config.get('network', {})
        self.api_url = self.network_config.get('api_url', 'http://localhost:8000/api')
        self.timeout = self.network_config.get('timeout', 30)
        self.max_retries = self.network_config.get('max_retries', 3)
        self.session = requests.Session()
        
        # Configurar headers por defecto
        self.session.headers.update({
            'User-Agent': 'XennovaVision/1.0',
            'Content-Type': 'application/json'
        })
    
    def is_connected(self) -> bool:
        """Verifica conectividad de red"""
        try:
            # Probar con Google DNS (más rápido que hacer HTTP)
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except:
            return False
    
    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """Envía un evento al servidor con reintentos"""
        if not self.is_connected():
            logger.warning("Sin conexión de red")
            return False
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.api_url}/events",
                    json=event_data,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 201]:
                    return True
                else:
                    logger.warning(f"Error HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout en intento {attempt + 1}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error de red en intento {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Backoff exponencial
        
        return False

class UltraServiceFactory:
    """Fábrica ultra-optimizada para crear todos los servicios del sistema"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa la fábrica con configuración
        
        Args:
            config: Diccionario de configuración. Si es None, usa configuración por defecto
        """
        self.config = config or self._get_default_config()
        logger.info("🏭 UltraServiceFactory inicializada")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto optimizada"""
        return {
            'camera': {
                'type': 'usb',
                'device_id': 1,  # Basado en los tests
                'resolution': [1280, 720],  # Resolución óptima encontrada
                'fps': 30,
                'frame_buffer_size': 5
            },
            'detection': {
                'confidence_threshold': 0.6,
                'max_detections': 5,
                'processing_size': [640, 480],
                'skip_frames': 1
            },
            'storage': {
                'database': 'data/events.db',
                'max_events': 10000
            },
            'network': {
                'api_url': 'http://localhost:8000/api',
                'timeout': 30,
                'max_retries': 3
            },
            'performance': {
                'threads': 4,
                'batch_size': 1,
                'use_gpu': False
            }
        }
    
    def create_camera_service(self) -> ICameraService:
        """Crea servicio ultra de cámara"""
        logger.info("📷 Creando UltraCameraService...")
        return UltraCameraService(self.config.get('camera', {}))
    
    def create_plate_detector(self) -> IPlateDetector:
        """Crea detector ultra de placas"""
        logger.info("🤖 Creando UltraSmartDetector...")
        return UltraSmartDetector(self.config.get('detection', {}))
    
    def create_storage_service(self) -> IEventStorage:
        """Crea servicio ultra de almacenamiento"""
        logger.info("💾 Creando UltraEventStorage...")
        return UltraEventStorage(self.config)
    
    def create_network_service(self) -> INetworkService:
        """Crea servicio ultra de red"""
        logger.info("🌐 Creando UltraNetworkService...")
        return UltraNetworkService(self.config)
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Retorna la configuración optimizada actual"""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Actualiza la configuración de la fábrica"""
        self.config.update(new_config)
        logger.info("⚙️ Configuración actualizada")