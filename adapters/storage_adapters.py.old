"""
Adaptadores para diferentes tipos de almacenamiento
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from core.interfaces import IEventStorage
from core.models import PlateEvent
from core.exceptions import StorageError
from utils.logger import get_logger

logger = get_logger(__name__)

class JSONStorageAdapter(IEventStorage):
    """Adaptador para almacenamiento en JSON"""
    
    def __init__(self, file_path: str = "events.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def store_event(self, event: PlateEvent) -> bool:
        """Almacena un evento en JSON"""
        try:
            # Cargar eventos existentes
            events = self._load_events()
            
            # Agregar nuevo evento
            event_dict = {
                'id': event.id,
                'plate_number': event.plate_number,
                'confidence': event.confidence,
                'timestamp': event.timestamp.isoformat(),
                'bbox': event.bbox,
                'image_path': event.image_path,
                'sent': event.sent
            }
            
            events.append(event_dict)
            
            # Guardar eventos
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Evento almacenado: {event.plate_number}")
            return True
            
        except Exception as e:
            logger.error(f"Error almacenando evento: {e}")
            raise StorageError(f"Error almacenando evento: {e}")
    
    def get_pending_events(self) -> List[PlateEvent]:
        """Obtiene eventos pendientes de envío"""
        try:
            events = self._load_events()
            pending_events = []
            
            for event_dict in events:
                if not event_dict.get('sent', False):
                    event = PlateEvent(
                        id=event_dict['id'],
                        plate_number=event_dict['plate_number'],
                        confidence=event_dict['confidence'],
                        timestamp=datetime.fromisoformat(event_dict['timestamp']),
                        bbox=tuple(event_dict['bbox']),
                        image_path=event_dict.get('image_path'),
                        sent=event_dict['sent']
                    )
                    pending_events.append(event)
            
            return pending_events
            
        except Exception as e:
            logger.error(f"Error obteniendo eventos pendientes: {e}")
            return []
    
    def mark_as_sent(self, event_id: str) -> bool:
        """Marca un evento como enviado"""
        try:
            events = self._load_events()
            
            for event in events:
                if event['id'] == event_id:
                    event['sent'] = True
                    break
            
            # Guardar cambios
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Evento marcado como enviado: {event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error marcando evento como enviado: {e}")
            return False
    
    def _load_events(self) -> List[Dict[str, Any]]:
        """Carga eventos desde el archivo JSON"""
        if not self.file_path.exists():
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

class SQLiteStorageAdapter(IEventStorage):
    """Adaptador para almacenamiento en SQLite"""
    
    def __init__(self, db_path: str = "events.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Inicializa la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS plate_events (
                        id TEXT PRIMARY KEY,
                        plate_number TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        bbox_x1 INTEGER,
                        bbox_y1 INTEGER,
                        bbox_x2 INTEGER,
                        bbox_y2 INTEGER,
                        image_path TEXT,
                        sent BOOLEAN DEFAULT FALSE
                    )
                ''')
                conn.commit()
                logger.info("Base de datos SQLite inicializada")
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            raise StorageError(f"Error inicializando base de datos: {e}")
    
    def store_event(self, event: PlateEvent) -> bool:
        """Almacena un evento en SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO plate_events 
                    (id, plate_number, confidence, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path, sent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id,
                    event.plate_number,
                    event.confidence,
                    event.timestamp.isoformat(),
                    event.bbox[0] if event.bbox else None,
                    event.bbox[1] if event.bbox else None,
                    event.bbox[2] if event.bbox else None,
                    event.bbox[3] if event.bbox else None,
                    event.image_path,
                    event.sent
                ))
                conn.commit()
                logger.debug(f"Evento almacenado en SQLite: {event.plate_number}")
                return True
        except Exception as e:
            logger.error(f"Error almacenando evento en SQLite: {e}")
            raise StorageError(f"Error almacenando evento: {e}")
    
    def get_pending_events(self) -> List[PlateEvent]:
        """Obtiene eventos pendientes de envío"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT id, plate_number, confidence, timestamp, 
                           bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_path, sent
                    FROM plate_events 
                    WHERE sent = FALSE
                    ORDER BY timestamp
                ''')
                
                events = []
                for row in cursor.fetchall():
                    bbox = None
                    if all(x is not None for x in row[3:7]):
                        bbox = (row[3], row[4], row[5], row[6])
                    
                    event = PlateEvent(
                        id=row[0],
                        plate_number=row[1],
                        confidence=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        bbox=bbox,
                        image_path=row[8],
                        sent=bool(row[9])
                    )
                    events.append(event)
                
                return events
        except Exception as e:
            logger.error(f"Error obteniendo eventos pendientes de SQLite: {e}")
            return []
    
    def mark_as_sent(self, event_id: str) -> bool:
        """Marca un evento como enviado"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE plate_events 
                    SET sent = TRUE 
                    WHERE id = ?
                ''', (event_id,))
                conn.commit()
                logger.debug(f"Evento marcado como enviado en SQLite: {event_id}")
                return True
        except Exception as e:
            logger.error(f"Error marcando evento como enviado en SQLite: {e}")
            return False