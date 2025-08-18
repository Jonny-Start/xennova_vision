import json
import sqlite3
import os
from typing import List
from datetime import datetime
from ..core.interfaces import IEventStorage
from ..core.models import PlateEvent

class FileEventStorage(IEventStorage):
    """Almacenamiento en archivos para sistemas embebidos"""
    
    def __init__(self, config: dict):
        self.storage_path = config.get('path', '/tmp/plate_events')
        self.max_events = config.get('max_events', 1000)
        os.makedirs(self.storage_path, exist_ok=True)
    
    def save_event(self, event: PlateEvent) -> bool:
        """Guarda un evento en archivo JSON"""
        try:
            file_path = os.path.join(self.storage_path, f"{event.id}.json")
            with open(file_path, 'w') as f:
                json.dump(event.to_dict(), f)
            return True
        except Exception as e:
            print(f"Error guardando evento: {e}")
            return False
    
    def get_pending_events(self) -> List[PlateEvent]:
        """Obtiene eventos pendientes de envío"""
        events = []
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_path, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if not data.get('sent', False):
                            event = PlateEvent(
                                id=data['id'],
                                plate_text=data['plate_text'],
                                timestamp=datetime.fromisoformat(data['timestamp']),
                                confidence=data['confidence'],
                                image_path=data.get('image_path'),
                                metadata=data.get('metadata'),
                                sent=data.get('sent', False)
                            )
                            events.append(event)
        except Exception as e:
            print(f"Error obteniendo eventos pendientes: {e}")
        
        return events[:self.max_events]
    
    def mark_as_sent(self, event_id: str) -> bool:
        """Marca un evento como enviado"""
        try:
            file_path = os.path.join(self.storage_path, f"{event_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                data['sent'] = True
                with open(file_path, 'w') as f:
                    json.dump(data, f)
                return True
        except Exception as e:
            print(f"Error marcando evento como enviado: {e}")
        return False

class SQLiteEventStorage(IEventStorage):
    """Almacenamiento en SQLite para sistemas desktop"""
    
    def __init__(self, config: dict):
        self.db_path = config.get('database_path', './plate_events.db')
        self.max_events = config.get('max_events', 10000)
        self._initialize_database()
    
    def _initialize_database(self):
        """Inicializa la base de datos SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS plate_events (
                    id TEXT PRIMARY KEY,
                    plate_text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    image_path TEXT,
                    metadata TEXT,
                    sent BOOLEAN DEFAULT FALSE
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error inicializando base de datos: {e}")
    
    def save_event(self, event: PlateEvent) -> bool:
        """Guarda un evento en SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO plate_events 
                (id, plate_text, timestamp, confidence, image_path, metadata, sent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.plate_text,
                event.timestamp.isoformat(),
                event.confidence,
                event.image_path,
                json.dumps(event.metadata) if event.metadata else None,
                event.sent
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error guardando evento en SQLite: {e}")
            return False
    
    def get_pending_events(self) -> List[PlateEvent]:
        """Obtiene eventos pendientes de envío"""
        events = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, plate_text, timestamp, confidence, image_path, metadata, sent
                FROM plate_events 
                WHERE sent = FALSE 
                LIMIT ?
            """, (self.max_events,))
            
            for row in cursor.fetchall():
                event = PlateEvent(
                    id=row[0],
                    plate_text=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    confidence=row[3],
                    image_path=row[4],
                    metadata=json.loads(row[5]) if row[5] else None,
                    sent=bool(row[6])
                )
                events.append(event)
            
            conn.close()
        except Exception as e:
            print(f"Error obteniendo eventos pendientes de SQLite: {e}")
        
        return events
    
    def mark_as_sent(self, event_id: str) -> bool:
        """Marca un evento como enviado"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE plate_events 
                SET sent = TRUE 
                WHERE id = ?
            """, (event_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error marcando evento como enviado en SQLite: {e}")
            return False