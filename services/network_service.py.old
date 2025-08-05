import requests
import time
from typing import Dict, Any, Optional
from core.interfaces import INetworkService
from core.exceptions import NetworkError
from utils.logger import get_logger

logger = get_logger(__name__)

class HTTPNetworkService(INetworkService):
    """Servicio de red HTTP"""
    
    def __init__(self, endpoint: str, timeout: int = 30, retry_attempts: int = 3):
        self.endpoint = endpoint
        self.timeout = timeout
        self.retry_attempts = retry_attempts
    
    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """Envía un evento al endpoint"""
        if not self.endpoint:
            logger.warning("No hay endpoint configurado")
            return False
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    self.endpoint,
                    json=event_data,
                    timeout=self.timeout,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    logger.info(f"Evento enviado exitosamente: {event_data.get('plate_number', 'N/A')}")
                    return True
                else:
                    logger.warning(f"Error HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Error de red (intento {attempt + 1}/{self.retry_attempts}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
        
        raise NetworkError(f"Falló el envío después de {self.retry_attempts} intentos")
    
    def is_connected(self) -> bool:
        """Verifica si hay conexión"""
        if not self.endpoint:
            return False
        
        try:
            response = requests.get(self.endpoint, timeout=5)
            return response.status_code < 500
        except:
            return False