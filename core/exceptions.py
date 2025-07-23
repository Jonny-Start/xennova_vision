"""
Excepciones personalizadas para el sistema de reconocimiento de placas
"""

class PlateSystemError(Exception):
    """Excepción base para errores del sistema de placas"""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class CameraError(PlateSystemError):
    """Excepción para errores de cámara"""
    
    def __init__(self, message: str, device_id: int = None):
        self.device_id = device_id
        error_code = f"CAM_{device_id}" if device_id is not None else "CAM"
        super().__init__(message, error_code)

class AIModelError(PlateSystemError):
    """Excepción para errores del modelo de IA"""
    
    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        error_code = f"AI_{model_name}" if model_name else "AI"
        super().__init__(message, error_code)

class NetworkError(PlateSystemError):
    """Excepción para errores de red"""
    
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        error_code = f"NET_{status_code}" if status_code else "NET"
        super().__init__(message, error_code)

class StorageError(PlateSystemError):
    """Excepción para errores de almacenamiento"""
    
    def __init__(self, message: str, storage_type: str = None):
        self.storage_type = storage_type
        error_code = f"STORE_{storage_type}" if storage_type else "STORE"
        super().__init__(message, error_code)

class ConfigurationError(PlateSystemError):
    """Excepción para errores de configuración"""
    
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        error_code = f"CONFIG_{config_key}" if config_key else "CONFIG"
        super().__init__(message, error_code)

class ValidationError(PlateSystemError):
    """Excepción para errores de validación"""
    
    def __init__(self, message: str, field_name: str = None):
        self.field_name = field_name
        error_code = f"VALID_{field_name}" if field_name else "VALID"
        super().__init__(message, error_code)

class ProcessingError(PlateSystemError):
    """Excepción para errores de procesamiento"""
    
    def __init__(self, message: str, stage: str = None):
        self.stage = stage
        error_code = f"PROC_{stage}" if stage else "PROC"
        super().__init__(message, error_code)