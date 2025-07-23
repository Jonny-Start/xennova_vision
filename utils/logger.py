"""
Sistema de logging configurado
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Formatter con colores para la consola"""
    
    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Verde
        'WARNING': '\033[33m',   # Amarillo
        'ERROR': '\033[31m',     # Rojo
        'CRITICAL': '\033[35m',  # Magenta
        'ENDC': '\033[0m'        # Fin color
    }
    
    def format(self, record):
        # Agregar color al nivel
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['ENDC']}"
        
        return super().format(record)

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Configura y retorna un logger
    
    Args:
        name: Nombre del logger
        level: Nivel de logging
        log_file: Archivo de log (opcional)
        console: Si mostrar en consola
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    
    # Evitar duplicar handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Formato para logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formatter con colores para consola
    color_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para consola
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
    
    # Handler para archivo
    if log_file:
        # Crear directorio si no existe
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Obtiene un logger configurado
    
    Args:
        name: Nombre del logger (generalmente __name__)
        level: Nivel de logging
    
    Returns:
        Logger configurado
    """
    # Crear directorio de logs
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Archivo de log con timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"plate_system_{timestamp}.log"
    
    return setup_logger(
        name=name,
        level=level,
        log_file=str(log_file),
        console=True
    )

# Logger principal del sistema
system_logger = get_logger("plate_system")

def log_info(message: str):
    """Log de información"""
    system_logger.info(message)

def log_error(message: str):
    """Log de error"""
    system_logger.error(message)

def log_warning(message: str):
    """Log de advertencia"""
    system_logger.warning(message)

def log_debug(message: str):
    """Log de debug"""
    system_logger.debug(message)