import os
import sys
import subprocess
import platform
import psutil

class AutoInstaller:
    def __init__(self):
        self.is_embedded = self._detect_embedded()
        self.python_cmd = self._get_python_command()
    
    def _detect_embedded(self) -> bool:
        """Detecta si estamos en un sistema embebido"""
        # Detectar Raspberry Pi
        if os.path.exists('/proc/device-tree/model'):
            try:
                with open('/proc/device-tree/model', 'r') as f:
                    model = f.read().lower()
                    if 'raspberry pi' in model:
                        return True
            except:
                pass
        
        # Detectar Jetson
        if os.path.exists('/proc/device-tree/nvidia,dtsfilename'):
            return True
        
        # Detectar por recursos limitados
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        if memory_gb < 8 and cpu_count < 4:
            return True
        
        return False
    
    def _get_python_command(self):
        """Obtiene el comando de Python apropiado"""
        for cmd in ['python3', 'python']:
            try:
                result = subprocess.run([cmd, '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0 and '3.' in result.stdout:
                    return cmd
            except FileNotFoundError:
                continue
        return 'python3'
    
    def install_system_dependencies(self):
        """Instala dependencias del sistema"""
        print("🔧 Instalando dependencias del sistema...")
        
        if platform.system() == 'Linux':
            # Detectar distribución
            try:
                with open('/etc/os-release', 'r') as f:
                    os_info = f.read().lower()
                
                if 'ubuntu' in os_info or 'debian' in os_info:
                    self._install_apt_packages()
                elif 'fedora' in os_info or 'centos' in os_info:
                    self._install_yum_packages()
            except:
                print("⚠️  No se pudo detectar la distribución. Instala manualmente:")
                print("   - tesseract-ocr")
                print("   - libopencv-dev")
                print("   - python3-dev")
    
    def _install_apt_packages(self):
        """Instala paquetes con apt (Ubuntu/Debian)"""
        packages = [
            'tesseract-ocr',
            'tesseract-ocr-spa',  # Español
            'libopencv-dev',
            'python3-dev',
            'python3-pip',
            'libgstreamer1.0-dev',  # Para cámaras CSI
            'libgstreamer-plugins-base1.0-dev'
        ]
        
        try:
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            subprocess.run(['sudo', 'apt', 'install', '-y'] + packages, check=True)
            print("✅ Dependencias del sistema instaladas")
        except subprocess.CalledProcessError:
            print("❌ Error instalando dependencias del sistema")
    
    def _install_yum_packages(self):
        """Instala paquetes con yum/dnf (Fedora/CentOS)"""
        packages = [
            'tesseract',
            'tesseract-langpack-spa',
            'opencv-devel',
            'python3-devel',
            'python3-pip'
        ]
        
        try:
            cmd = 'dnf' if os.path.exists('/usr/bin/dnf') else 'yum'
            subprocess.run(['sudo', cmd, 'install', '-y'] + packages, check=True)
            print("✅ Dependencias del sistema instaladas")
        except subprocess.CalledProcessError:
            print("❌ Error instalando dependencias del sistema")
    
    def install_python_dependencies(self):
        """Instala dependencias de Python"""
        print(f"🐍 Instalando dependencias de Python para {'embebido' if self.is_embedded else 'desktop'}...")
        
        requirements_file = 'requirements/embedded.txt' if self.is_embedded else 'requirements/desktop.txt'
        
        try:
            subprocess.run([
                self.python_cmd, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], check=True)
            
            subprocess.run([
                self.python_cmd, '-m', 'pip', 'install', '-r', requirements_file
            ], check=True)
            
            print("✅ Dependencias de Python instaladas")
        except subprocess.CalledProcessError:
            print("❌ Error instalando dependencias de Python")
            return False
        
        return True
    
    def create_config_files(self):
        """Crea archivos de configuración"""
        print("📝 Creando archivos de configuración...")
        
        # Crear directorios
        os.makedirs('config', exist_ok=True)
        
        # Configuración embebida
        embedded_config = {
            "camera": {
                "type": "csi" if self.is_embedded else "usb",
                "device_id": 0,
                "resolution": [640, 480] if self.is_embedded else [1280, 720],
                "fps": 15 if self.is_embedded else 30
            },
            "ai_model": {
                "type": "lightweight",
                "ocr_engine": "tesseract",
                "confidence_threshold": 0.7
            },
            "storage": {
                "type": "file",
                "path": "/tmp/plate_events" if self.is_embedded else "./plate_events",
                "max_events": 1000
            },
            "network": {
                "endpoint": "https://your-api-endpoint.com/plate-events",
                "timeout": 5,
                "retry_attempts": 3,
                "batch_size": 10
            },
            "capture_interval": 2.0 if self.is_embedded else 1.0,
            "log_level": "INFO"
        }
        
        # Configuración desktop
        desktop_config = {
            "camera": {
                "type": "usb",
                "device_id": 0,
                "resolution": [1920, 1080],
                "fps": 30
            },
            "ai_model": {
                "type": "advanced",
                "ocr_engine": "easyocr",
                "yolo_model": "yolov8n.pt",
                "confidence_threshold": 0.8
            },
            "storage": {
                "type": "sqlite",
                "database_path": "./plate_events.db",
                "max_events": 10000
            },
            "network": {
                "endpoint": "https://your-api-endpoint.com/plate-events",
                "timeout": 10,
                "retry_attempts": 5,
                "batch_size": 50
            },
            "capture_interval": 0.5,
            "log_level": "DEBUG"
        }
        
        import json
        
        with open('config/embedded_config.json', 'w') as f:
            json.dump(embedded_config, f, indent=2)
        
        with open('config/desktop_config.json', 'w') as f:
            json.dump(desktop_config, f, indent=2)
        
        print("✅ Archivos de configuración creados")
    
    def run_tests(self):
        """Ejecuta pruebas básicas"""
        print("🧪 Ejecutando pruebas básicas...")
        
        try:
            # Test de importaciones
            subprocess.run([
                self.python_cmd, '-c', 
                'import cv2, numpy, requests; print("✅ Importaciones básicas OK")'
            ], check=True)
            
            if self.is_embedded:
                subprocess.run([
                    self.python_cmd, '-c', 
                    'import pytesseract; print("✅ Tesseract OK")'
                ], check=True)
            else:
                subprocess.run([
                    self.python_cmd, '-c', 
                    'import easyocr; print("✅ EasyOCR OK")'
                ], check=True)
            
            print("✅ Todas las pruebas pasaron")
            return True
            
        except subprocess.CalledProcessError:
            print("❌ Algunas pruebas fallaron")
            return False
    
    def install(self):
        """Ejecuta la instalación completa"""
        print("🚀 Iniciando instalación automática...")
        print(f"📱 Hardware detectado: {'Embebido' if self.is_embedded else 'Desktop'}")
        print(f"🐍 Comando Python: {self.python_cmd}")
        
        # Instalar dependencias del sistema
        if platform.system() == 'Linux':
            self.install_system_dependencies()
        
        # Instalar dependencias de Python
        if not self.install_python_dependencies():
            return False
        
        # Crear archivos de configuración
        self.create_config_files()
        
        # Ejecutar pruebas
        if not self.run_tests():
            print("⚠️  Instalación completada con advertencias")
            return False
        
        print("🎉 ¡Instalación completada exitosamente!")
        print("📋 Próximos pasos:")
        print("   1. Edita config/embedded_config.json o config/desktop_config.json")
        print("   2. Configura tu endpoint en la sección 'network'")
        print("   3. Ejecuta: python main.py")
        
        return True

if __name__ == "__main__":
    installer = AutoInstaller()
    installer.install()