#!/bin/bash
# Script de instalación automática para Xennova Vision mejorado

echo "🚀 INSTALANDO XENNOVA VISION MEJORADO"
echo "====================================="

# Verificar que estamos en el directorio correcto
if [ ! -f "main.py" ] || [ ! -d "services" ]; then
    echo "❌ Error: Ejecuta este script desde la raíz del proyecto xennova_vision"
    exit 1
fi

echo "📁 Verificando estructura del proyecto..."

# Crear backup de archivos originales
echo "💾 Creando backup de archivos originales..."
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
cp services/service_factory.py backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp config/desktop_config.json backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp services/__init__.py backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

echo "📥 Instalando archivos nuevos..."

# 1. Reemplazar service_factory.py
if [ -f "enhanced_service_factory.py" ]; then
    cp enhanced_service_factory.py services/service_factory.py
    echo "✅ services/service_factory.py actualizado"
else
    echo "❌ No se encontró enhanced_service_factory.py"
fi

# 2. Agregar detector inteligente
if [ -f "smart_plate_detector.py" ]; then
    cp smart_plate_detector.py services/smart_plate_detector.py
    echo "✅ services/smart_plate_detector.py agregado"
else
    echo "❌ No se encontró smart_plate_detector.py"
fi

# 3. Actualizar configuración
if [ -f "optimized_config.json" ]; then
    cp optimized_config.json config/desktop_config.json
    echo "✅ config/desktop_config.json actualizado"
else
    echo "❌ No se encontró optimized_config.json"
fi

# 4. Actualizar __init__.py de services
if [ -f "services_init.py" ]; then
    cp services_init.py services/__init__.py
    echo "✅ services/__init__.py actualizado"
else
    echo "❌ No se encontró services_init.py"
fi

# 5. Copiar script de prueba a la raíz
if [ -f "test_cameras.py" ]; then
    cp test_cameras.py ./test_cameras.py
    echo "✅ test_cameras.py copiado a la raíz"
else
    echo "❌ No se encontró test_cameras.py"
fi

# Limpiar archivos temporales de la raíz
echo "🧹 Limpiando archivos temporales..."
rm -f enhanced_service_factory.py smart_plate_detector.py optimized_config.json services_init.py 2>/dev/null || true

echo ""
echo "🎉 ¡INSTALACIÓN COMPLETADA!"
echo "========================="
echo ""
echo "📋 ARCHIVOS INSTALADOS:"
echo "   ✅ services/service_factory.py (reemplazado)"
echo "   ✅ services/smart_plate_detector.py (nuevo)"
echo "   ✅ services/__init__.py (actualizado)"
echo "   ✅ config/desktop_config.json (actualizado)"
echo "   ✅ test_cameras.py (nuevo, en raíz)"
echo ""
echo "🔧 PRÓXIMOS PASOS:"
echo "   1. Instalar dependencias: pip install pytesseract"
echo "   2. Instalar Tesseract: brew install tesseract"
echo "   3. Probar cámaras: python test_cameras.py"
echo "   4. Ejecutar sistema: python main.py --continuous"
echo ""
echo "💾 Backup guardado en: backup/$(date +%Y%m%d_%H%M%S)/"
