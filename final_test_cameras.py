#!/usr/bin/env python3
"""
Script de prueba final - Xennova Vision
Completamente compatible con la estructura existente
"""

import cv2
import json
import time
import sys
import os
import numpy as np

# Import absoluto del factory corregido
from fixed_service_factory import ServiceFactory

def show_available_cameras():
    """Muestra todas las cámaras disponibles"""
    print("=" * 50)
    print("🔍 DETECTANDO CÁMARAS DISPONIBLES")
    print("=" * 50)

    available = ServiceFactory.get_available_cameras()

    if not available:
        print("❌ No se encontraron cámaras USB")
        print("💡 Asegúrate de que:")
        print("   • Tienes una cámara USB conectada")
        print("   • La cámara no está siendo usada por otra aplicación")
        print("   • Los drivers están instalados correctamente")
        return None

    print(f"\n📱 Se encontraron {len(available)} cámaras:")
    for device_id, info in available.items():
        print(f"   [{device_id}] {info}")

    return available

def choose_camera(available_cameras):
    """Permite elegir una cámara"""
    if not available_cameras:
        return None

    print("\n🎥 SELECCIÓN DE CÁMARA:")

    # Si solo hay una, usarla automáticamente
    if len(available_cameras) == 1:
        camera_id = list(available_cameras.keys())[0]
        print(f"✅ Usando única cámara disponible: {available_cameras[camera_id]}")
        return camera_id

    # Permitir elegir
    print("¿Qué cámara quieres usar?")
    try:
        choice = input(f"Ingresa el número (0-{max(available_cameras.keys())}): ").strip()
        camera_id = int(choice)

        if camera_id in available_cameras:
            print(f"✅ Seleccionaste: {available_cameras[camera_id]}")
            return camera_id
        else:
            print(f"❌ Opción inválida. Usando cámara 0")
            return 0
    except:
        print("❌ Entrada inválida. Usando cámara 0")
        return 0

def choose_detector_mode():
    """Permite elegir el modo del detector"""
    print("\n🧠 MODOS DE DETECTOR:")
    print("   [1] Smart - Análisis inteligente con confirmación múltiple")
    print("       • Confirma la placa 3 veces antes de estar seguro")
    print("       • Dice 'CONFIRMADA' cuando está 100% seguro")
    print("       • Mejor precisión, un poco más lento")
    print()
    print("   [2] Fast  - Detección rápida y directa")
    print("       • Procesamiento súper rápido")
    print("       • Menos confirmaciones")
    print("       • Mejor para pruebas rápidas")

    try:
        choice = input("\nElige modo (1-2, default=1): ").strip()

        if choice == '2':
            print("✅ Modo Fast seleccionado")
            return 'fast'
        else:
            print("✅ Modo Smart seleccionado (recomendado)")
            return 'smart'
    except:
        return 'smart'

def create_config(camera_id, detector_mode):
    """Crea configuración para la prueba"""
    return {
        "camera": {
            "device_id": camera_id,
            "resolution": [1280, 720],  # Resolución optimizada
            "fps": 30
        },
        "ai_model": {
            "detector_mode": detector_mode,
            "confidence_threshold": 0.85,
            "min_confirmations": 3
        },
        "storage": {
            "type": "simple"
        },
        "network": {
            "endpoint": "http://localhost:3000/api/placa"
        }
    }

def run_detection_loop(services, duration=60):
    """Ejecuta el loop principal de detección"""
    camera = services['camera']
    detector = services['detector']
    config = services['config']

    print(f"\n🚀 INICIANDO SISTEMA DE DETECCIÓN")
    print("=" * 50)
    print(f"⏱️  Duración máxima: {duration} segundos")
    print(f"🧠 Detector: {config['ai_model']['detector_mode'].upper()}")
    print(f"🎥 Cámara: dispositivo {config['camera']['device_id']}")
    print(f"📐 Resolución: {config['camera']['resolution'][0]}x{config['camera']['resolution'][1]}")
    print()
    print("🎯 INSTRUCCIONES:")
    print("   • Muestra una placa colombiana frente a la cámara")
    print("   • Mantén la placa estable y bien iluminada")
    print("   • Presiona 'q' en la ventana para salir")
    print("   • Presiona 'r' para reiniciar el historial del detector")
    print("-" * 50)

    start_time = time.time()
    frame_count = 0
    total_detections = 0
    confirmed_plates = set()
    last_detection_time = 0

    try:
        while time.time() - start_time < duration:
            # Capturar frame
            frame = camera.capture_frame()

            if frame is None:
                print("⚠️ No se pudo capturar frame, reintentando...")
                time.sleep(0.1)
                continue

            frame_count += 1
            current_time = time.time()

            # Detectar placas (no en cada frame para optimizar)
            if current_time - last_detection_time > 0.1:  # Cada 100ms
                detections = detector.detect_plates(frame)
                last_detection_time = current_time

                # Procesar detecciones
                for detection in detections:
                    total_detections += 1
                    plate_text = detection.plate_number
                    confidence = detection.confidence

                    # Determinar si es confirmación
                    is_confirmed = confidence > 0.9

                    if is_confirmed:
                        confirmed_plates.add(plate_text)
                        # No imprimir cada vez para evitar spam
            else:
                detections = []

            # Crear frame de visualización
            display_frame = frame.copy()

            # Información del sistema
            mode_text = f"Modo: {config['ai_model']['detector_mode'].upper()}"
            cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Mostrar detecciones en el frame
            for detection in detections:
                if detection.bbox:
                    x1, y1, x2, y2 = detection.bbox

                    # Color y estado basado en confianza
                    if detection.confidence > 0.9:
                        color = (0, 0, 255)  # Rojo para confirmadas
                        status = "CONFIRMADA"
                        thickness = 3
                    elif detection.confidence > 0.7:
                        color = (0, 255, 255)  # Amarillo para candidatas
                        status = "CANDIDATA"
                        thickness = 2
                    else:
                        color = (255, 0, 0)  # Azul para detecciones débiles
                        status = "DETECTADA"
                        thickness = 1

                    # Dibujar bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                    # Texto de la placa
                    plate_text = f"{detection.plate_number}"
                    confidence_text = f"({detection.confidence:.2f})"

                    # Fondo para el texto
                    text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1-35), (x1 + text_size[0] + 10, y1), color, -1)

                    # Texto de la placa
                    cv2.putText(display_frame, plate_text, (x1+5, y1-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Confianza
                    cv2.putText(display_frame, confidence_text, (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Estado
                    cv2.putText(display_frame, status, (x1, y2+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Estadísticas en tiempo real
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Panel de estadísticas
            stats_y = display_frame.shape[0] - 80
            cv2.rectangle(display_frame, (10, stats_y-5), (500, display_frame.shape[0]-10), (0, 0, 0), -1)

            stats_lines = [
                f"FPS: {fps:.1f}",
                f"Frames: {frame_count}",
                f"Detecciones: {total_detections}",
                f"Confirmadas: {len(confirmed_plates)}"
            ]

            for i, line in enumerate(stats_lines):
                cv2.putText(display_frame, line, (15, stats_y + 15 + i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Lista de placas confirmadas
            if confirmed_plates:
                plates_text = f"Placas: {', '.join(sorted(confirmed_plates))}"
                cv2.putText(display_frame, plates_text, (15, stats_y + 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Mostrar ventana
            cv2.imshow("Xennova Vision - Detector de Placas Inteligente", display_frame)

            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️ Saliendo por petición del usuario")
                break
            elif key == ord('r'):
                if hasattr(detector, 'reset_history'):
                    detector.reset_history()
                    confirmed_plates.clear()
                    total_detections = 0
                    print("🔄 Historial reiniciado")

            # Pequeña pausa para no saturar CPU
            time.sleep(0.01)

        # Estadísticas finales
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS FINALES")
        print("=" * 60)
        print(f"⏱️  Tiempo total: {elapsed:.1f} segundos")
        print(f"🖼️  Frames procesados: {frame_count}")
        print(f"⚡ FPS promedio: {fps:.1f}")
        print(f"🔍 Detecciones totales: {total_detections}")
        print(f"✅ Placas confirmadas: {len(confirmed_plates)}")

        if confirmed_plates:
            print("\n🎯 PLACAS CONFIRMADAS:")
            for i, plate in enumerate(sorted(confirmed_plates), 1):
                print(f"   {i}. 🚀 {plate}")
        else:
            print("\n💡 CONSEJOS PARA MEJORES RESULTADOS:")
            print("   • Usa iluminación uniforme y brillante")
            print("   • Mantén la placa perpendicular a la cámara")
            print("   • Asegúrate de que la placa esté limpia y legible")
            print("   • Evita reflejos y sombras")
            print("   • Mantén la placa estable por unos segundos")

        print("\n🔧 CONFIGURACIÓN USADA:")
        print(f"   • Detector: {config['ai_model']['detector_mode']}")
        print(f"   • Cámara: {config['camera']['device_id']}")
        print(f"   • Resolución: {config['camera']['resolution'][0]}x{config['camera']['resolution'][1]}")
        print(f"   • Umbral confianza: {config['ai_model']['confidence_threshold']}")

    except KeyboardInterrupt:
        print("\n⏹️ Interrumpido por el usuario (Ctrl+C)")

    finally:
        cv2.destroyAllWindows()
        camera.release()

def main():
    """Función principal"""
    print("🚗 XENNOVA VISION - SISTEMA INTELIGENTE DE DETECCIÓN DE PLACAS")
    print("=" * 70)
    print("🎯 Versión optimizada con detector inteligente")
    print("🇨🇴 Especializado en placas colombianas")
    print()

    # Paso 1: Detectar cámaras
    available_cameras = show_available_cameras()
    if not available_cameras:
        print("\n❌ No se puede continuar sin cámara")
        print("💡 Conecta una cámara USB y vuelve a intentar")
        return

    # Paso 2: Elegir cámara
    camera_id = choose_camera(available_cameras)
    if camera_id is None:
        return

    # Paso 3: Elegir modo de detector
    detector_mode = choose_detector_mode()

    # Paso 4: Configurar duración
    try:
        duration_input = input("\n⏱️ ¿Cuántos segundos de prueba? (default: 120): ").strip()
        duration = int(duration_input) if duration_input else 120
    except:
        duration = 120

    # Paso 5: Crear configuración
    config = create_config(camera_id, detector_mode)

    # Paso 6: Crear servicios
    print("\n🔧 Inicializando servicios del sistema...")
    services = ServiceFactory.create_all_services(config)

    if not services:
        print("❌ Error crítico inicializando servicios")
        print("💡 Verifica que:")
        print("   • La cámara no esté siendo usada por otra app")
        print("   • Tienes permisos para acceder a la cámara")
        print("   • Las dependencias están instaladas (easyocr, pytesseract)")
        return

    # Paso 7: Ejecutar detección
    print("\n🎬 ¡Todo listo! Iniciando detección...")
    time.sleep(1)  # Pequeña pausa dramática

    run_detection_loop(services, duration)

    print("\n🎉 ¡SESIÓN DE DETECCIÓN COMPLETADA!")
    print("\n📝 PRÓXIMOS PASOS:")
    print("   • Para cambiar cámara: ejecuta este script nuevamente")
    print("   • Para integrar en tu sistema: usa fixed_service_factory.py")
    print("   • Para configuración permanente: edita config/desktop_config.json")
    print("\n¡Gracias por usar Xennova Vision! 🚗✨")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("\n🔧 SOLUCIÓN DE PROBLEMAS:")
        print("   1. Verifica dependencias: pip install opencv-python easyocr pytesseract")
        print("   2. Instala Tesseract: brew install tesseract (macOS)")
        print("   3. Cierra otras apps que usen la cámara")
        print("   4. Reinicia el terminal y vuelve a intentar")
        import traceback
        traceback.print_exc()
