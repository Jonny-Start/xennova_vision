#!/usr/bin/env python3
"""
Script de prueba para el sistema de detección de placas optimizado
Permite probar diferentes cámaras y modos de detección fácilmente
"""

import cv2
import json
import time
import sys
import os
from services.fixed_service_factory import ServiceFactory

def test_all_cameras():
    """Prueba todas las cámaras disponibles"""
    print("=" * 50)
    print("🔍 PROBANDO TODAS LAS CÁMARAS DISPONIBLES")
    print("=" * 50)

    available = ServiceFactory.get_available_cameras()

    if not available:
        print("❌ No se encontraron cámaras")
        return None

    print(f"\n📱 Cámaras encontradas: {len(available)}")
    for device_id, info in available.items():
        print(f"   {device_id}: {info}")

    # Permitir al usuario elegir
    print("\n¿Qué cámara quieres usar?")
    try:
        choice = int(input("Ingresa el número del dispositivo (0-9): "))
        if choice in available:
            print(f"✅ Seleccionaste: {available[choice]}")
            return choice
        else:
            print("❌ Opción inválida, usando cámara 0")
            return 0
    except:
        print("❌ Entrada inválida, usando cámara 0")
        return 0

def create_test_config(camera_id, detector_mode='smart'):
    """Crea configuración de prueba"""
    return {
        "camera": {
            "type": "usb",
            "device_id": camera_id,
            "resolution": [1280, 720],
            "fps": 30
        },
        "ai_model": {
            "detector_mode": detector_mode,
            "confidence_threshold": 0.85,
            "min_confirmations": 3,
            "enable_cache": True
        },
        "display": {
            "show_window": True,
            "window_size": [800, 600]
        }
    }

def run_detection_test(camera_id, detector_mode='smart', duration=60):
    """Ejecuta prueba de detección"""
    print(f"\n🚀 INICIANDO PRUEBA CON DETECTOR '{detector_mode.upper()}'")
    print(f"⏱️  Duración: {duration} segundos")
    print("🎯 Busca una placa frente a la cámara")
    print("❌ Presiona 'q' para salir antes")
    print("-" * 50)

    # Crear configuración y servicios
    config = create_test_config(camera_id, detector_mode)

    try:
        services = ServiceFactory.create_optimized_services()
        services['config'] = config

        camera = ServiceFactory.create_camera_adapter(config)
        detector = ServiceFactory.create_plate_detector(config)

        if not camera.is_connected():
            print(f"❌ No se pudo conectar a la cámara {camera_id}")
            return

        if not detector.is_ready():
            print("❌ Detector no está listo")
            return

        print("✅ Servicios iniciados correctamente")
        print("🔄 Iniciando detección...")

        start_time = time.time()
        frame_count = 0
        detections = 0
        confirmed_plates = set()

        while time.time() - start_time < duration:
            # Capturar frame
            frame_data = camera.capture_frame()
            if not frame_data:
                continue

            frame_count += 1

            # Detectar placa
            result = detector.detect_plate(frame_data)

            # Mostrar frame si está configurado
            if config.get('display', {}).get('show_window', False):
                import numpy as np
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Agregar información al frame
                info_text = f"Modo: {detector_mode.upper()}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if result.plate_text:
                    detections += 1
                    plate_text = f"Placa: {result.plate_text}"
                    confidence_text = f"Confianza: {result.confidence:.2f}"
                    status_text = f"Estado: {result.additional_info or 'Detectada'}"

                    cv2.putText(frame, plate_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, status_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Si está confirmada
                    if result.is_valid and "RAPIDO" in str(result.additional_info):
                        confirmed_plates.add(result.plate_text)
                        cv2.putText(frame, "¡CONFIRMADA!", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Estadísticas
                fps = frame_count / (time.time() - start_time)
                stats = f"FPS: {fps:.1f} | Frames: {frame_count} | Detecciones: {detections}"
                cv2.putText(frame, stats, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow(f"Detector {detector_mode.upper()}", frame)

                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Log de detecciones importantes
            if result.plate_text and result.is_valid:
                print(f"🎯 {result.additional_info}: {result.plate_text} (confianza: {result.confidence:.2f})")

            time.sleep(0.1)  # Pequeña pausa para no saturar

        # Estadísticas finales
        elapsed = time.time() - start_time
        fps = frame_count / elapsed

        print("\n" + "=" * 50)
        print("📊 ESTADÍSTICAS FINALES")
        print("=" * 50)
        print(f"⏱️  Tiempo total: {elapsed:.1f}s")
        print(f"🖼️  Frames procesados: {frame_count}")
        print(f"⚡ FPS promedio: {fps:.1f}")
        print(f"🔍 Detecciones totales: {detections}")
        print(f"✅ Placas confirmadas: {len(confirmed_plates)}")

        for plate in confirmed_plates:
            print(f"   🚀 {plate}")

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        cv2.destroyAllWindows()

def main():
    """Función principal"""
    print("🚗 SISTEMA DE DETECCIÓN DE PLACAS - MODO PRUEBA")
    print("=" * 60)

    # Probar cámaras disponibles
    camera_id = test_all_cameras()
    if camera_id is None:
        print("❌ No se puede continuar sin cámara")
        return

    # Elegir modo de detector
    print("\n🧠 ¿Qué modo de detector quieres probar?")
    print("1. Smart (Análisis inteligente con confirmación)")
    print("2. Fast (Optimizado para velocidad)")
    print("3. Advanced (Con YOLO si está disponible)")

    mode_map = {'1': 'smart', '2': 'fast', '3': 'advanced'}

    try:
        choice = input("Elige opción (1-3): ").strip()
        detector_mode = mode_map.get(choice, 'smart')
    except:
        detector_mode = 'smart'

    print(f"✅ Modo seleccionado: {detector_mode.upper()}")

    # Duración de la prueba
    try:
        duration = int(input("\n⏱️ ¿Cuántos segundos quieres probar? (default: 60): ") or 60)
    except:
        duration = 60

    # Ejecutar prueba
    run_detection_test(camera_id, detector_mode, duration)

    print("\n🎉 ¡Prueba completada!")
    print("💡 Consejos:")
    print("   • Usa iluminación uniforme")
    print("   • Mantén la placa estable frente a la cámara")
    print("   • El modo 'smart' es mejor para precisión")
    print("   • El modo 'fast' es mejor para velocidad")

if __name__ == "__main__":
    main()
