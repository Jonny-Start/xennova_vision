import cv2
import time
import threading
from typing import Dict, List, Optional
import numpy as np
from services.factory_service import UltraServiceFactory

class UltraCameraTester:
    """
    Probador ultra-completo de cámaras:
    - Detección automática de todas las cámaras
    - Pruebas de rendimiento exhaustivas
    - Análisis de calidad de imagen
    - Recomendaciones de configuración
    - Reporte detallado de compatibilidad
    """
    
    def __init__(self):
        self.detected_cameras = {}
        self.test_results = {}
        self.recommendations = {}
        
        # Configuraciones de prueba
        self.test_resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]
        
        self.test_fps_values = [15, 30, 60]
        
        # Métricas de calidad
        self.quality_metrics = [
            'brightness',
            'contrast',
            'sharpness',
            'noise_level',
            'color_balance'
        ]

    def run_complete_test(self) -> Dict:
        """Ejecuta prueba completa de todas las cámaras"""
        print("🧪 Iniciando prueba completa de cámaras...")
        print("=" * 60)
        
        # Fase 1: Detección
        self._detect_all_cameras()
        
        if not self.detected_cameras:
            print("❌ No se detectaron cámaras")
            return {'error': 'No cameras detected'}
        
        # Fase 2: Pruebas individuales
        for camera_id in self.detected_cameras:
            print(f"\n🎥 Probando cámara {camera_id}...")
            self._test_single_camera(camera_id)
        
        # Fase 3: Análisis y recomendaciones
        self._generate_recommendations()
        
        # Fase 4: Reporte final
        report = self._generate_final_report()
        
        print("\n✅ Prueba completa finalizada")
        return report

    def _detect_all_cameras(self):
        """Detecta todas las cámaras disponibles"""
        print("🔍 Detectando cámaras disponibles...")
        
        for i in range(20):  # Buscar en más dispositivos
            try:
                cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    # Probar captura básica
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Obtener información básica
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        
                        # Detectar tipo de cámara
                        camera_type = self._detect_camera_type(i, cap)
                        
                        self.detected_cameras[i] = {
                            'id': i,
                            'type': camera_type,
                            'default_resolution': (width, height),
                            'default_fps': fps,
                            'available': True,
                            'frame_sample': frame.copy()
                        }
                        
                        print(f"  ✅ Cámara {i}: {camera_type} - {width}x{height}@{fps}fps")
                
                cap.release()
                
            except Exception as e:
                print(f"  ⚠️ Error probando cámara {i}: {e}")
                continue
        
        print(f"📊 Total detectadas: {len(self.detected_cameras)} cámaras")

    def _detect_camera_type(self, camera_id: int, cap) -> str:
        """Detecta el tipo de cámara"""
        try:
            # Obtener propiedades para identificar tipo
            backend = cap.getBackendName()
            
            # Intentar obtener más información
            brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = cap.get(cv2.CAP_PROP_CONTRAST)
            
            # Clasificar por características
            if 'DirectShow' in backend:
                return 'USB/Webcam (DirectShow)'
            elif 'V4L2' in backend:
                return 'USB/V4L2 (Linux)'
            elif 'AVFoundation' in backend:
                return 'USB/AVFoundation (macOS)'
            elif brightness == -1 and contrast == -1:
                return 'IP Camera/Virtual'
            else:
                return f'USB Camera ({backend})'
                
        except:
            return 'Unknown Type'

    def _test_single_camera(self, camera_id: int):
        """Prueba exhaustiva de una cámara individual"""
        camera_info = self.detected_cameras[camera_id]
        
        test_result = {
            'camera_id': camera_id,
            'basic_info': camera_info,
            'resolution_tests': {},
            'fps_tests': {},
            'quality_analysis': {},
            'stability_test': {},
            'performance_metrics': {},
            'compatibility_score': 0,
            'recommended_config': {}
        }
        
        print(f"  📐 Probando resoluciones...")
        self._test_resolutions(camera_id, test_result)
        
        print(f"  🎬 Probando FPS...")
        self._test_fps_capabilities(camera_id, test_result)
        
        print(f"  🖼️ Analizando calidad...")
        self._analyze_image_quality(camera_id, test_result)
        
        print(f"  ⚡ Probando estabilidad...")
        self._test_stability(camera_id, test_result)
        
        print(f"  📊 Calculando métricas...")
        self._calculate_performance_metrics(camera_id, test_result)
        
        self.test_results[camera_id] = test_result

    def _test_resolutions(self, camera_id: int, test_result: Dict):
        """Prueba diferentes resoluciones"""
        resolution_results = {}
        
        for width, height in self.test_resolutions:
            try:
                cap = cv2.VideoCapture(camera_id)
                
                if cap.isOpened():
                    # Configurar resolución
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Verificar resolución real
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Probar captura
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Medir tiempo de captura
                        start_time = time.time()
                        for _ in range(10):
                            ret, _ = cap.read()
                        capture_time = (time.time() - start_time) / 10
                        
                        resolution_results[f"{width}x{height}"] = {
                            'requested': (width, height),
                            'actual': (actual_width, actual_height),
                            'supported': True,
                            'capture_time': capture_time,
                            'frame_size': frame.shape,
                            'quality_score': self._calculate_frame_quality(frame)
                        }
                        
                        print(f"    ✅ {width}x{height} -> {actual_width}x{actual_height} ({capture_time:.3f}s)")
                    else:
                        resolution_results[f"{width}x{height}"] = {
                            'requested': (width, height),
                            'supported': False,
                            'error': 'Cannot capture frame'
                        }
                        print(f"    ❌ {width}x{height} - No se puede capturar")
                
                cap.release()
                
            except Exception as e:
                resolution_results[f"{width}x{height}"] = {
                    'requested': (width, height),
                    'supported': False,
                    'error': str(e)
                }
                print(f"    ❌ {width}x{height} - Error: {e}")
        
        test_result['resolution_tests'] = resolution_results

    def _test_fps_capabilities(self, camera_id: int, test_result: Dict):
        """Prueba capacidades de FPS"""
        fps_results = {}
        
        for target_fps in self.test_fps_values:
            try:
                cap = cv2.VideoCapture(camera_id)
                
                if cap.isOpened():
                    # Configurar FPS
                    cap.set(cv2.CAP_PROP_FPS, target_fps)
                    
                    # Verificar FPS configurado
                    actual_fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Medir FPS real
                    measured_fps = self._measure_actual_fps(cap, duration=3)
                    
                    fps_results[f"{target_fps}fps"] = {
                        'target': target_fps,
                        'configured': actual_fps,
                        'measured': measured_fps,
                        'accuracy': abs(measured_fps - target_fps) / target_fps if target_fps > 0 else 1,
                        'stable': abs(measured_fps - actual_fps) < 2
                    }
                    
                    print(f"    📊 {target_fps}fps -> Config: {actual_fps}, Real: {measured_fps:.1f}")
                
                cap.release()
                
            except Exception as e:
                fps_results[f"{target_fps}fps"] = {
                    'target': target_fps,
                    'error': str(e)
                }
                print(f"    ❌ {target_fps}fps - Error: {e}")
        
        test_result['fps_tests'] = fps_results

    def _measure_actual_fps(self, cap, duration: int = 3) -> float:
        """Mide el FPS real de la cámara"""
        frame_count = 0
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
        
        actual_duration = time.time() - start_time
        return frame_count / actual_duration if actual_duration > 0 else 0

    def _analyze_image_quality(self, camera_id: int, test_result: Dict):
        """Analiza la calidad de imagen"""
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                # Capturar múltiples frames para análisis
                frames = []
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames.append(frame)
                
                if frames:
                    # Analizar calidad promedio
                    quality_analysis = {}
                    
                    for metric in self.quality_metrics:
                        scores = [self._calculate_quality_metric(frame, metric) for frame in frames]
                        quality_analysis[metric] = {
                            'average': sum(scores) / len(scores),
                            'std_dev': np.std(scores),
                            'min': min(scores),
                            'max': max(scores)
                        }
                    
                    # Análisis adicional
                    quality_analysis['overall_score'] = self._calculate_overall_quality(frames[0])
                    quality_analysis['consistency'] = self._calculate_frame_consistency(frames)
                    
                    test_result['quality_analysis'] = quality_analysis
                    
                    print(f"    🎨 Calidad general: {quality_analysis['overall_score']:.2f}/10")
                
            cap.release()
            
        except Exception as e:
            test_result['quality_analysis'] = {'error': str(e)}
            print(f"    ❌ Error analizando calidad: {e}")

    def _calculate_quality_metric(self, frame: np.ndarray, metric: str) -> float:
        """Calcula una métrica específica de calidad"""
        try:
            if metric == 'brightness':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return np.mean(gray) / 255.0 * 10
            
            elif metric == 'contrast':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return np.std(gray) / 128.0 * 10
            
            elif metric == 'sharpness':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                return np.var(laplacian) / 1000.0
            
            elif metric == 'noise_level':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                noise = cv2.fastNlMeansDenoising(gray) - gray
                return 10 - (np.std(noise) / 25.0)
            
            elif metric == 'color_balance':
                b, g, r = cv2.split(frame)
                balance = 1 - (np.std([np.mean(b), np.mean(g), np.mean(r)]) / 255.0)
                return balance * 10
            
            return 5.0  # Valor por defecto
            
        except:
            return 0.0

    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calcula calidad general de un frame"""
        try:
            scores = []
            for metric in self.quality_metrics:
                score = self._calculate_quality_metric(frame, metric)
                scores.append(score)
            
            return sum(scores) / len(scores) if scores else 0.0
        except:
            return 0.0

    def _calculate_overall_quality(self, frame: np.ndarray) -> float:
        """Calcula calidad general ponderada"""
        try:
            # Pesos para diferentes métricas
            weights = {
                'brightness': 0.2,
                'contrast': 0.25,
                'sharpness': 0.3,
                'noise_level': 0.15,
                'color_balance': 0.1
            }
            
            weighted_score = 0
            for metric, weight in weights.items():
                score = self._calculate_quality_metric(frame, metric)
                weighted_score += score * weight
            
            return weighted_score
        except:
            return 0.0

    def _calculate_frame_consistency(self, frames: List[np.ndarray]) -> float:
        """Calcula consistencia entre frames"""
        try:
            if len(frames) < 2:
                return 10.0
            
            differences = []
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                mean_diff = np.mean(diff)
                differences.append(mean_diff)
            
            consistency = 10 - (np.mean(differences) / 25.5)
            return max(0, min(10, consistency))
        except:
            return 0.0

    def _test_stability(self, camera_id: int, test_result: Dict):
        """Prueba estabilidad de la cámara"""
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if cap.isOpened():
                print(f"    ⏱️ Probando estabilidad (30s)...")
                
                start_time = time.time()
                frame_count = 0
                error_count = 0
                fps_measurements = []
                
                last_fps_time = start_time
                fps_frame_count = 0
                
                while time.time() - start_time < 30:  # 30 segundos
                    try:
                        ret, frame = cap.read()
                        
                        if ret and frame is not None:
                            frame_count += 1
                            fps_frame_count += 1
                            
                            # Medir FPS cada segundo
                            current_time = time.time()
                            if current_time - last_fps_time >= 1.0:
                                fps = fps_frame_count / (current_time - last_fps_time)
                                fps_measurements.append(fps)
                                fps_frame_count = 0
                                last_fps_time = current_time
                        else:
                            error_count += 1
                        
                        time.sleep(0.01)  # Pequeña pausa
                        
                    except Exception:
                        error_count += 1
                
                duration = time.time() - start_time
                
                stability_result = {
                    'duration': duration,
                    'total_frames': frame_count,
                    'errors': error_count,
                    'error_rate': error_count / (frame_count + error_count) if (frame_count + error_count) > 0 else 1,
                    'average_fps': frame_count / duration,
                    'fps_stability': np.std(fps_measurements) if fps_measurements else 0,
                    'stability_score': self._calculate_stability_score(frame_count, error_count, fps_measurements)
                }
                
                test_result['stability_test'] = stability_result
                
                print(f"    📊 Frames: {frame_count}, Errores: {error_count}, Score: {stability_result['stability_score']:.1f}/10")
            
            cap.release()
            
        except Exception as e:
            test_result['stability_test'] = {'error': str(e)}
            print(f"    ❌ Error en prueba de estabilidad: {e}")

    def _calculate_stability_score(self, frames: int, errors: int, fps_measurements: List[float]) -> float:
        """Calcula score de estabilidad"""
        try:
            # Penalizar errores
            error_penalty = min(5, errors * 0.5)
            
            # Penalizar variabilidad de FPS
            fps_penalty = 0
            if fps_measurements:
                fps_std = np.std(fps_measurements)
                fps_penalty = min(3, fps_std * 0.1)
            
            # Score base
            base_score = 10
            
            # Penalizar si muy pocos frames
            if frames < 100:
                base_score -= 2
            
            final_score = base_score - error_penalty - fps_penalty
            return max(0, min(10, final_score))
        except:
            return 0.0

    def _calculate_performance_metrics(self, camera_id: int, test_result: Dict):
        """Calcula métricas de rendimiento"""
        try:
            metrics = {}
            
            # Score de resoluciones
            resolution_scores = []
            for res_test in test_result['resolution_tests'].values():
                if res_test.get('supported', False):
                    score = res_test.get('quality_score', 0) * (1 / max(0.001, res_test.get('capture_time', 1)))
                    resolution_scores.append(score)
            
            metrics['resolution_performance'] = sum(resolution_scores) / len(resolution_scores) if resolution_scores else 0
            
            # Score de FPS
            fps_scores = []
            for fps_test in test_result['fps_tests'].values():
                if 'accuracy' in fps_test:
                    accuracy_score = (1 - fps_test['accuracy']) * 10
                    stability_bonus = 2 if fps_test.get('stable', False) else 0
                    fps_scores.append(accuracy_score + stability_bonus)
            
            metrics['fps_performance'] = sum(fps_scores) / len(fps_scores) if fps_scores else 0
            
            # Score de calidad
            quality_data = test_result.get('quality_analysis', {})
            metrics['quality_performance'] = quality_data.get('overall_score', 0)
            
            # Score de estabilidad
            stability_data = test_result.get('stability_test', {})
            metrics['stability_performance'] = stability_data.get('stability_score', 0)
            
            # Score general
            scores = [
                metrics['resolution_performance'] * 0.25,
                metrics['fps_performance'] * 0.25,
                metrics['quality_performance'] * 0.3,
                metrics['stability_performance'] * 0.2
            ]
            
            metrics['overall_performance'] = sum(scores)
            
            test_result['performance_metrics'] = metrics
            test_result['compatibility_score'] = metrics['overall_performance']
            
        except Exception as e:
            test_result['performance_metrics'] = {'error': str(e)}

    def _generate_recommendations(self):
        """Genera recomendaciones para cada cámara"""
        for camera_id, test_result in self.test_results.items():
            recommendations = {
                'recommended_for_use': False,
                'best_resolution': None,
                'best_fps': None,
                'configuration': {},
                'warnings': [],
                'optimizations': []
            }
            
            try:
                # Analizar resultados
                compatibility_score = test_result.get('compatibility_score', 0)
                
                if compatibility_score >= 7:
                    recommendations['recommended_for_use'] = True
                    recommendations['rating'] = 'Excelente'
                elif compatibility_score >= 5:
                    recommendations['recommended_for_use'] = True
                    recommendations['rating'] = 'Buena'
                elif compatibility_score >= 3:
                    recommendations['recommended_for_use'] = True
                    recommendations['rating'] = 'Aceptable'
                    recommendations['warnings'].append('Rendimiento limitado')
                else:
                    recommendations['rating'] = 'No recomendada'
                    recommendations['warnings'].append('Rendimiento muy bajo')
                
                # Encontrar mejor resolución
                best_res_score = 0
                best_resolution = None
                
                for res_name, res_data in test_result.get('resolution_tests', {}).items():
                    if res_data.get('supported', False):
                        score = res_data.get('quality_score', 0) / max(0.001, res_data.get('capture_time', 1))
                        if score > best_res_score:
                            best_res_score = score
                            best_resolution = res_data['actual']
                
                recommendations['best_resolution'] = best_resolution
                
                # Encontrar mejor FPS
                best_fps_score = 0
                best_fps = None
                
                for fps_name, fps_data in test_result.get('fps_tests', {}).items():
                    if 'accuracy' in fps_data:
                        score = (1 - fps_data['accuracy']) * 10
                        if fps_data.get('stable', False):
                            score += 2
                        if score > best_fps_score:
                            best_fps_score = score
                            best_fps = fps_data['target']
                
                recommendations['best_fps'] = best_fps
                
                # Configuración recomendada
                recommendations['configuration'] = {
                    'device_id': camera_id,
                    'resolution': best_resolution or (640, 480),
                    'fps': best_fps or 15,
                    'buffer_size': 3 if compatibility_score < 5 else 5
                }
                
                # Optimizaciones específicas
                if compatibility_score < 5:
                    recommendations['optimizations'].extend([
                        'Usar resolución baja (640x480)',
                        'Reducir FPS a 15',
                        'Activar skip_frames=2'
                    ])
                
                if test_result.get('stability_test', {}).get('error_rate', 0) > 0.1:
                    recommendations['warnings'].append('Alta tasa de errores')
                    recommendations['optimizations'].append('Aumentar timeouts')
                
                self.recommendations[camera_id] = recommendations
                
            except Exception as e:
                self.recommendations[camera_id] = {'error': str(e)}

    def _generate_final_report(self) -> Dict:
        """Genera reporte final completo"""
        report = {
            'summary': {
                'total_cameras': len(self.detected_cameras),
                'tested_cameras': len(self.test_results),
                'recommended_cameras': 0,
                'best_camera': None,
                'test_duration': time.time()
            },
            'cameras': {},
            'recommendations': self.recommendations,
            'system_config': None
        }
        
        best_score = 0
        best_camera_id = None
        
        # Procesar cada cámara
        for camera_id, test_result in self.test_results.items():
            camera_report = {
                'basic_info': test_result['basic_info'],
                'compatibility_score': test_result.get('compatibility_score', 0),
                'performance_summary': test_result.get('performance_metrics', {}),
                'recommendation': self.recommendations.get(camera_id, {}),
                'detailed_results': test_result
            }
            
            report['cameras'][camera_id] = camera_report
            
            # Contar recomendadas
            if self.recommendations.get(camera_id, {}).get('recommended_for_use', False):
                report['summary']['recommended_cameras'] += 1
            
            # Encontrar mejor cámara
            score = test_result.get('compatibility_score', 0)
            if score > best_score:
                best_score = score
                best_camera_id = camera_id
        
        report['summary']['best_camera'] = best_camera_id
        
        # Generar configuración del sistema
        if best_camera_id is not None:
            best_config = self.recommendations.get(best_camera_id, {}).get('configuration', {})
            report['system_config'] = {
                'camera': {
                    'type': 'usb',
                    'device_id': best_config.get('device_id', 0),
                    'resolution': best_config.get('resolution', [640, 480]),
                    'fps': best_config.get('fps', 15),
                    'frame_buffer_size': best_config.get('buffer_size', 3)
                },
                'performance': {
                    'skip_frames': 2 if best_score < 5 else 1,
                    'processing_threads': 2 if best_score < 5 else 4
                }
            }
        
        return report

    def print_detailed_report(self):
        """Imprime reporte detallado en consola"""
        print("\n" + "="*80)
        print("🎥 REPORTE COMPLETO DE PRUEBAS DE CÁMARAS")
        print("="*80)
        
        if not self.test_results:
            print("❌ No hay resultados de pruebas disponibles")
            return
        
        # Resumen general
        total_cameras = len(self.detected_cameras)
        recommended = sum(1 for r in self.recommendations.values() if r.get('recommended_for_use', False))
        
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"  Cámaras detectadas: {total_cameras}")
        print(f"  Cámaras probadas: {len(self.test_results)}")
        print(f"  Cámaras recomendadas: {recommended}")
        
        # Detalles por cámara
        for camera_id, test_result in self.test_results.items():
            print(f"\n🎥 CÁMARA {camera_id}")
            print("-" * 40)
            
            # Información básica
            basic_info = test_result['basic_info']
            print(f"  Tipo: {basic_info['type']}")
            print(f"  Resolución por defecto: {basic_info['default_resolution']}")
            print(f"  FPS por defecto: {basic_info['default_fps']}")
            
            # Puntuación de compatibilidad
            score = test_result.get('compatibility_score', 0)
            recommendation = self.recommendations.get(camera_id, {})
            rating = recommendation.get('rating', 'Sin evaluar')
            
            print(f"  Puntuación: {score:.1f}/10 ({rating})")
            print(f"  Recomendada: {'✅ SÍ' if recommendation.get('recommended_for_use', False) else '❌ NO'}")
            
            # Configuración recomendada
            if 'configuration' in recommendation:
                config = recommendation['configuration']
                print(f"  Configuración recomendada:")
                print(f"    - Resolución: {config.get('resolution', 'N/A')}")
                print(f"    - FPS: {config.get('fps', 'N/A')}")
                print(f"    - Buffer: {config.get('buffer_size', 'N/A')}")
            
            # Advertencias
            warnings = recommendation.get('warnings', [])
            if warnings:
                print(f"  ⚠️ Advertencias:")
                for warning in warnings:
                    print(f"    - {warning}")
            
            # Optimizaciones
            optimizations = recommendation.get('optimizations', [])
            if optimizations:
                print(f"  🔧 Optimizaciones sugeridas:")
                for opt in optimizations:
                    print(f"    - {opt}")
        
        # Mejor cámara
        best_camera = max(self.test_results.items(), key=lambda x: x[1].get('compatibility_score', 0))
        if best_camera:
            camera_id, result = best_camera
            print(f"\n🏆 MEJOR CÁMARA: Cámara {camera_id}")
            print(f"  Puntuación: {result.get('compatibility_score', 0):.1f}/10")
            print(f"  Tipo: {result['basic_info']['type']}")
        
        print("\n" + "="*80)

    def save_report_to_file(self, filename: str = None):
        """Guarda reporte completo en archivo JSON"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_test_report_{timestamp}.json"
        
        try:
            import json
            
            report = self._generate_final_report()
            
            # Convertir numpy arrays a listas para JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            clean_report = convert_numpy(report)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(clean_report, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Reporte guardado en: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Error guardando reporte: {e}")
            return None

def main():
    """Función principal para ejecutar las pruebas"""
    print("🚀 Iniciando UltraCameraTester...")
    
    tester = UltraCameraTester()
    
    try:
        # Ejecutar pruebas completas
        report = tester.run_complete_test()
        
        if 'error' not in report:
            # Mostrar reporte detallado
            tester.print_detailed_report()
            
            # Guardar reporte
            tester.save_report_to_file()
            
            # Preguntar si quiere probar la mejor cámara
            best_camera = report['summary']['best_camera']
            if best_camera is not None:
                print(f"\n🎯 ¿Quieres probar la cámara {best_camera} en tiempo real? (s/n): ", end="")
                response = input().lower().strip()
                
                if response in ['s', 'si', 'sí', 'y', 'yes']:
                    print("🚀 Iniciando prueba en tiempo real...")
                    test_best_camera_live(best_camera, report)
        else:
            print(f"❌ Error en las pruebas: {report['error']}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Pruebas interrumpidas por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

def test_best_camera_live(camera_id: int):
    """Prueba la mejor cámara en tiempo real"""
    try:
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ No se pudo abrir la cámara {camera_id}")
            return
        
        print("🎥 Prueba en tiempo real iniciada")
        print("Presiona 'q' para salir, 'c' para capturar screenshot")
        
        # Corregir el problema de WINDOW_RESIZABLE
        try:
            cv2.namedWindow('Camera Test - Live Preview', cv2.WINDOW_NORMAL)
        except:
            cv2.namedWindow('Camera Test - Live Preview')
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if ret:
                frame_count += 1
                
                # Agregar información overlay
                current_time = time.time()
                fps = frame_count / (current_time - start_time)
                
                # Dibujar información
                cv2.putText(frame, f"Camera {camera_id} - FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 'c' to capture", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Camera Test - Live Preview', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"camera_{camera_id}_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot guardado: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        final_fps = frame_count / (time.time() - start_time)
        print(f"✅ Prueba completada - FPS promedio: {final_fps:.1f}")
        
    except Exception as e:
        print(f"❌ Error en prueba en vivo: {e}")

if __name__ == "__main__":
    main()