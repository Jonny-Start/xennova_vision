#!/usr/bin/env python3
"""
XENNOVA VISION ALPR SYSTEM - Optimized Version
High-performance License Plate Recognition System
Compatible with existing project structure
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import queue
import json
from datetime import datetime

# Core imports - Fixed to match your project structure
from utils.logger import get_logger
from utils.hardware_detector import HardwareDetector
from config.settings import ConfigManager
from services.factory_service import UltraServiceFactory
from services.camera_service import UltraCameraService
from services.detector_service import UltraSmartDetector
from core.exceptions import XennovaVisionError

class XennovaVisionSystem:
    """
    Optimized ALPR System with async processing and performance improvements
    """
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize the XENNOVA VISION system"""
        # ✅ Asignar parámetros PRIMERO
        self.debug = debug
        self.verbose = debug  # verbose = debug por simplicidad
        self.config_path = config_path
        
        # System info
        self.version = "2.0 (Optimized)"
        self.logger = None  # ✅ Inicializar como None
        
        # Services
        self.config_manager = None
        self.camera_service = None
        self.detector_service = None
        self.hardware_detector = None
        
        # Threading
        self.threads = []
        self.shutdown_event = threading.Event()
        self.running = False
        
        # Queues for async processing
        self.frame_queue = queue.Queue(maxsize=50)
        self.detection_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Statistics
        self.stats = {
            'start_time': None,
            'frames_processed': 0,
            'detections_made': 0,
            'last_detection_time': None
        }
        
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        # ✅ Usar self.debug si está disponible
        log_level = logging.DEBUG if getattr(self, 'debug', False) else logging.INFO
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create logger
        self.logger = logging.getLogger(__name__)
        
        if getattr(self, 'debug', False):
            self.logger.debug("🐛 Debug logging enabled")
        
        self.logger.info("🚀 XENNOVA VISION ALPR System Starting...")
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load and validate configuration using ConfigManager"""
        try:
            self.logger.info("📋 Loading configuration...")
            
            # Use existing ConfigManager
            self.config_manager = ConfigManager()
            config = self.config_manager.config
            
            # Auto-detect hardware for optimization
            self.hardware_detector = HardwareDetector()
            hw_info = self.hardware_detector.get_system_info()  # ✅ Método correcto
            
            # Optimize config based on hardware
            self._optimize_config_for_hardware(config, hw_info)
            
            self.logger.info(f"✅ Configuration loaded successfully")
            self.logger.info(f"🖥️ Hardware: {self.hardware_detector.get_hardware_type()}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            raise XennovaVisionError(f"Configuration error: {e}")
    
    def _optimize_config_for_hardware(self, config: Dict[str, Any], hw_info: Dict[str, Any]) -> None:
        """Optimize configuration based on detected hardware"""
        
        # CPU optimization
        cpu_cores = hw_info.get('cpu_count', 4)  # ✅ Nombre correcto
        if 'processing' not in config:
            config['processing'] = {}
        config['processing']['max_workers'] = min(cpu_cores - 1, 8)
        
        # Memory optimization
        total_memory = hw_info.get('memory_gb', 8)  # ✅ Nombre correcto
        if 'camera' not in config:
            config['camera'] = {}
        if 'detection' not in config:
            config['detection'] = {}
            
        if total_memory < 4:
            # Low memory mode
            config['camera']['frame_buffer_size'] = 3
            config['detection']['batch_size'] = 1
            config['camera']['resolution'] = [640, 480]
            config['processing']['frame_skip'] = 3
        elif total_memory < 8:
            # Medium memory mode
            config['camera']['frame_buffer_size'] = 5
            config['detection']['batch_size'] = 2
            config['processing']['frame_skip'] = 2
        else:
            # High memory mode
            config['camera']['frame_buffer_size'] = 10
            config['detection']['batch_size'] = 4
            config['processing']['frame_skip'] = 1
            
        # GPU optimization
        if hw_info.get('has_gpu', False):
            config['detection']['use_gpu'] = True
            config['ai'] = config.get('ai', {})
            config['ai']['use_gpu'] = True
        
        self.logger.info(f"⚙️ Configuration optimized for hardware")
    
    def initialize_services(self) -> None:
        """Initialize all services with error handling"""
        try:
            self.logger.info("🔧 Initializing services...")
            
            # Get config from ConfigManager
            config = self.config_manager.config
            
            # Create service factory with optimized config
            self.service_factory = UltraServiceFactory(config)
            
            # Initialize camera service
            self.logger.info("📷 Initializing camera service...")
            self.camera_service = self.service_factory.create_camera_service()
            if not self.camera_service:
                raise XennovaVisionError("Failed to initialize camera service")
            
            # Initialize detector service
            self.logger.info("🤖 Initializing detector service...")
            self.detector_service = self.service_factory.create_plate_detector()
            if not self.detector_service:
                raise XennovaVisionError("Failed to initialize detector service")
            
            # Initialize detector (important!)
            if hasattr(self.detector_service, 'initialize'):
                if not self.detector_service.initialize():
                    raise XennovaVisionError("Failed to initialize detector engine")
            
            self.logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {e}")
            raise XennovaVisionError(f"Service initialization error: {e}")
    
    def setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def frame_capture_worker(self) -> None:
        """Optimized frame capture worker thread"""
        self.logger.info("🎥 Frame capture worker started")
        frame_skip_counter = 0
        skip_frames = self.config_manager.config.get('processing', {}).get('frame_skip', 2)
        
        while not self.shutdown_event.is_set():
            try:
                # Check if camera service is available
                if not self.camera_service.is_available():
                    time.sleep(0.1)
                    continue
                
                # ✅ Capture frame using correct method
                frame = self.camera_service.capture_frame()
                if frame is None:
                    time.sleep(0.01)  # Brief pause if no frame
                    continue
                
                # Smart frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter % skip_frames != 0:
                    continue
                
                # Add to processing queue (non-blocking)
                try:
                    self.frame_queue.put(frame, timeout=0.01)
                    self.stats['frames_processed'] += 1
                except queue.Full:
                    # Drop frame if queue is full (prevents memory buildup)
                    pass
                    
            except Exception as e:
                self.logger.error(f"❌ Frame capture error: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def detection_worker(self) -> None:
        """Optimized detection worker thread"""
        self.logger.info("🔍 Detection worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get frame from queue
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process detection
                detections = self.detector_service.detect_plates(frame)
                
                if detections:
                    self.stats['detections_made'] += len(detections)
                    
                    # Add to result queue
                    result = {
                        'timestamp': datetime.now().isoformat(),
                        'detections': detections,
                        'frame_id': self.stats['frames_processed']
                    }
                    
                    try:
                        self.detection_queue.put(result, timeout=0.01)
                    except queue.Full:
                        pass  # Drop result if queue is full
                
                self.frame_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"❌ Detection error: {e}")
    
    def result_processor_worker(self) -> None:
        """Process and output detection results"""
        self.logger.info("📊 Result processor worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get detection result
                try:
                    result = self.detection_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process result
                self._process_detection_result(result)
                self.detection_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"❌ Result processing error: {e}")
    
    def _process_detection_result(self, result: Dict[str, Any]) -> None:
        """Process individual detection result"""
        timestamp = result['timestamp']
        detections = result['detections']
        
        for detection in detections:
            # Handle both dict and object formats
            if hasattr(detection, 'plate_number'):
                plate_text = detection.plate_number
                confidence = detection.confidence
                bbox = getattr(detection, 'bbox', [])
            else:
                plate_text = detection.get('plate_text', detection.get('plate_number', 'Unknown'))
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', [])
            
            # Log detection with emoji for visibility
            self.logger.info(f"🚗 PLATE DETECTED: {plate_text} (confidence: {confidence:.2f})")
            
            # Add to result queue for external processing
            try:
                self.result_queue.put({
                    'timestamp': timestamp,
                    'plate': plate_text,
                    'confidence': confidence,
                    'bbox': bbox
                }, timeout=0.01)
            except queue.Full:
                pass
    
    def performance_monitor_worker(self) -> None:
        """Monitor and log performance metrics"""
        self.logger.info("📈 Performance monitor started")
        
        while not self.shutdown_event.is_set():
            try:
                time.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                
                if self.stats['start_time']:
                    elapsed = current_time - self.stats['start_time']
                    fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
                    self.stats['fps'] = fps
                    
                    # Log performance with emojis
                    self.logger.info(
                        f"📊 Performance: "
                        f"FPS: {fps:.2f} | "
                        f"Frames: {self.stats['frames_processed']} | "
                        f"Detections: {self.stats['detections_made']} | "
                        f"Queues: F:{self.frame_queue.qsize()} "
                        f"D:{self.detection_queue.qsize()} R:{self.result_queue.qsize()}"
                    )
                    
                    # Get detector stats if available
                    if hasattr(self.detector_service, 'get_stats'):
                        detector_stats = self.detector_service.get_stats()
                        self.logger.info(f"🤖 Detector: {detector_stats}")
                
            except Exception as e:
                self.logger.error(f"❌ Performance monitoring error: {e}")
    
    def start_workers(self) -> None:
        """Start all worker threads"""
        self.stats['start_time'] = time.time()
        
        # Create and start worker threads
        workers = [
            ('FrameCapture', self.frame_capture_worker),
            ('Detection', self.detection_worker),
            ('ResultProcessor', self.result_processor_worker),
            ('PerformanceMonitor', self.performance_monitor_worker)
        ]
        
        for name, worker_func in workers:
            thread = threading.Thread(target=worker_func, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)
            self.logger.info(f"🚀 Started {name} worker thread")
    
    def run_continuous(self) -> None:
        """Run the system in continuous mode"""
        try:
            # Setup
            self.setup_logging()
            config = self.load_configuration()
            self.initialize_services()
            self.setup_signal_handlers()
            
            # ✅ Initialize camera (no start method)
            self.logger.info("📷 Initializing camera service...")
            if not self.camera_service.initialize():
                raise XennovaVisionError("Camera failed to initialize")
            
            # Wait a moment for camera to stabilize
            time.sleep(2)
            
            if not self.camera_service.is_available():  # ✅ Método correcto
                raise XennovaVisionError("Camera is not available")
            
            self.logger.info("✅ Camera service initialized successfully")
            
            # Start worker threads
            self.start_workers()
            self.running = True
            
            self.logger.info("🎯 System running! Press Ctrl+C to stop.")
            self.logger.info("👀 Watching for license plates...")
            
            # Main loop - just keep alive and monitor
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1)
                
                # Health check
                if not self.camera_service.is_available():  # ✅ Método correcto
                    self.logger.error("❌ Camera service stopped unexpectedly")
                    break
            
        except KeyboardInterrupt:
            self.logger.info("⌨️ Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"💥 System error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            raise
        finally:
            self.shutdown()
    
    def run_single_shot(self, image_path: str) -> None:
        """Run detection on a single image"""
        try:
            self.setup_logging()
            self.logger.info(f"🖼️ Processing single image: {image_path}")
            
            config = self.load_configuration()
            self.initialize_services()
            
            # Process image
            import cv2
            frame = cv2.imread(image_path)
            if frame is None:
                raise XennovaVisionError(f"Could not load image: {image_path}")
            
            self.logger.info(f"📐 Image size: {frame.shape}")
            
            # ✅ Agregar debug del detector
            if self.debug:
                self.logger.debug("🔍 Starting plate detection...")
                
                # Mostrar estadísticas del detector antes
                if hasattr(self.detector_service, 'get_stats'):
                    stats_before = self.detector_service.get_stats()
                    self.logger.debug(f"📊 Detector stats before: {stats_before}")
            
            detections = self.detector_service.detect_plates(frame)
            
            # ✅ Más debug después de la detección
            if self.debug:
                self.logger.debug(f"🎯 Detection result type: {type(detections)}")
                self.logger.debug(f"🎯 Detection result: {detections}")
                
                if hasattr(self.detector_service, 'get_stats'):
                    stats_after = self.detector_service.get_stats()
                    self.logger.debug(f"📊 Detector stats after: {stats_after}")
            
            if detections:
                self.logger.info(f"🎯 Found {len(detections)} plate(s):")
                for i, detection in enumerate(detections, 1):
                    if hasattr(detection, 'plate_number'):
                        plate_text = detection.plate_number
                        confidence = detection.confidence
                    else:
                        plate_text = detection.get('plate_text', detection.get('plate_number', 'Unknown'))
                        confidence = detection.get('confidence', 0.0)
                    
                    self.logger.info(f"  {i}. 🚗 {plate_text} (confidence: {confidence:.2f})")
            else:
                self.logger.info("❌ No plates detected")
                if self.debug:
                    self.logger.debug("💡 Try adjusting detection parameters or check image quality")
                
        except Exception as e:
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.error(f"💥 Single shot processing error: {e}")
            else:
                print(f"💥 Single shot processing error: {e}")
                
            if hasattr(self, 'debug') and self.debug:
                import traceback
                traceback.print_exc()
            raise
    
    def shutdown(self) -> None:
        """Graceful shutdown"""
        if not self.running:
            return
            
        self.logger.info("🛑 Shutting down system...")
        self.running = False
        self.shutdown_event.set()
        
        # ✅ Stop camera service using correct method
        if self.camera_service:
            try:
                self.camera_service.release()  # ✅ Método correcto
                self.logger.info("📷 Camera service stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping camera: {e}")
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        # Cleanup detector
        if self.detector_service and hasattr(self.detector_service, 'cleanup'):
            self.detector_service.cleanup()
        
        # Final stats
        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
            avg_fps = self.stats['frames_processed'] / total_time if total_time > 0 else 0
            
            self.logger.info(f"📊 Final Statistics:")
            self.logger.info(f"  ⏱️ Runtime: {total_time:.2f}s")
            self.logger.info(f"  🎞️ Frames processed: {self.stats['frames_processed']}")
            self.logger.info(f"  🎯 Detections made: {self.stats['detections_made']}")
            self.logger.info(f"  📈 Average FPS: {avg_fps:.2f}")
        
        self.logger.info("✅ System shutdown complete")


def main():
    """Main entry point with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description="🚀 XENNOVA VISION ALPR System - Optimized License Plate Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Examples:
  python main.py --continuous                    # Run with auto-detected config
  python main.py --continuous --debug           # Run with debug logging
  python main.py --image img/test/testPlaca.jpg # Process single image
  python main.py --test-camera                  # Test camera only
  python main.py --version                      # Show version info
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--continuous', 
        action='store_true',
        help='🔄 Run in continuous detection mode'
    )
    mode_group.add_argument(
        '--image', 
        type=str,
        help='🖼️ Process a single image file'
    )
    mode_group.add_argument(
        '--test-camera', 
        action='store_true',
        help='🧪 Test camera functionality only'
    )
    
    # Configuration options
    parser.add_argument(
        '--config', 
        type=str,
        help='📋 Path to configuration file (auto-detected if not specified)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='🐛 Enable debug logging'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='📢 Enable verbose output'
    )
    parser.add_argument(
        '--version', 
        action='version',
        version='🚀 XENNOVA VISION ALPR System v2.0 (Optimized)'
    )
    
    args = parser.parse_args()
    
    # Handle test camera mode
    if args.test_camera:
        print("🧪 Running camera test...")
        import subprocess
        subprocess.run([sys.executable, "camera_test.py"])
        return
    
    # Create and run system
    try:
        system = XennovaVisionSystem(
            config_path=args.config,
            debug=args.debug or args.verbose
        )
        
        if args.continuous:
            system.run_continuous()
        elif args.image:
            system.run_single_shot(args.image)
            
    except XennovaVisionError as e:
        print(f"❌ XENNOVA VISION Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        if args.debug or (hasattr(args, 'verbose') and args.verbose):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()