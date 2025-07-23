import unittest
from unittest.mock import Mock, patch
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import PlateEvent, DetectionResult
from services.plate_detector import LightweightPlateDetector
from utils.hardware_detector import HardwareDetector

class TestPlateEvent(unittest.TestCase):
    def test_plate_event_creation(self):
        """Prueba la creación de eventos de placa"""
        from datetime import datetime
        
        event = PlateEvent(
            id="test-123",
            plate_text="ABC123",
            timestamp=datetime.now(),
            confidence=0.95
        )
        
        self.assertEqual(event.plate_text, "ABC123")
        self.assertEqual(event.confidence, 0.95)
        self.assertFalse(event.sent)
    
    def test_plate_event_to_dict(self):
        """Prueba la serialización a diccionario"""
        from datetime import datetime
        
        event = PlateEvent(
            id="test-123",
            plate_text="ABC123",
            timestamp=datetime.now(),
            confidence=0.95
        )
        
        event_dict = event.to_dict()
        self.assertIn('id', event_dict)
        self.assertIn('plate_text', event_dict)
        self.assertIn('confidence', event_dict)

class TestHardwareDetector(unittest.TestCase):
    def test_hardware_detection(self):
        """Prueba la detección de hardware"""
        detector = HardwareDetector()
        platform_info = detector.get_platform_info()
        
        self.assertIn('system', platform_info)
        self.assertIn('memory_gb', platform_info)
        self.assertIn('cpu_count', platform_info)
        self.assertIn('is_embedded', platform_info)

class TestDetectionResult(unittest.TestCase):
    def test_detection_result_creation(self):
        """Prueba la creación de resultados de detección"""
        result = DetectionResult(
            plate_text="XYZ789",
            confidence=0.88,
            bounding_box={"x": 100, "y": 50, "w": 200, "h": 80},
            processing_time=0.15,
            success=True
        )
        
        self.assertEqual(result.plate_text, "XYZ789")
        self.assertEqual(result.confidence, 0.88)
        self.assertTrue(result.success)

if __name__ == '__main__':
    unittest.main()