import pytest
import numpy as np
from src.models.threat_detector import ThreatDetector

@pytest.fixture
def threat_detector():
    return ThreatDetector()

def test_predict_threat_level():
    detector = ThreatDetector()
    input_data = np.array([
        [0.1, 0.2, 0.3, 0.4],  # Normal traffic
        [0.9, 0.8, 0.7, 0.9],  # Suspicious traffic
    ])
    
    predictions = detector.predict(input_data)
    
    assert predictions.shape == (2,)
    assert 0 <= predictions[0] <= 1  # Normal traffic should have low threat score
    assert 0.5 < predictions[1] <= 1  # Suspicious traffic should have high threat score

@pytest.mark.asyncio
async def test_async_threat_detection():
    detector = ThreatDetector()
    input_data = np.array([[0.5, 0.5, 0.5, 0.5]])
    
    result = await detector.async_predict(input_data)
    
    assert isinstance(result, dict)
    assert 'threat_level' in result
    assert 'confidence' in result
    assert 0 <= result['threat_level'] <= 1
    assert 0 <= result['confidence'] <= 1

def test_invalid_input():
    detector = ThreatDetector()
    with pytest.raises(ValueError):
        detector.predict(np.array([]))  # Empty input
    
    with pytest.raises(ValueError):
        detector.predict(np.array([[1, 2]]))  # Wrong input shape 