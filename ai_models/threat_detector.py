import sys
import json
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

class ThreatDetector:
    def __init__(self):
        # Initialize the model and scaler
        self.model = self._load_model()
        self.scaler = StandardScaler()
        
        # Define severity levels
        self.severity_levels = {
            0: "LOW",
            1: "MEDIUM",
            2: "HIGH",
            3: "CRITICAL"
        }

    def _load_model(self):
        try:
            # Load the pre-trained model
            model_path = "ai_models/threat_detection_model.h5"
            return keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return None

    def preprocess_data(self, data):
        try:
            # Extract relevant features from the input data
            features = [
                data.get('packet_size', 0),
                data.get('protocol_type', 0),
                data.get('service', 0),
                data.get('flag', 0),
                data.get('src_bytes', 0),
                data.get('dst_bytes', 0),
                data.get('land', 0),
                data.get('wrong_fragment', 0),
                data.get('urgent', 0)
            ]
            
            # Scale the features
            scaled_features = self.scaler.fit_transform([features])
            return scaled_features
        except Exception as e:
            print(f"Error preprocessing data: {e}", file=sys.stderr)
            return None

    def detect_threat(self, data):
        try:
            if self.model is None:
                return {
                    "isThreat": False,
                    "severity": "LOW",
                    "description": "Model not loaded",
                    "confidence": 0.0,
                    "metadata": {"error": "Model not available"}
                }

            # Preprocess the input data
            processed_data = self.preprocess_data(data)
            if processed_data is None:
                return {
                    "isThreat": False,
                    "severity": "LOW",
                    "description": "Failed to preprocess data",
                    "confidence": 0.0,
                    "metadata": {"error": "Data preprocessing failed"}
                }

            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            confidence = float(np.max(prediction))
            threat_class = int(np.argmax(prediction))
            
            # Determine if it's a threat
            is_threat = threat_class > 0
            
            # Get severity level
            severity = self.severity_levels.get(threat_class, "LOW")
            
            # Generate description
            description = self._generate_description(threat_class, confidence)
            
            return {
                "isThreat": is_threat,
                "severity": severity,
                "description": description,
                "confidence": confidence,
                "metadata": {
                    "threat_class": threat_class,
                    "raw_prediction": prediction.tolist()
                }
            }
        except Exception as e:
            print(f"Error in threat detection: {e}", file=sys.stderr)
            return {
                "isThreat": False,
                "severity": "LOW",
                "description": "Error in threat detection",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

    def _generate_description(self, threat_class, confidence):
        threat_types = {
            0: "Normal traffic",
            1: "Probe attack",
            2: "DoS attack",
            3: "R2L attack",
            4: "U2R attack"
        }
        
        threat_type = threat_types.get(threat_class, "Unknown")
        confidence_percentage = f"{confidence * 100:.2f}%"
        
        return f"Detected {threat_type} with {confidence_percentage} confidence"

def main():
    detector = ThreatDetector()
    
    while True:
        try:
            # Read input from stdin
            line = sys.stdin.readline()
            if not line:
                break
                
            # Parse input data
            data = json.loads(line)
            
            # Detect threat
            result = detector.detect_threat(data)
            
            # Write result to stdout
            print(json.dumps(result))
            sys.stdout.flush()
            
        except json.JSONDecodeError as e:
            print(json.dumps({
                "isThreat": False,
                "severity": "LOW",
                "description": "Invalid JSON input",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }), file=sys.stdout)
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({
                "isThreat": False,
                "severity": "LOW",
                "description": "Internal error",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }), file=sys.stdout)
            sys.stdout.flush()

if __name__ == "__main__":
    main() 