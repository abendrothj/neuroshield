import numpy as np
import tensorflow as tf
import json
import logging
import os
from threat_detection_system import ThreatDetectionSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_test.log'),
        logging.StreamHandler()
    ]
)

def test_model():
    """Test the deployed model with sample data"""
    try:
        # Get the absolute path to the model
        model_dir = os.path.join(os.path.dirname(__file__), "models", "threat_detection_20250403_212211")
        model_path = os.path.join(model_dir, "model.keras")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logging.info(f"Loading model from {model_path}")
        
        # Initialize the system with the model path
        system = ThreatDetectionSystem(model_path=model_dir)
        
        # Load the model
        system.model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully")
        
        # Generate test data with correct number of features (39)
        test_data = np.random.randn(5, 39)  # 5 samples, 39 features
        test_data = np.clip(test_data, 0, 1)  # Normalize to [0,1] range
        
        # Make predictions
        predictions = system.model.predict(test_data)
        logging.info("Predictions made successfully")
        
        # Print results
        for i, pred in enumerate(predictions):
            threat_class = np.argmax(pred)
            confidence = float(pred[threat_class])
            threat_name = system.threat_types.get(threat_class, f"Unknown ({threat_class})")
            
            logging.info(f"Sample {i+1}:")
            logging.info(f"  Predicted Threat: {threat_name}")
            logging.info(f"  Confidence: {confidence:.4f}")
            logging.info(f"  Raw Probabilities: {pred}")
            
        # Load and display metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            logging.info("\nModel Metadata:")
            for key, value in metadata.items():
                logging.info(f"  {key}: {value}")
                
    except Exception as e:
        logging.error(f"Error testing model: {str(e)}")
        raise

if __name__ == "__main__":
    test_model() 