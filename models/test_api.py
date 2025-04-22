import requests
import json
import numpy as np
import logging
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_test.log'),
        logging.StreamHandler()
    ]
)

def test_api():
    """Test the API with sample data"""
    try:
        # API configuration
        base_url = "http://localhost:5000"
        
        # Test health endpoint
        logging.info("Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health")
        health_data = health_response.json()
        logging.info(f"Health check response: {json.dumps(health_data, indent=2)}")
        
        if health_data["status"] != "healthy":
            raise Exception("API is not healthy")
            
        # Generate sample data
        num_samples = 3
        num_features = 39  # Match the model's input shape (39 features)
        
        samples = []
        for i in range(num_samples):
            # Generate random features
            features = np.random.randn(num_features)
            features = np.clip(features, 0, 1)  # Normalize to [0,1] range
            
            # Create sample with feature names
            sample = {f"feature_{j}": float(features[j]) for j in range(num_features)}
            samples.append(sample)
        
        # Test analyze endpoint
        logging.info("\nTesting analyze endpoint...")
        analyze_response = requests.post(
            f"{base_url}/analyze",
            json={"data": samples}
        )
        
        if analyze_response.status_code != 200:
            logging.error(f"Error response: {analyze_response.text}")
            raise Exception(f"API returned error: {analyze_response.status_code}")
            
        analyze_data = analyze_response.json()
        logging.info(f"Raw response: {json.dumps(analyze_data, indent=2)}")
        
        # Print results - handle different response formats
        logging.info("\nAnalysis Results:")
        if "results" in analyze_data:
            for i, result in enumerate(analyze_data["results"]):
                logging.info(f"\nSample {i+1}:")
                if "threat_level" in result:
                    logging.info(f"  Threat Level: {result['threat_level']}")
                if "confidence" in result:
                    logging.info(f"  Confidence: {result['confidence']:.4f}")
                if "probabilities" in result:
                    logging.info(f"  Probabilities: {result['probabilities']}")
        else:
            logging.info(f"Unexpected response format: {analyze_data}")
            
        if "processing_time" in analyze_data:
            logging.info(f"\nProcessing Time: {analyze_data['processing_time']:.4f} seconds")
        
        # Test metrics endpoint
        logging.info("\nTesting metrics endpoint...")
        metrics_response = requests.get(f"{base_url}/metrics")
        logging.info("Metrics endpoint response received")
        
    except Exception as e:
        logging.error(f"Error testing API: {str(e)}")
        raise

if __name__ == "__main__":
    test_api() 