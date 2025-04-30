#!/usr/bin/env python3

"""
NeuraShield Threat Detection Daemon
This daemon continuously monitors for threats and logs them to the blockchain
"""

import os
import sys
import time
import logging
import argparse
import json
import numpy as np
import pandas as pd
import requests
import signal
import threading
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/threat_detection_daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('threat_detection_daemon')

# Configuration
DEFAULT_MODEL_PATH = os.environ.get('MODEL_PATH', 'models/trained/threat_detection_model')
DEFAULT_INTERVAL = float(os.environ.get('MONITORING_INTERVAL', '5.0'))  # seconds
DEFAULT_DATA_SOURCE = os.environ.get('DATA_SOURCE', 'api')  # 'api' or 'file'
DEFAULT_API_URL = os.environ.get('DATA_API_URL', 'http://localhost:5000/api/v1/network-data')
DEFAULT_INPUT_FILE = os.environ.get('INPUT_FILE', 'data/network_traffic.csv')
BLOCKCHAIN_API_URL = os.environ.get('BLOCKCHAIN_API_URL', 'http://localhost:3000/api/v1/events')
BLOCKCHAIN_ENABLED = os.environ.get('BLOCKCHAIN_ENABLED', 'true').lower() == 'true'

# Threat classification mapping
THREAT_TYPES = {
    0: "Normal",
    1: "Attack"  # Can be expanded to multiple attack types
}

class ThreatDetectionDaemon:
    def __init__(self, model_path, interval, data_source, api_url=None, input_file=None):
        self.model_path = model_path
        self.interval = interval
        self.data_source = data_source
        self.api_url = api_url
        self.input_file = input_file
        self.model = None
        self.scaler = None
        self.running = False
        self.monitor_thread = None
        self.total_processed = 0
        self.threats_detected = 0
        
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
            # Attempt to load scaler if exists
            scaler_path = os.path.join(os.path.dirname(self.model_path), 'scaler.pkl')
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.warning("No scaler found, will use StandardScaler for preprocessing")
                self.scaler = StandardScaler()
                
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_data(self, data):
        """Preprocess data for prediction"""
        try:
            # Handle different input formats
            if isinstance(data, pd.DataFrame):
                # Select only numerical features and exclude 'timestamp' and 'flag' columns
                feature_cols = [col for col in data.columns if col not in ['timestamp', 'flag']]
                
                # Convert categorical features to numeric
                for col in data.select_dtypes(include=['object']).columns:
                    if col in feature_cols:
                        data[col] = pd.factorize(data[col])[0]
                
                # Fill missing values
                data = data.fillna(0)
                
                # Use only the feature columns
                data = data[feature_cols].values
                logger.info(f"Preprocessed data shape: {data.shape}, features: {feature_cols}")
            elif isinstance(data, list):
                # Convert list of dictionaries to dataframe
                data = pd.DataFrame(data)
                return self.preprocess_data(data)
            
            # Apply scaling
            if self.scaler:
                try:
                    data = self.scaler.transform(data)
                except Exception as e:
                    logger.warning(f"Scaling error: {str(e)}. Attempting to fit and transform.")
                    # If the scaler hasn't been fitted yet or dimensions don't match
                    data = self.scaler.fit_transform(data)
            
            return data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def predict(self, data):
        """Make predictions"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            if processed_data is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(processed_data)
            
            # Process results
            results = []
            for i, prediction in enumerate(predictions):
                predicted_class = np.argmax(prediction)
                confidence = float(prediction[predicted_class])
                
                result = {
                    "sample_id": i,
                    "predicted_class": int(predicted_class),
                    "prediction": THREAT_TYPES.get(predicted_class, "Unknown"),
                    "confidence": confidence,
                    "probabilities": {
                        class_name: float(prob) 
                        for class_name, prob in zip(
                            [THREAT_TYPES.get(j, f"Class_{j}") for j in range(len(prediction))], 
                            prediction
                        )
                    },
                    "timestamp": datetime.now().isoformat(),
                    "source": data[i] if isinstance(data, list) and i < len(data) else f"Sample_{i}"
                }
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def fetch_data_from_api(self):
        """Fetch data from API"""
        try:
            response = requests.get(self.api_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error fetching data from API: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error connecting to API: {str(e)}")
            return None
    
    def fetch_data_from_file(self):
        """Fetch data from file"""
        try:
            # Read a batch of data from the file
            # This is a simplified implementation - in a real system,
            # you'd want to track which lines you've already processed
            if not os.path.exists(self.input_file):
                logger.error(f"Input file not found: {self.input_file}")
                return None
            
            df = pd.read_csv(self.input_file)
            return df
        except Exception as e:
            logger.error(f"Error reading data from file: {str(e)}")
            return None
    
    def log_to_blockchain(self, prediction_data):
        """Log threat detection results to the blockchain"""
        if not BLOCKCHAIN_ENABLED:
            logger.info("Blockchain logging is disabled")
            return True
        
        try:
            # Prepare the event data
            event_data = {
                'id': f"threat-{datetime.now().strftime('%Y%m%d%H%M%S')}-{prediction_data.get('sample_id', 'unknown')}",
                'timestamp': prediction_data.get('timestamp', datetime.now().isoformat()),
                'confidence': prediction_data.get('confidence', 0),
                'prediction': prediction_data.get('prediction'),
                'source_data': prediction_data.get('source'),
                'probabilities': prediction_data.get('probabilities'),
                'summary': f"Threat detection: {prediction_data.get('prediction')} with {prediction_data.get('confidence', 0):.2f} confidence"
            }
            
            # Send to blockchain API
            response = requests.post(
                BLOCKCHAIN_API_URL,
                json=event_data,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Successfully logged to blockchain: {result.get('eventId')}")
                return True
            else:
                logger.error(f"Failed to log to blockchain: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging to blockchain: {str(e)}")
            return False
    
    def monitor(self):
        """Continuously monitor for threats"""
        logger.info("Starting monitoring thread")
        self.running = True
        
        while self.running:
            try:
                # Fetch data
                data = None
                if self.data_source == 'api':
                    data = self.fetch_data_from_api()
                elif self.data_source == 'file':
                    data = self.fetch_data_from_file()
                else:
                    logger.error(f"Unknown data source: {self.data_source}")
                    time.sleep(self.interval)
                    continue
                
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    logger.warning("No data to process")
                    time.sleep(self.interval)
                    continue
                
                # Make predictions
                results = self.predict(data)
                if results is None:
                    logger.warning("Failed to make predictions")
                    time.sleep(self.interval)
                    continue
                
                # Process results
                for result in results:
                    self.total_processed += 1
                    
                    # Log threats to blockchain
                    if result['prediction'] != 'Normal' and result['confidence'] > 0.5:
                        self.threats_detected += 1
                        logger.warning(
                            f"Threat detected: {result['prediction']} "
                            f"(Confidence: {result['confidence']:.2f})"
                        )
                        self.log_to_blockchain(result)
                
                # Print periodic status
                if self.total_processed % 100 == 0:
                    logger.info(
                        f"Status: Processed {self.total_processed} samples, "
                        f"detected {self.threats_detected} threats"
                    )
                
                # Sleep until next check
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.interval)
    
    def start(self):
        """Start the monitoring thread"""
        if not self.load_model():
            logger.error("Failed to load model, cannot start monitoring")
            return False
        
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("Monitoring thread is already running")
            return True
        
        # Start the monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"Threat detection daemon started with interval {self.interval}s")
        return True
    
    def stop(self):
        """Stop the monitoring thread"""
        logger.info("Stopping threat detection daemon")
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop cleanly")
        
        logger.info(
            f"Daemon stopped. Processed {self.total_processed} samples, "
            f"detected {self.threats_detected} threats"
        )

def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}")
    if daemon:
        daemon.stop()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='NeuraShield Threat Detection Daemon')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH,
                       help='Path to the trained model')
    parser.add_argument('--interval', type=float, default=DEFAULT_INTERVAL,
                       help='Monitoring interval in seconds')
    parser.add_argument('--data-source', type=str, choices=['api', 'file'], default=DEFAULT_DATA_SOURCE,
                       help='Source of network data')
    parser.add_argument('--api-url', type=str, default=DEFAULT_API_URL,
                       help='URL for the network data API')
    parser.add_argument('--input-file', type=str, default=DEFAULT_INPUT_FILE,
                       help='Path to input data file')
    args = parser.parse_args()
    
    # Create daemon instance
    global daemon
    daemon = ThreatDetectionDaemon(
        model_path=args.model_path,
        interval=args.interval,
        data_source=args.data_source,
        api_url=args.api_url,
        input_file=args.input_file
    )
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start daemon
    if daemon.start():
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            daemon.stop()
    
if __name__ == "__main__":
    main() 