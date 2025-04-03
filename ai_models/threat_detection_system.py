import numpy as np
import time
from datetime import datetime
import json
import os
from threat_detection_model import ThreatDetectionModel
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threat_detection.log'),
        logging.StreamHandler()
    ]
)

class ThreatDetectionSystem:
    def __init__(self, model_path: str = "ai_models/threat_detection"):
        self.model_path = model_path
        self.model = None
        self.threat_types = {
            0: "Normal",
            1: "DDoS",
            2: "Brute Force",
            3: "Port Scan",
            4: "Malware"
        }
        self.threat_thresholds = {
            "DDoS": 0.5,
            "Brute Force": 0.6,
            "Port Scan": 0.4,
            "Malware": 0.7
        }
        self.alert_history = []
        self.scaler = MinMaxScaler()
        
    def load_unsw_dataset(self, dataset_path: str) -> tuple:
        """Load and preprocess UNSW-NB15 dataset"""
        try:
            # Find the CSV file in the dataset directory
            csv_file = None
            for file in os.listdir(dataset_path):
                if file.endswith('.csv'):
                    csv_file = os.path.join(dataset_path, file)
                    break
            
            if not csv_file:
                raise FileNotFoundError("No CSV file found in dataset directory")
            
            logging.info(f"Loading dataset from: {csv_file}")
            
            # Load dataset with proper handling of missing values
            df = pd.read_csv(csv_file, low_memory=False, na_values=[' ', '-', 'nan', 'NaN'])
            
            # Print column names for debugging
            logging.info(f"Available columns: {df.columns.tolist()}")
            
            # Select relevant features (using numeric column names from the dataset)
            feature_columns = [
                '0.072974',  # duration
                '4238',      # source bytes
                '60788',     # destination bytes
                '31',        # source ttl
                '29',        # destination ttl
                '7',         # source loss
                '30',        # destination loss
                '458245.4375',  # source load
                '6571546.5',    # destination load
                '72',           # source packets
                '72.1',         # destination packets
                '255',          # source window
                '255.1',        # destination window
                '1003293149',   # source tcp base sequence number
                '1003585034',   # destination tcp base sequence number
                '59',           # source tcp window
                '844',          # destination tcp window
                '0',            # source tcp retransmission
                '0.1',          # destination tcp retransmission
                '62.04531',     # source tcp round trip time
                '61.899776',    # destination tcp round trip time
                '1421955842',   # source tcp round trip time variance
                '1421955842.1', # destination tcp round trip time variance
                '1.02269',      # source tcp round trip time minimum
                '0.997042',     # destination tcp round trip time minimum
                '0.002317',     # source tcp round trip time maximum
                '0.002173',     # destination tcp round trip time maximum
                '0.000144',     # source tcp round trip time mean
                '0.2',          # destination tcp round trip time mean
                '0.3',          # source tcp round trip time standard deviation
                '0.4',          # destination tcp round trip time standard deviation
                '0.5',          # source tcp round trip time minimum
                '0.6',          # destination tcp round trip time minimum
                '13',           # source tcp round trip time maximum
                '13.1',         # destination tcp round trip time maximum
                '6',            # source tcp round trip time mean
                '7.1',          # destination tcp round trip time mean
                '1',            # source tcp round trip time standard deviation
                '1.1'           # destination tcp round trip time standard deviation
            ]
            
            # Clean and prepare features
            X = df[feature_columns].copy()
            
            # Log feature statistics before preprocessing
            logging.info("\nFeature statistics before preprocessing:")
            logging.info(X.describe())
            
            # Fill missing values with column means
            X = X.fillna(X.mean())
            
            # Convert to numeric, coercing errors to NaN
            X = X.apply(pd.to_numeric, errors='coerce')
            
            # Fill any remaining NaN values with 0
            X = X.fillna(0)
            
            # Convert to numpy array
            X = X.values
            
            # Debug label column
            logging.info("\nLabel column unique values:")
            logging.info(df['0.7'].unique())
            
            # Map numeric labels to threat types
            # Assuming 0 is Normal, 1 is DoS, 2 is Brute Force, 3 is Port Scan, 4 is Malware
            y = df['0.7'].astype(int).values
            
            # Log label distribution
            unique_labels, label_counts = np.unique(y, return_counts=True)
            logging.info("\nLabel distribution:")
            for label, count in zip(unique_labels, label_counts):
                label_name = self.threat_types.get(label, "Unknown")
                logging.info(f"{label_name}: {count} samples")
            
            # Normalize features
            X = self.scaler.fit_transform(X)
            
            # Log feature statistics after normalization
            logging.info("\nFeature statistics after normalization:")
            logging.info(pd.DataFrame(X).describe())
            
            logging.info(f"\nDataset loaded successfully. Shape: {X.shape}")
            logging.info(f"Number of classes: {len(np.unique(y))}")
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise

    def generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from UNSW-NB15 dataset"""
        try:
            # Load local dataset
            dataset_path = "UNSW_NB15"  # Local dataset path
            X, y = self.load_unsw_dataset(dataset_path)
            
            # Balance classes if needed
            X, y = self.balance_classes(X, y)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return X_train, y_train
            
        except Exception as e:
            logging.error(f"Error generating training data: {str(e)}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 64) -> Dict:
        """Train the threat detection model"""
        try:
            # Balance classes
            X_balanced, y_balanced = self.balance_classes(X, y)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_balanced, y_balanced,
                test_size=0.2,
                random_state=42
            )
            
            # Initialize model with correct input shape
            self.model = ThreatDetectionModel(
                input_shape=(X.shape[1],),  # Use actual number of features
                num_classes=len(np.unique(y))  # Use actual number of classes
            )
            
            # Train the model
            history = self.model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
            
            # Save the model
            self.model.save(self.model_path)
            
            return history
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
    
    def load_model(self) -> None:
        """Load a trained model"""
        if self.model is None:
            self.model = ThreatDetectionModel()
            self.model.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
    
    def generate_realtime_data(self) -> np.ndarray:
        """Generate simulated real-time network data with more realistic patterns"""
        # Randomly choose between normal traffic and threats
        is_threat = np.random.random() < 0.3  # 30% chance of threat
        
        if is_threat:
            # Choose a random threat type
            threat_type = np.random.choice([1, 2, 3, 4])
            threat_pattern = self.threat_types[threat_type]
            
            # Generate threat-specific patterns
            if threat_pattern == "DDoS":
                # DDoS pattern: high traffic with bursts
                base_metrics = np.random.normal(0.8, 0.1, 3)
                noise = np.random.normal(0, 0.1, 6)
                # Add burst pattern
                if np.random.random() < 0.3:
                    base_metrics *= 1.2
            elif threat_pattern == "Brute Force":
                # Brute force pattern: high authentication attempts
                base_metrics = np.random.normal(0.85, 0.1, 3)
                noise = np.random.normal(0, 0.1, 6)
                # Add intensive attempt pattern
                if np.random.random() < 0.4:
                    base_metrics *= 1.1
            elif threat_pattern == "Port Scan":
                # Port scan pattern: moderate-high activity with spikes
                base_metrics = np.random.normal(0.7, 0.1, 3)
                noise = np.random.normal(0, 0.1, 6)
                # Add scanning pattern
                if np.random.random() < 0.2:
                    base_metrics *= 1.15
            else:  # Malware
                # Malware pattern: consistently high activity
                base_metrics = np.random.normal(0.9, 0.05, 3)
                noise = np.random.normal(0, 0.05, 6)
                # Add infection pattern
                if np.random.random() < 0.1:
                    base_metrics *= 1.1
        else:
            # Normal traffic pattern
            base_metrics = np.random.normal(0.1, 0.05, 3)
            noise = np.random.normal(0, 0.05, 6)
            # Add occasional small spikes
            if np.random.random() < 0.1:
                base_metrics *= 1.2
        
        # Add correlation between features
        noise[1] = noise[0] * 0.7 + noise[1] * 0.3
        noise[2] = noise[1] * 0.5 + noise[2] * 0.5
        
        # Combine features
        features = np.concatenate([base_metrics, noise])
        
        # Ensure values are between 0 and 1
        return np.clip(features, 0, 1)
    
    def analyze_threat(self, features: np.ndarray) -> Dict:
        """Analyze network traffic for threats"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Make prediction
            predictions = self.model.predict(features)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
            
            # Get threat type name
            threat_type = self.threat_types.get(predicted_class, "Unknown")
            
            # Update statistics
            self.total_alerts += 1
            self.threat_counts[threat_type] = self.threat_counts.get(threat_type, 0) + 1
            
            return {
                "threat_type": threat_type,
                "confidence": float(confidence),
                "raw_predictions": predictions[0].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing threat: {str(e)}")
            raise
    
    def monitor_realtime(self, duration: int = 60, interval: float = 1.0) -> None:
        """Monitor network traffic in real-time"""
        logging.info(f"Starting real-time monitoring for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Generate simulated real-time data
            features = self.generate_realtime_data()
            
            # Analyze for threats
            analysis = self.analyze_threat(features)
            
            # Log threats
            if analysis["is_threat"]:
                logging.warning(
                    f"Threat detected: {analysis['threat_type']} "
                    f"(Confidence: {analysis['confidence']:.2f})"
                )
                self.alert_history.append(analysis)
            
            # Save alert history periodically
            if len(self.alert_history) % 10 == 0:
                self.save_alert_history()
            
            time.sleep(interval)
    
    def save_alert_history(self) -> None:
        """Save alert history to file"""
        with open("alert_history.json", "w") as f:
            json.dump(self.alert_history, f, indent=2)
        logging.info("Alert history saved")
    
    def get_statistics(self) -> Dict:
        """Get statistics about detected threats"""
        if not self.alert_history:
            return {"total_alerts": 0}
        
        threat_counts = {}
        for alert in self.alert_history:
            threat_type = alert["threat_type"]
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        return {
            "total_alerts": len(self.alert_history),
            "threat_counts": threat_counts,
            "latest_alert": self.alert_history[-1] if self.alert_history else None
        }

    def balance_classes(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes using SMOTE"""
        try:
            # Create SMOTE instance
            smote = SMOTE(random_state=42)
            
            # Log original class distribution
            unique_labels, label_counts = np.unique(y, return_counts=True)
            logging.info("\nOriginal class distribution:")
            for label, count in zip(unique_labels, label_counts):
                label_name = self.threat_types.get(label, "Unknown")
                logging.info(f"{label_name}: {count} samples")
            
            # Apply SMOTE
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Log balanced class distribution
            unique_labels, label_counts = np.unique(y_balanced, return_counts=True)
            logging.info("\nBalanced class distribution:")
            for label, count in zip(unique_labels, label_counts):
                label_name = self.threat_types.get(label, "Unknown")
                logging.info(f"{label_name}: {count} samples")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logging.error(f"Error balancing classes: {str(e)}")
            raise

    def predict_threat(self, features: np.ndarray) -> Dict:
        """Make predictions on new data"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Ensure features are normalized
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(features_scaled)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions, axis=1)[0]
            
            # Get threat type name
            threat_type = self.threat_types.get(predicted_class, "Unknown")
            
            return {
                "threat_type": threat_type,
                "confidence": float(confidence),
                "raw_predictions": predictions[0].tolist()
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            raise

    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """Make predictions on a batch of data"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Ensure features are normalized
            features_scaled = self.scaler.transform(features)
            
            # Make predictions
            predictions = self.model.predict(features_scaled)
            
            results = []
            for pred in predictions:
                predicted_class = np.argmax(pred)
                confidence = np.max(pred)
                threat_type = self.threat_types.get(predicted_class, "Unknown")
                
                results.append({
                    "threat_type": threat_type,
                    "confidence": float(confidence),
                    "raw_predictions": pred.tolist()
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error making batch predictions: {str(e)}")
            raise

    def generate_sample_data(self) -> np.ndarray:
        """Generate sample network traffic data"""
        # Generate 39 features to match the model's input shape
        features = np.random.rand(1, 39)
        
        # Scale features to match training data range
        features = self.scaler.transform(features)
        
        return features

def main():
    # Create system instance
    system = ThreatDetectionSystem()
    
    # Train new model (comment out if using existing model)
    logging.info("Training new model...")
    system.train_model(n_samples=1000, epochs=50)
    
    # Start real-time monitoring
    logging.info("Starting real-time monitoring...")
    system.monitor_realtime(duration=60, interval=1.0)
    
    # Print statistics
    stats = system.get_statistics()
    logging.info("\nMonitoring Statistics:")
    logging.info(f"Total alerts: {stats['total_alerts']}")
    logging.info("Threat distribution:")
    for threat_type, count in stats['threat_counts'].items():
        logging.info(f"  {threat_type}: {count}")

if __name__ == "__main__":
    main() 