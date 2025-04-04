import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import time

# Make metrics optional
try:
    from metrics import (
        start_metrics_server,
        update_model_metrics,
        record_prediction
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logging.warning("Metrics module not available. Metrics collection disabled.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threat_detection_model.log'),
        logging.StreamHandler()
    ]
)

class ThreatDetectionModel:
    def __init__(self, input_shape: Optional[Tuple[int, ...]] = None, num_classes: Optional[int] = None):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_version = "1.0.0"
        self.training_history = None
        self.model_name = "threat_detection"
        
        # Start metrics server only if available
        if METRICS_AVAILABLE:
            try:
                start_metrics_server(port=8000)
            except Exception as e:
                logging.warning(f"Failed to start metrics server: {str(e)}")
        
    def build_model(self) -> None:
        """Build the neural network model"""
        try:
            if self.input_shape is None or self.num_classes is None:
                raise ValueError("Input shape and number of classes must be specified")
            
            self.model = Sequential([
                Dense(128, activation='relu', input_shape=self.input_shape),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logging.info(f"Model built successfully with input shape {self.input_shape} and {self.num_classes} classes")
            
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 64, callbacks=None) -> Dict:
        """Train the model"""
        try:
            if self.model is None:
                self.build_model()
            
            # Create model checkpoint
            checkpoint_path = f"checkpoints/model_{self.model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            os.makedirs("checkpoints", exist_ok=True)
            
            if callbacks is None:
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
                ]
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.training_history = history.history
            
            # Update metrics if available
            if METRICS_AVAILABLE:
                try:
                    final_accuracy = history.history['val_accuracy'][-1]
                    memory_usage = self.model.count_params() * 4  # Approximate memory usage in bytes
                    update_model_metrics(
                        model_name=self.model_name,
                        accuracy=final_accuracy,
                        memory_usage=memory_usage
                    )
                except Exception as e:
                    logging.warning(f"Error updating metrics: {str(e)}")
            
            # Log training results
            logging.info(f"Training completed with {epochs} epochs")
            logging.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
            logging.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
            
            return history.history
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            start_time = time.time()
            predictions = self.model.predict(X, verbose=0)
            duration = time.time() - start_time
            
            # Record prediction metrics if available
            if METRICS_AVAILABLE:
                try:
                    record_prediction(
                        model_name=self.model_name,
                        duration=duration,
                        success=True
                    )
                except Exception as e:
                    logging.warning(f"Error recording prediction metrics: {str(e)}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            if METRICS_AVAILABLE:
                try:
                    record_prediction(
                        model_name=self.model_name,
                        duration=0,
                        success=False
                    )
                except:
                    pass
            raise
    
    def save(self, path: str) -> None:
        """Save the model and metadata"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save the model using the newer recommended .keras format if possible
            model_path = os.path.join(path, "model.keras")
            try:
                # Try newer save method first
                self.model.save(model_path, save_format="keras")
            except TypeError:
                # Fall back to older save method (will save as H5)
                logging.info("Falling back to default save format")
                self.model.save(model_path)
                
            # Save metadata
            metadata = {
                "model_version": self.model_version,
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "training_history": self.training_history,
                "timestamp": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logging.info(f"Model saved to {path} using Keras format")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load the model and metadata"""
        try:
            # Check for newer .keras format first
            model_path = os.path.join(path, "model.keras")
            if not os.path.exists(model_path):
                # Try legacy .h5 format as fallback
                model_path = os.path.join(path, "model.h5")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found at {path}")
            
            self.model = load_model(model_path)
            
            # Load metadata
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get("model_version", self.model_version)
                    self.input_shape = metadata.get("input_shape", self.input_shape)
                    self.num_classes = metadata.get("num_classes", self.num_classes)
                    self.training_history = metadata.get("training_history", self.training_history)
            
            logging.info(f"Model loaded from {path}")
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            return {
                "model_version": self.model_version,
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "layers": len(self.model.layers),
                "trainable_params": self.model.count_params(),
                "training_history": self.training_history
            }
            
        except Exception as e:
            logging.error(f"Error getting model info: {str(e)}")
            raise

def create_sample_data(n_samples=1000):
    """Create sample data for testing"""
    X = np.random.randn(n_samples, 9)  # 9 features
    y = np.random.randint(0, 5, n_samples)  # 5 classes
    return X, y

if __name__ == "__main__":
    # Example usage
    model = ThreatDetectionModel()
    
    # Create sample data
    X_train, y_train = create_sample_data(1000)
    X_val, y_val = create_sample_data(200)
    
    # Train the model
    history = model.train(X_train, y_train, X_val, y_val, epochs=10)
    
    # Save the model
    model.save("ai_models/threat_detection") 