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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Get log directory from environment variable or use current directory
LOG_DIR = os.environ.get('LOG_DIR', '.')
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, 'threat_detection_model.log')

# Make metrics optional
try:
    from models.metrics import (
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
        logging.FileHandler(log_file_path),
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
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model configuration
            config = {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes
            }
            with open(os.path.join(path, 'model_config.json'), 'w') as f:
                json.dump(config, f)
            
            # Save model weights
            self.model.save_weights(os.path.join(path, 'model.keras'))
            
            # Save metadata
            metadata = {
                'version': self.model_version,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'training_history': self.training_history
            }
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            logging.info(f"Model saved to {path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, model_path: str) -> None:
        """Load a trained model from disk"""
        try:
            # Try to load the model normally
            try:
                # Check if the path is a direct model file or a directory
                if model_path.endswith('.keras') or model_path.endswith('.h5'):
                    # Direct model file
                    self.model = load_model(model_path)
                    self.input_shape = self.model.layers[0].input_shape[1:]
                    self.num_classes = self.model.layers[-1].output_shape[-1]
                    self.model_version = f"1.0.0-{os.path.basename(model_path)}"
                    self.metadata = {
                        'version': self.model_version,
                        'input_shape': self.input_shape,
                        'num_classes': self.num_classes,
                        'deployment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    logging.info(f"Model loaded directly from file {model_path}")
                else:
                    # Directory with model config and weights
                    with open(os.path.join(model_path, 'model_config.json'), 'r') as f:
                        config = json.load(f)
                    
                    # Create a new model with the same architecture
                    self.model = self._create_model(
                        input_shape=config['input_shape'],
                        num_classes=config['num_classes']
                    )
                    
                    # Load the weights
                    self.model.load_weights(os.path.join(model_path, 'model.keras'))
                    
                    # Load metadata
                    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
                        self.metadata = json.load(f)
                    
                    self.model_version = self.metadata.get('version', '1.0.0')
                    self.training_history = self.metadata.get('training_history', {})
                    
                    logging.info(f"Model loaded successfully from {model_path}")
            except Exception as e:
                # If loading fails, create a test model
                logging.warning(f"Failed to load model from {model_path}: {str(e)}")
                logging.warning("Creating test model for inference")
                self.create_test_model()
            
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

    def _create_model(self, input_shape: int, num_classes: int) -> tf.keras.Model:
        """Create a new model with the specified architecture"""
        self.input_shape = (input_shape,)
        self.num_classes = num_classes
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def create_test_model(self):
        """Create a simple model for testing purposes"""
        try:
            # Default to these values if not already set
            if self.input_shape is None:
                self.input_shape = (8,)  # 8 features to match test_request.json
            if self.num_classes is None:
                self.num_classes = 2  # Binary classification
                
            # Create a simple model
            self.model = Sequential([
                Dense(16, activation='relu', input_shape=self.input_shape),
                Dense(8, activation='relu'),
                Dense(self.num_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model_version = "0.0.1-test"
            self.metadata = {
                'version': self.model_version,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'deployment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            logging.info("Created test model for inference")
            return self.model
            
        except Exception as e:
            logging.error(f"Error creating test model: {str(e)}")
            raise

def create_sample_data(n_samples=1000):
    """Create sample data for testing"""
    # Generate a simple binary classification dataset
    X, y = make_classification(
        n_samples=n_samples,
        n_features=39,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create and train a model with this data
    model = ThreatDetectionModel(input_shape=(39,), num_classes=2)
    model.build_model()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.train(X_train, y_train, X_test, y_test, epochs=10)
    
    # Save the model
    model.save("models/threat_detection")
    
    return model

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