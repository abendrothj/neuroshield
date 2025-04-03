import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import os
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict
import logging

class ThreatDetectionModel(tf.keras.Model):
    def __init__(self, input_shape: tuple = (39,), num_classes: int = 2):
        """Initialize the threat detection model"""
        super(ThreatDetectionModel, self).__init__()
        
        # Store input shape and number of classes
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Define layers
        self.dense1 = Dense(128, activation='relu')
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(0.3)
        
        self.dense2 = Dense(64, activation='relu')
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(0.3)
        
        self.dense3 = Dense(32, activation='relu')
        self.bn3 = BatchNormalization()
        self.dropout3 = Dropout(0.3)
        
        self.output_layer = Dense(num_classes, activation='softmax')
        
    def build(self, input_shape):
        """Build the model"""
        # Call build on each layer to initialize their weights
        self.dense1.build(input_shape)
        self.bn1.build(self.dense1.compute_output_shape(input_shape))
        self.dropout1.build(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape)))
        
        self.dense2.build(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape))))
        self.bn2.build(self.dense2.compute_output_shape(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape)))))
        self.dropout2.build(self.bn2.compute_output_shape(self.dense2.compute_output_shape(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape))))))
        
        self.dense3.build(self.dropout2.compute_output_shape(self.bn2.compute_output_shape(self.dense2.compute_output_shape(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape)))))))
        self.bn3.build(self.dense3.compute_output_shape(self.dropout2.compute_output_shape(self.bn2.compute_output_shape(self.dense2.compute_output_shape(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape))))))))
        self.dropout3.build(self.bn3.compute_output_shape(self.dense3.compute_output_shape(self.dropout2.compute_output_shape(self.bn2.compute_output_shape(self.dense2.compute_output_shape(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape)))))))))
        
        self.output_layer.build(self.dropout3.compute_output_shape(self.bn3.compute_output_shape(self.dense3.compute_output_shape(self.dropout2.compute_output_shape(self.bn2.compute_output_shape(self.dense2.compute_output_shape(self.dropout1.compute_output_shape(self.bn1.compute_output_shape(self.dense1.compute_output_shape(input_shape))))))))))
        
        # Mark the model as built
        self.built = True
        
    def call(self, inputs):
        """Forward pass"""
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        return self.output_layer(x)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train the model"""
        try:
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=self.model_path,
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
            
            # Train the model
            history = self.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return history.history
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
    
    def predict(self, X):
        """Make predictions on new data"""
        return super().predict(X)
    
    def save(self, filepath, overwrite=True):
        """Save the model"""
        super().save(f"{filepath}.keras", overwrite=overwrite)
    
    def load(self, filepath):
        """Load the model"""
        return keras.models.load_model(f"{filepath}.keras")

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
    history = model.train_model(X_train, y_train, X_val, y_val, epochs=10)
    
    # Save the model
    model.save("ai_models/threat_detection") 