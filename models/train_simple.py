#!/usr/bin/env python3

"""
Simplified NeuraShield AI Model Training Script
This script trains the threat detection model outside of Docker
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

# Add the ai_models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models"))

class ModelTrainer:
    def __init__(self, model_type='residual', learning_rate=0.001):
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self, input_shape):
        """Build either a simple or advanced model based on model_type"""
        if self.model_type == 'residual':
            return self._build_residual_model(input_shape)
        elif self.model_type == 'advanced':
            return self._build_advanced_model(input_shape)
        else:
            return self._build_simple_model(input_shape)
    
    def _build_simple_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _build_residual_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        
        # First residual block
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        residual = x
        
        # Second residual block
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.add([x, residual])
        
        # Final layers
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_advanced_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Feature extraction
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Multiple residual blocks
        for _ in range(3):
            residual = x
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.add([x, residual])
        
        # Final layers
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with early stopping and model checkpointing"""
        self.model = self.build_model((X_train.shape[1],))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Create output directory
        output_dir = Path('models') / f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / 'model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(output_dir / 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        history_df = pd.DataFrame(self.history.history)
        history_df.to_csv(output_dir / 'training_history.csv', index=False)
        
        return output_dir
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2]
        }
        
        return metrics

def load_unsw_dataset(dataset_path):
    """Load the UNSW-NB15 dataset"""
    logging.info(f"Loading dataset from {dataset_path}")
    
    # Find training file
    train_file = os.path.join(dataset_path, "UNSW_NB15_training-set.csv")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Load dataset
    df = pd.read_csv(train_file, low_memory=False)
    logging.info(f"Dataset loaded with shape: {df.shape}")
    
    # Display column information
    logging.info(f"Columns: {df.columns.tolist()}")
    
    # Identify feature and target columns
    # Adjust these based on the actual dataset structure
    X = df.drop(['id', 'label', 'attack_cat'], axis=1, errors='ignore')
    y = df['label'].values if 'label' in df.columns else df.iloc[:, -1].values
    
    # Convert categorical features to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.factorize(X[col])[0]
    
    # Fill missing values
    X = X.fillna(0)
    
    # Convert to numpy arrays
    X = X.values
    
    logging.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description='Train NeuraShield threat detection model')
    parser.add_argument('--dataset-path', type=str, required=True,
                      help='Path to the dataset directory')
    parser.add_argument('--model-type', type=str, default='residual',
                      choices=['simple', 'residual', 'advanced'],
                      help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate for the optimizer')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    # Add your data loading and preprocessing code here
    
    # Initialize and train model
    trainer = ModelTrainer(
        model_type=args.model_type,
        learning_rate=args.learning_rate
    )
    
    logging.info(f"Training {args.model_type} model...")
    output_dir = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    logging.info(f"Test metrics: {metrics}")
    
    logging.info(f"Model saved to {output_dir}")

if __name__ == '__main__':
    main() 