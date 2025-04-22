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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add the ai_models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models"))

def build_model(input_shape, num_classes):
    """Build a simple threat detection model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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
    # Set paths
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models/datasets/UNSW_NB15")
    model_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_output_path, exist_ok=True)
    
    # Load dataset
    try:
        X, y = load_unsw_dataset(dataset_path)
        logging.info("Dataset loaded successfully")
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build model
    num_classes = len(np.unique(y))
    input_shape = X_train.shape[1]
    model = build_model(input_shape, num_classes)
    logging.info(f"Model built with input shape {input_shape} and {num_classes} classes")
    
    # Train model
    epochs = 20
    batch_size = 64
    logging.info(f"Training model with {epochs} epochs and batch_size={batch_size}")
    
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )
    
    # Evaluate model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = np.mean(y_pred == y_test)
    logging.info(f"Model accuracy: {accuracy:.4f}")
    
    # Print classification report
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_output_path, f"threat_detection_{timestamp}.keras")
    model.save(model_file)
    logging.info(f"Model saved to {model_file}")

if __name__ == "__main__":
    main() 