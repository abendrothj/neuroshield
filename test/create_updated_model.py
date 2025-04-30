#!/usr/bin/env python3

"""
Create a simple test model for threat detection testing with all features from CSV
"""

import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Creating test threat detection model matching CSV format...")

# Read the actual CSV to match the format exactly
try:
    csv_path = "/home/jub/Cursor/neurashield/data/test_network_traffic.csv"
    df = pd.read_csv(csv_path)
    print(f"CSV data shape: {df.shape}")
    print(f"CSV columns: {df.columns.tolist()}")
    
    # Convert categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.factorize(df[col])[0]

    # Extract features (all columns except timestamp and flag)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'flag']]
    num_features = len(feature_cols)
    print(f"Using {num_features} features: {feature_cols}")
    
    # Create a model with the right input shape
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create training data based on the CSV
    X_data = df[feature_cols].values
    
    # Generate synthetic training data with the right shape
    normal_samples = 10
    suspicious_samples = 5
    X_train = np.vstack([
        # Normal samples
        np.random.normal(loc=1.0, scale=0.2, size=(normal_samples, num_features)) * X_data[0],
        # Suspicious samples
        np.random.normal(loc=1.0, scale=0.2, size=(suspicious_samples, num_features)) * X_data[2]
    ])
    
    # Labels (0=normal, 1=suspicious)
    y_train = np.array([0] * normal_samples + [1] * suspicious_samples)
    
    # Train the model
    print(f"Training model with {len(X_train)} samples, shape {X_train.shape}")
    model.fit(X_train, y_train, epochs=10, verbose=1)
    
    # Create and save a scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Paths
    model_dir = '/home/jub/Cursor/neurashield/models/trained'
    model_path = os.path.join(model_dir, 'threat_detection_model.h5')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    # Save the model in H5 format
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_path)
    print(f'Saved test model to {model_path}')
    
    # Save the scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'Saved scaler to {scaler_path}')
    
    # Test predictions on original data
    X_scaled = scaler.transform(X_data)
    predictions = model.predict(X_scaled)
    
    print("\nTest predictions for CSV data:")
    for i, pred in enumerate(predictions):
        pred_class = np.argmax(pred)
        confidence = pred[pred_class]
        print(f"Row {i+1}: Class {pred_class} ({'suspicious' if pred_class == 1 else 'normal'}) with {confidence:.4f} confidence")
    
except Exception as e:
    print(f"Error: {str(e)}")
    raise 