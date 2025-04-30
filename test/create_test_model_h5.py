#!/usr/bin/env python3

"""
Create a simple test model for threat detection testing (H5 format)
"""

import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler

print("Creating test threat detection model in H5 format...")

# Create a very simple model that detects 'suspicious' activity
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Create some fake training data to ensure the model accepts input
X_train = np.array([
    # bytes_sent, bytes_received, packets_sent, packets_received, duration_ms
    [1500, 5000, 10, 15, 257],      # normal
    [2000, 3000, 12, 10, 150],      # normal
    [1800, 4500, 8, 12, 200],       # normal
    [200000, 1000, 150, 5, 100],    # normal
    [180000, 2000, 140, 8, 120],    # normal
    [450, 20000, 300, 20, 30000],   # suspicious
    [500, 25000, 280, 25, 28000],   # suspicious
    [600, 22000, 320, 18, 32000],   # suspicious
])

y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1])  # 0=normal, 1=suspicious

# Train the model for a few epochs just to initialize weights
model.fit(X_train, y_train, epochs=5, verbose=0)

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

# Test predictions
X_test = np.array([
    [1500, 5000, 10, 15, 257],      # normal
    [450, 20000, 300, 20, 30000],   # suspicious
])
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

print("\nTest predictions:")
for i, pred in enumerate(predictions):
    pred_class = np.argmax(pred)
    confidence = pred[pred_class]
    print(f"Sample {i+1}: Class {pred_class} ({'suspicious' if pred_class == 1 else 'normal'}) with {confidence:.4f} confidence") 