#!/usr/bin/env python3

"""
NeuraShield Threat Prediction Script
This script performs threat prediction on new network traffic data
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def preprocess_input(input_data, scaler=None):
    """Preprocess input data for prediction"""
    # Convert to numpy array if it's a dataframe
    if isinstance(input_data, pd.DataFrame):
        # Convert categorical features to numeric
        for col in input_data.select_dtypes(include=['object']).columns:
            input_data[col] = pd.factorize(input_data[col])[0]
        
        # Fill missing values
        input_data = input_data.fillna(0)
        
        # Convert to numpy array
        input_data = input_data.values
    
    # Normalize features if scaler is provided
    if scaler:
        input_data = scaler.transform(input_data)
    else:
        # Simple normalization if no scaler provided
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data)
    
    return input_data

def predict_threats(model, input_data):
    """Make threat predictions"""
    predictions = model.predict(input_data)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create results
    results = []
    for i, (pred_class, pred_probs) in enumerate(zip(predicted_classes, predictions)):
        result = {
            "sample_id": i,
            "predicted_class": int(pred_class),
            "prediction": "Normal" if pred_class == 0 else "Attack",
            "confidence": float(pred_probs[pred_class]),
            "probabilities": {
                "Normal": float(pred_probs[0]),
                "Attack": float(pred_probs[1])
            }
        }
        results.append(result)
    
    return results

def save_results(results, output_file):
    """Save prediction results to a file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Predict threats using the trained model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--input-file', type=str, required=True, help='Path to input data CSV file')
    parser.add_argument('--output-file', type=str, default='predictions.json', 
                       help='Path to save prediction results')
    parser.add_argument('--scaler-file', type=str, help='Path to fitted scaler (optional)')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return
    
    # Load scaler if provided
    scaler = None
    if args.scaler_file and os.path.exists(args.scaler_file):
        try:
            with open(args.scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            logging.info(f"Scaler loaded from {args.scaler_file}")
        except Exception as e:
            logging.warning(f"Could not load scaler: {str(e)}")
    
    # Load input data
    try:
        input_data = pd.read_csv(args.input_file)
        logging.info(f"Input data loaded with shape: {input_data.shape}")
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        return
    
    # Preprocess input data
    processed_data = preprocess_input(input_data, scaler)
    
    # Make predictions
    results = predict_threats(model, processed_data)
    logging.info(f"Made predictions for {len(results)} samples")
    
    # Log summary
    attack_count = sum(1 for r in results if r["prediction"] == "Attack")
    normal_count = sum(1 for r in results if r["prediction"] == "Normal")
    logging.info(f"Prediction summary: {normal_count} normal, {attack_count} attack")
    
    # Save results
    save_results(results, args.output_file)

if __name__ == "__main__":
    main() 