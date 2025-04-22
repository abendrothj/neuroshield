#!/usr/bin/env python3

"""
NeuraShield Model Deployment Script
This script deploys a trained model for use in the threat detection API
"""

import os
import sys
import logging
import argparse
import json
import shutil
import tensorflow as tf
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def prepare_serving_model(model_path, output_dir, version_number=None):
    """Prepare the model for TensorFlow Serving"""
    
    # Load the trained model
    logging.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Create a timestamp for the version if not provided
    if version_number is None:
        version_number = int(datetime.now().strftime("%Y%m%d%H%M%S"))
    
    # Create the output directory structure
    serving_dir = os.path.join(output_dir, str(version_number))
    os.makedirs(serving_dir, exist_ok=True)
    
    # Save the model in SavedModel format for TensorFlow Serving
    # Fix for newer TensorFlow versions
    logging.info(f"Saving model for serving at {serving_dir}")
    
    # Save model in newer format
    save_path = os.path.join(serving_dir, "model.keras")
    model.save(save_path)
    
    # Create model metadata
    metadata = {
        "version": version_number,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "tensorflow",
        "input_shape": [int(dim) if dim is not None else None for dim in model.input_shape],
        "output_shape": [int(dim) if dim is not None else None for dim in model.output_shape],
        "classes": ["Normal", "Attack"]  # Update based on your model's classes
    }
    
    # Save metadata
    with open(os.path.join(serving_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Model deployed successfully to {serving_dir}")
    return serving_dir

def update_symlink(serving_dir):
    """Update the 'latest' symlink to point to the new model version"""
    latest_link = os.path.join(os.path.dirname(serving_dir), "latest")
    
    # Remove existing symlink if it exists
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    
    # Create new symlink
    os.symlink(serving_dir, latest_link)
    logging.info(f"Updated 'latest' symlink to point to {serving_dir}")

def copy_to_api_dir(serving_dir, api_model_dir):
    """Copy the deployed model to the API models directory"""
    if not os.path.exists(api_model_dir):
        os.makedirs(api_model_dir, exist_ok=True)
    
    # Get the version number from the serving dir
    version = os.path.basename(serving_dir)
    target_dir = os.path.join(api_model_dir, f"threat_detection_{version}")
    
    # Copy the model files
    shutil.copytree(serving_dir, target_dir)
    logging.info(f"Copied model to API directory: {target_dir}")
    
    # Update the latest symlink in the API directory
    latest_link = os.path.join(api_model_dir, "latest")
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(target_dir, latest_link)
    
    return target_dir

def main():
    parser = argparse.ArgumentParser(description='Deploy a trained threat detection model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--output-dir', type=str, default='models/serving', 
                        help='Directory to save the serving model')
    parser.add_argument('--api-model-dir', type=str, default='ai_models/models',
                        help='Directory where the API looks for models')
    parser.add_argument('--version', type=int, help='Model version number')
    args = parser.parse_args()
    
    try:
        # Create the serving model
        serving_dir = prepare_serving_model(
            args.model_path, 
            args.output_dir,
            args.version
        )
        
        # Update the 'latest' symlink
        update_symlink(serving_dir)
        
        # Copy to API directory if specified
        if args.api_model_dir:
            copy_to_api_dir(serving_dir, args.api_model_dir)
        
        logging.info("Model deployment completed successfully")
        
    except Exception as e:
        logging.error(f"Error during model deployment: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 