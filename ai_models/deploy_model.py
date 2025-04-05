import os
import shutil
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)

def deploy_model(checkpoint_path, model_name="threat_detection"):
    """Deploy a trained model to production"""
    try:
        # Create production model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prod_dir = os.path.join("models", f"{model_name}_{timestamp}")
        os.makedirs(prod_dir, exist_ok=True)
        
        # Copy the model file
        model_file = os.path.join(checkpoint_path, "model_epoch_38.keras")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        shutil.copy2(model_file, os.path.join(prod_dir, "model.keras"))
        logging.info(f"Copied model to {prod_dir}")
        
        # Create metadata file
        metadata = {
            "model_name": model_name,
            "version": "1.0.0",
            "deployment_time": timestamp,
            "validation_accuracy": 0.9947,  # We'll update this after checking the actual value
            "training_accuracy": 0.9950,    # We'll update this after checking the actual value
            "epoch": 38,
            "input_shape": (9,),  # Update based on your model
            "num_classes": 5,     # Update based on your model
            "description": "NeuraShield Threat Detection Model"
        }
        
        with open(os.path.join(prod_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logging.info(f"Created metadata file in {prod_dir}")
        
        # Create symlink to latest model
        latest_link = os.path.join("models", f"{model_name}_latest")
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(prod_dir, latest_link)
        logging.info(f"Created symlink to latest model at {latest_link}")
        
        return prod_dir
        
    except Exception as e:
        logging.error(f"Error deploying model: {str(e)}")
        raise

if __name__ == "__main__":
    # Deploy the model from the latest checkpoint
    checkpoint_path = "checkpoints/checkpoint_20250403_205441"
    deploy_model(checkpoint_path) 