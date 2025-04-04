#!/usr/bin/env python3

"""
NeuraShield AI Model Training Script
This script trains the threat detection model with comprehensive analytics and visualization
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ThreatDetectionSystem and ThreatDetectionModel
from threat_detection_system import ThreatDetectionSystem
from threat_detection_model import ThreatDetectionModel

# Configure TensorFlow to use available devices
physical_devices = tf.config.list_physical_devices()
print(f"Available physical devices: {physical_devices}")

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Using GPU for training")
    try:
        # Configure TensorFlow to use memory growth
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set to True for GPUs")
    except:
        print("Error setting memory growth for GPUs")
else:
    print("No GPU available, using CPU for training")

# Set up logging
log_dir = "training_logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/training_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def create_output_directories():
    """Create directories for output files"""
    directories = ["models", "plots", "reports"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    return {dir_name: f"{dir_name}_{timestamp}" for dir_name in directories}

def load_and_prepare_data(system):
    """Load and prepare the dataset for training"""
    logging.info("Loading and preparing dataset...")
    
    try:
        # Try to load the dataset from the UNSW_NB15 directory
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "UNSW_NB15")
        
        if os.path.exists(dataset_path):
            logging.info(f"Loading dataset from {dataset_path}")
            X, y = system.load_unsw_dataset(dataset_path)
        else:
            # If dataset doesn't exist, generate synthetic data
            logging.warning(f"Dataset path {dataset_path} not found. Generating synthetic data.")
            X, y = system.generate_training_data(n_samples=10000)
        
        # Display data statistics
        logging.info(f"Data shape: X: {X.shape}, y: {y.shape}")
        unique_classes, class_counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            class_name = system.threat_types.get(cls, f"Unknown ({cls})")
            logging.info(f"Class {class_name}: {count} samples ({count/len(y)*100:.2f}%)")
        
        # Balance classes
        X_balanced, y_balanced = system.balance_classes(X, y)
        logging.info(f"Balanced data shape: X: {X_balanced.shape}, y: {y_balanced.shape}")
        
        # Split into train, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logging.info(f"Training set: {X_train.shape[0]} samples")
        logging.info(f"Validation set: {X_val.shape[0]} samples")
        logging.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def train_model(system, X_train, y_train, X_val, y_val, output_dirs):
    """Train the model with the prepared dataset"""
    logging.info("Training model...")
    
    try:
        # Initialize model with correct input shape and number of classes
        input_shape = (X_train.shape[1],)
        num_classes = len(np.unique(y_train))
        
        system.model = system.model or ThreatDetectionModel(
            input_shape=input_shape, 
            num_classes=num_classes
        )
        
        # Train the model with early stopping and model checkpointing
        epochs = 150
        batch_size = 128
        
        # Create model checkpoint directory
        checkpoint_dir = os.path.join("checkpoints", f"checkpoint_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define callbacks
        checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, 
                monitor='val_loss', 
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=5, 
                min_lr=0.0001
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join("logs", f"run_{timestamp}"),
                histogram_freq=1
            )
        ]
        
        # Train the model
        history = system.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Save the model
        model_path = os.path.join(output_dirs["models"], "threat_detection_model")
        os.makedirs(model_path, exist_ok=True)
        system.model.save(model_path)
        logging.info(f"Model saved to {model_path}")
        
        return history, model_path
        
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(system, X_test, y_test, history, output_dirs):
    """Evaluate the trained model and generate performance metrics"""
    logging.info("Evaluating model...")
    
    try:
        # Generate predictions
        y_pred_proba = system.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # Create classification report
        class_names = [system.threat_types.get(i, f"Unknown ({i})") for i in range(len(system.threat_types))]
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"\nClassification Report:\n{report}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dirs["plots"], "confusion_matrix.png"))
        plt.close()
        
        # Plot training history (accuracy)
        plt.figure(figsize=(10, 6))
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dirs["plots"], "accuracy.png"))
        plt.close()
        
        # Plot training history (loss)
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dirs["plots"], "loss.png"))
        plt.close()
        
        # Generate ROC curves for each class
        plt.figure(figsize=(10, 8))
        for i in range(len(system.threat_types)):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{system.threat_types.get(i, f"Class {i}")}: (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dirs["plots"], "roc_curves.png"))
        plt.close()
        
        # Save evaluation results
        evaluation_results = {
            "accuracy": float(accuracy),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "timestamp": timestamp,
            "model_path": os.path.join(output_dirs["models"], "threat_detection_model")
        }
        
        with open(os.path.join(output_dirs["reports"], "evaluation_results.json"), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logging.info(f"Evaluation results saved to {os.path.join(output_dirs['reports'], 'evaluation_results.json')}")
        
        return accuracy, report, cm
        
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        raise

def test_real_world_scenarios(system, output_dirs):
    """Test the model with real-world scenarios"""
    logging.info("Testing with real-world scenarios...")
    
    try:
        # Define some real-world scenarios
        scenarios = [
            {"name": "Normal Traffic", "features": system.generate_realtime_data()},
            {"name": "DDoS Attack", "features": np.random.normal(0.7, 0.2, system.model.input_shape[0])},
            {"name": "Brute Force Attack", "features": np.random.normal(0.8, 0.15, system.model.input_shape[0])},
            {"name": "Port Scan", "features": np.random.normal(0.6, 0.25, system.model.input_shape[0])},
            {"name": "Malware", "features": np.random.normal(0.9, 0.1, system.model.input_shape[0])}
        ]
        
        # Ensure all features are in range [0, 1]
        for scenario in scenarios:
            scenario["features"] = np.clip(scenario["features"], 0, 1)
        
        # Test each scenario
        scenario_results = []
        for scenario in scenarios:
            result = system.analyze_threat(scenario["features"])
            logging.info(f"Scenario: {scenario['name']}")
            logging.info(f"  Prediction: {result['prediction']}")
            logging.info(f"  Is Threat: {result['is_threat']}")
            logging.info(f"  Confidence: {result['confidence']:.2f}%")
            scenario_results.append({
                "scenario": scenario["name"],
                "prediction": result["prediction"],
                "is_threat": result["is_threat"],
                "confidence": float(result["confidence"])
            })
        
        # Save scenario results
        with open(os.path.join(output_dirs["reports"], "scenario_results.json"), 'w') as f:
            json.dump(scenario_results, f, indent=2)
        
        logging.info(f"Scenario results saved to {os.path.join(output_dirs['reports'], 'scenario_results.json')}")
        
        return scenario_results
        
    except Exception as e:
        logging.error(f"Error testing scenarios: {str(e)}")
        raise

def main():
    """Main function to run the training process"""
    try:
        logging.info("Starting NeuraShield AI model training...")
        
        # Create output directories
        output_dirs = create_output_directories()
        
        # Initialize the ThreatDetectionSystem
        system = ThreatDetectionSystem()
        
        # Load and prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(system)
        
        # Train model
        history, model_path = train_model(system, X_train, y_train, X_val, y_val, output_dirs)
        
        # Evaluate model
        accuracy, report, cm = evaluate_model(system, X_test, y_test, history, output_dirs)
        
        # Test with real-world scenarios
        scenario_results = test_real_world_scenarios(system, output_dirs)
        
        # Generate training summary
        summary = {
            "timestamp": timestamp,
            "model_path": model_path,
            "accuracy": float(accuracy),
            "training_samples": int(X_train.shape[0]),
            "validation_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "features": int(X_train.shape[1]),
            "classes": len(system.threat_types),
            "output_directories": output_dirs,
            "tensorflow_version": tf.__version__,
            "physical_devices": [str(device) for device in physical_devices],
        }
        
        with open(os.path.join(output_dirs["reports"], "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training summary saved to {os.path.join(output_dirs['reports'], 'training_summary.json')}")
        logging.info("NeuraShield AI model training completed successfully!")
        
        # Return information for use in interactive environment
        return {
            "system": system,
            "model_path": model_path,
            "accuracy": accuracy,
            "output_dirs": output_dirs
        }
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 