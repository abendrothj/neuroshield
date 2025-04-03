from threat_detection_system import ThreatDetectionSystem
import logging
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import tensorflow as tf

# Configure TensorFlow to use available devices
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices()
logging.info(f"Available physical devices: {physical_devices}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def plot_training_history(history, save_dir='training_plots'):
    """Plot training history metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{save_dir}/accuracy.png')
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir='training_plots'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()

def train_full_model():
    """Train a full-capability model with enhanced features"""
    logging.info("Starting full-capability model training...")
    
    # Create system instance with adjusted thresholds
    system = ThreatDetectionSystem()
    system.threat_thresholds = {
        "DDoS": 0.65,        # Lowered threshold for better detection
        "Brute Force": 0.70,  # Lowered threshold for better detection
        "Port Scan": 0.60,    # Lowered threshold for better detection
        "Malware": 0.80       # Lowered threshold for better detection
    }
    
    # Generate larger training dataset with more threat samples
    logging.info("Generating training data...")
    X, y = system.generate_training_data(n_samples=10000)  # Increased samples
    
    # Split data with stratification
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with enhanced parameters
    logging.info("Training model with enhanced parameters...")
    history = system.train_model(
        n_samples=10000,
        epochs=150,  # Increased epochs
        batch_size=128  # Increased batch size
    )
    
    if history is not None:
        # Plot training history
        logging.info("Generating training plots...")
        plot_training_history(history)
    else:
        logging.warning("No training history available for plotting")
    
    # Evaluate model
    logging.info("Evaluating model performance...")
    val_predictions = system.model.predict(X_val)
    val_pred_classes = np.argmax(val_predictions, axis=1)
    
    # Generate classification report
    report = classification_report(y_val, val_pred_classes)
    logging.info("\nClassification Report:")
    logging.info(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_val, val_pred_classes)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"ai_models/threat_detection_full_{timestamp}"
    system.model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Test with simulated real-time data
    logging.info("\nTesting model with simulated real-time data...")
    test_samples = 100
    threat_detected = 0
    
    # Generate test samples with known threats
    for i in range(test_samples):
        if i < 20:  # 20% normal traffic
            features = system.generate_realtime_data()
        else:  # 80% threat samples
            # Generate threat-specific features
            threat_type = np.random.choice([1, 2, 3, 4])  # Random threat type
            base_metrics = np.random.normal(
                system.threat_thresholds[system.threat_types[threat_type]] - 0.1,
                0.1,
                3
            )
            noise = np.random.normal(0, 0.1, 6)
            features = np.concatenate([base_metrics, noise])
            features = np.clip(features, 0, 1)
        
        analysis = system.analyze_threat(features)
        if analysis["is_threat"]:
            threat_detected += 1
    
    logging.info(f"Threat detection rate: {threat_detected/test_samples*100:.2f}%")
    
    # Save training summary
    summary = {
        "timestamp": timestamp,
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "epochs": 150,
        "batch_size": 128,
        "threat_thresholds": system.threat_thresholds,
        "threat_detection_rate": threat_detected/test_samples,
        "model_path": model_path,
        "classification_report": report,
        "tensorflow_version": tf.__version__,
        "available_devices": [device.device_type for device in physical_devices]
    }
    
    import json
    with open(f"training_summary_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logging.info("Full-capability model training completed successfully!")

if __name__ == "__main__":
    train_full_model() 