import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
from threat_detection_system import ThreatDetectionSystem
from threat_detection_model import ThreatDetectionModel
import os
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unsw_training.log'),
        logging.StreamHandler()
    ]
)

def plot_training_history(history, save_path):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize the system
    system = ThreatDetectionSystem()
    
    # Load and preprocess the dataset
    print("Loading UNSW-NB15 dataset...")
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "UNSW_NB15")
    X, y = system.load_unsw_dataset(dataset_path)
    
    # Get unique classes and their counts
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"\nNumber of classes in dataset: {num_classes}")
    print("Class distribution:")
    for cls in unique_classes:
        count = np.sum(y == cls)
        print(f"Class {cls}: {count} samples")
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    print("\nCreating model...")
    model = ThreatDetectionModel(input_shape=(X_train.shape[1],), num_classes=num_classes)
    
    # Find the most recent model file
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.h5')]
    if model_files:
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        previous_model_path = os.path.join(models_dir, latest_model)
        print(f"Loading previous model for transfer learning: {latest_model}")
        model.load(previous_model_path)
        # Freeze some layers to prevent catastrophic forgetting
        for layer in model.layers[:-2]:  # Freeze all except last two layers
            layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, f'unsw_threat_detection_{timestamp}_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the scaler
    import joblib
    joblib.dump(system.scaler, os.path.join(models_dir, f'unsw_threat_detection_{timestamp}_scaler.joblib'))
    
    # Plot training history
    plot_training_history(history, os.path.join(models_dir, f'unsw_threat_detection_{timestamp}_history.png'))
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=[system.threat_types.get(i, f"Class {i}") for i in range(num_classes)]))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, 
                         [system.threat_types.get(i, f"Class {i}") for i in range(num_classes)],
                         os.path.join(models_dir, f'unsw_threat_detection_{timestamp}_confusion_matrix.png'))
    
    print(f"\nModel and artifacts saved in: {models_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main() 