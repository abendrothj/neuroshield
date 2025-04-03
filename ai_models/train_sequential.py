import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from datetime import datetime
from threat_detection_system import ThreatDetectionSystem
from threat_detection_model import ThreatDetectionModel
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import glob
from imblearn.over_sampling import SMOTE
from collections import Counter
import psutil
import warnings
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def find_csv_files(datasets_dir):
    """Recursively find all CSV files in the datasets directory"""
    csv_files = []
    for root, _, files in os.walk(datasets_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def handle_categorical_features(df):
    """Handle categorical features using one-hot encoding"""
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Remove label columns from categorical columns
    label_columns = [col for col in ['label', 'Label', 'class', 'Class'] if col in df.columns]
    categorical_columns = [col for col in categorical_columns if col not in label_columns]
    
    if categorical_columns:
        print(f"Found categorical columns: {categorical_columns}")
        
        # Drop date/timestamp columns as they're not needed for training
        date_columns = [col for col in df.columns if any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
        if date_columns:
            print(f"Dropping date columns: {date_columns}")
            df = df.drop(columns=date_columns)
            categorical_columns = [col for col in categorical_columns if col not in date_columns]
        
        # Special handling for numeric-like columns
        numeric_like_columns = [col for col in categorical_columns if any(x in col.lower() for x in 
            ['port', 'duration', 'pkts', 'len', 'mean', 'std', 'max', 'min', 'avg', 'ratio', 'count', 'cnt', 
             'byts', 'iat', 'win', 'subflow'])]
        for col in numeric_like_columns:
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                categorical_columns.remove(col)
                print(f"Converted numeric-like column: {col}")
            except:
                print(f"Warning: Could not convert column to numeric: {col}")
        
        # Handle flag columns (convert to numeric)
        flag_columns = [col for col in categorical_columns if any(x in col.lower() for x in ['flag', 'flags'])]
        for col in flag_columns:
            try:
                # Convert flag columns to numeric (assuming they contain numeric values)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                categorical_columns.remove(col)
                print(f"Converted flag column to numeric: {col}")
            except:
                print(f"Warning: Could not convert flag column to numeric: {col}")
        
        # One-hot encode essential categorical columns
        essential_categorical = ['Protocol']  # Add other essential categorical columns here
        truly_categorical = [col for col in categorical_columns if col in essential_categorical]
        if truly_categorical:
            print(f"One-hot encoding essential categorical columns: {truly_categorical}")
            df_encoded = pd.get_dummies(df, columns=truly_categorical, prefix=truly_categorical)
            return df_encoded
    
    return df

def load_dataset(dataset_path, scaler=None):
    """Load and preprocess a single dataset"""
    print(f"\nLoading dataset from: {dataset_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path, low_memory=False, na_values=[' ', '-', 'nan', 'NaN', 'inf', '-inf'])
        
        # Handle categorical features
        df = handle_categorical_features(df)
        
        # Handle labels (adjust based on dataset)
        label_column = next((col for col in ['label', 'Label', 'class', 'Class'] if col in df.columns), None)
        if label_column is None:
            raise ValueError("No label column found in dataset")
        
        # Convert all columns to numeric before creating numpy array
        for col in df.columns:
            if col != label_column:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        X = df.drop(columns=[label_column]).values.astype(np.float64)
        y = df[label_column].values
        
        # Convert labels to numeric if needed
        if not np.issubdtype(y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Fill missing values and handle infinities
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Validate data
        if np.any(np.isinf(X)):
            print("Warning: Found infinite values, replacing with 0")
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(X)):
            print("Warning: Found NaN values, replacing with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if scaler is None:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        
        # Final validation
        if not np.all(np.isfinite(X)):
            raise ValueError("Data contains invalid values after preprocessing")
        
        print(f"Memory usage after loading {os.path.basename(dataset_path)}: {get_memory_usage():.2f} MB")
        print(f"Feature shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {Counter(y)}")
        return X, y, scaler
        
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {str(e)}")
        return None, None, None

def balance_dataset(X, y):
    """Balance the dataset using SMOTE"""
    print("\nBalancing dataset...")
    print(f"Original class distribution: {Counter(y)}")
    
    try:
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"Balanced class distribution: {Counter(y_balanced)}")
        return X_balanced, y_balanced
    except Exception as e:
        print(f"Warning: Could not balance dataset: {str(e)}")
        return X, y

def learning_rate_schedule(epoch, lr):
    """Learning rate schedule for progressive training"""
    if epoch < 5:
        return lr
    elif epoch < 10:
        return lr * 0.5
    else:
        return lr * 0.1

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

def train_on_dataset(model, X, y, dataset_name, models_dir, timestamp, scaler, is_first_dataset=False):
    """Train model on a single dataset"""
    print(f"\nTraining on dataset: {dataset_name}")
    
    # Balance the dataset
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
    
    # Get number of classes
    num_classes = len(np.unique(y_balanced))
    
    # Create dataset-specific directory
    dataset_dir = os.path.join(models_dir, f'dataset_{dataset_name}_{timestamp}')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Compile model with appropriate learning rate
    if is_first_dataset:
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    else:
        # Use lower learning rate for transfer learning
        model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    # Define callbacks with more frequent checkpointing
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased patience
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(dataset_dir, 'model_checkpoint_{epoch:02d}_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        ModelCheckpoint(
            filepath=os.path.join(dataset_dir, 'model_final.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        LearningRateScheduler(learning_rate_schedule)
    ]
    
    # Train the model with class weights
    class_weights = {}
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    for class_idx in range(num_classes):
        class_weights[class_idx] = total_samples / (num_classes * class_counts.get(class_idx, 1))
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,  # Increased epochs
        batch_size=32,  # Reduced batch size for better generalization
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights,  # Add class weights
        verbose=1
    )
    
    # Save the final model for this dataset
    model.save(os.path.join(dataset_dir, 'model_completed.h5'))
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, os.path.join(dataset_dir, 'scaler.joblib'))
    
    # Save training history
    history_dict = history.history
    with open(os.path.join(dataset_dir, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f)
    
    # Plot training history
    plot_training_history(history, os.path.join(dataset_dir, 'training_history.png'))
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average=None)
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i}:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1-score:  {f:.4f}")
    
    # Save classification report
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    with open(os.path.join(dataset_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, 
                         [f"Class {i}" for i in range(num_classes)],
                         os.path.join(dataset_dir, 'confusion_matrix.png'))
    
    # Save dataset info
    dataset_info = {
        'dataset_name': dataset_name,
        'timestamp': timestamp,
        'num_classes': num_classes,
        'feature_shape': X.shape,
        'class_distribution': dict(Counter(y)),  # Fixed Counter serialization
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss)
    }
    with open(os.path.join(dataset_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f)
    
    print(f"\nAll artifacts saved in: {dataset_dir}")
    return model

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all CSV files in the datasets directory
    datasets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    if not os.path.exists(datasets_dir):
        raise ValueError(f"Datasets directory not found: {datasets_dir}")
    
    dataset_paths = find_csv_files(datasets_dir)
    if not dataset_paths:
        raise ValueError(f"No CSV files found in {datasets_dir}")
    
    print(f"Found {len(dataset_paths)} CSV files in {datasets_dir}")
    for path in dataset_paths:
        print(f"  - {os.path.basename(path)}")
    
    # Find the most recent model file
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.h5')]
    if model_files:
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        previous_model_path = os.path.join(models_dir, latest_model)
        print(f"\nLoading previous model for transfer learning: {latest_model}")
        try:
            # Load the previous model
            model = ThreatDetectionModel(input_shape=(None,), num_classes=None)  # Shape will be set by first dataset
            model.load(previous_model_path)
            print("Successfully loaded previous model")
        except Exception as e:
            print(f"Warning: Could not load previous model: {str(e)}")
            model = None
    else:
        model = None
    
    # Train on each dataset sequentially
    scaler = None
    for i, dataset_path in enumerate(dataset_paths):
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        print(f"\nProcessing dataset {i+1}/{len(dataset_paths)}: {dataset_name}")
        
        # Load and preprocess the dataset
        X, y, scaler = load_dataset(dataset_path, scaler)
        if X is None or y is None:
            print(f"Skipping dataset {dataset_name} due to loading error")
            continue
        
        # Create new model if needed
        if model is None:
            model = ThreatDetectionModel(input_shape=(X.shape[1],), num_classes=len(np.unique(y)))
            print("Created new model")
        
        # Train on this dataset
        is_first_dataset = (i == 0)
        model = train_on_dataset(model, X, y, dataset_name, models_dir, timestamp, scaler, is_first_dataset)
        
        # Freeze some layers for transfer learning
        if not is_first_dataset:
            for layer in model.layers[:-2]:  # Freeze all except last two layers
                layer.trainable = False
            print("Frozen early layers for transfer learning")
    
    print(f"\nTraining completed. All artifacts saved in: {models_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main() 