import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
        
        # Special handling for timestamp/date columns
        date_columns = [col for col in df.columns if any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
        for col in date_columns:
            try:
                # Try different date formats
                for fmt in ['%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt)
                        print(f"Converted {col} using format {fmt}")
                        break
                    except:
                        continue
                
                # Extract time features
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_second'] = df[col].dt.second
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_year'] = df[col].dt.year
                
                # Drop original column
                df = df.drop(columns=[col])
                print(f"Extracted time features from {col}")
            except Exception as e:
                print(f"Warning: Could not process date column {col}: {str(e)}")
        
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
        
        # Only one-hot encode essential categorical columns (like Protocol)
        essential_categorical = ['Protocol']  # Add other essential categorical columns here
        truly_categorical = [col for col in categorical_columns if col in essential_categorical]
        if truly_categorical:
            print(f"One-hot encoding essential categorical columns: {truly_categorical}")
            df_encoded = pd.get_dummies(df, columns=truly_categorical, prefix=truly_categorical)
            return df_encoded
    
    return df

def align_features(df, common_features=None):
    """Align features across datasets"""
    if common_features is None:
        # First dataset sets the features
        return df, list(df.columns)
    
    # Add missing features with zeros
    for feature in common_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Remove extra features
    df = df[common_features]
    return df, common_features

def load_dataset(dataset_path, scaler=None, common_features=None):
    """Load and preprocess a single dataset"""
    print(f"\nLoading dataset from: {dataset_path}")
    
    try:
        # Load the dataset
        df = pd.read_csv(dataset_path, low_memory=False, na_values=[' ', '-', 'nan', 'NaN'])
        
        # Handle categorical features
        df = handle_categorical_features(df)
        
        # Align features
        df, common_features = align_features(df, common_features)
        
        # Handle labels (adjust based on dataset)
        label_column = next((col for col in ['label', 'Label', 'class', 'Class'] if col in df.columns), None)
        if label_column is None:
            raise ValueError("No label column found in dataset")
        
        X = df.drop(columns=[label_column]).values
        y = df[label_column].values
        
        # Convert labels to numeric if needed
        if not np.issubdtype(y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Fill missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        if scaler is None:
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        else:
            X = scaler.transform(X)
        
        print(f"Memory usage after loading {os.path.basename(dataset_path)}: {get_memory_usage():.2f} MB")
        print(f"Feature shape: {X.shape}")
        return X, y, scaler, common_features
        
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {str(e)}")
        return None, None, None, None

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
    if epoch < 10:
        return lr
    elif epoch < 20:
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

def main():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize the system
    system = ThreatDetectionSystem()
    
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
    
    # Initialize lists to store combined data
    all_X = []
    all_y = []
    scaler = None
    common_features = None
    
    # Load and combine all datasets
    for dataset_path in dataset_paths:
        X, y, scaler, common_features = load_dataset(dataset_path, scaler, common_features)
        if X is not None and y is not None:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X or not all_y:
        raise ValueError("No valid datasets were loaded")
    
    # Combine all datasets
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"Total memory usage after loading all datasets: {get_memory_usage():.2f} MB")
    
    # Balance the combined dataset
    X, y = balance_dataset(X, y)
    
    # Get unique classes and their counts
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    print(f"\nTotal number of classes across all datasets: {num_classes}")
    print("Class distribution:")
    for cls in unique_classes:
        count = np.sum(y == cls)
        print(f"Class {cls}: {count} samples")
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create and compile the model
    print("\nCreating model...")
    model = ThreatDetectionModel(input_shape=(X_train.shape[1],), num_classes=num_classes)
    
    # Find the most recent model file
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.h5')]
    if model_files:
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        previous_model_path = os.path.join(models_dir, latest_model)
        print(f"Loading previous model for transfer learning: {latest_model}")
        try:
            model.load(previous_model_path)
            # Freeze some layers to prevent catastrophic forgetting
            for layer in model.layers[:-2]:  # Freeze all except last two layers
                layer.trainable = False
        except Exception as e:
            print(f"Warning: Could not load previous model: {str(e)}")
    
    model.compile(optimizer=Adam(learning_rate=0.001),  # Start with higher learning rate
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # Increased patience
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=os.path.join(models_dir, f'multi_dataset_threat_detection_{timestamp}_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        LearningRateScheduler(learning_rate_schedule)
    ]
    
    # Train the model
    print("\nTraining model on combined datasets...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,  # Increased batch size
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, os.path.join(models_dir, f'multi_dataset_threat_detection_{timestamp}_scaler.joblib'))
    
    # Plot training history
    plot_training_history(history, os.path.join(models_dir, f'multi_dataset_threat_detection_{timestamp}_history.png'))
    
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
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, 
                         [f"Class {i}" for i in range(num_classes)],
                         os.path.join(models_dir, f'multi_dataset_threat_detection_{timestamp}_confusion_matrix.png'))
    
    print(f"\nModel and artifacts saved in: {models_dir}")
    print(f"Timestamp: {timestamp}")

if __name__ == "__main__":
    main() 