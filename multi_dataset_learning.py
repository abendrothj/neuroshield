#!/usr/bin/env python3

"""
NeuraShield Multi-Dataset Transfer Learning
This script implements transfer learning across multiple cybersecurity datasets
with feature standardization to enhance threat detection capabilities
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import logging
import pickle
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix

# Add the ai_models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models"))

# Import the custom modules
from ai_models.feature_engineering import apply_feature_engineering, unsw_feature_engineering, dimensionality_reduction
from ai_models.advanced_models import build_residual_nn, build_conv_nn, build_sequential_nn
from ai_models.ensemble_models import EnsembleModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class DatasetStandardizer:
    """Standardizes features across different cybersecurity datasets"""
    
    def __init__(self, common_features=None):
        """
        Initialize the standardizer
        
        Args:
            common_features: List of common features across datasets
        """
        self.common_features = common_features
        self.scaler = None
        self.variance_threshold = None
        self.pca = None
        self.feature_mapping = {}
    
    def _identify_common_features(self, datasets):
        """
        Identify common features across datasets
        
        Args:
            datasets: Dictionary of dataframes keyed by dataset name
        
        Returns:
            List of common feature names
        """
        # Start with features from first dataset
        common = set(datasets[list(datasets.keys())[0]].columns)
        
        # Find intersection with other datasets
        for name, df in datasets.items():
            common = common.intersection(set(df.columns))
        
        # Remove target variables and ID fields
        features_to_exclude = ['id', 'label', 'target', 'class', 'attack_cat']
        common = common - set(features_to_exclude)
        
        logging.info(f"Identified {len(common)} common features across datasets")
        return list(common)
    
    def _create_feature_mapping(self, datasets):
        """
        Create mapping of dataset-specific features to common feature space
        
        Args:
            datasets: Dictionary of dataframes keyed by dataset name
        """
        if not self.common_features:
            self.common_features = self._identify_common_features(datasets)
        
        # For each dataset, create a mapping from its features to common space
        for name, df in datasets.items():
            # For common features, use direct mapping
            mapping = {col: col for col in self.common_features if col in df.columns}
            
            # For dataset-specific features, use semantic mapping where possible
            # This requires domain knowledge and is implemented partially
            
            # Example mappings from UNSW-NB15 to CIC-IDS features
            semantic_maps = {
                'UNSW-NB15': {
                    'sbytes': 'src_bytes',
                    'dbytes': 'dst_bytes',
                    'sttl': 'src_ttl',
                    'dttl': 'dst_ttl',
                    'sloss': 'src_loss',
                    'dloss': 'dst_loss',
                    'sload': 'src_load',
                    'dload': 'dst_load'
                },
                'CIC-IDS2017': {
                    'Source Bytes': 'src_bytes',
                    'Destination Bytes': 'dst_bytes',
                    'Flow Duration': 'duration',
                    'Flow Packets/s': 'rate'
                },
                'NSL-KDD': {
                    'src_bytes': 'src_bytes',
                    'dst_bytes': 'dst_bytes',
                    'duration': 'duration'
                }
            }
            
            # Add semantic mappings if this dataset has a predefined mapping
            if name in semantic_maps:
                for src, dst in semantic_maps[name].items():
                    if src in df.columns and dst not in mapping:
                        mapping[src] = dst
            
            self.feature_mapping[name] = mapping
            logging.info(f"Created feature mapping for {name}: {len(mapping)} features mapped")
    
    def fit(self, datasets):
        """
        Fit standardizer to multiple datasets
        
        Args:
            datasets: Dictionary of dataframes keyed by dataset name
        """
        # Create feature mapping
        self._create_feature_mapping(datasets)
        
        # Combine samples from all datasets for fitting
        combined_data = []
        for name, df in datasets.items():
            # Extract mapped features
            mapping = self.feature_mapping[name]
            mapped_df = pd.DataFrame()
            
            for src, dst in mapping.items():
                if src in df.columns:
                    try:
                        # Convert to numeric and handle missing values
                        mapped_df[dst] = pd.to_numeric(df[src], errors='coerce').fillna(0)
                    except Exception as e:
                        logging.warning(f"Error converting column {src} in dataset {name}: {str(e)}")
                        # Use zeros as fallback
                        mapped_df[dst] = 0
            
            # Fill missing mapped features with zeros
            for feature in self.common_features:
                if feature not in mapped_df.columns:
                    mapped_df[feature] = 0
            
            # Add this dataset to combined data if it has content
            if not mapped_df.empty and len(mapped_df.columns) > 0:
                combined_data.append(mapped_df)
            else:
                logging.warning(f"Skipping empty mapped DataFrame for {name}")
        
        if not combined_data:
            raise ValueError("No valid data after mapping features. Check your datasets.")
        
        # Concatenate all mapped datasets
        combined_df = pd.concat(combined_data, ignore_index=True)
        logging.info(f"Combined data for standardization: {combined_df.shape}")
        
        # Check for non-numeric columns and handle them
        for col in combined_df.columns:
            if not np.issubdtype(combined_df[col].dtype, np.number):
                logging.warning(f"Column {col} is not numeric. Converting to numeric.")
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
        
        # Remove low-variance features
        try:
            self.variance_threshold = VarianceThreshold(threshold=0.01)
            data_var = self.variance_threshold.fit_transform(combined_df)
            
            # Get feature names after variance threshold
            var_support = self.variance_threshold.get_support()
            self.selected_features = [f for f, s in zip(combined_df.columns, var_support) if s]
            
            # Fit scaler
            self.scaler = StandardScaler()
            self.scaler.fit(data_var)
            
            logging.info(f"Standardizer fit complete. Selected {len(self.selected_features)} features")
        except Exception as e:
            logging.error(f"Error fitting standardizer: {str(e)}")
            # Create minimal fallback standardizer
            self.variance_threshold = None
            self.selected_features = list(combined_df.columns)
            self.scaler = StandardScaler()
            self.scaler.fit(combined_df)
            
            logging.warning(f"Using fallback standardizer with {len(self.selected_features)} features")
        
        return self
    
    def transform(self, df, dataset_name):
        """
        Transform a dataset to standardized feature space
        
        Args:
            df: Dataframe to transform
            dataset_name: Name of the dataset
            
        Returns:
            Transformed dataframe
        """
        # Apply feature mapping
        mapping = self.feature_mapping.get(dataset_name, {})
        mapped_df = pd.DataFrame()
        
        for src, dst in mapping.items():
            if src in df.columns:
                # Ensure the column is numeric before adding it
                try:
                    mapped_df[dst] = pd.to_numeric(df[src], errors='coerce').fillna(0)
                except Exception as e:
                    logging.warning(f"Error converting column {src} to numeric: {str(e)}")
                    # Use zeros as fallback
                    mapped_df[dst] = 0
        
        # Fill missing mapped features with zeros
        for feature in self.common_features:
            if feature not in mapped_df.columns:
                mapped_df[feature] = 0
        
        # Make sure the DataFrame is not empty
        if mapped_df.empty or len(mapped_df.columns) == 0:
            logging.error(f"Empty mapped DataFrame for {dataset_name}")
            # Return a dummy array with the right number of features
            dummy_array = np.zeros((df.shape[0], len(self.selected_features)))
            return dummy_array
        
        # Apply variance threshold
        try:
            data_var = self.variance_threshold.transform(mapped_df)
        except Exception as e:
            logging.error(f"Error applying variance threshold: {str(e)}")
            # Return dummy data as fallback
            data_var = np.zeros((mapped_df.shape[0], len(self.selected_features)))
            return data_var
        
        # Apply scaler
        try:
            data_scaled = self.scaler.transform(data_var)
        except Exception as e:
            logging.error(f"Error applying scaler: {str(e)}")
            # Return unscaled data as fallback
            return data_var
        
        return data_scaled
    
    def save(self, filepath):
        """Save standardizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'common_features': self.common_features,
                'scaler': self.scaler,
                'variance_threshold': self.variance_threshold,
                'pca': self.pca,
                'feature_mapping': self.feature_mapping,
                'selected_features': self.selected_features
            }, f)
        
        logging.info(f"Standardizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load standardizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        standardizer = cls(common_features=data['common_features'])
        standardizer.scaler = data['scaler']
        standardizer.variance_threshold = data['variance_threshold']
        standardizer.pca = data['pca']
        standardizer.feature_mapping = data['feature_mapping']
        standardizer.selected_features = data['selected_features']
        
        logging.info(f"Standardizer loaded from {filepath}")
        return standardizer


def load_unsw_dataset(dataset_path):
    """Load the UNSW-NB15 dataset"""
    logging.info(f"Loading UNSW-NB15 dataset from {dataset_path}")
    
    # Find training and testing files
    train_file = os.path.join(dataset_path, "UNSW_NB15_training-set.csv")
    test_file = os.path.join(dataset_path, "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Training or testing file not found in {dataset_path}")
    
    # Load datasets
    train_df = pd.read_csv(train_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)
    
    logging.info(f"UNSW-NB15 - Training data shape: {train_df.shape}")
    logging.info(f"UNSW-NB15 - Testing data shape: {test_df.shape}")
    
    # Apply feature engineering
    train_df = unsw_feature_engineering(train_df)
    test_df = unsw_feature_engineering(test_df)
    
    return {
        'train': train_df,
        'test': test_df,
        'features': [c for c in train_df.columns if c not in ['id', 'label', 'attack_cat']],
        'target': 'label'
    }


def load_cicids_dataset(dataset_path):
    """Load the CIC-IDS2017 dataset"""
    logging.info(f"Loading CIC-IDS2017 dataset from {dataset_path}")
    
    # CIC-IDS2017 has multiple CSV files for different days and attack types
    csv_files = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if f.endswith('.csv') and 'CIC-IDS-2017' in f
    ]
    
    if not csv_files:
        raise FileNotFoundError(f"No CIC-IDS2017 CSV files found in {dataset_path}")
    
    # Load and concatenate all files
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file, low_memory=False)
        dataframes.append(df)
    
    full_df = pd.concat(dataframes, ignore_index=True)
    
    # Rename target column if necessary
    if 'Label' in full_df.columns:
        full_df = full_df.rename(columns={'Label': 'label'})
    
    # Convert categorical labels to binary for consistency
    full_df['binary_label'] = full_df['label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df['binary_label']
    )
    
    logging.info(f"CIC-IDS2017 - Training data shape: {train_df.shape}")
    logging.info(f"CIC-IDS2017 - Testing data shape: {test_df.shape}")
    
    return {
        'train': train_df,
        'test': test_df,
        'features': [c for c in train_df.columns if c not in ['label', 'binary_label']],
        'target': 'binary_label'
    }


def load_nslkdd_dataset(dataset_path):
    """Load the NSL-KDD dataset"""
    logging.info(f"Loading NSL-KDD dataset from {dataset_path}")
    
    # Find training and testing files
    train_file = os.path.join(dataset_path, "KDDTrain+.txt")
    test_file = os.path.join(dataset_path, "KDDTest+.txt")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Training or testing file not found in {dataset_path}")
    
    # NSL-KDD dataset columns
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'class', 'difficulty'
    ]
    
    # Load datasets
    train_df = pd.read_csv(train_file, header=None, names=columns)
    test_df = pd.read_csv(test_file, header=None, names=columns)
    
    # Convert categorical labels to binary
    train_df['binary_label'] = train_df['class'].apply(
        lambda x: 0 if x == 'normal' else 1
    )
    test_df['binary_label'] = test_df['class'].apply(
        lambda x: 0 if x == 'normal' else 1
    )
    
    # Convert categorical features to numeric
    for col in ['protocol_type', 'service', 'flag']:
        train_df[col] = train_df[col].astype('category').cat.codes
        test_df[col] = test_df[col].astype('category').cat.codes
    
    logging.info(f"NSL-KDD - Training data shape: {train_df.shape}")
    logging.info(f"NSL-KDD - Testing data shape: {test_df.shape}")
    
    return {
        'train': train_df,
        'test': test_df,
        'features': [c for c in train_df.columns if c not in ['class', 'difficulty', 'binary_label']],
        'target': 'binary_label'
    }


def load_cse_cic_ids2018_dataset(dataset_path):
    """Load the CSE-CIC-IDS2018 dataset"""
    logging.info(f"Loading CSE-CIC-IDS2018 dataset from {dataset_path}")
    
    # CSE-CIC-IDS2018 has multiple parquet files for different attack scenarios
    parquet_files = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if f.endswith('.parquet') and 'TrafficForML_CICFlowMeter' in f
    ]
    
    if not parquet_files:
        raise FileNotFoundError(f"No CSE-CIC-IDS2018 parquet files found in {dataset_path}")
    
    # Load and concatenate all files
    dataframes = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        # Extract the attack type from the filename
        attack_type = os.path.basename(file).split('-')[0]
        df['attack_type'] = attack_type
        dataframes.append(df)
    
    full_df = pd.concat(dataframes, ignore_index=True)
    
    # Make sure there's a 'label' column (usually is 'Label')
    if 'Label' in full_df.columns:
        full_df = full_df.rename(columns={'Label': 'label'})
    
    # Convert categorical labels to binary
    full_df['binary_label'] = full_df['label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df['binary_label']
    )
    
    logging.info(f"CSE-CIC-IDS2018 - Training data shape: {train_df.shape}")
    logging.info(f"CSE-CIC-IDS2018 - Testing data shape: {test_df.shape}")
    
    return {
        'train': train_df,
        'test': test_df,
        'features': [c for c in train_df.columns if c not in ['label', 'binary_label', 'attack_type']],
        'target': 'binary_label'
    }


def load_cic_ddos19_dataset(dataset_path):
    """Load the CIC-DDoS19 dataset"""
    logging.info(f"Loading CIC-DDoS19 dataset from {dataset_path}")
    
    # Find the CSV file
    csv_files = [
        os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
        if f.endswith('.csv') and ('Random_combine' in f or 'DDoS' in f)
    ]
    
    if not csv_files:
        raise FileNotFoundError(f"No CIC-DDoS19 CSV files found in {dataset_path}")
    
    # Load the entire file first to get column names correctly
    logging.info(f"Reading CSV file: {csv_files[0]}")
    
    try:
        # Use pandas' automatic column name cleaning
        df = pd.read_csv(csv_files[0], low_memory=False, skipinitialspace=True)
        logging.info(f"CSV file loaded with shape: {df.shape}")
        logging.info(f"Column names: {df.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Clean up columns by removing spaces and special characters from names
        df.columns = [col.strip().replace(' ', '_').replace('/', '_').replace('-', '_') for col in df.columns]
        
        # Identify and remove problematic columns
        # IP addresses, timestamps, etc.
        drop_patterns = ['IP', 'Timestamp', 'Flow_ID']
        drop_cols = [col for col in df.columns if any(pattern in col for pattern in drop_patterns)]
        
        if drop_cols:
            logging.info(f"Dropping problematic columns: {drop_cols}")
            df = df.drop(columns=drop_cols, errors='ignore')
        
        # Find the label column
        label_col = None
        # Look for Label column in various formats
        label_candidates = ['Label', 'label', 'CLASS', 'class', 'attack', 'Attack']
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        # If no label column found, use the last column
        if label_col is None:
            label_col = df.columns[-1]
            logging.warning(f"No label column found, using last column: {label_col}")
        
        logging.info(f"Using {label_col} as the label column")
        
        # Create binary label
        try:
            if df[label_col].dtype == object:
                # For string labels, convert "benign" to 0, everything else to 1
                df['binary_label'] = df[label_col].apply(
                    lambda x: 0 if str(x).lower() == 'benign' else 1)
            else:
                # For numeric labels, 0=benign, anything else=attack
                df['binary_label'] = df[label_col].apply(
                    lambda x: 0 if x == 0 or x == 0.0 else 1)
                
            logging.info(f"Binary label distribution: {df['binary_label'].value_counts().to_dict()}")
        except Exception as e:
            logging.error(f"Error creating binary labels: {str(e)}")
            # If we can't create labels, just set everything as an attack
            df['binary_label'] = 1
            logging.warning("Setting all samples as attacks due to label processing error")
        
        # Feature columns - everything except the label and binary label
        feature_cols = [col for col in df.columns if col != label_col and col != 'binary_label']
        
        # Convert all columns to numeric format
        for col in feature_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Error converting column {col} to numeric: {str(e)}")
                # Replace with zeros if conversion fails
                df[col] = 0
        
        # Fill missing values
        df = df.fillna(0)
        
        # Create stratified train/test split
        logging.info(f"Creating train/test split for dataset with shape: {df.shape}")
        try:
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, 
                stratify=df['binary_label'] if len(df['binary_label'].unique()) > 1 else None
            )
        except Exception as e:
            logging.error(f"Error in train/test split: {str(e)}")
            # Fallback to random split if stratification fails
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        logging.info(f"CIC-DDoS19 - Training data shape: {train_df.shape}")
        logging.info(f"CIC-DDoS19 - Testing data shape: {test_df.shape}")
        
        return {
            'train': train_df,
            'test': test_df,
            'features': feature_cols,
            'target': 'binary_label'
        }
        
    except Exception as e:
        logging.error(f"Error processing CIC-DDoS19 dataset: {str(e)}")
        raise


def transfer_learning(base_model_path, target_dataset, output_dir, epochs=20, batch_size=32):
    """
    Implement transfer learning from base model to target dataset
    
    Args:
        base_model_path: Path to base model
        target_dataset: Target dataset dict with train, test, features, target
        output_dir: Output directory
        epochs: Number of epochs for fine-tuning
        batch_size: Batch size for training
    """
    logging.info(f"Implementing transfer learning from {base_model_path}")
    
    # Load base model
    try:
        logging.info(f"Attempting to load model from: {base_model_path}")
        
        # Check if the path exists first
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Model path does not exist: {base_model_path}")
            
        # Try to load the model
        if os.path.isdir(base_model_path) and "ensemble_model" in base_model_path:
            # This is an ensemble model directory
            from ai_models.ensemble_models import EnsembleModel
            base_model = EnsembleModel.load(base_model_path)
            # Get the meta-model for transfer learning
            if hasattr(base_model, 'meta_model') and base_model.meta_model is not None:
                base_model = base_model.meta_model
            else:
                raise ValueError("Cannot extract meta-model from ensemble for transfer learning")
        else:
            # Regular Keras model
            base_model = tf.keras.models.load_model(base_model_path)
            
        logging.info(f"Base model loaded successfully with {len(base_model.layers)} layers")
    except Exception as e:
        error_msg = f"Error loading base model: {str(e)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Extract target dataset components
    X_train = target_dataset['X_train']
    y_train = target_dataset['y_train']
    X_val = target_dataset['X_val']
    y_val = target_dataset['y_val']
    X_test = target_dataset['X_test']
    y_test = target_dataset['y_test']
    
    # Create model for transfer learning
    # Freeze base layers
    for layer in base_model.layers[:-2]:  # Keep last two layers trainable
        layer.trainable = False
        logging.info(f"Freezing layer: {layer.name}")
    
    # Replace the output layer if number of classes differs
    num_classes = len(np.unique(y_train))
    
    if base_model.layers[-1].units != num_classes:
        logging.info(f"Replacing output layer to match {num_classes} classes")
        # Remove last layer and add a new output layer
        x = base_model.layers[-2].output
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        transfer_model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    else:
        transfer_model = base_model
    
    # Compile model with a low learning rate
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune on target dataset
    os.makedirs(output_dir, exist_ok=True)
    history = transfer_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, "best_model.keras"),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
    
    # Evaluate model
    evaluation = transfer_model.evaluate(X_test, y_test)
    logging.info(f"Transfer model evaluation - Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")
    
    # Save model
    transfer_model.save(os.path.join(output_dir, "transfer_model.keras"))
    
    # Save training history
    with open(os.path.join(output_dir, "transfer_history.pkl"), 'wb') as f:
        pickle.dump(history.history, f)
    
    # Create predictions on test set for analysis
    y_pred = transfer_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate classification report
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "transfer_classification_report.csv"))
    
    # Generate confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Label with class names if available
    class_names = [f"Class_{i}" for i in range(num_classes)]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Display values in the matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, "transfer_confusion_matrix.png"))
    plt.close()
    
    return transfer_model, history


def prepare_datasets(args):
    """
    Load and prepare datasets for multi-dataset learning
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of prepared datasets
    """
    datasets = {}
    
    # Load UNSW-NB15 dataset if specified
    if args.unsw_path:
        try:
            unsw_data = load_unsw_dataset(args.unsw_path)
            datasets['UNSW-NB15'] = unsw_data
        except Exception as e:
            logging.error(f"Error loading UNSW-NB15 dataset: {str(e)}")
    
    # Load CIC-IDS2017 dataset if specified
    if args.cicids_path:
        try:
            cicids_data = load_cicids_dataset(args.cicids_path)
            datasets['CIC-IDS2017'] = cicids_data
        except Exception as e:
            logging.error(f"Error loading CIC-IDS2017 dataset: {str(e)}")
    
    # Load NSL-KDD dataset if specified
    if args.nslkdd_path:
        try:
            nslkdd_data = load_nslkdd_dataset(args.nslkdd_path)
            datasets['NSL-KDD'] = nslkdd_data
        except Exception as e:
            logging.error(f"Error loading NSL-KDD dataset: {str(e)}")
    
    # Load CSE-CIC-IDS2018 dataset if specified
    if args.cse_cic_ids2018_path:
        try:
            cse_cicids_data = load_cse_cic_ids2018_dataset(args.cse_cic_ids2018_path)
            datasets['CSE-CIC-IDS2018'] = cse_cicids_data
        except Exception as e:
            logging.error(f"Error loading CSE-CIC-IDS2018 dataset: {str(e)}")
    
    # Load CIC-DDoS19 dataset if specified
    if args.cic_ddos19_path:
        try:
            cic_ddos_data = load_cic_ddos19_dataset(args.cic_ddos19_path)
            datasets['CIC-DDoS19'] = cic_ddos_data
        except Exception as e:
            logging.error(f"Error loading CIC-DDoS19 dataset: {str(e)}")
    
    if not datasets:
        raise ValueError("No datasets were successfully loaded")
    
    logging.info(f"Loaded {len(datasets)} datasets: {', '.join(datasets.keys())}")
    
    # Create and fit standardizer
    standardizer = DatasetStandardizer()
    standardizer.fit({name: data['train'] for name, data in datasets.items()})
    
    # Save standardizer
    os.makedirs(args.output_dir, exist_ok=True)
    standardizer.save(os.path.join(args.output_dir, "standardizer.pkl"))
    
    # Standardize datasets and prepare for training
    prepared_datasets = {}
    
    for name, data in datasets.items():
        # Prepare features and target
        X_train = standardizer.transform(data['train'], name)
        
        if name == 'UNSW-NB15':
            y_train = data['train']['label'].values 
            X_test = standardizer.transform(data['test'], name)
            y_test = data['test']['label'].values
        elif name == 'CIC-IDS2017':
            y_train = data['train']['binary_label'].values
            X_test = standardizer.transform(data['test'], name)
            y_test = data['test']['binary_label'].values
        elif name == 'NSL-KDD':
            y_train = data['train']['binary_label'].values
            X_test = standardizer.transform(data['test'], name)
            y_test = data['test']['binary_label'].values
        elif name == 'CSE-CIC-IDS2018':
            y_train = data['train']['binary_label'].values
            X_test = standardizer.transform(data['test'], name)
            y_test = data['test']['binary_label'].values
        elif name == 'CIC-DDoS19':
            y_train = data['train']['binary_label'].values
            X_test = standardizer.transform(data['test'], name)
            y_test = data['test']['binary_label'].values
        
        # Split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        prepared_datasets[name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'num_features': X_train.shape[1],
            'num_classes': len(np.unique(y_train))
        }
        
        logging.info(f"Prepared {name} dataset:")
        logging.info(f"  - Training features: {X_train.shape}")
        logging.info(f"  - Validation features: {X_val.shape}")
        logging.info(f"  - Testing features: {X_test.shape}")
        logging.info(f"  - Number of classes: {prepared_datasets[name]['num_classes']}")
    
    return prepared_datasets, standardizer


def train_base_model(dataset_name, dataset, output_dir, model_type='residual', epochs=30, batch_size=32):
    """
    Train base model on a specific dataset
    
    Args:
        dataset_name: Name of the dataset
        dataset: Prepared dataset dictionary
        output_dir: Output directory
        model_type: Type of model to train
        epochs: Number of epochs
        batch_size: Batch size
    """
    logging.info(f"Training base model on {dataset_name} dataset")
    
    # Extract dataset components
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    num_features = dataset['num_features']
    num_classes = dataset['num_classes']
    
    # Create timestamp for model versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir
    os.makedirs(model_dir, exist_ok=True)
    
    # Build appropriate model
    if model_type == 'residual':
        model = build_residual_nn(num_features, num_classes, units=128, num_blocks=3)
    elif model_type == 'conv':
        model = build_conv_nn(num_features, num_classes)
    elif model_type == 'sequential':
        model = build_sequential_nn(num_features, num_classes)
    elif model_type == 'ensemble':
        # Use ensemble model with residual networks
        ensemble = EnsembleModel(num_features, num_classes)
        
        # Train base models
        base_histories = ensemble.train_base_models(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        # Train meta-model
        meta_history = ensemble.train_meta_model(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=batch_size
        )
        
        # Save ensemble model
        ensemble_dir = os.path.join(model_dir, "ensemble_model")
        os.makedirs(ensemble_dir, exist_ok=True)
        ensemble.save(ensemble_dir)
        
        # For consistency, also save as a keras model if possible
        try:
            ensemble_keras_path = os.path.join(model_dir, "best_model.keras")
            if hasattr(ensemble, 'meta_model') and ensemble.meta_model is not None:
                ensemble.meta_model.save(ensemble_keras_path)
                logging.info(f"Saved ensemble meta model to {ensemble_keras_path}")
        except Exception as e:
            logging.warning(f"Could not save ensemble as keras model: {str(e)}")
        
        # Evaluate model
        y_pred = ensemble.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred_classes == y_test)
        logging.info(f"Ensemble model accuracy on test set: {accuracy:.4f}")
        
        # Generate classification report
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(model_dir, "classification_report.csv"))
        
        return ensemble, meta_history
    
    # Train standard model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, "best_model.keras"),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
    )
    
    # Save the final model - use a consistent name
    model.save(os.path.join(model_dir, "final_model.keras"))
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{dataset_name} - {model_type} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, f"{model_type}_accuracy.png"))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{dataset_name} - {model_type} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, f"{model_type}_loss.png"))
    plt.close()
    
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test)
    logging.info(f"Model evaluation - Loss: {evaluation[0]}, Accuracy: {evaluation[1]}")
    
    # Generate predictions and classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(model_dir, "classification_report.csv"))
    
    return model, history


def perform_transfer_learning(args, datasets, standardizer):
    """
    Perform transfer learning between datasets
    
    Args:
        args: Command line arguments
        datasets: Dictionary of prepared datasets
        standardizer: DatasetStandardizer instance
    """
    # Train base model on source dataset
    source_name = args.source_dataset
    target_name = args.target_dataset
    
    if source_name not in datasets or target_name not in datasets:
        raise ValueError(f"Source or target dataset not found. Available datasets: {list(datasets.keys())}")
    
    # Create output directories
    base_model_dir = os.path.join(args.output_dir, f"{source_name}_base")
    os.makedirs(base_model_dir, exist_ok=True)
    
    transfer_model_dir = os.path.join(args.output_dir, f"{source_name}_to_{target_name}")
    os.makedirs(transfer_model_dir, exist_ok=True)
    
    # Train base model on source dataset
    logging.info(f"Training base model on {source_name}")
    base_model, base_history = train_base_model(
        source_name, 
        datasets[source_name],
        base_model_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Find the actual model path that was saved
    # Check different possible filenames in priority order
    possible_model_paths = [
        os.path.join(base_model_dir, "best_model.keras"),
        os.path.join(base_model_dir, "final_model.keras"),
        os.path.join(base_model_dir, f"{args.model_type}_model.keras")
    ]
    
    # For ensemble models
    if args.model_type == 'ensemble':
        possible_model_paths.insert(0, os.path.join(base_model_dir, "ensemble_model"))
    
    # Find the first existing model file
    base_model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            base_model_path = path
            logging.info(f"Found base model at: {base_model_path}")
            break
    
    if base_model_path is None:
        # If no model file is found, save the model we just trained
        base_model_path = os.path.join(base_model_dir, "base_model.keras")
        if hasattr(base_model, 'save'):
            base_model.save(base_model_path)
            logging.info(f"Saved base model to: {base_model_path}")
        else:
            raise FileNotFoundError(f"No model file found in {base_model_dir} and cannot save current model")
    
    # Perform transfer learning to target dataset
    logging.info(f"Performing transfer learning from {source_name} to {target_name}")
    
    transfer_model, transfer_history = transfer_learning(
        base_model_path,
        datasets[target_name],
        transfer_model_dir,
        epochs=args.transfer_epochs,
        batch_size=args.batch_size
    )
    
    # Save standardizer with the transfer model
    standardizer.save(os.path.join(transfer_model_dir, "standardizer.pkl"))
    
    logging.info(f"Transfer learning complete. Models saved to {transfer_model_dir}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-dataset transfer learning for threat detection')
    
    # Dataset paths
    parser.add_argument('--unsw-path', type=str,
                        help='Path to UNSW-NB15 dataset')
    parser.add_argument('--cicids-path', type=str,
                        help='Path to CIC-IDS2017 dataset')
    parser.add_argument('--nslkdd-path', type=str,
                        help='Path to NSL-KDD dataset')
    parser.add_argument('--cse-cic-ids2018-path', type=str,
                        help='Path to CSE-CIC-IDS2018 dataset')
    parser.add_argument('--cic-ddos19-path', type=str,
                        help='Path to CIC-DDoS19 dataset')
    
    # Transfer learning settings
    parser.add_argument('--source-dataset', type=str, default='UNSW-NB15',
                        help='Source dataset for transfer learning')
    parser.add_argument('--target-dataset', type=str, default='CIC-IDS2017',
                        help='Target dataset for transfer learning')
    
    # Model settings
    parser.add_argument('--model-type', type=str, default='residual',
                        choices=['residual', 'conv', 'sequential', 'ensemble'],
                        help='Type of model to train')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for base model training')
    parser.add_argument('--transfer-epochs', type=int, default=20,
                        help='Number of epochs for transfer learning')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='models/multi_dataset',
                        help='Directory to save models and results')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file and console
    log_file = os.path.join(args.output_dir, "multi_dataset_learning.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load and prepare datasets
        datasets, standardizer = prepare_datasets(args)
        
        # Perform transfer learning
        perform_transfer_learning(args, datasets, standardizer)
        
        # Log completion time
        elapsed_time = time.time() - start_time
        logging.info(f"Multi-dataset learning completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error during multi-dataset learning: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 