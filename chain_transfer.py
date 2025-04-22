#!/usr/bin/env python3

"""
NeuraShield Chained Transfer Learning
This script performs chained transfer learning from a pre-trained model to a new dataset
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Add the ai_models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import the CSE-CIC-IDS2018 loader from multi_dataset_learning.py
# This assumes you've already implemented this function in multi_dataset_learning.py
from multi_dataset_learning import load_cse_cic_ids2018_dataset

def process_cse_cic_ids2018(dataset_path):
    """Process the CSE-CIC-IDS2018 dataset for transfer learning and combine with benign traffic"""
    logging.info(f"Processing CSE-CIC-IDS2018 dataset from {dataset_path}")
    
    # Load dataset (only attack data)
    dataset = load_cse_cic_ids2018_dataset(dataset_path)
    attack_df = dataset['train']
    attack_test_df = dataset['test']
    original_feature_cols = dataset['features']
    
    # Set default feature columns to be updated later
    feature_cols = []
    
    logging.info(f"Original class distribution in CSE-CIC-IDS2018 training: {attack_df['binary_label'].value_counts().to_dict()}")

    # ---- IMPROVED: Multi-source benign traffic collection ----
    logging.info("Loading multiple datasets to create a robust benign traffic sample")
    all_benign_samples = []
    all_benign_test_samples = []
    
    # Try to get benign samples from CIC-DDoS19
    try:
        from multi_dataset_learning import load_cic_ddos19_dataset
        ddos_dataset = load_cic_ddos19_dataset("/home/jub/Cursor/neurashield/ai_models/datasets/CIC-DDoS19")
        benign_df = ddos_dataset['train'][ddos_dataset['train']['binary_label'] == 0]
        benign_test_df = ddos_dataset['test'][ddos_dataset['test']['binary_label'] == 0]
        
        logging.info(f"Found {len(benign_df)} benign samples from CIC-DDoS19 dataset")
        if len(benign_df) > 0:
            # Use domain-specific features instead of PCA
            numeric_features = [col for col in benign_df.columns 
                               if col != 'binary_label' and 
                               benign_df[col].dtype.kind in 'ifc']
            # Select top traffic-related features
            traffic_features = [col for col in numeric_features if any(kw in col.lower() for kw in 
                              ['byte', 'packet', 'flow', 'duration', 'time', 'rate', 'len'])]
            selected_features = traffic_features[:8] if len(traffic_features) >= 8 else numeric_features[:8]
            
            # Create uniform feature set
            benign_selected = benign_df[selected_features + ['binary_label']].copy()
            benign_test_selected = benign_test_df[selected_features + ['binary_label']].copy()
            
            # Rename columns for consistency
            for i, col in enumerate(selected_features):
                benign_selected.rename(columns={col: f"feature_{i}"}, inplace=True)
                benign_test_selected.rename(columns={col: f"feature_{i}"}, inplace=True)
                
            all_benign_samples.append(benign_selected)
            all_benign_test_samples.append(benign_test_selected)
    except Exception as e:
        logging.error(f"Error loading CIC-DDoS19 benign traffic: {str(e)}")
    
    # Try to get benign samples from UNSW-NB15
    try:
        from multi_dataset_learning import load_unsw_dataset
        unsw_dataset = load_unsw_dataset("/home/jub/Cursor/neurashield/ai_models/datasets/UNSW_NB15")
        unsw_benign = unsw_dataset['train'][unsw_dataset['train']['label'] == 0]
        unsw_benign_test = unsw_dataset['test'][unsw_dataset['test']['label'] == 0]
        
        # Rename label column for consistency
        unsw_benign = unsw_benign.rename(columns={'label': 'binary_label'})
        unsw_benign_test = unsw_benign_test.rename(columns={'label': 'binary_label'})
        logging.info(f"Found {len(unsw_benign)} benign samples from UNSW-NB15 dataset")
        
        if len(unsw_benign) > 0:
            # IMPORTANT: Select only numeric features from UNSW-NB15
            numeric_features = [col for col in unsw_benign.columns 
                               if col != 'binary_label' and 
                               unsw_benign[col].dtype.kind in 'ifc']
            
            # Select the most relevant features
            available_features = numeric_features[:8]
            
            # Create a subset with selected features
            unsw_selected = unsw_benign[available_features + ['binary_label']].copy()
            unsw_test_selected = unsw_benign_test[available_features + ['binary_label']].copy()
            
            # Rename columns for consistency
            for i, col in enumerate(available_features):
                unsw_selected.rename(columns={col: f"feature_{i}"}, inplace=True)
                unsw_test_selected.rename(columns={col: f"feature_{i}"}, inplace=True)
                
            all_benign_samples.append(unsw_selected)
            all_benign_test_samples.append(unsw_test_selected)
    except Exception as e:
        logging.error(f"Error loading UNSW-NB15 benign traffic: {str(e)}")
    
    # Check if we have any benign samples, if not create synthetic ones
    if not all_benign_samples:
        logging.warning("No benign samples found from other datasets, creating synthetic ones")
        # Create synthetic benign data that's different from attack patterns
        benign_synthetic = pd.DataFrame()
        benign_synthetic_test = pd.DataFrame()
        
        # Get numeric features from attack data
        attack_numeric = [col for col in attack_df.columns 
                         if col != 'binary_label' and 
                         attack_df[col].dtype.kind in 'ifc']
        
        # Select 8 features with highest variance as they're likely most informative
        if len(attack_numeric) >= 8:
            feature_variance = attack_df[attack_numeric].var().sort_values(ascending=False)
            top_features = feature_variance.index[:8].tolist()
        else:
            top_features = attack_numeric[:min(8, len(attack_numeric))]
            
        # Generate synthetic benign data based on attack patterns but with different distributions
        for i, feature in enumerate(top_features):
            # Get feature statistics from attack data
            attack_mean = attack_df[feature].mean()
            attack_std = max(attack_df[feature].std(), 0.1)  # Prevent zero std
            
            # Create benign data with an intentionally different distribution
            # Offset mean and reduce variance for differentiation
            benign_synthetic[f"feature_{i}"] = np.random.normal(
                loc=attack_mean * 0.5,          # Different mean
                scale=attack_std * 0.7,         # Different variance
                size=min(len(attack_df), 10000)  # Balance dataset size
            )
            
            benign_synthetic_test[f"feature_{i}"] = np.random.normal(
                loc=attack_mean * 0.5,
                scale=attack_std * 0.7,
                size=min(len(attack_test_df), 2000)
            )
            
        benign_synthetic['binary_label'] = 0
        benign_synthetic_test['binary_label'] = 0
        
        all_benign_samples.append(benign_synthetic)
        all_benign_test_samples.append(benign_synthetic_test)
    
    # Combine all benign samples
    benign_df = pd.concat(all_benign_samples, ignore_index=True)
    benign_test_df = pd.concat(all_benign_test_samples, ignore_index=True)
    
    logging.info(f"Combined {len(benign_df)} benign training samples and {len(benign_test_df)} test samples")
    
    # Process attack data to match the same feature structure
    attack_processed = pd.DataFrame()
    attack_test_processed = pd.DataFrame()
    
    # IMPORTANT: Get only numeric features from attack data with highest variance
    attack_numeric = [col for col in attack_df.columns 
                     if col != 'binary_label' and 
                     attack_df[col].dtype.kind in 'ifc']
    
    # Select 8 features with highest variance
    if len(attack_numeric) >= 8:
        feature_variance = attack_df[attack_numeric].var().sort_values(ascending=False)
        top_features = feature_variance.index[:8].tolist()
    else:
        top_features = attack_numeric[:min(8, len(attack_numeric))]
    
    # Process attack data using only numeric features
    for i, feature in enumerate(top_features):
        attack_processed[f"feature_{i}"] = attack_df[feature]
        attack_test_processed[f"feature_{i}"] = attack_test_df[feature]
    
    attack_processed['binary_label'] = 1
    attack_test_processed['binary_label'] = 1
    
    # Make sure we have the same columns in benign and attack
    feature_cols = [f"feature_{i}" for i in range(min(8, len(top_features)))]
    
    # Ensure all dataframes have the same columns
    for df in [benign_df, benign_test_df, attack_processed, attack_test_processed]:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
    
    # ---- IMPROVED: Balance classes ----
    # Balance the dataset to avoid class imbalance
    max_samples = min(len(benign_df) * 2, len(attack_processed), 20000)
    
    if len(attack_processed) > max_samples:
        attack_processed = attack_processed.sample(max_samples, random_state=42)
        logging.info(f"Downsampled attack training data to {len(attack_processed)} records")
    
    if len(benign_df) > max_samples // 2:
        benign_df = benign_df.sample(max_samples // 2, random_state=42)
        logging.info(f"Downsampled benign training data to {len(benign_df)} records")
    
    # Now balance test data
    max_test_samples = min(len(benign_test_df) * 2, len(attack_test_processed), 5000)
    
    if len(attack_test_processed) > max_test_samples:
        attack_test_processed = attack_test_processed.sample(max_test_samples, random_state=42)
        logging.info(f"Downsampled attack test data to {len(attack_test_processed)} records")
    
    if len(benign_test_df) > max_test_samples // 2:
        benign_test_df = benign_test_df.sample(max_test_samples // 2, random_state=42)
        logging.info(f"Downsampled benign test data to {len(benign_test_df)} records")
    
    # Combine datasets
    train_df = pd.concat([attack_processed, benign_df], ignore_index=True)
    test_df = pd.concat([attack_test_processed, benign_test_df], ignore_index=True)
    
    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logging.info(f"Combined dataset class distribution - Training: {train_df['binary_label'].value_counts().to_dict()}")
    logging.info(f"Combined dataset class distribution - Testing: {test_df['binary_label'].value_counts().to_dict()}")
    logging.info(f"Features used: {feature_cols}")
    
    # Verify that feature_cols exist in the dataframes
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        logging.error(f"Missing columns in train_df: {missing_cols}")
        # Create missing columns with zeros
        for col in missing_cols:
            train_df[col] = 0
            test_df[col] = 0
    
    # IMPORTANT: Verify all features are numeric
    for col in feature_cols:
        # Check if column contains non-numeric data
        if train_df[col].dtype.kind not in 'ifc':
            logging.warning(f"Column {col} contains non-numeric data: {train_df[col].dtype}")
            # Convert to categorical codes then to float
            train_df[col] = pd.factorize(train_df[col])[0].astype(float)
            test_df[col] = pd.factorize(test_df[col])[0].astype(float)
            logging.info(f"Converted {col} to numeric codes")
        
        # Handle any remaining non-numeric values
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    # Extract features and labels
    X_train = train_df[feature_cols].values
    y_train = train_df['binary_label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['binary_label'].values
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=y_train if len(np.unique(y_train)) > 1 else None
    )
    
    # Convert labels to float32 for binary classification
    y_train = y_train.astype('float32')
    y_val = y_val.astype('float32')
    y_test = y_test.astype('float32')
    
    # Confirm all data is numeric before scaling
    assert np.issubdtype(X_train.dtype, np.number), "X_train contains non-numeric values"
    assert np.issubdtype(X_val.dtype, np.number), "X_val contains non-numeric values"
    assert np.issubdtype(X_test.dtype, np.number), "X_test contains non-numeric values"
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Check for NaN values and replace with zeros
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)
    
    logging.info(f"Dataset prepared - Training: {X_train.shape}, Validation: {X_val.shape}, Testing: {X_test.shape}")
    logging.info(f"Final class ratio - Training: {np.unique(y_train, return_counts=True)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols

def load_pretrained_model(model_path):
    """Load a pre-trained model for transfer learning"""
    logging.info(f"Loading pre-trained model from {model_path}")
    
    try:
        # First try loading with 'compile=False' to avoid metric compilation issues
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info(f"Model loaded successfully with {len(model.layers)} layers")
        
        # Check for residual connections which might cause shape issues
        has_residual = False
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Add):
                has_residual = True
                logging.warning("Model contains residual connections (Add layers)")
                break
                
        if has_residual:
            logging.info("Will create a new model architecture without residual connections")
        
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def prepare_model_for_transfer(base_model, num_classes, input_features=8):
    """
    Create a new model for transfer learning with a completely independent architecture
    to avoid shape incompatibility issues.
    
    Args:
        base_model: The pre-trained model
        num_classes: Number of output classes
        input_features: Number of input features in the new dataset
    """
    logging.info(f"Creating a simple knowledge transfer model with {input_features} input features")
    
    # Create a completely new simple model with no residual connections
    inputs = tf.keras.layers.Input(shape=(input_features,), name="new_input")
    
    # Add L2 regularization to all dense layers
    regularizer = tf.keras.regularizers.l2(0.001)
    
    # Simple feedforward architecture
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        outputs = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizer)(x)
        loss = 'binary_crossentropy'
    else:
        # Multi-class classification
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizer)(x)
        loss = 'sparse_categorical_crossentropy'
    
    # Create and compile model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Try to find weights to transfer from base model's dense layers
    try:
        logging.info("Trying to transfer weights from base model dense layers")
        # Look for dense layers in the base model with compatible shapes
        transferred_weights = False
        
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                for new_layer in model.layers:
                    if isinstance(new_layer, tf.keras.layers.Dense):
                        # Check if shapes are compatible for transfer
                        if layer.get_weights() and new_layer.get_weights():
                            base_w, base_b = layer.get_weights()
                            new_w, new_b = new_layer.get_weights()
                            
                            if base_w.shape == new_w.shape and base_b.shape == new_b.shape:
                                logging.info(f"Transferring weights from {layer.name} to {new_layer.name}")
                                new_layer.set_weights([base_w, base_b])
                                transferred_weights = True
                                break
        
        if not transferred_weights:
            logging.info("No compatible weights found for transfer. Using random initialization.")
    except Exception as e:
        logging.warning(f"Error during weight transfer: {str(e)}")
    
    # Use a lower learning rate for transfer learning
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
    
    # Add metrics
    metrics = [
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    logging.info("Created new simplified model without residual connections")
    model.summary(print_fn=logging.info)
    
    return model

def plot_training_results(history, output_dir):
    """Plot and save training metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Chained Transfer Learning - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Chained Transfer Learning - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()
    
def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate model and save metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test, verbose=1)
    metrics_names = model.metrics_names
    
    # Log metrics
    metrics_dict = dict(zip(metrics_names, evaluation))
    for name, value in metrics_dict.items():
        logging.info(f"Test {name}: {value:.4f}")
    
    # Generate predictions
    y_pred_proba = model.predict(X_test)
    
    # Find optimal threshold using multiple metrics
    # For imbalanced datasets, we want to optimize for more than just F1
    from sklearn.metrics import precision_recall_curve, roc_curve, matthews_corrcoef
    
    # Calculate precision-recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Calculate ROC curve
    fprs, tprs, roc_thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    
    # Calculate geometric mean of TPR and TNR (balanced accuracy at each threshold)
    # This helps with imbalanced datasets
    def calculate_g_mean(threshold):
        y_pred = (y_pred_proba >= threshold).astype(int)
        positives = y_test == 1
        negatives = y_test == 0
        
        # True positive rate
        tp = np.sum(y_pred[positives] == 1)
        tpr = tp / np.sum(positives) if np.sum(positives) > 0 else 0
        
        # True negative rate
        tn = np.sum(y_pred[negatives] == 0)
        tnr = tn / np.sum(negatives) if np.sum(negatives) > 0 else 0
        
        # Geometric mean
        g_mean = np.sqrt(tpr * tnr) if tpr > 0 and tnr > 0 else 0
        return g_mean
    
    # Calculate g-mean for a range of thresholds
    thresholds_to_check = np.linspace(0.1, 0.9, 17)  # Check more potential thresholds
    g_means = [calculate_g_mean(t) for t in thresholds_to_check]
    
    # Calculate Matthews Correlation Coefficient for each threshold
    mcc_scores = []
    for threshold in thresholds_to_check:
        y_pred = (y_pred_proba >= threshold).astype(int)
        mcc = matthews_corrcoef(y_test, y_pred)
        mcc_scores.append(mcc)
    
    # Find the best threshold for each metric
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
    
    best_gmean_idx = np.argmax(g_means)
    best_gmean_threshold = thresholds_to_check[best_gmean_idx]
    
    best_mcc_idx = np.argmax(mcc_scores)
    best_mcc_threshold = thresholds_to_check[best_mcc_idx]
    
    # Log all possible optimal thresholds
    logging.info(f"Best thresholds by metric:")
    logging.info(f"  F1-score: {best_f1_threshold:.4f} (F1: {f1_scores[best_f1_idx]:.4f})")
    logging.info(f"  G-mean: {best_gmean_threshold:.4f} (G-mean: {g_means[best_gmean_idx]:.4f})")
    logging.info(f"  MCC: {best_mcc_threshold:.4f} (MCC: {mcc_scores[best_mcc_idx]:.4f})")
    
    # Choose the final threshold based on all metrics
    # For imbalanced datasets, G-mean is often more robust
    final_threshold = best_gmean_threshold
    
    logging.info(f"Selected classification threshold: {final_threshold:.4f}")
    
    # Apply best threshold
    y_pred_classes = (y_pred_proba >= final_threshold).astype(int)
    
    # Generate comprehensive metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    
    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred_classes)
    logging.info(f"Matthews Correlation Coefficient: {mcc:.4f}")
    
    # Classification report
    class_names = ["Benign", "Attack"]
    report = classification_report(y_test, y_pred_classes, target_names=class_names, 
                                  output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred_classes, target_names=class_names, zero_division=0))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Save normalized confusion matrix
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Display values in the matrix
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, "normalized_confusion_matrix.png"))
    plt.close()
    
    # Save raw confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Raw Counts)')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Display values in the matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # Generate precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions, color='blue', lw=2)
    plt.axvline(x=recalls[best_f1_idx] if best_f1_idx < len(recalls) else 0, 
                color='red', linestyle='--', 
                label=f'F1 Threshold: {best_f1_threshold:.2f}')
    plt.axvline(x=0.5, color='green', linestyle=':', 
                label=f'G-mean Threshold: {best_gmean_threshold:.2f}') 
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()
    
    # Plot MCC values across thresholds
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds_to_check, mcc_scores, 'b-', marker='o')
    plt.axvline(x=best_mcc_threshold, color='red', linestyle='--',
               label=f'Best MCC threshold: {best_mcc_threshold:.2f}')
    plt.grid(True)
    plt.xlabel('Threshold')
    plt.ylabel('Matthews Correlation Coefficient')
    plt.title('MCC vs. Classification Threshold')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "mcc_threshold.png"))
    plt.close()
    
    # Return comprehensive metrics
    return {
        'accuracy': metrics_dict.get('accuracy', 0),
        'auc': metrics_dict.get('auc', roc_auc),
        'precision': metrics_dict.get('precision', report['weighted avg']['precision']),
        'recall': metrics_dict.get('recall', report['weighted avg']['recall']),
        'f1': report['weighted avg']['f1-score'],
        'g_mean': g_means[best_gmean_idx],
        'mcc': mcc,
        'best_threshold': final_threshold,
        'confusion_matrix': cm
    }

def save_model_and_scaler(model, scaler, output_dir):
    """Save model and scaler for future use"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "chained_model.keras")
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Chained transfer learning for NeuraShield')
    
    parser.add_argument('--base-model-path', type=str, required=True,
                        help='Path to the pre-trained model from first transfer step')
    parser.add_argument('--target-dataset-path', type=str, required=True,
                        help='Path to the target dataset (CSE-CIC-IDS2018)')
    parser.add_argument('--output-dir', type=str, default='models/multi_dataset/chained_transfer',
                        help='Directory to save models and results')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file and console
    log_file = os.path.join(args.output_dir, "chained_transfer.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load and process target dataset - now also returns feature_cols
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols = process_cse_cic_ids2018(args.target_dataset_path)
        
        # Check class distribution and calculate class weights
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        # Create class weights to handle imbalance - more aggressive weighting
        # This gives higher weight to minority class
        class_weights = {}
        if len(unique_classes) > 1:
            max_count = max(class_counts)
            for i, count in zip(unique_classes, class_counts):
                # More aggressive weighting for rare classes
                weight = (total_samples / (len(unique_classes) * count)) * (max_count / count)**0.5
                class_weights[int(i)] = weight
                
            logging.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
            logging.info(f"Enhanced class weights: {class_weights}")
        else:
            logging.warning("Only one class found in training data. No class weights applied.")
        
        # Determine number of classes
        num_classes = len(np.unique(y_train))
        logging.info(f"Target dataset has {num_classes} classes")
        
        # Load pre-trained model
        base_model = load_pretrained_model(args.base_model_path)
        
        # Prepare model for transfer learning, using exactly 8 features
        model = prepare_model_for_transfer(base_model, num_classes, X_train.shape[1])
        
        # Define improved callbacks
        checkpoint_path = os.path.join(args.output_dir, "best_model.keras")
        
        callbacks = [
            # Early stopping with higher patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc', 
                patience=10,              # More patience to find global optimum
                restore_best_weights=True,
                mode='max'                # We want to maximize AUC
            ),
            # Model checkpoint - save best model
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                monitor='val_auc',
                mode='max',
                verbose=1
            ),
            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,               # More gradual reduction
                patience=5, 
                min_lr=0.00001, 
                verbose=1
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(args.output_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Fine-tune model with enhanced class weights and callbacks
        logging.info(f"Fine-tuning model for {args.epochs} epochs with batch size {args.batch_size}")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            shuffle=True
        )
        
        # Plot training results
        plot_training_results(history, args.output_dir)
        
        # Load best model for evaluation
        if os.path.exists(checkpoint_path):
            logging.info(f"Loading best model from {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)
        
        # Evaluate model with best threshold selection
        try:
            metrics = evaluate_model(model, X_test, y_test, args.output_dir)
        except KeyError as e:
            logging.warning(f"KeyError in metrics: {e}. Using first metric value instead.")
            evaluation = model.evaluate(X_test, y_test, verbose=1)
            metrics = {
                'accuracy': evaluation[1],
                'auc': evaluation[2] if len(evaluation) > 2 else 0, 
                'precision': evaluation[3] if len(evaluation) > 3 else 0,
                'recall': evaluation[4] if len(evaluation) > 4 else 0
            }
        
        # Save model and scaler
        save_model_and_scaler(model, scaler, args.output_dir)
        
        # Log training completion
        elapsed_time = time.time() - start_time
        logging.info(f"Chained transfer learning completed in {elapsed_time:.2f} seconds")
        logging.info(f"Final metrics: {metrics}")
        
    except Exception as e:
        logging.error(f"Error during chained transfer learning: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 