#!/usr/bin/env python3

"""
NeuraShield Advanced AI Training Script
This script implements advanced training with ensemble models and feature engineering
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

# Import the custom modules
from ai_models.feature_engineering import apply_feature_engineering, unsw_feature_engineering, dimensionality_reduction
from ai_models.advanced_models import build_residual_nn, build_sequential_nn, build_conv_nn, build_hybrid_nn, focal_loss
from ai_models.ensemble_models import EnsembleModel, HybridEnsemble, SpecializedEnsemble

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_unsw_dataset(dataset_path, apply_feature_engineering=True):
    """Load the UNSW-NB15 dataset with advanced feature engineering"""
    logging.info(f"Loading dataset from {dataset_path}")
    
    # Find training and testing files
    train_file = os.path.join(dataset_path, "UNSW_NB15_training-set.csv")
    test_file = os.path.join(dataset_path, "UNSW_NB15_testing-set.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"Training or testing file not found in {dataset_path}")
    
    # Load datasets
    train_df = pd.read_csv(train_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)
    
    logging.info(f"Training data loaded with shape: {train_df.shape}")
    logging.info(f"Testing data loaded with shape: {test_df.shape}")
    
    # Apply feature engineering if requested
    if apply_feature_engineering:
        logging.info("Applying feature engineering...")
        
        # UNSW-specific feature engineering
        train_df = unsw_feature_engineering(train_df)
        test_df = unsw_feature_engineering(test_df)
        
        logging.info(f"Feature engineered training data shape: {train_df.shape}")
    
    # Separate features and labels
    X_train = train_df.drop(['id', 'label', 'attack_cat'], axis=1, errors='ignore')
    y_train = train_df['label'].values if 'label' in train_df.columns else train_df.iloc[:, -1].values
    
    X_test = test_df.drop(['id', 'label', 'attack_cat'], axis=1, errors='ignore')
    y_test = test_df['label'].values if 'label' in test_df.columns else test_df.iloc[:, -1].values
    
    # Convert categorical features to numeric
    for col in X_train.select_dtypes(include=['object']).columns:
        # Ensure test set has same columns
        if col in X_test.columns:
            # Get all unique values from both train and test
            all_values = pd.concat([X_train[col], X_test[col]]).unique()
            # Create mapping
            value_map = {val: i for i, val in enumerate(all_values)}
            # Apply mapping
            X_train[col] = X_train[col].map(value_map)
            X_test[col] = X_test[col].map(value_map)
            # Fill any missing values
            X_train[col] = X_train[col].fillna(-1)
            X_test[col] = X_test[col].fillna(-1)
    
    # Fill any remaining missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Ensure test set has same columns as train set
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    
    # Ensure columns are in the same order
    X_test = X_test[X_train.columns]
    
    # Convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    
    logging.info(f"Final feature shapes - Training: {X_train.shape}, Testing: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def plot_training_results(history, model_name, output_dir):
    """Plot and save training metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy.png"))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{model_name}_loss.png"))
    plt.close()
    
    # Plot AUC if available
    if 'auc' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title(f'{model_name} AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model_name}_auc.png"))
        plt.close()

def evaluate_model(model, X_test, y_test, class_names, output_dir):
    """Evaluate model and save metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    logging.info(f"Model accuracy on test set: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, target_names=class_names))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
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
    
    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def train_ensemble_model(X_train, y_train, X_val, y_val, input_shape, num_classes, model_dir):
    """Train ensemble model"""
    logging.info("Training ensemble model...")
    
    # Create ensemble model with consistent config - avoid CNN model
    # Use only residual neural networks with different configurations
    models_config = [
        {"type": "residual", "params": {"units": 128, "num_blocks": 3}},
        {"type": "residual", "params": {"units": 256, "num_blocks": 2}},
        {"type": "residual", "params": {"units": 192, "num_blocks": 4}}
    ]
    ensemble = EnsembleModel(input_shape, num_classes, models_config=models_config)
    
    # Use a consistent batch size that divides evenly into the dataset size
    # This helps avoid shape mismatches in TensorFlow's internal operations
    batch_size = 32  # Smaller batch size for more stable training
    
    # Train base models
    base_histories = ensemble.train_base_models(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
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
        batch_size=batch_size  # Use same batch size for consistency
    )
    
    # Save ensemble model
    os.makedirs(model_dir, exist_ok=True)
    ensemble.save(os.path.join(model_dir, "ensemble_model"))
    
    return ensemble, meta_history

def train_hybrid_ensemble(X_train, y_train, X_val, y_val, input_shape, num_classes, model_dir):
    """Train hybrid ensemble of neural network and traditional ML models"""
    logging.info("Training hybrid ensemble model...")
    
    # Create hybrid ensemble
    hybrid = HybridEnsemble(input_shape, num_classes)
    
    # Use a smaller, consistent batch size
    batch_size = 32
    
    # Train models
    hybrid.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=batch_size
    )
    
    # Save hybrid ensemble
    os.makedirs(model_dir, exist_ok=True)
    hybrid.save(os.path.join(model_dir, "hybrid_ensemble"))
    
    return hybrid

def train_specialized_ensemble(X_train, y_train, X_val, y_val, input_shape, threat_types, model_dir):
    """Train specialized models for different threat types"""
    logging.info("Training specialized ensemble model...")
    
    # Create specialized ensemble
    specialized = SpecializedEnsemble(input_shape, threat_types)
    
    # Use a smaller, consistent batch size
    batch_size = 32
    
    # Train specialized models
    specialized.train_specialized_models(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=batch_size
    )
    
    # Train meta-model
    specialized.train_meta_model(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=batch_size
    )
    
    # Save specialized ensemble
    os.makedirs(model_dir, exist_ok=True)
    specialized.save(os.path.join(model_dir, "specialized_ensemble"))
    
    return specialized

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train advanced AI models for threat detection')
    
    parser.add_argument('--dataset-path', type=str, default='ai_models/datasets/UNSW_NB15',
                        help='Path to UNSW-NB15 dataset')
    parser.add_argument('--model-type', type=str, default='ensemble',
                        choices=['residual', 'conv', 'sequential', 'hybrid', 'ensemble', 'specialized'],
                        help='Type of model to train')
    parser.add_argument('--output-dir', type=str, default='models/advanced',
                        help='Directory to save models and results')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--feature-engineering', action='store_true',
                        help='Apply feature engineering to the input data')
    parser.add_argument('--ensemble-size', type=int, default=3,
                        help='Number of models in the ensemble')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging to file and console
    log_file = os.path.join(args.output_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Start timing
    start_time = time.time()
    
    # Load and prepare data
    try:
        X_train, y_train, X_test, y_test = load_unsw_dataset(
            args.dataset_path,
            apply_feature_engineering=args.feature_engineering
        )
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Save the scaler
        with open(os.path.join(args.output_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        
        # Determine class names
        num_classes = len(np.unique(y_train))
        if num_classes == 2:
            class_names = ["Normal", "Attack"]
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        # Log dataset information
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Class distribution in training set: {np.bincount(y_train)}")
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Validation set shape: {X_val.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # Create timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(args.output_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save input parameters
        with open(os.path.join(model_dir, "training_params.json"), 'w') as f:
            import json
            json.dump(vars(args), f, indent=2)
        
        # Train the selected model type
        if args.model_type == 'residual':
            # Train residual neural network
            logging.info("Training residual neural network...")
            model = build_residual_nn(X_train.shape[1], num_classes, units=256, num_blocks=4)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(model_dir, "best_model.keras"),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Plot training results
            plot_training_results(history, "residual_nn", model_dir)
            
            # Save the final model
            model.save(os.path.join(model_dir, "final_model.keras"))
            
        elif args.model_type == 'conv':
            # Train convolutional neural network
            logging.info("Training convolutional neural network...")
            model = build_conv_nn(X_train.shape[1], num_classes)
            
            # Reshape input data for Conv1D
            sequence_length = 15
            features_per_timestep = X_train.shape[1] // sequence_length
            if features_per_timestep < 1:
                features_per_timestep = 1
                sequence_length = X_train.shape[1]
            
            X_train_reshaped = X_train.reshape(X_train.shape[0], sequence_length, features_per_timestep)
            X_val_reshaped = X_val.reshape(X_val.shape[0], sequence_length, features_per_timestep)
            X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, features_per_timestep)
            
            history = model.fit(
                X_train_reshaped, y_train,
                validation_data=(X_val_reshaped, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(model_dir, "best_model.keras"),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Plot training results
            plot_training_results(history, "conv_nn", model_dir)
            
            # Save the final model
            model.save(os.path.join(model_dir, "final_model.keras"))
            
            # Update variables for evaluation
            X_test = X_test_reshaped
            
        elif args.model_type == 'sequential':
            # Train sequential neural network (LSTM/GRU)
            logging.info("Training sequential neural network...")
            model = build_sequential_nn(X_train.shape[1], num_classes)
            
            # Reshape input data for LSTM
            sequence_length = 15
            features_per_timestep = X_train.shape[1] // sequence_length
            if features_per_timestep < 1:
                features_per_timestep = 1
                sequence_length = X_train.shape[1]
            
            X_train_reshaped = X_train.reshape(X_train.shape[0], sequence_length, features_per_timestep)
            X_val_reshaped = X_val.reshape(X_val.shape[0], sequence_length, features_per_timestep)
            X_test_reshaped = X_test.reshape(X_test.shape[0], sequence_length, features_per_timestep)
            
            history = model.fit(
                X_train_reshaped, y_train,
                validation_data=(X_val_reshaped, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(model_dir, "best_model.keras"),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Plot training results
            plot_training_results(history, "sequential_nn", model_dir)
            
            # Save the final model
            model.save(os.path.join(model_dir, "final_model.keras"))
            
            # Update variables for evaluation
            X_test = X_test_reshaped
            
        elif args.model_type == 'hybrid':
            # Train hybrid neural network
            logging.info("Training hybrid neural network...")
            model = build_hybrid_nn(X_train.shape[1], num_classes)
            
            # Prepare inputs for hybrid model
            static_features = X_train.shape[1] // 2
            seq_features = X_train.shape[1] - static_features
            static_features = max(static_features, 5)
            seq_features = max(seq_features, 10)
            
            X_train_static = X_train[:, :static_features]
            X_train_seq = X_train[:, -seq_features:]
            
            X_val_static = X_val[:, :static_features]
            X_val_seq = X_val[:, -seq_features:]
            
            X_test_static = X_test[:, :static_features]
            X_test_seq = X_test[:, -seq_features:]
            
            # Reshape sequential features
            sequence_length = 10
            features_per_timestep = seq_features // sequence_length
            if features_per_timestep < 1:
                features_per_timestep = 1
                sequence_length = seq_features
                
            X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], sequence_length, features_per_timestep)
            X_val_seq = X_val_seq.reshape(X_val_seq.shape[0], sequence_length, features_per_timestep)
            X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], sequence_length, features_per_timestep)
            
            history = model.fit(
                [X_train_static, X_train_seq], y_train,
                validation_data=([X_val_static, X_val_seq], y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint(
                        os.path.join(model_dir, "best_model.keras"),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                ]
            )
            
            # Plot training results
            plot_training_results(history, "hybrid_nn", model_dir)
            
            # Save the final model
            model.save(os.path.join(model_dir, "final_model.keras"))
            
            # Update variables for evaluation
            X_test = [X_test_static, X_test_seq]
            
        elif args.model_type == 'ensemble':
            # Train ensemble model
            model, history = train_ensemble_model(
                X_train, y_train,
                X_val, y_val,
                X_train.shape[1],
                num_classes,
                model_dir
            )
            
            # Plot training results
            plot_training_results(history, "ensemble", model_dir)
            
        elif args.model_type == 'specialized':
            # Define specialized threat types (can be customized)
            threat_types = ["Normal"]
            if num_classes == 2:
                threat_types.append("Attack")
            else:
                # Create attack categories
                for i in range(1, num_classes):
                    threat_types.append(f"Attack_{i}")
            
            # Train specialized ensemble
            model = train_specialized_ensemble(
                X_train, y_train,
                X_val, y_val,
                X_train.shape[1],
                threat_types,
                model_dir
            )
        
        # Evaluate the model
        logging.info("Evaluating model on test set...")
        metrics = evaluate_model(model, X_test, y_test, class_names, model_dir)
        
        # Log training completion
        elapsed_time = time.time() - start_time
        logging.info(f"Training completed in {elapsed_time:.2f} seconds")
        logging.info(f"Model saved to {model_dir}")
        logging.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 