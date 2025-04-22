#!/usr/bin/env python3

"""
NeuraShield Model Fine-Tuning Tool
This script enables fine-tuning of threat detection models with new data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add the ai_models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models"))

def load_model(model_path):
    """Load saved model from path"""
    try:
        if os.path.isdir(model_path):
            # Check if this is an ensemble model
            if os.path.exists(os.path.join(model_path, "ensemble_config.pkl")):
                from ensemble_models import EnsembleModel
                model = EnsembleModel.load(model_path)
                model_type = "ensemble"
            elif os.path.exists(os.path.join(model_path, "hybrid_config.pkl")):
                from ensemble_models import HybridEnsemble
                model = HybridEnsemble.load(model_path)
                model_type = "hybrid"
            elif os.path.exists(os.path.join(model_path, "specialized_config.pkl")):
                from ensemble_models import SpecializedEnsemble
                model = SpecializedEnsemble.load(model_path)
                model_type = "specialized"
            else:
                # Regular model saved in directory
                model_files = [f for f in os.listdir(model_path) if f.endswith('.keras')]
                if not model_files:
                    raise FileNotFoundError(f"No .keras model files found in {model_path}")
                model = tf.keras.models.load_model(os.path.join(model_path, model_files[0]))
                model_type = "standard"
        else:
            # Direct model file
            model = tf.keras.models.load_model(model_path)
            model_type = "standard"
        
        logging.info(f"Model loaded successfully from {model_path} (type: {model_type})")
        return model, model_type
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def load_data(data_path, label_col=None, scaler_path=None):
    """Load and preprocess data for fine-tuning"""
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
            # Handle label column
            if label_col is not None:
                if label_col in df.columns:
                    X = df.drop(columns=[label_col])
                    y = df[label_col]
                else:
                    raise ValueError(f"Label column '{label_col}' not found in data")
            else:
                # Assume last column is the label
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
            
            # Apply scaling if provided
            if scaler_path:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_scaled = scaler.transform(X)
            else:
                # Create new scaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
            X_data = X_scaled
            y_data = y.values
            feature_names = X.columns.tolist()
            
        elif data_path.endswith('.npz'):
            # Load numpy arrays
            data = np.load(data_path)
            X_data = data['X']
            y_data = data['y']
            feature_names = None
            
            # Apply scaling if provided
            if scaler_path:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_data = scaler.transform(X_data)
            else:
                # Create new scaler
                scaler = StandardScaler()
                X_data = scaler.fit_transform(X_data)
        else:
            # Try loading pickle file
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            X_data = data['X']
            y_data = data['y']
            feature_names = data.get('feature_names', None)
            
            # Apply scaling if provided
            if scaler_path:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_data = scaler.transform(X_data)
            
        logging.info(f"Data loaded with shape: X={X_data.shape}, y={y_data.shape}")
        return X_data, y_data, scaler, feature_names
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def apply_feature_engineering(X, feature_names=None, advanced=True):
    """Apply feature engineering to the input data"""
    try:
        from feature_engineering import generate_statistical_features, generate_interaction_features
        
        logging.info("Applying feature engineering...")
        X_engineered = X.copy()
        
        if advanced and feature_names is not None:
            # Apply advanced feature engineering based on domain knowledge
            from feature_engineering import generate_advanced_features
            X_additional = generate_advanced_features(X, feature_names)
            
            if X_additional is not None and X_additional.shape[1] > 0:
                X_engineered = np.concatenate([X_engineered, X_additional], axis=1)
                logging.info(f"Added {X_additional.shape[1]} advanced features")
        
        # Apply statistical features
        X_stats = generate_statistical_features(X_engineered)
        if X_stats is not None and X_stats.shape[1] > 0:
            X_engineered = np.concatenate([X_engineered, X_stats], axis=1)
            logging.info(f"Added {X_stats.shape[1]} statistical features")
        
        # Apply interaction features
        X_interactions = generate_interaction_features(X, top_n=5)
        if X_interactions is not None and X_interactions.shape[1] > 0:
            X_engineered = np.concatenate([X_engineered, X_interactions], axis=1)
            logging.info(f"Added {X_interactions.shape[1]} interaction features")
        
        logging.info(f"Feature engineering complete. New shape: {X_engineered.shape}")
        return X_engineered
    except ImportError:
        logging.warning("Feature engineering module not available. Using original features.")
        return X
    except Exception as e:
        logging.error(f"Error in feature engineering: {str(e)}")
        return X

def finetune_standard_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.0001):
    """Fine-tune a standard Keras model"""
    try:
        # Compile model with reduced learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Fine-tune model
        logging.info(f"Fine-tuning standard model for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    except Exception as e:
        logging.error(f"Error fine-tuning standard model: {str(e)}")
        return None, None

def finetune_ensemble_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.0001):
    """Fine-tune an ensemble model"""
    try:
        # For ensemble models, we fine-tune the individual models and then reoptimize weights
        logging.info("Fine-tuning ensemble model...")
        
        # Use the ensemble's own fine-tuning method
        history = model.fine_tune(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return model, history
    except Exception as e:
        logging.error(f"Error fine-tuning ensemble model: {str(e)}")
        return None, None

def finetune_hybrid_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.0001):
    """Fine-tune a hybrid ensemble model"""
    try:
        # For hybrid models, fine-tune neural networks and retrain traditional models
        logging.info("Fine-tuning hybrid ensemble model...")
        
        # Use the hybrid ensemble's own fine-tuning method
        history = model.fine_tune(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return model, history
    except Exception as e:
        logging.error(f"Error fine-tuning hybrid model: {str(e)}")
        return None, None

def finetune_specialized_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.0001):
    """Fine-tune a specialized ensemble model"""
    try:
        # For specialized ensembles, fine-tune each specialized model
        logging.info("Fine-tuning specialized ensemble model...")
        
        # Use the specialized ensemble's own fine-tuning method
        history = model.fine_tune(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return model, history
    except Exception as e:
        logging.error(f"Error fine-tuning specialized model: {str(e)}")
        return None, None

def plot_training_history(history, output_dir):
    """Plot training history metrics"""
    if history is None or not hasattr(history, 'history'):
        logging.warning("No history data available to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy During Fine-tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Fine-tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fine_tuning_history.png'), dpi=300)
    plt.close()
    
    # Save history data
    with open(os.path.join(output_dir, 'fine_tuning_history.json'), 'w') as f:
        json.dump(history.history, f)

def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate model performance after fine-tuning"""
    try:
        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        logging.info(f"Model accuracy after fine-tuning: {accuracy:.4f}")
        
        # Save evaluation results
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        np.savetxt(os.path.join(output_dir, 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
        
        # Print classification report
        logging.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        return accuracy, y_pred, y_pred_proba
    except Exception as e:
        logging.error(f"Error evaluating model: {str(e)}")
        return None, None, None

def compare_performance(original_model, finetuned_model, X_test, y_test, output_dir):
    """Compare performance between original and fine-tuned models"""
    try:
        # Get predictions from both models
        orig_pred_proba = original_model.predict(X_test)
        orig_pred = np.argmax(orig_pred_proba, axis=1)
        
        ft_pred_proba = finetuned_model.predict(X_test)
        ft_pred = np.argmax(ft_pred_proba, axis=1)
        
        # Calculate accuracies
        orig_acc = np.mean(orig_pred == y_test)
        ft_acc = np.mean(ft_pred == y_test)
        
        logging.info(f"Original model accuracy: {orig_acc:.4f}")
        logging.info(f"Fine-tuned model accuracy: {ft_acc:.4f}")
        logging.info(f"Accuracy improvement: {(ft_acc - orig_acc) * 100:.2f}%")
        
        # Find examples where predictions differ
        diff_indices = np.where(orig_pred != ft_pred)[0]
        
        if len(diff_indices) > 0:
            diff_df = pd.DataFrame({
                'Index': diff_indices,
                'True_Label': y_test[diff_indices],
                'Original_Prediction': orig_pred[diff_indices],
                'Finetuned_Prediction': ft_pred[diff_indices],
                'Correct_Before': orig_pred[diff_indices] == y_test[diff_indices],
                'Correct_After': ft_pred[diff_indices] == y_test[diff_indices],
            })
            
            os.makedirs(output_dir, exist_ok=True)
            diff_df.to_csv(os.path.join(output_dir, 'prediction_differences.csv'), index=False)
            
            # Count improvements and regressions
            improvements = sum(~diff_df['Correct_Before'] & diff_df['Correct_After'])
            regressions = sum(diff_df['Correct_Before'] & ~diff_df['Correct_After'])
            
            logging.info(f"Prediction differences: {len(diff_indices)}")
            logging.info(f"Improvements: {improvements} predictions corrected")
            logging.info(f"Regressions: {regressions} predictions newly incorrect")
            
            # Plot comparison
            plt.figure(figsize=(10, 6))
            labels = ['Original Model', 'Fine-tuned Model']
            accuracies = [orig_acc, ft_acc]
            
            plt.bar(labels, accuracies, color=['blue', 'green'])
            plt.ylim([min(orig_acc, ft_acc) * 0.9, max(orig_acc, ft_acc) * 1.05])
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            
            # Add text annotations
            for i, acc in enumerate(accuracies):
                plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
            
            # Add improvement arrow if positive
            if ft_acc > orig_acc:
                plt.annotate(
                    f"+{(ft_acc - orig_acc) * 100:.2f}%",
                    xy=(1, ft_acc),
                    xytext=(0.5, (orig_acc + ft_acc) / 2),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                    ha='center',
                    fontsize=12,
                    fontweight='bold',
                    color='green'
                )
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
            plt.close()
        
        return orig_acc, ft_acc
    except Exception as e:
        logging.error(f"Error comparing model performance: {str(e)}")
        return None, None

def save_finetuned_model(model, model_type, output_dir):
    """Save the fine-tuned model"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if model_type in ['ensemble', 'hybrid', 'specialized']:
            # Use the model's own save method
            model.save(output_dir)
        else:
            # Standard Keras model
            model.save(os.path.join(output_dir, 'finetuned_model.keras'))
        
        logging.info(f"Fine-tuned model saved to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error saving fine-tuned model: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune threat detection model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model to fine-tune')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to new data for fine-tuning')
    parser.add_argument('--output-dir', type=str, default='finetuned_model',
                        help='Directory to save fine-tuned model and results')
    parser.add_argument('--label-col', type=str,
                        help='Name of the label column in CSV data')
    parser.add_argument('--scaler-path', type=str,
                        help='Path to saved scaler for data preprocessing')
    parser.add_argument('--validation-split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Fraction of data to use for testing')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for fine-tuning')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--feature-engineering', action='store_true',
                        help='Apply feature engineering to new data')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original model
    original_model, model_type = load_model(args.model_path)
    if original_model is None:
        return 1
    
    # Make a copy of the original model for comparison
    if model_type in ['ensemble', 'hybrid', 'specialized']:
        # For complex models, we'll keep reference to the original
        comparison_model = original_model
    else:
        # For standard Keras models, make a copy
        comparison_model = tf.keras.models.clone_model(original_model)
        comparison_model.set_weights(original_model.get_weights())
    
    # Load and preprocess data
    X_data, y_data, scaler, feature_names = load_data(
        args.data_path, 
        label_col=args.label_col,
        scaler_path=args.scaler_path
    )
    if X_data is None or y_data is None:
        return 1
    
    # Apply feature engineering if specified
    if args.feature_engineering:
        X_data = apply_feature_engineering(X_data, feature_names)
    
    # Save scaler if it was created
    if args.scaler_path is None and scaler is not None:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler saved to {scaler_path}")
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_data, y_data, test_size=args.test_split, random_state=42, stratify=y_data
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=args.validation_split / (1 - args.test_split),
        random_state=42, 
        stratify=y_train_val
    )
    
    logging.info(f"Data split: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples, {X_test.shape[0]} test samples")
    
    # Fine-tune the model based on its type
    if model_type == 'standard':
        finetuned_model, history = finetune_standard_model(
            original_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
        )
    elif model_type == 'ensemble':
        finetuned_model, history = finetune_ensemble_model(
            original_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
        )
    elif model_type == 'hybrid':
        finetuned_model, history = finetune_hybrid_model(
            original_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
        )
    elif model_type == 'specialized':
        finetuned_model, history = finetune_specialized_model(
            original_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate
        )
    else:
        logging.error(f"Unsupported model type: {model_type}")
        return 1
    
    if finetuned_model is None:
        return 1
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Evaluate fine-tuned model
    accuracy, _, _ = evaluate_model(finetuned_model, X_test, y_test, output_dir)
    if accuracy is None:
        return 1
    
    # Compare with original model
    compare_performance(comparison_model, finetuned_model, X_test, y_test, output_dir)
    
    # Save the fine-tuned model
    save_finetuned_model(finetuned_model, model_type, os.path.join(output_dir, 'model'))
    
    logging.info(f"Fine-tuning complete. Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 