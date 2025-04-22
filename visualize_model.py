#!/usr/bin/env python3

"""
NeuraShield Model Visualization Tool
This script visualizes the performance of trained threat detection models
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pickle
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import logging

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
            elif os.path.exists(os.path.join(model_path, "hybrid_config.pkl")):
                from ensemble_models import HybridEnsemble
                model = HybridEnsemble.load(model_path)
            elif os.path.exists(os.path.join(model_path, "specialized_config.pkl")):
                from ensemble_models import SpecializedEnsemble
                model = SpecializedEnsemble.load(model_path)
            else:
                # Regular model saved in directory
                model_files = [f for f in os.listdir(model_path) if f.endswith('.keras')]
                if not model_files:
                    raise FileNotFoundError(f"No .keras model files found in {model_path}")
                model = tf.keras.models.load_model(os.path.join(model_path, model_files[0]))
        else:
            # Direct model file
            model = tf.keras.models.load_model(model_path)
        
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def load_data(data_path, preprocessed=False, scaler_path=None):
    """Load test data for visualization"""
    try:
        if preprocessed:
            # Load preprocessed data directly
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            X_test = data['X_test']
            y_test = data['y_test']
        else:
            # Load raw data and preprocess
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                
                # Assume last column is the label if not specified
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                # Apply scaling if provided
                if scaler_path:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    X = scaler.transform(X)
                
                X_test = X
                y_test = y
            else:
                # Assume numpy arrays
                data = np.load(data_path)
                X_test = data['X_test']
                y_test = data['y_test']
        
        logging.info(f"Data loaded with shape: X={X_test.shape}, y={y_test.shape}")
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrix with annotations"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(
        cm_norm, 
        annot=cm,  # Show raw counts inside cells
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Calculate and log metrics
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    logging.info(f"Overall Accuracy: {accuracy:.4f}")
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        logging.info(f"Class {class_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def plot_roc_curves(y_true, y_pred_proba, class_names, output_dir):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Convert to one-hot encoding for multi-class ROC
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.4f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300)
    plt.close()

def plot_precision_recall_curves(y_true, y_pred_proba, class_names, output_dir):
    """Plot precision-recall curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # Convert to one-hot encoding for multi-class precision-recall
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    # Calculate and plot precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AUC = {pr_auc:.4f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "precision_recall_curves.png"), dpi=300)
    plt.close()

def visualize_feature_importance(model, X_test, y_test, feature_names, output_dir, n_features=20):
    """Visualize feature importance using perturbation analysis"""
    if not hasattr(model, 'predict'):
        logging.warning("Model doesn't support feature importance visualization")
        return
    
    # Get baseline predictions
    baseline_preds = model.predict(X_test)
    baseline_accuracy = np.mean(np.argmax(baseline_preds, axis=1) == y_test)
    
    # Store feature importance scores
    importance_scores = []
    
    # Perturb each feature and measure impact
    for i in range(X_test.shape[1]):
        # Save original values
        original_values = X_test[:, i].copy()
        
        # Perturb the feature (randomize values)
        X_test[:, i] = np.random.permutation(X_test[:, i])
        
        # Get predictions with perturbed feature
        perturbed_preds = model.predict(X_test)
        perturbed_accuracy = np.mean(np.argmax(perturbed_preds, axis=1) == y_test)
        
        # Calculate importance (drop in accuracy)
        importance = baseline_accuracy - perturbed_accuracy
        importance_scores.append(importance)
        
        # Restore original values
        X_test[:, i] = original_values
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Select top N features
    top_features = importance_df.head(n_features)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {n_features} Feature Importance')
    plt.xlabel('Importance (Drop in Accuracy)')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()
    
    # Save feature importance data
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

def visualize_tsne(X_test, y_test, class_names, output_dir, perplexity=30, n_components=2):
    """Visualize data distribution using t-SNE"""
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_test)
    
    # Create DataFrame for visualization
    tsne_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1] if n_components > 1 else np.zeros(X_tsne.shape[0]),
        'label': [class_names[y] for y in y_test]
    })
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='x', y='y', hue='label', data=tsne_df, palette='viridis', alpha=0.7)
    plt.title('t-SNE Visualization of Data Distribution')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tsne_visualization.png"), dpi=300)
    plt.close()

def analyze_errors(X_test, y_test, y_pred, class_names, output_dir):
    """Analyze misclassified examples"""
    # Find misclassified examples
    misclassified = np.where(y_test != y_pred)[0]
    
    if len(misclassified) == 0:
        logging.info("No misclassifications found")
        return
    
    # Create DataFrame with misclassification details
    error_df = pd.DataFrame({
        'True_Label': [class_names[y_test[i]] for i in misclassified],
        'Predicted_Label': [class_names[y_pred[i]] for i in misclassified],
        'Sample_Index': misclassified
    })
    
    # Create confusion matrix for misclassifications
    error_cm = pd.crosstab(
        [class_names[y] for y in y_test[misclassified]],
        [class_names[y] for y in y_pred[misclassified]],
        rownames=['True'],
        colnames=['Predicted']
    )
    
    # Save error analysis
    os.makedirs(output_dir, exist_ok=True)
    error_df.to_csv(os.path.join(output_dir, "misclassifications.csv"), index=False)
    error_cm.to_csv(os.path.join(output_dir, "error_confusion_matrix.csv"))
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='True_Label', hue='Predicted_Label', data=error_df)
    plt.title('Distribution of Classification Errors')
    plt.xlabel('True Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Predicted as')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=300)
    plt.close()
    
    # Log error statistics
    logging.info(f"Total misclassifications: {len(misclassified)} ({len(misclassified)/len(y_test)*100:.2f}%)")
    logging.info("\nError confusion matrix:")
    logging.info(error_cm)

def plot_calibration_curve(y_true, y_pred_proba, class_names, output_dir, n_bins=10):
    """Plot reliability (calibration) curve"""
    plt.figure(figsize=(10, 8))
    
    # Convert to one-hot encoding for multi-class calibration
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    # Plot calibration curve for each class
    for i, class_name in enumerate(class_names):
        # Get predicted probabilities for this class
        probs = y_pred_proba[:, i]
        
        # Create bins based on predicted probabilities
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate actual frequencies in each bin
        bin_sums = np.bincount(bin_indices, weights=y_true_onehot[:, i], minlength=n_bins)
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_actual = np.zeros(n_bins)
        nonzero = bin_counts > 0
        bin_actual[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot calibration curve
        plt.plot(bin_centers, bin_actual, marker='o', label=class_name)
    
    # Plot ideal calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Frequency')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=300)
    plt.close()

def visualize_model_performance(model, X_test, y_test, feature_names, output_dir, class_names=None):
    """Visualize model performance with various metrics"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine class names if not provided
    num_classes = len(np.unique(y_test))
    if class_names is None:
        if num_classes == 2:
            class_names = ["Normal", "Attack"]
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names, output_dir)
    
    # Plot ROC curves
    plot_roc_curves(y_test, y_pred_proba, class_names, output_dir)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(y_test, y_pred_proba, class_names, output_dir)
    
    # Plot calibration curve
    plot_calibration_curve(y_test, y_pred_proba, class_names, output_dir)
    
    # Visualize feature importance (if applicable)
    if feature_names is not None:
        visualize_feature_importance(model, X_test, y_test, feature_names, output_dir)
    
    # Visualize data distribution with t-SNE
    visualize_tsne(X_test, y_test, class_names, output_dir)
    
    # Analyze misclassifications
    analyze_errors(X_test, y_test, y_pred, class_names, output_dir)
    
    # Generate and save classification report
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))
    
    # Log overall performance
    accuracy = np.mean(y_pred == y_test)
    logging.info(f"Overall accuracy: {accuracy:.4f}")
    logging.info(f"Classification report:\n{classification_report(y_test, y_pred, target_names=class_names)}")
    
    # Create summary visualization
    create_summary_dashboard(
        accuracy,
        report_df,
        os.path.join(output_dir, "confusion_matrix.png"),
        os.path.join(output_dir, "roc_curves.png"),
        os.path.join(output_dir, "precision_recall_curves.png"),
        output_dir
    )

def create_summary_dashboard(accuracy, report_df, cm_path, roc_path, pr_path, output_dir):
    """Create a summary dashboard with key visualizations"""
    plt.figure(figsize=(15, 12))
    
    # Set up the grid
    gs = plt.GridSpec(3, 2, height_ratios=[1, 2, 2])
    
    # Top row: Summary statistics
    ax_stats = plt.subplot(gs[0, :])
    ax_stats.axis('off')
    
    # Header with accuracy
    ax_stats.text(0.5, 0.8, f"Model Performance Summary", 
                 ha='center', va='center', fontsize=18, fontweight='bold')
    ax_stats.text(0.5, 0.5, f"Overall Accuracy: {accuracy:.4f}", 
                 ha='center', va='center', fontsize=14)
    
    # Add class metrics
    class_metrics = []
    for idx, row in report_df.iterrows():
        if idx not in ['accuracy', 'macro avg', 'weighted avg']:
            class_metrics.append(f"{idx}: Precision={row['precision']:.3f}, Recall={row['recall']:.3f}, F1={row['f1-score']:.3f}")
    
    metric_text = "\n".join(class_metrics)
    ax_stats.text(0.5, 0.2, metric_text, ha='center', va='center', fontsize=10)
    
    # Middle row: Confusion Matrix and ROC Curve
    ax_cm = plt.subplot(gs[1, 0])
    ax_roc = plt.subplot(gs[1, 1])
    
    # Add images if they exist
    if os.path.exists(cm_path):
        cm_img = plt.imread(cm_path)
        ax_cm.imshow(cm_img)
        ax_cm.set_title("Confusion Matrix")
        ax_cm.axis('off')
    
    if os.path.exists(roc_path):
        roc_img = plt.imread(roc_path)
        ax_roc.imshow(roc_img)
        ax_roc.set_title("ROC Curves")
        ax_roc.axis('off')
    
    # Bottom row: Precision-Recall Curve and Feature Importance
    ax_pr = plt.subplot(gs[2, 0])
    ax_fi = plt.subplot(gs[2, 1])
    
    if os.path.exists(pr_path):
        pr_img = plt.imread(pr_path)
        ax_pr.imshow(pr_img)
        ax_pr.set_title("Precision-Recall Curves")
        ax_pr.axis('off')
    
    fi_path = os.path.join(output_dir, "feature_importance.png")
    if os.path.exists(fi_path):
        fi_img = plt.imread(fi_path)
        ax_fi.imshow(fi_img)
        ax_fi.set_title("Feature Importance")
        ax_fi.axis('off')
    else:
        tsne_path = os.path.join(output_dir, "tsne_visualization.png")
        if os.path.exists(tsne_path):
            tsne_img = plt.imread(tsne_path)
            ax_fi.imshow(tsne_img)
            ax_fi.set_title("t-SNE Visualization")
            ax_fi.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_dashboard.png"), dpi=300)
    plt.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize model performance')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model or model directory')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to test data (CSV or numpy file)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--feature-names', type=str,
                        help='Path to JSON file with feature names')
    parser.add_argument('--class-names', type=str,
                        help='Path to JSON file with class names')
    parser.add_argument('--scaler-path', type=str,
                        help='Path to saved scaler for data preprocessing')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Indicate if the data is already preprocessed')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load model
    model = load_model(args.model_path)
    if model is None:
        return 1
    
    # Load test data
    X_test, y_test = load_data(
        args.data_path,
        preprocessed=args.preprocessed,
        scaler_path=args.scaler_path
    )
    if X_test is None or y_test is None:
        return 1
    
    # Load feature names if provided
    feature_names = None
    if args.feature_names:
        try:
            with open(args.feature_names, 'r') as f:
                feature_names = json.load(f)
        except Exception as e:
            logging.warning(f"Error loading feature names: {str(e)}")
    
    # Load class names if provided
    class_names = None
    if args.class_names:
        try:
            with open(args.class_names, 'r') as f:
                class_names = json.load(f)
        except Exception as e:
            logging.warning(f"Error loading class names: {str(e)}")
    
    # Visualize model performance
    visualize_model_performance(
        model, X_test, y_test, 
        feature_names=feature_names,
        output_dir=args.output_dir,
        class_names=class_names
    )
    
    logging.info(f"Visualizations saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 