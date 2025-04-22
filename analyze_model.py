#!/usr/bin/env python3

"""
NeuraShield Model Analysis Tool
This script performs comprehensive analysis of threat detection models
and generates detailed performance reports
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Add the ai_models directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_models"))

def load_model(model_path):
    """Load the model from the specified path"""
    try:
        import tensorflow as tf
        
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

def load_data(data_path, preprocessed=False, scaler_path=None):
    """Load data for analysis"""
    try:
        import tensorflow as tf
        
        if preprocessed:
            # Load preprocessed data directly
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            X_data = data['X_test'] if 'X_test' in data else data['X']
            y_data = data['y_test'] if 'y_test' in data else data['y']
            feature_names = data.get('feature_names', None)
            class_names = data.get('class_names', None)
        else:
            # Load raw data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                
                # Assume last column is the label
                X_data = df.iloc[:, :-1].values
                y_data = df.iloc[:, -1].values
                
                # Get feature names
                feature_names = df.columns[:-1].tolist()
                
                # Apply scaling if provided
                if scaler_path:
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    X_data = scaler.transform(X_data)
            elif data_path.endswith('.npz'):
                # Load numpy arrays
                data = np.load(data_path)
                X_data = data['X']
                y_data = data['y']
                feature_names = None
                class_names = None
            else:
                # Try loading pickle file
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                X_data = data['X']
                y_data = data['y']
                feature_names = data.get('feature_names', None)
                class_names = data.get('class_names', None)
        
        if class_names is None:
            # Try to determine class names from the data
            unique_classes = np.unique(y_data)
            if len(unique_classes) == 2:
                class_names = ["Normal", "Attack"]
            else:
                class_names = [f"Class_{i}" for i in range(len(unique_classes))]
        
        logging.info(f"Data loaded with shape: X={X_data.shape}, y={y_data.shape}")
        logging.info(f"Class distribution: {np.bincount(y_data.astype(int))}")
        
        return X_data, y_data, feature_names, class_names
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def analyze_model_architecture(model, model_type):
    """Analyze model architecture and complexity"""
    try:
        import tensorflow as tf
        
        model_info = {}
        
        if model_type == "standard":
            # Standard Keras model
            model_info["type"] = "Standard Neural Network"
            model_info["layers"] = len(model.layers)
            model_info["parameters"] = model.count_params()
            
            # Count trainable vs non-trainable parameters
            trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])
            non_trainable_params = sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
            model_info["trainable_parameters"] = int(trainable_params)
            model_info["non_trainable_parameters"] = int(non_trainable_params)
            
            # Get layer types
            layer_types = {}
            for layer in model.layers:
                layer_type = layer.__class__.__name__
                if layer_type in layer_types:
                    layer_types[layer_type] += 1
                else:
                    layer_types[layer_type] = 1
            model_info["layer_types"] = layer_types
            
            # Get input and output shapes
            model_info["input_shape"] = tuple(model.input_shape[1:])
            model_info["output_shape"] = tuple(model.output_shape[1:])
            
        elif model_type == "ensemble":
            # Ensemble model
            model_info["type"] = "Ensemble Neural Network"
            model_info["num_models"] = len(model.models)
            model_info["meta_model_type"] = model.meta_model.__class__.__name__ if hasattr(model, 'meta_model') else "None"
            
            # Count total parameters across all models
            total_params = 0
            for sub_model in model.models:
                total_params += sub_model.count_params()
            if hasattr(model, 'meta_model'):
                total_params += model.meta_model.count_params()
            model_info["total_parameters"] = total_params
            
            # Get base model types
            model_info["base_model_types"] = [m.__class__.__name__ for m in model.models]
            
            # Get input and output shapes
            model_info["input_shape"] = model.input_shape
            model_info["output_shape"] = model.output_shape
            
        elif model_type == "hybrid":
            # Hybrid ensemble
            model_info["type"] = "Hybrid Ensemble"
            model_info["nn_models"] = len(model.nn_models) if hasattr(model, 'nn_models') else 0
            model_info["ml_models"] = len(model.ml_models) if hasattr(model, 'ml_models') else 0
            
            # Get model types
            if hasattr(model, 'nn_models'):
                model_info["nn_model_types"] = [m.__class__.__name__ for m in model.nn_models]
            if hasattr(model, 'ml_models'):
                model_info["ml_model_types"] = [m.__class__.__name__ for m in model.ml_models]
            
            # Get weights if available
            if hasattr(model, 'weights'):
                model_info["voting_weights"] = model.weights.tolist() if isinstance(model.weights, np.ndarray) else model.weights
            
        elif model_type == "specialized":
            # Specialized ensemble
            model_info["type"] = "Specialized Ensemble"
            model_info["specialized_models"] = len(model.specialized_models) if hasattr(model, 'specialized_models') else 0
            model_info["threat_types"] = model.threat_types if hasattr(model, 'threat_types') else "Unknown"
            
            # Get model types
            if hasattr(model, 'specialized_models'):
                model_info["model_types"] = {threat: m.__class__.__name__ for threat, m in model.specialized_models.items()}
            
            # Get meta model info
            if hasattr(model, 'meta_model'):
                model_info["meta_model_type"] = model.meta_model.__class__.__name__
        
        return model_info
    except Exception as e:
        logging.error(f"Error analyzing model architecture: {str(e)}")
        return {"type": model_type, "error": str(e)}

def analyze_performance(model, X_data, y_data, class_names):
    """Analyze model performance metrics"""
    try:
        import tensorflow as tf
        
        # Get predictions
        y_pred_proba = model.predict(X_data)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy and other metrics
        accuracy = np.mean(y_pred == y_data)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_data, y_pred)
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "support": int(np.sum(cm[i, :]))
            }
        
        # Calculate ROC AUC for each class
        roc_auc_scores = {}
        # Convert to one-hot for multi-class ROC
        y_data_onehot = tf.keras.utils.to_categorical(y_data, num_classes=len(class_names))
        
        for i, class_name in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(y_data_onehot[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                roc_auc_scores[class_name] = roc_auc
            except Exception as e:
                logging.warning(f"Error calculating ROC AUC for class {class_name}: {str(e)}")
                roc_auc_scores[class_name] = None
        
        # Combine results
        performance_metrics = {
            "accuracy": accuracy,
            "class_metrics": class_metrics,
            "confusion_matrix": cm.tolist(),
            "roc_auc_scores": roc_auc_scores
        }
        
        return performance_metrics
    except Exception as e:
        logging.error(f"Error analyzing model performance: {str(e)}")
        return {"error": str(e)}

def analyze_feature_importance(model, model_type, X_data, y_data, feature_names):
    """Analyze feature importance using various methods"""
    try:
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_data.shape[1])]
        
        # Initialize feature importance results
        feature_importance = {
            "method": "perturbation",
            "importance_scores": {},
            "top_features": []
        }
        
        # Get baseline predictions
        y_pred_proba = model.predict(X_data)
        y_pred = np.argmax(y_pred_proba, axis=1)
        baseline_accuracy = np.mean(y_pred == y_data)
        
        # Store feature importance scores
        importance_scores = []
        
        # Analyze a subset of features if there are too many
        max_features_to_analyze = 100
        if X_data.shape[1] > max_features_to_analyze:
            logging.info(f"Analyzing importance for top {max_features_to_analyze} features (out of {X_data.shape[1]})")
            feature_indices = np.random.choice(X_data.shape[1], max_features_to_analyze, replace=False)
        else:
            feature_indices = range(X_data.shape[1])
        
        # Perturb each feature and measure impact
        for i in feature_indices:
            # Save original values
            original_values = X_data[:, i].copy()
            
            # Perturb the feature (randomize values)
            X_data[:, i] = np.random.permutation(X_data[:, i])
            
            # Get predictions with perturbed feature
            perturbed_preds = model.predict(X_data)
            perturbed_pred_labels = np.argmax(perturbed_preds, axis=1)
            perturbed_accuracy = np.mean(perturbed_pred_labels == y_data)
            
            # Calculate importance (drop in accuracy)
            importance = baseline_accuracy - perturbed_accuracy
            importance_scores.append((i, importance))
            
            # Restore original values
            X_data[:, i] = original_values
        
        # Sort features by importance
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to dictionary
        for i, score in importance_scores:
            feature_importance["importance_scores"][feature_names[i]] = float(score)
        
        # Get top features
        top_n = min(20, len(importance_scores))
        for i, score in importance_scores[:top_n]:
            feature_importance["top_features"].append({
                "name": feature_names[i],
                "importance": float(score)
            })
        
        return feature_importance
    except Exception as e:
        logging.error(f"Error analyzing feature importance: {str(e)}")
        return {"method": "error", "error": str(e)}

def analyze_error_patterns(y_data, y_pred, class_names):
    """Analyze error patterns and misclassifications"""
    try:
        # Find misclassified examples
        misclassified_indices = np.where(y_data != y_pred)[0]
        total_samples = len(y_data)
        misclassified_count = len(misclassified_indices)
        
        # Calculate error rate
        error_rate = misclassified_count / total_samples
        
        # Create confusion matrix for misclassifications
        error_cm = np.zeros((len(class_names), len(class_names)))
        for idx in misclassified_indices:
            true_label = y_data[idx]
            pred_label = y_pred[idx]
            error_cm[true_label, pred_label] += 1
        
        # Get top misclassification patterns
        error_patterns = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and error_cm[i, j] > 0:
                    error_patterns.append({
                        "true_class": class_names[i],
                        "predicted_class": class_names[j],
                        "count": int(error_cm[i, j]),
                        "percentage": float(error_cm[i, j] / total_samples * 100)
                    })
        
        # Sort error patterns by count
        error_patterns.sort(key=lambda x: x["count"], reverse=True)
        
        # Summarize results
        error_analysis = {
            "total_samples": total_samples,
            "misclassified_count": misclassified_count,
            "error_rate": error_rate,
            "error_patterns": error_patterns[:10]  # Top 10 error patterns
        }
        
        return error_analysis
    except Exception as e:
        logging.error(f"Error analyzing error patterns: {str(e)}")
        return {"error": str(e)}

def generate_visualizations(model, X_data, y_data, y_pred, y_pred_proba, class_names, feature_importance, output_dir):
    """Generate key visualizations for the analysis report"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = {}
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_data, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=cm, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        visualizations["confusion_matrix"] = cm_path
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        y_data_onehot = tf.keras.utils.to_categorical(y_data, num_classes=len(class_names))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_data_onehot[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(output_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        visualizations["roc_curves"] = roc_path
        
        # 3. Feature Importance
        if feature_importance and "top_features" in feature_importance:
            plt.figure(figsize=(12, 8))
            top_features = feature_importance["top_features"]
            
            # Check if we have at least one feature with importance > 0
            has_important_features = any(f["importance"] > 0 for f in top_features)
            
            if has_important_features and len(top_features) > 0:
                feature_names = [f["name"] for f in top_features]
                importance_scores = [f["importance"] for f in top_features]
                
                # Sort by importance
                sorted_indices = np.argsort(importance_scores)
                feature_names = [feature_names[i] for i in sorted_indices]
                importance_scores = [importance_scores[i] for i in sorted_indices]
                
                plt.barh(feature_names, importance_scores)
                plt.xlabel('Importance Score')
                plt.title('Feature Importance')
                plt.tight_layout()
                
                fi_path = os.path.join(output_dir, "feature_importance.png")
                plt.savefig(fi_path, dpi=300)
                visualizations["feature_importance"] = fi_path
            
            plt.close()
        
        # 4. Class Distribution
        plt.figure(figsize=(10, 6))
        class_counts = np.bincount(y_data.astype(int))
        plt.bar(class_names, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        class_dist_path = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(class_dist_path, dpi=300)
        plt.close()
        visualizations["class_distribution"] = class_dist_path
        
        # 5. Error Distribution
        if np.any(y_data != y_pred):
            plt.figure(figsize=(10, 6))
            error_indices = np.where(y_data != y_pred)[0]
            error_true_labels = y_data[error_indices]
            
            error_counts = np.bincount(error_true_labels.astype(int), minlength=len(class_names))
            plt.bar(class_names, error_counts)
            plt.xlabel('True Class')
            plt.ylabel('Error Count')
            plt.title('Distribution of Errors by True Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            error_dist_path = os.path.join(output_dir, "error_distribution.png")
            plt.savefig(error_dist_path, dpi=300)
            plt.close()
            visualizations["error_distribution"] = error_dist_path
        
        return visualizations
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
        return {"error": str(e)}

def generate_html_report(analysis_results, output_dir):
    """Generate an HTML report from the analysis results"""
    try:
        report_path = os.path.join(output_dir, "analysis_report.html")
        
        # Extract data
        model_info = analysis_results["model_architecture"]
        performance = analysis_results["performance"]
        feature_imp = analysis_results["feature_importance"]
        error_analysis = analysis_results["error_patterns"]
        visualizations = analysis_results["visualizations"]
        
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NeuraShield Model Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .section {{
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .image-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .image-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.1);
                }}
                .metric-card {{
                    display: inline-block;
                    width: 200px;
                    margin: 10px;
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                    background-color: #fff;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                .good {{
                    color: #27ae60;
                }}
                .medium {{
                    color: #f39c12;
                }}
                .poor {{
                    color: #e74c3c;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NeuraShield Model Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="section">
                    <h2>Model Information</h2>
                    <table>
                        <tr>
                            <th>Property</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add model information
        for key, value in model_info.items():
            if key not in ["layer_types", "base_model_types", "nn_model_types", "ml_model_types"]:
                html_content += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td>{value}</td>
                </tr>
                """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Performance Metrics</h2>
                    
                    <div class="metric-card">
                        <h3>Accuracy</h3>
        """
        
        # Add accuracy with color coding
        accuracy = performance["accuracy"]
        accuracy_class = "good" if accuracy > 0.9 else "medium" if accuracy > 0.7 else "poor"
        html_content += f"""
                        <div class="metric-value {accuracy_class}">{accuracy:.4f}</div>
                    </div>
        """
        
        # Add class metrics
        for class_name, metrics in performance["class_metrics"].items():
            precision = metrics["precision"]
            recall = metrics["recall"]
            f1 = metrics["f1"]
            
            # Determine color classes
            precision_class = "good" if precision > 0.9 else "medium" if precision > 0.7 else "poor"
            recall_class = "good" if recall > 0.9 else "medium" if recall > 0.7 else "poor"
            f1_class = "good" if f1 > 0.9 else "medium" if f1 > 0.7 else "poor"
            
            html_content += f"""
                    <div class="metric-card">
                        <h3>{class_name}</h3>
                        <p>Precision: <span class="{precision_class}">{precision:.4f}</span></p>
                        <p>Recall: <span class="{recall_class}">{recall:.4f}</span></p>
                        <p>F1: <span class="{f1_class}">{f1:.4f}</span></p>
                    </div>
            """
        
        # Add visualizations
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
        """
        
        for viz_name, viz_path in visualizations.items():
            if viz_path and os.path.exists(viz_path):
                rel_path = os.path.relpath(viz_path, output_dir)
                title = viz_name.replace('_', ' ').title()
                html_content += f"""
                    <div class="image-container">
                        <h3>{title}</h3>
                        <img src="{rel_path}" alt="{title}">
                    </div>
                """
        
        # Add feature importance
        html_content += """
                </div>
                
                <div class="section">
                    <h2>Feature Importance</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Importance Score</th>
                        </tr>
        """
        
        if "top_features" in feature_imp:
            for feature in feature_imp["top_features"]:
                html_content += f"""
                <tr>
                    <td>{feature["name"]}</td>
                    <td>{feature["importance"]:.6f}</td>
                </tr>
                """
        
        # Add error analysis
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Error Analysis</h2>
        """
        
        # Convert error analysis into HTML
        if error_analysis and "total_samples" in error_analysis:
            total_samples = error_analysis["total_samples"]
            misclassified = error_analysis["misclassified_count"]
            error_rate = error_analysis["error_rate"] * 100
            
            html_content += f"""
                    <p>Total samples: {total_samples}</p>
                    <p>Misclassified samples: {misclassified} ({error_rate:.2f}%)</p>
                    
                    <h3>Top Error Patterns</h3>
                    <table>
                        <tr>
                            <th>True Class</th>
                            <th>Predicted Class</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
            """
            
            for pattern in error_analysis["error_patterns"]:
                html_content += f"""
                    <tr>
                        <td>{pattern["true_class"]}</td>
                        <td>{pattern["predicted_class"]}</td>
                        <td>{pattern["count"]}</td>
                        <td>{pattern["percentage"]:.2f}%</td>
                    </tr>
                """
        else:
            html_content += """
                    <p>No error analysis available.</p>
            """
        
        # Close HTML
        html_content += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"HTML report generated and saved to {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Error generating HTML report: {str(e)}")
        return None

def analyze_model(model_path, data_path, output_dir, preprocessed=False, scaler_path=None):
    """Perform comprehensive model analysis"""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        model, model_type = load_model(model_path)
        if model is None:
            return False
        
        # Load data
        X_data, y_data, feature_names, class_names = load_data(
            data_path, preprocessed=preprocessed, scaler_path=scaler_path
        )
        if X_data is None:
            return False
        
        # Get predictions
        y_pred_proba = model.predict(X_data)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Analyze model architecture
        model_architecture = analyze_model_architecture(model, model_type)
        
        # Analyze performance
        performance = analyze_performance(model, X_data, y_data, class_names)
        
        # Analyze feature importance
        feature_importance = analyze_feature_importance(model, model_type, X_data, y_data, feature_names)
        
        # Analyze error patterns
        error_patterns = analyze_error_patterns(y_data, y_pred, class_names)
        
        # Generate visualizations
        visualizations = generate_visualizations(
            model, X_data, y_data, y_pred, y_pred_proba, 
            class_names, feature_importance, output_dir
        )
        
        # Combine results
        analysis_results = {
            "model_architecture": model_architecture,
            "performance": performance,
            "feature_importance": feature_importance,
            "error_patterns": error_patterns,
            "visualizations": visualizations
        }
        
        # Save results as JSON
        json_path = os.path.join(output_dir, "analysis_results.json")
        with open(json_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Generate HTML report
        html_report = generate_html_report(analysis_results, output_dir)
        
        logging.info(f"Analysis complete. Results saved to {output_dir}")
        return True
    except Exception as e:
        logging.error(f"Error in model analysis: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze threat detection model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model to analyze')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to test data for analysis')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--preprocessed', action='store_true',
                        help='Indicate if the data is already preprocessed')
    parser.add_argument('--scaler-path', type=str,
                        help='Path to saved scaler for data preprocessing')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    # Analyze model
    success = analyze_model(
        args.model_path,
        args.data_path,
        output_dir,
        preprocessed=args.preprocessed,
        scaler_path=args.scaler_path
    )
    
    if success:
        logging.info(f"Analysis complete. Results saved to {output_dir}")
        return 0
    else:
        logging.error("Analysis failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    import tensorflow as tf
    sys.exit(main()) 